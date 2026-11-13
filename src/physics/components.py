"""Engine components as small functions: each takes a gas flow in, gives a flow out.

A Flow is the state of the gas at one station: mass flow, total temperature,
total pressure and fuel-air ratio. Power is in BTU/s (1 BTU/s = 1.055 kW).

Run the demo from the repo root:
    python -m src.physics.components
"""
from dataclasses import dataclass, replace

import numpy as np

from src.data_loader import load_train
from src.physics import gas

GC = 32.174      # lbm*ft / (lbf*s^2), converts mass to force units
J = 778.17       # ft*lbf per BTU
R_FT = gas.R_GAS * J   # gas constant in ft*lbf/(lbm*degR), about 53.35


@dataclass(frozen=True)
class Flow:
    w: float          # mass flow, lbm/s
    t: float          # total temperature, degR
    p: float          # total pressure, psia
    far: float = 0.0  # fuel-air ratio (0 = pure air)

    @property
    def h(self) -> float:
        """Enthalpy per lbm, BTU/lbm."""
        return float(gas.enthalpy(self.t, self.far))


def split(flow: Flow, fraction: float) -> tuple[Flow, Flow]:
    """Split a flow in two (e.g. bypass vs core, or a cooling bleed).
    Returns (remaining, taken), where taken = fraction of the mass flow."""
    taken = replace(flow, w=flow.w * fraction)
    return replace(flow, w=flow.w - taken.w), taken


def compressor(flow: Flow, pr: float, eta: float) -> tuple[Flow, float]:
    """Raise pressure by pr. Returns (exit flow, power absorbed in BTU/s)."""
    t_out = float(gas.compressor_exit_temp(flow.t, pr, eta, flow.far))
    out = replace(flow, t=t_out, p=flow.p * pr)
    return out, flow.w * (out.h - flow.h)


def combustor(flow: Flow, t_out: float, dp: float = 0.05,
              eta_b: float = 0.995) -> tuple[Flow, float]:
    """Burn fuel until the exit reaches t_out. dp = fractional pressure loss.
    Returns (exit flow, fuel flow in lbm/s).

    Energy balance per lbm of air (fuel enthalpy is included in LHV):
        h_air(T3) + far * eta_b * LHV = (1 + far) * h_gas(T4, far)
    h_gas depends on far, so iterate a few times.
    """
    far = 0.02
    for _ in range(20):
        h_out = float(gas.enthalpy(t_out, far))
        far = (h_out - flow.h) / (eta_b * gas.LHV - h_out)
    wf = flow.w * far
    return Flow(w=flow.w + wf, t=t_out, p=flow.p * (1.0 - dp), far=far), wf


def turbine(flow: Flow, power: float, eta: float) -> Flow:
    """Extract a given power (BTU/s). The pressure ratio follows from it."""
    h_out = flow.h - power / flow.w
    t_out = float(gas.temp_from_enthalpy(h_out, flow.far))
    pr = float(gas.turbine_pressure_ratio(flow.t, t_out, eta, flow.far))
    return replace(flow, t=t_out, p=flow.p / pr)


def mix(main: Flow, extra: Flow) -> Flow:
    """Mix two flows (e.g. cooling air rejoining the gas path).
    Energy and fuel are conserved; the main stream sets the pressure."""
    w = main.w + extra.w
    fuel = main.w * main.far / (1 + main.far) + extra.w * extra.far / (1 + extra.far)
    far = fuel / (w - fuel)
    h = (main.w * main.h + extra.w * extra.h) / w
    return Flow(w=w, t=float(gas.temp_from_enthalpy(h, far)), p=main.p, far=far)


def nozzle(flow: Flow, p_amb: float) -> dict:
    """Convergent nozzle. Returns gross thrust (lbf), exit velocity (ft/s),
    exit area (in^2) and whether the throat is choked (sonic)."""
    g = float(gas.gamma(flow.t, flow.far))
    p_crit = flow.p / ((g + 1) / 2) ** (g / (g - 1))   # pressure at Mach 1
    choked = p_crit > p_amb
    p_exit = p_crit if choked else p_amb
    t_static = float(gas.isentropic_temp(flow.t, p_exit / flow.p, flow.far))
    dh = flow.h - float(gas.enthalpy(t_static, flow.far))
    v = np.sqrt(2 * GC * J * dh)
    rho = p_exit * 144 / (R_FT * t_static)            # lbm/ft^3
    area = flow.w / (rho * v)                          # ft^2
    thrust = flow.w * v / GC + (p_exit - p_amb) * 144 * area
    return {"thrust": thrust, "velocity": v, "area_in2": area * 144, "choked": choked}


def demo() -> None:
    """Push a new FD001 engine through the components, using sensor values
    where they exist and guesses (marked) where they don't. Step 3 replaces
    the guesses with a proper calibration."""
    df = load_train()
    new = df[df["cycle"] <= 20].mean(numeric_only=True)
    t2, p2, t24, t30, p30 = new["s1"], new["s5"], new["s2"], new["s3"], new["s7"]
    t50_measured, bpr, p15 = new["s4"], new["s15"], new["s6"]

    W_CORE = 100.0    # guess; temperatures and pressures don't depend on it
    ETA_LPC, ETA_FAN, ETA_HPT, ETA_LPT = 0.88, 0.90, 0.89, 0.90   # guesses
    T4 = 2800.0       # turbine inlet temperature, degR (guess)

    inlet = Flow(w=W_CORE * (1 + bpr), t=t2, p=p2)
    core, bypass = split(inlet, bpr / (1 + bpr))

    # Fan (bypass side): pressure ratio from P15/P2
    bypass_out, p_fan = compressor(bypass, p15 / p2, ETA_FAN)
    # LPC: pick the pressure ratio that reproduces measured T24
    s24, p_lpc = compressor(core, float(gas.compressor_pressure_ratio(t2, t24, ETA_LPC)), ETA_LPC)
    # HPC: measured pressure ratio; efficiency from measured T30
    pr_hpc = p30 / s24.p
    eta_hpc = float(gas.compressor_efficiency(t24, t30, pr_hpc))
    s30, p_hpc = compressor(s24, pr_hpc, eta_hpc)
    s40, wf = combustor(s30, T4)
    s45 = turbine(s40, p_hpc, ETA_HPT)              # HPT drives the HPC
    s50 = turbine(s45, p_lpc + p_fan, ETA_LPT)       # LPT drives fan + LPC
    core_noz, fan_noz = nozzle(s50, p2), nozzle(bypass_out, p2)

    print(f"{'station':<22}{'T (degR)':>10}{'P (psia)':>10}{'W (lbm/s)':>11}")
    for name, f in [("2  inlet (core)", core), ("24 LPC exit", s24),
                    ("30 HPC exit", s30), ("40 combustor exit", s40),
                    ("45 HPT exit", s45), ("50 LPT exit", s50),
                    ("13 fan exit (bypass)", bypass_out)]:
        print(f"{name:<22}{f.t:10.1f}{f.p:10.2f}{f.w:11.1f}")
    print(f"\nHPC efficiency with variable cp: {eta_hpc:.4f}  (step 1, constant cp: 0.8202)")
    print(f"Fuel-air ratio {s40.far:.4f}, fuel flow {wf:.2f} lbm/s")
    print(f"Power  HPC {p_hpc / 0.9478:,.0f} kW   LPC {p_lpc / 0.9478:,.0f} kW   "
          f"fan {p_fan / 0.9478:,.0f} kW")
    print(f"Nozzles: core choked={core_noz['choked']}, bypass choked={fan_noz['choked']}; "
          f"gross thrust {core_noz['thrust'] + fan_noz['thrust']:,.0f} lbf")
    print(f"\nT50 predicted {s50.t:.1f}  vs measured {t50_measured:.1f} degR "
          f"(error {s50.t - t50_measured:+.1f})")


if __name__ == "__main__":
    demo()
