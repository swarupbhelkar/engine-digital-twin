"""Week 2, step 3: the calibrated design point (a new FD001 engine).

Engine layout (station numbers as in C-MAPSS):

  2 --fan--> 13 --bypass nozzle
  |
  +--LPC--> 24 --HPC--> 30 --combustor--> 40 --HPT--> 45 --LPT--> 50 --core nozzle
                         |                    ^           ^
                         +-- W31 coolant -----+           |
                         +-- W32 coolant -----------------+

The two coolant flows (W31, W32) leave the HPC exit, skip the combustor and
rejoin the gas in front of each turbine. They are measured in FD001.

What is measured and what is not:
  measured  : T2, P2, T24, T30, T50, P15, P30, BPR, W31, W32, fuel flow
  assumed   : fan, LPC, HPT and LPT efficiencies (typical values), burner loss
  solved for: core mass flow and turbine inlet temperature T4,
              so that the model reproduces measured fuel flow AND T50.

Fuel flow is not a direct C-MAPSS sensor. phi = fuel flow / Ps30, so
fuel flow = phi * Ps30. Read in lbm/hour this gives about 24,700 lb/h,
a sensible number for a ~90,000 lbf engine. This is an ASSUMPTION about
units; see the note at the end of the output.

Run from the repo root:
    python -m src.physics.design_point
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from scipy.optimize import fsolve

from src.data_loader import load_train
from src.physics import gas
from src.physics.components import Flow, compressor, combustor, mix, nozzle, split, turbine

REPORTS = Path(__file__).resolve().parents[2] / "reports"
HEALTHY_CYCLES = 20


@dataclass
class EngineInputs:
    """Everything needed to run the engine once, front to back."""
    t2: float
    p2: float
    w_core: float     # core mass flow into the LPC, lbm/s
    bpr: float        # bypass ratio
    pr_fan: float
    pr_lpc: float
    pr_hpc: float
    w31: float        # HPT coolant, lbm/s
    w32: float        # LPT coolant, lbm/s
    t4: float         # combustor exit temperature, degR
    eta_fan: float = 0.90
    eta_lpc: float = 0.88
    eta_hpc: float = 0.85
    eta_hpt: float = 0.89
    eta_lpt: float = 0.90
    dp_burner: float = 0.05


def run_engine(x: EngineInputs) -> dict:
    """Run the gas through every component. Returns stations, powers, thrust."""
    inlet = Flow(w=x.w_core * (1 + x.bpr), t=x.t2, p=x.p2)
    core, bypass = split(inlet, x.bpr / (1 + x.bpr))

    s13, pw_fan = compressor(bypass, x.pr_fan, x.eta_fan)
    s24, pw_lpc = compressor(core, x.pr_lpc, x.eta_lpc)
    s30, pw_hpc = compressor(s24, x.pr_hpc, x.eta_hpc)

    # Coolant leaves at HPC exit; the rest goes to the combustor
    s30_rest, c31 = split(s30, x.w31 / s30.w)
    burner_in, c32 = split(s30_rest, x.w32 / s30_rest.w)
    s40, wf = combustor(burner_in, x.t4, x.dp_burner)

    s41 = mix(s40, c31)                          # HPT coolant joins at HPT inlet
    s45 = turbine(s41, pw_hpc, x.eta_hpt)        # HP spool: HPT drives HPC
    s46 = mix(s45, c32)                          # LPT coolant joins at LPT inlet
    s50 = turbine(s46, pw_fan + pw_lpc, x.eta_lpt)  # LP spool: LPT drives fan + LPC

    core_noz, fan_noz = nozzle(s50, x.p2), nozzle(s13, x.p2)
    return {
        "stations": {"2": core, "13": s13, "24": s24, "30": s30, "40": s40,
                     "41": s41, "45": s45, "50": s50},
        "power_btu_s": {"fan": pw_fan, "lpc": pw_lpc, "hpc": pw_hpc},
        "wf": wf,
        "pr_hpt": s41.p / s45.p,
        "pr_lpt": s46.p / s50.p,
        "thrust": core_noz["thrust"] + fan_noz["thrust"],
        "nozzles": {"core": core_noz, "bypass": fan_noz},
    }


def healthy_targets() -> dict:
    """Fleet-mean sensor values over the first 20 cycles of every engine."""
    df = load_train()
    n = df[df["cycle"] <= HEALTHY_CYCLES].mean(numeric_only=True)
    return {
        "t2": n["s1"], "p2": n["s5"], "t24": n["s2"], "t30": n["s3"], "t50": n["s4"],
        "p15": n["s6"], "p30": n["s7"], "bpr": n["s15"], "w31": n["s20"], "w32": n["s21"],
        "wf": n["s12"] * n["s11"] / 3600.0,     # lbm/h -> lbm/s (assumption, see top)
    }


def calibrate(tg: dict) -> EngineInputs:
    """Find the design-point inputs that reproduce the healthy sensors."""
    # 1) Direct from sensors: pressure ratios and HPC efficiency
    eta_lpc = EngineInputs.eta_lpc
    pr_lpc = float(gas.compressor_pressure_ratio(tg["t2"], tg["t24"], eta_lpc))
    pr_hpc = tg["p30"] / (tg["p2"] * pr_lpc)
    eta_hpc = float(gas.compressor_efficiency(tg["t24"], tg["t30"], pr_hpc))

    def inputs(w_core, t4):
        return EngineInputs(t2=tg["t2"], p2=tg["p2"], w_core=w_core, bpr=tg["bpr"],
                            pr_fan=tg["p15"] / tg["p2"], pr_lpc=pr_lpc, pr_hpc=pr_hpc,
                            w31=tg["w31"], w32=tg["w32"], t4=t4, eta_hpc=eta_hpc)

    # 2) Two unknowns, two equations: match fuel flow and T50.
    #    Errors are divided by the target so both are in "fraction" units.
    def errors(guess):
        w_core, t4 = guess
        r = run_engine(inputs(w_core, t4))
        return [r["wf"] / tg["wf"] - 1.0, r["stations"]["50"].t / tg["t50"] - 1.0]

    (w_core, t4), info, ok, msg = fsolve(errors, x0=[300.0, 2800.0], full_output=True)
    if ok != 1:
        raise RuntimeError(f"calibration did not converge: {msg}")
    return inputs(float(w_core), float(t4))


def main() -> None:
    tg = healthy_targets()
    x = calibrate(tg)
    r = run_engine(x)
    st = r["stations"]

    print("Calibrated design point (new FD001 engine)\n")
    print(f"{'station':<24}{'T (degR)':>10}{'P (psia)':>10}{'W (lbm/s)':>11}")
    names = {"2": "2  core inlet", "24": "24 LPC exit", "30": "30 HPC exit",
             "40": "40 combustor exit", "41": "41 HPT inlet (+W31)", "45": "45 HPT exit",
             "50": "50 LPT exit", "13": "13 fan exit (bypass)"}
    for k, name in names.items():
        f = st[k]
        print(f"{name:<24}{f.t:10.1f}{f.p:10.2f}{f.w:11.1f}")

    print("\nModel vs sensor:")
    checks = [("T24", st["24"].t, tg["t24"]), ("T30", st["30"].t, tg["t30"]),
              ("T50", st["50"].t, tg["t50"]), ("P30", st["30"].p, tg["p30"]),
              ("P15", st["13"].p, tg["p15"]), ("fuel lbm/s", r["wf"], tg["wf"])]
    for name, model, data in checks:
        print(f"  {name:<11} model {model:9.2f}  sensor {data:9.2f}  "
              f"error {100 * (model / data - 1):+.3f} %")

    print("\nSolved and derived values:")
    print(f"  core flow     {x.w_core:7.1f} lbm/s   total airflow {x.w_core * (1 + x.bpr):,.0f} lbm/s")
    print(f"  T4            {x.t4:7.1f} degR  ({x.t4 / 1.8:.0f} K)")
    print(f"  HPC eff       {x.eta_hpc:7.4f}")
    print(f"  HPT PR {r['pr_hpt']:.2f}, LPT PR {r['pr_lpt']:.2f}, fuel-air ratio {st['40'].far:.4f}")
    print(f"  coolant       {100 * (x.w31 + x.w32) / x.w_core:.1f} % of core flow")
    print(f"  gross thrust  {r['thrust']:,.0f} lbf (indicative)")

    out = REPORTS / "design_point.json"
    out.write_text(json.dumps(asdict(x), indent=2))
    print(f"\nSaved {out.relative_to(REPORTS.parent)} (used by step 4)")
    print("\nNote: fuel flow = phi * Ps30 read as lbm/h is an assumption. If it is "
          "wrong, core flow and thrust scale with it; temperatures and pressures do not.")


if __name__ == "__main__":
    main()
