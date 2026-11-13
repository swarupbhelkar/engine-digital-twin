"""Week 2, step 4: off-design. Make the engine find its own operating point.

At the design point we GAVE the model the pressure ratios (from sensors).
A twin has to PREDICT them: if the HPC wears, where does the engine settle?

The engine settles where these rules all hold at once:

  1. Fan speed is held by the controller, so the fan swallows the same
     total airflow at the same pressure ratio (P15 and Nf are flat in FD001).
  2. The HPC passes a fixed CORRECTED flow, W*sqrt(T24)/P24, set by its
     geometry. Wear (fouling, tip clearance) shrinks it: "flow capacity".
  3. The HPT and LPT inlet nozzles are CHOKED (flow at Mach 1), so each
     passes a fixed corrected flow W*sqrt(T)/P. Physically: a choked hole
     can only pass so much gas for a given pressure and temperature.
  4. The core nozzle has a fixed area.
  5. On each shaft, turbine power = compressor power.
     (Built into run_engine: each turbine extracts exactly what is needed.)

Unknowns the solver adjusts:  core flow, LPC PR, HPC PR, T4  (4)
Equations it must satisfy:    rules 2, 3 (two nozzles), 4    (4)

Coolant flows stay a fixed fraction of HPC exit flow (fixed bleed holes),
and the controller adds whatever fuel is needed to keep fan speed: T4 floats.

Run from the repo root:
    python -m src.physics.off_design
"""
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.optimize import fsolve

from src.data_loader import load_train
from src.physics.components import Flow, mix, nozzle
from src.physics.design_point import EngineInputs, run_engine

REPORTS = Path(__file__).resolve().parents[2] / "reports"


@dataclass(frozen=True)
class Health:
    """Component health as fractional changes from new (0 = new engine).
    eta_hpc = -0.01 means HPC efficiency is 1% lower (relative, 0.852 -> 0.844)."""
    eta_hpc: float = 0.0
    flow_hpc: float = 0.0


def corrected_flow(f: Flow) -> float:
    """W*sqrt(T)/P. Constant for a choked nozzle or a compressor at fixed geometry."""
    return f.w * np.sqrt(f.t) / f.p


def load_design() -> EngineInputs:
    return EngineInputs(**json.loads((REPORTS / "design_point.json").read_text()))


class OffDesignEngine:
    def __init__(self, design: EngineInputs):
        self.d = design
        self.w_total = design.w_core * (1 + design.bpr)   # rule 1: fixed by the fan
        self.f31 = design.w31 / design.w_core             # coolant fractions (fixed holes)
        self.f32 = design.w32 / design.w_core
        # Reference values of the "fixed" quantities, taken at the design point
        self.ref = self._quantities(run_engine(design), design)

    def inputs(self, w_core, pr_lpc, pr_hpc, t4, health: Health) -> EngineInputs:
        return replace(self.d, w_core=w_core, bpr=(self.w_total - w_core) / w_core,
                       pr_lpc=pr_lpc, pr_hpc=pr_hpc, t4=t4,
                       w31=self.f31 * w_core, w32=self.f32 * w_core,
                       eta_hpc=self.d.eta_hpc * (1 + health.eta_hpc))

    @staticmethod
    def _quantities(r: dict, x: EngineInputs) -> dict:
        """The four quantities that the rules hold fixed."""
        st = r["stations"]
        c32 = Flow(w=x.w32, t=st["30"].t, p=st["30"].p)
        s46 = mix(st["45"], c32)                     # LPT inlet, after its coolant
        return {"hpc": corrected_flow(replace(st["24"], w=x.w_core)),
                "hpt": corrected_flow(st["41"]),
                "lpt": corrected_flow(s46),
                "noz": r["nozzles"]["core"]["area_in2"]}

    def solve(self, health: Health = Health()) -> dict:
        d = self.d
        scale = np.array([d.w_core, d.pr_lpc, d.pr_hpc, d.t4])

        def errors(z):
            x = self.inputs(*(z * scale), health)
            q = self._quantities(run_engine(x), x)
            return [q["hpc"] / (self.ref["hpc"] * (1 + health.flow_hpc)) - 1,
                    q["hpt"] / self.ref["hpt"] - 1,
                    q["lpt"] / self.ref["lpt"] - 1,
                    q["noz"] / self.ref["noz"] - 1]

        z, info, ok, msg = fsolve(errors, x0=np.ones(4), full_output=True, xtol=1e-12)
        if ok != 1:
            raise RuntimeError(f"off-design solve failed for {health}: {msg}")
        x = self.inputs(*(z * scale), health)
        return {"inputs": x, "result": run_engine(x)}


def sensors(sol: dict) -> dict:
    """Model outputs named like the C-MAPSS sensors."""
    x, r = sol["inputs"], sol["result"]
    st = r["stations"]
    return {"T24": st["24"].t, "T30": st["30"].t, "T50": st["50"].t, "P30": st["30"].p,
            "BPR": x.bpr, "W31": x.w31, "W32": x.w32, "fuel": r["wf"]}


def fd001_change() -> dict:
    """Measured % change, new engine (first 20 cycles) -> worn (last 20 cycles)."""
    df = load_train()
    new, old = df[df["cycle"] <= 20], df[df["rul"] < 20]
    cols = {"T24": "s2", "T30": "s3", "T50": "s4", "P30": "s7", "BPR": "s15",
            "W31": "s20", "W32": "s21"}
    out = {k: 100 * (old[c].mean() / new[c].mean() - 1) for k, c in cols.items()}
    out["fuel"] = 100 * ((old["s12"] * old["s11"]).mean() / (new["s12"] * new["s11"]).mean() - 1)
    return out


def main() -> None:
    eng = OffDesignEngine(load_design())
    base = sensors(eng.solve())

    # Sanity check: with no damage the solver must land back on the design point
    d = eng.d
    print(f"No damage -> core flow {base['W31'] / eng.f31:.2f} (design {d.w_core:.2f}), "
          f"T50 {base['T50']:.2f} degR\n")

    cases = {"eff -1%": Health(eta_hpc=-0.01),
             "flow -1%": Health(flow_hpc=-0.01),
             "both -1%": Health(eta_hpc=-0.01, flow_hpc=-0.01)}
    data = fd001_change()

    pct = {name: {k: 100 * (v / base[k] - 1) for k, v in sensors(eng.solve(h)).items()}
           for name, h in cases.items()}

    print("Sensor response, % change from a new engine")
    print(f"{'sensor':<7}" + "".join(f"{n:>11}" for n in cases) + f"{'FD001 data':>13}")
    for k in base:
        row = "".join(f"{pct[n][k]:+11.3f}" for n in cases)
        print(f"{k:<7}{row}{data[k]:+13.3f}")
    print("\nRead it as: does the model move each sensor the SAME WAY as the data?")

    rows = ["sensor," + ",".join(cases) + ",fd001_data"]
    rows += [f"{k}," + ",".join(f"{pct[n][k]:.4f}" for n in cases) + f",{data[k]:.4f}"
             for k in base]
    out = REPORTS / "hpc_sensitivity.csv"
    out.write_text("\n".join(rows) + "\n")
    print(f"Saved {out.relative_to(REPORTS.parent)}")


if __name__ == "__main__":
    main()
