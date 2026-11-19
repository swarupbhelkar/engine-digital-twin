"""Week 5, step 5: what the Fluent cascade runs say about the engine.

Input: cfd/results/cascade_fluent.csv - the converged values Fluent reported for
each case (last iteration of the report files), copied by hand:
  p0_in   mass-weighted total pressure at the inlet, Pa
  p0_out  mass-weighted total pressure at the outlet, Pa
  p_in    area-weighted static pressure at the inlet, Pa
  mdot    mass flow per metre of span, kg/s (held at 30.15 by the outlet condition)

Loss coefficient (rotor frame):  omega = (p0_in - p0_out) / (p0_in - p_in)
Link to the engine (step 1a): if every rotor and stator row of the HPC gained the
same loss rise, the HPC efficiency would change by efficiency_drop_from_loss().
Week 2 found -1.49 % at FD001 failure, which needs a rise of about 0.0082.

All three cases run at the same mass flow and inlet Mach (0.75), so any change
in loss comes from the blade surface or shape.

Run from the repo root:
    python -m src.cad_cfd.cascade_results
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.cad_cfd.hpc_stage import design_stage, efficiency_drop_from_loss

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "cfd" / "results"
FAILURE_ETA = -1.49          # % HPC efficiency change at FD001 failure (Week 2)
GAMMA = 1.375                # air at about 600 K (cp 1050 J/kg K)


def analyse() -> pd.DataFrame:
    s = design_stage()
    df = pd.read_csv(RESULTS / "cascade_fluent.csv")
    df["omega"] = (df["p0_in"] - df["p0_out"]) / (df["p0_in"] - df["p_in"])
    g = (GAMMA - 1) / GAMMA
    df["mach_in"] = (((df["p0_in"] / df["p_in"]) ** g - 1) * 2 / (GAMMA - 1)) ** 0.5
    clean = df.loc[df["case"] == "healthy", "omega"].iloc[0]
    df["omega_rise"] = df["omega"] - clean
    df["omega_rise_pct"] = 100 * df["omega_rise"] / clean
    df["d_eta_hpc_pct"] = [100 * efficiency_drop_from_loss(s, d) for d in df["omega_rise"]]
    df["times_failure"] = df["d_eta_hpc_pct"] / FAILURE_ETA
    return df


def main() -> None:
    df = analyse()
    s = design_stage()
    needed = (FAILURE_ETA / 100) / efficiency_drop_from_loss(s, 1.0)
    out = df.round({"omega": 5, "mach_in": 3, "omega_rise": 5, "omega_rise_pct": 1,
                    "d_eta_hpc_pct": 2, "times_failure": 2})
    out.to_csv(RESULTS / "cascade_summary.csv", index=False)
    pd.set_option("display.width", 140)
    print(out[["case", "mdot", "mach_in", "omega", "omega_rise", "omega_rise_pct",
               "d_eta_hpc_pct", "times_failure"]].to_string(index=False))
    print(f"\nLoss rise needed for the FD001 failure level ({FAILURE_ETA} %): {needed:.4f}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    colours = {"healthy": "#006837", "fouled": "#D9822B", "eroded": "#7A4FA3"}
    a1.bar(df["case"], df["omega"], color=[colours[c] for c in df["case"]])
    for x, v in zip(df["case"], df["omega"]):
        a1.text(x, v + 0.0008, f"{v:.4f}", ha="center", fontsize=9)
    a1.set_ylabel("loss coefficient omega")
    a1.set_title("2D cascade, HPC stage 5 rotor (Fluent, M = 0.75)")
    a2.barh(df["case"], df["d_eta_hpc_pct"], color=[colours[c] for c in df["case"]])
    a2.axvline(FAILURE_ETA, color="k", ls="--", lw=1)
    a2.set_xlabel("HPC efficiency change if every row wore alike (%)")
    a2.set_title("Link to the engine")
    a2.invert_yaxis()
    a2.text(FAILURE_ETA, -0.4, " FD001 failure (Week 2)", fontsize=8, va="top")
    fig.tight_layout()
    path = ROOT / "reports" / "figures" / "17_cascade_loss.png"
    fig.savefig(path, dpi=150)
    print(f"Wrote cfd/results/cascade_summary.csv, {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
