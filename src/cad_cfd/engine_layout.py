"""Week 5, step 2: rough engine layout for the CAD cutaway + HPC health colours.

Part A - layout. The flow passage at each station must carry the Week 2 mass
flow at the station's temperature and pressure:
    area = mass flow / (density * axial velocity)
  - HPC: same velocity triangles as step 1 (repeating stages), constant mean
    radius from the core speed Nc. So the HPC here matches the CFD blade.
  - Other stations: an assumed axial Mach number and hub/tip ratio (typical
    values for a high-bypass turbofan; see STATIONS).
  - Lengths: stages x STAGE_LENGTH x axial chord, chord = blade height / aspect.
This is for PROPORTIONS in the cutaway, not a real engine drawing.

Part B - colours. HPC colour from the twin health index (1 = new, 0 = typical
failure damage) of the Week 4 demo engine 52: red-yellow-green scale.

Outputs:
  cad/engine_layout.csv      station, x, hub and tip radius (mm), for Fusion
  cad/components.csv         axial start/end of each component (mm)
  cad/health_colours.csv     cycle, health, efficiency loss, RUL, colour (hex, RGB)
  reports/figures/14_engine_layout.png

Needs data/processed/twin_state_demo.csv: run python -m src.twin_state first.

Run from the repo root:
    python -m src.cad_cfd.engine_layout
"""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps

from src.cad_cfd.hpc_stage import ASPECT, LBM_TO_KG, PSI_TO_PA, R_SI, design_stage
from src.physics import gas
from src.physics.design_point import run_engine
from src.physics.off_design import load_design

ROOT = Path(__file__).resolve().parents[2]
STAGE_LENGTH = 2.6     # stage length / axial chord (rotor + stator + gaps)
STAGGER = 40.0         # deg, turns chord into axial chord

# Components front to back: (name, stages, blade aspect ratio, front, back station).
# Ducts and the combustor have no blades: their length is LENGTHS x HPC mean radius.
COMPONENTS = [("Fan", 1, 3.0, "2", "2"), ("LPC", 3, ASPECT, "2c", "24L"),
              ("Duct", 0, None, "24L", "24"), ("HPC", 10, ASPECT, "24", "30"),
              ("Combustor", 0, None, "30", "41"), ("HPT", 2, 1.3, "41", "45"),
              ("LPT", 6, 3.0, "45", "50")]
LENGTHS = {"Duct": 0.6, "Combustor": 0.7}

# Non-HPC stations: (station, name, Week 2 station key, axial Mach, hub/tip ratio, flow)
STATIONS = [("2", "fan face", "2", 0.60, 0.30, "total"),
            ("2c", "LPC inlet", "2", 0.50, 0.72, "core"),
            ("24L", "LPC exit", "24", 0.45, 0.80, "core"),
            ("41", "HPT inlet", "41", 0.20, None, "core"),
            ("45", "HPT exit", "45", 0.40, None, "core"),
            ("50", "LPT exit", "50", 0.45, 0.62, "core")]


def flux_area(w_kg, t0_k, p0_pa, mach, gam):
    """Flow area (m2) carrying w_kg at total T, P and axial Mach number:
    W sqrt(T0) / (A P0) = sqrt(gam/R) M (1 + (gam-1)/2 M^2) ^ -((gam+1) / (2(gam-1)))"""
    f = math.sqrt(gam / R_SI) * mach * (1 + (gam - 1) / 2 * mach ** 2) ** (
        -(gam + 1) / (2 * (gam - 1)))
    return w_kg * math.sqrt(t0_k) / (p0_pa * f)


def hub_tip(area, ratio=None, r_mean=None):
    """Hub and tip radius (m) from area and EITHER hub/tip ratio OR mean radius."""
    if r_mean is not None:
        h = area / (2 * math.pi * r_mean)
        return r_mean - h / 2, r_mean + h / 2
    tip = math.sqrt(area / (math.pi * (1 - ratio ** 2)))
    return ratio * tip, tip


def hpc_station(st, s):
    """HPC passage at a station: same Cx and absolute angle as the step 1 stage."""
    t0, p0 = st.t / 1.8, st.p * PSI_TO_PA
    cp = float(gas.cp(st.t)) / gas.KJKGK_TO_BTULBMR * 1000
    gam = float(gas.gamma(st.t))
    c = s.cx / math.cos(math.radians(s.alpha1))
    t = t0 - c ** 2 / (2 * cp)
    rho = p0 * (t / t0) ** (gam / (gam - 1)) / (R_SI * t)
    return hub_tip(st.w * LBM_TO_KG / (rho * s.cx), r_mean=s.r_mean)


def layout() -> tuple[pd.DataFrame, pd.DataFrame]:
    d = load_design()
    st = run_engine(d)["stations"]
    s = design_stage()
    r_mean_hp = s.r_mean
    radii = {}
    for key, name, stn, mach, ratio, flow in STATIONS:
        f = st[stn]
        w = f.w * (1 + d.bpr) if flow == "total" else f.w
        area = flux_area(w * LBM_TO_KG, f.t / 1.8, f.p * PSI_TO_PA, mach,
                         float(gas.gamma(f.t, f.far)))
        if ratio is None:   # turbines: HPT sits at the HP spool radius, HPT exit slightly up
            rm = r_mean_hp * (1.0 if key == "41" else 1.08)
            radii[key] = (name, *hub_tip(area, r_mean=rm))
        else:
            radii[key] = (name, *hub_tip(area, ratio))
    radii["24"] = ("HPC inlet", *hpc_station(st["24"], s))
    radii["30"] = ("HPC exit", *hpc_station(st["30"], s))

    # Axial positions: each component starts where the previous one ends.
    # Core passage rows start at the LPC (the fan is a separate disc in comps).
    x, rows, comps = 0.0, [], []
    for name, stages, aspect, front, back in COMPONENTS:
        if stages == 0:
            length = LENGTHS[name] * r_mean_hp
        else:
            heights = [radii[k][2] - radii[k][1] for k in (front, back)]
            chord = np.mean(heights) / aspect
            length = stages * STAGE_LENGTH * chord * math.cos(math.radians(STAGGER))
        comps.append({"component": name, "stages": stages,
                      "x_start_mm": 1000 * x, "x_end_mm": 1000 * (x + length),
                      "r_hub_mm": 1000 * radii[front][1], "r_tip_mm": 1000 * radii[front][2]})
        if name != "Fan":
            for k, xi in ((front, x), (back, x + length)):
                if rows and rows[-1]["station"] == k:
                    continue                          # shared station, listed once
                nm, rh, rt = radii[k]
                rows.append({"station": k, "name": nm, "x_mm": 1000 * xi,
                             "r_hub_mm": 1000 * rh, "r_tip_mm": 1000 * rt})
        x += length
    return pd.DataFrame(rows).round(1), pd.DataFrame(comps).round(1)


def health_colours() -> pd.DataFrame:
    """HPC colour at four moments of demo engine 52, plus the failure reference.
    Health is smoothed over 10 cycles (single cycles are noisy) and clipped to 0-1."""
    demo = pd.read_csv(ROOT / "data" / "processed" / "twin_state_demo.csv")
    demo["hi_smooth"] = demo["hi"].rolling(10, min_periods=1).mean().clip(0, 1)
    last = int(demo["cycle"].max())
    mid = int(demo.loc[demo["hi_smooth"] <= 0.7, "cycle"].min())   # first cycle at 0.7
    picks = [("new", 10), ("mid-wear", mid), ("now (last cycle)", last)]
    cmap = colormaps["RdYlGn"]
    rows = []
    for label, cyc in picks:
        r = demo[demo["cycle"] == cyc].iloc[0]
        rows.append({"label": label, "cycle": cyc, "hi": r["hi_smooth"],
                     "d_eta_pct": r["d_eta_pct"], "rul_pred": r["rul_pred"],
                     "status": r["engine_status"]})
    rows.append({"label": "typical failure (reference)", "cycle": np.nan, "hi": 0.0,
                 "d_eta_pct": -1.49, "rul_pred": 0, "status": "ACTION"})
    out = pd.DataFrame(rows)
    rgb = [tuple(int(round(255 * c)) for c in cmap(h)[:3]) for h in out["hi"]]
    out["hex"] = ["#{:02X}{:02X}{:02X}".format(*c) for c in rgb]
    out["rgb"] = [f"{c[0]} {c[1]} {c[2]}" for c in rgb]
    return out.round(3)


def plot(lay, comps, colours, path):
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    c_idx = comps.set_index("component")
    for ax, (_, c) in zip(axes, colours.iloc[:3].iterrows()):
        fan = c_idx.loc["Fan"]
        ax.fill_between([fan["x_start_mm"], fan["x_end_mm"]], fan["r_hub_mm"],
                        fan["r_tip_mm"], color="0.85", ec="k", lw=0.8)
        ax.fill_between(lay["x_mm"], lay["r_hub_mm"], lay["r_tip_mm"],
                        color="0.85", ec="k", lw=0.8)
        hpc = c_idx.loc["HPC"]
        m = lay[(lay["x_mm"] >= hpc["x_start_mm"]) & (lay["x_mm"] <= hpc["x_end_mm"])]
        ax.fill_between(m["x_mm"], m["r_hub_mm"], m["r_tip_mm"], color=c["hex"],
                        ec="k", lw=1.2)
        for name, p in c_idx.iterrows():
            if name != "Duct":
                label = "Burner" if name == "Combustor" else name
                ax.text((p["x_start_mm"] + p["x_end_mm"]) / 2, 150, label,
                        ha="center", fontsize=9)
        ax.set_title(f"{c['label']}: cycle {int(c['cycle'])} - health {c['hi']:.2f}, "
                     f"HPC eff. {c['d_eta_pct']:+.2f} %, RUL {c['rul_pred']:.0f}, "
                     f"{c['status']}", fontsize=10, loc="left")
        ax.set_ylim(0, 1700)
        ax.set_aspect("equal")
        ax.set_ylabel("radius (mm)")
    axes[-1].set_xlabel("axial (mm)")
    fig.suptitle("Engine half-section (rough, from Week 2 flows), HPC coloured by twin "
                 "health - demo engine 52")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    lay, comps = layout()
    colours = health_colours()
    (ROOT / "cad").mkdir(exist_ok=True)
    lay.to_csv(ROOT / "cad" / "engine_layout.csv", index=False)
    comps.to_csv(ROOT / "cad" / "components.csv", index=False)
    colours.to_csv(ROOT / "cad" / "health_colours.csv", index=False)
    fig = ROOT / "reports" / "figures" / "14_engine_layout.png"
    plot(lay, comps, colours, fig)

    pd.set_option("display.width", 120)
    print("Flow passage (mm):\n", lay.to_string(index=False))
    print("\nComponents (mm):\n", comps.to_string(index=False))
    print("\nHPC colours for the cutaway:\n",
          colours[["label", "cycle", "hi", "d_eta_pct", "rul_pred", "status", "hex", "rgb"]]
          .to_string(index=False))
    print(f"\nWrote cad/engine_layout.csv, cad/components.csv, cad/health_colours.csv, "
          f"{fig.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
