"""Week 2, step 5: estimate HPC health for every engine, every cycle.

Method: gas path analysis (GPA), the standard way engine makers do this.

1. Influence matrix H, from the step 4 model:
       sensor % change = H @ [efficiency % change, flow capacity % change]
   Step 4 showed the response is close to linear for small damage.
2. For each engine, sensor % change is measured against THAT engine's own
   first 20 cycles. This removes build-to-build differences between engines.
3. Each cycle: find the two health numbers that best explain the 7 sensor
   changes (weighted least squares). Weights = 1 / sensor noise, so a noisy
   sensor counts for less.

Fuel flow is left out of the fit (it relies on Ps30, which is physically
suspect - see step 3) and is used only as an independent check.

Run from the repo root:
    python -m src.physics.health_fit
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.data_loader import load_train
from src.physics.off_design import Health, OffDesignEngine, load_design, sensors

REPORTS = Path(__file__).resolve().parents[2] / "reports"
FIT_SENSORS = {"T24": "s2", "T30": "s3", "T50": "s4", "P30": "s7",
               "BPR": "s15", "W31": "s20", "W32": "s21"}
BASELINE_CYCLES = 20
SMOOTH = 10            # rolling mean window, cycles


def influence_matrix(eng: OffDesignEngine, step: float = 0.005) -> pd.DataFrame:
    """% sensor change per 1 % health change, by central differences."""
    cols = {}
    for name, field in [("eta", "eta_hpc"), ("flow", "flow_hpc")]:
        up = sensors(eng.solve(Health(**{field: +step})))
        dn = sensors(eng.solve(Health(**{field: -step})))
        mid = sensors(eng.solve())
        cols[name] = {k: (up[k] - dn[k]) / mid[k] / (2 * step) for k in mid}
    return pd.DataFrame(cols)        # rows: sensors, columns: eta, flow


def sensor_deviations(df: pd.DataFrame) -> pd.DataFrame:
    """% change of each sensor from the engine's own baseline, smoothed."""
    out = df[["unit", "cycle", "rul"]].copy()
    for name, col in {**FIT_SENSORS, "fuel": None}.items():
        x = df["s12"] * df["s11"] if name == "fuel" else df[col]
        base = x.groupby(df["unit"]).transform(lambda s: s.iloc[:BASELINE_CYCLES].mean())
        dev = 100 * (x / base - 1)
        out[name] = dev.groupby(df["unit"]).transform(
            lambda s: s.rolling(SMOOTH, min_periods=1).mean())
    return out


def fit_health(dev: pd.DataFrame, H: pd.DataFrame, noise: pd.Series) -> pd.DataFrame:
    """Weighted least squares for every row at once."""
    names = list(FIT_SENSORS)
    w = 1.0 / noise[names].to_numpy()              # weight per sensor
    A = H.loc[names].to_numpy() * w[:, None]       # 7 x 2
    Y = dev[names].to_numpy() * w                  # rows x 7
    solution = np.linalg.pinv(A) @ Y.T             # 2 x rows
    out = dev[["unit", "cycle", "rul"]].copy()
    out["d_eta_pct"], out["d_flow_pct"] = solution
    # how much of each sensor the fit explains
    fitted = (H.loc[names].to_numpy() @ solution).T
    for i, n in enumerate(names):
        out[f"fit_{n}"] = fitted[:, i]
    return out


def plot(health: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, col, title in [(axes[0], "d_eta_pct", "HPC efficiency change (%)"),
                           (axes[1], "d_flow_pct", "HPC flow capacity change (%)")]:
        for _, g in health.groupby("unit"):
            ax.plot(-g["rul"], g[col], color="0.78", lw=0.5)
        med = health.groupby("rul")[col].median().sort_index()
        med = med[med.index <= 250]
        ax.plot(-med.index, med.values, color="C3", lw=2, label="fleet median")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(title)
        ax.set_xlabel("cycles before failure (0 = failure)")
        ax.set_xlim(-250, 0)
    axes[0].legend(loc="lower left")
    fig.suptitle("FD001: HPC health estimated from 7 sensors with the physics model")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    eng = OffDesignEngine(load_design())
    H = influence_matrix(eng)
    print("Influence matrix (% sensor change per 1 % health change):")
    print(H.round(3).to_string(), "\n")

    df = load_train()
    dev = sensor_deviations(df)
    # noise = scatter of the UNSMOOTHED % change in the healthy period
    raw = df.copy()
    for n, c in FIT_SENSORS.items():
        base = raw[c].groupby(raw["unit"]).transform(lambda s: s.iloc[:BASELINE_CYCLES].mean())
        raw[n] = 100 * (raw[c] / base - 1)
    noise = raw[raw["cycle"] <= BASELINE_CYCLES][list(FIT_SENSORS)].std()

    health = fit_health(dev, H, noise)
    health.round(5).to_csv(REPORTS / "hpc_health_fd001.csv", index=False)

    # Health at failure, per engine (mean of last 10 cycles)
    end = health[health["rul"] < 10].groupby("unit")[["d_eta_pct", "d_flow_pct"]].mean()
    end.round(4).to_csv(REPORTS / "hpc_loss_per_engine_fd001.csv")
    print("HPC health at failure (100 engines):")
    for col, name in [("d_eta_pct", "efficiency"), ("d_flow_pct", "flow capacity")]:
        s = end[col]
        print(f"  {name:<14} mean {s.mean():+.2f} %  std {s.std():.2f}  "
              f"range {s.min():+.2f} to {s.max():+.2f}")

    # Check 1: full nonlinear model at the fleet-mean damage vs measured change
    h = Health(eta_hpc=end["d_eta_pct"].mean() / 100, flow_hpc=end["d_flow_pct"].mean() / 100)
    new, worn = sensors(eng.solve()), sensors(eng.solve(h))
    meas = dev[dev["rul"] < 10][list(FIT_SENSORS) + ["fuel"]].mean()
    print("\nFleet at failure: physics model (nonlinear) vs data, % change")
    print(f"  {'sensor':<6}{'model':>8}{'data':>8}")
    for k in list(FIT_SENSORS) + ["fuel"]:
        tag = "   <- not used in fit" if k == "fuel" else ""
        print(f"  {k:<6}{100 * (worn[k] / new[k] - 1):+8.3f}{meas[k]:+8.3f}{tag}")

    # Check 2: is the health estimate a good degradation signal?
    late = health[health["rul"] <= 150]
    rho_eta = spearmanr(late["d_eta_pct"], late["rul"]).statistic
    rho_flow = spearmanr(late["d_flow_pct"], late["rul"]).statistic
    # Same test for the best single sensor from Week 1 (Ps30), same smoothing
    ps30 = df["s11"].groupby(df["unit"]).transform(lambda s: s.rolling(SMOOTH, min_periods=1).mean())
    rho_ps30 = spearmanr(ps30[df["rul"] <= 150], df.loc[df["rul"] <= 150, "rul"]).statistic
    print(f"\nSpearman |rho| with RUL, last 150 cycles, 10-cycle smoothing:")
    print(f"  HPC efficiency estimate {abs(rho_eta):.3f}   flow estimate {abs(rho_flow):.3f}   "
          f"best single sensor Ps30 {abs(rho_ps30):.3f}")

    fig = REPORTS / "figures" / "06_hpc_health_vs_cycles.png"
    plot(health, fig)
    print(f"\nSaved reports/hpc_health_fd001.csv, reports/hpc_loss_per_engine_fd001.csv,")
    print(f"      reports/figures/{fig.name}")


if __name__ == "__main__":
    main()
