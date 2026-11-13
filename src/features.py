"""Week 3, step 1: one feature table for train AND test engines.

Each row = one engine, one cycle. Columns:
  - smoothed sensors (10-cycle rolling mean, per engine)
  - physics health from Week 2: d_eta_pct, d_flow_pct (HPC efficiency and
    flow capacity change, from gas path analysis)
  - hi: physics health index, 1 = new, 0 = typical failure level
  - rul (true), rul_capped (target used for training, capped at 125)

Key rule: anything LEARNED from data (sensor noise weights, failure-level
loss for the health index) comes from the TRAIN set only, then is applied
unchanged to test. Otherwise the test score would be cheating.

Run from the repo root:
    python -m src.features
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import load_test, load_train
from src.physics.health_fit import (BASELINE_CYCLES, FIT_SENSORS, SMOOTH,
                                    fit_health, influence_matrix,
                                    sensor_deviations)
from src.physics.off_design import OffDesignEngine, load_design

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures"

# 14 sensors that change with wear in FD001 (Week 1). The other 7 are flat.
SENSOR_COLS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11",
               "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
RUL_CAP = 125   # early life looks "healthy"; capping is the standard C-MAPSS target


def sensor_noise(train: pd.DataFrame) -> pd.Series:
    """Scatter of each fit sensor (% change) in the healthy first cycles. Train only."""
    pct = {}
    for name, col in FIT_SENSORS.items():
        base = train[col].groupby(train["unit"]).transform(
            lambda s: s.iloc[:BASELINE_CYCLES].mean())
        pct[name] = 100 * (train[col] / base - 1)
    pct = pd.DataFrame(pct)
    return pct[train["cycle"] <= BASELINE_CYCLES].std()


def smoothed_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """10-cycle rolling mean of each useful sensor, per engine (uses past cycles only)."""
    out = df[SENSOR_COLS].groupby(df["unit"]).transform(
        lambda s: s.rolling(SMOOTH, min_periods=1).mean())
    return out


def build(df: pd.DataFrame, H: pd.DataFrame, noise: pd.Series) -> pd.DataFrame:
    """Feature table for one dataset (train or test)."""
    health = fit_health(sensor_deviations(df), H, noise)
    feats = df[["unit", "cycle", "rul"]].copy()
    feats[SENSOR_COLS] = smoothed_sensors(df)
    feats["d_eta_pct"] = health["d_eta_pct"].to_numpy()
    feats["d_flow_pct"] = health["d_flow_pct"].to_numpy()
    feats["rul_capped"] = feats["rul"].clip(upper=RUL_CAP)
    return feats


def add_health_index(feats: pd.DataFrame, eta_fail: float, flow_fail: float) -> None:
    """hi = 1 - (fraction of the typical failure-level loss already used up).

    Averages efficiency and flow (they are 99 % correlated, so averaging
    mostly cuts noise). hi can go a bit below 0 or above 1: some engines
    fail with more or less damage than average, and early cycles are noisy.
    """
    used = 0.5 * (feats["d_eta_pct"] / eta_fail + feats["d_flow_pct"] / flow_fail)
    feats["hi"] = 1 - used


def plot(train: pd.DataFrame, test: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for _, g in train.groupby("unit"):
        ax.plot(-g["rul"], g["hi"], color="0.8", lw=0.5)
    med = train.groupby("rul")["hi"].median().sort_index()
    med = med[med.index <= 250]
    ax.plot(-med.index, med.values, color="C3", lw=2, label="fleet median")
    ax.axhline(1, color="k", lw=0.6)
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.set_xlim(-250, 0)
    ax.set_xlabel("cycles before failure (0 = failure)")
    ax.set_ylabel("health index (1 = new, 0 = typical failure)")
    ax.set_title("Train engines (100)")
    ax.legend(loc="lower left")

    ax = axes[1]
    last = test.groupby("unit").tail(1)
    ax.scatter(last["rul"], last["hi"], s=14, color="C0")
    ax.set_xlabel("true RUL at last test cycle")
    ax.set_ylabel("health index at last test cycle")
    ax.set_title("Test engines (100): physics HI vs true RUL")
    fig.suptitle("FD001: physics health index from the Week 2 gas path analysis")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    train_raw, test_raw = load_train(), load_test()
    H = influence_matrix(OffDesignEngine(load_design()))
    noise = sensor_noise(train_raw)            # train only

    train = build(train_raw, H, noise)
    test = build(test_raw, H, noise)

    # Typical loss at failure, from TRAIN engines only (mean of last 10 cycles)
    end = train[train["rul"] < 10].groupby("unit")[["d_eta_pct", "d_flow_pct"]].mean()
    eta_fail, flow_fail = end["d_eta_pct"].mean(), end["d_flow_pct"].mean()
    add_health_index(train, eta_fail, flow_fail)
    add_health_index(test, eta_fail, flow_fail)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train.round(5).to_csv(OUT_DIR / "features_train.csv", index=False)
    test.round(5).to_csv(OUT_DIR / "features_test.csv", index=False)

    print(f"Failure-level loss (train mean): efficiency {eta_fail:+.2f} %, "
          f"flow {flow_fail:+.2f} %")
    print(f"train: {len(train)} rows, test: {len(test)} rows, "
          f"{train.shape[1]} columns")

    last = test.groupby("unit").tail(1)
    rho = last[["hi", "rul"]].corr(method="spearman").iloc[0, 1]
    print(f"\nTest engines, last cycle: Spearman rho(HI, true RUL) = {rho:+.3f}")
    print("Health index at last test cycle, by true RUL band:")
    bands = pd.cut(last["rul"], [0, 25, 50, 100, 150, 200])
    print(last.groupby(bands, observed=True)["hi"].agg(["count", "mean"]).round(2).to_string())

    fig = FIG_DIR / "07_health_index_train_test.png"
    plot(train, test, fig)
    print(f"\nSaved data/processed/features_train.csv, features_test.csv")
    print(f"      reports/figures/{fig.name}")


if __name__ == "__main__":
    main()