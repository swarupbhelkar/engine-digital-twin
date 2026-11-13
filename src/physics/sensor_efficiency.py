"""Week 2, step 1: HPC efficiency straight from the sensors.

The HPC has temperature probes at inlet (T24) and outlet (T30) and a
pressure probe at outlet (P30). If we also knew the inlet pressure P24,
the gas equations would give HPC efficiency directly, cycle by cycle.

C-MAPSS has no P24 sensor. So we make ONE assumption: the efficiency of
the low-pressure compression (fan core side + LPC, from station 2 to 24).
From T2 -> T24 and that efficiency we get the pressure ratio, hence P24.

Run from the repo root:
    python -m src.physics.sensor_efficiency
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data_loader import load_train
from src.physics.gas import compressor_efficiency, compressor_pressure_ratio

ETA_LOW = 0.88      # assumed efficiency, station 2 -> 24 (fan core + LPC)
HEALTHY_CYCLES = 20  # first cycles of each engine, treated as "new engine"
FIG_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"

# C-MAPSS column names for the stations we need
T2, T24, T30, P2, P30 = "s1", "s2", "s3", "s5", "s7"


def hpc_efficiency(df: pd.DataFrame, eta_low: float = ETA_LOW) -> pd.Series:
    """HPC isentropic efficiency for every row of a C-MAPSS dataframe."""
    pr_low = compressor_pressure_ratio(df[T2], df[T24], eta_low)
    p24 = df[P2] * pr_low
    return compressor_efficiency(df[T24], df[T30], df[P30] / p24)


def main() -> None:
    df = load_train()
    df["eta_hpc"] = hpc_efficiency(df)
    # Sensor noise is large, so smooth each engine with a 10-cycle rolling mean
    df["eta_smooth"] = df.groupby("unit")["eta_hpc"].transform(
        lambda s: s.rolling(10, min_periods=1).mean()
    )

    healthy = df[df["cycle"] <= HEALTHY_CYCLES]
    worn = df[df["rul"] < HEALTHY_CYCLES]   # last 20 cycles before failure

    # 1) What does a new engine look like?
    print("Healthy engine (fleet mean, first 20 cycles):")
    for col, name in [(T2, "T2"), (T24, "T24"), (T30, "T30"), (P2, "P2"), (P30, "P30")]:
        print(f"  {name:4s} = {healthy[col].mean():8.2f}")
    pr_low = compressor_pressure_ratio(healthy[T2].mean(), healthy[T24].mean(), ETA_LOW)
    print(f"  -> LPC pressure ratio {pr_low:.2f}, "
          f"HPC pressure ratio {healthy[P30].mean() / (healthy[P2].mean() * pr_low):.2f}")

    # 2) HPC efficiency, new vs worn
    eta_new, eta_worn = healthy["eta_hpc"].mean(), worn["eta_hpc"].mean()
    print(f"\nHPC efficiency: new {eta_new:.4f}, last 20 cycles {eta_worn:.4f}, "
          f"change {100 * (eta_worn - eta_new):+.2f} points")

    # 3) How much does the one assumption matter?
    print("\nSensitivity to the assumed LPC efficiency:")
    for eta_low in (0.85, 0.88, 0.91):
        e = hpc_efficiency(df, eta_low)
        new, old = e[df["cycle"] <= HEALTHY_CYCLES].mean(), e[df["rul"] < HEALTHY_CYCLES].mean()
        print(f"  eta_low {eta_low:.2f}: HPC new {new:.4f}, change {100 * (old - new):+.2f} points")

    # 4) Plot: every engine against cycles-to-failure
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for _, g in df.groupby("unit"):
        ax.plot(-g["rul"], g["eta_smooth"], color="0.75", lw=0.6)
    median = df.groupby("rul")["eta_smooth"].median().sort_index()
    median = median[median.index <= 250]
    ax.plot(-median.index, median.values, color="C3", lw=2, label="fleet median")
    ax.set_xlabel("cycles before failure (0 = failure)")
    ax.set_ylabel("HPC isentropic efficiency")
    ax.set_title("FD001: HPC efficiency computed from T24, T30, P30 (10-cycle mean)")
    ax.set_xlim(-250, 0)
    ax.legend()
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "05_hpc_efficiency_from_sensors.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out.relative_to(FIG_DIR.parents[1])}")


if __name__ == "__main__":
    main()
