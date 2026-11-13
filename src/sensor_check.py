"""Week 4, step 1: does each sensor agree with the physics?

Real sensors fail: a thermocouple drifts, a pressure line leaks. If the twin
trusts a bad sensor, it reports engine wear that is not there (or misses wear
that is). The physics model gives us a way to catch this.

Idea (analytic redundancy):
  The HPC has only 2 health numbers (efficiency, flow capacity), but we
  measure 7 sensors. So the sensors are redundant. Any 6 of them already say
  what the health is, and the physics then PREDICTS the 7th:
      residual = measured sensor - physics prediction from the other 6
  For a good sensor the residual is just noise. For a faulty sensor it grows,
  and it grows most for THAT sensor, which tells us which one is bad.

Why "the other 6" and not all 7: T24 is so much quieter than the rest that
the all-7 fit follows it almost exactly. A bad T24 would then drag the health
numbers along with it and its own residual would stay small.

Two things learned from TRAIN engines (assumed to have no sensor faults):
  1. Model mismatch. The 0D model is not perfect (Week 2: T50 response about
     16 % low), so residuals creep as wear grows, e.g. T50 reads +0.28 %
     above the model near failure. Learned per sensor as
         expected residual = a * damage + b * damage^2
     (damage = average HPC loss in %, from the other 6 sensors), then
     subtracted. Without this, 52 of the 100 clean train engines raise a
     false alarm near the end of life, exactly when the twin matters most.
  2. Scatter of the corrected residual per sensor. z = residual / scatter.

Alarm rule: the same sensor has the largest |z|, and |z| > Z_ALARM, for
PERSIST cycles in a row. One fault per engine is assumed. Checks start after
the 20-cycle baseline, because health is measured against those cycles.

When a sensor is alarmed, the twin stops using it: from that cycle on, health
comes from the other 6 sensors only.

Run from the repo root:
    python -m src.sensor_check
"""
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import load_test, load_train
from src.features import sensor_noise
from src.physics.health_fit import (BASELINE_CYCLES, FIT_SENSORS, fit_health,
                                    influence_matrix, sensor_deviations)
from src.physics.off_design import OffDesignEngine, load_design

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
NAMES = list(FIT_SENSORS)      # T24, T30, T50, P30, BPR, W31, W32
Z_ALARM = 4.0                  # residual must exceed 4 x its normal scatter ...
PERSIST = 5                    # ... on the same sensor for 5 cycles in a row


@dataclass
class Calibration:
    """Everything learned from train engines. Applied unchanged to test."""
    H: pd.DataFrame            # influence matrix (physics), 8 sensors x 2
    noise: pd.Series           # raw sensor noise, % (weights for the fit)
    mismatch: pd.DataFrame     # a, b per sensor (model mismatch curve)
    scatter: pd.Series         # std of corrected residual per sensor, %


def leave_one_out(dev: pd.DataFrame, H: pd.DataFrame, noise: pd.Series):
    """For each sensor: fit health from the OTHER 6, predict this one.

    Returns
      resid  : rows x 7, measured - predicted (% change)
      health : dict sensor -> (d_eta, d_flow) arrays, fitted without it
    """
    Hm = H.loc[NAMES].to_numpy()                   # 7 x 2
    w = 1.0 / noise[NAMES].to_numpy()
    Y = dev[NAMES].to_numpy()                      # rows x 7
    resid = np.zeros_like(Y)
    health = {}
    for i, name in enumerate(NAMES):
        keep = [j for j in range(len(NAMES)) if j != i]
        A = Hm[keep] * w[keep, None]               # 6 x 2, weighted
        sol = np.linalg.pinv(A) @ (Y[:, keep] * w[keep]).T     # 2 x rows
        resid[:, i] = Y[:, i] - Hm[i] @ sol
        health[name] = (sol[0], sol[1])
    return pd.DataFrame(resid, columns=NAMES, index=dev.index), health


def damage_of(health_pair) -> np.ndarray:
    """Average HPC loss in % (positive = worn). About 1.5 at failure."""
    d_eta, d_flow = health_pair
    return -(d_eta + d_flow) / 2


def mismatch_curve(damage: np.ndarray, coef: pd.Series) -> np.ndarray:
    return coef["a"] * damage + coef["b"] * damage ** 2


def calibrate(train_raw: pd.DataFrame) -> Calibration:
    H = influence_matrix(OffDesignEngine(load_design()))
    noise = sensor_noise(train_raw)
    dev = sensor_deviations(train_raw)
    resid, health = leave_one_out(dev, H, noise)
    use = (dev["cycle"] > BASELINE_CYCLES).to_numpy()

    mismatch, scatter = {}, {}
    for name in NAMES:
        dmg = damage_of(health[name])
        X = np.c_[dmg, dmg ** 2][use]
        (a, b), *_ = np.linalg.lstsq(X, resid[name].to_numpy()[use], rcond=None)
        mismatch[name] = {"a": a, "b": b}
        corrected = resid[name].to_numpy() - mismatch_curve(dmg, pd.Series({"a": a, "b": b}))
        scatter[name] = corrected[use].std()
    return Calibration(H, noise, pd.DataFrame(mismatch).T, pd.Series(scatter))


def z_scores(dev: pd.DataFrame, cal: Calibration):
    """z per sensor per cycle, plus the leave-one-out health (needed later)."""
    resid, health = leave_one_out(dev, cal.H, cal.noise)
    z = pd.DataFrame(index=dev.index)
    for name in NAMES:
        corr = resid[name] - mismatch_curve(damage_of(health[name]), cal.mismatch.loc[name])
        z[name] = corr / cal.scatter[name]
    return z, health


def alarms(dev: pd.DataFrame, z: pd.DataFrame) -> pd.Series:
    """Name of the alarmed sensor per row ('' = none). Once raised, it stays."""
    absz = z[NAMES].abs()
    worst = absz.idxmax(axis=1).where(absz.max(axis=1) > Z_ALARM, "")
    worst = worst.where(dev["cycle"] > BASELINE_CYCLES, "")

    out = pd.Series("", index=dev.index, dtype=object)
    for _, idx in dev.groupby("unit").groups.items():
        run, last, alarmed = 0, "", ""
        for i in idx:
            if not alarmed:
                w = worst.at[i]
                run = run + 1 if (w and w == last) else (1 if w else 0)
                last = w
                if run >= PERSIST:
                    alarmed = w
            out.at[i] = alarmed
    return out


def check(raw: pd.DataFrame, cal: Calibration) -> pd.DataFrame:
    """Full sensor check for one dataset. One row per engine per cycle.

    Columns: z_<sensor>, alarm (sensor name or ''), health with all 7 sensors
    (d_eta_all, d_flow_all) and the health the twin TRUSTS (d_eta_pct,
    d_flow_pct): all 7 sensors until an alarm, then the other 6.
    """
    dev = sensor_deviations(raw)
    z, health_loo = z_scores(dev, cal)
    full = fit_health(dev, cal.H, cal.noise)

    out = dev[["unit", "cycle", "rul"]].copy()
    for name in NAMES:
        out[f"z_{name}"] = z[name]
    out["alarm"] = alarms(dev, z)
    out["d_eta_all"] = full["d_eta_pct"].to_numpy()
    out["d_flow_all"] = full["d_flow_pct"].to_numpy()

    out["d_eta_pct"] = out["d_eta_all"].copy()
    out["d_flow_pct"] = out["d_flow_all"].copy()
    for name in NAMES:
        rows = (out["alarm"] == name).to_numpy()
        out.loc[rows, "d_eta_pct"] = health_loo[name][0][rows]
        out.loc[rows, "d_flow_pct"] = health_loo[name][1][rows]
    return out


def plot_mismatch(train_raw: pd.DataFrame, cal: Calibration, path: Path) -> None:
    """Why the mismatch correction is needed: residual vs life, T24 and T50."""
    dev = sensor_deviations(train_raw)
    resid, _ = leave_one_out(dev, cal.H, cal.noise)
    z, _ = z_scores(dev, cal)
    late = dev["rul"] <= 250
    rul = dev.loc[late, "rul"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for row, name in enumerate(["T24", "T50"]):
        for col, (series, when) in enumerate([
                (resid[name] / cal.scatter[name], "before correction"),
                (z[name], "after mismatch correction")]):
            ax = axes[row, col]
            s = series[late]
            lo = s.groupby(rul).quantile(0.05).sort_index()
            hi = s.groupby(rul).quantile(0.95).sort_index()
            med = s.groupby(rul).median().sort_index()
            ax.fill_between(-lo.index, lo.values, hi.values, color="C0", alpha=0.2,
                            label="5-95 % of engines")
            ax.plot(-med.index, med.values, color="C0", lw=2, label="median")
            ax.axhline(Z_ALARM, color="C3", ls="--", lw=1, label=f"alarm level (|z| = {Z_ALARM:g})")
            ax.axhline(-Z_ALARM, color="C3", ls="--", lw=1)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_title(f"{name}, {when}")
            ax.set_xlim(-250, 0)
        axes[row, 0].set_ylabel(f"{name} residual / normal scatter (z)")
    for ax in axes[1]:
        ax.set_xlabel("cycles before failure (0 = failure)")
    axes[0, 0].legend(loc="lower left", fontsize=8)
    fig.suptitle("FD001 train engines: each sensor vs the physics prediction from the other 6")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    train_raw, test_raw = load_train(), load_test()
    cal = calibrate(train_raw)

    table = cal.mismatch.copy()
    table["scatter_pct"] = cal.scatter
    table["noise_pct"] = cal.noise[NAMES]
    table.round(5).to_csv(REPORTS / "sensor_check_calibration.csv", index_label="sensor")
    print("Calibration from train engines (residual = % change):")
    print(f"  {'sensor':<6}{'mismatch at failure':>21}{'scatter':>9}{'raw noise':>11}")
    for name in NAMES:
        at_fail = mismatch_curve(np.array(1.5), cal.mismatch.loc[name])
        print(f"  {name:<6}{at_fail:+20.3f} %{cal.scatter[name]:8.3f} %"
              f"{cal.noise[name]:10.3f} %")

    print(f"\nAlarm rule: |z| > {Z_ALARM:g} on the same sensor for {PERSIST} cycles")
    for label_, raw in [("train", train_raw), ("test", test_raw)]:
        res = check(raw, cal)
        alarmed = res.groupby("unit")["alarm"].last()
        n_alarm = (alarmed != "").sum()
        print(f"  {label_}: {n_alarm} of {len(alarmed)} clean engines raise an alarm "
              f"(false alarms){': ' + ', '.join(alarmed[alarmed != ''].tolist()) if n_alarm else ''}")
        print(f"         largest |z| seen anywhere: "
              f"{res.loc[res['cycle'] > BASELINE_CYCLES, [f'z_{n}' for n in NAMES]].abs().max().max():.1f}")

    fig = REPORTS / "figures" / "10_sensor_residual_mismatch.png"
    plot_mismatch(train_raw, cal, fig)
    print(f"\nSaved reports/sensor_check_calibration.csv, reports/figures/{fig.name}")


if __name__ == "__main__":
    main()