"""Week 4, step 2: break sensors on purpose and see if the twin notices.

C-MAPSS has no sensor faults, so we add them ourselves to the TEST engines
(the sensor check was calibrated on train only, so this is a fair test).

Two fault types, one sensor at a time, starting at a random cycle:
  bias  : sudden offset (a connector or calibration fault)
  drift : offset that grows steadily (a thermocouple ageing). It reaches
          the given size after 50 cycles and keeps growing.
Sizes are multiples of each sensor's own noise (2, 3, 4, 6 x noise), so a
"4 x" fault is equally hard to see on every sensor. In physical units this
is very different: 4 x noise on T24 is ~1.2 degR, on T30 ~15.5 degR.

Questions answered:
  1. Detection: does an alarm go off after the fault starts?
  2. Isolation: does it name the right sensor?
  3. Delay: how many cycles from fault start to alarm?
  4. Does it matter? Remaining life predicted from the physics health,
     with the faulty sensor trusted (naive) vs with the twin's check
     (trusted), compared with the prediction on the clean engine.

Only test engines with at least 80 cycles are used (77 engines), so each
fault has time to show. Fault start: random cycle between 30 and
(last cycle - 30), same for every run (seed 0).

Run from the repo root (about 2 minutes):
    python -m src.fault_injection
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import load_test, load_train
from src.features import RUL_CAP, add_health_index
from src.physics.health_fit import FIT_SENSORS
from src.rul_rf import make_model
from src.sensor_check import NAMES, calibrate, check

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
SIZES = [2, 3, 4, 6]               # x sensor noise
TYPES = ["bias", "drift"]
DRIFT_CYCLES = 50                  # drift reaches full size after 50 cycles
MIN_CYCLES = 80
PHYS_COLS = ["d_eta_pct", "d_flow_pct", "hi"]


def inject(df: pd.DataFrame, sensor: str, kind: str, size: float,
           onset: pd.Series, noise_pct: float) -> pd.DataFrame:
    """Copy of df with a fault on one sensor. size in multiples of noise."""
    out = df.copy()
    col = FIT_SENSORS[sensor]
    start = out["unit"].map(onset)
    since = (out["cycle"] - start).clip(lower=0)          # cycles since onset
    on = out["cycle"] >= start
    frac = np.where(on, 1.0, 0.0) if kind == "bias" else since / DRIFT_CYCLES
    out[col] = out[col] * (1 + size * noise_pct / 100 * frac)
    return out


def health_table(res: pd.DataFrame, eta_fail: float, flow_fail: float,
                 which: str) -> pd.DataFrame:
    """Physics features from a check() result. which = 'trusted' or 'naive'."""
    t = res[["unit", "cycle", "rul"]].copy()
    if which == "trusted":
        t["d_eta_pct"], t["d_flow_pct"] = res["d_eta_pct"], res["d_flow_pct"]
    else:
        t["d_eta_pct"], t["d_flow_pct"] = res["d_eta_all"], res["d_flow_all"]
    add_health_index(t, eta_fail, flow_fail)
    return t


def main() -> None:
    train_raw, test_raw = load_train(), load_test()
    cal = calibrate(train_raw)

    # RUL model on physics health only (Week 3 model C), trained on clean train
    tr = check(train_raw, cal)
    end = tr[tr["rul"] < 10].groupby("unit")[["d_eta_pct", "d_flow_pct"]].mean()
    eta_fail, flow_fail = end["d_eta_pct"].mean(), end["d_flow_pct"].mean()
    tr_feat = health_table(tr, eta_fail, flow_fail, "trusted")
    rf = make_model().fit(tr_feat[PHYS_COLS], tr_feat["rul"].clip(upper=RUL_CAP))

    lens = test_raw.groupby("unit")["cycle"].max()
    units = lens[lens >= MIN_CYCLES].index
    test = test_raw[test_raw["unit"].isin(units)].copy()
    rng = np.random.default_rng(0)
    onset = pd.Series(rng.integers(30, (lens[units] - 30).to_numpy()), index=units)

    clean = check(test, cal)
    clean_last = health_table(clean, eta_fail, flow_fail, "trusted").groupby("unit").tail(1)
    rul_clean = rf.predict(clean_last[PHYS_COLS])

    rows = []
    for kind in TYPES:
        for size in SIZES:
            print(f"{kind} {size} x noise ...", flush=True)
            for sensor in NAMES:
                faulty = inject(test, sensor, kind, size, onset, cal.noise[sensor])
                res = check(faulty, cal)
                last = res.groupby("unit")["alarm"].last()
                first = res[res["alarm"] != ""].groupby("unit")["cycle"].min()
                delay = (first - onset).reindex(units)          # NaN = no alarm
                early = (delay < 0).fillna(False)

                rul = {}
                for which in ["naive", "trusted"]:
                    h = health_table(res, eta_fail, flow_fail, which).groupby("unit").tail(1)
                    rul[which] = rf.predict(h[PHYS_COLS])
                rows.append({
                    "type": kind, "size_x_noise": size, "sensor": sensor,
                    "size_physical_pct": round(size * cal.noise[sensor], 3),
                    "detected_pct": round(100 * ((last != "") & ~early).mean(), 1),
                    "isolated_pct": round(100 * ((last == sensor) & ~early).mean(), 1),
                    "early_alarm_pct": round(100 * early.mean(), 1),
                    "median_delay": delay[delay >= 0].median(),
                    "rul_shift_naive": round(np.abs(rul["naive"] - rul_clean).mean(), 1),
                    "rul_shift_trusted": round(np.abs(rul["trusted"] - rul_clean).mean(), 1),
                })

    res = pd.DataFrame(rows)
    res.to_csv(REPORTS / "fault_injection_results.csv", index=False)

    print(f"\n{len(units)} test engines, one fault each, per run")
    for kind in TYPES:
        sub = res[res["type"] == kind]
        print(f"\n{kind.upper()}: % of engines where the right sensor is named "
              f"(rows = sensor, columns = size x noise)")
        print(sub.pivot(index="sensor", columns="size_x_noise",
                        values="isolated_pct").loc[NAMES].to_string())
        print(f"{kind}: median cycles from fault start to alarm")
        print(sub.pivot(index="sensor", columns="size_x_noise",
                        values="median_delay").loc[NAMES].to_string())

    print("\nDoes it matter? Mean change in predicted RUL at the last cycle vs the")
    print("clean engine (cycles), physics-only RUL model, averaged over sensors:")
    summary = res.groupby(["type", "size_x_noise"])[["rul_shift_naive", "rul_shift_trusted"]].mean()
    print(summary.round(1).to_string())
    worst = res.sort_values("rul_shift_naive").iloc[-1]
    print(f"\nWorst case: {worst['type']} {worst['size_x_noise']} x noise on "
          f"{worst['sensor']}: RUL off by {worst['rul_shift_naive']:.0f} cycles if trusted, "
          f"{worst['rul_shift_trusted']:.0f} with the check")

    fig = REPORTS / "figures" / "11_fault_injection.png"
    plot(res, fig)
    print(f"\nSaved reports/fault_injection_results.csv, reports/figures/{fig.name}")


def plot(res: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, kind in zip(axes[:2], TYPES):
        sub = res[res["type"] == kind]
        for i, sensor in enumerate(NAMES):
            s = sub[sub["sensor"] == sensor]
            ax.plot(s["size_x_noise"], s["isolated_pct"], marker="o",
                    color=f"C{i}", label=sensor)
        ax.set_title(f"{kind}: right sensor named")
        ax.set_xlabel("fault size (x sensor noise)")
        ax.set_ylim(-3, 103)
        ax.set_xticks(SIZES)
    axes[0].set_ylabel("% of test engines")
    axes[0].legend(fontsize=8, loc="lower right")

    ax = axes[2]
    m = res.groupby("size_x_noise")[["rul_shift_naive", "rul_shift_trusted"]].mean()
    x = np.arange(len(m))
    ax.bar(x - 0.2, m["rul_shift_naive"], 0.4, color="0.6", label="faulty sensor trusted")
    ax.bar(x + 0.2, m["rul_shift_trusted"], 0.4, color="C3", label="with sensor check")
    ax.set_xticks(x, [f"{s} x" for s in m.index])
    ax.set_xlabel("fault size (x sensor noise), bias + drift, all sensors")
    ax.set_ylabel("mean RUL error caused by the fault (cycles)")
    ax.set_title("Effect on predicted remaining life")
    ax.legend(fontsize=8)
    fig.suptitle("FD001 test engines with injected sensor faults")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()