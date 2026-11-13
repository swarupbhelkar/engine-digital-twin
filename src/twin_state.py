"""Week 4, step 3: the twin state. One row per engine per cycle.

This is what the dashboard (Week 6) will read. Each cycle the twin answers:
  1. Is every sensor telling the truth?  (sensor check, step 1)
  2. How worn is the HPC?   efficiency and flow-capacity loss, health index
  3. How fast is it wearing? health index lost per 100 cycles, last 30 cycles
  4. How long until it needs removing?  predicted RUL with a range
  5. What should the operator do?  engine status + sensor status

RUL model: the Week 3 random forest on PHYSICS health only (model C), now fed
the health the twin trusts. Model C is a little less accurate than the best
models (test RMSE 17.4 vs LSTM 14.1), but it reads only the 3 health numbers,
so when a sensor fails and the check drops it, the RUL is protected too
(step 2). The LSTM reads raw sensors and has no such protection.

RUL range (80 %): how wrong is the model usually? Measured honestly with
5-fold cross-validation on TRAIN engines (split by engine): the 10th and
90th percentile of (true - predicted), in 3 bands of predicted RUL, because
errors are bigger when lots of life is left. range = prediction + those.
(The spread between the 200 trees was tried first: far too narrow, only
47 % of test engines inside. Trees agree with each other more than with
the truth.)

Status rules (simple and readable; an airline would set its own):
  engine: ACTION if predicted RUL <= 30 cycles, WATCH if <= 75, else OK
  sensor: FAULT <name> once the sensor check raises an alarm, else OK

Why not just extrapolate the health trend? Tried: a straight line through
the last 30-80 cycles of health gives test RMSE 29-36 cycles. HPC wear
accelerates near the end, so a straight line predicts too much life. The
wear rate is still shown, because a rising rate is useful to see.

Outputs (in data/processed/, regenerate any time):
  twin_state_test.csv : all 100 test engines, clean data
  twin_state_demo.csv : test engine 52 with an injected T30 drift (reads
                        low, hiding wear), to show the twin catching it

Run from the repo root (about 30 seconds):
    python -m src.twin_state
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_predict

from src.data_loader import load_test, load_train
from src.fault_injection import PHYS_COLS, health_table, inject
from src.features import RUL_CAP
from src.rul_rf import make_model, nasa_score, rmse
from src.sensor_check import NAMES, Calibration, calibrate, check

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
RATE_WINDOW = 30               # cycles used for the wear rate
ACTION_RUL, WATCH_RUL = 30, 75
RANGE_BANDS = [0, 40, 80, np.inf]   # predicted-RUL bands for the error range
# Demo: test engine 52, chosen because the clean twin is accurate there
# (predicts 27 cycles, truth 29), so any difference in the demo is the fault.
# T30 drifts LOW, which hides wear: the dangerous direction.
DEMO = {"unit": 52, "sensor": "T30", "kind": "drift", "size": -4, "onset": 90}


class Twin:
    """Everything learned from train engines, bundled so it is built once."""

    def __init__(self, train_raw: pd.DataFrame):
        self.cal: Calibration = calibrate(train_raw)
        res = check(train_raw, self.cal)
        end = res[res["rul"] < 10].groupby("unit")[["d_eta_pct", "d_flow_pct"]].mean()
        self.eta_fail = end["d_eta_pct"].mean()
        self.flow_fail = end["d_flow_pct"].mean()
        feats = health_table(res, self.eta_fail, self.flow_fail, "trusted")
        X, y = feats[PHYS_COLS], feats["rul"].clip(upper=RUL_CAP)
        self.rf = make_model().fit(X, y)

        # Typical error, from 5-fold cross-validation grouped by engine
        cv_pred = cross_val_predict(make_model(), X, y, groups=feats["unit"],
                                    cv=GroupKFold(n_splits=5))
        band = pd.cut(cv_pred, RANGE_BANDS, include_lowest=True)
        err = pd.Series(y.to_numpy() - cv_pred).groupby(band, observed=False)
        self.err_low = err.quantile(0.10).to_numpy()      # one value per band
        self.err_high = err.quantile(0.90).to_numpy()

    def state(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Twin state for every engine and cycle in raw."""
        res = check(raw, self.cal)
        h = health_table(res, self.eta_fail, self.flow_fail, "trusted")
        naive = health_table(res, self.eta_fail, self.flow_fail, "naive")

        out = res[["unit", "cycle"]].copy()
        out["rul_true"] = res["rul"]                   # known here; not in service
        out["d_eta_pct"] = h["d_eta_pct"]
        out["d_flow_pct"] = h["d_flow_pct"]
        out["hi"] = h["hi"]
        out["hi_if_trusting_all"] = naive["hi"]        # what a twin without the check sees
        out["wear_rate"] = wear_rate(out)

        pred = self.rf.predict(h[PHYS_COLS])
        band = np.searchsorted(RANGE_BANDS[1:-1], pred)    # 0, 1 or 2
        out["rul_pred"] = pred
        out["rul_if_trusting_all"] = self.rf.predict(naive[PHYS_COLS])
        out["rul_low"] = np.clip(pred + self.err_low[band], 0, None)
        out["rul_high"] = pred + self.err_high[band]

        z = res[[f"z_{n}" for n in NAMES]].abs()
        out["worst_sensor"] = z.idxmax(axis=1).str[2:]
        out["worst_z"] = z.max(axis=1)
        out["sensor_status"] = np.where(res["alarm"] != "", "FAULT " + res["alarm"], "OK")
        out["engine_status"] = np.select(
            [out["rul_pred"] <= ACTION_RUL, out["rul_pred"] <= WATCH_RUL],
            ["ACTION", "WATCH"], "OK")
        return out


def wear_rate(state: pd.DataFrame) -> pd.Series:
    """Health index lost per 100 cycles: least-squares slope over the last
    RATE_WINDOW cycles, per engine, using past cycles only.
    slope = cov(cycle, hi) / var(cycle), with rolling means."""
    def roll(s: pd.Series) -> pd.Series:
        return s.groupby(state["unit"]).transform(
            lambda x: x.rolling(RATE_WINDOW, min_periods=10).mean())
    t, y = state["cycle"].astype(float), state["hi"]
    cov = roll(t * y) - roll(t) * roll(y)
    var = roll(t * t) - roll(t) ** 2
    return -100 * cov / var


def report_test(state: pd.DataFrame) -> None:
    last = state.groupby("unit").tail(1)
    true = last["rul_true"].clip(upper=RUL_CAP)
    print(f"Test engines, last cycle (100 engines, RUL capped at {RUL_CAP}):")
    print(f"  RUL RMSE {rmse(last['rul_pred'], true):.2f} cycles, "
          f"NASA score {nasa_score(last['rul_pred'], true):.0f}")
    inside = ((true >= last["rul_low"]) & (true <= last["rul_high"])).mean()
    print(f"  true RUL inside the 80 % range: {100 * inside:.0f} % of engines")

    band = pd.cut(last["rul_true"], [-1, ACTION_RUL, WATCH_RUL, 1000],
                  labels=[f"true <= {ACTION_RUL}", f"true {ACTION_RUL + 1}-{WATCH_RUL}",
                          f"true > {WATCH_RUL}"])
    print("\nEngine status at the last cycle vs true remaining life (count of engines):")
    print(pd.crosstab(band, last["engine_status"])
          .reindex(columns=["OK", "WATCH", "ACTION"], fill_value=0).to_string())
    print(f"\n  sensor alarms on clean test data: "
          f"{(last['sensor_status'] != 'OK').sum()} engine(s)")


def plot_demo(demo: pd.DataFrame, clean: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    alarm = demo.loc[demo["sensor_status"] != "OK", "cycle"]
    t_alarm = alarm.min() if len(alarm) else None

    ax = axes[0]
    ax.plot(demo["cycle"], demo["worst_z"].where(demo["worst_sensor"] == DEMO["sensor"]),
            color="C3", lw=0.8, label=f"|z| of {DEMO['sensor']} (when it is the worst sensor)")
    ax.plot(demo["cycle"], demo["worst_z"], color="0.7", lw=0.6, zorder=0,
            label="largest |z| of any sensor")
    ax.axhline(4, color="C3", ls="--", lw=1, label="alarm level")
    ax.set_ylabel("sensor check |z|")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    ax.plot(clean["cycle"], clean["hi"], color="0.6", lw=1, label="clean sensors (truth)")
    ax.plot(demo["cycle"], demo["hi_if_trusting_all"], color="C1", lw=1,
            label="faulty sensor trusted")
    ax.plot(demo["cycle"], demo["hi"], color="C0", lw=1.5, label="twin (with sensor check)")
    ax.set_ylabel("health index (1 new, 0 failure)")
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[2]
    ax.fill_between(demo["cycle"], demo["rul_low"], demo["rul_high"], color="C0", alpha=0.2,
                    label="80 % range")
    ax.plot(demo["cycle"], demo["rul_if_trusting_all"], color="C1", lw=1,
            label="faulty sensor trusted")
    ax.plot(demo["cycle"], demo["rul_pred"], color="C0", lw=1.5, label="twin predicted RUL")
    ax.plot(demo["cycle"], demo["rul_true"].clip(upper=RUL_CAP), "k--", lw=1,
            label=f"true RUL (capped {RUL_CAP})")
    ax.set_ylabel("remaining life (cycles)")
    ax.set_xlabel("cycle")
    ax.legend(fontsize=8, loc="lower left")

    for ax in axes:
        ax.axvline(DEMO["onset"], color="C1", lw=0.8, ls=":")
        if t_alarm is not None:
            ax.axvline(t_alarm, color="C3", lw=0.8, ls=":")
    axes[0].text(DEMO["onset"], axes[0].get_ylim()[1] * 0.92, " fault starts", color="C1", fontsize=8)
    if t_alarm is not None:
        axes[0].text(t_alarm, axes[0].get_ylim()[1] * 0.80, " alarm", color="C3", fontsize=8)
    unit = int(demo["unit"].iloc[0])
    fig.suptitle(f"Twin demo: test engine {unit}, {DEMO['sensor']} {DEMO['kind']} "
                 f"({DEMO['size']} x noise) injected at cycle {DEMO['onset']}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    train_raw, test_raw = load_train(), load_test()
    twin = Twin(train_raw)

    state = twin.state(test_raw)
    PROC.mkdir(parents=True, exist_ok=True)
    state.round(4).to_csv(PROC / "twin_state_test.csv", index=False)
    report_test(state)

    # Demo: one engine with a slow T30 drift
    unit = DEMO["unit"]
    one = test_raw[test_raw["unit"] == unit]
    faulty = inject(one, DEMO["sensor"], DEMO["kind"], DEMO["size"],
                    pd.Series({unit: DEMO["onset"]}), twin.cal.noise[DEMO["sensor"]])
    demo, clean = twin.state(faulty), twin.state(one)
    demo.round(4).to_csv(PROC / "twin_state_demo.csv", index=False)

    first = demo.loc[demo["sensor_status"] != "OK", "cycle"]
    end_d, end_c = demo.iloc[-1], clean.iloc[-1]
    print(f"\nDemo: test engine {unit} ({len(one)} cycles), {DEMO['sensor']} "
          f"{DEMO['kind']} {DEMO['size']} x noise from cycle {DEMO['onset']}")
    print(f"  alarm: {end_d['sensor_status']} at cycle "
          f"{int(first.min()) if len(first) else 'never'}")
    print(f"  last cycle health index: clean {end_c['hi']:.2f}, twin {end_d['hi']:.2f}, "
          f"faulty sensor trusted {end_d['hi_if_trusting_all']:.2f}")
    print(f"  last cycle predicted RUL: clean {end_c['rul_pred']:.0f}, twin "
          f"{end_d['rul_pred']:.0f} ({end_d['engine_status']}), faulty sensor trusted "
          f"{end_d['rul_if_trusting_all']:.0f}, true {end_d['rul_true']:.0f}")

    fig = REPORTS / "figures" / "12_twin_demo.png"
    plot_demo(demo, clean, fig)
    print(f"\nSaved data/processed/twin_state_test.csv, twin_state_demo.csv,")
    print(f"      reports/figures/{fig.name}")


if __name__ == "__main__":
    main()