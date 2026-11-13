"""Week 3, step 2: random-forest RUL baseline. Does physics help?

Three models, same random forest, different inputs:
  A  sensors only     : 14 smoothed sensors (what a pure data model sees)
  B  sensors + physics: A plus HPC efficiency loss, flow loss, health index
  C  physics only     : just the 3 physics health numbers

Target: RUL capped at 125 (early life all looks "healthy", so we don't ask
the model to tell cycle 20 from cycle 60).

Two scores:
  1. Cross-validation on TRAIN, split by engine (5 folds). A model never
     sees cycles from an engine it is scored on.
  2. The official TEST score: predict RUL at each test engine's LAST cycle,
     compare with RUL_FD001.txt. Reported as RMSE (cycles) and the NASA
     score (late predictions punished more than early ones).

Needs data/processed/*.csv, so run `python -m src.features` first.

Run from the repo root:
    python -m src.rul_rf
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

from src.data_loader import label
from src.features import RUL_CAP, SENSOR_COLS

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

PHYSICS_COLS = ["d_eta_pct", "d_flow_pct", "hi"]
FEATURE_SETS = {
    "A sensors only": SENSOR_COLS,
    "B sensors + physics": SENSOR_COLS + PHYSICS_COLS,
    "C physics only": PHYSICS_COLS,
}


def make_model() -> RandomForestRegressor:
    # min_samples_leaf=10: each leaf averages >= 10 cycles, smooths sensor noise
    return RandomForestRegressor(n_estimators=200, min_samples_leaf=10,
                                 max_features=0.5, n_jobs=-1, random_state=0)


def rmse(pred, true) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(true)) ** 2)))


def nasa_score(pred, true) -> float:
    """PHM08 score. d = pred - true. Late (d > 0) costs more: an engine
    you think has life left but fails is worse than one pulled early."""
    d = np.asarray(pred) - np.asarray(true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def cv_rmse(train: pd.DataFrame, cols: list[str]) -> float:
    """5-fold CV, grouped by engine, scored on all cycles (capped RUL)."""
    errs = []
    for fit_idx, val_idx in GroupKFold(n_splits=5).split(train, groups=train["unit"]):
        m = make_model().fit(train.iloc[fit_idx][cols], train.iloc[fit_idx]["rul_capped"])
        pred = m.predict(train.iloc[val_idx][cols])
        errs.append(rmse(pred, train.iloc[val_idx]["rul_capped"]))
    return float(np.mean(errs))


def plot(last: pd.DataFrame, preds: dict, model_b, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    true = last["rul"].clip(upper=RUL_CAP)
    for name, c in [("A sensors only", "0.6"), ("B sensors + physics", "C3")]:
        ax.scatter(true, preds[name], s=14, color=c, label=name)
    ax.plot([0, RUL_CAP], [0, RUL_CAP], "k--", lw=0.8)
    ax.set_xlabel(f"true RUL (capped at {RUL_CAP})")
    ax.set_ylabel("predicted RUL")
    ax.set_title("Test engines, last cycle")
    ax.legend(loc="upper left")

    ax = axes[1]
    cols = FEATURE_SETS["B sensors + physics"]
    imp = pd.Series(model_b.feature_importances_, index=cols).sort_values()
    names = [c if c in PHYSICS_COLS else label(c) for c in imp.index]
    colors = ["C3" if c in PHYSICS_COLS else "0.6" for c in imp.index]
    ax.barh(names, imp.values, color=colors)
    ax.set_xlabel("importance (share of error reduction)")
    ax.set_title("Model B: what the forest relies on (red = physics)")
    fig.suptitle("FD001 random-forest RUL baseline")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    train = pd.read_csv(PROC / "features_train.csv")
    test = pd.read_csv(PROC / "features_test.csv")
    last = test.groupby("unit").tail(1)         # one row per test engine
    true_capped = last["rul"].clip(upper=RUL_CAP)

    rows, preds, models = [], {}, {}
    for name, cols in FEATURE_SETS.items():
        print(f"{name}: cross-validating ...", flush=True)
        cv = cv_rmse(train, cols)
        m = make_model().fit(train[cols], train["rul_capped"])
        p = m.predict(last[cols])
        preds[name], models[name] = p, m
        rows.append({"model": name, "n_features": len(cols),
                     "cv_rmse_train": round(cv, 2),
                     "test_rmse": round(rmse(p, true_capped), 2),
                     "test_nasa_score": round(nasa_score(p, true_capped), 1)})

    res = pd.DataFrame(rows)
    res.to_csv(REPORTS / "rul_rf_results.csv", index=False)
    print("\nRandom forest, RUL capped at 125 (lower is better):")
    print(res.to_string(index=False))

    # Where do errors come from? Split test engines by true RUL.
    err = pd.DataFrame({n: np.abs(p - true_capped.to_numpy()) for n, p in preds.items()})
    err["band"] = pd.cut(last["rul"].to_numpy(), [0, 50, 100, 200],
                         labels=["near failure (<=50)", "mid (51-100)", "early (>100)"])
    print("\nMean absolute error on test engines, by true RUL:")
    print(err.groupby("band", observed=True).mean().round(1).to_string())

    fig = REPORTS / "figures" / "08_rf_rul_test.png"
    plot(last, preds, models["B sensors + physics"], fig)
    print(f"\nSaved reports/rul_rf_results.csv, reports/figures/{fig.name}")


if __name__ == "__main__":
    main()