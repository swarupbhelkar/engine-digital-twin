"""Week 3, step 3: LSTM RUL model. Same question as step 2: does physics help?

Random forest (step 2) sees ONE cycle at a time. An LSTM reads a WINDOW of
the last 30 cycles in order, so it can see the trend (how fast health is
falling), not just the current level.

Same inputs as step 2:
  A  sensors only      : 14 smoothed sensors
  B  sensors + physics : A plus HPC efficiency loss, flow loss, health index

How training works:
  - Every input is scaled to mean 0, std 1 using TRAIN statistics only.
  - 15 of the 100 train engines are held out for early stopping: after each
    pass over the data (epoch) we check error on them and keep the best
    weights. This stops the network memorising the training engines.
  - Neural nets start from random weights, so results vary run to run.
    We train 3 seeds and average their predictions (a small ensemble).

Needs data/processed/*.csv, so run `python -m src.features` first.

Run from the repo root (takes a few minutes on CPU):
    python -m src.rul_lstm
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from src.features import RUL_CAP, SENSOR_COLS
from src.rul_rf import PHYSICS_COLS, nasa_score, rmse

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

WINDOW = 30          # cycles the LSTM looks back
SEEDS = [0, 1, 2]
MAX_EPOCHS = 60
PATIENCE = 10        # stop if validation hasn't improved for this many epochs
N_VAL_ENGINES = 15

FEATURE_SETS = {
    "A sensors only": SENSOR_COLS,
    "B sensors + physics": SENSOR_COLS + PHYSICS_COLS,
}


class RULNet(nn.Module):
    """LSTM reads the window; a small linear layer turns its final state into RUL."""

    def __init__(self, n_inputs: int, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(n_inputs, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(hidden, 32),
                                  nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):                 # x: (batch, WINDOW, n_inputs)
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)   # use the last time step


def windows(df: pd.DataFrame, cols: list[str], last_only: bool = False):
    """Cut each engine into overlapping 30-cycle windows.

    Window ending at cycle t holds cycles t-29 .. t, target = capped RUL at t.
    Engines shorter than WINDOW are padded by repeating their first cycle.
    last_only=True gives one window per engine (its last cycle): the test case.
    """
    X, y = [], []
    for _, g in df.groupby("unit"):
        a = g[cols].to_numpy(np.float32)
        if len(a) < WINDOW:
            a = np.vstack([np.repeat(a[:1], WINDOW - len(a), axis=0), a])
        r = g["rul_capped"].to_numpy(np.float32)[-len(a):] if len(g) >= WINDOW else \
            np.concatenate([np.full(WINDOW - len(g), np.nan, np.float32),
                            g["rul_capped"].to_numpy(np.float32)])
        ends = [len(a)] if last_only else range(WINDOW, len(a) + 1)
        for e in ends:
            X.append(a[e - WINDOW:e])
            y.append(r[e - 1])
    return torch.tensor(np.array(X)), torch.tensor(np.array(y))


def train_one(Xtr, ytr, Xval, yval, n_inputs: int, seed: int) -> RULNet:
    torch.manual_seed(seed)
    net = RULNet(n_inputs)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    best, best_state, wait = np.inf, None, 0
    gen = torch.Generator().manual_seed(seed)
    for epoch in range(MAX_EPOCHS):
        net.train()
        for idx in torch.randperm(len(Xtr), generator=gen).split(256):
            opt.zero_grad()
            # target scaled to 0..1 (divide by 125): easier for the network
            loss = loss_fn(net(Xtr[idx]), ytr[idx] / RUL_CAP)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val = rmse(net(Xval).numpy() * RUL_CAP, yval.numpy())
        if val < best - 0.01:
            best, wait = val, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    net.load_state_dict(best_state)
    print(f"    seed {seed}: stopped after {epoch + 1} epochs, "
          f"best validation RMSE {best:.2f}", flush=True)
    return net


def main() -> None:
    torch.set_num_threads(max(1, torch.get_num_threads()))
    train = pd.read_csv(PROC / "features_train.csv")
    test = pd.read_csv(PROC / "features_test.csv")
    rng = np.random.default_rng(0)
    val_units = rng.choice(train["unit"].unique(), N_VAL_ENGINES, replace=False)
    fit_df = train[~train["unit"].isin(val_units)]
    val_df = train[train["unit"].isin(val_units)]
    true_last = test.groupby("unit")["rul"].last().clip(upper=RUL_CAP).to_numpy()

    rows, ens_preds = [], {}
    for name, cols in FEATURE_SETS.items():
        print(f"{name}:", flush=True)
        mean, std = fit_df[cols].mean(), fit_df[cols].std()   # fit engines only

        def scaled(d):
            d = d.copy()
            d[cols] = (d[cols] - mean) / std
            return d

        Xtr, ytr = windows(scaled(fit_df), cols)
        Xval, yval = windows(scaled(val_df), cols)
        Xte, _ = windows(scaled(test), cols, last_only=True)

        seed_preds = []
        for s in SEEDS:
            net = train_one(Xtr, ytr, Xval, yval, len(cols), s)
            with torch.no_grad():
                seed_preds.append(net(Xte).numpy() * RUL_CAP)
        seed_rmse = [rmse(p, true_last) for p in seed_preds]
        ens = np.mean(seed_preds, axis=0)
        ens_preds[name] = ens
        rows.append({"model": f"LSTM {name}",
                     "test_rmse_single_seed": f"{np.mean(seed_rmse):.2f} +/- {np.std(seed_rmse):.2f}",
                     "test_rmse": round(rmse(ens, true_last), 2),
                     "test_nasa_score": round(nasa_score(ens, true_last), 1)})

    res = pd.DataFrame(rows)
    res.to_csv(REPORTS / "rul_lstm_results.csv", index=False)

    # Side by side with the random forest from step 2
    rf = pd.read_csv(REPORTS / "rul_rf_results.csv")
    rf = rf[rf["model"].isin(FEATURE_SETS)]
    both = pd.concat([
        pd.DataFrame({"model": "RF " + rf["model"], "test_rmse": rf["test_rmse"],
                      "test_nasa_score": rf["test_nasa_score"]}),
        res[["model", "test_rmse", "test_nasa_score"]]], ignore_index=True)
    print("\nLSTM, 3-seed average (single-seed spread shown):")
    print(res.to_string(index=False))
    print("\nAll models, test set, RUL capped at 125 (lower is better):")
    print(both.to_string(index=False))

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for name, c in [("A sensors only", "0.6"), ("B sensors + physics", "C3")]:
        ax.scatter(true_last, ens_preds[name], s=14, color=c, label=f"LSTM {name}")
    ax.plot([0, RUL_CAP], [0, RUL_CAP], "k--", lw=0.8)
    ax.set_xlabel(f"true RUL (capped at {RUL_CAP})")
    ax.set_ylabel("predicted RUL")
    ax.set_title("FD001 LSTM, test engines, last cycle")
    ax.legend(loc="upper left")
    fig.tight_layout()
    path = REPORTS / "figures" / "09_lstm_rul_test.png"
    fig.savefig(path, dpi=150)
    print(f"\nSaved reports/rul_lstm_results.csv, reports/figures/{path.name}")


if __name__ == "__main__":
    main()