"""Checks for the LSTM window cutter (fast: no training)."""
import numpy as np
import pandas as pd

from src.rul_lstm import WINDOW, RULNet, windows


def _engine(unit, n):
    return pd.DataFrame({"unit": unit, "cycle": np.arange(1, n + 1),
                         "x": np.arange(n, dtype=float),
                         "rul_capped": np.arange(n - 1, -1, -1, dtype=float)})


def test_windows_train_and_last():
    df = pd.concat([_engine(1, 40), _engine(2, 35)])
    X, y = windows(df, ["x"])
    assert X.shape == (11 + 6, WINDOW, 1)        # 40-30+1 and 35-30+1 windows
    assert y[0] == 10 and X[0, -1, 0] == 29      # first window ends at cycle 30
    X, y = windows(df, ["x"], last_only=True)
    assert X.shape == (2, WINDOW, 1) and y.tolist() == [0, 0]


def test_short_engine_is_padded():
    X, y = windows(_engine(1, 20), ["x"], last_only=True)
    assert X.shape == (1, WINDOW, 1)
    assert X[0, 0, 0] == 0 and X[0, -1, 0] == 19 and y[0] == 0


def test_net_output_shape():
    X, _ = windows(_engine(1, 40), ["x"])
    assert RULNet(1)(X).shape == (len(X),)