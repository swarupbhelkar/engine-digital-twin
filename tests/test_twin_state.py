"""Checks for the Week 4 twin state."""
import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_test, load_train
from src.twin_state import Twin, wear_rate


@pytest.fixture(scope="module")
def twin_and_state():
    twin = Twin(load_train())
    return twin, twin.state(load_test())


def test_wear_rate_of_a_straight_line():
    """hi falling 0.002 per cycle = 0.2 health index lost per 100 cycles."""
    cyc = np.arange(1, 61)
    df = pd.DataFrame({"unit": 1, "cycle": cyc, "hi": 1 - 0.002 * cyc})
    rate = wear_rate(df)
    assert rate.iloc[:9].isna().all()             # not enough history yet
    assert np.allclose(rate.iloc[9:], 0.2)


def test_state_columns_and_values(twin_and_state):
    _, s = twin_and_state
    assert len(s) == 13096
    assert set(s["engine_status"]) <= {"OK", "WATCH", "ACTION"}
    assert (s["rul_low"] <= s["rul_pred"]).all() and (s["rul_pred"] <= s["rul_high"]).all()
    assert (s["rul_low"] >= 0).all()


def test_test_score_and_safety(twin_and_state):
    """Same accuracy as Week 3 model C, and no engine near failure marked OK."""
    _, s = twin_and_state
    last = s.groupby("unit").tail(1)
    err = last["rul_pred"] - last["rul_true"].clip(upper=125)
    assert np.sqrt(np.mean(err ** 2)) < 19
    assert not ((last["rul_true"] <= 30) & (last["engine_status"] == "OK")).any()
    inside = (last["rul_true"].clip(upper=125) >= last["rul_low"]) & \
             (last["rul_true"].clip(upper=125) <= last["rul_high"])
    assert 0.65 < inside.mean() < 0.95             # roughly the promised 80 %
