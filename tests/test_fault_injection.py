"""Checks that injected faults have the intended shape and size."""
import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_test
from src.fault_injection import DRIFT_CYCLES, inject


@pytest.fixture(scope="module")
def engine():
    t = load_test()
    return t[t["unit"] == 1].reset_index(drop=True)      # 31 cycles


def test_bias_is_a_step(engine):
    onset = pd.Series({1: 10})
    out = inject(engine, "T30", "bias", 4, onset, noise_pct=0.25)
    ratio = out["s3"] / engine["s3"] - 1
    assert np.allclose(ratio[engine["cycle"] < 10], 0)
    assert np.allclose(ratio[engine["cycle"] >= 10], 0.04 * 0.25)   # 4 x 0.25 %
    assert out["s2"].equals(engine["s2"])                            # others untouched


def test_drift_grows_from_zero(engine):
    onset = pd.Series({1: 5})
    out = inject(engine, "T50", "drift", 2, onset, noise_pct=0.3)
    ratio = (out["s4"] / engine["s4"] - 1).to_numpy()
    cyc = engine["cycle"].to_numpy()
    expected = 2 * 0.3 / 100 * np.clip(cyc - 5, 0, None) / DRIFT_CYCLES
    assert np.allclose(ratio, expected)