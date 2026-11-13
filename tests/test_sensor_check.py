"""Checks for the Week 4 sensor check (physics vs data residuals)."""
import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_test, load_train
from src.physics.health_fit import FIT_SENSORS, sensor_deviations
from src.sensor_check import (NAMES, calibrate, check, leave_one_out,
                              z_scores)


@pytest.fixture(scope="module")
def setup():
    train = load_train()
    return train, load_test(), calibrate(train)


def test_leave_one_out_exact_on_perfect_data(setup):
    """If sensors follow the physics exactly, every residual is zero."""
    _, _, cal = setup
    health = np.array([[-0.5, -0.5], [-1.5, -1.4], [0.0, 0.0]])   # 3 fake cycles
    dev = pd.DataFrame(health @ cal.H.loc[NAMES].to_numpy().T, columns=NAMES)
    resid, loo = leave_one_out(dev, cal.H, cal.noise)
    assert np.abs(resid.to_numpy()).max() < 1e-9
    assert np.allclose(loo["T30"][0], health[:, 0])     # efficiency recovered


def test_mismatch_correction_removes_drift(setup):
    """Near failure, corrected residuals of clean engines average about 0."""
    train, _, cal = setup
    dev = sensor_deviations(train)
    z, _ = z_scores(dev, cal)
    late = z[dev["rul"] < 25].mean()
    assert late.abs().max() < 0.5


def test_no_false_alarms_on_clean_train(setup):
    train, _, cal = setup
    res = check(train, cal)
    assert (res["alarm"] == "").all()


def test_injected_bias_is_caught_and_isolated(setup):
    """6 x noise bias on T30 from cycle 40: alarm names T30, and the trusted
    health stays close to the clean engine's health."""
    _, test, cal = setup
    lens = test.groupby("unit")["cycle"].max()
    units = lens[lens >= 100].index[:10]
    clean = test[test["unit"].isin(units)].copy()
    faulty = clean.copy()
    col = FIT_SENSORS["T30"]
    on = faulty["cycle"] >= 40
    faulty.loc[on, col] *= 1 + 6 * cal.noise["T30"] / 100

    res_clean, res = check(clean, cal), check(faulty, cal)
    last = res.groupby("unit").tail(1)
    assert (last["alarm"] == "T30").all()

    first = res[res["alarm"] != ""].groupby("unit")["cycle"].min()
    assert first.min() >= 40 and first.max() <= 60     # caught within 20 cycles

    end_clean = res_clean.groupby("unit").tail(1)
    err_naive = (last["d_eta_all"].to_numpy() - end_clean["d_eta_pct"].to_numpy())
    err_trusted = (last["d_eta_pct"].to_numpy() - end_clean["d_eta_pct"].to_numpy())
    assert np.abs(err_trusted).mean() < 0.5 * np.abs(err_naive).mean()