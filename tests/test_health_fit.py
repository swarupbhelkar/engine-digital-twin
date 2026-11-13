"""Checks for the HPC health fit. Run: python -m pytest -q"""
import pandas as pd
import pytest

from src.physics.health_fit import FIT_SENSORS, fit_health, influence_matrix
from src.physics.off_design import Health, OffDesignEngine, load_design, sensors


@pytest.fixture(scope="module")
def eng():
    return OffDesignEngine(load_design())


@pytest.fixture(scope="module")
def H(eng):
    return influence_matrix(eng)


def test_fit_recovers_known_damage(eng, H):
    # Make "fake data" with the full nonlinear model, then see if the fit finds it
    true_eta, true_flow = -1.2, -0.8          # percent
    new = sensors(eng.solve())
    worn = sensors(eng.solve(Health(eta_hpc=true_eta / 100, flow_hpc=true_flow / 100)))
    row = {k: 100 * (worn[k] / new[k] - 1) for k in FIT_SENSORS}
    dev = pd.DataFrame([{"unit": 1, "cycle": 1, "rul": 0, **row}])
    noise = pd.Series(1.0, index=list(FIT_SENSORS))
    got = fit_health(dev, H, noise).iloc[0]
    assert got["d_eta_pct"] == pytest.approx(true_eta, abs=0.05)
    assert got["d_flow_pct"] == pytest.approx(true_flow, abs=0.05)


def test_no_change_means_healthy(H):
    dev = pd.DataFrame([{"unit": 1, "cycle": 1, "rul": 0, **{k: 0.0 for k in FIT_SENSORS}}])
    got = fit_health(dev, H, pd.Series(1.0, index=list(FIT_SENSORS))).iloc[0]
    assert got["d_eta_pct"] == pytest.approx(0.0, abs=1e-12)
    assert got["d_flow_pct"] == pytest.approx(0.0, abs=1e-12)


def test_efficiency_loss_heats_the_exhaust(H):
    # Efficiency UP should cool T50, so the coefficient is negative
    assert H.loc["T50", "eta"] < 0