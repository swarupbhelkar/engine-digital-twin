"""Checks for the calibrated design point. Run: python -m pytest -q"""
import pytest

from src.physics.design_point import calibrate, healthy_targets, run_engine


@pytest.fixture(scope="module")
def design():
    tg = healthy_targets()
    x = calibrate(tg)
    return tg, x, run_engine(x)


def test_matches_every_healthy_sensor(design):
    tg, _, r = design
    st = r["stations"]
    for model, data in [(st["24"].t, tg["t24"]), (st["30"].t, tg["t30"]),
                        (st["50"].t, tg["t50"]), (st["30"].p, tg["p30"]),
                        (st["13"].p, tg["p15"]), (r["wf"], tg["wf"])]:
        assert model == pytest.approx(data, rel=1e-4)


def test_spools_are_in_power_balance(design):
    # Each turbine delivers exactly what its compressors absorb
    _, x, r = design
    st, pw = r["stations"], r["power_btu_s"]
    hpt_out = st["41"].w * (st["41"].h - st["45"].h)
    assert hpt_out == pytest.approx(pw["hpc"], rel=1e-6)


def test_mass_is_conserved(design):
    _, x, r = design
    # core nozzle flow = core inlet air + fuel (coolant leaves and comes back)
    assert r["stations"]["50"].w == pytest.approx(x.w_core + r["wf"], rel=1e-9)


def test_values_are_physically_sensible(design):
    _, x, r = design
    assert 2600 < x.t4 < 3400               # degR: typical take-off TIT
    assert 0.80 < x.eta_hpc < 0.92
    assert 0.015 < r["stations"]["40"].far < 0.035
    assert r["pr_hpt"] > 1 and r["pr_lpt"] > 1
