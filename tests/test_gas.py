"""Checks for the gas model and components. Run from the repo root:
    python -m pytest -q
"""
import pytest

from src.physics import gas
from src.physics.components import Flow, combustor, compressor, mix, turbine


def test_cp_matches_air_tables():
    # Air tables: cp = 1.005 kJ/kgK at 300 K (540 degR), 1.142 at 1000 K (1800 degR)
    assert gas.cp(540.0) / gas.KJKGK_TO_BTULBMR == pytest.approx(1.005, abs=0.003)
    assert gas.cp(1800.0) / gas.KJKGK_TO_BTULBMR == pytest.approx(1.142, abs=0.003)


def test_gamma_is_1_4_at_room_temperature():
    assert gas.gamma(540.0) == pytest.approx(1.40, abs=0.002)


def test_inverse_functions_round_trip():
    for t in (600.0, 1500.0, 2800.0):
        assert gas.temp_from_enthalpy(gas.enthalpy(t, 0.02), 0.02) == pytest.approx(t)
        assert gas.temp_from_entropy_fn(gas.entropy_fn(t)) == pytest.approx(t)


def test_compressor_functions_agree():
    t_out = gas.compressor_exit_temp(642.0, 19.0, 0.85)
    assert gas.compressor_efficiency(642.0, t_out, 19.0) == pytest.approx(0.85)
    assert gas.compressor_pressure_ratio(642.0, t_out, 0.85) == pytest.approx(19.0)


def test_lower_efficiency_means_hotter_exit():
    assert gas.compressor_exit_temp(642.0, 19.0, 0.84) > gas.compressor_exit_temp(642.0, 19.0, 0.86)


def test_turbine_gives_back_compressor_power():
    # Energy conservation: the turbine removes exactly the power asked for
    air, power = compressor(Flow(w=100.0, t=642.0, p=28.5), 19.0, 0.85)
    hot, _ = combustor(air, 2800.0)
    out = turbine(hot, power, 0.89)
    assert hot.w * (hot.h - out.h) == pytest.approx(power)
    assert out.p < hot.p


def test_combustor_fuel_air_ratio_is_realistic():
    hot, wf = combustor(Flow(w=100.0, t=1587.0, p=554.0), 2800.0)
    assert 0.015 < hot.far < 0.03
    assert hot.w == pytest.approx(100.0 + wf)


def test_mixing_conserves_energy():
    a, b = Flow(w=90.0, t=2000.0, p=100.0, far=0.02), Flow(w=10.0, t=1000.0, p=110.0)
    m = mix(a, b)
    assert m.w == pytest.approx(100.0)
    assert m.w * m.h == pytest.approx(a.w * a.h + b.w * b.h)
    assert 1000.0 < m.t < 2000.0
