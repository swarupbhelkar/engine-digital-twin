"""Checks for the off-design solver. Run: python -m pytest -q"""
import pytest

from src.physics.off_design import Health, OffDesignEngine, load_design, sensors


@pytest.fixture(scope="module")
def engine():
    return OffDesignEngine(load_design())


@pytest.fixture(scope="module")
def new(engine):
    return sensors(engine.solve())


def test_no_damage_returns_design_point(engine):
    x = engine.solve()["inputs"]
    d = engine.d
    for got, want in [(x.w_core, d.w_core), (x.pr_lpc, d.pr_lpc),
                      (x.pr_hpc, d.pr_hpc), (x.t4, d.t4)]:
        assert got == pytest.approx(want, rel=1e-6)


def test_efficiency_loss_makes_engine_run_hotter(engine, new):
    worn = sensors(engine.solve(Health(eta_hpc=-0.01)))
    assert worn["T30"] > new["T30"]      # more compressor work wasted as heat
    assert worn["T50"] > new["T50"]      # turbines work harder, exhaust hotter
    assert worn["fuel"] > new["fuel"]    # same fan speed costs more fuel


def test_combined_damage_matches_fd001_directions(engine, new):
    worn = sensors(engine.solve(Health(eta_hpc=-0.01, flow_hpc=-0.01)))
    rises = ["T24", "T30", "T50", "BPR", "fuel"]
    falls = ["P30", "W31", "W32"]
    assert all(worn[k] > new[k] for k in rises)
    assert all(worn[k] < new[k] for k in falls)


def test_solver_handles_large_damage(engine):
    sol = engine.solve(Health(eta_hpc=-0.03, flow_hpc=-0.03))
    assert sol["inputs"].t4 > engine.d.t4
