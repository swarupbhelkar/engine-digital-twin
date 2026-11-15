"""Checks for the Week 5 HPC stage and blade geometry. Run: python -m pytest -q"""
import math

import numpy as np
import pytest

from src.cad_cfd import blade_geometry as bg
from src.cad_cfd.hpc_stage import N_STAGES, design_stage, efficiency_drop_from_loss


@pytest.fixture(scope="module")
def stage():
    return design_stage()


def test_stage_is_a_sensible_compressor_stage(stage):
    # Mid-HPC stage: PR 1.2-1.5, subsonic inlet, diffusion inside the de Haller limit
    assert 1.2 < stage.pr < 1.5
    assert 0.5 < stage.mach_w1 < 1.0
    assert stage.de_haller >= 0.72
    assert 1.0 < stage.solidity < 1.8
    assert stage.eta < stage.eta_poly          # compressor: isentropic < polytropic


def test_all_stages_multiply_to_hpc_pressure_ratio():
    prs = [design_stage(k).pr for k in range(1, N_STAGES + 1)]
    from src.physics.off_design import load_design
    assert np.prod(prs) == pytest.approx(load_design().pr_hpc, rel=1e-6)


def test_blade_angles_match_the_flow(stage):
    ang = bg.metal_angles(stage)
    # Leading edge camber slope = inlet metal angle; Carter deviation closes the loop
    assert ang["stagger"] + ang["camber"] / 2 == pytest.approx(ang["b1_metal"])
    assert ang["b2_metal"] + ang["deviation"] == pytest.approx(stage.beta2, abs=1e-6)


def test_profiles_are_closed_and_erosion_shortens_the_blade(stage):
    healthy, eroded = bg.profile(stage, "healthy"), bg.profile(stage, "eroded")
    for xy in (healthy, eroded):
        assert np.allclose(xy[0], xy[-1])                    # closed loop at the TE
    a = math.radians(bg.metal_angles(stage)["stagger"])
    along = np.array([math.cos(a), math.sin(a)])
    chord_h = np.ptp(healthy @ along)
    chord_e = np.ptp(eroded @ along)
    assert chord_e < chord_h
    assert chord_h == pytest.approx(stage.chord, rel=0.01)


def test_efficiency_link_is_linear_and_negative(stage):
    d1 = efficiency_drop_from_loss(stage, 0.01)
    assert d1 < 0
    assert efficiency_drop_from_loss(stage, 0.02) == pytest.approx(2 * d1)


def test_engine_layout_is_consistent_with_the_hpc_stage(stage):
    from src.cad_cfd.engine_layout import layout
    lay, comps = layout()
    assert (lay["r_tip_mm"] > lay["r_hub_mm"]).all()
    assert comps["x_end_mm"].is_monotonic_increasing
    hpc = lay[lay["station"].isin(["24", "30"])]
    mean_r = (hpc["r_hub_mm"] + hpc["r_tip_mm"]) / 2
    assert np.allclose(mean_r, 1000 * stage.r_mean, atol=0.2)
    # Blades get shorter through the HPC as the air is compressed
    h = (hpc["r_tip_mm"] - hpc["r_hub_mm"]).to_numpy()
    assert h[1] < h[0]


def test_health_colours_go_from_green_to_red():
    from src.cad_cfd.engine_layout import ROOT, health_colours
    if not (ROOT / "data" / "processed" / "twin_state_demo.csv").exists():
        pytest.skip("run python -m src.twin_state first")
    c = health_colours()
    assert c["hi"].is_monotonic_decreasing
    red = c["rgb"].str.split().str[0].astype(int)
    green = c["rgb"].str.split().str[1].astype(int)
    assert green.iloc[0] > red.iloc[0] and red.iloc[-1] > green.iloc[-1]
