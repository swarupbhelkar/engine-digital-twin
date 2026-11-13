"""Checks for the Week 3 feature table."""
import pandas as pd
import pytest

from src.data_loader import load_test, load_train
from src.features import (RUL_CAP, SENSOR_COLS, add_health_index, build,
                          sensor_noise)
from src.physics.health_fit import influence_matrix
from src.physics.off_design import OffDesignEngine, load_design


@pytest.fixture(scope="module")
def tables():
    tr_raw, te_raw = load_train(), load_test()
    H = influence_matrix(OffDesignEngine(load_design()))
    noise = sensor_noise(tr_raw)
    tr, te = build(tr_raw, H, noise), build(te_raw, H, noise)
    end = tr[tr["rul"] < 10].groupby("unit")[["d_eta_pct", "d_flow_pct"]].mean()
    for t in (tr, te):
        add_health_index(t, end["d_eta_pct"].mean(), end["d_flow_pct"].mean())
    return tr, te


def test_shapes_and_no_missing(tables):
    tr, te = tables
    assert len(tr) == 20631 and len(te) == 13096
    for t in (tr, te):
        assert t[SENSOR_COLS + ["d_eta_pct", "d_flow_pct", "hi"]].notna().all().all()
        assert t["rul_capped"].max() == RUL_CAP


def test_health_index_new_vs_failed(tables):
    tr, _ = tables
    new = tr[tr["cycle"] <= 20]["hi"].mean()
    failed = tr[tr["rul"] < 10]["hi"].mean()
    assert new == pytest.approx(1.0, abs=0.05)
    assert failed == pytest.approx(0.0, abs=0.1)


def test_health_index_tracks_test_rul(tables):
    _, te = tables
    last = te.groupby("unit").tail(1)
    rho = last[["hi", "rul"]].corr(method="spearman").iloc[0, 1]
    assert rho > 0.8