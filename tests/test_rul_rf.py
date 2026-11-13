"""Checks for the RUL scoring functions (fast: no model training)."""
import pytest

from src.rul_rf import nasa_score, rmse


def test_rmse():
    assert rmse([10, 20], [10, 20]) == 0
    assert rmse([13, 16], [10, 20]) == pytest.approx(3.5355, abs=1e-3)


def test_nasa_score_punishes_late_more():
    early = nasa_score([90], [100])    # predicted 10 cycles too early
    late = nasa_score([110], [100])    # predicted 10 cycles too late
    assert nasa_score([50], [50]) == 0
    assert late > early > 0