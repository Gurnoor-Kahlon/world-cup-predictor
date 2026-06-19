"""Tests for evaluation metrics and the walk-forward backtester."""

import math

import numpy as np
import pytest

from evaluation import (
    Backtester,
    accuracy_score,
    brier_score,
    calibration_curve_data,
    log_loss_score,
)


# --------------------------------------------------------------------------- #
# Metric functions
# --------------------------------------------------------------------------- #
def test_perfect_predictions_score_perfectly():
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    actual = np.array([0, 1, 2])
    assert accuracy_score(probs, actual) == 1.0
    assert brier_score(probs, actual) == pytest.approx(0.0, abs=1e-12)
    assert log_loss_score(probs, actual) == pytest.approx(0.0, abs=1e-9)


def test_uniform_predictions_have_expected_log_loss():
    probs = np.full((5, 3), 1 / 3)
    actual = np.array([0, 1, 2, 0, 1])
    # Log loss of a uniform 3-class prediction is ln(3).
    assert log_loss_score(probs, actual) == pytest.approx(math.log(3), abs=1e-9)


def test_accuracy_uses_argmax():
    probs = np.array([[0.6, 0.3, 0.1], [0.1, 0.2, 0.7]])
    actual = np.array([0, 0])   # second one is wrong
    assert accuracy_score(probs, actual) == pytest.approx(0.5)


def test_brier_known_value():
    probs = np.array([[0.7, 0.2, 0.1]])
    actual = np.array([0])
    # (0.7-1)^2 + 0.2^2 + 0.1^2 = 0.09 + 0.04 + 0.01 = 0.14
    assert brier_score(probs, actual) == pytest.approx(0.14, abs=1e-9)


def test_calibration_curve_structure():
    rng = np.random.default_rng(0)
    pred = rng.random(200)
    actual = (rng.random(200) < pred).astype(float)
    cal = calibration_curve_data(pred, actual, n_bins=5)
    assert {"mean_predicted", "observed_frequency", "count"} <= set(cal.columns)
    assert cal["count"].sum() == 200


# --------------------------------------------------------------------------- #
# Backtester
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def backtest_result(sample_matches):
    # Use a slice for speed; still exercises the full walk-forward path.
    subset = sample_matches.head(600)
    return Backtester(subset).run(test_fraction=0.2)


def test_backtest_returns_all_models(backtest_result):
    metrics = backtest_result["metrics"]
    for model in ("elo", "poisson", "ml", "ensemble"):
        assert model in metrics
        assert 0.0 <= metrics[model]["accuracy"] <= 1.0
        assert metrics[model]["log_loss"] > 0


def test_backtest_predictions_are_valid_distributions(backtest_result):
    preds = backtest_result["predictions"]
    assert len(preds) == backtest_result["n_test"]
    for model in ("elo", "poisson", "ensemble"):
        cols = [f"{model}_H", f"{model}_D", f"{model}_A"]
        sums = preds[cols].sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-6)


def test_backtest_beats_uniform_baseline(backtest_result):
    # A useful model should not be *worse* than blind guessing (ln 3 ~ 1.0986).
    assert backtest_result["metrics"]["ensemble"]["log_loss"] < math.log(3)


def test_backtest_tournament_filter(sample_matches):
    res = Backtester(sample_matches.head(800)).run(test_fraction=0.25,
                                                   tournament="Friendly")
    assert res["n_test"] > 0
    assert (res["predictions"]["tournament"].str.contains("Friendly")).all()
