"""Tests for 2026 World Cup mode (host advantage + simulation)."""

import pytest

import tournament_2026 as wc
from tournament_2026 import (
    advance_probability,
    is_host,
    match_probabilities,
    simulate_bracket,
)


def test_is_host():
    assert is_host("USA")
    assert is_host("Mexico")
    assert not is_host("Brazil")


def test_host_advantage_helps_the_host(fitted_predictor):
    # USA is a 2026 host and appears in the sample data.
    neutral = match_probabilities(fitted_predictor, "USA", "Brazil", host_fraction=0.0)
    hosted = match_probabilities(fitted_predictor, "USA", "Brazil", host_fraction=1.0)
    # With host advantage, USA's win probability should rise.
    assert hosted[0] > neutral[0]


def test_match_probabilities_sum_to_one(fitted_predictor):
    probs = match_probabilities(fitted_predictor, "USA", "Germany")
    assert sum(probs) == pytest.approx(1.0, abs=1e-9)


def test_advance_probability_bounds(fitted_predictor):
    p = advance_probability(fitted_predictor, "Mexico", "Argentina")
    assert 0.0 <= p <= 1.0


def test_simulate_bracket_sums_to_one(fitted_predictor):
    bracket = ["USA", "Brazil", "Mexico", "Germany"]
    odds = simulate_bracket(fitted_predictor, bracket, n_sims=800, seed=3)
    assert sum(odds.values()) == pytest.approx(1.0, abs=1e-9)


def test_simulate_bracket_rejects_bad_size(fitted_predictor):
    with pytest.raises(ValueError):
        simulate_bracket(fitted_predictor, ["USA", "Brazil", "Mexico"], n_sims=10)


def test_default_bracket_includes_available_hosts(fitted_predictor):
    bracket = wc.default_bracket(fitted_predictor, size=8)
    assert len(bracket) == 8
    # USA and Mexico are in the sample data and should be seeded in.
    assert "USA" in bracket and "Mexico" in bracket
