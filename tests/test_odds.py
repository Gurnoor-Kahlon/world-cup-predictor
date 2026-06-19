"""Tests for the odds conversion utilities."""


import pytest

from odds_converter import (
    add_margin,
    decimal_to_probability,
    odds_table,
    probability_to_american,
    probability_to_decimal,
    probability_to_fractional,
    remove_margin,
)


def test_decimal_from_probability():
    assert probability_to_decimal(0.5) == 2.0
    assert probability_to_decimal(0.25) == 4.0
    assert probability_to_decimal(1.0) == 1.0


def test_decimal_zero_probability_is_infinite():
    assert probability_to_decimal(0.0) == float("inf")
    assert probability_to_decimal(-0.1) == float("inf")


def test_decimal_probability_round_trip():
    for p in (0.1, 0.33, 0.5, 0.8):
        assert decimal_to_probability(probability_to_decimal(p)) == pytest.approx(p, abs=1e-2)


def test_american_odds_sign():
    # Even money is -100.
    assert probability_to_american(0.5) == -100
    # Favourite (p > 0.5) gives a negative line.
    assert probability_to_american(0.8) < 0
    # Underdog (p < 0.5) gives a positive line.
    assert probability_to_american(0.25) == 300


def test_fractional_is_clean_string():
    frac = probability_to_fractional(0.5)
    assert "/" in frac
    num, den = frac.split("/")
    assert int(num) > 0 and int(den) > 0


def test_add_and_remove_margin():
    fair = [0.5, 0.3, 0.2]
    priced = add_margin(fair, margin=0.05)
    assert sum(priced) == pytest.approx(1.05, abs=1e-9)
    # Removing the margin renormalises back to 1.
    assert sum(remove_margin(priced)) == pytest.approx(1.0, abs=1e-9)


def test_remove_margin_recovers_fair_ratios():
    fair = [0.5, 0.3, 0.2]
    recovered = remove_margin(add_margin(fair, 0.08))
    for a, b in zip(fair, recovered):
        assert a == pytest.approx(b, abs=1e-9)


def test_odds_table_structure():
    rows = odds_table([0.5, 0.3, 0.2], ["Home", "Draw", "Away"], margin=0.05)
    assert len(rows) == 3
    expected_keys = {"outcome", "probability", "decimal", "fractional", "american"}
    for row in rows:
        assert expected_keys.issubset(row.keys())
    assert rows[0]["outcome"] == "Home"
