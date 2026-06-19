"""Convert probabilities to and from betting odds.

Three common formats are supported:

    * **Decimal** (European): total return per 1 unit staked, e.g. 2.50
    * **Fractional** (UK): profit/stake, e.g. "3/2"
    * **American** (moneyline): +150 (underdog) or -200 (favourite)

A probability of ``p`` corresponds to *fair* decimal odds of ``1 / p``. Real
bookmakers add a margin (the "overround") so the implied probabilities sum to
more than 100% — :func:`add_margin` lets you simulate that.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence

import config


def probability_to_decimal(prob: float) -> float:
    """Fair decimal odds for a probability. Returns ``inf`` for p <= 0."""
    if prob <= 0:
        return float("inf")
    return round(1.0 / prob, 2)


def decimal_to_probability(decimal_odds: float) -> float:
    """Implied probability from decimal odds."""
    if decimal_odds <= 0:
        return 0.0
    return 1.0 / decimal_odds


def probability_to_american(prob: float) -> int:
    """Fair American (moneyline) odds for a probability."""
    if prob <= 0:
        return 0
    if prob >= 0.5:
        # Favourite: negative line.
        return -round(100 * prob / (1 - prob))
    # Underdog: positive line.
    return round(100 * (1 - prob) / prob)


def probability_to_fractional(prob: float, max_denominator: int = 20) -> str:
    """Fair fractional odds (e.g. '7/2') for a probability."""
    if prob <= 0:
        return "∞"
    decimal = 1.0 / prob
    frac = Fraction(decimal - 1).limit_denominator(max_denominator)
    return f"{frac.numerator}/{frac.denominator}"


def add_margin(probs: Sequence[float], margin: float = config.DEFAULT_BOOKMAKER_MARGIN) -> list[float]:
    """Inflate fair probabilities by a bookmaker margin (overround).

    The returned values sum to ``1 + margin`` and, when turned into decimal
    odds, look like market prices rather than fair ones.
    """
    return [p * (1.0 + margin) for p in probs]


def remove_margin(implied_probs: Sequence[float]) -> list[float]:
    """Strip the overround from a set of implied probabilities (renormalise)."""
    total = sum(implied_probs)
    if total <= 0:
        n = len(implied_probs)
        return [1.0 / n] * n
    return [p / total for p in implied_probs]


def odds_table(probs: Sequence[float], labels: Sequence[str],
               margin: float = 0.0) -> list[dict]:
    """Build a tidy table of every odds format for a set of outcomes.

    Parameters
    ----------
    probs : fair probabilities (should sum to ~1)
    labels : outcome names, same length as ``probs``
    margin : optional bookmaker margin to apply to the displayed odds
    """
    priced = add_margin(probs, margin) if margin else list(probs)
    rows = []
    for label, fair_p, market_p in zip(labels, probs, priced):
        rows.append({
            "outcome": label,
            "probability": round(fair_p, 4),
            "decimal": probability_to_decimal(market_p),
            "fractional": probability_to_fractional(market_p),
            "american": probability_to_american(market_p),
        })
    return rows
