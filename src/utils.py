"""Small shared helper functions used across the project.

Nothing here is football-specific magic — just reusable numeric and formatting
utilities kept in one place so the other modules stay focused.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Numeric helpers
# --------------------------------------------------------------------------- #
def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide two numbers, returning ``default`` instead of raising on /0."""
    if denominator == 0 or denominator is None:
        return default
    return numerator / denominator


def normalize_probabilities(probs: Sequence[float]) -> list[float]:
    """Scale a list of non-negative numbers so they sum to 1.

    Falls back to a uniform distribution if the total is zero.
    """
    arr = np.asarray(probs, dtype=float)
    arr = np.clip(arr, 0.0, None)          # no negative probabilities
    total = arr.sum()
    if total <= 0:
        return [1.0 / len(arr)] * len(arr)
    return (arr / total).tolist()


def blend_distributions(
    distributions: dict[str, Sequence[float]],
    weights: dict[str, float],
) -> list[float]:
    """Weighted average of several probability distributions of equal length.

    Only keys present in *both* dicts are used. The result is re-normalised.
    """
    keys = [k for k in weights if k in distributions]
    if not keys:
        raise ValueError("No matching keys between distributions and weights.")

    length = len(next(iter(distributions.values())))
    combined = np.zeros(length, dtype=float)
    total_weight = 0.0
    for key in keys:
        dist = np.asarray(distributions[key], dtype=float)
        if len(dist) != length:
            raise ValueError("All distributions must have the same length.")
        combined += weights[key] * dist
        total_weight += weights[key]

    if total_weight > 0:
        combined /= total_weight
    return normalize_probabilities(combined)


def clamp(value: float, low: float, high: float) -> float:
    """Constrain ``value`` to the inclusive range [low, high]."""
    return max(low, min(high, value))


# --------------------------------------------------------------------------- #
# Football helpers
# --------------------------------------------------------------------------- #
def result_label(home_score: int, away_score: int) -> str:
    """Return 'H' (home win), 'D' (draw) or 'A' (away win) for a scoreline."""
    if home_score > away_score:
        return "H"
    if home_score < away_score:
        return "A"
    return "D"


def points_for_result(result: str, perspective: str = "home") -> int:
    """Football league points (3/1/0) for a result from one team's perspective."""
    if result == "D":
        return 1
    if perspective == "home":
        return 3 if result == "H" else 0
    return 3 if result == "A" else 0


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def as_percentages(probs: Iterable[float], decimals: int = 1) -> list[float]:
    """Convert probabilities (0..1) to rounded percentages (0..100)."""
    return [round(100 * p, decimals) for p in probs]
