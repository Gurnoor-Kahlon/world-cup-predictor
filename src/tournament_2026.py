"""2026 World Cup mode — host advantage and tournament simulation.

The 2026 World Cup is co-hosted by the **USA, Canada and Mexico**. This module
adds a *partial* home advantage for those hosts (some host games are effectively
neutral, so we scale it by ``config.HOST_ADVANTAGE_FRACTION``) and provides a
host-aware Monte-Carlo bracket simulation.

> ⚠️ Qualified teams and groups for 2026 are not finalised, so the default team
> list here is a **placeholder** built from the strongest teams in the loaded
> data. Swap in the real teams/groups once they are confirmed.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np

import config
from predictor import MatchPredictor
from utils import clamp, normalize_probabilities

HOSTS = list(config.WORLD_CUP_2026_HOSTS)


def is_host(team: str) -> bool:
    """True if ``team`` is one of the 2026 co-hosts."""
    return team in HOSTS


def available_hosts(predictor: MatchPredictor) -> list[str]:
    """Hosts that actually appear in the loaded dataset."""
    return [h for h in HOSTS if h in predictor.teams]


def default_bracket(predictor: MatchPredictor, size: int = 8) -> list[str]:
    """A placeholder power-of-two bracket: strongest teams, hosts included.

    This is **illustrative only** — not the real 2026 field.
    """
    predictor._ensure_fitted()
    ranked = sorted(predictor.teams, key=lambda t: predictor.elo.get_rating(t), reverse=True)
    chosen = [t for t in available_hosts(predictor)]
    for t in ranked:
        if len(chosen) >= size:
            break
        if t not in chosen:
            chosen.append(t)
    return chosen[:size]


def _heads(predictor: MatchPredictor, home: str, away: str, neutral: bool) -> list[float]:
    """[P(home), P(draw), P(away)] from the Poisson + Elo heads (fast, no ML)."""
    elo_diff = predictor.elo.get_rating(home) - predictor.elo.get_rating(away)
    lam_h, lam_a = predictor.poisson.expected_goals(home, away, neutral, elo_diff)
    p_pois = predictor.poisson.result_probabilities(predictor.poisson.scoreline_grid(lam_h, lam_a))
    p_elo = predictor.elo.match_probabilities(home, away, neutral)
    w = predictor.blend_weights
    tot = w["poisson"] + w["elo"]
    return normalize_probabilities([
        (w["poisson"] * p_pois[i] + w["elo"] * p_elo[i]) / tot for i in range(3)
    ])


def match_probabilities(
    predictor: MatchPredictor,
    team_a: str,
    team_b: str,
    host_fraction: float = config.HOST_ADVANTAGE_FRACTION,
) -> list[float]:
    """Host-aware [P(A win), P(draw), P(B win)].

    If exactly one side is a host, its full-home probabilities are blended with
    the neutral ones by ``host_fraction`` (0 = fully neutral, 1 = full home).
    """
    neutral = _heads(predictor, team_a, team_b, neutral=True)
    a_host, b_host = is_host(team_a), is_host(team_b)

    if a_host and not b_host:
        home = _heads(predictor, team_a, team_b, neutral=False)
    elif b_host and not a_host:
        # Compute from B's home perspective, then flip back to A's perspective.
        b_home = _heads(predictor, team_b, team_a, neutral=False)
        home = [b_home[2], b_home[1], b_home[0]]
    else:
        return neutral  # both hosts or neither: treat as neutral

    frac = clamp(host_fraction, 0.0, 1.0)
    blended = [(1 - frac) * neutral[i] + frac * home[i] for i in range(3)]
    return normalize_probabilities(blended)


def advance_probability(
    predictor: MatchPredictor,
    team_a: str,
    team_b: str,
    host_fraction: float = config.HOST_ADVANTAGE_FRACTION,
) -> float:
    """P(team_a advances) for a host-aware knockout tie (draw -> shootout split)."""
    p_a, p_draw, p_b = match_probabilities(predictor, team_a, team_b, host_fraction)
    # Elo-nudged shootout split, mirroring MatchPredictor._shootout_split.
    exp = predictor.elo.win_expectation(team_a, team_b, neutral=True)
    shootout = clamp(0.5 + 0.18 * (2 * exp - 1), 0.35, 0.65)
    return clamp(p_a + p_draw * shootout, 0.0, 1.0)


def simulate_bracket(
    predictor: MatchPredictor,
    bracket: list[str],
    n_sims: int = 5000,
    seed: int = 0,
    host_fraction: float = config.HOST_ADVANTAGE_FRACTION,
) -> dict[str, float]:
    """Monte-Carlo a single-elimination bracket with host advantage.

    Returns each team's estimated title probability. ``bracket`` length must be a
    power of two; adjacent entries meet in round one.
    """
    size = len(bracket)
    if size < 2 or (size & (size - 1)) != 0:
        raise ValueError("Bracket size must be a power of two (2, 4, 8, 16...).")
    predictor._ensure_fitted()

    @lru_cache(maxsize=4096)
    def pair_prob(a: str, b: str) -> float:
        return advance_probability(predictor, a, b, host_fraction)

    rng = np.random.default_rng(seed)
    titles = {team: 0 for team in bracket}
    for _ in range(n_sims):
        alive = list(bracket)
        while len(alive) > 1:
            nxt = []
            for i in range(0, len(alive), 2):
                a, b = alive[i], alive[i + 1]
                nxt.append(a if rng.random() < pair_prob(a, b) else b)
            alive = nxt
        titles[alive[0]] += 1
    return {team: titles[team] / n_sims for team in bracket}
