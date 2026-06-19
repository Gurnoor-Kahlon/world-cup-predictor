"""The three statistical / ML engines behind a prediction.

This module deliberately keeps each technique in its own small, testable class:

    EloRatingSystem  – long-run team strength, updated match by match.
    PoissonGoalModel – attack/defence strengths -> expected goals -> scoreline grid.
    ResultClassifier – a gradient-boosting (or optional XGBoost/LightGBM) model
                       that learns Win/Draw/Loss from engineered features.

``predictor.py`` combines them into a single, explained prediction.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import poisson

import config
from utils import clamp, normalize_probabilities


# ===========================================================================
# 1. Elo rating system
# ===========================================================================
class EloRatingSystem:
    """A classic Elo engine adapted for football.

    Ratings start at ``ELO_INITIAL`` and move after every match based on the
    result, a home-advantage adjustment and the importance of the competition.
    """

    def __init__(
        self,
        k_factor: float = config.ELO_K,
        initial: float = config.ELO_INITIAL,
        home_advantage: float = config.ELO_HOME_ADVANTAGE,
    ):
        self.k_factor = k_factor
        self.initial = initial
        self.home_advantage = home_advantage
        self.ratings: dict[str, float] = {}
        # Per-match pre-match ratings, aligned to the order matches were fitted.
        self.history: list[dict] = []

    # -- internal helpers ---------------------------------------------------
    def get_rating(self, team: str) -> float:
        """Current rating for a team (creates it at the initial value if new)."""
        return self.ratings.setdefault(team, self.initial)

    @staticmethod
    def _expected_score(rating_a: float, rating_b: float) -> float:
        """Elo expectation: probability-like score for A vs B (0..1)."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def _tournament_weight(self, tournament: str) -> float:
        name = str(tournament).lower()
        for key, weight in config.ELO_TOURNAMENT_WEIGHTS.items():
            if key in name:
                return weight
        return config.ELO_DEFAULT_TOURNAMENT_WEIGHT

    @staticmethod
    def _margin_multiplier(goal_diff: int) -> float:
        """Bigger wins move ratings a little more (diminishing returns)."""
        return math.log(abs(goal_diff) + 1) + 1.0

    # -- fitting ------------------------------------------------------------
    def fit(self, matches: pd.DataFrame) -> "EloRatingSystem":
        """Process matches in chronological order, learning ratings.

        Records the pre-match ratings of every game so feature engineering can
        use point-in-time (no-leakage) values.
        """
        self.ratings = {}
        self.history = []
        for row in matches.itertuples(index=False):
            self.update_match(
                row.home_team, row.away_team, row.home_score, row.away_score,
                neutral=bool(getattr(row, "neutral", False)),
                tournament=getattr(row, "tournament", ""),
                record_history=True,
            )
        return self

    def update_match(
        self,
        home: str,
        away: str,
        home_score: int,
        away_score: int,
        neutral: bool = False,
        tournament: str = "",
        record_history: bool = False,
    ) -> None:
        """Apply a single match's Elo update (used by both fit and backtests).

        Set ``record_history=True`` to append the pre-match ratings to
        ``self.history`` (needed for leak-free feature engineering).
        """
        r_home = self.get_rating(home)
        r_away = self.get_rating(away)
        if record_history:
            self.history.append({"home_elo": r_home, "away_elo": r_away})

        # Home advantage only applies at non-neutral venues.
        adv = 0.0 if neutral else self.home_advantage
        exp_home = self._expected_score(r_home + adv, r_away)
        exp_away = 1.0 - exp_home

        # Actual result as a score: win=1, draw=0.5, loss=0.
        if home_score > away_score:
            s_home, s_away = 1.0, 0.0
        elif home_score < away_score:
            s_home, s_away = 0.0, 1.0
        else:
            s_home = s_away = 0.5

        weight = self._tournament_weight(tournament)
        margin = self._margin_multiplier(home_score - away_score)
        k = self.k_factor * weight * margin

        self.ratings[home] = r_home + k * (s_home - exp_home)
        self.ratings[away] = r_away + k * (s_away - exp_away)

    # -- prediction-time heuristics ----------------------------------------
    def win_expectation(self, home: str, away: str, neutral: bool = False) -> float:
        """Elo expected score for the home team (0..1), venue-adjusted."""
        adv = 0.0 if neutral else self.home_advantage
        return self._expected_score(self.get_rating(home) + adv, self.get_rating(away))

    def match_probabilities(self, home: str, away: str, neutral: bool = False) -> list[float]:
        """Heuristic [P(home win), P(draw), P(away win)] from Elo alone.

        Draws are modelled as more likely when the two sides are close in
        strength. This is a transparent heuristic, used only as a minor anchor
        in the ensemble (the Poisson and ML heads do the heavy lifting).
        """
        exp_home = self.win_expectation(home, away, neutral)
        # Draw probability peaks (~0.28) for even games, shrinks for mismatches.
        draw = 0.28 * (1.0 - abs(2 * exp_home - 1)) + 0.06
        draw = clamp(draw, 0.05, 0.34)
        home_win = (1.0 - draw) * exp_home
        away_win = (1.0 - draw) * (1.0 - exp_home)
        return normalize_probabilities([home_win, draw, away_win])


# ===========================================================================
# 2. Poisson goal model
# ===========================================================================
class PoissonGoalModel:
    """Expected-goals model built from team attack & defence strengths.

    For a fixture it produces two Poisson rate parameters (lambda_home,
    lambda_away), then a full grid of scoreline probabilities.
    """

    def __init__(
        self,
        max_goals: int = config.MAX_GOALS,
        home_advantage: float = config.POISSON_HOME_ADVANTAGE,
        elo_scale: float = config.ELO_TO_GOALS_SCALE,
        dixon_coles: bool = config.DIXON_COLES_ENABLED,
        rho: float = config.DIXON_COLES_RHO,
    ):
        self.max_goals = max_goals
        self.home_advantage = home_advantage
        self.elo_scale = elo_scale
        self.dixon_coles = dixon_coles
        self.rho = rho
        self.league_avg: float = config.MIN_EXPECTED_GOALS
        self.attack: dict[str, float] = {}
        self.defence: dict[str, float] = {}

    def fit(self, matches: pd.DataFrame) -> "PoissonGoalModel":
        """Estimate per-team attack/defence relative to the league average."""
        total_goals = matches["home_score"].sum() + matches["away_score"].sum()
        n_team_matches = 2 * len(matches)
        self.league_avg = max(total_goals / n_team_matches, config.MIN_EXPECTED_GOALS)

        # Aggregate goals scored & conceded per team (home and away combined).
        scored: dict[str, list[int]] = {}
        conceded: dict[str, list[int]] = {}
        for row in matches.itertuples(index=False):
            scored.setdefault(row.home_team, []).append(row.home_score)
            conceded.setdefault(row.home_team, []).append(row.away_score)
            scored.setdefault(row.away_team, []).append(row.away_score)
            conceded.setdefault(row.away_team, []).append(row.home_score)

        for team in scored:
            avg_scored = float(np.mean(scored[team]))
            avg_conceded = float(np.mean(conceded[team]))
            # Strength of 1.0 == exactly league average.
            self.attack[team] = avg_scored / self.league_avg
            self.defence[team] = avg_conceded / self.league_avg
        return self

    def _strength(self, team: str, table: dict[str, float]) -> float:
        """Look up a strength, defaulting to league-average (1.0) for new teams."""
        return table.get(team, 1.0)

    def expected_goals(
        self,
        home: str,
        away: str,
        neutral: bool = False,
        elo_diff: float = 0.0,
    ) -> tuple[float, float]:
        """Expected goals for each side.

        Combines the attack*defence/avg model, a home-venue multiplier and a
        small adjustment from the Elo rating gap (``elo_diff`` = home - away).
        """
        atk_home = self._strength(home, self.attack)
        atk_away = self._strength(away, self.attack)
        def_home = self._strength(home, self.defence)
        def_away = self._strength(away, self.defence)

        lam_home = atk_home * def_away * self.league_avg
        lam_away = atk_away * def_home * self.league_avg

        if not neutral:
            lam_home *= self.home_advantage

        # Elo nudge: stronger team scores a touch more, concedes a touch less.
        # sqrt split keeps the total expected goals roughly stable.
        factor = math.exp(self.elo_scale * (elo_diff / 100.0))
        lam_home *= math.sqrt(factor)
        lam_away /= math.sqrt(factor)

        lam_home = clamp(lam_home, config.MIN_EXPECTED_GOALS, float(self.max_goals))
        lam_away = clamp(lam_away, config.MIN_EXPECTED_GOALS, float(self.max_goals))
        return lam_home, lam_away

    def scoreline_grid(
        self, lam_home: float, lam_away: float, dixon_coles: Optional[bool] = None
    ) -> np.ndarray:
        """Matrix ``P[i, j]`` = probability of home i goals, away j goals.

        When Dixon-Coles is enabled, the four low-score cells (0-0, 1-0, 0-1,
        1-1) are nudged to better match real football, where independent Poisson
        slightly mis-prices tight games.
        """
        goals = np.arange(self.max_goals + 1)
        home_pmf = poisson.pmf(goals, lam_home)
        away_pmf = poisson.pmf(goals, lam_away)
        grid = np.outer(home_pmf, away_pmf)

        use_dc = self.dixon_coles if dixon_coles is None else dixon_coles
        if use_dc:
            grid = self._apply_dixon_coles(grid, lam_home, lam_away)

        # Renormalise (the truncated tail beyond max_goals loses a little mass).
        return grid / grid.sum()

    def _apply_dixon_coles(self, grid: np.ndarray, lam: float, mu: float) -> np.ndarray:
        """Multiply the four low-score cells by the Dixon-Coles tau factor."""
        rho = self.rho
        tau = {
            (0, 0): 1.0 - lam * mu * rho,
            (0, 1): 1.0 + lam * rho,
            (1, 0): 1.0 + mu * rho,
            (1, 1): 1.0 - rho,
        }
        grid = grid.copy()
        for (i, j), factor in tau.items():
            grid[i, j] *= max(factor, 0.0)   # keep probabilities non-negative
        return grid

    @staticmethod
    def result_probabilities(grid: np.ndarray) -> list[float]:
        """[P(home win), P(draw), P(away win)] from a scoreline grid."""
        home_win = float(np.tril(grid, -1).sum())   # home goals > away goals
        away_win = float(np.triu(grid, 1).sum())    # away goals > home goals
        draw = float(np.trace(grid))
        return normalize_probabilities([home_win, draw, away_win])

    def top_scorelines(self, grid: np.ndarray, n: int = 5) -> list[tuple[int, int, float]]:
        """The ``n`` most likely exact scorelines as (home, away, probability)."""
        flat = [
            (i, j, float(grid[i, j]))
            for i in range(grid.shape[0])
            for j in range(grid.shape[1])
        ]
        flat.sort(key=lambda x: x[2], reverse=True)
        return flat[:n]


# ===========================================================================
# 3. Machine-learning result classifier
# ===========================================================================
def _build_estimator():
    """Return the best available gradient-boosting classifier.

    Prefers XGBoost or LightGBM if installed, otherwise falls back to
    scikit-learn's GradientBoostingClassifier (always available). The fallback
    means the project works with only the default requirements installed.
    """
    try:  # pragma: no cover - depends on optional install
        from xgboost import XGBClassifier

        return ("xgboost", XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.9, eval_metric="mlogloss", random_state=42,
        ))
    except Exception:
        pass
    try:  # pragma: no cover - depends on optional install
        from lightgbm import LGBMClassifier

        return ("lightgbm", LGBMClassifier(
            n_estimators=300, max_depth=-1, learning_rate=0.05,
            subsample=0.9, random_state=42, verbose=-1,
        ))
    except Exception:
        pass

    from sklearn.ensemble import GradientBoostingClassifier

    return ("sklearn-gbdt", GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42,
    ))


class ResultClassifier:
    """Thin wrapper around a gradient-boosting classifier for H/D/A outcomes."""

    CLASSES = ["H", "D", "A"]

    def __init__(self):
        self.backend, self.estimator = _build_estimator()
        self.feature_names: list[str] = []
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ResultClassifier":
        self.feature_names = list(X.columns)
        self.estimator.fit(X.values, y.values)
        self._fitted = True
        return self

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def predict_proba(self, features: dict[str, float]) -> dict[str, float]:
        """Predict {'H':.., 'D':.., 'A':..} for a single feature row."""
        if not self._fitted:
            raise RuntimeError("ResultClassifier must be fit before predicting.")
        row = np.array([[features.get(name, 0.0) for name in self.feature_names]])
        proba = self.estimator.predict_proba(row)[0]
        # Map model's internal class order to our fixed H/D/A order.
        result = {cls: 0.0 for cls in self.CLASSES}
        for cls, p in zip(self.estimator.classes_, proba):
            result[str(cls)] = float(p)
        return result
