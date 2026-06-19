"""Turn raw match history into model features.

The same feature definitions are used for **training** (looping over history,
always looking only at the past to avoid leakage) and for **prediction** (using
the most recent available history for each team). Sharing one ``_feature_row``
function guarantees the two never drift apart.

Features (all from the home team's perspective unless noted):
    elo_diff           home Elo minus away Elo (pre-match, venue-neutral)
    form_points_diff   recent points-per-game gap (last N matches)
    form_gd_diff       recent goal-difference gap
    attack_diff        recent goals-scored-per-game gap
    defence_diff       recent goals-conceded-per-game gap (home minus away)
    sos_diff           strength-of-schedule gap (avg opponent Elo faced)
    h2h_home           head-to-head balance for the home team (-1..1)
    rest_diff          rest-days gap (capped)
    neutral            1 if neutral venue else 0
    stage_importance   tournament-stage weight (see config.STAGE_IMPORTANCE)
"""

from __future__ import annotations

from collections import defaultdict, deque

import pandas as pd

import config
from model import EloRatingSystem
from utils import points_for_result

FEATURE_COLUMNS = [
    "elo_diff",
    "form_points_diff",
    "form_gd_diff",
    "attack_diff",
    "defence_diff",
    "sos_diff",
    "h2h_home",
    "rest_diff",
    "neutral",
    "stage_importance",
]

REST_CAP_DAYS = 30
DEFAULT_REST_DAYS = 14


def _tournament_importance(tournament: str) -> float:
    """Approximate stage importance for *training* rows (no explicit stage)."""
    name = str(tournament).lower()
    if "world cup" in name and "qualif" not in name:
        return 1.3
    if "qualif" in name:
        return 1.0
    if "friendly" in name:
        return 0.9
    return 1.0


def _recent_stats(window: deque) -> dict:
    """Summarise a team's recent matches (a deque of per-match dicts)."""
    if not window:
        return {"ppg": 1.0, "gd": 0.0, "attack": 1.0, "defence": 1.0, "sos": config.ELO_INITIAL}
    n = len(window)
    return {
        "ppg": sum(m["points"] for m in window) / n,
        "gd": sum(m["scored"] - m["conceded"] for m in window) / n,
        "attack": sum(m["scored"] for m in window) / n,
        "defence": sum(m["conceded"] for m in window) / n,
        "sos": sum(m["opp_elo"] for m in window) / n,
    }


def _feature_row(
    home_stats: dict,
    away_stats: dict,
    elo_diff: float,
    h2h_home: float,
    rest_diff: float,
    neutral: bool,
    stage_importance: float,
) -> dict:
    """Assemble the final feature dict from pre-computed pieces."""
    return {
        "elo_diff": elo_diff,
        "form_points_diff": home_stats["ppg"] - away_stats["ppg"],
        "form_gd_diff": home_stats["gd"] - away_stats["gd"],
        "attack_diff": home_stats["attack"] - away_stats["attack"],
        "defence_diff": home_stats["defence"] - away_stats["defence"],
        "sos_diff": home_stats["sos"] - away_stats["sos"],
        "h2h_home": h2h_home,
        "rest_diff": rest_diff,
        "neutral": 1.0 if neutral else 0.0,
        "stage_importance": stage_importance,
    }


class FeatureEngineer:
    """Builds training tables and single-match prediction features."""

    def __init__(self, form_window: int = config.FORM_WINDOW):
        self.form_window = form_window

    # -- training -----------------------------------------------------------
    def build_training_table(
        self, matches: pd.DataFrame, elo: EloRatingSystem
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Create (X, y) for the ML classifier with no future leakage.

        ``elo`` must already be fitted on the same ``matches`` so that
        ``elo.history`` provides pre-match ratings aligned by row order.
        """
        if len(elo.history) != len(matches):
            raise ValueError("Elo history length does not match the dataset; "
                             "fit the EloRatingSystem on these matches first.")

        recent: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.form_window))
        last_date: dict[str, pd.Timestamp | None] = {}
        h2h: dict[frozenset, list] = defaultdict(list)
        has_dates = "date" in matches.columns

        rows, targets = [], []
        for i, row in enumerate(matches.itertuples(index=False)):
            home, away = row.home_team, row.away_team
            home_elo = elo.history[i]["home_elo"]
            away_elo = elo.history[i]["away_elo"]

            home_stats = _recent_stats(recent[home])
            away_stats = _recent_stats(recent[away])
            h2h_home = self._h2h_balance(h2h[frozenset((home, away))], home)
            rest_diff = self._rest_diff(last_date, home, away,
                                        getattr(row, "date", None) if has_dates else None)
            stage = _tournament_importance(getattr(row, "tournament", ""))

            rows.append(_feature_row(home_stats, away_stats, home_elo - away_elo,
                                     h2h_home, rest_diff, bool(getattr(row, "neutral", False)),
                                     stage))
            targets.append(row.result)

            # --- update running state AFTER recording the row (no leakage) ---
            self._update_state(recent, last_date, h2h, row, home_elo, away_elo, has_dates)

        X = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
        y = pd.Series(targets, name="result")
        return X, y

    # -- prediction ---------------------------------------------------------
    def latest_features(
        self,
        matches: pd.DataFrame,
        elo: EloRatingSystem,
        home: str,
        away: str,
        neutral: bool,
        stage: str,
    ) -> dict:
        """Features for an upcoming fixture using the most recent history."""
        recent_home = self._tail_window(matches, elo, home)
        recent_away = self._tail_window(matches, elo, away)
        home_stats = _recent_stats(recent_home)
        away_stats = _recent_stats(recent_away)

        h2h_list = self._all_meetings(matches, home, away)
        h2h_home = self._h2h_balance(h2h_list, home)

        rest_diff = self._latest_rest_diff(matches, home, away)
        elo_diff = elo.get_rating(home) - elo.get_rating(away)
        stage_importance = config.STAGE_IMPORTANCE.get(stage, 1.0)

        return _feature_row(home_stats, away_stats, elo_diff, h2h_home,
                            rest_diff, neutral, stage_importance)

    # -- helpers ------------------------------------------------------------
    def _update_state(self, recent, last_date, h2h, row, home_elo, away_elo, has_dates):
        home, away = row.home_team, row.away_team
        res = row.result
        recent[home].append({
            "scored": row.home_score, "conceded": row.away_score,
            "points": points_for_result(res, "home"), "opp_elo": away_elo,
        })
        recent[away].append({
            "scored": row.away_score, "conceded": row.home_score,
            "points": points_for_result(res, "away"), "opp_elo": home_elo,
        })
        h2h[frozenset((home, away))].append({
            "home": home, "away": away,
            "home_score": row.home_score, "away_score": row.away_score,
        })
        if has_dates:
            d = getattr(row, "date", None)
            last_date[home] = d
            last_date[away] = d

    @staticmethod
    def _h2h_balance(meetings: list, team: str) -> float:
        """Head-to-head balance in [-1, 1] from ``team``'s perspective."""
        if not meetings:
            return 0.0
        wins = losses = 0
        for m in meetings:
            team_is_home = m["home"] == team
            gf = m["home_score"] if team_is_home else m["away_score"]
            ga = m["away_score"] if team_is_home else m["home_score"]
            wins += gf > ga
            losses += gf < ga
        return (wins - losses) / len(meetings)

    @staticmethod
    def _rest_diff(last_date, home, away, match_date) -> float:
        if match_date is None:
            return 0.0
        def rest(team):
            prev = last_date.get(team)
            if prev is None or pd.isna(prev):
                return DEFAULT_REST_DAYS
            days = (match_date - prev).days
            return max(0, min(days, REST_CAP_DAYS))
        return rest(home) - rest(away)

    def _tail_window(self, matches, elo, team) -> deque:
        """Most recent ``form_window`` matches for a team, as stat dicts."""
        window: deque = deque(maxlen=self.form_window)
        team_rows = matches[(matches["home_team"] == team) | (matches["away_team"] == team)]
        for row in team_rows.tail(self.form_window).itertuples(index=False):
            is_home = row.home_team == team
            scored = row.home_score if is_home else row.away_score
            conceded = row.away_score if is_home else row.home_score
            opp = row.away_team if is_home else row.home_team
            window.append({
                "scored": scored, "conceded": conceded,
                "points": points_for_result(row.result, "home" if is_home else "away"),
                "opp_elo": elo.get_rating(opp),
            })
        return window

    @staticmethod
    def _all_meetings(matches, home, away) -> list:
        mask = (
            ((matches["home_team"] == home) & (matches["away_team"] == away))
            | ((matches["home_team"] == away) & (matches["away_team"] == home))
        )
        return [
            {"home": r.home_team, "away": r.away_team,
             "home_score": r.home_score, "away_score": r.away_score}
            for r in matches[mask].itertuples(index=False)
        ]

    @staticmethod
    def _latest_rest_diff(matches, home, away) -> float:
        if "date" not in matches.columns:
            return 0.0
        # Without a fixture date we cannot know real rest; treat as equal.
        return 0.0
