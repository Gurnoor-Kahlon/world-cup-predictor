"""High-level match predictor that ties the whole pipeline together.

``MatchPredictor`` loads everything once (Elo, Poisson, ML classifier) and then
answers questions about individual fixtures with a rich, explained result:
probabilities, expected goals, likely scorelines, fair odds, a confidence rating,
a plain-English explanation, and — for knockout games — who is likely to advance.

It also offers a Monte-Carlo :meth:`simulate_tournament` for title odds.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import config
from data_loader import list_teams, load_matches, load_team_metadata
from feature_engineering import FeatureEngineer, _recent_stats
from model import EloRatingSystem, PoissonGoalModel, ResultClassifier
from utils import blend_distributions, clamp, normalize_probabilities


class MatchPredictor:
    """Train once, then predict any fixture."""

    def __init__(
        self,
        matches: pd.DataFrame | None = None,
        team_metadata: pd.DataFrame | None = None,
        form_window: int = config.FORM_WINDOW,
        blend_weights: dict[str, float] | None = None,
    ):
        self.matches = matches if matches is not None else load_matches()
        self.team_metadata = (
            team_metadata if team_metadata is not None else load_team_metadata()
        )
        self.teams = list_teams(self.matches)
        self.fe = FeatureEngineer(form_window=form_window)
        # Ensemble weights are configurable per-instance (default from config).
        self.blend_weights = dict(blend_weights or config.BLEND_WEIGHTS)

        self.elo = EloRatingSystem()
        self.poisson = PoissonGoalModel()
        self.classifier = ResultClassifier()
        self._fitted = False

        # How many matches each team has — used for confidence.
        counts = pd.concat([self.matches["home_team"], self.matches["away_team"]])
        self.team_match_counts = counts.value_counts().to_dict()

    # ----------------------------------------------------------------- fit
    def fit(self) -> MatchPredictor:
        """Train every component on the loaded history."""
        self.elo.fit(self.matches)
        self.poisson.fit(self.matches)
        X, y = self.fe.build_training_table(self.matches, self.elo)
        # A classifier needs at least two outcome classes to train.
        if y.nunique() >= 2 and len(X) >= 10:
            self.classifier.fit(X, y)
        self._fitted = True
        return self

    def _ensure_fitted(self):
        if not self._fitted:
            self.fit()

    # ------------------------------------------------------- persistence
    def save(self, path: Path | str | None = None) -> Path:
        """Persist the fitted predictor to disk with joblib.

        Saves the whole object (Elo ratings, Poisson strengths, trained ML
        model and the match history) so it can be reloaded without retraining.
        """
        self._ensure_fitted()
        path = Path(path) if path is not None else config.MODEL_ARTIFACT_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        self._pair_advance_prob.cache_clear()  # don't pickle a stale cache
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: Path | str | None = None) -> MatchPredictor:
        """Load a predictor previously saved with :meth:`save`."""
        path = Path(path) if path is not None else config.MODEL_ARTIFACT_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"No saved model at '{path}'. Train and save one first, e.g. "
                f"`python src/main.py --save-model`."
            )
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"File '{path}' did not contain a MatchPredictor.")
        return obj

    # ------------------------------------------------------------- predict
    def predict(
        self,
        home: str,
        away: str,
        stage: str = config.GROUP_STAGE,
        neutral: bool = False,
        context: dict | None = None,
    ) -> dict:
        """Predict a single fixture. See module docstring for the output shape.

        ``context`` is an optional dict for real-world adjustments you supply,
        e.g. ``{"home_injuries": 2, "rest_days_home": 3, "rest_days_away": 6}``.
        Supported keys: ``home_injuries``/``away_injuries`` (count of key players
        unavailable), ``home_xg_adjust``/``away_xg_adjust`` (multipliers),
        ``rest_days_home``/``rest_days_away``.
        """
        self._ensure_fitted()
        context = context or {}

        elo_home = self.elo.get_rating(home)
        elo_away = self.elo.get_rating(away)
        elo_diff = elo_home - elo_away

        # --- Poisson head -------------------------------------------------
        lam_home, lam_away = self.poisson.expected_goals(home, away, neutral, elo_diff)
        lam_home, lam_away = self._apply_context(lam_home, lam_away, context)
        grid = self.poisson.scoreline_grid(lam_home, lam_away)
        poisson_probs = self.poisson.result_probabilities(grid)

        # --- Elo head -----------------------------------------------------
        elo_probs = self.elo.match_probabilities(home, away, neutral)

        # --- ML head ------------------------------------------------------
        ml_probs = self._ml_probabilities(home, away, neutral, stage, context)

        # --- Blend --------------------------------------------------------
        components = {"poisson": poisson_probs, "elo": elo_probs}
        if ml_probs is not None:
            components["ml"] = ml_probs
        final = blend_distributions(components, self.blend_weights)
        home_win, draw, away_win = final

        # --- Extras -------------------------------------------------------
        top = [
            {"score": f"{i}-{j}", "home_goals": i, "away_goals": j, "prob": round(p, 4)}
            for i, j, p in self.poisson.top_scorelines(grid, n=5)
        ]
        confidence = self._confidence(final, poisson_probs, ml_probs, home, away)
        explanation = self._explain(home, away, neutral, stage, elo_home, elo_away,
                                    lam_home, lam_away, final, context)

        result = {
            "home": home,
            "away": away,
            "stage": stage,
            "neutral": neutral,
            "probabilities": {"home_win": home_win, "draw": draw, "away_win": away_win},
            "component_probabilities": {
                "poisson": poisson_probs,
                "elo": elo_probs,
                "ml": ml_probs,
            },
            "expected_goals": {"home": round(lam_home, 2), "away": round(lam_away, 2)},
            "top_scorelines": top,
            "elo": {"home": round(elo_home, 1), "away": round(elo_away, 1)},
            "confidence": confidence,
            "explanation": explanation["summary"],
            "key_factors": explanation["factors"],
            "blend_weights": {k: v for k, v in self.blend_weights.items()
                              if k in components},
        }

        # Knockout: who advances if the 90-min result is a draw?
        if stage in config.KNOCKOUT_STAGES:
            result["knockout"] = self._advance_probabilities(
                home, away, neutral, home_win, draw, away_win, context
            )
        return result

    # ------------------------------------------------------- internal bits
    def _apply_context(self, lam_home, lam_away, context) -> tuple[float, float]:
        """Apply user-supplied real-world adjustments to expected goals."""
        # Each unavailable key player trims ~4% off a side's expected goals.
        home_inj = float(context.get("home_injuries", 0) or 0)
        away_inj = float(context.get("away_injuries", 0) or 0)
        lam_home *= clamp(1.0 - 0.04 * home_inj, 0.6, 1.0)
        lam_away *= clamp(1.0 - 0.04 * away_inj, 0.6, 1.0)
        # Free-form multipliers (e.g. from an xG feed).
        lam_home *= float(context.get("home_xg_adjust", 1.0) or 1.0)
        lam_away *= float(context.get("away_xg_adjust", 1.0) or 1.0)
        lam_home = clamp(lam_home, config.MIN_EXPECTED_GOALS, float(self.poisson.max_goals))
        lam_away = clamp(lam_away, config.MIN_EXPECTED_GOALS, float(self.poisson.max_goals))
        return lam_home, lam_away

    def _ml_probabilities(self, home, away, neutral, stage, context):
        """Get [H, D, A] from the classifier, or None if it could not train."""
        if not self.classifier.is_fitted:
            return None
        feats = self.fe.latest_features(self.matches, self.elo, home, away, neutral, stage)
        if "rest_days_home" in context or "rest_days_away" in context:
            rh = float(context.get("rest_days_home", 7))
            ra = float(context.get("rest_days_away", 7))
            feats["rest_diff"] = clamp(rh - ra, -30, 30)
        proba = self.classifier.predict_proba(feats)
        return normalize_probabilities([proba["H"], proba["D"], proba["A"]])

    def _confidence(self, final, poisson_probs, ml_probs, home, away) -> dict:
        """A 0-100 confidence score with human-readable reasons.

        Built from three transparent ingredients (documented, not magic):
          * decisiveness – how far the top probability is above a coin toss,
          * agreement    – how closely the Poisson and ML heads agree,
          * data depth   – how much match history both teams have.
        """
        decisiveness = clamp((max(final) - 1 / 3) / (1 - 1 / 3), 0, 1)

        if ml_probs is not None:
            l1 = sum(abs(a - b) for a, b in zip(poisson_probs, ml_probs))
            agreement = clamp(1 - l1 / 2, 0, 1)
        else:
            agreement = 0.5  # only one statistical head available

        min_games = min(self.team_match_counts.get(home, 0),
                        self.team_match_counts.get(away, 0))
        data_depth = clamp(min_games / max(config.MIN_MATCHES_FOR_CONFIDENCE, 1), 0, 1)

        score = 100 * (0.5 * decisiveness + 0.3 * agreement + 0.2 * data_depth)
        score = round(score)
        if score >= 67:
            label = "High"
        elif score >= 40:
            label = "Medium"
        else:
            label = "Low"

        reasons = []
        reasons.append("one outcome is clearly favoured" if decisiveness > 0.5
                       else "the outcome is fairly open")
        reasons.append("the statistical and ML models agree" if agreement > 0.7
                       else "the models partly disagree")
        if data_depth < 0.5:
            reasons.append("limited match history for at least one team")
        return {"score": score, "label": label, "reasons": reasons}

    def _advance_probabilities(self, home, away, neutral, home_win, draw, away_win, context):
        """Split the draw into who progresses (extra time / penalties)."""
        p_home_shootout = self._shootout_split(home, away, neutral, context)
        return {
            "home_advance": round(home_win + draw * p_home_shootout, 4),
            "away_advance": round(away_win + draw * (1 - p_home_shootout), 4),
            "home_shootout_edge": round(p_home_shootout, 3),
        }

    def _shootout_split(self, home, away, neutral, context) -> float:
        """Probability the home side wins a tie that goes to ET/penalties.

        Mostly a coin toss, nudged slightly by Elo, and overridable with a
        historical penalty record via ``context['home_shootout_winrate']``.
        """
        if "home_shootout_winrate" in context:
            return clamp(float(context["home_shootout_winrate"]), 0.05, 0.95)
        exp = self.elo.win_expectation(home, away, neutral)  # 0..1
        return clamp(0.5 + 0.18 * (2 * exp - 1), 0.35, 0.65)

    def _explain(self, home, away, neutral, stage, elo_home, elo_away,
                 lam_home, lam_away, final, context) -> dict:
        """Build the key-factor table and a plain-English summary."""
        home_stats = _recent_stats(self.fe._tail_window(self.matches, self.elo, home))
        away_stats = _recent_stats(self.fe._tail_window(self.matches, self.elo, away))

        def favours(home_val, away_val, higher_is_better=True):
            if abs(home_val - away_val) < 1e-9:
                return "Even"
            home_better = home_val > away_val
            if not higher_is_better:
                home_better = not home_better
            return home if home_better else away

        ppg_fav = favours(home_stats["ppg"], away_stats["ppg"])
        atk_fav_v = favours(home_stats["attack"], away_stats["attack"])
        def_fav_v = favours(home_stats["defence"], away_stats["defence"], higher_is_better=False)
        factors = [
            {"factor": "Elo rating", "home": round(elo_home, 0), "away": round(elo_away, 0),
             "favours": favours(elo_home, elo_away)},
            {"factor": "Recent form (pts/game)", "home": round(home_stats["ppg"], 2),
             "away": round(away_stats["ppg"], 2), "favours": ppg_fav},
            {"factor": "Attack (goals/game)", "home": round(home_stats["attack"], 2),
             "away": round(away_stats["attack"], 2), "favours": atk_fav_v},
            {"factor": "Defence (conceded/game)", "home": round(home_stats["defence"], 2),
             "away": round(away_stats["defence"], 2), "favours": def_fav_v},
            {"factor": "Expected goals", "home": round(lam_home, 2), "away": round(lam_away, 2),
             "favours": favours(lam_home, lam_away)},
        ]

        # --- narrative sentences ----------------------------------------
        bits = []
        form_fav = favours(home_stats["ppg"], away_stats["ppg"])
        if form_fav != "Even":
            bits.append(f"{form_fav} have the better recent form")
        atk_fav = favours(home_stats["attack"], away_stats["attack"])
        if atk_fav != "Even":
            bits.append(f"{atk_fav} have the stronger attack")
        def_fav = favours(home_stats["defence"], away_stats["defence"], higher_is_better=False)
        if def_fav != "Even":
            bits.append(f"{def_fav} have the more solid defence")

        if neutral:
            bits.append("the venue is neutral, so no home advantage is applied")
        else:
            bits.append(f"{home} get a home-venue boost")

        if context.get("home_injuries"):
            bits.append(f"{home} are weakened by {int(context['home_injuries'])} key absentee(s)")
        if context.get("away_injuries"):
            bits.append(f"{away} are weakened by {int(context['away_injuries'])} key absentee(s)")

        spread = max(final) - min(final)
        shape = ("the odds are fairly one-sided" if spread > 0.45
                 else "the odds are close" if spread < 0.18 else "there is a moderate favourite")
        summary = (
            f"{home} vs {away} ({stage}): " + "; ".join(bits) + f". Overall, {shape}."
        )
        return {"summary": summary, "factors": factors}

    # -------------------------------------------------- tournament sim
    @lru_cache(maxsize=4096)  # noqa: B019  (cache is cleared in save()/simulate)
    def _pair_advance_prob(self, home: str, away: str) -> float:
        """Fast cached P(home advances) for a neutral knockout tie.

        Uses only the Poisson + Elo heads (no ML) so thousands of simulations
        run quickly.
        """
        elo_diff = self.elo.get_rating(home) - self.elo.get_rating(away)
        lam_h, lam_a = self.poisson.expected_goals(home, away, True, elo_diff)
        grid = self.poisson.scoreline_grid(lam_h, lam_a)
        p_h, p_d, p_a = self.poisson.result_probabilities(grid)
        e_h, e_d, e_a = self.elo.match_probabilities(home, away, True)
        # Blend the two heads (renormalised).
        w = self.blend_weights
        tot = w["poisson"] + w["elo"]
        win = (w["poisson"] * p_h + w["elo"] * e_h) / tot
        drw = (w["poisson"] * p_d + w["elo"] * e_d) / tot
        shootout = self._shootout_split(home, away, True, {})
        return clamp(win + drw * shootout, 0.0, 1.0)

    def simulate_tournament(self, bracket: list[str], n_sims: int = 5000,
                            seed: int = 0) -> dict:
        """Monte-Carlo a single-elimination bracket and return title odds.

        ``bracket`` is an ordered list of teams whose length is a power of two
        (4, 8, 16...). Adjacent pairs meet in round one.
        """
        size = len(bracket)
        if size < 2 or (size & (size - 1)) != 0:
            raise ValueError("Bracket size must be a power of two (2, 4, 8, 16...).")
        self._ensure_fitted()
        self._pair_advance_prob.cache_clear()

        rng = np.random.default_rng(seed)
        champions = {team: 0 for team in bracket}
        for _ in range(n_sims):
            alive = list(bracket)
            while len(alive) > 1:
                nxt = []
                for i in range(0, len(alive), 2):
                    a, b = alive[i], alive[i + 1]
                    p = self._pair_advance_prob(a, b)
                    nxt.append(a if rng.random() < p else b)
                alive = nxt
            champions[alive[0]] += 1

        return {team: champions[team] / n_sims for team in bracket}
