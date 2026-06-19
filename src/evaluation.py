"""Backtesting and evaluation for the prediction models.

This is what turns the project from "a model" into "a *measured* model". It
runs a **leak-free walk-forward backtest**: models are trained on the past, then
asked to predict each future match using only information available before kick-off,
and scored with proper metrics:

    * **accuracy**  – share of matches whose most likely outcome was correct
    * **log loss**  – rewards well-calibrated probabilities (lower is better)
    * **Brier score** – mean squared error of the probability vector (lower better)
    * **calibration** – do "70% home win" predictions actually happen ~70% of the time?

It compares four models head-to-head: Elo-only, Poisson-only, ML-only and the
Ensemble. Nothing here claims the model is good — it simply *measures* it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from feature_engineering import FeatureEngineer, _tournament_importance
from model import EloRatingSystem, PoissonGoalModel, ResultClassifier
from utils import blend_distributions

# Fixed outcome ordering used throughout evaluation.
OUTCOMES = ["H", "D", "A"]
OUTCOME_INDEX = {o: i for i, o in enumerate(OUTCOMES)}


# =========================================================================== #
# Metric functions (operate on a (n, 3) probability matrix + actual indices)
# =========================================================================== #
def _as_arrays(probs: np.ndarray, actual_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probs, dtype=float)
    actual_idx = np.asarray(actual_idx, dtype=int)
    return probs, actual_idx


def accuracy_score(probs: np.ndarray, actual_idx: np.ndarray) -> float:
    """Share of matches where argmax(prob) equals the actual outcome."""
    probs, actual_idx = _as_arrays(probs, actual_idx)
    if len(probs) == 0:
        return float("nan")
    return float(np.mean(np.argmax(probs, axis=1) == actual_idx))


def log_loss_score(probs: np.ndarray, actual_idx: np.ndarray, eps: float = 1e-15) -> float:
    """Multiclass log loss (cross-entropy). Lower is better."""
    probs, actual_idx = _as_arrays(probs, actual_idx)
    if len(probs) == 0:
        return float("nan")
    p_actual = probs[np.arange(len(probs)), actual_idx]
    p_actual = np.clip(p_actual, eps, 1.0)
    return float(-np.mean(np.log(p_actual)))


def brier_score(probs: np.ndarray, actual_idx: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error vs the one-hot outcome."""
    probs, actual_idx = _as_arrays(probs, actual_idx)
    if len(probs) == 0:
        return float("nan")
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(probs)), actual_idx] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def calibration_curve_data(
    pred_prob: np.ndarray, actual: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Reliability-curve data for a single binary event (e.g. "home win").

    Returns one row per non-empty bin with the mean predicted probability, the
    observed frequency and the count.
    """
    pred_prob = np.asarray(pred_prob, dtype=float)
    actual = np.asarray(actual, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(pred_prob, bins) - 1, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append({
            "bin_lower": bins[b],
            "bin_upper": bins[b + 1],
            "mean_predicted": float(pred_prob[mask].mean()),
            "observed_frequency": float(actual[mask].mean()),
            "count": int(mask.sum()),
        })
    return pd.DataFrame(rows)


# =========================================================================== #
# Walk-forward backtester
# =========================================================================== #
class Backtester:
    """Train-on-past / test-on-future evaluation of every model."""

    def __init__(
        self,
        matches: pd.DataFrame,
        blend_weights: dict[str, float] | None = None,
        form_window: int = config.FORM_WINDOW,
    ):
        # Chronological order is essential for a fair walk-forward.
        if "date" in matches.columns:
            matches = matches.sort_values("date", kind="stable")
        self.matches = matches.reset_index(drop=True)
        self.blend_weights = dict(blend_weights or config.BLEND_WEIGHTS)
        self.fe = FeatureEngineer(form_window=form_window)

    def run(
        self,
        test_fraction: float = config.BACKTEST_TEST_FRACTION,
        tournament: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        models: list[str] | None = None,
    ) -> dict:
        """Run the backtest and return metrics, per-match predictions and calibration.

        Parameters
        ----------
        test_fraction : fraction of the most recent matches used as the test set.
        tournament : if given, only evaluate matches whose tournament contains this
            (case-insensitive substring). State is still updated on every match.
        start_date, end_date : optional ISO dates limiting which test matches are
            *scored* (state still updates on all of them).
        models : subset of ["elo", "poisson", "ml", "ensemble"] to compare.
        """
        models = models or config.BACKTEST_MODELS
        df = self.matches
        n = len(df)
        split_idx = max(int(n * (1 - test_fraction)), 1)
        train = df.iloc[:split_idx]
        if split_idx >= n:
            raise ValueError("test_fraction leaves no test matches.")

        # --- fit on the training portion only ----------------------------
        elo = EloRatingSystem().fit(train)
        poisson = PoissonGoalModel().fit(train)
        classifier = ResultClassifier()
        ml_ok = False
        X, y = self.fe.build_training_table(train, elo)
        if y.nunique() >= 2 and len(X) >= 10:
            classifier.fit(X, y)
            ml_ok = True

        has_dates = "date" in df.columns
        start = pd.to_datetime(start_date) if start_date else None
        end = pd.to_datetime(end_date) if end_date else None

        records: list[dict] = []
        # --- walk forward through the test matches -----------------------
        for pos in range(split_idx, n):
            row = df.iloc[pos]
            home, away = row["home_team"], row["away_team"]
            neutral = bool(row.get("neutral", False))
            actual = row["result"]

            scored = self._passes_filters(row, tournament, start, end, has_dates)
            if scored:
                history = df.iloc[:pos]   # everything strictly before this match
                preds = self._model_predictions(
                    home, away, neutral, row.get("tournament", ""),
                    elo, poisson, classifier, ml_ok, history, models,
                )
                rec = {
                    "date": row.get("date", pd.NaT),
                    "home_team": home, "away_team": away,
                    "tournament": row.get("tournament", ""),
                    "actual": actual,
                }
                for m, p in preds.items():
                    rec[f"{m}_H"], rec[f"{m}_D"], rec[f"{m}_A"] = p
                records.append(rec)

            # Always update Elo online so ratings stay current for later matches.
            elo.update_match(home, away, row["home_score"], row["away_score"],
                             neutral=neutral, tournament=row.get("tournament", ""))

        if not records:
            raise ValueError("No test matches matched the given filters.")

        predictions = pd.DataFrame(records)
        metrics = self._compute_metrics(predictions, models)
        calibration = self._calibration(predictions, models)
        return {
            "metrics": metrics,
            "comparison_table": self._comparison_table(metrics),
            "predictions": predictions,
            "calibration": calibration,
            "n_test": len(predictions),
            "n_train": split_idx,
        }

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _passes_filters(row, tournament, start, end, has_dates) -> bool:
        if tournament and tournament.lower() not in str(row.get("tournament", "")).lower():
            return False
        if has_dates and (start is not None or end is not None):
            d = row.get("date", pd.NaT)
            if pd.isna(d):
                return False
            if start is not None and d < start:
                return False
            if end is not None and d > end:
                return False
        return True

    def _model_predictions(self, home, away, neutral, tournament, elo, poisson,
                           classifier, ml_ok, history, models) -> dict:
        """Return {model_name: [P(H), P(D), P(A)]} for the requested models."""
        elo_probs = elo.match_probabilities(home, away, neutral)
        elo_diff = elo.get_rating(home) - elo.get_rating(away)
        lam_h, lam_a = poisson.expected_goals(home, away, neutral, elo_diff)
        poisson_probs = poisson.result_probabilities(poisson.scoreline_grid(lam_h, lam_a))

        ml_probs = None
        if ml_ok and len(history) > 0:
            feats = self.fe.latest_features(history, elo, home, away, neutral, "Group")
            # Match the stage-importance encoding used during training.
            feats["stage_importance"] = _tournament_importance(tournament)
            proba = classifier.predict_proba(feats)
            ml_probs = [proba["H"], proba["D"], proba["A"]]

        out = {}
        if "elo" in models:
            out["elo"] = elo_probs
        if "poisson" in models:
            out["poisson"] = poisson_probs
        if "ml" in models and ml_probs is not None:
            out["ml"] = ml_probs
        if "ensemble" in models:
            comps = {"elo": elo_probs, "poisson": poisson_probs}
            if ml_probs is not None:
                comps["ml"] = ml_probs
            out["ensemble"] = blend_distributions(comps, self.blend_weights)
        return out

    @staticmethod
    def _compute_metrics(predictions: pd.DataFrame, models: list[str]) -> dict:
        actual_idx = predictions["actual"].map(OUTCOME_INDEX).to_numpy()
        metrics = {}
        for m in models:
            cols = [f"{m}_{o}" for o in OUTCOMES]
            if not all(c in predictions.columns for c in cols):
                continue
            probs = predictions[cols].to_numpy()
            metrics[m] = {
                "accuracy": accuracy_score(probs, actual_idx),
                "log_loss": log_loss_score(probs, actual_idx),
                "brier": brier_score(probs, actual_idx),
                "n": int(len(probs)),
            }
        return metrics

    @staticmethod
    def _comparison_table(metrics: dict) -> pd.DataFrame:
        rows = [{"model": m, **vals} for m, vals in metrics.items()]
        table = pd.DataFrame(rows)
        if not table.empty:
            table = table.sort_values("log_loss").reset_index(drop=True)
        return table

    @staticmethod
    def _calibration(predictions: pd.DataFrame, models: list[str], n_bins: int = 10) -> dict:
        """Calibration data for the P(home win) prediction of each model."""
        home_win = (predictions["actual"] == "H").to_numpy(dtype=float)
        out = {}
        for m in models:
            col = f"{m}_H"
            if col in predictions.columns:
                out[m] = calibration_curve_data(predictions[col].to_numpy(), home_win, n_bins)
        return out
