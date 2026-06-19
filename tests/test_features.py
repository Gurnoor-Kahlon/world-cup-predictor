"""Tests for data loading and feature engineering."""

import pandas as pd
import pytest

from data_loader import DataValidationError, list_teams, load_matches
from feature_engineering import FEATURE_COLUMNS, FeatureEngineer, _recent_stats
from model import EloRatingSystem


# --------------------------------------------------------------------------- #
# Data loader
# --------------------------------------------------------------------------- #
def test_sample_data_loads(sample_matches):
    assert len(sample_matches) > 0
    assert {"home_team", "away_team", "home_score", "away_score", "result"} <= set(sample_matches.columns)
    assert set(sample_matches["result"].unique()) <= {"H", "D", "A"}


def test_missing_required_column_raises(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"home_team": ["A"], "away_team": ["B"]}).to_csv(bad, index=False)
    with pytest.raises(DataValidationError):
        load_matches(bad)


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_matches("does_not_exist_12345.csv")


def test_list_teams_sorted_unique(sample_matches):
    teams = list_teams(sample_matches)
    assert teams == sorted(teams)
    assert len(teams) == len(set(teams))


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def test_recent_stats_defaults_on_empty():
    from collections import deque
    stats = _recent_stats(deque())
    assert set(stats) == {"ppg", "gd", "attack", "defence", "sos"}


def test_training_table_shape_and_validity(sample_matches):
    elo = EloRatingSystem().fit(sample_matches)
    fe = FeatureEngineer()
    X, y = fe.build_training_table(sample_matches, elo)

    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == len(sample_matches) == len(y)
    assert not X.isna().any().any()          # no missing features
    assert set(y.unique()) <= {"H", "D", "A"}


def test_training_table_requires_fitted_elo(sample_matches):
    unfitted = EloRatingSystem()  # history is empty
    with pytest.raises(ValueError):
        FeatureEngineer().build_training_table(sample_matches, unfitted)


def test_latest_features_has_all_columns(sample_matches):
    elo = EloRatingSystem().fit(sample_matches)
    fe = FeatureEngineer()
    teams = list_teams(sample_matches)
    feats = fe.latest_features(sample_matches, elo, teams[0], teams[1],
                              neutral=True, stage="Final")
    assert set(feats.keys()) == set(FEATURE_COLUMNS)
    assert feats["neutral"] == 1.0


def test_h2h_balance_perspective():
    fe = FeatureEngineer()
    meetings = [
        {"home": "A", "away": "B", "home_score": 2, "away_score": 0},  # A win
        {"home": "B", "away": "A", "home_score": 1, "away_score": 1},  # draw
    ]
    # A won once, drew once -> balance = (1 - 0) / 2 = 0.5 from A's view.
    assert fe._h2h_balance(meetings, "A") == pytest.approx(0.5)
    # Mirror image from B's perspective.
    assert fe._h2h_balance(meetings, "B") == pytest.approx(-0.5)
