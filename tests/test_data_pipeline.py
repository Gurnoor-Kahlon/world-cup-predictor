"""Tests for the data pipeline: validation, path resolution and prepare()."""

import sys
from pathlib import Path

import pandas as pd
import pytest

import config
from data_loader import (
    DataValidationError,
    load_matches,
    resolve_matches_path,
    validate_matches,
)

# Make the standalone prepare script importable.
sys.path.insert(0, str(Path(config.PROJECT_ROOT) / "scripts"))
import prepare_data  # noqa: E402


def test_resolve_prefers_sample_when_no_processed():
    # In a clean checkout there is no processed/matches.csv, so the sample wins.
    if not config.PROCESSED_MATCHES_PATH.exists():
        assert resolve_matches_path() == config.SAMPLE_MATCHES_PATH


def test_validate_matches_raises_on_missing_columns():
    bad = pd.DataFrame({"home_team": ["A"], "away_team": ["B"]})
    with pytest.raises(DataValidationError):
        validate_matches(bad, source="bad")


def test_rename_to_canonical_handles_aliases():
    raw = pd.DataFrame({
        "Date": ["2020-01-01"], "HomeTeam": ["A"], "AwayTeam": ["B"],
        "FTHG": [2], "FTAG": [1], "Competition": ["Cup"],
    })
    renamed = prepare_data._rename_to_canonical(raw)
    for col in ("date", "home_team", "away_team", "home_score", "away_score", "tournament"):
        assert col in renamed.columns


def test_prepare_writes_loadable_processed_file(tmp_path):
    raw = pd.DataFrame({
        "date": ["2020-01-01", "2020-02-01", "bad-date"],
        "home_team": ["A", "B", "A"],
        "away_team": ["B", "A", "B"],
        "home_score": [2, 1, None],   # last row is unusable and should be dropped
        "away_score": [0, 1, 3],
        "tournament": ["Friendly", "Friendly", "Friendly"],
    })
    raw_path = tmp_path / "raw.csv"
    out_path = tmp_path / "processed.csv"
    raw.to_csv(raw_path, index=False)

    df = prepare_data.prepare(raw_path, out_path)
    assert out_path.exists()
    assert len(df) == 2   # the row with a missing score was dropped

    # The processed file loads cleanly through the normal loader.
    loaded = load_matches(out_path)
    assert {"home_team", "away_team", "result"} <= set(loaded.columns)


def test_prepare_missing_input_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare_data.prepare(tmp_path / "nope.csv", tmp_path / "out.csv")
