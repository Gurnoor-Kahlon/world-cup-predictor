"""Shared pytest fixtures.

``src/`` is on the path via ``pyproject.toml`` ([tool.pytest.ini_options]
pythonpath = ["src"]), so tests can import the modules directly.
"""

from __future__ import annotations

import pandas as pd
import pytest

from data_loader import load_matches
from predictor import MatchPredictor


@pytest.fixture(scope="session")
def sample_matches() -> pd.DataFrame:
    """The bundled sample dataset, loaded once per test session."""
    return load_matches()


@pytest.fixture(scope="session")
def fitted_predictor(sample_matches) -> MatchPredictor:
    """A MatchPredictor trained on the sample data (trained once, reused)."""
    return MatchPredictor(matches=sample_matches).fit()


@pytest.fixture(scope="session")
def synthetic_matches() -> pd.DataFrame:
    """A tiny, deterministic dataset where 'Strong' clearly beats 'Weak'.

    Useful for asserting that the model ranks an obviously better team higher.
    """
    rows = []
    day = pd.Timestamp("2020-01-01")
    for i in range(40):
        # Strong beats Weak comfortably, and beats Mid narrowly.
        rows.append(("Strong", "Weak", 3, 0))
        rows.append(("Strong", "Mid", 2, 1))
        rows.append(("Mid", "Weak", 2, 0))
        rows.append(("Weak", "Mid", 0, 1))
    matches = pd.DataFrame(rows, columns=["home_team", "away_team", "home_score", "away_score"])
    matches.insert(0, "date", [day + pd.Timedelta(days=d) for d in range(len(matches))])
    matches["tournament"] = "Friendly"
    matches["neutral"] = False
    # Re-run the loader's normalisation (adds the 'result' column, etc.).
    from utils import result_label
    matches["result"] = [result_label(h, a) for h, a in
                         zip(matches["home_score"], matches["away_score"])]
    return matches


@pytest.fixture(scope="session")
def synthetic_predictor(synthetic_matches) -> MatchPredictor:
    return MatchPredictor(matches=synthetic_matches).fit()
