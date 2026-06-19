"""Load and validate historical match data.

The rest of the project depends on a clean, predictable DataFrame, so all the
messy "is this CSV actually usable?" logic lives here. Swap in real data by
pointing ``config.MATCHES_PATH`` at your own file with the same columns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

import config
from utils import result_label


class DataValidationError(Exception):
    """Raised when an input dataset is missing required columns or is empty."""


def load_matches(path: Optional[Path | str] = None) -> pd.DataFrame:
    """Load a match-history CSV into a validated, sorted DataFrame.

    Adds convenience columns used everywhere downstream:
        * ``date``    – parsed to datetime (if a date column exists)
        * ``neutral`` – boolean (defaults to False if absent)
        * ``result``  – 'H' / 'D' / 'A'

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist.
    DataValidationError
        If required columns are missing or the file has no rows.
    """
    path = Path(path) if path is not None else config.MATCHES_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Match data not found at '{path}'. Generate the sample data with "
            f"`python data/generate_sample_data.py` or set config.MATCHES_PATH."
        )

    df = pd.read_csv(path)

    missing = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"Dataset '{path.name}' is missing required columns: {missing}. "
            f"Required: {config.REQUIRED_COLUMNS}"
        )
    if df.empty:
        raise DataValidationError(f"Dataset '{path.name}' contains no rows.")

    # Parse dates if present, otherwise keep insertion order.
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date", kind="stable").reset_index(drop=True)

    # Coerce scores to integers, dropping rows we cannot interpret.
    for col in ("home_score", "away_score"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    bad = df[["home_score", "away_score"]].isna().any(axis=1)
    if bad.any():
        df = df[~bad].reset_index(drop=True)
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # Normalise optional columns.
    if "neutral" not in df.columns:
        df["neutral"] = False
    df["neutral"] = _coerce_bool(df["neutral"])
    if "tournament" not in df.columns:
        df["tournament"] = "Unknown"
    df["tournament"] = df["tournament"].fillna("Unknown").astype(str)

    # Strip stray whitespace from team names.
    df["home_team"] = df["home_team"].astype(str).str.strip()
    df["away_team"] = df["away_team"].astype(str).str.strip()

    # Derived result column.
    df["result"] = [
        result_label(h, a) for h, a in zip(df["home_score"], df["away_score"])
    ]
    return df


def _coerce_bool(series: pd.Series) -> pd.Series:
    """Best-effort conversion of a column to booleans."""
    if series.dtype == bool:
        return series
    truthy = {"true", "1", "yes", "y", "t"}
    return (
        series.astype(str).str.strip().str.lower().isin(truthy)
    )


def list_teams(matches: pd.DataFrame) -> list[str]:
    """Return the sorted list of unique team names appearing in the data."""
    teams = pd.unique(
        pd.concat([matches["home_team"], matches["away_team"]], ignore_index=True)
    )
    return sorted(str(t) for t in teams)


def load_team_metadata(path: Optional[Path | str] = None) -> Optional[pd.DataFrame]:
    """Load optional per-team metadata (rank, market value...).

    Returns ``None`` if the file does not exist — it is purely optional and the
    pipeline works fine without it.
    """
    path = Path(path) if path is not None else config.TEAMS_PATH
    if not path.exists():
        return None
    meta = pd.read_csv(path)
    if "team" in meta.columns:
        meta["team"] = meta["team"].astype(str).str.strip()
    return meta
