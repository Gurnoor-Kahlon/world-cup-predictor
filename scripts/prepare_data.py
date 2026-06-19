"""Validate and convert raw data into the project's processed format.

Reads a raw results CSV (default ``data/raw/results.csv``, e.g. the file fetched
by ``scripts/download_data.py``) and writes a clean, validated dataset to
``data/processed/matches.csv``. Once that file exists, the whole project (app,
CLI, backtest) automatically uses it instead of the bundled sample data.

The raw file may use slightly different column names; the mapping below covers
the common international-results schema and is easy to extend.

Usage
-----
    python scripts/prepare_data.py
    python scripts/prepare_data.py --input data/raw/my_results.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
from data_loader import DataValidationError, validate_matches  # noqa: E402

# Map raw column names -> the project's canonical names. Extend as needed.
COLUMN_ALIASES = {
    "home_team": ["home_team", "home", "hometeam", "team_home"],
    "away_team": ["away_team", "away", "awayteam", "team_away"],
    "home_score": ["home_score", "home_goals", "fthg", "score_home"],
    "away_score": ["away_score", "away_goals", "ftag", "score_away"],
    "date": ["date", "match_date", "datetime"],
    "tournament": ["tournament", "competition", "comp"],
    "neutral": ["neutral", "neutral_venue", "is_neutral"],
    "city": ["city"],
    "country": ["country"],
}

KEEP_COLUMNS = ["date", "home_team", "away_team", "home_score", "away_score",
                "tournament", "neutral", "city", "country"]


def _rename_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Rename recognised columns to the canonical schema (case-insensitive)."""
    lower = {c.lower(): c for c in df.columns}
    rename = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lower:
                rename[lower[alias]] = canonical
                break
    return df.rename(columns=rename)


def prepare(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Clean + validate raw matches and write the processed CSV."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at '{input_path}'.\n"
            f"Download it first:  python scripts/download_data.py"
        )

    raw = pd.read_csv(input_path)
    df = _rename_to_canonical(raw)
    validate_matches(df, source=input_path.name)

    # Coerce scores, drop rows we cannot use.
    for col in ("home_score", "away_score"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["home_team", "away_team", "home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date", kind="stable")

    # Keep only known columns that exist.
    cols = [c for c in KEEP_COLUMNS if c in df.columns]
    df = df[cols].reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    dropped = before - len(df)
    teams = pd.unique(pd.concat([df["home_team"], df["away_team"]]))
    print(f"Prepared {len(df):,} matches ({dropped} unusable rows dropped).")
    if "date" in df.columns and df["date"].notna().any():
        print(f"Date range : {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"Teams      : {len(teams)}")
    print(f"Written to : {output_path}")
    print("\nThe project will now use this dataset automatically.")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare raw data into processed/matches.csv.")
    parser.add_argument("--input", default=str(ROOT / "data" / "raw" / "results.csv"),
                        help="path to the raw results CSV")
    parser.add_argument("--output", default=str(config.PROCESSED_MATCHES_PATH),
                        help="where to write the processed matches CSV")
    args = parser.parse_args()

    try:
        prepare(Path(args.input), Path(args.output))
    except (FileNotFoundError, DataValidationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
