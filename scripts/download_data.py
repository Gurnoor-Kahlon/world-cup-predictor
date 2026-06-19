"""Download real, public international-match data into ``data/raw/``.

This script does **not** fabricate data. By default it fetches the well-known,
openly published *International football results* dataset (the same data behind
the popular Kaggle dataset, maintained by martj42 on GitHub):

    https://github.com/martj42/international_results

If you have no internet access, the script prints clear manual instructions
instead of failing silently. After downloading, run ``scripts/prepare_data.py``
to validate and convert it into the project's processed format.

Usage
-----
    python scripts/download_data.py                 # default real results dataset
    python scripts/download_data.py --url <CSV_URL> # any CSV in the same schema
    python scripts/download_data.py --fifa-url <URL>  # optional FIFA ranking CSV
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

# Real, public source (CC0-licensed results compilation).
DEFAULT_RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

MANUAL_INSTRUCTIONS = f"""
Could not download automatically. To add real data manually:

1. Open the dataset page and download the CSV:
     - GitHub : https://github.com/martj42/international_results
     - Kaggle : https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
2. Save it as:
     {RAW_DIR / 'results.csv'}
3. Then run:
     python scripts/prepare_data.py

See docs/data_sources.md for FIFA ranking, Elo and xG sources.
"""


def _download(url: str, dest: Path, timeout: int = 30) -> bool:
    """Download ``url`` to ``dest``. Returns True on success, False on failure."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"Downloading {url} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "world-cup-predictor"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        dest.write_bytes(data)
        size_kb = len(data) / 1024
        print(f"Saved {size_kb:,.0f} KB -> {dest}")
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download real match data into data/raw/.")
    parser.add_argument("--url", default=DEFAULT_RESULTS_URL,
                        help="CSV URL for match results (same schema as the sample data)")
    parser.add_argument("--fifa-url", default=None,
                        help="optional CSV URL for FIFA ranking data")
    parser.add_argument("--elo-url", default=None,
                        help="optional CSV URL for external Elo ratings")
    args = parser.parse_args()

    ok = _download(args.url, RAW_DIR / "results.csv")
    if args.fifa_url:
        _download(args.fifa_url, RAW_DIR / "fifa_ranking.csv")
    if args.elo_url:
        _download(args.elo_url, RAW_DIR / "elo_ratings.csv")

    if ok:
        print("\nDone. Next step:\n    python scripts/prepare_data.py")
    else:
        print(MANUAL_INSTRUCTIONS)
        sys.exit(1)


if __name__ == "__main__":
    main()
