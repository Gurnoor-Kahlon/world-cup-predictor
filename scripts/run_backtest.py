"""Run a walk-forward backtest and write the results to data/processed/.

Examples
--------
    python scripts/run_backtest.py
    python scripts/run_backtest.py --test-fraction 0.3
    python scripts/run_backtest.py --tournament "World Cup"
    python scripts/run_backtest.py --start-date 2018-01-01 --end-date 2022-12-31
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src/ importable when run as a standalone script.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
from data_loader import load_matches  # noqa: E402
from evaluation import Backtester  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the prediction models.")
    parser.add_argument("--test-fraction", type=float, default=config.BACKTEST_TEST_FRACTION,
                        help="fraction of most-recent matches used for testing")
    parser.add_argument("--tournament", default=None,
                        help="only score matches whose tournament contains this text")
    parser.add_argument("--start-date", default=None, help="ISO date lower bound for test matches")
    parser.add_argument("--end-date", default=None, help="ISO date upper bound for test matches")
    parser.add_argument("--output", default=str(config.BACKTEST_RESULTS_PATH),
                        help="where to write the per-match predictions CSV")
    args = parser.parse_args()

    print("Loading data...")
    matches = load_matches()
    print(f"Loaded {len(matches)} matches. Running walk-forward backtest "
          f"(test_fraction={args.test_fraction})...\n")

    bt = Backtester(matches)
    result = bt.run(
        test_fraction=args.test_fraction,
        tournament=args.tournament,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    table = result["comparison_table"]
    print(f"Trained on {result['n_train']} matches, scored on {result['n_test']} test matches.\n")
    print("Model comparison (sorted by log loss, lower = better):")
    print("-" * 60)
    print(f"{'model':<10}{'accuracy':>10}{'log_loss':>12}{'brier':>10}")
    for _, r in table.iterrows():
        print(f"{r['model']:<10}{r['accuracy']:>10.3f}{r['log_loss']:>12.4f}{r['brier']:>10.4f}")
    print("-" * 60)

    # Write outputs.
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result["predictions"].to_csv(out_path, index=False)
    table.to_csv(out_path.with_name("backtest_comparison.csv"), index=False)
    print(f"\nWrote per-match predictions -> {out_path}")
    print(f"Wrote comparison table     -> {out_path.with_name('backtest_comparison.csv')}")

    # Helpful, honest interpretation.
    best = table.iloc[0]["model"]
    print(f"\nBest log loss: '{best}'. Remember: these numbers describe the *sample* "
          "data. Re-run on real data (see docs/data_sources.md) for meaningful results.")


if __name__ == "__main__":
    main()
