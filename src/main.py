"""Command-line entry point: train the models and print a prediction.

Examples
--------
    python src/main.py                                   # default sample fixture
    python src/main.py --home Brazil --away Germany --stage final
    python src/main.py --list-teams
    python src/main.py --simulate Brazil France Argentina Spain
"""

from __future__ import annotations

import argparse

import config
from odds_converter import odds_table
from predictor import MatchPredictor


def _print_prediction(result: dict) -> None:
    p = result["probabilities"]
    home, away = result["home"], result["away"]
    venue = "neutral venue" if result["neutral"] else f"{home} at home"

    print("=" * 60)
    print(f"{home} vs {away} - {result['stage']} ({venue})")
    print("=" * 60)

    print("\nResult probabilities")
    print(f"  {home} win .... {p['home_win'] * 100:5.1f}%")
    print(f"  Draw ......... {p['draw'] * 100:5.1f}%")
    print(f"  {away} win .... {p['away_win'] * 100:5.1f}%")

    eg = result["expected_goals"]
    print(f"\nExpected goals:  {home} {eg['home']}  -  {eg['away']} {away}")

    print("\nFair odds (decimal | fractional | American)")
    rows = odds_table(
        [p["home_win"], p["draw"], p["away_win"]],
        [f"{home} win", "Draw", f"{away} win"],
        margin=config.DEFAULT_BOOKMAKER_MARGIN,
    )
    for r in rows:
        print(f"  {r['outcome']:>18}: {r['decimal']:>6}  {r['fractional']:>6}  {r['american']:>5}")

    print("\nMost likely scorelines")
    for s in result["top_scorelines"][:3]:
        print(f"  {s['score']} ...... {s['prob'] * 100:4.1f}%")

    c = result["confidence"]
    print(f"\nConfidence: {c['label']} ({c['score']}/100) - {', '.join(c['reasons'])}")

    if "knockout" in result:
        k = result["knockout"]
        print(f"\nTo advance:  {home} {k['home_advance'] * 100:.1f}%  |  "
              f"{away} {k['away_advance'] * 100:.1f}%")

    print(f"\nWhy: {result['explanation']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="World Cup Match Predictor (CLI).")
    parser.add_argument("--home", default="Brazil", help="home / team A")
    parser.add_argument("--away", default="Germany", help="away / team B")
    parser.add_argument("--stage", default=config.GROUP_STAGE,
                        choices=config.TOURNAMENT_STAGES, help="tournament stage")
    parser.add_argument("--neutral", action="store_true", help="neutral venue")
    parser.add_argument("--list-teams", action="store_true", help="print available teams and exit")
    parser.add_argument("--simulate", nargs="+", metavar="TEAM",
                        help="simulate a knockout bracket (power-of-two teams)")
    args = parser.parse_args()

    print("Loading data and training models...")
    predictor = MatchPredictor().fit()
    print(f"Trained on {len(predictor.matches)} matches "
          f"(ML backend: {predictor.classifier.backend}).\n")

    if args.list_teams:
        print("Available teams:")
        for team in predictor.teams:
            print(f"  - {team}")
        return

    if args.simulate:
        print(f"Simulating bracket: {', '.join(args.simulate)}")
        odds = predictor.simulate_tournament(args.simulate, n_sims=5000)
        print("\nTitle odds:")
        for team, prob in sorted(odds.items(), key=lambda kv: kv[1], reverse=True):
            print(f"  {team:<14} {prob * 100:5.1f}%")
        return

    for team in (args.home, args.away):
        if team not in predictor.teams:
            print(f"WARNING: '{team}' is not in the dataset; using default ratings.")

    result = predictor.predict(args.home, args.away, stage=args.stage, neutral=args.neutral)
    print()
    _print_prediction(result)


if __name__ == "__main__":
    main()
