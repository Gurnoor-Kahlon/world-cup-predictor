"""Generate the bundled *synthetic* sample dataset.

This script creates a realistic-looking history of international football matches
so the project runs out of the box. It is **synthetic** (clearly not real results),
but it is generated from a principled model rather than pure noise:

    * each team has a hidden "quality" that maps to an attack strength and a
      defence strength (goals it tends to score / concede vs an average team),
    * each match's expected goals follow the standard
      ``attack * opponent_defence / league_average`` model, with a home-venue bump,
    * actual goals are drawn from a Poisson distribution around those expectations.

Because the sample data is produced by the *same family* of model the predictor
later tries to recover, the project behaves sensibly end-to-end.

Run it with:

    python data/generate_sample_data.py

It is deterministic (fixed random seed), so re-running reproduces the same files.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Folder this script lives in (the data/ directory).
DATA_DIR = Path(__file__).resolve().parent

SEED = 42
LEAGUE_AVG_GOALS = 1.35     # average goals a team scores against an average team
HOME_MULTIPLIER = 1.25      # home teams score a bit more (skipped at neutral venues)

# 16 national teams with a hidden "quality" in roughly [-1, 1].
# Higher quality -> stronger attack and stronger defence.
TEAMS = {
    "Brazil":        {"quality": 0.95, "confederation": "CONMEBOL", "fifa_rank": 3},
    "France":        {"quality": 0.92, "confederation": "UEFA",     "fifa_rank": 2},
    "Argentina":     {"quality": 0.90, "confederation": "CONMEBOL", "fifa_rank": 1},
    "Spain":         {"quality": 0.80, "confederation": "UEFA",     "fifa_rank": 8},
    "Germany":       {"quality": 0.78, "confederation": "UEFA",     "fifa_rank": 16},
    "England":       {"quality": 0.75, "confederation": "UEFA",     "fifa_rank": 5},
    "Portugal":      {"quality": 0.72, "confederation": "UEFA",     "fifa_rank": 6},
    "Netherlands":   {"quality": 0.68, "confederation": "UEFA",     "fifa_rank": 7},
    "Belgium":       {"quality": 0.60, "confederation": "UEFA",     "fifa_rank": 4},
    "Croatia":       {"quality": 0.45, "confederation": "UEFA",     "fifa_rank": 10},
    "Uruguay":       {"quality": 0.42, "confederation": "CONMEBOL", "fifa_rank": 11},
    "Mexico":        {"quality": 0.20, "confederation": "CONCACAF", "fifa_rank": 15},
    "Japan":         {"quality": 0.10, "confederation": "AFC",      "fifa_rank": 18},
    "USA":           {"quality": 0.05, "confederation": "CONCACAF", "fifa_rank": 13},
    "Nigeria":       {"quality": -0.05, "confederation": "CAF",     "fifa_rank": 28},
    "South Korea":   {"quality": -0.10, "confederation": "AFC",     "fifa_rank": 23},
}

# Rough squad market values (in €m) — *sample* numbers, only used as optional
# metadata to demonstrate how extra features could be wired in later.
MARKET_VALUE_M = {
    "Brazil": 1100, "France": 1050, "Argentina": 720, "Spain": 900, "Germany": 880,
    "England": 1300, "Portugal": 950, "Netherlands": 800, "Belgium": 650,
    "Croatia": 380, "Uruguay": 360, "Mexico": 230, "Japan": 220, "USA": 290,
    "Nigeria": 200, "South Korea": 170,
}


def team_strengths(quality: float) -> tuple[float, float]:
    """Map a hidden quality score to (attack, defence) expected-goals factors.

    * ``attack``  – goals this team scores against an average opponent.
    * ``defence`` – goals this team concedes against an average opponent
      (lower is better).
    """
    attack = LEAGUE_AVG_GOALS * (1.0 + 0.45 * quality)
    defence = LEAGUE_AVG_GOALS * (1.0 - 0.45 * quality)
    return attack, defence


def expected_goals(home, away, attack, defence, neutral):
    """Expected goals for a fixture using the attack*defence/avg model."""
    home_mult = 1.0 if neutral else HOME_MULTIPLIER
    lam_home = attack[home] * defence[away] / LEAGUE_AVG_GOALS * home_mult
    lam_away = attack[away] * defence[home] / LEAGUE_AVG_GOALS
    return max(lam_home, 0.05), max(lam_away, 0.05)


def main() -> None:
    rng = np.random.default_rng(SEED)
    teams = list(TEAMS)

    # Precompute each team's attack/defence factors (with a touch of fixed noise).
    attack, defence = {}, {}
    for name, info in TEAMS.items():
        a, d = team_strengths(info["quality"])
        attack[name] = a * (1 + 0.05 * rng.standard_normal())
        defence[name] = d * (1 + 0.05 * rng.standard_normal())

    rows = []

    def play(home, away, day, tournament, neutral):
        lam_h, lam_a = expected_goals(home, away, attack, defence, neutral)
        hs = int(rng.poisson(lam_h))
        as_ = int(rng.poisson(lam_a))
        rows.append({
            "date": day.isoformat(),
            "home_team": home,
            "away_team": away,
            "home_score": hs,
            "away_score": as_,
            "tournament": tournament,
            "neutral": neutral,
        })

    # --- Yearly qualifier-style round robin (2011..2024) ---------------------
    for year in range(2011, 2025):
        order = teams.copy()
        rng.shuffle(order)
        match_day = date(year, 3, 1)
        for i in range(len(order)):
            for j in range(i + 1, len(order)):
                home, away = order[i], order[j]
                # alternate who hosts across years for variety
                if year % 2 == 0:
                    home, away = away, home
                play(home, away, match_day, "FIFA World Cup qualification", False)
                match_day += timedelta(days=2)

        # A short friendly window mid-year
        rng.shuffle(order)
        fday = date(year, 9, 5)
        for k in range(0, len(order) - 1, 2):
            play(order[k], order[k + 1], fday, "Friendly", False)
            fday += timedelta(days=3)

    # --- Mock World Cups (2014, 2018, 2022) on neutral ground ----------------
    for wc_year in (2014, 2018, 2022):
        # top-8 ranked teams contest a simple neutral mini-tournament
        contenders = sorted(teams, key=lambda t: TEAMS[t]["fifa_rank"])[:8]
        rng.shuffle(contenders)
        wday = date(wc_year, 6, 14)
        # group stage: round robin within two groups of 4
        for g in (contenders[:4], contenders[4:]):
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    play(g[i], g[j], wday, "FIFA World Cup", True)
                    wday += timedelta(days=1)
        # a few knockout-style neutral matches
        for a, b in [(contenders[0], contenders[5]),
                     (contenders[1], contenders[4]),
                     (contenders[2], contenders[7]),
                     (contenders[3], contenders[6])]:
            play(a, b, wday, "FIFA World Cup", True)
            wday += timedelta(days=2)

    matches = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    matches_path = DATA_DIR / "sample_matches.csv"
    matches.to_csv(matches_path, index=False)

    # --- Team metadata table -------------------------------------------------
    teams_df = pd.DataFrame([
        {
            "team": name,
            "confederation": info["confederation"],
            "fifa_rank": info["fifa_rank"],
            "squad_market_value_m": MARKET_VALUE_M[name],
        }
        for name, info in TEAMS.items()
    ]).sort_values("fifa_rank").reset_index(drop=True)
    teams_path = DATA_DIR / "teams.csv"
    teams_df.to_csv(teams_path, index=False)

    print(f"Wrote {len(matches)} matches -> {matches_path}")
    print(f"Wrote {len(teams_df)} teams   -> {teams_path}")


if __name__ == "__main__":
    main()
