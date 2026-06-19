"""Central configuration: file paths and tunable model constants.

Keeping every "magic number" here makes the project easy to understand and tweak.
Change a value in this file and the whole pipeline (model, predictor, app) follows.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Project root = the folder that contains src/, data/, app/ ...
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Default data files. Point MATCHES_PATH at your own CSV to use real data.
MATCHES_PATH = DATA_DIR / "sample_matches.csv"
TEAMS_PATH = DATA_DIR / "teams.csv"

# Columns every match dataset must provide.
REQUIRED_COLUMNS = ["home_team", "away_team", "home_score", "away_score"]

# --------------------------------------------------------------------------- #
# Elo rating system
# --------------------------------------------------------------------------- #
ELO_INITIAL = 1500.0        # every team starts here
ELO_K = 32.0                # base learning rate (how fast ratings move)
ELO_HOME_ADVANTAGE = 65.0   # rating points added to the home side (0 at neutral)

# Competitions matter by different amounts; World Cup games move ratings more.
# Matched against a lower-cased substring of the "tournament" column.
ELO_TOURNAMENT_WEIGHTS = {
    "world cup": 1.6,
    "qualification": 1.1,
    "uefa euro": 1.4,
    "copa": 1.4,
    "nations league": 1.2,
    "friendly": 0.8,
}
ELO_DEFAULT_TOURNAMENT_WEIGHT = 1.0

# --------------------------------------------------------------------------- #
# Poisson goal model
# --------------------------------------------------------------------------- #
MAX_GOALS = 8               # size of the scoreline grid (0..MAX_GOALS each side)
POISSON_HOME_ADVANTAGE = 1.25   # multiplier on home expected goals (1.0 at neutral)
# How strongly the Elo gap nudges expected goals (per 100 Elo points).
ELO_TO_GOALS_SCALE = 0.10
MIN_EXPECTED_GOALS = 0.15   # floor so lambdas never hit zero

# --------------------------------------------------------------------------- #
# Form / feature engineering
# --------------------------------------------------------------------------- #
FORM_WINDOW = 6             # matches in the "recent form" window
MIN_MATCHES_FOR_CONFIDENCE = 8   # below this, confidence is penalised

# --------------------------------------------------------------------------- #
# Ensemble blend weights (Win/Draw/Loss probabilities)
# --------------------------------------------------------------------------- #
# These are combined then re-normalised, so they need not sum to 1.
BLEND_WEIGHTS = {
    "poisson": 0.45,    # statistical attack/defence + Elo model
    "ml": 0.45,         # machine-learning classifier
    "elo": 0.10,        # pure-Elo heuristic head (sanity anchor)
}

# --------------------------------------------------------------------------- #
# Tournament stages
# --------------------------------------------------------------------------- #
GROUP_STAGE = "Group"
KNOCKOUT_STAGES = ["Round of 32", "Round of 16", "Quarter-final", "Semi-final", "Final"]
TOURNAMENT_STAGES = [GROUP_STAGE] + KNOCKOUT_STAGES

# Importance weight per stage (used as a feature and to scale rating impact).
STAGE_IMPORTANCE = {
    "Group": 1.0,
    "Round of 32": 1.1,
    "Round of 16": 1.2,
    "Quarter-final": 1.35,
    "Semi-final": 1.5,
    "Final": 1.6,
}

# --------------------------------------------------------------------------- #
# Odds
# --------------------------------------------------------------------------- #
# Bookmaker margin ("overround") applied when showing market-style odds.
DEFAULT_BOOKMAKER_MARGIN = 0.05
