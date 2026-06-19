"""Optional FastAPI backend exposing the predictor over HTTP.

Run it with:

    uvicorn api.main:app --reload --port 8000

Then open the interactive docs at http://localhost:8000/docs

Endpoints
---------
    GET /health
    GET /teams
    GET /predict?home_team=Brazil&away_team=Germany&stage=Final&neutral=true
    GET /simulate-tournament?teams=Brazil,France,Argentina,Spain&n_sims=3000
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query

# Make the src/ modules importable.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import config  # noqa: E402
from predictor import MatchPredictor  # noqa: E402

app = FastAPI(
    title="World Cup Match Predictor API",
    description="Predict international football match probabilities and odds. "
                "Educational project — predictions are estimates, not betting advice.",
    version="1.0.0",
)


@lru_cache(maxsize=1)
def get_predictor() -> MatchPredictor:
    """Train (or reload) the predictor once and cache it for the process."""
    return MatchPredictor().fit()


@app.get("/health")
def health() -> dict:
    """Liveness check plus a little model metadata."""
    p = get_predictor()
    return {
        "status": "ok",
        "n_matches": int(len(p.matches)),
        "n_teams": len(p.teams),
        "ml_backend": p.classifier.backend,
    }


@app.get("/teams")
def teams() -> dict:
    """List the teams the model knows about."""
    return {"teams": get_predictor().teams}


@app.get("/predict")
def predict(
    home_team: str = Query(..., description="home / team A"),
    away_team: str = Query(..., description="away / team B"),
    stage: str = Query(config.GROUP_STAGE, description="tournament stage"),
    neutral: bool = Query(False, description="neutral venue?"),
) -> dict:
    """Predict a single fixture."""
    p = get_predictor()
    if stage not in config.TOURNAMENT_STAGES:
        raise HTTPException(400, f"Unknown stage '{stage}'. "
                                 f"Valid: {config.TOURNAMENT_STAGES}")
    if home_team == away_team:
        raise HTTPException(400, "home_team and away_team must differ.")
    for team in (home_team, away_team):
        if team not in p.teams:
            raise HTTPException(404, f"Unknown team '{team}'. See GET /teams.")
    return p.predict(home_team, away_team, stage=stage, neutral=neutral)


@app.get("/simulate-tournament")
def simulate_tournament(
    teams: str = Query(..., description="comma-separated bracket (power of two)"),
    n_sims: int = Query(3000, ge=100, le=50000),
) -> dict:
    """Monte-Carlo a single-elimination bracket and return title odds."""
    p = get_predictor()
    bracket = [t.strip() for t in teams.split(",") if t.strip()]
    unknown = [t for t in bracket if t not in p.teams]
    if unknown:
        raise HTTPException(404, f"Unknown teams: {unknown}. See GET /teams.")
    try:
        odds = p.simulate_tournament(bracket, n_sims=n_sims)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    ranked = dict(sorted(odds.items(), key=lambda kv: kv[1], reverse=True))
    return {"bracket": bracket, "n_sims": n_sims, "title_odds": ranked}
