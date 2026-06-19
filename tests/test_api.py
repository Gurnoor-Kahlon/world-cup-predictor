"""Tests for the optional FastAPI backend.

Skipped automatically if FastAPI is not installed (it is an optional dependency).
"""

import sys
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

# Import the app (api/ is a package at the project root).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from api.main import app  # noqa: E402

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["n_teams"] > 0


def test_teams():
    r = client.get("/teams")
    assert r.status_code == 200
    assert "Brazil" in r.json()["teams"]


def test_predict_ok():
    r = client.get("/predict", params={"home_team": "Brazil", "away_team": "Germany",
                                       "stage": "Final", "neutral": True})
    assert r.status_code == 200
    probs = r.json()["probabilities"]
    assert abs(sum(probs.values()) - 1.0) < 1e-6


def test_predict_unknown_team_404():
    r = client.get("/predict", params={"home_team": "Atlantis", "away_team": "Germany"})
    assert r.status_code == 404


def test_predict_same_team_400():
    r = client.get("/predict", params={"home_team": "Brazil", "away_team": "Brazil"})
    assert r.status_code == 400


def test_simulate_tournament():
    r = client.get("/simulate-tournament",
                   params={"teams": "Brazil,France,Argentina,Spain", "n_sims": 500})
    assert r.status_code == 200
    odds = r.json()["title_odds"]
    assert abs(sum(odds.values()) - 1.0) < 1e-6
