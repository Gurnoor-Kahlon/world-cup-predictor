"""Shared helpers for the Streamlit app: cached resources and charts.

Keeping the heavy lifting here keeps each page in ``streamlit_app.py`` short and
readable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make the src/ modules importable when run via `streamlit run app/...`.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import config  # noqa: E402
from evaluation import Backtester  # noqa: E402
from predictor import MatchPredictor  # noqa: E402

# Palette.
HOME_COLOR = "#1f77b4"
DRAW_COLOR = "#7f7f7f"
AWAY_COLOR = "#d62728"


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading data and training models...")
def get_predictor() -> MatchPredictor:
    """Train the predictor once and cache it across pages and reruns."""
    return MatchPredictor().fit()


@st.cache_data(show_spinner="Running walk-forward backtest...")
def run_backtest(test_fraction: float = config.BACKTEST_TEST_FRACTION) -> dict:
    """Run (and cache) a backtest. Returns plain data so it caches cleanly."""
    predictor = get_predictor()
    bt = Backtester(predictor.matches, blend_weights=predictor.blend_weights)
    result = bt.run(test_fraction=test_fraction)
    return {
        "comparison": result["comparison_table"],
        "calibration": result["calibration"],
        "predictions": result["predictions"],
        "n_train": result["n_train"],
        "n_test": result["n_test"],
    }


# --------------------------------------------------------------------------- #
# Match-selection widget shared by several pages
# --------------------------------------------------------------------------- #
def team_selectors(predictor: MatchPredictor, key: str, with_context: bool = False):
    """Render team/stage pickers in the sidebar and return the selection."""
    teams = predictor.teams
    di_home = teams.index("Brazil") if "Brazil" in teams else 0
    di_away = teams.index("Germany") if "Germany" in teams else min(1, len(teams) - 1)

    home = st.sidebar.selectbox("Team A (home)", teams, index=di_home, key=f"{key}_home")
    away = st.sidebar.selectbox("Team B (away)", teams, index=di_away, key=f"{key}_away")
    stage = st.sidebar.selectbox("Stage", config.TOURNAMENT_STAGES, key=f"{key}_stage")
    neutral = st.sidebar.checkbox("Neutral venue", value=stage != config.GROUP_STAGE,
                                  key=f"{key}_neutral")

    context = {}
    if with_context:
        st.sidebar.divider()
        st.sidebar.caption("Optional real-world context")
        context = {
            "home_injuries": st.sidebar.number_input(f"{home} key players out", 0, 11, 0,
                                                     key=f"{key}_hinj"),
            "away_injuries": st.sidebar.number_input(f"{away} key players out", 0, 11, 0,
                                                     key=f"{key}_ainj"),
            "rest_days_home": st.sidebar.slider(f"{home} rest days", 1, 14, 7, key=f"{key}_hrest"),
            "rest_days_away": st.sidebar.slider(f"{away} rest days", 1, 14, 7, key=f"{key}_arest"),
        }
    return home, away, stage, neutral, context


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def probability_bar(home, away, probs):
    labels = [f"{home} win", "Draw", f"{away} win"]
    values = [probs["home_win"], probs["draw"], probs["away_win"]]
    fig = go.Figure(go.Bar(
        x=[v * 100 for v in values], y=labels, orientation="h",
        marker_color=[HOME_COLOR, DRAW_COLOR, AWAY_COLOR],
        text=[f"{v * 100:.1f}%" for v in values], textposition="auto",
    ))
    fig.update_layout(xaxis_title="Probability (%)", yaxis=dict(autorange="reversed"),
                      height=240, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def strength_radar(predictor, home, away):
    elo_h, elo_a = predictor.elo.get_rating(home), predictor.elo.get_rating(away)
    atk_h = predictor.poisson.attack.get(home, 1.0)
    atk_a = predictor.poisson.attack.get(away, 1.0)
    def_h = 2 - predictor.poisson.defence.get(home, 1.0)
    def_a = 2 - predictor.poisson.defence.get(away, 1.0)
    all_elos = list(predictor.elo.ratings.values()) or [config.ELO_INITIAL]
    lo, hi = min(all_elos), max(all_elos)

    def norm_elo(r):
        return 2 * (r - lo) / (hi - lo) if hi > lo else 1.0

    cats = ["Elo strength", "Attack", "Defence"]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[norm_elo(elo_h), atk_h, def_h], theta=cats,
                                  fill="toself", name=home, line_color=HOME_COLOR))
    fig.add_trace(go.Scatterpolar(r=[norm_elo(elo_a), atk_a, def_a], theta=cats,
                                  fill="toself", name=away, line_color=AWAY_COLOR))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 2])),
                      height=340, margin=dict(l=20, r=20, t=30, b=20))
    return fig


def scoreline_heatmap(predictor, home, away, neutral, max_display=6):
    """Heatmap of P(home i goals, away j goals) from the Poisson grid."""
    elo_diff = predictor.elo.get_rating(home) - predictor.elo.get_rating(away)
    lam_h, lam_a = predictor.poisson.expected_goals(home, away, neutral, elo_diff)
    grid = predictor.poisson.scoreline_grid(lam_h, lam_a)
    g = grid[:max_display + 1, :max_display + 1] * 100
    fig = px.imshow(
        g, x=[str(j) for j in range(max_display + 1)],
        y=[str(i) for i in range(max_display + 1)],
        labels=dict(x=f"{away} goals", y=f"{home} goals", color="Prob %"),
        color_continuous_scale="Blues", text_auto=".1f", aspect="auto",
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis=dict(autorange="reversed"))
    return fig


def comparison_grouped_bar(predictor, home, away):
    """Grouped bar comparing several team metrics from recent form + model."""
    from feature_engineering import _recent_stats
    hs = _recent_stats(predictor.fe._tail_window(predictor.matches, predictor.elo, home))
    as_ = _recent_stats(predictor.fe._tail_window(predictor.matches, predictor.elo, away))
    metrics = {
        "Form (pts/game)": (hs["ppg"], as_["ppg"]),
        "Goals scored/game": (hs["attack"], as_["attack"]),
        "Goals conceded/game": (hs["defence"], as_["defence"]),
        "Attack strength": (predictor.poisson.attack.get(home, 1.0),
                            predictor.poisson.attack.get(away, 1.0)),
    }
    rows = []
    for label, (h, a) in metrics.items():
        rows.append({"metric": label, "team": home, "value": round(h, 2)})
        rows.append({"metric": label, "team": away, "value": round(a, 2)})
    df = pd.DataFrame(rows)
    fig = px.bar(df, x="metric", y="value", color="team", barmode="group",
                 color_discrete_map={home: HOME_COLOR, away: AWAY_COLOR})
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="", legend_title="")
    return fig


def comparison_table_df(predictor, home, away):
    from feature_engineering import _recent_stats
    hs = _recent_stats(predictor.fe._tail_window(predictor.matches, predictor.elo, home))
    as_ = _recent_stats(predictor.fe._tail_window(predictor.matches, predictor.elo, away))
    return pd.DataFrame([
        {"Metric": "Elo rating", home: round(predictor.elo.get_rating(home)),
         away: round(predictor.elo.get_rating(away))},
        {"Metric": "Form (pts/game)", home: round(hs["ppg"], 2), away: round(as_["ppg"], 2)},
        {"Metric": "Goals scored/game", home: round(hs["attack"], 2),
         away: round(as_["attack"], 2)},
        {"Metric": "Goals conceded/game", home: round(hs["defence"], 2),
         away: round(as_["defence"], 2)},
        {"Metric": "Matches in data", home: predictor.team_match_counts.get(home, 0),
         away: predictor.team_match_counts.get(away, 0)},
    ])


def calibration_chart(cal_df: pd.DataFrame, model: str):
    """Reliability diagram for one model's P(home win)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                             line=dict(dash="dash", color=DRAW_COLOR)))
    if not cal_df.empty:
        fig.add_trace(go.Scatter(x=cal_df["mean_predicted"], y=cal_df["observed_frequency"],
                                 mode="lines+markers", name=model, line_color=HOME_COLOR))
    fig.update_layout(xaxis_title="Predicted P(home win)", yaxis_title="Observed frequency",
                      height=360, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_range=[0, 1], yaxis_range=[0, 1])
    return fig


def title_odds_bar(odds: dict):
    df = (pd.DataFrame({"Team": list(odds), "Title chance": list(odds.values())})
          .sort_values("Title chance", ascending=False))
    fig = px.bar(df, x="Title chance", y="Team", orientation="h",
                 text=df["Title chance"].map(lambda v: f"{v * 100:.1f}%"))
    fig.update_layout(xaxis_tickformat=".0%", yaxis=dict(autorange="reversed"),
                      height=max(300, 40 * len(df)), margin=dict(l=10, r=10, t=10, b=10))
    return fig
