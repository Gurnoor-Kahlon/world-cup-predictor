"""Interactive Streamlit dashboard for the World Cup Match Predictor.

Run from the project root with:

    streamlit run app/streamlit_app.py
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
from odds_converter import odds_table  # noqa: E402
from predictor import MatchPredictor  # noqa: E402

st.set_page_config(page_title="World Cup Match Predictor", page_icon="⚽", layout="wide")

# Brand-ish colours.
HOME_COLOR = "#1f77b4"
DRAW_COLOR = "#7f7f7f"
AWAY_COLOR = "#d62728"


@st.cache_resource(show_spinner="Loading data and training models...")
def get_predictor() -> MatchPredictor:
    """Train the predictor once and cache it across reruns."""
    return MatchPredictor().fit()


def probability_bar(home, away, probs):
    """Horizontal bar of Win/Draw/Loss probabilities."""
    labels = [f"{home} win", "Draw", f"{away} win"]
    values = [probs["home_win"], probs["draw"], probs["away_win"]]
    fig = go.Figure(go.Bar(
        x=[v * 100 for v in values],
        y=labels,
        orientation="h",
        marker_color=[HOME_COLOR, DRAW_COLOR, AWAY_COLOR],
        text=[f"{v * 100:.1f}%" for v in values],
        textposition="auto",
    ))
    fig.update_layout(
        xaxis_title="Probability (%)", yaxis=dict(autorange="reversed"),
        height=240, margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def strength_radar(predictor, home, away):
    """Radar comparing several strength dimensions of the two teams."""
    elo_h, elo_a = predictor.elo.get_rating(home), predictor.elo.get_rating(away)
    atk_h = predictor.poisson.attack.get(home, 1.0)
    atk_a = predictor.poisson.attack.get(away, 1.0)
    # Defence: invert so "higher = better" on the chart.
    def_h = 2 - predictor.poisson.defence.get(home, 1.0)
    def_a = 2 - predictor.poisson.defence.get(away, 1.0)

    # Normalise Elo to a 0..2 scale around the league for comparability.
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
                      height=320, margin=dict(l=20, r=20, t=30, b=20))
    return fig


def render_prediction(predictor, home, away, stage, neutral, context):
    result = predictor.predict(home, away, stage=stage, neutral=neutral, context=context)
    probs = result["probabilities"]
    eg = result["expected_goals"]

    # --- headline metrics ---
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{home} win", f"{probs['home_win'] * 100:.1f}%")
    c2.metric("Draw", f"{probs['draw'] * 100:.1f}%")
    c3.metric(f"{away} win", f"{probs['away_win'] * 100:.1f}%")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Result probabilities")
        st.plotly_chart(probability_bar(home, away, probs), width="stretch")

        st.subheader("Expected score")
        st.markdown(
            f"### {home} **{eg['home']}** – **{eg['away']}** {away}"
            + ("  \n_(neutral venue)_" if neutral else f"  \n_({home} at home)_")
        )

        st.subheader("Most likely scorelines")
        sl = pd.DataFrame(result["top_scorelines"])
        sl["probability"] = (sl["prob"] * 100).round(1).astype(str) + "%"
        st.table(sl[["score", "probability"]].rename(
            columns={"score": "Scoreline", "probability": "Chance"}).head(5))

    with right:
        st.subheader("Team strength")
        st.plotly_chart(strength_radar(predictor, home, away), width="stretch")

        conf = result["confidence"]
        st.subheader("Confidence")
        st.progress(conf["score"] / 100.0, text=f"{conf['label']} ({conf['score']}/100)")
        st.caption("Because " + ", ".join(conf["reasons"]) + ".")

    # --- knockout advance ---
    if "knockout" in result:
        k = result["knockout"]
        st.subheader("Who advances? (extra time / penalties)")
        a1, a2 = st.columns(2)
        a1.metric(f"{home} advances", f"{k['home_advance'] * 100:.1f}%")
        a2.metric(f"{away} advances", f"{k['away_advance'] * 100:.1f}%")

    # --- odds ---
    st.subheader("Fair odds")
    rows = odds_table([probs["home_win"], probs["draw"], probs["away_win"]],
                      [f"{home} win", "Draw", f"{away} win"],
                      margin=config.DEFAULT_BOOKMAKER_MARGIN)
    odds_df = pd.DataFrame(rows)[["outcome", "decimal", "fractional", "american"]]
    odds_df.columns = ["Outcome", "Decimal", "Fractional", "American"]
    st.table(odds_df)
    st.caption(f"Odds include a {int(config.DEFAULT_BOOKMAKER_MARGIN * 100)}% bookmaker "
               "margin to resemble market prices.")

    # --- explanation & key factors ---
    st.subheader("Why this prediction?")
    st.info(result["explanation"])
    factors = pd.DataFrame(result["key_factors"])
    factors.columns = ["Factor", home, away, "Favours"]
    st.table(factors)


def render_simulation(predictor):
    st.subheader("Monte-Carlo tournament simulation")
    st.write("Pick a knockout bracket (4, 8 or 16 teams). Adjacent picks meet in round one.")

    default = [t for t in ["Brazil", "France", "Argentina", "Spain",
                           "Germany", "England", "Portugal", "Netherlands"]
               if t in predictor.teams]
    bracket = st.multiselect("Bracket (in seeding order)", predictor.teams, default=default)
    n_sims = st.slider("Simulations", 1000, 20000, 5000, step=1000)

    size = len(bracket)
    if size not in (2, 4, 8, 16):
        st.warning("Select exactly 2, 4, 8 or 16 teams to run the simulation.")
        return
    if st.button("Run simulation", type="primary"):
        with st.spinner(f"Running {n_sims:,} tournaments..."):
            odds = predictor.simulate_tournament(bracket, n_sims=n_sims)
        df = (pd.DataFrame({"Team": list(odds), "Title chance": list(odds.values())})
              .sort_values("Title chance", ascending=False))
        fig = px.bar(df, x="Title chance", y="Team", orientation="h",
                     text=df["Title chance"].map(lambda v: f"{v * 100:.1f}%"))
        fig.update_layout(xaxis_tickformat=".0%", yaxis=dict(autorange="reversed"),
                          height=400, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")


def main():
    st.title("⚽ World Cup Match Predictor")
    st.caption("Elo + Poisson + machine learning. Educational project — predictions "
               "are estimates, not guarantees.")

    predictor = get_predictor()

    with st.sidebar:
        st.header("Match setup")
        teams = predictor.teams
        default_home = teams.index("Brazil") if "Brazil" in teams else 0
        default_away = teams.index("Germany") if "Germany" in teams else min(1, len(teams) - 1)
        home = st.selectbox("Team A (home)", teams, index=default_home)
        away = st.selectbox("Team B (away)", teams, index=default_away)
        stage = st.selectbox("Tournament stage", config.TOURNAMENT_STAGES)
        neutral = st.checkbox("Neutral venue", value=stage != config.GROUP_STAGE)

        st.divider()
        st.header("Optional real-world context")
        st.caption("These let you fold in information the sample data doesn't contain.")
        home_inj = st.number_input(f"{home} key players out", 0, 11, 0)
        away_inj = st.number_input(f"{away} key players out", 0, 11, 0)
        rest_home = st.slider(f"{home} rest days", 1, 14, 7)
        rest_away = st.slider(f"{away} rest days", 1, 14, 7)

    if home == away:
        st.warning("Please pick two different teams.")
        return

    context = {
        "home_injuries": home_inj, "away_injuries": away_inj,
        "rest_days_home": rest_home, "rest_days_away": rest_away,
    }

    tab_pred, tab_sim, tab_about = st.tabs(["🔮 Prediction", "🏆 Tournament sim", "ℹ️ About"])
    with tab_pred:
        render_prediction(predictor, home, away, stage, neutral, context)
    with tab_sim:
        render_simulation(predictor)
    with tab_about:
        st.markdown(
            """
            **How it works.** Each prediction blends three components trained on the
            match history: an **Elo** rating system, a **Poisson** expected-goals
            model and a **gradient-boosting** classifier. See `docs/model_explanation.md`.

            **Data.** The app ships with synthetic **sample data** so it runs out of the
            box. Connect real data by replacing `data/sample_matches.csv` — see
            `docs/data_sources.md`.

            **Disclaimer.** Predictions are statistical estimates, **not** guaranteed
            outcomes, and this is **not** betting advice.
            """
        )


if __name__ == "__main__":
    main()
