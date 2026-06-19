"""Page render functions for the Streamlit dashboard.

Kept separate from ``streamlit_app.py`` (which only wires up navigation) so each
page can be imported and unit-tested in isolation with ``AppTest.from_function``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure this app/ directory is importable for `app_lib`.
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import pandas as pd
import streamlit as st

import app_lib as lib  # adds src/ to sys.path as a side effect

import config  # noqa: E402
import tournament_2026 as wc  # noqa: E402
from odds_converter import odds_table  # noqa: E402

DISCLAIMER = ("Educational project — predictions are statistical **estimates, not "
              "guarantees**, and this is **not** betting advice.")


def page_predictor():
    st.title("🔮 Match Predictor")
    st.caption(DISCLAIMER)
    predictor = lib.get_predictor()
    st.sidebar.header("Match setup")
    home, away, stage, neutral, context = lib.team_selectors(predictor, "pred", with_context=True)
    if home == away:
        st.warning("Please pick two different teams.")
        return

    result = predictor.predict(home, away, stage=stage, neutral=neutral, context=context)
    probs = result["probabilities"]
    eg = result["expected_goals"]

    c1, c2, c3 = st.columns(3)
    c1.metric(f"{home} win", f"{probs['home_win'] * 100:.1f}%")
    c2.metric("Draw", f"{probs['draw'] * 100:.1f}%")
    c3.metric(f"{away} win", f"{probs['away_win'] * 100:.1f}%")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Result probabilities")
        st.plotly_chart(lib.probability_bar(home, away, probs), width="stretch")
        st.subheader("Expected score")
        st.markdown(f"### {home} **{eg['home']}** – **{eg['away']}** {away}"
                    + ("  \n_(neutral venue)_" if neutral else f"  \n_({home} at home)_"))
        st.subheader("Most likely scorelines")
        sl = pd.DataFrame(result["top_scorelines"]).head(5)
        sl["Chance"] = (sl["prob"] * 100).round(1).astype(str) + "%"
        st.table(sl[["score", "Chance"]].rename(columns={"score": "Scoreline"}))
    with right:
        st.subheader("Team strength")
        st.plotly_chart(lib.strength_radar(predictor, home, away), width="stretch")
        conf = result["confidence"]
        st.subheader("Confidence")
        st.progress(conf["score"] / 100.0, text=f"{conf['label']} ({conf['score']}/100)")
        st.caption("Because " + ", ".join(conf["reasons"]) + ".")

    if "knockout" in result:
        k = result["knockout"]
        st.subheader("Who advances? (extra time / penalties)")
        a1, a2 = st.columns(2)
        a1.metric(f"{home} advances", f"{k['home_advance'] * 100:.1f}%")
        a2.metric(f"{away} advances", f"{k['away_advance'] * 100:.1f}%")

    st.subheader("Fair odds")
    rows = odds_table([probs["home_win"], probs["draw"], probs["away_win"]],
                      [f"{home} win", "Draw", f"{away} win"],
                      margin=config.DEFAULT_BOOKMAKER_MARGIN)
    odds_df = pd.DataFrame(rows)[["outcome", "decimal", "fractional", "american"]]
    odds_df.columns = ["Outcome", "Decimal", "Fractional", "American"]
    st.table(odds_df)
    st.caption(f"Includes a {int(config.DEFAULT_BOOKMAKER_MARGIN * 100)}% bookmaker margin.")

    st.subheader("Why this prediction?")
    st.info(result["explanation"])
    factors = pd.DataFrame(result["key_factors"])
    factors.columns = ["Factor", home, away, "Favours"]
    st.table(factors)


def page_comparison():
    st.title("📊 Team Comparison")
    st.caption("Compare two teams on Elo, form, attack and defence.")
    predictor = lib.get_predictor()
    st.sidebar.header("Teams")
    home, away, _, _, _ = lib.team_selectors(predictor, "cmp")
    if home == away:
        st.warning("Please pick two different teams.")
        return

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Metric comparison")
        st.plotly_chart(lib.comparison_grouped_bar(predictor, home, away), width="stretch")
    with right:
        st.subheader("Strength radar")
        st.plotly_chart(lib.strength_radar(predictor, home, away), width="stretch")

    st.subheader("Side-by-side")
    st.table(lib.comparison_table_df(predictor, home, away))


def page_heatmap():
    st.title("🔥 Scoreline Heatmap")
    st.caption("Probability of every exact scoreline, from the Poisson model.")
    predictor = lib.get_predictor()
    st.sidebar.header("Match setup")
    home, away, _, neutral, _ = lib.team_selectors(predictor, "heat")
    if home == away:
        st.warning("Please pick two different teams.")
        return

    elo_diff = predictor.elo.get_rating(home) - predictor.elo.get_rating(away)
    lam_h, lam_a = predictor.poisson.expected_goals(home, away, neutral, elo_diff)
    c1, c2 = st.columns(2)
    c1.metric(f"{home} expected goals", f"{lam_h:.2f}")
    c2.metric(f"{away} expected goals", f"{lam_a:.2f}")
    st.plotly_chart(lib.scoreline_heatmap(predictor, home, away, neutral), width="stretch")
    st.caption(f"Brighter cells are more likely. Rows = {home} goals, columns = {away} goals.")


def page_simulator():
    st.title("🏆 Tournament Simulator")
    st.caption("Monte-Carlo a knockout bracket to estimate each team's title chance.")
    predictor = lib.get_predictor()
    default = [t for t in ["Brazil", "France", "Argentina", "Spain",
                           "Germany", "England", "Portugal", "Netherlands"]
               if t in predictor.teams]
    bracket = st.multiselect("Bracket (seeding order; 2/4/8/16 teams)",
                             predictor.teams, default=default)
    n_sims = st.slider("Simulations", 1000, 20000, 5000, step=1000)
    if len(bracket) not in (2, 4, 8, 16):
        st.warning("Select exactly 2, 4, 8 or 16 teams.")
        return
    if st.button("Run simulation", type="primary"):
        with st.spinner(f"Running {n_sims:,} tournaments..."):
            odds = predictor.simulate_tournament(bracket, n_sims=n_sims)
        st.plotly_chart(lib.title_odds_bar(odds), width="stretch")


def page_2026():
    st.title("🌎 2026 World Cup Mode")
    st.caption("Co-hosted by USA, Canada & Mexico — hosts get a partial home advantage.")
    predictor = lib.get_predictor()

    hosts = wc.available_hosts(predictor)
    st.info(f"Hosts present in the loaded data: **{', '.join(hosts) or 'none'}**. "
            "Qualified teams and groups for 2026 are not final, so the default bracket "
            "below is a **placeholder** built from the strongest available teams.")

    size = st.selectbox("Bracket size", [4, 8, 16], index=1)
    default = wc.default_bracket(predictor, size=size)
    bracket = st.multiselect("Bracket (seeding order)", predictor.teams, default=default)
    host_fraction = st.slider("Host advantage strength", 0.0, 1.0,
                              float(config.HOST_ADVANTAGE_FRACTION), 0.05)
    n_sims = st.slider("Simulations", 1000, 20000, 5000, step=1000, key="wc26_sims")

    if len(bracket) not in (2, 4, 8, 16):
        st.warning("Select exactly 2, 4, 8 or 16 teams.")
        return
    if st.button("Simulate 2026", type="primary"):
        with st.spinner(f"Running {n_sims:,} tournaments with host advantage..."):
            odds = wc.simulate_bracket(predictor, bracket, n_sims=n_sims,
                                       host_fraction=host_fraction)
        st.plotly_chart(lib.title_odds_bar(odds), width="stretch")
        hosted = [t for t in bracket if wc.is_host(t)]
        if hosted:
            st.caption(f"Host advantage applied to: {', '.join(hosted)}.")


def page_backtesting():
    st.title("📈 Backtesting Dashboard")
    st.caption("Leak-free walk-forward evaluation. Lower log loss / Brier is better.")
    test_fraction = st.slider("Test fraction (most recent matches held out)",
                              0.1, 0.4, float(config.BACKTEST_TEST_FRACTION), 0.05)
    res = lib.run_backtest(test_fraction)
    st.write(f"Trained on **{res['n_train']:,}** matches, scored on **{res['n_test']:,}** "
             "test matches.")

    st.subheader("Model comparison")
    table = res["comparison"].rename(columns={"model": "Model", "accuracy": "Accuracy",
                                              "log_loss": "Log loss", "brier": "Brier", "n": "N"})
    st.dataframe(table.style.format({"Accuracy": "{:.3f}", "Log loss": "{:.4f}",
                                     "Brier": "{:.4f}"}), width="stretch")
    st.caption("Baseline: blind 1/3-1/3-1/3 guessing scores a log loss of ln(3) ≈ 1.099. "
               "A model worth keeping should beat that.")

    st.subheader("Calibration (P home win)")
    keys = list(res["calibration"].keys())
    default_idx = keys.index("ensemble") if "ensemble" in keys else 0
    model = st.selectbox("Model", keys, index=default_idx)
    st.plotly_chart(lib.calibration_chart(res["calibration"][model], model), width="stretch")
    st.caption("Points near the dashed diagonal mean well-calibrated probabilities.")


def page_about():
    st.title("ℹ️ About & Methodology")
    st.markdown(
        """
        ### How a prediction is made
        Each prediction is an **ensemble** of three components trained on historical
        international matches:

        1. **Elo rating system** — long-run team strength, updated after every match
           with margin-of-victory and tournament-importance weighting.
        2. **Poisson goal model** — attack/defence strengths → expected goals → a full
           scoreline grid, with an optional **Dixon–Coles** low-score correction.
        3. **Machine-learning classifier** — gradient boosting on engineered features
           (Elo gap, form, head-to-head, rest, strength of schedule…).

        Their Win/Draw/Loss probabilities are **blended** with configurable weights and
        re-normalised. For knockouts, the draw is split into "who advances".

        ### Honest evaluation
        The **Backtesting** page runs a leak-free walk-forward test and compares
        Elo-only / Poisson-only / ML-only / Ensemble on accuracy, log loss and Brier
        score. See `docs/backtesting.md` and `docs/model_explanation.md`.

        ### Data
        Ships with synthetic **sample data** so it runs instantly. Connect real data
        with `python scripts/download_data.py && python scripts/prepare_data.py`
        (see `docs/data_sources.md`).

        ### Limitations & disclaimer
        Football is high-variance; probabilities are estimates, not certainties.
        """
    )
    st.warning(DISCLAIMER)


# All pages, in display order, for the navigation to import.
ALL_PAGES = [
    (page_predictor, "Match Predictor", "🔮", True),
    (page_comparison, "Team Comparison", "📊", False),
    (page_heatmap, "Scoreline Heatmap", "🔥", False),
    (page_simulator, "Tournament Simulator", "🏆", False),
    (page_2026, "2026 World Cup", "🌎", False),
    (page_backtesting, "Backtesting", "📈", False),
    (page_about, "About", "ℹ️", False),
]
