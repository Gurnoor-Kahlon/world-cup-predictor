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
    lib.data_source_caption(predictor)
    st.sidebar.header("Match setup")
    home, away, stage, neutral, context = lib.team_selectors(predictor, "pred", with_context=True)
    if home == away:
        st.info("👈 Pick **two different teams** in the sidebar to see a prediction.")
        return

    try:
        result = predictor.predict(home, away, stage=stage, neutral=neutral, context=context)
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.error(f"Sorry, the prediction failed: {exc}")
        return
    probs = result["probabilities"]
    eg = result["expected_goals"]

    st.subheader(f"{home} vs {away}")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{home} win", f"{probs['home_win'] * 100:.1f}%",
              help="Probability the home/Team A side wins in 90 minutes.")
    c2.metric("Draw", f"{probs['draw'] * 100:.1f}%",
              help="Probability the match is level after 90 minutes.")
    c3.metric(f"{away} win", f"{probs['away_win'] * 100:.1f}%",
              help="Probability the away/Team B side wins in 90 minutes.")
    st.divider()

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

    st.divider()
    st.subheader("Fair odds")
    rows = odds_table([probs["home_win"], probs["draw"], probs["away_win"]],
                      [f"{home} win", "Draw", f"{away} win"],
                      margin=config.DEFAULT_BOOKMAKER_MARGIN)
    odds_df = pd.DataFrame(rows)[["outcome", "decimal", "fractional", "american"]]
    odds_df.columns = ["Outcome", "Decimal", "Fractional", "American"]
    st.table(odds_df)
    st.caption(f"Fair odds derived from the model probabilities, plus a "
               f"{int(config.DEFAULT_BOOKMAKER_MARGIN * 100)}% bookmaker margin to resemble "
               "market prices. Not betting advice.")

    st.divider()
    st.subheader("Why this prediction?")
    st.info(result["explanation"])
    factors = pd.DataFrame(result["key_factors"])
    factors.columns = ["Factor", home, away, "Favours"]
    st.table(factors)


def page_comparison():
    st.title("📊 Team Comparison")
    st.caption("Compare two teams on Elo, form, attack and defence.")
    predictor = lib.get_predictor()
    lib.data_source_caption(predictor)
    st.sidebar.header("Teams")
    home, away, _, _, _ = lib.team_selectors(predictor, "cmp")
    if home == away:
        st.info("👈 Pick **two different teams** in the sidebar to compare them.")
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
    lib.data_source_caption(predictor)
    st.sidebar.header("Match setup")
    home, away, _, neutral, _ = lib.team_selectors(predictor, "heat")
    if home == away:
        st.info("👈 Pick **two different teams** in the sidebar to see the heatmap.")
        return

    elo_diff = predictor.elo.get_rating(home) - predictor.elo.get_rating(away)
    lam_h, lam_a = predictor.poisson.expected_goals(home, away, neutral, elo_diff)
    c1, c2 = st.columns(2)
    c1.metric(f"{home} expected goals", f"{lam_h:.2f}",
              help="Average goals the model expects this team to score.")
    c2.metric(f"{away} expected goals", f"{lam_a:.2f}",
              help="Average goals the model expects this team to score.")
    st.plotly_chart(lib.scoreline_heatmap(predictor, home, away, neutral), width="stretch")
    st.caption(f"Brighter cells are more likely. Rows = {home} goals, columns = {away} goals. "
               "Built from two independent Poisson distributions (plus the Dixon–Coles "
               "low-score correction).")


def page_simulator():
    st.title("🏆 Tournament Simulator")
    st.caption("Monte-Carlo a knockout bracket to estimate each team's title chance.")
    predictor = lib.get_predictor()
    lib.data_source_caption(predictor)
    default = [t for t in ["Brazil", "France", "Argentina", "Spain",
                           "Germany", "England", "Portugal", "Netherlands"]
               if t in predictor.teams]
    bracket = st.multiselect(
        "Bracket (seeding order; 2/4/8/16 teams)", predictor.teams, default=default,
        help="Adjacent picks meet in round one, then winners advance until one remains.")
    n_sims = st.slider("Simulations", 1000, 20000, 5000, step=1000,
                       help="More simulations = smoother, more stable title odds.")
    if len(bracket) not in (2, 4, 8, 16):
        st.warning("Select exactly **2, 4, 8 or 16** teams to run the simulation "
                   f"(currently {len(bracket)} selected).")
        return
    if st.button("Run simulation", type="primary"):
        with st.spinner(f"Running {n_sims:,} tournaments..."):
            odds = predictor.simulate_tournament(bracket, n_sims=n_sims)
        favourite = max(odds, key=odds.get)
        st.metric("Most likely champion", favourite, f"{odds[favourite] * 100:.1f}% title chance")
        st.plotly_chart(lib.title_odds_bar(odds), width="stretch")
    else:
        st.info("Set up your bracket above, then press **Run simulation**.")


def page_2026():
    st.title("🌎 2026 World Cup Mode")
    st.caption("Co-hosted by USA, Canada & Mexico — hosts get a partial home advantage.")
    predictor = lib.get_predictor()

    hosts = wc.available_hosts(predictor)
    st.info(f"Hosts present in the loaded data: **{', '.join(hosts) or 'none'}**. "
            "Qualified teams and groups for 2026 are not final, so the default bracket "
            "below is a **placeholder** built from the strongest available teams.")

    size = st.selectbox("Bracket size", [4, 8, 16], index=1,
                        help="How many teams contest the knockout bracket.")
    default = wc.default_bracket(predictor, size=size)
    bracket = st.multiselect("Bracket (seeding order)", predictor.teams, default=default)
    host_fraction = st.slider(
        "Host advantage strength", 0.0, 1.0, float(config.HOST_ADVANTAGE_FRACTION), 0.05,
        help="0 = treat hosts as neutral; 1 = full home advantage for USA/Canada/Mexico.")
    n_sims = st.slider("Simulations", 1000, 20000, 5000, step=1000, key="wc26_sims",
                       help="More simulations = smoother, more stable title odds.")

    if len(bracket) not in (2, 4, 8, 16):
        st.warning("Select exactly **2, 4, 8 or 16** teams "
                   f"(currently {len(bracket)} selected).")
        return
    if st.button("Simulate 2026", type="primary"):
        with st.spinner(f"Running {n_sims:,} tournaments with host advantage..."):
            odds = wc.simulate_bracket(predictor, bracket, n_sims=n_sims,
                                       host_fraction=host_fraction)
        favourite = max(odds, key=odds.get)
        st.metric("Most likely champion", favourite, f"{odds[favourite] * 100:.1f}% title chance")
        st.plotly_chart(lib.title_odds_bar(odds), width="stretch")
        hosted = [t for t in bracket if wc.is_host(t)]
        if hosted:
            st.caption(f"Host advantage applied to: {', '.join(hosted)}.")
    else:
        st.info("Adjust the bracket and host advantage, then press **Simulate 2026**.")


def page_backtesting():
    st.title("📈 Backtesting Dashboard")
    st.caption("Leak-free walk-forward evaluation — train on the past, predict the future, "
               "then score it honestly.")
    predictor = lib.get_predictor()
    lib.data_source_caption(predictor)

    test_fraction = st.slider(
        "Test fraction (most recent matches held out)", 0.1, 0.4,
        float(config.BACKTEST_TEST_FRACTION), 0.05,
        help="Share of the newest matches used only for testing, never for training.")
    res = lib.run_backtest(test_fraction)
    st.write(f"Trained on **{res['n_train']:,}** matches, scored on **{res['n_test']:,}** "
             "test matches.")

    comp = res["comparison"]
    # Headline cards for the ensemble (the model the app actually uses).
    ens = comp[comp["model"] == "ensemble"]
    if not ens.empty:
        row = ens.iloc[0]
        m1, m2, m3 = st.columns(3)
        m1.metric("Ensemble accuracy", f"{row['accuracy'] * 100:.1f}%",
                  help="Share of test matches whose most likely outcome was correct.")
        m2.metric("Ensemble log loss", f"{row['log_loss']:.4f}",
                  help="Lower is better. Beats the 1.099 blind-guess baseline if < ln(3).")
        m3.metric("Ensemble Brier", f"{row['brier']:.4f}",
                  help="Lower is better. Mean squared error of the probability vector.")

    with st.expander("What do these metrics mean? (plain English)"):
        st.markdown(
            """
            - **Accuracy** — how often the single most likely outcome was right. Simple,
              but ignores *how confident* the model was.
            - **Log loss** — rewards probabilities that are both correct **and** well
              calibrated, and punishes confident mistakes hard. **Lower is better.**
              Blind 1/3-1/3-1/3 guessing scores `ln(3) ≈ 1.099`, so anything below that
              is doing real work.
            - **Brier score** — the average squared error between the predicted
              probabilities and what actually happened. **Lower is better** (0 = perfect).
            - **Calibration** — when the model says "60%", does it happen ~60% of the
              time? The chart below plots predicted vs actual; closer to the dashed line
              is better calibrated.
            """
        )

    st.subheader("Model comparison")
    table = comp.rename(columns={"model": "Model", "accuracy": "Accuracy",
                                 "log_loss": "Log loss", "brier": "Brier", "n": "N"})
    st.dataframe(table.style.format({"Accuracy": "{:.3f}", "Log loss": "{:.4f}",
                                     "Brier": "{:.4f}"}), width="stretch")
    st.caption("Models compared: Elo-only, Poisson-only, ML-only and the blended Ensemble. "
               "Results above describe the loaded dataset only — no claim is made about "
               "real-world accuracy.")

    st.subheader("Calibration (P home win)")
    keys = list(res["calibration"].keys())
    default_idx = keys.index("ensemble") if "ensemble" in keys else 0
    model = st.selectbox("Model", keys, index=default_idx,
                         help="Which model's calibration to plot.")
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
