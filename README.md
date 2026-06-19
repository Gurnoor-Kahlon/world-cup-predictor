# ⚽ World Cup Match Predictor

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![CI](https://github.com/your-username/world-cup-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/world-cup-predictor/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<!-- After pushing, replace `your-username` above (and in the clone URL below) with your GitHub username so the CI badge resolves. -->

> Estimate the most realistic possible **odds, expected goals and likely scorelines**
> for international football matches — using Elo ratings, team form, a Poisson goal
> model and machine learning, all wrapped in an interactive Streamlit dashboard.

A portfolio-grade project that blends classic football analytics with ML. It ships
with **sample data so it runs immediately**, and is structured so you can plug in
**real datasets and APIs** later without rewriting the core.

---

## ✨ Features

For any **Team A vs Team B** matchup at a chosen tournament stage, the app predicts:

- 🟢 **Win / Draw / Loss probabilities** (calibrated, summing to 100%)
- 🎯 **Expected goals** for each team
- 🔢 **Most likely scorelines** (from a Poisson scoreline grid)
- 📈 **Fair odds** in decimal, fractional and American formats
- 🧭 **Confidence rating** (how sure the model is, and why)
- 💬 **Plain-English explanation** of *why* it made the prediction
- 🥅 **Knockout mode** – converts draw probability into "who advances" using a
  shootout/extra-time split for round-of-16 → final
- 🎲 **Monte Carlo tournament simulation** to estimate each team's title chance

### Factors the model uses

| Used today (from match history)              | Designed-in hooks for real data later        |
|----------------------------------------------|----------------------------------------------|
| Elo rating (auto-computed)                   | FIFA ranking                                 |
| Recent form (rolling results & goals)        | Injuries / suspensions                       |
| Head-to-head record                          | Player club form & minutes played            |
| Goals scored & conceded rates                | Squad market value                           |
| Strength of schedule (opponent quality)      | Manager history                              |
| Home / neutral-venue advantage               | Travel distance & altitude                   |
| Tournament stage importance                  | Weather / stadium                            |
| Rest days between matches                    | Expected goals (xG) feeds (FBref/StatsBomb)  |
| Penalty-shootout split for knockouts         | Historical World Cup performance weighting   |

> The factors on the right are **not invented with fake numbers**. The code exposes
> clean inputs (an optional `context` dict and extra CSV columns) plus
> [`docs/data_sources.md`](docs/data_sources.md) explaining exactly where to get
> each one and how to wire it in.

---

## 🖼️ Screenshots

> _Placeholder — add your own screenshots/GIFs here after running the app._

| Prediction view | Team comparison | Tournament sim |
|---|---|---|
| ![prediction](app/assets/screenshot_prediction.png) | ![comparison](app/assets/screenshot_comparison.png) | ![simulation](app/assets/screenshot_simulation.png) |

---

## 🗂️ Project structure

```
world-cup-predictor/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── data/
│   ├── raw/                     # drop real downloaded datasets here
│   ├── processed/               # generated artefacts (git-ignored)
│   ├── sample_matches.csv       # bundled synthetic sample dataset
│   └── teams.csv                # optional per-team metadata (rank, value…)
├── notebooks/
│   └── model_exploration.ipynb  # walkthrough of the pipeline
├── src/
│   ├── main.py                  # CLI: train + print a sample prediction
│   ├── config.py                # paths & tunable constants
│   ├── data_loader.py           # load/validate match data
│   ├── feature_engineering.py   # form, h2h, rest, strength-of-schedule…
│   ├── model.py                 # Elo + Poisson + ML classifier
│   ├── predictor.py             # orchestrates a full prediction
│   ├── odds_converter.py        # probabilities ↔ odds
│   └── utils.py                 # small shared helpers
├── app/
│   ├── streamlit_app.py         # interactive dashboard
│   └── assets/                  # images/screenshots
├── tests/
│   ├── test_features.py
│   ├── test_odds.py
│   └── test_predictor.py
└── docs/
    ├── model_explanation.md     # how the maths/ML works
    └── data_sources.md          # where to get real data
```

---

## 🚀 Installation

```bash
# 1. clone
git clone https://github.com/<your-username>/world-cup-predictor.git
cd world-cup-predictor

# 2. (recommended) create a virtual environment
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate

# 3. install dependencies
pip install -r requirements.txt
```

## ▶️ How to run

**The web app:**

```bash
streamlit run app/streamlit_app.py
```

Then open the URL Streamlit prints (usually <http://localhost:8501>).

**The command line (quick check, no browser):**

```bash
python src/main.py --home Brazil --away Germany --stage final
```

**The tests:**

```bash
pytest
```

---

## 🧠 How the model works

The prediction is an **ensemble of three complementary components**, each grounded
in real match data (see [`docs/model_explanation.md`](docs/model_explanation.md) for
the full write-up):

1. **Elo rating system** — every team starts at 1500 and its rating moves after each
   match based on the result, the goal margin and the importance of the competition
   (World Cup games count more than friendlies). Elo captures *long-run strength*.

2. **Poisson goal model** — from historical scoring/conceding rates we derive each
   team's **attack** and **defence** strength, adjust expected goals by the Elo gap
   and home/neutral venue, then use the **Poisson distribution** to build a full
   grid of scoreline probabilities. Summing the grid gives Win/Draw/Loss and the
   most likely scores.

3. **Machine-learning classifier** — a `GradientBoostingClassifier` (with optional
   XGBoost/LightGBM backends) trained on engineered features: Elo gap, recent form,
   goal differentials, rest days, head-to-head and strength of schedule. It learns
   patterns the formulas miss.

The three result-probability sets are **blended** with configurable weights and
re-normalised. **Confidence** reflects how much the components agree and how much
history exists for both teams. For knockout stages, draw probability is split into
"who advances" using a shootout/extra-time model.

> No magic, no random numbers: every probability traces back to data and documented
> formulas. The weights and constants live in [`src/config.py`](src/config.py).

---

## 📊 Example prediction output

```text
Brazil vs Germany — Final (neutral venue)

  Result probabilities
    Brazil win .......... 42.7%
    Draw ................ 25.1%
    Germany win ......... 32.2%

  Expected goals:  Brazil 1.46  -  1.21 Germany
  Fair decimal odds:  Brazil 2.34  |  Draw 3.98  |  Germany 3.11

  Most likely scorelines
    1-1 ....... 11.8%
    1-0 ....... 10.4%
    2-1 ........ 8.6%

  Confidence: Medium (components broadly agree; solid match history)

  Why: Brazil have the stronger recent form and a higher Elo rating, while
  Germany's defence has been slightly leakier. The venue is neutral, so no
  home advantage is applied — hence the odds are close rather than one-sided.
```

> Numbers above are illustrative of the output *format*. Actual values depend on the
> data you load.

---

## 🔌 Data sources

The app runs on bundled **synthetic sample data** out of the box. To use real data,
drop a CSV into `data/` with these columns and point `config.py` at it:

| column       | description                              |
|--------------|------------------------------------------|
| `date`       | match date (`YYYY-MM-DD`)                |
| `home_team`  | home team name                           |
| `away_team`  | away team name                           |
| `home_score` | goals scored by the home team            |
| `away_score` | goals scored by the away team            |
| `tournament` | competition name (optional)              |
| `neutral`    | `True`/`False` neutral venue (optional)  |

Recommended free/legal sources (details and links in
[`docs/data_sources.md`](docs/data_sources.md)):

- **Kaggle** – "International football results from 1872 to 2017" (matches the schema)
- **football-data.co.uk** – club match results & odds
- **World Football Elo Ratings** & **FIFA rankings**
- **FBref** / **StatsBomb open data** – advanced stats & xG
- **API-Football**, **Sportmonks** – live fixtures & lineups (API keys)
- **Transfermarkt** – market values (respect their terms of use)

---

## ⚠️ Limitations

- Default data is **synthetic** — useful for demos and testing, not real forecasting.
- Football is **high-variance**; even good models are wrong often. Treat outputs as
  probability estimates, not certainties.
- Many requested factors (injuries, lineups, weather…) require external data you
  must connect; until then they don't influence the numbers.
- The ML model is only as good as the history it's trained on; small datasets =
  noisy models.

## 🛣️ Future improvements

- Connect a live fixtures/lineups API and auto-refresh ratings
- Add injuries, suspensions and squad market value as real features
- Probability **calibration** (Platt / isotonic) and back-testing with log-loss/Brier
- Dixon–Coles low-score correction for the Poisson model
- Player-level xG aggregation from StatsBomb/FBref
- Full bracket builder with live Monte Carlo title odds

## 🚫 Disclaimer

This project is for **education and portfolio purposes only**. Predictions are
statistical **estimates, not guaranteed outcomes**. It is **not** betting advice, and
no claims are made about accuracy or profitability. Please gamble responsibly — or
better yet, just enjoy the football. 🏆

## 📄 License

Released under the [MIT License](LICENSE).
