# ⚽ World Cup Match Predictor

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Ruff](https://img.shields.io/badge/lint-ruff-D7FF64?logo=ruff&logoColor=black)](https://docs.astral.sh/ruff/)
[![CI](https://github.com/Gurnoor-Kahlon/world-cup-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/Gurnoor-Kahlon/world-cup-predictor/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Estimate realistic **odds, expected goals and likely scorelines** for international
> football matches — using **Elo ratings**, a **Poisson + Dixon–Coles** goal model and
> **machine learning**, blended into one explainable ensemble, with an interactive
> **Streamlit** dashboard, a **FastAPI** backend and a **leak-free backtest** that
> actually measures the models.

A portfolio-grade football-analytics + ML project. It ships with **sample data so it
runs instantly**, and includes a one-command pipeline to swap in **real public data**.

---

## ✨ Features

For any **Team A vs Team B** matchup at a chosen stage, the model predicts:

- 🟢 **Win / Draw / Loss probabilities** (calibrated, summing to 100%)
- 🎯 **Expected goals** for each team
- 🔢 **Most likely scorelines** + a full **scoreline heatmap**
- 📈 **Fair odds** in decimal, fractional and American formats
- 🧭 **Confidence rating** with reasons
- 💬 **Plain-English explanation** of *why* it made the prediction
- 🥅 **Knockout mode** — converts draws into "who advances" (extra time / penalties)
- 🌎 **2026 World Cup mode** — host advantage for USA / Canada / Mexico
- 🎲 **Monte-Carlo tournament simulation** for title odds
- 📊 **Backtesting dashboard** — accuracy, log loss, Brier score, calibration

### What the model uses

| Used today (from match history)              | Designed-in hooks for real data later        |
|----------------------------------------------|----------------------------------------------|
| Elo rating (auto-computed)                   | FIFA ranking                                 |
| Recent form (rolling results & goals)        | Injuries / suspensions (`context`)           |
| Head-to-head record                          | Player club form & minutes                   |
| Goals scored & conceded rates                | Squad market value                           |
| Strength of schedule                         | Manager history                              |
| Home / neutral / **host** advantage          | Travel distance & altitude                   |
| Tournament stage importance                  | Weather / stadium                            |
| Rest days; penalty-shootout split            | Expected goals (xG) feeds (FBref/StatsBomb)  |

> The right-hand column is **not faked**. The code exposes clean inputs (a `context`
> dict and extra CSV columns) and [`docs/data_sources.md`](docs/data_sources.md)
> explains exactly where to get each one.

---

## 🖼️ Screenshots & demo

> _Placeholders — add your own after running the app (see below)._

| Match Predictor | Scoreline Heatmap | Backtesting |
|---|---|---|
| ![predictor](app/assets/screenshot_predictor.png) | ![heatmap](app/assets/screenshot_heatmap.png) | ![backtest](app/assets/screenshot_backtest.png) |

**To create them:** run `streamlit run app/streamlit_app.py`, open each page, and
screenshot. For a demo GIF, record the window with [ScreenToGif](https://www.screentogif.com/)
(Windows) or [LICEcap](https://www.cockos.com/licecap/), save as
`app/assets/demo.gif`, and embed it here with `![demo](app/assets/demo.gif)`.

---

## 🚀 Quick start

```bash
git clone https://github.com/Gurnoor-Kahlon/world-cup-predictor.git
cd world-cup-predictor

python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Then pick one:

```bash
# Interactive dashboard
streamlit run app/streamlit_app.py

# Command line
python src/main.py --home Brazil --away Germany --stage Final --neutral

# REST API (optional extra)
pip install -e ".[api]"
uvicorn api.main:app --reload --port 8000   # docs at /docs

# Backtest the models
python scripts/run_backtest.py

# Run the tests
pytest
```

A `Makefile` wraps these (`make app`, `make api`, `make backtest`, `make test`,
`make lint`). A `Dockerfile` is included — see [`docs/deployment.md`](docs/deployment.md).

---

## 🧠 How the model works

A prediction is an **ensemble** of three components, each grounded in real match data
(full write-up in [`docs/model_explanation.md`](docs/model_explanation.md)):

1. **Elo rating system** — long-run strength, updated after each match with
   margin-of-victory and tournament-importance weighting and a home/neutral adjustment.
2. **Poisson goal model** — attack/defence strengths → expected goals → a full
   scoreline grid, with an optional **Dixon–Coles** low-score correction.
3. **Machine-learning classifier** — gradient boosting (scikit-learn; optional
   XGBoost/LightGBM) on engineered features (Elo gap, form, head-to-head, rest,
   strength of schedule…).

Their Win/Draw/Loss probabilities are **blended** with configurable weights
(`config.BLEND_WEIGHTS`) and re-normalised. Confidence reflects model agreement and
data depth; knockouts split the draw into "who advances".

> No random numbers — every probability traces to data and documented formulas.

---

## 📊 Example prediction

```text
Brazil vs Germany - Final (neutral venue)

  Result probabilities      Expected goals:  Brazil 1.72 - 1.39 Germany
    Brazil win ....  44.5%   Fair decimal odds:  Brazil 2.14 | Draw 3.68 | Germany 3.21
    Draw .........  25.9%
    Germany win ...  29.6%   Most likely scoreline: 1-1 (11.2%)

  Why: Brazil have the stronger recent form and a higher Elo rating; the venue is
  neutral, so no home advantage is applied — hence the odds are moderate, not one-sided.
```

*(Values from the real-data pipeline; exact numbers depend on the dataset loaded.)*

---

## 📈 Backtesting & evaluation

The project **measures** itself with a leak-free walk-forward backtest (train on the
past, predict each future match, score with proper metrics). Example on the bundled
sample data (1,380 train / 460 test), via `python scripts/run_backtest.py`:

| model     | accuracy | log_loss | brier  |
|-----------|---------:|---------:|-------:|
| ensemble  |   0.483  |  1.0325  | 0.6186 |
| poisson   |   0.500  |  1.0392  | 0.6224 |
| elo       |   0.476  |  1.0687  | 0.6432 |
| ml        |   0.454  |  1.0713  | 0.6412 |

The **ensemble has the best (lowest) log loss**, beating the `ln(3) ≈ 1.099` blind-guess
baseline — i.e. its probabilities are better than guessing. Accuracy in the high-40s/50%
is expected for a draw-heavy 3-way outcome. **These numbers describe the sample data
only**; connect real data for meaningful figures. Details:
[`docs/backtesting.md`](docs/backtesting.md).

---

## 🗂️ Project structure

```
world-cup-predictor/
├── app/
│   ├── streamlit_app.py        # multipage navigation entry
│   ├── pages_views.py          # the 7 dashboard pages
│   └── app_lib.py              # cached resources + charts
├── api/
│   └── main.py                 # FastAPI backend
├── src/
│   ├── config.py               # all tunable constants + paths
│   ├── data_loader.py          # load/validate/auto-resolve data
│   ├── feature_engineering.py  # leak-free point-in-time features
│   ├── model.py                # Elo + Poisson(+Dixon–Coles) + ML
│   ├── predictor.py            # ensemble, explanation, sim, save/load
│   ├── evaluation.py           # metrics + walk-forward Backtester
│   ├── tournament_2026.py      # host advantage + 2026 simulation
│   ├── odds_converter.py       # probabilities ↔ odds
│   ├── utils.py                # shared helpers
│   └── main.py                 # CLI
├── scripts/
│   ├── download_data.py        # fetch real public data
│   ├── prepare_data.py         # validate + convert to processed/
│   └── run_backtest.py         # backtest CLI
├── data/                       # sample_matches.csv, teams.csv, raw/, processed/
├── docs/                       # model_explanation, data_sources, backtesting, deployment
├── notebooks/model_exploration.ipynb
├── tests/                      # pytest suite (69 tests)
├── Dockerfile · Makefile · .env.example · pyproject.toml · requirements.txt
```

---

## 🔌 Data sources

Runs on bundled **synthetic sample data** out of the box. To use **real** data:

```bash
python scripts/download_data.py   # real public results -> data/raw/
python scripts/prepare_data.py    # -> data/processed/matches.csv (auto-detected)
```

The loader then uses the real data automatically. Recommended sources (links and
per-factor guidance in [`docs/data_sources.md`](docs/data_sources.md)): the
martj42/Kaggle international-results dataset, football-data.co.uk, World Football Elo,
FIFA rankings, FBref/StatsBomb (xG), API-Football/Sportmonks (live), Transfermarkt
(market value).

---

## ⚠️ Limitations

- Default data is **synthetic** — great for demos/tests, not real forecasting.
- Football is **high-variance**; even good models are wrong often.
- Several listed factors (injuries, lineups, weather…) require external data you must
  connect; until then they don't influence the numbers.
- Probabilities are backtested but **not yet re-calibrated** (Platt/isotonic).

## 🛣️ Future improvements

- Probability calibration + richer back-testing (rolling, per-confederation)
- Real injuries / squad value / xG as first-class features
- Full 48-team 2026 group + bracket builder once the draw is final
- MLE-fitted Dixon–Coles `rho`; time-decay weighting of older matches
- Model artifact registry + scheduled retraining

## 🚫 Disclaimer

For **education and portfolio purposes only**. Predictions are statistical
**estimates, not guaranteed outcomes**, and this is **not** betting advice. No claims
are made about real-world accuracy or profitability. Please gamble responsibly — or
just enjoy the football. 🏆

## 📄 License

Released under the [MIT License](LICENSE).
