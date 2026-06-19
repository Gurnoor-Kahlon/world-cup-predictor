# ⚽ World Cup Match Predictor

A Python + Streamlit project that estimates the probable outcome and **odds** of
international football matches. It combines several classic football-analytics
techniques into one explainable pipeline:

- **Elo ratings** – a running strength score for every team, updated after each match.
- **Team form** – recent results and goal trends over a rolling window.
- **Expected goals (xG-style) estimates** – attack/defence strength per team.
- **Poisson modelling** – turns expected goals into a full scoreline probability grid.
- **Machine learning** – a classifier trained on engineered features for Win / Draw / Loss.
- **Odds conversion** – converts probabilities into fair decimal/American odds.

> ⚠️ **Honest disclaimer:** This is a **portfolio / educational project**. It ships
> with small **sample data** so it runs out of the box. No claims are made about
> real-world predictive accuracy or betting profitability. Do not use it to gamble.

---

## 🗂️ Project structure

```
world-cup-predictor/
├── app/
│   └── streamlit_app.py        # Streamlit UI
├── src/wc_predictor/           # Importable Python package
│   ├── config.py               # Paths & tunable constants
│   ├── data_loader.py          # Load/validate match data
│   ├── features.py             # Feature engineering (form, etc.)
│   ├── elo.py                  # Elo rating engine
│   ├── poisson_model.py        # Expected goals + Poisson scoreline grid
│   ├── predictor.py            # Combines everything into a prediction
│   └── odds.py                 # Probability → odds conversion
├── data/
│   ├── raw/sample_matches.csv  # Bundled sample dataset
│   └── processed/              # Generated artefacts (git-ignored)
├── tests/                      # pytest suite
├── notebooks/                  # Optional exploration
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 🚀 Quickstart

```bash
# 1. (recommended) create a virtual environment
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. run the app
streamlit run app/streamlit_app.py
```

Then open the URL Streamlit prints (usually <http://localhost:8501>), pick two
teams, and view the predicted result probabilities and fair odds.

---

## 🧪 Running the tests

```bash
pip install -r requirements.txt
pytest
```

---

## 🔌 Using your own data

The project reads a simple CSV of historical matches. To use real data, replace
`data/raw/sample_matches.csv` (or point `config.py` at a new file) with a CSV that
has the same columns:

| column      | description                          |
|-------------|--------------------------------------|
| `date`      | match date (`YYYY-MM-DD`)            |
| `home_team` | home team name                       |
| `away_team` | away team name                       |
| `home_score`| goals scored by the home team        |
| `away_score`| goals scored by the away team        |
| `tournament`| competition name (optional)          |
| `neutral`   | `True`/`False` neutral venue (optional) |

A great free real-world dataset with this shape is the
["International football results" dataset on Kaggle](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017).

---

## 🛣️ Roadmap

- [ ] Sample data + data loader
- [ ] Feature engineering (form / rolling stats)
- [ ] Elo rating engine
- [ ] Poisson expected-goals model
- [ ] Combined match predictor
- [ ] Odds conversion
- [ ] Streamlit app
- [ ] Test suite

## 📄 License

Released under the [MIT License](LICENSE).
