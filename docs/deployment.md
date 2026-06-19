# Deployment

Several ways to run and ship the project, from a laptop to the cloud.

## 1. Local (virtualenv)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## 2. Docker

A `Dockerfile` is included (Streamlit app, with a healthcheck):

```bash
docker build -t world-cup-predictor .
docker run -p 8501:8501 world-cup-predictor
# open http://localhost:8501
```

To run the **API** in the same image instead:

```bash
docker run -p 8000:8000 world-cup-predictor \
  python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## 3. Streamlit Community Cloud (free)

1. Push this repo to GitHub (already done if you cloned it).
2. Go to <https://share.streamlit.io>, "New app", and select the repo.
3. Set **Main file path** to `app/streamlit_app.py`.
4. Deploy. Streamlit Cloud installs `requirements.txt` automatically.

## 4. FastAPI backend (any host)

```bash
pip install -e ".[api]"
uvicorn api.main:app --host 0.0.0.0 --port 8000
# docs at http://localhost:8000/docs
```

Deploy the API to any platform that runs a Python process (Render, Railway, Fly.io,
a VM, etc.) using the same `uvicorn` command. Containerise with the Dockerfile for
reproducibility.

## Configuration / environment

Copy `.env.example` to `.env` and adjust. The only functional override today is:

| Variable           | Purpose                                                        |
|--------------------|----------------------------------------------------------------|
| `WCP_MATCHES_PATH` | Use a custom matches CSV instead of auto-detected sample/real data |
| `PORT`             | Port for the app/API (informational; pass to the run command)  |

## Connecting real data before deploying

```bash
python scripts/download_data.py     # real public data -> data/raw/
python scripts/prepare_data.py      # -> data/processed/matches.csv (auto-used)
```

Note: `data/processed/` and `data/raw/*.csv` are git-ignored, so for a cloud
deployment either commit a prepared dataset deliberately, bake it into the Docker
image, or run the prepare step at startup.

## Performance notes

- The model trains in seconds on the sample data and is cached in the app
  (`st.cache_resource`) and API (`lru_cache`) so it trains once per process.
- For large real datasets, pre-train and persist an artifact
  (`python src/main.py --save-model`) and load it with `MatchPredictor.load()`.
