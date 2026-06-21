# 🚀 Quickstart

Get the World Cup Match Predictor running in ~2 minutes. It works out of the box on
bundled **sample data** — no API keys or downloads required.

> Run every command from the **project root** (the folder that contains `app/`,
> `src/` and `requirements.txt`).

## 1. Check your Python

You need **Python 3.9 or newer**.

```bash
python --version        # Windows
python3 --version       # macOS / Linux
```

## 2. Create and activate a virtual environment

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (cmd.exe)**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should now see `(.venv)` at the start of your prompt.

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Run it

| What | Command |
|------|---------|
| 🖥️ Dashboard | `streamlit run app/streamlit_app.py` |
| ⌨️ CLI prediction | `python src/main.py --home Brazil --away Germany --stage Final --neutral` |
| 📈 Backtest | `python scripts/run_backtest.py` |
| ✅ Tests | `pytest` |

The dashboard opens at <http://localhost:8501>. Press `Ctrl+C` in the terminal to stop it.

Prefer shortcuts? With `make` installed: `make app`, `make cli`, `make backtest`, `make test`.

---

## 🧯 Troubleshooting

**`streamlit: command not found` (or `'streamlit' is not recognized`)**
The virtual environment isn't active, or its scripts aren't on PATH. Either activate
the venv (step 2) or run it via Python directly:
```bash
python -m streamlit run app/streamlit_app.py
```

**`ModuleNotFoundError` / missing dependencies**
The venv isn't active or dependencies aren't installed. Activate it (step 2) and:
```bash
pip install -r requirements.txt
```

**Python version errors (e.g. syntax errors on install)**
You're on Python < 3.9. Check with `python --version` and install a newer Python from
<https://www.python.org/downloads/>. On macOS/Linux use `python3` explicitly.

**`FileNotFoundError` / "Match data not found" / empty data**
You're either in the wrong folder or the sample data is missing. Run from the project
root, and regenerate the sample data if needed:
```bash
python data/generate_sample_data.py
```

**Running from the wrong folder**
All commands assume the **project root**. If you see import errors or missing files,
`cd` into the `world-cup-predictor` directory first.

**Port 8501 already in use**
Another Streamlit app is running. Use a different port:
```bash
streamlit run app/streamlit_app.py --server.port 8502
```

**(Optional) Use the REST API**
```bash
pip install -e ".[api]"
uvicorn api.main:app --reload --port 8000     # docs at http://localhost:8000/docs
```

Still stuck? Open an issue on the repository with the exact command and full error text.
