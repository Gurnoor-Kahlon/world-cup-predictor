# Convenience commands. Run `make help` to list them.
# (On Windows, use Git Bash / WSL, or run the underlying commands directly.)

PYTHON ?= python

.PHONY: help install install-dev test lint format typecheck app cli api backtest data sample model docker clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime dependencies
	$(PYTHON) -m pip install -r requirements.txt

install-dev:  ## Install runtime + dev + api extras (editable)
	$(PYTHON) -m pip install -e ".[dev,api]"

test:  ## Run the test suite
	$(PYTHON) -m pytest -q

lint:  ## Lint with ruff
	$(PYTHON) -m ruff check .

format:  ## Auto-fix lint issues with ruff
	$(PYTHON) -m ruff check . --fix

typecheck:  ## Type-check src/ with mypy
	$(PYTHON) -m mypy src

app:  ## Launch the Streamlit dashboard
	$(PYTHON) -m streamlit run app/streamlit_app.py

cli:  ## Run an example CLI prediction (Brazil vs Germany, final)
	$(PYTHON) src/main.py --home Brazil --away Germany --stage Final --neutral

api:  ## Launch the FastAPI server (needs the api extra)
	$(PYTHON) -m uvicorn api.main:app --reload --port 8000

backtest:  ## Run the walk-forward backtest
	$(PYTHON) scripts/run_backtest.py

data:  ## Download + prepare real data into data/processed/
	$(PYTHON) scripts/download_data.py && $(PYTHON) scripts/prepare_data.py

sample:  ## Regenerate the synthetic sample dataset
	$(PYTHON) data/generate_sample_data.py

model:  ## Train and save a model artifact to models/
	$(PYTHON) src/main.py --save-model

docker:  ## Build the Docker image
	docker build -t world-cup-predictor .

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__ models
