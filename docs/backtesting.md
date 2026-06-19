# Backtesting & evaluation

A model is only as trustworthy as its evaluation. This project ships a
**leak-free walk-forward backtest** ([`src/evaluation.py`](../src/evaluation.py))
so you can measure the models honestly rather than trust a marketing claim.

## How the backtest works

1. **Chronological split.** Matches are sorted by date and the most recent
   `test_fraction` (default 25%) is held out as the test set.
2. **Train on the past.** Elo, the Poisson model and the ML classifier are fitted
   on the training portion only.
3. **Predict the future, one match at a time.** For each test match, every model
   predicts using *only* information available before kick-off (history up to, but
   not including, that match). Elo is then updated online so ratings stay current.
4. **Score with proper metrics** and compare four models head-to-head.

This avoids the classic mistake of letting the model peek at the future.

## Metrics

| Metric | Meaning | Better |
|---|---|---|
| **Accuracy** | share of matches where the most likely outcome was correct | higher |
| **Log loss** | cross-entropy; punishes confident wrong probabilities | **lower** |
| **Brier score** | mean squared error of the full probability vector | **lower** |
| **Calibration** | do "70%" predictions happen ~70% of the time? | closer to diagonal |

> A useful sanity baseline: blind 1/3–1/3–1/3 guessing has a log loss of
> `ln(3) ≈ 1.0986`. Any model worth keeping should beat that.

## Running it

```bash
python scripts/run_backtest.py                       # default 25% test split
python scripts/run_backtest.py --test-fraction 0.3
python scripts/run_backtest.py --tournament "World Cup"
python scripts/run_backtest.py --start-date 2018-01-01 --end-date 2022-12-31
```

Outputs are written to:
- `data/processed/backtest_results.csv` — per-match predictions for every model
- `data/processed/backtest_comparison.csv` — the summary comparison table

You can also explore results interactively in the **Backtesting** page of the
Streamlit app.

## Example output (bundled synthetic sample data)

Running `python scripts/run_backtest.py` on the bundled sample data
(1,380 train / 460 test matches) produces results in this ballpark:

| model     | accuracy | log_loss | brier  |
|-----------|---------:|---------:|-------:|
| ensemble  |   0.483  |  1.0325  | 0.6186 |
| poisson   |   0.500  |  1.0392  | 0.6224 |
| elo       |   0.476  |  1.0687  | 0.6432 |
| ml        |   0.454  |  1.0713  | 0.6412 |

**How to read this honestly:**
- The **ensemble has the best (lowest) log loss**, beating the `ln(3) ≈ 1.0986`
  uniform baseline — i.e. its probabilities are better calibrated than guessing.
- Accuracy in the high-40s/50% is expected for a 3-way outcome with lots of
  draws; football is genuinely hard to predict.
- These numbers describe the **synthetic sample data only**. They are *not* a
  claim about real-world performance — connect real data
  (see [`data_sources.md`](data_sources.md)) and re-run to get meaningful figures.

## Limitations of this backtest

- The Poisson and ML models are fitted once on the training split (only Elo
  updates online). A fully online refit each step would be more rigorous but much
  slower; this is a deliberate, documented trade-off.
- Rest-day and stage features are simplified at prediction time (the deployed
  model rarely knows a future fixture's exact rest days).
- No probability re-calibration (Platt/isotonic) is applied yet — a good next step.
