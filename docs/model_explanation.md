# How the model works

This document explains the maths and machine learning behind a prediction, in
plain language. Every number the app shows traces back to the match data and the
formulas below — **nothing is random**.

A prediction is an **ensemble** of three components whose Win/Draw/Loss
probabilities are blended (weights in [`src/config.py`](../src/config.py),
`BLEND_WEIGHTS`) and re-normalised.

---

## 1. Elo rating system (`EloRatingSystem`)

Elo measures **long-run team strength** with a single number. Every team starts at
`1500`. After each match, the winner takes points from the loser:

```
expected_home = 1 / (1 + 10 ** ((rating_away - rating_home - home_adv) / 400))
new_rating    = rating + K * weight * margin * (actual - expected)
```

- `actual` is `1` for a win, `0.5` for a draw, `0` for a loss.
- `home_adv` (≈65 pts) is added to the home side, and skipped at neutral venues.
- `weight` makes important competitions count more (World Cup > friendly).
- `margin` lets bigger winning margins move ratings a little more
  (`log(|goal_diff| + 1) + 1`).

Because ratings are updated **in chronological order**, we also record each game's
**pre-match** ratings. Feature engineering uses those so the ML model never "sees
the future".

---

## 2. Poisson goal model (`PoissonGoalModel`)

Goals in football are well approximated by a **Poisson distribution**. We first
estimate each team's **attack** and **defence** strength relative to the league
average goals per team per match (`L`):

```
attack[t]  = (avg goals team t scores)   / L      # 1.0 == average
defence[t] = (avg goals team t concedes) / L      # lower is better
```

For a fixture we compute expected goals (the Poisson rate `λ`):

```
λ_home = attack[home] * defence[away] * L   (× home multiplier if not neutral)
λ_away = attack[away] * defence[home] * L
```

A small **Elo adjustment** then nudges the stronger team's `λ` up and the weaker
team's down (split with a square root so total goals stay stable).

With `λ_home` and `λ_away` we build a **scoreline grid** — the probability of every
exact score `i–j` up to 8 goals each:

```
P(i, j) = Poisson(i; λ_home) * Poisson(j; λ_away)
```

Summing the grid gives the outcome probabilities and the most likely scorelines:

- **Home win** = sum below the diagonal (home goals > away goals)
- **Draw** = sum on the diagonal
- **Away win** = sum above the diagonal

---

## 3. Machine-learning classifier (`ResultClassifier`)

A **gradient-boosting** classifier (scikit-learn by default, or XGBoost/LightGBM if
installed) learns Win/Draw/Loss from engineered features built for every historical
match (see [`feature_engineering.py`](../src/feature_engineering.py)):

| Feature | Meaning |
|---|---|
| `elo_diff` | pre-match Elo gap |
| `form_points_diff` | recent points-per-game gap (last N games) |
| `form_gd_diff` | recent goal-difference gap |
| `attack_diff` / `defence_diff` | recent scoring / conceding gaps |
| `sos_diff` | strength of schedule (avg opponent Elo) |
| `h2h_home` | head-to-head balance for the home team |
| `rest_diff` | rest-days gap |
| `neutral` | neutral venue flag |
| `stage_importance` | tournament-stage weight |

The classifier captures interactions the formulas miss (e.g. "good form *and* a
rested squad").

---

## 4. Blending, confidence and knockouts

**Blend.** The three probability sets are combined with the configured weights and
re-normalised so they sum to 100%.

**Confidence** (0–100) is built from three transparent ingredients:
- *decisiveness* — how far the top probability sits above a coin toss,
- *agreement* — how closely the Poisson and ML heads agree,
- *data depth* — how much history both teams have.

It is a **heuristic guide**, not a calibrated certainty.

**Knockout stages.** A draw after 90 minutes is resolved by extra time/penalties.
We split the draw probability into "who advances" using a mostly-coin-toss model
nudged by Elo (and overridable with a real penalty-shootout record via the
`context` argument).

**Tournament simulation.** `simulate_tournament` runs a single-elimination bracket
thousands of times using cached pairwise advance probabilities to estimate each
team's title chance (Monte-Carlo).

---

## Honest limitations

- Default data is **synthetic**; real forecasting needs real data.
- Goals are assumed independent across teams (a real Dixon–Coles model adds a
  low-score correction — a good future improvement).
- Probabilities are **not** formally calibrated or back-tested here. Adding
  log-loss/Brier back-testing is on the roadmap.
