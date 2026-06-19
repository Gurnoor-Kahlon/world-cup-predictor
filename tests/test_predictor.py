"""Tests for the Elo/Poisson engines and the end-to-end MatchPredictor."""

import pytest

from model import EloRatingSystem, PoissonGoalModel


# --------------------------------------------------------------------------- #
# Elo
# --------------------------------------------------------------------------- #
def test_elo_ranks_stronger_team_higher(synthetic_matches):
    elo = EloRatingSystem().fit(synthetic_matches)
    assert elo.get_rating("Strong") > elo.get_rating("Mid") > elo.get_rating("Weak")


def test_elo_history_aligned(synthetic_matches):
    elo = EloRatingSystem().fit(synthetic_matches)
    assert len(elo.history) == len(synthetic_matches)
    # First-ever ratings should be the initial value.
    assert elo.history[0]["home_elo"] == pytest.approx(elo.initial)


def test_elo_match_probabilities_sum_to_one(synthetic_matches):
    elo = EloRatingSystem().fit(synthetic_matches)
    probs = elo.match_probabilities("Strong", "Weak")
    assert sum(probs) == pytest.approx(1.0, abs=1e-9)
    # Strong at home vs Weak should be the most likely outcome.
    assert probs[0] > probs[2]


# --------------------------------------------------------------------------- #
# Poisson
# --------------------------------------------------------------------------- #
def test_poisson_result_probabilities_sum_to_one(synthetic_matches):
    pois = PoissonGoalModel().fit(synthetic_matches)
    lam_h, lam_a = pois.expected_goals("Strong", "Weak", neutral=True, elo_diff=120)
    grid = pois.scoreline_grid(lam_h, lam_a)
    assert grid.sum() == pytest.approx(1.0, abs=1e-9)
    probs = pois.result_probabilities(grid)
    assert sum(probs) == pytest.approx(1.0, abs=1e-9)


def test_poisson_stronger_team_scores_more(synthetic_matches):
    pois = PoissonGoalModel().fit(synthetic_matches)
    lam_h, lam_a = pois.expected_goals("Strong", "Weak", neutral=True, elo_diff=0)
    assert lam_h > lam_a


def test_top_scorelines_sorted_and_bounded(synthetic_matches):
    pois = PoissonGoalModel().fit(synthetic_matches)
    grid = pois.scoreline_grid(1.5, 1.1)
    top = pois.top_scorelines(grid, n=5)
    assert len(top) == 5
    probs = [p for _, _, p in top]
    assert probs == sorted(probs, reverse=True)   # descending
    assert all(0 <= p <= 1 for p in probs)


def test_dixon_coles_changes_low_scores_but_keeps_valid_grid(synthetic_matches):
    pois = PoissonGoalModel().fit(synthetic_matches)
    plain = pois.scoreline_grid(1.3, 1.1, dixon_coles=False)
    dc = pois.scoreline_grid(1.3, 1.1, dixon_coles=True)
    # Both remain valid probability grids.
    assert plain.sum() == pytest.approx(1.0, abs=1e-9)
    assert dc.sum() == pytest.approx(1.0, abs=1e-9)
    assert (dc >= 0).all()
    # The correction actually moves the 1-1 cell (rho != 0).
    assert dc[1, 1] != pytest.approx(plain[1, 1], abs=1e-6)


# --------------------------------------------------------------------------- #
# End-to-end MatchPredictor
# --------------------------------------------------------------------------- #
def test_prediction_probabilities_sum_to_one(fitted_predictor):
    res = fitted_predictor.predict("Brazil", "Germany", stage="Group")
    p = res["probabilities"]
    assert sum(p.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(0 <= v <= 1 for v in p.values())


def test_prediction_has_expected_keys(fitted_predictor):
    res = fitted_predictor.predict("Brazil", "Germany")
    for key in ("probabilities", "expected_goals", "top_scorelines",
                "confidence", "explanation", "key_factors", "elo"):
        assert key in res
    assert res["expected_goals"]["home"] > 0
    assert 0 <= res["confidence"]["score"] <= 100
    assert res["confidence"]["label"] in {"Low", "Medium", "High"}


def test_knockout_advance_probabilities(fitted_predictor):
    res = fitted_predictor.predict("Brazil", "Germany", stage="Final", neutral=True)
    assert "knockout" in res
    k = res["knockout"]
    # Someone always advances: the two advance probabilities sum to 1.
    assert k["home_advance"] + k["away_advance"] == pytest.approx(1.0, abs=1e-6)


def test_group_stage_has_no_knockout(fitted_predictor):
    res = fitted_predictor.predict("Brazil", "Germany", stage="Group")
    assert "knockout" not in res


def test_stronger_team_favoured(synthetic_predictor):
    res = synthetic_predictor.predict("Strong", "Weak", stage="Group", neutral=True)
    p = res["probabilities"]
    assert p["home_win"] > p["away_win"]


def test_context_injuries_reduce_expected_goals(fitted_predictor):
    base = fitted_predictor.predict("Brazil", "Germany", stage="Group")
    weakened = fitted_predictor.predict("Brazil", "Germany", stage="Group",
                                        context={"home_injuries": 4})
    assert weakened["expected_goals"]["home"] < base["expected_goals"]["home"]


def test_unknown_team_does_not_crash(fitted_predictor):
    res = fitted_predictor.predict("Atlantis", "Germany", stage="Group")
    assert sum(res["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)


def test_simulate_tournament_sums_to_one(fitted_predictor):
    odds = fitted_predictor.simulate_tournament(
        ["Brazil", "Germany", "France", "Spain"], n_sims=1000, seed=1)
    assert sum(odds.values()) == pytest.approx(1.0, abs=1e-9)
    assert all(0.0 <= v <= 1.0 for v in odds.values())


def test_simulate_tournament_rejects_non_power_of_two(synthetic_predictor):
    with pytest.raises(ValueError):
        synthetic_predictor.simulate_tournament(["Strong", "Weak", "Mid"], n_sims=10)


def test_configurable_blend_weights_affect_output(fitted_predictor, sample_matches):
    from predictor import MatchPredictor
    elo_heavy = MatchPredictor(matches=sample_matches,
                               blend_weights={"poisson": 0.0, "elo": 1.0, "ml": 0.0}).fit()
    base = fitted_predictor.predict("Brazil", "Germany")["probabilities"]
    tilted = elo_heavy.predict("Brazil", "Germany")["probabilities"]
    # Different weights should generally produce different probabilities.
    assert base != tilted


def test_save_and_load_round_trip(fitted_predictor, tmp_path):
    from predictor import MatchPredictor
    path = tmp_path / "model.joblib"
    fitted_predictor.save(path)
    assert path.exists()
    loaded = MatchPredictor.load(path)
    before = fitted_predictor.predict("Brazil", "Germany")["probabilities"]
    after = loaded.predict("Brazil", "Germany")["probabilities"]
    # A reloaded model reproduces the same prediction.
    assert before == after
