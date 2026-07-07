import numpy as np
import pytest

from pitman_yor_dp import (
    DirichletProcessCardinalityPrior,
    PitmanYorCardinalityPrior,
    birth_counts_from_label_sets,
    count_series_diagnostics,
    hill_tail_index,
    lifetimes_from_label_sets,
    live_counts_from_label_sets,
    loglog_survival_fit,
    new_target_vs_clutter_odds,
    novelty_prior_probability,
    phase_transition_table,
    survival_function,
)


def test_count_series_diagnostics_flags_overdispersion():
    diagnostics = count_series_diagnostics([0, 0, 1, 4, 7, 12])

    assert diagnostics.num_samples == 6
    assert diagnostics.maximum == 12
    assert diagnostics.zero_fraction == pytest.approx(2.0 / 6.0)
    assert diagnostics.dispersion_index > 1.0


def test_survival_function_uses_greater_equal_tail():
    xs, survival = survival_function([0, 1, 1, 3], min_value=0)

    assert xs.tolist() == [0, 1, 2, 3]
    assert survival.tolist() == pytest.approx([1.0, 0.75, 0.25, 0.25])


def test_hill_tail_index_returns_finite_positive_estimate():
    estimate = hill_tail_index([1, 2, 3, 5, 8, 13, 21, 34], top_k=3)

    assert estimate.threshold == pytest.approx(8.0)
    assert estimate.tail_count == 3
    assert estimate.hill_gamma > 0.0
    assert estimate.tail_index > 0.0


def test_loglog_survival_fit_returns_negative_slope_for_right_tail():
    samples = np.asarray([1, 1, 2, 2, 3, 5, 8, 13, 21])

    fit = loglog_survival_fit(samples, min_value=1)

    assert fit.num_points >= 2
    assert fit.slope < 0.0
    assert fit.implied_tail_index > 0.0


def test_label_set_helpers_extract_live_birth_and_lifetime_counts():
    frames = [
        {"a", "b"},
        {"a", "b", "c"},
        {"b", "c"},
        {"c", "d"},
    ]

    assert live_counts_from_label_sets(frames) == (2, 3, 2, 2)
    assert birth_counts_from_label_sets(frames) == (2, 1, 0, 1)
    assert lifetimes_from_label_sets(frames) == (1, 2, 3, 3)


def test_novelty_prior_probability_makes_exposure_choice_explicit():
    dp = DirichletProcessCardinalityPrior(strength=1.0)
    pyp = PitmanYorCardinalityPrior(strength=1.0, discount=0.5)

    assert novelty_prior_probability(dp, exposure_count=10, active_clusters=4) == pytest.approx(1.0 / 11.0)
    assert novelty_prior_probability(pyp, exposure_count=10, active_clusters=4) == pytest.approx(3.0 / 11.0)


def test_new_target_vs_clutter_odds_can_flip_with_discount():
    dp = DirichletProcessCardinalityPrior(strength=1.0)
    pyp = PitmanYorCardinalityPrior(strength=1.0, discount=0.5)

    dp_odds = new_target_vs_clutter_odds(
        dp,
        exposure_count=10,
        active_clusters=4,
        birth_to_clutter_likelihood_ratio=4.0,
    )
    pyp_odds = new_target_vs_clutter_odds(
        pyp,
        exposure_count=10,
        active_clusters=4,
        birth_to_clutter_likelihood_ratio=4.0,
    )

    assert not dp_odds.favors_new_target
    assert pyp_odds.favors_new_target


def test_phase_transition_table_skips_impossible_cluster_counts():
    prior = PitmanYorCardinalityPrior(strength=1.0, discount=0.5)

    rows = phase_transition_table(
        prior,
        exposure_counts=(0, 1),
        active_clusters=(0, 1, 2),
        birth_to_clutter_likelihood_ratios=(0.5, 2.0),
    )

    assert len(rows) == 6
    assert all(row.active_clusters <= row.exposure_count for row in rows)


def test_invalid_tail_inputs_raise():
    with pytest.raises(ValueError):
        hill_tail_index([0, 1, 2], top_k=1)
    with pytest.raises(ValueError):
        hill_tail_index([1, 2], top_k=2)
    with pytest.raises(ValueError):
        new_target_vs_clutter_odds(
            PitmanYorCardinalityPrior(),
            exposure_count=1,
            active_clusters=1,
            birth_to_clutter_likelihood_ratio=0.0,
        )
