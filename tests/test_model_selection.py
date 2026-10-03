import pytest

from pitman_yor_dp import (
    compare_count_models,
    comparison_to_dict,
    fit_negative_binomial_count_model,
    fit_poisson_count_model,
)


def test_poisson_fit_uses_sample_mean_rate():
    fit = fit_poisson_count_model([0, 1, 2, 3])

    assert fit.model_name == "poisson"
    assert fit.parameters["rate"] == pytest.approx(1.5)
    assert fit.feasible
    assert fit.aic > 0.0


def test_negative_binomial_fit_is_infeasible_without_overdispersion():
    fit = fit_negative_binomial_count_model([1, 1, 1, 1])

    assert fit.model_name == "negative_binomial"
    assert not fit.feasible
    assert fit.aic == float("inf")


def test_negative_binomial_fit_handles_overdispersed_counts():
    fit = fit_negative_binomial_count_model([0, 0, 1, 5, 8, 13])

    assert fit.feasible
    assert fit.parameters["mean"] > 0.0
    assert fit.parameters["size"] > 0.0
    assert 0.0 < fit.parameters["success_probability"] < 1.0


def test_compare_count_models_recommends_overdispersed_family():
    comparison = compare_count_models(
        [0, 0, 1, 5, 8, 13],
        count_name="birth_count",
        heavy_tail_index_threshold=0.1,
    )

    assert comparison.count_name == "birth_count"
    assert comparison.recommended_family == "gamma_poisson_or_negative_binomial"
    assert comparison.best_fit is None
    assert comparison.fits[1].aic is None
    serialized = comparison_to_dict(comparison)
    assert serialized["count_name"] == "birth_count"
    assert serialized["recommended_family"] == "gamma_poisson_or_negative_binomial"


def test_tail_heuristic_does_not_select_a_partition_prior_or_lifetime_model():
    comparison = compare_count_models(
        [1, 1, 1, 2, 3, 5, 8, 13, 21, 34],
        count_name="lifetime",
        heavy_tail_top_k=3,
        heavy_tail_index_threshold=5.0,
    )

    assert comparison.recommended_family == "boundary_aware_survival_analysis"
    assert comparison.best_fit is None
    assert comparison.fits == ()
    assert comparison_to_dict(comparison)["tail_evidence"] == "not_tested"


def test_underdispersion_does_not_recommend_poisson_or_dp():
    comparison = compare_count_models([10, 10, 11, 10], count_name="N_t")
    assert comparison.recommended_family == "underdispersed_count_baselines"


def test_zero_counts_have_no_automatic_winner_and_serialize_as_strict_json():
    import json

    comparison = compare_count_models([0, 0, 0])
    assert comparison.best_fit is None
    assert comparison.recommended_family == "insufficient_variation"
    json.dumps(comparison_to_dict(comparison), allow_nan=False)


def test_heavy_tail_arguments_cannot_trigger_pyp_recommendation():
    result = compare_count_models([1, 1, 2, 3, 50, 1000], heavy_tail_index_threshold=100)
    assert result.recommended_family == "gamma_poisson_or_negative_binomial"
    assert comparison_to_dict(result)["tail_evidence"] == "not_tested"


def test_compare_count_models_rejects_empty_samples():
    with pytest.raises(ValueError):
        compare_count_models([])
