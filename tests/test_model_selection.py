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
    assert comparison.best_fit is not None
    serialized = comparison_to_dict(comparison)
    assert serialized["count_name"] == "birth_count"
    assert serialized["recommended_family"] == "gamma_poisson_or_negative_binomial"


def test_compare_count_models_can_flag_possible_heavy_tail():
    comparison = compare_count_models(
        [1, 1, 1, 2, 3, 5, 8, 13, 21, 34],
        count_name="lifetime",
        heavy_tail_top_k=3,
        heavy_tail_index_threshold=5.0,
    )

    assert comparison.recommended_family == "pitman_yor_or_stable_process"
    assert "tail index" in comparison.recommendation_reason


def test_compare_count_models_rejects_empty_samples():
    with pytest.raises(ValueError):
        compare_count_models([])
