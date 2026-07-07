import pytest

from pitman_yor_dp import (
    SyntheticTrackingConfig,
    compare_count_models,
    simulate_count_series,
    simulate_track_records,
)


def test_deterministic_fixed_synthetic_sequence_is_predictable():
    config = SyntheticTrackingConfig(
        num_frames=4,
        birth_rate=1.0,
        birth_count_model="deterministic",
        lifetime_model="fixed",
        fixed_lifetime=2,
    )

    bundle = simulate_count_series(config)

    assert bundle.frames == (0, 1, 2, 3)
    assert bundle.birth_counts == (1, 1, 1, 1)
    assert bundle.live_counts == (1, 2, 2, 2)
    assert bundle.lifetimes == (1, 2, 2, 2)


def test_synthetic_generation_is_reproducible_with_seed():
    config = SyntheticTrackingConfig(
        num_frames=20,
        birth_rate=1.5,
        birth_count_model="negative_binomial",
        birth_overdispersion_size=0.8,
        lifetime_model="pareto",
        pareto_shape=1.5,
        seed=17,
    )

    assert simulate_track_records(config) == simulate_track_records(config)


def test_synthetic_counts_feed_model_selection():
    config = SyntheticTrackingConfig(
        num_frames=50,
        birth_rate=2.0,
        birth_count_model="negative_binomial",
        birth_overdispersion_size=0.5,
        lifetime_model="geometric",
        mean_lifetime=4.0,
        seed=3,
    )

    bundle = simulate_count_series(config)
    comparison = compare_count_models(bundle.birth_counts, count_name="synthetic_births")

    assert comparison.count_name == "synthetic_births"
    assert comparison.diagnostics.num_samples == 50
    assert comparison.best_fit is not None


def test_synthetic_config_validation():
    with pytest.raises(ValueError):
        SyntheticTrackingConfig(num_frames=0)
    with pytest.raises(ValueError):
        SyntheticTrackingConfig(birth_count_model="unsupported")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SyntheticTrackingConfig(lifetime_model="unsupported")  # type: ignore[arg-type]
