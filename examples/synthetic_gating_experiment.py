"""Run the diagnostics-first workflow on synthetic tracking labels.

This example provides a zero-data smoke test for the paper's first gating
experiment. It compares a mild Poisson/geometric scenario with a more bursty
negative-binomial/Pareto scenario.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from pitman_yor_dp import (
    SyntheticTrackingConfig,
    compare_count_models,
    comparison_to_dict,
    simulate_count_series,
)


SCENARIOS = {
    "mild_poisson_geometric": SyntheticTrackingConfig(
        num_frames=120,
        birth_rate=1.0,
        birth_count_model="poisson",
        lifetime_model="geometric",
        mean_lifetime=8.0,
        seed=11,
    ),
    "bursty_nb_pareto": SyntheticTrackingConfig(
        num_frames=120,
        birth_rate=1.0,
        birth_count_model="negative_binomial",
        birth_overdispersion_size=0.4,
        lifetime_model="pareto",
        pareto_shape=1.4,
        min_lifetime=2,
        seed=11,
    ),
}


def _scenario_summary(name: str, config: SyntheticTrackingConfig) -> dict[str, object]:
    bundle = simulate_count_series(config)
    comparisons = (
        compare_count_models(bundle.live_counts, count_name="N_t_live_count"),
        compare_count_models(bundle.birth_counts, count_name="B_t_birth_count"),
        compare_count_models(bundle.lifetimes, count_name="L_k_lifetime"),
    )
    return {
        "scenario": name,
        "config": asdict(config),
        "num_tracks": len(bundle.lifetimes),
        "count_model_comparisons": [comparison_to_dict(comparison) for comparison in comparisons],
    }


def main() -> None:
    summaries = [_scenario_summary(name, config) for name, config in SCENARIOS.items()]
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
