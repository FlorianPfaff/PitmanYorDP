# PitmanYorDP

Research code for Dirichlet-process and Pitman--Yor-process cardinality priors in multitarget tracking.

The goal is to keep reusable implementation code here, while notes, figures, and paper text live in [`FlorianPfaff/2026-07-PitmanYorDP-Paper`](https://github.com/FlorianPfaff/2026-07-PitmanYorDP-Paper).

## Scope

This repository provides a small installable Python package, `pitman_yor_dp`, for diagnostics and prior-predictive work around tracking counts. It is deliberately not a full multitarget tracker. The intended use is to decide which count in the tracking generative model plausibly needs a heavy-tailed prior before embedding the resulting prior terms in a tracker-specific likelihood, survival, birth, missed-detection, and clutter model.

The current code separates five concerns:

- `cardinality.py`: Dirichlet-process and Pitman--Yor Chinese-restaurant prior calculations for target-generated cluster counts.
- `diagnostics.py`: empirical diagnostics for live counts, birth counts, track lifetimes, survival curves, overdispersion, and exploratory tail-index estimates.
- `adapters.py`: generic conversion from frame-level track tables to `N_t`, `B_t`, and `L_k` count series.
- `model_selection.py`: Poisson versus negative-binomial screening and heuristic prior-family recommendations.
- `clutter.py`: explicit new-target-versus-clutter odds diagnostics for the singleton-track confound.

## Installation

```bash
python -m pip install -e ".[dev]"
```

## Example: prior predictive comparison

```python
from pitman_yor_dp import DirichletProcessCardinalityPrior, PitmanYorCardinalityPrior

dp = DirichletProcessCardinalityPrior(strength=1.0)
pyp = PitmanYorCardinalityPrior(strength=1.0, discount=0.5)

print(dp.predictive_probabilities((3, 1)))
print(pyp.predictive_probabilities((3, 1)))

print(dp.expected_number_of_clusters(20))
print(pyp.expected_number_of_clusters(20))
```

For current cluster sizes `n_1, ..., n_K`, the Pitman--Yor CRP predictive probabilities are

```text
P(existing k) = (n_k - d) / (theta + n)
P(new cluster) = (theta + d K) / (theta + n)
```

where `n = sum_k n_k`, `theta` is the strength parameter, and `d` is the discount parameter. The Dirichlet process is the `d = 0` special case.

## Example: diagnostics-first workflow

```python
from pitman_yor_dp import (
    PitmanYorCardinalityPrior,
    birth_counts_from_label_sets,
    compare_count_models,
    count_series_diagnostics,
    hill_tail_index,
    lifetimes_from_label_sets,
    new_target_vs_clutter_odds,
)

frames = [{"a", "b"}, {"a", "b", "c"}, {"b", "c"}, {"c", "d"}]

birth_counts = birth_counts_from_label_sets(frames)
lifetimes = lifetimes_from_label_sets(frames)

print(count_series_diagnostics(birth_counts))
print(hill_tail_index(lifetimes, top_k=2))
print(compare_count_models(birth_counts, count_name="B_t_birth_count"))

prior = PitmanYorCardinalityPrior(strength=1.0, discount=0.5)
print(
    new_target_vs_clutter_odds(
        prior,
        exposure_count=10,
        active_clusters=4,
        birth_to_clutter_likelihood_ratio=4.0,
    )
)
```

The `exposure_count` argument is intentionally explicit: a per-scan restaurant and an across-frame restaurant imply different meanings for `n` and should not be conflated.

## Example: CSV count diagnostics

For a CSV with at least

```text
frame,track_id
```

run:

```bash
python examples/count_diagnostics_from_tracks.py path/to/tracks.csv
```

The script emits JSON containing live counts, birth counts, lifetimes, Poisson/negative-binomial fits, and a first-pass prior-family recommendation. Use benchmark-specific scripts only to convert annotations into this minimal table schema.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
```

## Repository split

- `FlorianPfaff/PitmanYorDP`: reusable code, tests, simulation utilities.
- `FlorianPfaff/2026-07-PitmanYorDP-Paper`: notes, figures, and manuscript text.
