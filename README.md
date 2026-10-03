# PitmanYorDP

Research code for Dirichlet-process and Pitman--Yor-process cardinality priors in multitarget tracking.

The goal is to keep reusable implementation code here, while notes, figures, and paper text live in [`FlorianPfaff/2026-07-PitmanYorDP-Paper`](https://github.com/FlorianPfaff/2026-07-PitmanYorDP-Paper).

## Scope

Marginal birth, annotated-population, and observed-span statistics do not select a DP or PYP partition prior. Recommendations identify exploratory count baselines only. Negative-binomial fits currently use moments, so their AIC and mixed-estimator rankings are unavailable (`best_model` is null). Lifetimes require a separate boundary-aware analysis; a DP is not a Poisson count model.

For audited MOT17 ground truth, use:

```bash
python examples/batch_motchallenge_diagnostics.py MOT17Labels/train \
  --mot17-ground-truth --json-output report.json --output-dir reports/mot17 --quiet
```

This mode requires `seqinfo.ini`, keeps marked pedestrians at all visibility levels, preserves complete clip exposure, checks and deduplicates detector variants, and records hashes and identity boundary flags. Raw `B_t` includes the initial frame; comparisons exclude it. `L_k` is an observed span, not a complete physical lifetime. Generic input remains supported without the flag, but clip boundaries and frame rates may be unknown.

An exact structural diagnostic is available for windows of at most nine observations:

```bash
python examples/partition_structure_check.py --output partition-check.json
```

The `partitions` module conditions the full EPPF on valid point-target partitions. It verifies a two-frame equivalence after matching the entire detected-identity-count distribution and demonstrates a remaining three-frame occupancy effect. These are prior calculations, not tracker performance or live-cardinality inference.

This repository provides a small installable Python package, `pitman_yor_dp`, for descriptive diagnostics and partition-prior checks. It is deliberately not a full multitarget tracker. Tracker integration requires a normalized model of observations, survival, births, missed detections, and clutter.

The current code separates seven concerns:

- `cardinality.py`: Dirichlet-process and Pitman--Yor Chinese-restaurant prior calculations for target-generated cluster counts.
- `diagnostics.py`: empirical diagnostics for live counts, birth counts, track lifetimes, survival curves, overdispersion, and exploratory tail-index estimates.
- `adapters.py`: generic conversion from frame-level track tables to `N_t`, `B_t`, and `L_k` count series.
- `motchallenge.py`: MOTChallenge-style text-file conversion to the generic count-series schema.
- `model_selection.py`: Poisson versus negative-binomial screening and heuristic prior-family recommendations.
- `clutter.py`: explicit new-target-versus-clutter odds diagnostics for the singleton-track confound.
- `synthetic.py`: synthetic frame-level label generation for exercising the gating workflow without a benchmark file.

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

The script emits JSON containing observed counts, first appearances, identity spans, and exploratory baseline suggestions. Use the audited batch mode for MOT17 metadata and boundary handling.

## Example: MOTChallenge count diagnostics

For a MOTChallenge-style file such as `gt/gt.txt` with rows

```text
frame,id,bb_left,bb_top,bb_width,bb_height,conf,x,y,z
```

run:

```bash
python examples/count_diagnostics_from_motchallenge.py path/to/gt.txt --dataset MOT17-02 > mot_counts.json
```

Then render it into the paper repo with:

```bash
python ../2026-07-PitmanYorDP-Paper/scripts/render_count_diagnostic_tables.py empirical mot_counts.json \
  --dataset MOT17-02 \
  --output ../2026-07-PitmanYorDP-Paper/paper/tables/empirical_count_diagnostics_template.tex
```

## Example: synthetic gating smoke test

Run two built-in synthetic scenarios, one mild and one bursty/heavy-tailed:

```bash
python examples/synthetic_gating_experiment.py
```

This is useful for CI and for checking that the paper's first-stage diagnostic workflow is executable before a real benchmark adapter is added.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
```

## Repository split

- `FlorianPfaff/PitmanYorDP`: reusable code, tests, simulation utilities.
- `FlorianPfaff/2026-07-PitmanYorDP-Paper`: notes, figures, and manuscript text.
