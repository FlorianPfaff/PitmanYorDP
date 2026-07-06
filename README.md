# PitmanYorDP

Research code for Dirichlet-process and Pitman--Yor-process cardinality priors in multitarget tracking.

The goal is to keep reusable implementation code here, while notes, figures, and paper text live in [`FlorianPfaff/2026-07-PitmanYorDP-Paper`](https://github.com/FlorianPfaff/2026-07-PitmanYorDP-Paper).

## Scope

This repository currently provides a small installable Python package, `pitman_yor_dp`, for prior-predictive work around target-generated cluster counts. It is deliberately not a full multitarget tracker. The intended use is to combine these prior terms with tracker-specific likelihood, survival, birth, missed-detection, and clutter models.

## Installation

```bash
python -m pip install -e ".[dev]"
```

## Example

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

## Development

```bash
python -m pip install -e ".[dev]"
pytest
```

## Repository split

- `FlorianPfaff/PitmanYorDP`: reusable code, tests, simulation utilities.
- `FlorianPfaff/2026-07-PitmanYorDP-Paper`: notes, figures, and manuscript text.
