"""Simple empirical model-selection utilities for tracking count series.

These utilities are deliberately modest. They are meant to answer the first
screening question before tracker integration: are observed tracking counts
Poisson-like, merely overdispersed, or plausibly heavy-tailed?
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, lgamma, log
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike

from .diagnostics import CountSeriesDiagnostics, count_series_diagnostics, hill_tail_index


@dataclass(frozen=True)
class CountModelFit:
    """Maximum-likelihood-style fit summary for one count model."""

    model_name: str
    log_likelihood: float
    aic: float
    num_parameters: int
    parameters: Mapping[str, float]
    feasible: bool = True
    note: str = ""


@dataclass(frozen=True)
class CountModelComparison:
    """Comparison of simple count models for one empirical count series."""

    count_name: str
    diagnostics: CountSeriesDiagnostics
    fits: tuple[CountModelFit, ...]
    recommended_family: str
    recommendation_reason: str

    @property
    def best_fit(self) -> CountModelFit | None:
        """Return the feasible fit with the smallest AIC, if any."""

        feasible = [fit for fit in self.fits if fit.feasible]
        if not feasible:
            return None
        return min(feasible, key=lambda fit: fit.aic)


def fit_poisson_count_model(samples: ArrayLike) -> CountModelFit:
    """Fit a Poisson count model by maximum likelihood."""

    counts = _as_nonnegative_integer_array(samples, name="samples")
    if counts.size == 0:
        raise ValueError("samples must not be empty.")
    rate = float(np.mean(counts))
    log_likelihood = float(sum(_poisson_logpmf(int(count), rate) for count in counts))
    return CountModelFit(
        model_name="poisson",
        log_likelihood=log_likelihood,
        aic=_aic(log_likelihood, num_parameters=1),
        num_parameters=1,
        parameters={"rate": rate},
    )


def fit_negative_binomial_count_model(samples: ArrayLike) -> CountModelFit:
    """Fit a method-of-moments negative-binomial model.

    The parameterization uses mean ``mu`` and size ``r`` with
    ``Var[X] = mu + mu^2 / r``. If the empirical variance does not exceed the
    mean, the returned fit is marked infeasible because the negative-binomial
    overdispersion parameter is not identified by moments.
    """

    counts = _as_nonnegative_integer_array(samples, name="samples")
    if counts.size == 0:
        raise ValueError("samples must not be empty.")
    mean = float(np.mean(counts))
    variance = float(np.var(counts, ddof=1)) if counts.size > 1 else 0.0
    if mean <= 0.0:
        return CountModelFit(
            model_name="negative_binomial",
            log_likelihood=float("nan"),
            aic=float("inf"),
            num_parameters=2,
            parameters={"mean": mean, "size": float("inf"), "success_probability": 1.0},
            feasible=False,
            note="negative binomial is not informative when all counts are zero.",
        )
    if variance <= mean:
        return CountModelFit(
            model_name="negative_binomial",
            log_likelihood=float("nan"),
            aic=float("inf"),
            num_parameters=2,
            parameters={"mean": mean, "size": float("inf"), "success_probability": 1.0},
            feasible=False,
            note="empirical variance is not larger than the mean; Poisson-like counts do not identify overdispersion.",
        )

    size = mean * mean / (variance - mean)
    success_probability = size / (size + mean)
    log_likelihood = float(sum(_negative_binomial_logpmf(int(count), size, success_probability) for count in counts))
    return CountModelFit(
        model_name="negative_binomial",
        log_likelihood=log_likelihood,
        aic=_aic(log_likelihood, num_parameters=2),
        num_parameters=2,
        parameters={
            "mean": mean,
            "size": float(size),
            "success_probability": float(success_probability),
        },
    )


def compare_count_models(
    samples: ArrayLike,
    *,
    count_name: str = "count",
    dispersion_threshold: float = 1.25,
    heavy_tail_top_k: int | None = None,
    heavy_tail_index_threshold: float = 3.0,
) -> CountModelComparison:
    """Compare simple count baselines and return a prior-family recommendation.

    The recommendation is heuristic and should be treated as a screening result:

    - low dispersion: ordinary Poisson/RFS cardinality may be enough;
    - overdispersion without strong tail evidence: gamma-Poisson or negative
      binomial is the first baseline;
    - small Hill tail index: Pitman--Yor/stable-process motivation remains
      plausible and deserves a tracker-level test.
    """

    counts = _as_nonnegative_integer_array(samples, name="samples")
    if counts.size == 0:
        raise ValueError("samples must not be empty.")
    diagnostics = count_series_diagnostics(counts)
    fits = [fit_poisson_count_model(counts), fit_negative_binomial_count_model(counts)]

    tail_index: float | None = None
    positive_counts = counts[counts > 0]
    if positive_counts.size >= 4 and int(np.max(positive_counts)) > int(np.min(positive_counts)):
        try:
            tail_index = hill_tail_index(positive_counts, top_k=heavy_tail_top_k).tail_index
        except ValueError:
            tail_index = None

    if tail_index is not None and tail_index <= float(heavy_tail_index_threshold) and diagnostics.maximum >= 4:
        recommended_family = "pitman_yor_or_stable_process"
        reason = f"exploratory tail index {tail_index:.3g} is below threshold {heavy_tail_index_threshold:.3g}; verify with survival plots and clutter controls."
    elif isfinite(diagnostics.dispersion_index) and diagnostics.dispersion_index > float(dispersion_threshold):
        recommended_family = "gamma_poisson_or_negative_binomial"
        reason = f"dispersion index {diagnostics.dispersion_index:.3g} exceeds threshold {dispersion_threshold:.3g}, but heavy-tail evidence is weak or unavailable."
    else:
        recommended_family = "poisson_or_dirichlet_process_baseline"
        reason = f"dispersion index {diagnostics.dispersion_index:.3g} does not exceed threshold {dispersion_threshold:.3g}."

    sorted_fits = tuple(sorted(fits, key=lambda fit: (not fit.feasible, fit.aic)))
    return CountModelComparison(
        count_name=str(count_name),
        diagnostics=diagnostics,
        fits=sorted_fits,
        recommended_family=recommended_family,
        recommendation_reason=reason,
    )


def comparison_to_dict(comparison: CountModelComparison) -> dict[str, object]:
    """Serialize a count-model comparison to a JSON/CSV-friendly dictionary."""

    best_fit = comparison.best_fit
    return {
        "count_name": comparison.count_name,
        "num_samples": comparison.diagnostics.num_samples,
        "mean": comparison.diagnostics.mean,
        "variance": comparison.diagnostics.variance,
        "dispersion_index": comparison.diagnostics.dispersion_index,
        "zero_fraction": comparison.diagnostics.zero_fraction,
        "maximum": comparison.diagnostics.maximum,
        "best_model": None if best_fit is None else best_fit.model_name,
        "best_model_aic": None if best_fit is None else best_fit.aic,
        "recommended_family": comparison.recommended_family,
        "recommendation_reason": comparison.recommendation_reason,
    }


def comparisons_to_dicts(comparisons: tuple[CountModelComparison, ...]) -> tuple[dict[str, object], ...]:
    """Serialize multiple count-model comparisons."""

    return tuple(comparison_to_dict(comparison) for comparison in comparisons)


def _poisson_logpmf(count: int, rate: float) -> float:
    if rate < 0.0:
        raise ValueError("rate must be nonnegative.")
    if rate == 0.0:
        return 0.0 if count == 0 else float("-inf")
    return float(count * log(rate) - rate - lgamma(count + 1.0))


def _negative_binomial_logpmf(count: int, size: float, success_probability: float) -> float:
    if size <= 0.0 or not isfinite(size):
        raise ValueError("size must be positive and finite.")
    if not 0.0 < success_probability < 1.0:
        raise ValueError("success_probability must be in (0, 1).")
    return float(
        lgamma(count + size)
        - lgamma(size)
        - lgamma(count + 1.0)
        + size * log(success_probability)
        + count * log1p_negative(success_probability)
    )


def log1p_negative(probability: float) -> float:
    """Return ``log(1 - probability)`` with validation."""

    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be in (0, 1).")
    return float(log(1.0 - probability))


def _aic(log_likelihood: float, *, num_parameters: int) -> float:
    return float(2 * int(num_parameters) - 2.0 * float(log_likelihood))


def _as_nonnegative_integer_array(samples: ArrayLike, *, name: str) -> np.ndarray:
    values = np.asarray(samples)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if values.size == 0:
        return values.astype(int)
    if np.any(values < 0):
        raise ValueError(f"{name} must be nonnegative.")
    if not np.all(values == np.floor(values)):
        raise ValueError(f"{name} must contain integer values.")
    return values.astype(int)
