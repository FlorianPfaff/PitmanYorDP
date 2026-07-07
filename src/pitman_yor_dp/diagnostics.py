"""Diagnostics for deciding which tracking count should be heavy-tailed.

The functions here are intentionally data-facing rather than tracker-specific.
They help summarize live-cardinality, birth-count, clutter-count, and track-
lifetime samples before choosing between DP, Pitman--Yor, gamma-Poisson, or
beta-Bernoulli style priors.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class CountSeriesDiagnostics:
    """Basic diagnostics for nonnegative count samples."""

    num_samples: int
    mean: float
    variance: float
    dispersion_index: float
    zero_fraction: float
    maximum: int


@dataclass(frozen=True)
class TailIndexEstimate:
    """Hill-style estimate for a positive heavy-tail index.

    ``tail_index`` corresponds to the Pareto exponent ``alpha`` in
    ``P(X >= x) approximately x^{-alpha}``. The estimate is exploratory: for
    discrete tracking counts and short series, it should be used as a screening
    diagnostic rather than as a definitive model-selection criterion.
    """

    threshold: float
    tail_count: int
    hill_gamma: float
    tail_index: float


@dataclass(frozen=True)
class LogLogSurvivalFit:
    """Log-log survival regression diagnostic."""

    slope: float
    intercept: float
    implied_tail_index: float
    num_points: int
    min_value: float


def count_series_diagnostics(samples: ArrayLike) -> CountSeriesDiagnostics:
    """Return mean/variance diagnostics for nonnegative integer count samples."""

    counts = _as_nonnegative_integer_array(samples, name="samples")
    if counts.size == 0:
        raise ValueError("samples must not be empty.")
    mean = float(np.mean(counts))
    variance = float(np.var(counts, ddof=1)) if counts.size > 1 else 0.0
    dispersion_index = float(variance / mean) if mean > 0.0 else float("nan")
    zero_fraction = float(np.mean(counts == 0))
    return CountSeriesDiagnostics(
        num_samples=int(counts.size),
        mean=mean,
        variance=variance,
        dispersion_index=dispersion_index,
        zero_fraction=zero_fraction,
        maximum=int(np.max(counts)),
    )


def survival_function(samples: ArrayLike, *, min_value: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return empirical survival values ``P(X >= x)`` for integer thresholds."""

    counts = _as_nonnegative_integer_array(samples, name="samples")
    if counts.size == 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=float)
    if min_value is None:
        min_value = int(np.min(counts))
    min_value = _validate_nonnegative_int(min_value, "min_value")
    xs = np.arange(min_value, int(np.max(counts)) + 1, dtype=int)
    survival = np.asarray([np.mean(counts >= x) for x in xs], dtype=float)
    return xs, survival


def hill_tail_index(samples: ArrayLike, *, threshold: float | None = None, top_k: int | None = None) -> TailIndexEstimate:
    """Estimate a Pareto tail exponent with the Hill estimator.

    Provide either an explicit positive ``threshold`` or ``top_k``. If neither is
    supplied, ``top_k`` defaults to ``sqrt(n)`` on the positive samples. The
    estimator uses values strictly above the threshold.
    """

    values = _as_positive_float_array(samples, name="samples")
    if values.size < 2:
        raise ValueError("at least two positive samples are required.")
    if threshold is not None and top_k is not None:
        raise ValueError("provide either threshold or top_k, not both.")

    sorted_values = np.sort(values)
    if threshold is None:
        if top_k is None:
            top_k = max(1, int(np.sqrt(values.size)))
        top_k = _validate_positive_int(top_k, "top_k")
        if top_k >= values.size:
            raise ValueError("top_k must be smaller than the number of positive samples.")
        threshold = float(sorted_values[-top_k - 1])
    else:
        threshold = float(threshold)
        if not isfinite(threshold) or threshold <= 0.0:
            raise ValueError("threshold must be a positive finite number.")

    tail_values = values[values > threshold]
    if tail_values.size == 0:
        raise ValueError("no samples exceed the threshold.")
    logs = np.log(tail_values / threshold)
    hill_gamma = float(np.mean(logs))
    if hill_gamma <= 0.0:
        raise ValueError("tail samples must be strictly larger than the threshold.")
    return TailIndexEstimate(
        threshold=float(threshold),
        tail_count=int(tail_values.size),
        hill_gamma=hill_gamma,
        tail_index=float(1.0 / hill_gamma),
    )


def loglog_survival_fit(samples: ArrayLike, *, min_value: int = 1) -> LogLogSurvivalFit:
    """Fit ``log P(X >= x)`` against ``log x`` as a quick tail-shape diagnostic."""

    min_value = _validate_positive_int(min_value, "min_value")
    xs, survival = survival_function(samples, min_value=min_value)
    mask = (xs > 0) & (survival > 0.0)
    xs = xs[mask]
    survival = survival[mask]
    if xs.size < 2:
        raise ValueError("at least two positive survival points are required.")
    design = np.column_stack([np.log(xs.astype(float)), np.ones(xs.size)])
    slope, intercept = np.linalg.lstsq(design, np.log(survival), rcond=None)[0]
    return LogLogSurvivalFit(
        slope=float(slope),
        intercept=float(intercept),
        implied_tail_index=float(-slope),
        num_points=int(xs.size),
        min_value=float(xs[0]),
    )


def live_counts_from_label_sets(label_sets_by_frame: Iterable[Iterable[Hashable]]) -> tuple[int, ...]:
    """Return the live target count in each frame from per-frame label sets."""

    return tuple(len(set(labels)) for labels in label_sets_by_frame)


def birth_counts_from_label_sets(label_sets_by_frame: Iterable[Iterable[Hashable]]) -> tuple[int, ...]:
    """Return per-frame counts of labels first observed in that frame."""

    seen: set[Hashable] = set()
    births: list[int] = []
    for labels in label_sets_by_frame:
        current = set(labels)
        new_labels = current.difference(seen)
        births.append(len(new_labels))
        seen.update(current)
    return tuple(births)


def lifetimes_from_label_sets(label_sets_by_frame: Iterable[Iterable[Hashable]]) -> tuple[int, ...]:
    """Return inclusive first-to-last-frame lifetimes for observed labels.

    If a label disappears and later reappears, the gap is included. This is the
    right convention for persistent track identities; fragmented detections
    should be preprocessed into separate labels if gaps should terminate tracks.
    """

    first_seen: dict[Hashable, int] = {}
    last_seen: dict[Hashable, int] = {}
    for frame_index, labels in enumerate(label_sets_by_frame):
        for label in set(labels):
            first_seen.setdefault(label, frame_index)
            last_seen[label] = frame_index
    lifetimes = [last_seen[label] - first_seen[label] + 1 for label in first_seen]
    lifetimes.sort()
    return tuple(int(lifetime) for lifetime in lifetimes)


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


def _as_positive_float_array(samples: ArrayLike, *, name: str) -> np.ndarray:
    values = np.asarray(samples, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if values.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain finite values.")
    if np.any(values <= 0.0):
        raise ValueError(f"{name} must be strictly positive for tail-index estimation.")
    return values


def _validate_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a nonnegative integer.")
    if value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return int(value)


def _validate_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer.")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)
