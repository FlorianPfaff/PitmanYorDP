"""Exact finite-window association inference, not a live-population tracker.

Labels are canonical target identities; -1 denotes clutter. The inference API
never accepts ground truth. All priors are conditioned on observed frame counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import exp, lgamma, log, pi
from typing import Literal, Sequence
from types import MappingProxyType

import numpy as np

def _positive(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")


def logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    if maximum == -np.inf:
        return -np.inf
    return maximum + log(float(np.exp(values - maximum).sum()))


def _group_logsumexp(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    maximum = np.full(int(groups.max()) + 1, -np.inf)
    np.maximum.at(maximum, groups, values)
    shifted = np.zeros_like(values)
    finite = np.isfinite(values)
    shifted[finite] = np.exp(values[finite] - maximum[groups[finite]])
    sums = np.bincount(groups, weights=shifted, minlength=len(maximum))
    with np.errstate(divide="ignore"):
        return maximum + np.log(sums)


@dataclass(frozen=True)
class ObservationWindow:
    frame_times: tuple[float, ...]
    measurements: tuple[tuple[tuple[float, float], ...], ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.frame_times) <= 4 or len(self.measurements) != len(self.frame_times):
            raise ValueError("Expected one to four frames with matching measurement lists.")
        times = np.asarray(self.frame_times)
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("Frame times must be finite and strictly increasing.")
        if sum(self.frame_counts) > 9:
            raise ValueError("Exact inference is limited to nine observations; do not truncate.")
        for frame in self.measurements:
            for position in frame:
                if len(position) != 2 or not np.isfinite(position).all():
                    raise ValueError("Measurements must be finite two-dimensional positions.")

    @property
    def frame_counts(self) -> tuple[int, ...]:
        return tuple(len(frame) for frame in self.measurements)

    @property
    def positions(self) -> np.ndarray:
        return np.asarray([y for frame in self.measurements for y in frame], dtype=float).reshape(-1, 2)


@dataclass(frozen=True)
class MotionModel:
    position_std: float = 2.0
    velocity_std: float = 0.4
    acceleration_std: float = 0.1
    measurement_std: float = 0.5
    clutter_std: float = 6.0

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            _positive(value, name)


def trajectory_covariance(times: Sequence[float], motion: MotionModel, *, noisy: bool = True) -> np.ndarray:
    """One-coordinate integrated-white-acceleration CV covariance from time zero."""
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or not np.isfinite(t).all() or np.any(t < 0):
        raise ValueError("Trajectory times must be finite and nonnegative.")
    lower = np.minimum.outer(t, t)
    upper = np.maximum.outer(t, t)
    covariance = (motion.position_std**2 + motion.velocity_std**2 * np.outer(t, t)
                  + motion.acceleration_std**2 * lower**2 * (3 * upper - lower) / 6)
    if noisy:
        covariance = covariance + motion.measurement_std**2 * np.eye(len(t))
    return covariance


@dataclass(frozen=True)
class SurvivalModel:
    probabilities: tuple[float, ...] = (0.8,)
    weights: tuple[float, ...] = (1.0,)
    detection_probability: float = 0.8
    initial_birth_mass: float = 0.65

    def __post_init__(self) -> None:
        if not self.probabilities or len(self.weights) != len(self.probabilities):
            raise ValueError("Survival probabilities and weights must have equal nonzero length.")
        values = (*self.probabilities, self.detection_probability, self.initial_birth_mass)
        if any(not np.isfinite(p) or not 0 < p < 1 for p in values):
            raise ValueError("Survival, detection, and initial birth probabilities must lie in (0, 1).")
        if any(not np.isfinite(w) or w < 0 for w in self.weights) or not np.isclose(sum(self.weights), 1):
            raise ValueError("Mixture weights must be nonnegative and sum to one.")


@lru_cache(maxsize=256)
def observation_pattern_probabilities(num_frames: int, survival: SurvivalModel) -> tuple[float, ...]:
    """Integrate birth, mixture component, death, and missed detections, including empty patterns.

    Survival is per acquisition step, not per second. The last alive frame is
    right-censored at the window boundary. Birth precedes that frame's detection.
    """
    if not 1 <= num_frames <= 4:
        raise ValueError("Expected one to four frames.")
    birth = ([1.0] if num_frames == 1 else
             [survival.initial_birth_mass] + [(1 - survival.initial_birth_mass) / (num_frames - 1)] * (num_frames - 1))
    probabilities = np.zeros(1 << num_frames)
    pd = survival.detection_probability
    for s, weight in zip(survival.probabilities, survival.weights):
        for first, birth_probability in enumerate(birth):
            for last in range(first, num_frames):
                alive_probability = weight * birth_probability * s**(last - first)
                if last < num_frames - 1:
                    alive_probability *= 1 - s
                for bits in range(1 << (last - first + 1)):
                    hits = bits.bit_count()
                    probabilities[bits << first] += alive_probability * pd**hits * (1 - pd)**(last - first + 1 - hits)
    return tuple(float(p) for p in probabilities)


@dataclass(frozen=True)
class AssociationModel:
    kind: Literal["survival", "dp", "pyp", "oracle"] = "survival"
    survival: SurvivalModel = SurvivalModel()
    discount: float = 0.0
    strength: float = 1.0
    count_rate: float = 1.5
    target_probability: float = 0.8
    oracle_population: tuple[float, ...] = (0.0, 0.0, 0.5, 0.5)
    oracle_clutter_rate: float = 0.2

    def __post_init__(self) -> None:
        if self.kind not in {"survival", "dp", "pyp", "oracle"}:
            raise ValueError("Unknown association prior.")
        if not np.isfinite(self.discount) or not 0 <= self.discount < 1:
            raise ValueError("Discount must lie in [0, 1).")
        _positive(self.strength, "strength")
        _positive(self.count_rate, "count_rate")
        if not np.isfinite(self.target_probability) or not 0 < self.target_probability < 1:
            raise ValueError("Target probability must lie in (0, 1).")
        if (not self.oracle_population or any(not np.isfinite(p) or p < 0 for p in self.oracle_population)
                or not np.isclose(sum(self.oracle_population), 1)):
            raise ValueError("Oracle population probabilities must sum to one.")
        if not np.isfinite(self.oracle_clutter_rate) or self.oracle_clutter_rate < 0:
            raise ValueError("Oracle clutter rate must be finite and nonnegative.")


@dataclass(frozen=True)
class _Topology:
    hypotheses: tuple[tuple[int, ...], ...]
    frames: tuple[int, ...]
    block_masks: tuple[int, ...]
    block_indices: np.ndarray
    masks: np.ndarray
    counts: np.ndarray
    target_counts: np.ndarray
    mask_groups: tuple[np.ndarray, ...]
    count_groups: tuple[tuple[np.ndarray, ...], ...]
    group_ids: np.ndarray
    group_representatives: np.ndarray
    label_indices: object


@lru_cache(maxsize=128)
def _topology(frame_counts: tuple[int, ...]) -> _Topology:
    if sum(frame_counts) > 9 or any(type(n) is not int or n < 0 for n in frame_counts):
        raise ValueError("Expected nonnegative counts totaling at most nine.")
    frames = tuple(t for t, n in enumerate(frame_counts) for _ in range(n))
    labels: list[int] = []
    occupied: list[set[int]] = []
    hypotheses: list[tuple[int, ...]] = []

    def visit(i: int) -> None:
        if i == len(frames):
            hypotheses.append(tuple(labels))
            return
        labels.append(-1)
        visit(i + 1)
        labels.pop()
        for k, used in enumerate(occupied):
            if frames[i] not in used:
                used.add(frames[i])
                labels.append(k)
                visit(i + 1)
                labels.pop()
                used.remove(frames[i])
        labels.append(len(occupied))
        occupied.append({frames[i]})
        visit(i + 1)
        occupied.pop()
        labels.pop()

    visit(0)
    blocks = [tuple(sum(1 << i for i, label in enumerate(h) if label == k)
                    for k in range(max(h, default=-1) + 1)) for h in hypotheses]
    unique = (0, *sorted({block for row in blocks for block in row}))
    lookup = {block: i for i, block in enumerate(unique)}
    indices = np.zeros((len(blocks), len(frames)), dtype=np.int32)
    for i, row in enumerate(blocks):
        indices[i, :len(row)] = [lookup[block] for block in row]
    masks = np.asarray([sum(row) for row in blocks], dtype=np.int32)
    counts = np.asarray([len(row) for row in blocks], dtype=np.int32)
    target_counts = np.asarray([int(mask).bit_count() for mask in masks], dtype=np.int32)
    groups = tuple(np.flatnonzero(masks == mask) for mask in range(1 << len(frames)))
    count_groups = tuple(tuple(group[counts[group] == k] for k in np.unique(counts[group])) for group in groups)
    _, representatives, group_ids = np.unique(masks * (len(frames) + 1) + counts, return_index=True, return_inverse=True)
    for array in (indices, masks, counts, target_counts, *groups, *(g for gs in count_groups for g in gs)):
        array.setflags(write=False)
    representatives.setflags(write=False)
    group_ids.setflags(write=False)
    return _Topology(tuple(hypotheses), frames, unique, indices, masks, counts, target_counts, groups, count_groups,
                     group_ids, representatives, MappingProxyType({h: i for i, h in enumerate(hypotheses)}))


@lru_cache(maxsize=512)
def _log_prior(frame_counts: tuple[int, ...], model: AssociationModel) -> np.ndarray:
    topology = _topology(frame_counts)
    n = sum(frame_counts)
    patterns = observation_pattern_probabilities(len(frame_counts), model.survival)
    block_log = np.zeros(len(topology.block_masks))
    for i, mask in enumerate(topology.block_masks[1:], 1):
        pattern = sum(1 << topology.frames[j] for j in range(n) if mask & (1 << j))
        size = mask.bit_count()
        block_log[i] = log(patterns[pattern])
        if model.kind == "survival":
            block_log[i] += lgamma(size - model.discount) - lgamma(1 - model.discount) - lgamma(size)
    raw = block_log[topology.block_indices].sum(axis=1)
    if model.kind == "oracle":
        # Marginalize undetected members of the finite population. Poisson
        # clutter factorials cancel against within-frame random permutations.
        factors = []
        for k in range(n + 1):
            factors.append(sum(p * exp(lgamma(m + 1) - lgamma(m - k + 1)) * patterns[0]**(m - k)
                               for m, p in enumerate(model.oracle_population) if m >= k))
        with np.errstate(divide="ignore"):
            raw += np.log(np.asarray(factors))[topology.counts]
        clutter_counts = n - topology.target_counts
        if model.oracle_clutter_rate == 0:
            raw[clutter_counts > 0] = -np.inf
        else:
            raw += clutter_counts * log(model.oracle_clutter_rate)
        normalizer = logsumexp(raw)
        if not np.isfinite(normalizer):
            raise ValueError("The oracle assigns zero probability to these frame counts.")
        raw -= normalizer
    else:
        target = model.target_probability
        subset = topology.target_counts * log(target) + (n - topology.target_counts) * log(1 - target)
        if model.kind in {"dp", "pyp"}:
            d = model.discount if model.kind == "pyp" else 0.0
            block_log = np.asarray([0.0] + [lgamma(mask.bit_count() - d) - lgamma(1 - d) for mask in topology.block_masks[1:]])
            coefficients = np.zeros((n + 1, n + 1))
            for total in range(1, n + 1):
                for k in range(1, total + 1):
                    coefficients[total, k] = (sum(log(model.strength + j * d) for j in range(1, k))
                                              - lgamma(model.strength + total) + lgamma(model.strength + 1))
            raw = block_log[topology.block_indices].sum(axis=1) + coefficients[topology.target_counts, topology.counts]
            raw -= _group_logsumexp(raw, topology.masks)[topology.masks]
        else:
            # q(K | A,m) is a common Poisson law truncated to feasible K. The
            # discount changes only R(pi | A,K,m), never this count distribution.
            representatives = topology.group_representatives
            k_values = topology.counts[representatives]
            q = np.asarray([k * log(model.count_rate) - lgamma(k + 1) for k in k_values])
            masks = topology.masks[representatives]
            q -= _group_logsumexp(q, masks)[masks]
            raw += (q - _group_logsumexp(raw, topology.group_ids))[topology.group_ids]
        raw += subset
    raw.setflags(write=False)
    return raw


@dataclass(frozen=True)
class PreparedWindow:
    observations: ObservationWindow
    topology: _Topology
    log_likelihood: np.ndarray


@lru_cache(maxsize=512)
def _block_precision(times: tuple[float, ...], motion: MotionModel) -> tuple[np.ndarray, float]:
    covariance = trajectory_covariance(times, motion)
    cholesky = np.linalg.cholesky(covariance)
    inverse = np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, np.eye(len(times))))
    inverse.setflags(write=False)
    return inverse, 2 * float(np.log(np.diag(cholesky)).sum())


def prepare_window(observations: ObservationWindow, motion: MotionModel = MotionModel()) -> PreparedWindow:
    """Cache block likelihoods once for comparisons sharing the observation model."""
    topology = _topology(observations.frame_counts)
    positions = observations.positions
    relative_times = np.asarray(observations.frame_times) - observations.frame_times[0]
    logs = np.zeros(len(topology.block_masks))
    for i, mask in enumerate(topology.block_masks[1:], 1):
        members = [j for j in range(len(positions)) if mask & (1 << j)]
        times = tuple(float(relative_times[topology.frames[j]]) for j in members)
        precision, determinant = _block_precision(times, motion)
        y = positions[members]
        logs[i] = -len(members) * log(2 * pi) - determinant - 0.5 * float((y * (precision @ y)).sum())
    log_clutter = -log(2 * pi * motion.clutter_std**2) - (positions**2).sum(axis=1) / (2 * motion.clutter_std**2)
    likelihood = logs[topology.block_indices].sum(axis=1)
    for i, clutter_log in enumerate(log_clutter):
        likelihood += ((topology.masks & (1 << i)) == 0) * clutter_log
    likelihood.setflags(write=False)
    return PreparedWindow(observations, topology, likelihood)


@dataclass(frozen=True)
class AssociationPosterior:
    hypotheses: tuple[tuple[int, ...], ...]
    log_probabilities: np.ndarray
    count_probabilities: tuple[float, ...]
    log_evidence: float
    log_count_probabilities: tuple[float, ...]
    _label_indices: object

    @property
    def probabilities(self) -> np.ndarray:
        return np.exp(self.log_probabilities)

    @property
    def map_labels(self) -> tuple[int, ...]:
        return self.hypotheses[int(np.argmax(self.log_probabilities))]


def infer_associations(observations: ObservationWindow, model: AssociationModel = AssociationModel(),
                       motion: MotionModel = MotionModel()) -> AssociationPosterior:
    return infer_prepared(prepare_window(observations, motion), model)


def infer_prepared(prepared: PreparedWindow, model: AssociationModel) -> AssociationPosterior:
    joint = _log_prior(prepared.observations.frame_counts, model) + prepared.log_likelihood
    evidence = logsumexp(joint)
    if not np.isfinite(evidence):
        raise ValueError("No finite posterior mass; inspect the observation model and numerical scale.")
    probabilities = joint - evidence
    probabilities.setflags(write=False)
    log_counts = _group_logsumexp(probabilities, prepared.topology.counts)
    return AssociationPosterior(prepared.topology.hypotheses, probabilities, tuple(float(p) for p in np.exp(log_counts)), evidence,
                                tuple(float(p) for p in log_counts), prepared.topology.label_indices)


def canonical_labels(labels: Sequence[int]) -> tuple[int, ...]:
    identities: dict[int, int] = {}
    result = []
    for label in labels:
        if isinstance(label, (bool, np.bool_)) or not isinstance(label, (int, np.integer)) or label < -1:
            raise ValueError("Identity labels must be integers >= 0, or -1 for clutter.")
        if label == -1:
            result.append(-1)
        else:
            result.append(identities.setdefault(int(label), len(identities)))
    return tuple(result)


def score_associations(posterior: AssociationPosterior, truth: Sequence[int]) -> dict[str, float]:
    """Evaluate an already-computed posterior; labels cannot affect inference."""
    labels = canonical_labels(truth)
    try:
        index = posterior._label_indices[labels]
    except KeyError as error:
        raise ValueError("Truth is incompatible with the observation window.") from error
    n = len(labels)
    k = max(labels, default=-1) + 1
    log_count = posterior.log_count_probabilities[k]
    estimate = posterior.map_labels
    differences = [float((a == -1) != (b == -1)) for a, b in zip(labels, estimate)]
    differences += [float((labels[i] >= 0 and labels[i] == labels[j]) !=
                          (estimate[i] >= 0 and estimate[i] == estimate[j])) for i in range(n) for j in range(i)]
    cumulative = np.cumsum(posterior.count_probabilities)
    lower = int(np.searchsorted(cumulative, 0.05))
    upper = int(np.searchsorted(cumulative, 0.95))
    return {
        "association_nll": -float(posterior.log_probabilities[index]) / max(n, 1),
        "count_nll": -log_count,
        "count_brier": float(sum((p - float(j == k))**2 for j, p in enumerate(posterior.count_probabilities))),
        "count_coverage_90": float(lower <= k <= upper),
        "association_error": float(np.mean(differences)) if differences else 0.0,
        "expected_count": float(sum(j * p for j, p in enumerate(posterior.count_probabilities))),
    }
