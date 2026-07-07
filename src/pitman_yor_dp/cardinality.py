"""Cardinality priors based on Dirichlet and Pitman--Yor processes.

The module intentionally models *target-generated clusters* rather than a full
random finite set posterior. In a multitarget tracker, these probabilities should
be combined with likelihoods, survival, birth, missed-detection, and clutter
terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, lgamma, log
from numbers import Integral
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class PitmanYorCardinalityPrior:
    """Pitman--Yor Chinese-restaurant prior over target-generated clusters.

    Parameters
    ----------
    strength:
        Pitman--Yor strength parameter ``theta``. It must satisfy
        ``strength > -discount``. For the Dirichlet-process special case
        ``discount == 0``, this means ``strength > 0``.
    discount:
        Pitman--Yor discount parameter ``d`` with ``0 <= d < 1``. Positive
        values induce heavier-tailed cluster-count priors than the Dirichlet
        process.

    Notes
    -----
    Given current cluster sizes ``n_1, ..., n_K`` and
    ``n = sum_k n_k``, the CRP predictive probabilities are

    ``P(existing k) = (n_k - d) / (theta + n)``

    and

    ``P(new cluster) = (theta + d K) / (theta + n)``.
    """

    strength: float = 1.0
    discount: float = 0.0

    def __post_init__(self) -> None:
        strength = float(self.strength)
        discount = float(self.discount)
        if not isfinite(strength):
            raise ValueError("strength must be finite.")
        if not isfinite(discount):
            raise ValueError("discount must be finite.")
        if not 0.0 <= discount < 1.0:
            raise ValueError("discount must satisfy 0 <= discount < 1.")
        if strength <= -discount:
            raise ValueError("strength must satisfy strength > -discount.")
        object.__setattr__(self, "strength", strength)
        object.__setattr__(self, "discount", discount)

    @property
    def is_dirichlet_process(self) -> bool:
        """Return true if this is the Dirichlet-process special case."""

        return self.discount == 0.0

    def predictive_weights(self, cluster_sizes: Sequence[int]) -> tuple[float, ...]:
        """Return unnormalized existing-cluster and new-cluster weights.

        The final entry is the weight for a new cluster. Earlier entries follow
        the order of ``cluster_sizes``.
        """

        sizes = validate_cluster_sizes(cluster_sizes)
        if not sizes:
            return (1.0,)
        existing_weights = tuple(float(size) - self.discount for size in sizes)
        new_weight = self.strength + self.discount * len(sizes)
        return existing_weights + (float(new_weight),)

    def predictive_probabilities(self, cluster_sizes: Sequence[int]) -> tuple[float, ...]:
        """Return CRP predictive probabilities for existing clusters and a new cluster."""

        sizes = validate_cluster_sizes(cluster_sizes)
        if not sizes:
            return (1.0,)
        denominator = self.strength + float(sum(sizes))
        return tuple(weight / denominator for weight in self.predictive_weights(sizes))

    def predictive_log_probabilities(self, cluster_sizes: Sequence[int]) -> tuple[float, ...]:
        """Return log predictive probabilities for existing clusters and a new cluster."""

        return tuple(log(probability) for probability in self.predictive_probabilities(cluster_sizes))

    def log_exchangeable_partition_probability(self, cluster_sizes: Sequence[int]) -> float:
        """Return the EPPF log probability for one partition with given block sizes.

        The value is the probability of a particular exchangeable partition with
        these block sizes, not the marginal probability of the unordered size
        histogram.
        """

        sizes = validate_cluster_sizes(cluster_sizes)
        if not sizes:
            return 0.0

        num_observations = sum(sizes)
        num_clusters = len(sizes)
        log_probability = 0.0

        for cluster_index in range(1, num_clusters):
            log_probability += log(self.strength + self.discount * cluster_index)
        for observation_index in range(1, num_observations):
            log_probability -= log(self.strength + observation_index)
        for size in sizes:
            log_probability += _log_rising_factorial(1.0 - self.discount, size - 1)
        return float(log_probability)

    def exchangeable_partition_probability(self, cluster_sizes: Sequence[int]) -> float:
        """Return the EPPF probability for one partition with given block sizes."""

        return float(exp(self.log_exchangeable_partition_probability(cluster_sizes)))

    def cluster_count_pmf(self, num_observations: int) -> tuple[float, ...]:
        """Return ``P(K_n = k)`` for ``k = 0, ..., num_observations``.

        ``K_n`` is the number of occupied target-generated clusters after ``n``
        exchangeable observations under the CRP prior. Entry zero is one only
        when ``num_observations == 0``.
        """

        n = validate_nonnegative_int(num_observations, "num_observations")
        pmf = [1.0]
        for occupied_observations in range(n):
            next_pmf = [0.0] * (occupied_observations + 2)
            if occupied_observations == 0:
                next_pmf[1] = 1.0
                pmf = next_pmf
                continue
            denominator = self.strength + float(occupied_observations)
            for num_clusters, probability in enumerate(pmf):
                if probability == 0.0 or num_clusters == 0:
                    continue
                stay_weight = float(occupied_observations) - self.discount * float(num_clusters)
                new_weight = self.strength + self.discount * float(num_clusters)
                next_pmf[num_clusters] += probability * stay_weight / denominator
                next_pmf[num_clusters + 1] += probability * new_weight / denominator
            pmf = next_pmf
        return tuple(float(probability) for probability in pmf)

    def expected_number_of_clusters(self, num_observations: int) -> float:
        """Return the prior expected number of occupied clusters after ``n`` observations."""

        pmf = self.cluster_count_pmf(num_observations)
        return float(sum(num_clusters * probability for num_clusters, probability in enumerate(pmf)))

    def cluster_count_tail_probability(self, num_observations: int, min_clusters: int) -> float:
        """Return ``P(K_n >= min_clusters)`` under the prior predictive PMF."""

        min_clusters = validate_nonnegative_int(min_clusters, "min_clusters")
        pmf = self.cluster_count_pmf(num_observations)
        if min_clusters >= len(pmf):
            return 0.0
        return float(sum(pmf[min_clusters:]))

    def sample_partition(self, num_observations: int, rng: np.random.Generator | int | None = None) -> tuple[int, ...]:
        """Sample CRP cluster labels for ``num_observations`` observations.

        Labels are zero-based and ordered by first appearance. Sampling is meant
        for simulation and prior-predictive checks, not for posterior inference.
        """

        n = validate_nonnegative_int(num_observations, "num_observations")
        generator = _coerce_rng(rng)
        labels: list[int] = []
        sizes: list[int] = []

        for _ in range(n):
            probabilities = self.predictive_probabilities(sizes)
            sampled_index = int(generator.choice(len(probabilities), p=np.asarray(probabilities, dtype=float)))
            if sampled_index == len(sizes):
                labels.append(len(sizes))
                sizes.append(1)
            else:
                labels.append(sampled_index)
                sizes[sampled_index] += 1

        return tuple(labels)

    def sample_number_of_clusters(
        self,
        num_observations: int,
        *,
        num_samples: int,
        rng: np.random.Generator | int | None = None,
    ) -> np.ndarray:
        """Draw prior-predictive samples of ``K_n``."""

        num_samples = validate_nonnegative_int(num_samples, "num_samples")
        generator = _coerce_rng(rng)
        counts = np.empty(num_samples, dtype=int)
        for sample_index in range(num_samples):
            counts[sample_index] = len(set(self.sample_partition(num_observations, rng=generator)))
        return counts


class DirichletProcessCardinalityPrior(PitmanYorCardinalityPrior):
    """Dirichlet-process cardinality prior with zero Pitman--Yor discount."""

    def __init__(self, strength: float = 1.0) -> None:
        super().__init__(strength=strength, discount=0.0)


def empirical_pmf(samples: ArrayLike, *, max_value: int | None = None) -> tuple[float, ...]:
    """Return an empirical PMF from nonnegative integer samples."""

    sample_array = np.asarray(samples)
    if sample_array.ndim != 1:
        raise ValueError("samples must be one-dimensional.")
    if sample_array.size == 0:
        return ()
    if np.any(sample_array < 0):
        raise ValueError("samples must be nonnegative.")
    if not np.all(sample_array == np.floor(sample_array)):
        raise ValueError("samples must contain integer values.")

    integer_samples = sample_array.astype(int)
    if max_value is None:
        max_value = int(integer_samples.max())
    max_value = validate_nonnegative_int(max_value, "max_value")
    counts = np.bincount(integer_samples, minlength=max_value + 1)[: max_value + 1]
    return tuple((counts / float(integer_samples.size)).astype(float))


def validate_cluster_sizes(cluster_sizes: Sequence[int]) -> tuple[int, ...]:
    """Validate and normalize positive cluster sizes."""

    sizes = tuple(cluster_sizes)
    for size in sizes:
        if isinstance(size, bool) or not isinstance(size, Integral):
            raise TypeError("cluster sizes must be positive integers.")
        if int(size) <= 0:
            raise ValueError("cluster sizes must be positive integers.")
    return tuple(int(size) for size in sizes)


def validate_nonnegative_int(value: int, name: str) -> int:
    """Validate a nonnegative integer parameter."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a nonnegative integer.")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return value


def _log_rising_factorial(start: float, length: int) -> float:
    if length <= 0:
        return 0.0
    return float(lgamma(start + length) - lgamma(start))


def _coerce_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)
