"""Exact conditional partition diagnostics for tiny point-target windows.

These are priors over detected identity partitions conditional on frame exposure,
not posteriors over live targets. Enumeration is deliberately limited to nine
observations. Each identity can contribute at most one observation per frame.
"""

from __future__ import annotations

from collections import Counter
from math import exp, fsum, isfinite
from typing import Mapping, Sequence

from .cardinality import PitmanYorCardinalityPrior, validate_nonnegative_int

Partition = tuple[int, ...]


def valid_partitions(frame_counts: Sequence[int]) -> tuple[Partition, ...]:
    """Enumerate valid partitions once, using labels ordered by first appearance."""

    counts = tuple(validate_nonnegative_int(count, "frame count") for count in frame_counts)
    if sum(counts) > 9:
        raise ValueError("Exact enumeration is limited to nine observations.")
    frames = tuple(frame for frame, count in enumerate(counts) for _ in range(count))
    labels: list[int] = []
    block_frames: list[set[int]] = []
    partitions: list[Partition] = []

    def visit(index: int) -> None:
        if index == len(frames):
            partitions.append(tuple(labels))
            return
        frame = frames[index]
        for block in range(len(block_frames)):
            if frame not in block_frames[block]:
                labels.append(block)
                block_frames[block].add(frame)
                visit(index + 1)
                block_frames[block].remove(frame)
                labels.pop()
        labels.append(len(block_frames))
        block_frames.append({frame})
        visit(index + 1)
        block_frames.pop()
        labels.pop()

    visit(0)
    return tuple(partitions)


def conditional_partition_pmf(
    prior: PitmanYorCardinalityPrior,
    frame_counts: Sequence[int],
    *,
    num_clusters: int | None = None,
) -> dict[Partition, float]:
    """Condition the full EPPF on temporal validity and optionally on K.

    Normalize after enumerating all valid partitions. Greedily renormalizing CRP
    transitions after rejecting same-frame assignments defines a different law.
    """

    if num_clusters is not None:
        num_clusters = validate_nonnegative_int(num_clusters, "num_clusters")
    partitions = [
        partition for partition in valid_partitions(frame_counts)
        if num_clusters is None or len(set(partition)) == num_clusters
    ]
    if not partitions:
        raise ValueError("No valid partitions have the requested number of clusters.")
    log_weights = [
        prior.log_exchangeable_partition_probability(tuple(Counter(partition).values()))
        for partition in partitions
    ]
    offset = max(log_weights)
    weights = [exp(weight - offset) for weight in log_weights]
    normalizer = fsum(weights)
    return {partition: weight / normalizer for partition, weight in zip(partitions, weights)}


def cluster_count_distribution(pmf: Mapping[Partition, float]) -> dict[int, float]:
    """Marginalize a partition PMF over the number of detected identities."""

    _validate_pmf(pmf)
    counts = sorted({len(set(partition)) for partition in pmf})
    return {k: fsum(probability for partition, probability in pmf.items() if len(set(partition)) == k)
            for k in counts}


def match_cluster_count_distribution(
    source: Mapping[Partition, float], target: Mapping[Partition, float]
) -> dict[Partition, float]:
    """Keep source's conditional partitions given K and use target's entire K law.

    This diagnostic control need not itself be a DP or PYP. It is stronger than
    matching only the expected cluster count and is not an oracle tracker.
    """

    if source.keys() != target.keys():
        raise ValueError("Partition PMFs must have identical support.")
    source_counts = cluster_count_distribution(source)
    target_counts = cluster_count_distribution(target)
    if any(value <= 0.0 for value in source_counts.values()):
        raise ValueError("Source must put positive mass on every cluster count.")
    return {
        partition: probability * target_counts[len(set(partition))] / source_counts[len(set(partition))]
        for partition, probability in source.items()
    }


def _validate_pmf(pmf: Mapping[Partition, float]) -> None:
    if not pmf or any(not isfinite(value) or value < 0.0 for value in pmf.values()):
        raise ValueError("Expected a nonempty PMF with finite nonnegative probabilities.")
    if abs(fsum(pmf.values()) - 1.0) > 1e-10:
        raise ValueError("Partition probabilities must sum to one.")
