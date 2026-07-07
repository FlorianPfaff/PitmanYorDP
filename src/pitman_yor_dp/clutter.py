"""Diagnostics for the new-target versus clutter confound.

A Pitman--Yor discount can increase the prior probability of minting singleton
clusters. These helpers make that pressure explicit before it is embedded in a
tracker-specific likelihood model.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log

from .cardinality import PitmanYorCardinalityPrior, validate_nonnegative_int


@dataclass(frozen=True)
class NoveltyClutterOdds:
    """One diagnostic point for new-target versus clutter odds."""

    exposure_count: int
    active_clusters: int
    novelty_prior_probability: float
    birth_to_clutter_likelihood_ratio: float
    log_odds: float

    @property
    def favors_new_target(self) -> bool:
        """Return true when the diagnostic odds favor a new target over clutter."""

        return self.log_odds > 0.0


def novelty_prior_probability(
    prior: PitmanYorCardinalityPrior,
    *,
    exposure_count: int,
    active_clusters: int,
) -> float:
    """Return the Pitman--Yor prior probability of a new cluster.

    ``exposure_count`` is the number of observations already assigned in the
    exchangeable restaurant used by the diagnostic. In a per-scan model, it is a
    count of current-scan ambiguous target-generated observations; in an across-
    frame model, using it naively would conflate track age and cluster size.
    """

    exposure_count = validate_nonnegative_int(exposure_count, "exposure_count")
    active_clusters = validate_nonnegative_int(active_clusters, "active_clusters")
    if active_clusters > exposure_count:
        raise ValueError("active_clusters cannot exceed exposure_count.")
    if exposure_count == 0:
        if active_clusters != 0:
            raise ValueError("active_clusters must be zero when exposure_count is zero.")
        return 1.0
    denominator = prior.strength + float(exposure_count)
    numerator = prior.strength + prior.discount * float(active_clusters)
    if numerator < 0.0 or denominator <= 0.0:
        raise ValueError("invalid prior parameters for novelty probability.")
    return float(numerator / denominator)


def new_target_vs_clutter_odds(
    prior: PitmanYorCardinalityPrior,
    *,
    exposure_count: int,
    active_clusters: int,
    birth_to_clutter_likelihood_ratio: float,
) -> NoveltyClutterOdds:
    """Return log odds for a new target against clutter.

    ``birth_to_clutter_likelihood_ratio`` should be the marginal birth evidence
    divided by the clutter intensity for the measurement under consideration.
    Positive log odds mean that the prior-weighted diagnostic favors minting a
    new target over declaring clutter.
    """

    likelihood_ratio = float(birth_to_clutter_likelihood_ratio)
    if not isfinite(likelihood_ratio) or likelihood_ratio <= 0.0:
        raise ValueError("birth_to_clutter_likelihood_ratio must be positive and finite.")
    novelty_probability = novelty_prior_probability(
        prior,
        exposure_count=exposure_count,
        active_clusters=active_clusters,
    )
    if novelty_probability <= 0.0:
        log_odds = float("-inf")
    else:
        log_odds = log(novelty_probability) + log(likelihood_ratio)
    return NoveltyClutterOdds(
        exposure_count=int(exposure_count),
        active_clusters=int(active_clusters),
        novelty_prior_probability=float(novelty_probability),
        birth_to_clutter_likelihood_ratio=likelihood_ratio,
        log_odds=float(log_odds),
    )


def phase_transition_table(
    prior: PitmanYorCardinalityPrior,
    *,
    exposure_counts: tuple[int, ...],
    active_clusters: tuple[int, ...],
    birth_to_clutter_likelihood_ratios: tuple[float, ...],
) -> tuple[NoveltyClutterOdds, ...]:
    """Return a small grid for new-target/clutter phase-transition diagnostics."""

    rows: list[NoveltyClutterOdds] = []
    for exposure_count in exposure_counts:
        for num_clusters in active_clusters:
            if num_clusters > exposure_count:
                continue
            for likelihood_ratio in birth_to_clutter_likelihood_ratios:
                rows.append(
                    new_target_vs_clutter_odds(
                        prior,
                        exposure_count=exposure_count,
                        active_clusters=num_clusters,
                        birth_to_clutter_likelihood_ratio=likelihood_ratio,
                    )
                )
    return tuple(rows)
