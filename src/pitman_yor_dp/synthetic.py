"""Synthetic tracking-label generators for count-diagnostic experiments.

The generator produces only frame-level track labels, not measurements or target
states. This is enough for the first gating experiment: do the induced live
counts, birth counts, and lifetimes look Poisson-like, overdispersed, or plausibly
heavy-tailed?
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Literal

import numpy as np

from .adapters import CountSeriesBundle, TrackRecord, count_series_from_records

BirthCountModel = Literal["poisson", "negative_binomial", "deterministic"]
LifetimeModel = Literal["geometric", "pareto", "fixed"]


@dataclass(frozen=True)
class SyntheticTrackingConfig:
    """Configuration for synthetic frame-level track-label generation.

    Parameters
    ----------
    num_frames:
        Number of frames to generate.
    birth_rate:
        Mean number of new labels per frame for Poisson and negative-binomial
        births, or the deterministic number of births per frame rounded to the
        nearest nonnegative integer.
    birth_count_model:
        Birth-count model. Use ``"negative_binomial"`` for overdispersed but
        light-tailed birth counts.
    birth_overdispersion_size:
        Negative-binomial size parameter. Smaller values produce stronger
        overdispersion. The parameterization has
        ``Var[B_t] = birth_rate + birth_rate^2 / size``.
    lifetime_model:
        Track-lifetime model. Use ``"pareto"`` to create a simple heavy-tailed
        lifetime diagnostic case.
    mean_lifetime:
        Mean lifetime used by the geometric model.
    fixed_lifetime:
        Lifetime used by the fixed model.
    pareto_shape:
        Pareto shape. Smaller values produce heavier lifetime tails.
    min_lifetime:
        Minimum lifetime for Pareto samples.
    seed:
        Optional seed for reproducibility.
    """

    num_frames: int = 100
    birth_rate: float = 1.0
    birth_count_model: BirthCountModel = "poisson"
    birth_overdispersion_size: float = 2.0
    lifetime_model: LifetimeModel = "geometric"
    mean_lifetime: float = 10.0
    fixed_lifetime: int = 10
    pareto_shape: float = 2.0
    min_lifetime: int = 1
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.num_frames <= 0:
            raise ValueError("num_frames must be positive.")
        if self.birth_rate < 0.0:
            raise ValueError("birth_rate must be nonnegative.")
        if self.birth_count_model not in {"poisson", "negative_binomial", "deterministic"}:
            raise ValueError("birth_count_model must be 'poisson', 'negative_binomial', or 'deterministic'.")
        if self.birth_overdispersion_size <= 0.0:
            raise ValueError("birth_overdispersion_size must be positive.")
        if self.lifetime_model not in {"geometric", "pareto", "fixed"}:
            raise ValueError("lifetime_model must be 'geometric', 'pareto', or 'fixed'.")
        if self.mean_lifetime <= 0.0:
            raise ValueError("mean_lifetime must be positive.")
        if self.fixed_lifetime <= 0:
            raise ValueError("fixed_lifetime must be positive.")
        if self.pareto_shape <= 0.0:
            raise ValueError("pareto_shape must be positive.")
        if self.min_lifetime <= 0:
            raise ValueError("min_lifetime must be positive.")


def simulate_track_records(config: SyntheticTrackingConfig) -> tuple[TrackRecord, ...]:
    """Generate synthetic frame-level track records from ``config``."""

    rng = np.random.default_rng(config.seed)
    records: list[TrackRecord] = []
    next_track_id = 0

    for frame in range(config.num_frames):
        num_births = _sample_birth_count(config, rng)
        for _ in range(num_births):
            lifetime = _sample_lifetime(config, rng)
            track_id = f"synthetic-{next_track_id}"
            next_track_id += 1
            last_frame = min(config.num_frames - 1, frame + lifetime - 1)
            for active_frame in range(frame, last_frame + 1):
                records.append(
                    TrackRecord(
                        frame=active_frame,
                        track_id=track_id,
                        metadata={"birth_frame": frame, "sampled_lifetime": lifetime},
                    )
                )
    return tuple(records)


def simulate_count_series(config: SyntheticTrackingConfig) -> CountSeriesBundle:
    """Generate synthetic records and return count-series diagnostics input."""

    return count_series_from_records(
        simulate_track_records(config),
        include_empty_frames=True,
        start_frame=0,
        end_frame=config.num_frames - 1,
    )


def _sample_birth_count(config: SyntheticTrackingConfig, rng: np.random.Generator) -> int:
    if config.birth_count_model == "deterministic":
        return max(0, int(round(config.birth_rate)))
    if config.birth_count_model == "poisson":
        return int(rng.poisson(config.birth_rate))
    size = float(config.birth_overdispersion_size)
    if config.birth_rate == 0.0:
        return 0
    success_probability = size / (size + float(config.birth_rate))
    return int(rng.negative_binomial(size, success_probability))


def _sample_lifetime(config: SyntheticTrackingConfig, rng: np.random.Generator) -> int:
    if config.lifetime_model == "fixed":
        return int(config.fixed_lifetime)
    if config.lifetime_model == "geometric":
        success_probability = min(1.0, 1.0 / float(config.mean_lifetime))
        return int(rng.geometric(success_probability))
    sample = float(config.min_lifetime) * (1.0 + float(rng.pareto(config.pareto_shape)))
    return max(1, int(ceil(sample)))
