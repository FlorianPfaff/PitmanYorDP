"""Independent finite-population scenes for exact association experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .association import MotionModel, ObservationWindow, SurvivalModel, canonical_labels, trajectory_covariance


@dataclass(frozen=True)
class SceneConfig:
    num_frames: int = 3
    motion: MotionModel = MotionModel()
    survival: SurvivalModel = SurvivalModel()
    clutter_rate: float = 0.2
    population_probabilities: tuple[float, ...] = (0.0, 0.0, 0.5, 0.5)
    max_attempts: int = 10000

    def __post_init__(self) -> None:
        if type(self.num_frames) is not int or not 1 <= self.num_frames <= 4:
            raise ValueError("Scene length must be one to four frames.")
        if not np.isfinite(self.clutter_rate) or self.clutter_rate < 0:
            raise ValueError("Clutter rate must be finite and nonnegative.")
        if (not self.population_probabilities or any(not np.isfinite(p) or p < 0 for p in self.population_probabilities)
                or not np.isclose(sum(self.population_probabilities), 1)):
            raise ValueError("Population probabilities must sum to one.")
        if type(self.max_attempts) is not int or self.max_attempts <= 0:
            raise ValueError("max_attempts must be a positive integer.")


@dataclass(frozen=True)
class SimulatedScene:
    observations: ObservationWindow
    truth: tuple[int, ...]
    rejected_oversized: int
    seed: int


def simulate_association_scene(config: SceneConfig, seed: int) -> SimulatedScene:
    """Generate and shuffle a scene, explicitly conditioning on at most nine observations.

    Initial trajectory states are at the common window origin even for later
    births. This makes the shared Gaussian likelihood exact, independently of
    birth/death. It is a finite-window benchmark, not a physical FOV simulator.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    rng = np.random.default_rng(seed)
    length = config.num_frames
    times = tuple(float(t) for t in range(length))
    birth_probabilities = ([1.0] if length == 1 else [config.survival.initial_birth_mass]
                           + [(1 - config.survival.initial_birth_mass) / (length - 1)] * (length - 1))
    covariance = trajectory_covariance(times, config.motion, noisy=False)
    for attempt in range(config.max_attempts):
        frames: list[list[tuple[tuple[float, float], int]]] = [[] for _ in times]
        population = int(rng.choice(len(config.population_probabilities), p=config.population_probabilities))
        for identity in range(population):
            birth = int(rng.choice(length, p=birth_probabilities))
            survival = float(rng.choice(config.survival.probabilities, p=config.survival.weights))
            duration = int(rng.geometric(1 - survival))
            trajectory = rng.multivariate_normal(np.zeros(length), covariance, size=2).T
            for frame in range(birth, min(length, birth + duration)):
                if rng.random() < config.survival.detection_probability:
                    position = trajectory[frame] + rng.normal(0, config.motion.measurement_std, 2)
                    frames[frame].append(((float(position[0]), float(position[1])), identity))
        for frame in frames:
            for _ in range(int(rng.poisson(config.clutter_rate))):
                position = rng.normal(0, config.motion.clutter_std, 2)
                frame.append(((float(position[0]), float(position[1])), -1))
            rng.shuffle(frame)
        if sum(map(len, frames)) > 9:
            continue
        observations = ObservationWindow(times, tuple(tuple(position for position, _ in frame) for frame in frames))
        truth = canonical_labels(tuple(identity for frame in frames for _, identity in frame))
        return SimulatedScene(observations, truth, attempt, seed)
    raise RuntimeError("Too many oversized scenes; change the declared benchmark, not the measurements.")
