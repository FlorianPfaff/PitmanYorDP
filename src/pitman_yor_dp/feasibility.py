"""Locked development/pilot/confirmation workflow for the association study.

The balanced primary contrast is the fitted mixed-survival baseline minus its
count-matched discount extension. Positive NLL differences favor the extension.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
from itertools import product
import json
import os
from math import ceil, lgamma, log, sqrt
from pathlib import Path
import platform
import subprocess
import tempfile
from time import perf_counter

import numpy as np

from .association import (
    AssociationModel, MotionModel, SurvivalModel, _log_prior, infer_prepared,
    logsumexp, prepare_window, score_associations,
)
from .association_scenes import SceneConfig, simulate_association_scene


@dataclass(frozen=True)
class FeasibilityConfig:
    master_seed: int = 20261003
    development_scenes: int = 200
    pilot_scenes: int = 100
    minimum_confirmation: int = 200
    maximum_confirmation: int = 5000
    negative_control_scenes: int = 40
    ci_half_width: float = 0.01
    minimum_effect: float = 0.02
    count_harm_margin: float = 0.02
    frame_lengths: tuple[int, ...] = (3, 4)
    position_scales: tuple[float, ...] = (0.75, 3.0)
    clutter_rates: tuple[float, ...] = (0.05, 0.6)
    detection_probabilities: tuple[float, ...] = (0.65, 0.9)
    lifetime_kinds: tuple[str, ...] = ("homogeneous", "heterogeneous")
    discount_grid: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8)
    strength_grid: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0)
    count_rate_grid: tuple[float, ...] = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    survival_grid: tuple[float, ...] = (0.35, 0.4, 0.6, 0.8, 0.95)
    mixture_pairs: tuple[tuple[float, float], ...] = ((0.35, 0.8), (0.35, 0.95), (0.4, 0.95), (0.6, 0.95))
    mixture_weight_grid: tuple[float, ...] = (0.25, 0.5, 0.75)
    novelty_status: str = "unestablished"

    def __post_init__(self) -> None:
        for name in ("development_scenes", "pilot_scenes", "minimum_confirmation", "maximum_confirmation", "negative_control_scenes"):
            if type(getattr(self, name)) is not int or not 2 <= getattr(self, name) < 2**20:
                raise ValueError(f"{name} must be an integer in [2, 2**20).")
        if type(self.master_seed) is not int or not 0 <= self.master_seed < 2**32:
            raise ValueError("master_seed must be a nonnegative integer below 2**32.")
        if self.minimum_confirmation > self.maximum_confirmation or self.maximum_confirmation > 5000:
            raise ValueError("Confirmation must use between the declared minimum and 5000 scenes per regime.")
        for name in ("ci_half_width", "minimum_effect", "count_harm_margin"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not self.frame_lengths or any(t not in (3, 4) for t in self.frame_lengths):
            raise ValueError("Main experiments use three or four frames.")
        grids = (self.position_scales, self.clutter_rates, self.detection_probabilities,
                 self.lifetime_kinds, self.discount_grid, self.strength_grid, self.count_rate_grid,
                 self.survival_grid, self.mixture_pairs, self.mixture_weight_grid)
        if any(not grid for grid in grids) or self.discount_grid[0] != 0:
            raise ValueError("Nonempty grids and an initial zero-discount candidate are required.")
        if any(x not in {"homogeneous", "heterogeneous"} for x in self.lifetime_kinds):
            raise ValueError("Unknown lifetime regime.")
        for scale in self.position_scales:
            MotionModel(position_std=scale)
        for rate in self.clutter_rates:
            SceneConfig(clutter_rate=rate)
        for pd in self.detection_probabilities:
            SurvivalModel(detection_probability=pd)
        for d in self.discount_grid:
            AssociationModel(discount=d)
        for rate in (*self.strength_grid, *self.count_rate_grid):
            AssociationModel(count_rate=rate)
        for s in self.survival_grid:
            SurvivalModel((s,), (1.0,))
        for pair, weight in product(self.mixture_pairs, self.mixture_weight_grid):
            if len(pair) != 2:
                raise ValueError("Mixture pairs require two probabilities.")
            SurvivalModel(pair, (weight, 1 - weight))
        if self.novelty_status not in {"unestablished", "defensible", "not_distinct"}:
            raise ValueError("Novelty status must be an explicit assessed state.")

    @classmethod
    def from_dict(cls, data: dict) -> FeasibilityConfig:
        tuple_fields = ("frame_lengths", "position_scales", "clutter_rates", "detection_probabilities",
                        "lifetime_kinds", "discount_grid", "strength_grid", "count_rate_grid",
                        "survival_grid", "mixture_weight_grid")
        data = dict(data)
        for name in tuple_fields:
            if name in data:
                data[name] = tuple(data[name])
        if "mixture_pairs" in data:
            data["mixture_pairs"] = tuple(tuple(pair) for pair in data["mixture_pairs"])
        return cls(**data)


def _json_bytes(data: object) -> bytes:
    return (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def digest(data: object) -> str:
    return sha256(_json_bytes(data)).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_new(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(_json_bytes(data))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _provenance() -> dict:
    source = Path(__file__).parent
    files = {path.name: sha256(path.read_bytes()).hexdigest() for path in sorted(source.glob("*.py"))}
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {"source_sha256": digest(files), "source_files": files, "git_revision": revision,
            "python": platform.python_version(), "numpy": np.__version__, "hostname": platform.node()}


def scene_seed(master_seed: int, phase: int, regime: int, index: int) -> int:
    if not (0 <= phase < 256 and 0 <= regime < 4096 and 0 <= index < 2**20):
        raise ValueError("Seed namespace exceeded.")
    return (master_seed << 40) | (phase << 32) | (regime << 20) | index


def regimes(config: FeasibilityConfig) -> list[tuple[str, SceneConfig]]:
    rows = []
    for length, scale, clutter, pd, kind in product(config.frame_lengths, config.position_scales,
                                                 config.clutter_rates, config.detection_probabilities, config.lifetime_kinds):
        survival = SurvivalModel((0.8,), (1.0,), pd) if kind == "homogeneous" else SurvivalModel((0.4, 0.95), (0.5, 0.5), pd)
        name = f"t{length}-x{scale}-c{clutter}-pd{pd}-{kind}"
        rows.append((name, SceneConfig(length, MotionModel(position_std=scale), survival, clutter)))
    if len(rows) >= 4096 or len({name for name, _ in rows}) != len(rows):
        raise ValueError("Regimes must have unique names and fit the seed namespace.")
    return rows


def model_from_dict(data: dict) -> AssociationModel:
    data = dict(data)
    survival = dict(data["survival"])
    survival["probabilities"] = tuple(survival["probabilities"])
    survival["weights"] = tuple(survival["weights"])
    data["survival"] = SurvivalModel(**survival)
    data["oracle_population"] = tuple(data["oracle_population"])
    return AssociationModel(**data)


def _fit_models(config: FeasibilityConfig, scene_config: SceneConfig, scene_list: list) -> tuple[dict, list]:
    prepared = [prepare_window(scene.observations, scene_config.motion) for scene in scene_list]
    truth_indices = [item.topology.label_indices[scene.truth] for item, scene in zip(prepared, scene_list)]
    total_observations = sum(len(scene.truth) for scene in scene_list)
    total_target = sum(sum(label >= 0 for label in scene.truth) for scene in scene_list)
    target_probability = (total_target + 1) / (total_observations + 2)

    def count_loss(rate):
        scores = []
        for scene in scene_list:
            target_counts = [sum(label >= 0 for label in scene.truth[sum(scene.observations.frame_counts[:t]):sum(scene.observations.frame_counts[:t+1])])
                             for t in range(scene_config.num_frames)]
            k = max(scene.truth, default=-1) + 1
            feasible = range(max(target_counts), sum(target_counts) + 1)
            weights = np.asarray([j * log(rate) - lgamma(j + 1) for j in feasible])
            scores.append(logsumexp(weights) - (k * log(rate) - lgamma(k + 1)))
        return float(np.mean(scores))

    rate = min(config.count_rate_grid, key=count_loss)
    base = AssociationModel(target_probability=target_probability, count_rate=rate)
    evaluated = []

    def loss(model):
        values = []
        for item, index in zip(prepared, truth_indices):
            joint = _log_prior(item.observations.frame_counts, model) + item.log_likelihood
            values.append((logsumexp(joint) - float(joint[index])) / max(sum(item.observations.frame_counts), 1))
        value = float(np.mean(values))
        evaluated.append({"model": asdict(model), "development_association_nll": value})
        return value

    def select(models):
        scored = [(loss(model), i, model) for i, model in enumerate(models)]
        return min(scored, key=lambda row: (round(row[0], 12), row[1]))[2]

    pd = scene_config.survival.detection_probability
    homogeneous = [SurvivalModel((s,), (1.0,), pd) for s in config.survival_grid]
    mixed = homogeneous + [SurvivalModel(pair, (weight, 1 - weight), pd)
                           for pair, weight in product(config.mixture_pairs, config.mixture_weight_grid)]
    models = {}
    for name, survival_options in (("homogeneous", homogeneous), ("mixed", mixed)):
        models[name] = select([replace(base, survival=s) for s in survival_options])
        # Joint fitting includes every baseline configuration at d=0, so gains
        # cannot come merely from freezing a poorly tuned survival nuisance.
        models[name + "_discount"] = select([replace(base, survival=s, discount=d)
                                             for d in config.discount_grid for s in survival_options])
    models["dp"] = select([replace(base, kind="dp", strength=theta) for theta in config.strength_grid])
    models["pyp"] = select([replace(base, kind="pyp", strength=theta, discount=d)
                             for d in config.discount_grid for theta in config.strength_grid])
    models["oracle"] = replace(base, kind="oracle", survival=scene_config.survival,
                               oracle_population=scene_config.population_probabilities,
                               oracle_clutter_rate=scene_config.clutter_rate)
    models["mixed_discount_off"] = replace(models["mixed_discount"], discount=0)
    return {name: asdict(model) for name, model in models.items()}, evaluated


def _scene_scores(scene, scene_config, models):
    start = perf_counter()
    prepared = prepare_window(scene.observations, scene_config.motion)
    preparation_seconds = perf_counter() - start
    scores = {}
    for name, data in models.items():
        start = perf_counter()
        posterior = infer_prepared(prepared, model_from_dict(data))
        inference_seconds = perf_counter() - start
        scores[name] = {**score_associations(posterior, scene.truth), "inference_seconds": inference_seconds,
                        "log_evidence": posterior.log_evidence}
    return {"seed": scene.seed, "frame_counts": scene.observations.frame_counts,
            "truth": scene.truth, "rejected_oversized": scene.rejected_oversized,
            "likelihood_seconds": preparation_seconds, "scores": scores}


def _run_regime(task):
    phase, index, name, scene_config, config, count, models, output, fingerprint = task
    output = Path(output)
    if output.exists():
        existing = _read(output)
        if existing["fingerprint"] != fingerprint or existing["count"] != count:
            raise ValueError(f"Refusing to reuse incompatible result: {output}")
        return name
    scenes = [simulate_association_scene(scene_config, scene_seed(config.master_seed, phase, index, i)) for i in range(count)]
    data = {"regime": name, "count": count, "fingerprint": fingerprint, "scene_config": asdict(scene_config)}
    if phase == 1:
        fitted, candidates = _fit_models(config, scene_config, scenes)
        data.update(models=fitted, candidates=candidates,
                    seeds=[scene.seed for scene in scenes], rejected_oversized=sum(s.rejected_oversized for s in scenes))
    else:
        data["rows"] = [_scene_scores(scene, scene_config, models) for scene in scenes]
    _write_new(output, data)
    return name


def paired_summary(values: list[float]) -> dict:
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("At least two finite paired observations are required.")
    mean = float(np.mean(values))
    variance = float(np.var(values, ddof=1))
    half = 1.96 * sqrt(variance / len(values))
    return {"n": len(values), "mean": mean, "variance": variance, "ci95": [mean - half, mean + half]}


def balanced_summary(summaries: list[dict]) -> dict:
    if not summaries:
        raise ValueError("No regimes to summarize.")
    mean = float(np.mean([s["mean"] for s in summaries]))
    se = sqrt(sum(s["variance"] / s["n"] for s in summaries)) / len(summaries)
    return {"regimes": len(summaries), "mean": mean, "ci95": [mean - 1.96 * se, mean + 1.96 * se]}


def confirmation_size(pilot_variance: float, config: FeasibilityConfig) -> tuple[int, bool]:
    if not np.isfinite(pilot_variance) or pilot_variance < 0:
        raise ValueError("Pilot variance must be finite and nonnegative.")
    requested = max(config.minimum_confirmation, ceil(1.96**2 * pilot_variance / config.ci_half_width**2))
    return min(requested, config.maximum_confirmation), requested > config.maximum_confirmation


def _contrasts(rows):
    result = {}
    for baseline in ("homogeneous", "mixed"):
        for metric in ("association_nll", "count_nll"):
            key = baseline + ":" + metric
            result[key] = paired_summary([r["scores"][baseline][metric] - r["scores"][baseline + "_discount"][metric] for r in rows])
    result["fixed_nuisance:association_nll"] = paired_summary([
        r["scores"]["mixed_discount_off"]["association_nll"] - r["scores"]["mixed_discount"]["association_nll"] for r in rows])
    return result


def decide(config: FeasibilityConfig, primary: dict, count_gain: dict, heterogeneous: dict,
           fixed_nuisance: dict) -> dict:
    if config.novelty_status == "not_distinct":
        return {"decision": "stop", "reason": "The proposed contribution is not distinct from prior work."}
    if primary["ci95"][1] < config.minimum_effect:
        return {"decision": "stop", "reason": "The confirmed primary interval excludes the minimum worthwhile gain in the declared benchmark."}
    if count_gain["ci95"][1] < -config.count_harm_margin:
        return {"decision": "stop", "reason": "Detected-count log score violates the harm margin."}
    empirical_pass = (primary["mean"] >= config.minimum_effect and primary["ci95"][0] > 0
                      and heterogeneous["ci95"][0] > 0 and fixed_nuisance["ci95"][0] > 0
                      and count_gain["ci95"][0] >= -config.count_harm_margin)
    if empirical_pass and config.novelty_status == "defensible":
        return {"decision": "continue", "reason": "The prespecified empirical and novelty gates pass; full tracking remains untested."}
    if empirical_pass:
        return {"decision": "inconclusive", "reason": "The empirical gate passes, but novelty is not established."}
    return {"decision": "inconclusive", "reason": "Uncertainty or the heterogeneous-survival control prevents a continue decision."}


def run_study(config: FeasibilityConfig, output_dir: Path, *, stage: str = "all", workers: int = 1) -> dict:
    if stage not in {"development", "pilot", "lock", "confirmation", "report", "all"}:
        raise ValueError("Unknown study stage.")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer.")
    output_dir = Path(output_dir)
    provenance = _provenance()
    identity = {"config": asdict(config), "source_sha256": provenance["source_sha256"]}
    fingerprint = digest(identity)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        if _read(manifest_path)["fingerprint"] != fingerprint:
            raise ValueError("Configuration or inference source changed; use a new output directory.")
    else:
        _write_new(manifest_path, {**identity, "fingerprint": fingerprint, "provenance": provenance,
                                   "created_utc": datetime.now(timezone.utc).isoformat()})
    cells = regimes(config)

    def dispatch(phase, counts, parameters=None, lock_digest=None):
        directory = {1: "development", 2: "pilot", 3: "confirmation", 4: "negative_control"}[phase]
        tasks = []
        for i, (name, scene_config) in enumerate(cells):
            if phase == 4:
                scene_config = replace(scene_config, num_frames=2)
            parameters_for_cell = parameters[name] if parameters is not None else None
            task_identity = digest({"fingerprint": fingerprint, "phase": phase, "regime": name,
                                    "models": parameters_for_cell, "lock": lock_digest})
            tasks.append((phase, i, name, scene_config, config, counts[name], parameters_for_cell,
                          str(output_dir / directory / (name + ".json")), task_identity))
        if workers == 1:
            for task in tasks:
                print(f"{directory}: {_run_regime(task)}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for name in pool.map(_run_regime, tasks):
                    print(f"{directory}: {name}", flush=True)

    if stage in {"development", "all"}:
        dispatch(1, {name: config.development_scenes for name, _ in cells})
        if stage == "development":
            return {"stage": stage, "regimes": len(cells)}
    development = {name: _read(output_dir / "development" / (name + ".json")) for name, _ in cells}
    parameters = {name: data["models"] for name, data in development.items()}
    if stage in {"pilot", "all"}:
        dispatch(2, {name: config.pilot_scenes for name, _ in cells}, parameters)
        dispatch(4, {name: config.negative_control_scenes for name, _ in cells}, parameters)
        if stage == "pilot":
            return {"stage": stage, "regimes": len(cells)}
    pilot = {name: _read(output_dir / "pilot" / (name + ".json")) for name, _ in cells}
    lock_path = output_dir / "confirmation-lock.json"
    if stage in {"lock", "all"} and not lock_path.exists():
        allocation = {}
        for name, data in pilot.items():
            contrast = _contrasts(data["rows"])
            variance = max(contrast["mixed:association_nll"]["variance"], contrast["mixed:count_nll"]["variance"])
            count, capped = confirmation_size(variance, config)
            allocation[name] = {"scenes": count, "capped": capped, "pilot_variance": variance}
        payload = {"fingerprint": fingerprint, "parameters": parameters, "allocation": allocation,
                   "pilot_sha256": digest(pilot), "development_sha256": digest(development),
                   "primary": "balanced mixed minus mixed_discount association_nll",
                   "created_utc": datetime.now(timezone.utc).isoformat()}
        _write_new(lock_path, {"payload": payload, "sha256": digest(payload)})
    lock = _read(lock_path)
    payload = lock["payload"]
    if (digest(payload) != lock["sha256"] or payload["fingerprint"] != fingerprint
            or payload["parameters"] != parameters or payload["pilot_sha256"] != digest(pilot)
            or payload["development_sha256"] != digest(development)):
        raise ValueError("Confirmation lock no longer matches development, pilot, or source.")
    if stage == "lock":
        return {"stage": stage, "lock_sha256": lock["sha256"], "allocation": payload["allocation"]}
    if stage in {"confirmation", "all"}:
        dispatch(3, {name: data["scenes"] for name, data in payload["allocation"].items()}, parameters, lock["sha256"])
        if stage == "confirmation":
            return {"stage": stage, "lock_sha256": lock["sha256"]}
    confirmation = {name: _read(output_dir / "confirmation" / (name + ".json")) for name, _ in cells}
    for i, (name, _) in enumerate(cells):
        data = confirmation[name]
        expected = digest({"fingerprint": fingerprint, "phase": 3, "regime": name,
                           "models": parameters[name], "lock": lock["sha256"]})
        seeds = [scene_seed(config.master_seed, 3, i, j) for j in range(payload["allocation"][name]["scenes"])]
        if data["fingerprint"] != expected or [r["seed"] for r in data["rows"]] != seeds:
            raise ValueError("Confirmation records do not match the locked models, allocation, or seed namespace.")
    by_regime = {name: _contrasts(data["rows"]) for name, data in confirmation.items()}
    aggregate = {key: balanced_summary([data[key] for data in by_regime.values()]) for key in next(iter(by_regime.values()))}
    hetero = [data["mixed:association_nll"] for name, data in by_regime.items() if name.endswith("-heterogeneous")]
    heterogeneous = balanced_summary(hetero) if hetero else {"mean": 0, "ci95": [-1e100, 1e100], "regimes": 0}
    negative_controls = {name: _read(output_dir / "negative_control" / (name + ".json")) for name, _ in cells}
    control_error = 0.0
    # Use the same fitted nuisance parameters on both sides: independently
    # refitted survival models need not be identical in a two-frame test.
    for i, (name, scene_config) in enumerate(cells):
        candidate = model_from_dict(parameters[name]["mixed_discount"])
        for j in range(min(5, config.negative_control_scenes)):
            scene = simulate_association_scene(replace(scene_config, num_frames=2), scene_seed(config.master_seed, 4, i, j))
            item = prepare_window(scene.observations, scene_config.motion)
            a = infer_prepared(item, candidate)
            b = infer_prepared(item, replace(candidate, discount=0))
            control_error = max(control_error, float(np.max(np.abs(a.probabilities - b.probabilities))))
    if control_error > 1e-10:
        raise RuntimeError("The two-frame posterior-equivalence control failed.")
    model_means = {model: {metric: float(np.mean([np.mean([r["scores"][model][metric] for r in data["rows"]])
                                                  for data in confirmation.values()]))
                           for metric in ("association_nll", "count_nll", "count_brier", "count_coverage_90", "association_error", "inference_seconds")}
                   for model in next(iter(parameters.values()))}
    result = {"schema_version": 1, "fingerprint": fingerprint, "lock_sha256": lock["sha256"],
              "decision": decide(config, aggregate["mixed:association_nll"], aggregate["mixed:count_nll"], heterogeneous,
                                 aggregate["fixed_nuisance:association_nll"]),
              "aggregate": aggregate, "heterogeneous_primary": heterogeneous, "by_regime": by_regime,
              "model_means": model_means, "confirmation_scenes": sum(data["count"] for data in confirmation.values()),
              "rejected_oversized": sum(r["rejected_oversized"] for data in confirmation.values() for r in data["rows"]),
              "zero_observation_scenes": sum(not r["truth"] for data in confirmation.values() for r in data["rows"]),
              "negative_control_max_probability_error": control_error,
              "negative_control_sha256": digest(negative_controls), "confirmation_sha256": digest(confirmation),
              "capped_regimes": [name for name, data in payload["allocation"].items() if data["capped"]],
              "scope": "Conditional finite-window simulation; not real-data tracking or live-cardinality evidence."}
    report_path = output_dir / "decision.json"
    if report_path.exists():
        if digest(_read(report_path)) != digest(result):
            raise ValueError("Existing decision differs; preserve it and investigate the inputs.")
    else:
        _write_new(report_path, result)
    return result
