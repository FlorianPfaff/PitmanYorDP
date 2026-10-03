from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from pitman_yor_dp.feasibility import (
    FeasibilityConfig, balanced_summary, confirmation_size, decide, paired_summary,
    regimes, run_study, scene_seed,
)


def tiny_config():
    return FeasibilityConfig(development_scenes=3, pilot_scenes=3, minimum_confirmation=3,
                             maximum_confirmation=4, negative_control_scenes=2,
                             frame_lengths=(3,), position_scales=(0.75,), clutter_rates=(0.05,),
                             detection_probabilities=(0.65,), lifetime_kinds=("heterogeneous",),
                             discount_grid=(0, 0.5), strength_grid=(1,), count_rate_grid=(1,),
                             survival_grid=(0.8,), mixture_pairs=((0.4, 0.95),), mixture_weight_grid=(0.5,))


def test_seed_namespaces_are_disjoint():
    seeds = [scene_seed(20261003, phase, regime, i) for phase in (1, 2, 3, 4) for regime in range(32) for i in range(10)]
    assert len(seeds) == len(set(seeds))
    assert scene_seed(1, 1, 0, 0) != scene_seed(2, 1, 0, 0)


def test_config_roundtrip_and_full_factorial():
    config = FeasibilityConfig()
    assert FeasibilityConfig.from_dict(json.loads(json.dumps(asdict(config)))) == config
    assert len(regimes(config)) == 32
    with pytest.raises(ValueError):
        replace(config, frame_lengths=(3, 3))
        regimes(replace(config, frame_lengths=(3, 3)))


def test_balanced_estimator_does_not_weight_large_cells_more():
    one = paired_summary([1, 1])
    two = paired_summary([3] * 20)
    result = balanced_summary([one, two])
    assert result["mean"] == 2
    assert result["ci95"] == [2, 2]
    assert paired_summary([-1, 1])["ci95"] == pytest.approx([-1.96, 1.96])


def test_confirmation_allocation_uses_variance_and_respects_cap():
    config = FeasibilityConfig()
    assert confirmation_size(0, config) == (200, False)
    assert confirmation_size(100, config) == (5000, True)
    assert confirmation_size(0.01, config)[0] == 385
    with pytest.raises(ValueError):
        confirmation_size(float("nan"), config)


def test_decision_requires_novelty_gain_and_harm_controls():
    config = FeasibilityConfig()
    good = {"mean": 0.04, "ci95": [0.03, 0.05]}
    neutral = {"mean": 0.0, "ci95": [-0.01, 0.01]}
    assert decide(config, good, neutral, good, good)["decision"] == "inconclusive"
    assert decide(replace(config, novelty_status="defensible"), good, neutral, good, good)["decision"] == "continue"
    assert decide(config, neutral, neutral, neutral, neutral)["decision"] == "stop"
    harm = {"mean": -0.04, "ci95": [-0.05, -0.03]}
    assert decide(config, good, harm, good, good)["decision"] == "stop"
    ambiguous = {"mean": 0.025, "ci95": [-0.01, 0.06]}
    assert decide(config, ambiguous, neutral, good, good)["decision"] == "inconclusive"
    assert decide(replace(config, novelty_status="defensible"), good, neutral, good, neutral)["decision"] == "inconclusive"


def test_end_to_end_lock_resume_and_per_scene_provenance(tmp_path):
    config = tiny_config()
    result = run_study(config, tmp_path)
    assert result["confirmation_scenes"] in (3, 4)
    assert result["negative_control_max_probability_error"] < 1e-10
    assert result == run_study(config, tmp_path, stage="report")
    assert result == run_study(config, tmp_path)
    lock = json.loads((tmp_path / "confirmation-lock.json").read_text())
    assert result["lock_sha256"] == lock["sha256"]
    name = regimes(config)[0][0]
    sets = []
    for phase in ("development", "pilot", "confirmation", "negative_control"):
        data = json.loads((tmp_path / phase / (name + ".json")).read_text())
        sets.append(set(data["seeds"] if phase == "development" else (r["seed"] for r in data["rows"])))
    assert sum(map(len, sets)) == len(set.union(*sets))
    with pytest.raises(ValueError, match="Configuration or inference source changed"):
        run_study(replace(config, minimum_effect=0.03), tmp_path)
    lock["payload"]["primary"] = "different endpoint"
    (tmp_path / "confirmation-lock.json").write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="Confirmation lock"):
        run_study(config, tmp_path, stage="report")


def test_confirmation_cannot_run_before_a_lock(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_study(tiny_config(), tmp_path, stage="confirmation")


@pytest.mark.parametrize("change", [
    {"minimum_confirmation": 0}, {"maximum_confirmation": 5001}, {"discount_grid": (0.5,)},
    {"ci_half_width": 0}, {"discount_grid": (0, float("nan"))}, {"frame_lengths": (5,)},
    {"detection_probabilities": (1,)}, {"lifetime_kinds": ("unknown",)}, {"master_seed": True},
])
def test_bad_configs_fail(change):
    with pytest.raises(ValueError):
        replace(FeasibilityConfig(), **change)
