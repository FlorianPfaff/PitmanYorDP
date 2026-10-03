from dataclasses import replace
from math import exp, log, pi

import numpy as np
import pytest

from pitman_yor_dp.association import (
    AssociationModel, MotionModel, ObservationWindow, SurvivalModel, _log_prior,
    _topology, canonical_labels, infer_associations, observation_pattern_probabilities,
    prepare_window, score_associations, trajectory_covariance,
)
from pitman_yor_dp.association_scenes import SceneConfig, simulate_association_scene


def window(counts):
    return ObservationWindow(tuple(float(t) for t in range(len(counts))),
                             tuple(tuple((float(i), float(t) / 2) for i in range(n)) for t, n in enumerate(counts)))


@pytest.mark.parametrize("length", [1, 2, 3, 4])
def test_pattern_probabilities_integrate_birth_death_and_detection(length):
    model = SurvivalModel((0.4, 0.95), (0.3, 0.7), 0.65)
    probabilities = observation_pattern_probabilities(length, model)
    assert sum(probabilities) == pytest.approx(1)
    assert all(p > 0 for p in probabilities)
    assert observation_pattern_probabilities(1, model) == pytest.approx((0.35, 0.65))


@pytest.mark.parametrize("counts", [(0,), (2, 2), (2, 2, 1), (1, 1, 1, 1)])
@pytest.mark.parametrize("kind", ["survival", "dp", "pyp", "oracle"])
def test_prior_and_posterior_normalize(counts, kind):
    model = AssociationModel(kind=kind, discount=0.5)
    prior = np.exp(_log_prior(counts, model))
    posterior = infer_associations(window(counts), model)
    assert prior.sum() == pytest.approx(1)
    assert posterior.probabilities.sum() == pytest.approx(1)
    assert sum(posterior.count_probabilities) == pytest.approx(1)
    topology = _topology(counts)
    assert len(set(topology.hypotheses)) == len(topology.hypotheses)
    for hypothesis in topology.hypotheses:
        for frame in set(topology.frames):
            labels = [label for label, t in zip(hypothesis, topology.frames) if t == frame and label >= 0]
            assert len(labels) == len(set(labels))


def test_full_count_law_and_target_subset_law_are_preserved():
    counts = (2, 2, 1)
    topology = _topology(counts)
    model = AssociationModel(survival=SurvivalModel((0.3, 0.9), (0.4, 0.6)))
    baseline = np.exp(_log_prior(counts, model))
    candidate = np.exp(_log_prior(counts, replace(model, discount=0.7)))
    assert not np.allclose(baseline, candidate)
    for groups in topology.count_groups:
        for group in groups:
            assert baseline[group].sum() == pytest.approx(candidate[group].sum(), abs=1e-13)
    for mask, group in enumerate(topology.mask_groups):
        targets = mask.bit_count()
        assert baseline[group].sum() == pytest.approx(model.target_probability**targets * (1 - model.target_probability)**(5 - targets))


@pytest.mark.parametrize("counts", [(3,), (3, 3), (2, 0)])
def test_discount_cancels_for_at_most_two_frames_including_latent_clutter(counts):
    observations = window(counts)
    baseline = AssociationModel(survival=SurvivalModel((0.3, 0.95), (0.5, 0.5)))
    a = infer_associations(observations, baseline)
    b = infer_associations(observations, replace(baseline, discount=0.8))
    assert a.probabilities == pytest.approx(b.probabilities, abs=1e-12)
    assert a.log_evidence == pytest.approx(b.log_evidence)


def test_three_frame_residual_odds_and_zero_discount():
    counts = (2, 2, 1)
    topology = _topology(counts)
    a = topology.hypotheses.index((0, 1, 0, 2, 0))
    b = topology.hypotheses.index((0, 1, 0, 1, 2))
    baseline = _log_prior(counts, AssociationModel())
    candidate = _log_prior(counts, AssociationModel(discount=0.5))
    assert exp(candidate[a] - candidate[b] - baseline[a] + baseline[b]) == pytest.approx(1.5)
    assert _log_prior(counts, AssociationModel(discount=0)) == pytest.approx(baseline)


def test_single_measurement_matches_hand_calculation():
    observations = ObservationWindow((0.0,), (((0.0, 0.0),),))
    motion = MotionModel(position_std=2, measurement_std=1, clutter_std=4)
    model = AssociationModel(target_probability=0.6)
    posterior = infer_associations(observations, model, motion)
    target = 0.6 / (2 * pi * 5)
    clutter = 0.4 / (2 * pi * 16)
    assert posterior.log_evidence == pytest.approx(log(target + clutter))
    assert posterior.count_probabilities == pytest.approx((clutter / (target + clutter), target / (target + clutter)))


def test_oracle_single_measurement_matches_population_and_clutter_odds():
    observations = ObservationWindow((0.0,), (((0.0, 0.0),),))
    survival = SurvivalModel(detection_probability=0.75)
    model = AssociationModel(kind="oracle", survival=survival, oracle_population=(0.5, 0.5), oracle_clutter_rate=0.2)
    probability = np.exp(_log_prior((1,), model))
    target = 0.5 * 0.75
    clutter = (0.5 + 0.5 * 0.25) * 0.2
    assert probability == pytest.approx((clutter / (target + clutter), target / (target + clutter)))
    posterior = infer_associations(observations, model)
    assert np.isfinite(score_associations(posterior, (10,))["association_nll"])


def test_oracle_two_frame_combinatorics_and_undetected_targets():
    model = AssociationModel(kind="oracle", oracle_population=(0, 0, 1), oracle_clutter_rate=0)
    patterns = observation_pattern_probabilities(2, model.survival)
    topology = _topology((1, 1))
    prior = np.exp(_log_prior((1, 1), model))
    same = prior[topology.hypotheses.index((0, 0))]
    distinct = prior[topology.hypotheses.index((0, 1))]
    assert same / distinct == pytest.approx(patterns[0] * patterns[3] / (patterns[1] * patterns[2]))
    assert same + distinct == pytest.approx(1)


def test_optimized_eppf_matches_existing_partition_api():
    from pitman_yor_dp.cardinality import PitmanYorCardinalityPrior
    from pitman_yor_dp.partitions import conditional_partition_pmf
    counts = (2, 1, 1)
    model = AssociationModel(kind="pyp", strength=0.3, discount=0.6)
    topology = _topology(counts)
    logs = _log_prior(counts, model)
    complete = [(h, p) for h, p in zip(topology.hypotheses, np.exp(logs)) if -1 not in h]
    actual = {h: p / model.target_probability**4 for h, p in complete}
    expected = conditional_partition_pmf(PitmanYorCardinalityPrior(0.3, 0.6), counts)
    assert actual == pytest.approx(expected)


def test_empty_all_clutter_and_observation_gaps():
    posterior = infer_associations(window((0, 0, 0)))
    assert posterior.hypotheses == ((),)
    assert posterior.count_probabilities == (1,)
    assert posterior.log_evidence == 0
    assert score_associations(posterior, ())["association_nll"] == 0
    posterior = infer_associations(window((1, 0, 1)))
    assert np.isfinite(score_associations(posterior, (-1, -1))["association_nll"])
    assert np.isfinite(score_associations(posterior, (7, 7))["association_nll"])


def test_permutation_invariance_and_label_invariance():
    observations = window((2, 2, 1))
    changed = ObservationWindow(observations.frame_times, (observations.measurements[0][::-1], *observations.measurements[1:]))
    a = infer_associations(observations, AssociationModel(discount=0.6))
    b = infer_associations(changed, AssociationModel(discount=0.6))
    mapped = {canonical_labels((h[1], h[0], *h[2:])): p for h, p in zip(a.hypotheses, a.probabilities)}
    assert dict(zip(b.hypotheses, b.probabilities)) == pytest.approx(mapped)
    assert a.log_evidence == pytest.approx(b.log_evidence)
    assert score_associations(a, (0, 1, 0, 1, -1)) == score_associations(a, (17, 24, 17, 24, -1))


def test_likelihood_uses_actual_times_and_not_absolute_epoch():
    a = ObservationWindow((0, 1), (((0, 0),), ((2, 1),)))
    b = ObservationWindow((10, 11), a.measurements)
    c = ObservationWindow((0, 4), a.measurements)
    assert prepare_window(a).log_likelihood == pytest.approx(prepare_window(b).log_likelihood)
    assert not np.allclose(prepare_window(a).log_likelihood, prepare_window(c).log_likelihood)
    assert np.linalg.eigvalsh(trajectory_covariance((0, 1, 4), MotionModel())).min() > 0


def test_finite_simulator_is_reproducible_and_oracle_supports_truth():
    config = SceneConfig(survival=SurvivalModel((0.3, 0.95), (0.5, 0.5)))
    scene = simulate_association_scene(config, 42)
    assert scene == simulate_association_scene(config, 42)
    for seed in range(10):
        scene = simulate_association_scene(config, seed)
        oracle = AssociationModel(kind="oracle", survival=config.survival, oracle_clutter_rate=config.clutter_rate)
        assert np.isfinite(score_associations(infer_associations(scene.observations, oracle, config.motion), scene.truth)["association_nll"])


def test_explicit_rejection_is_reported_and_never_truncates():
    scene = simulate_association_scene(SceneConfig(num_frames=4, clutter_rate=3), 2)
    assert scene.rejected_oversized > 0
    assert len(scene.truth) <= 9
    with pytest.raises(RuntimeError):
        simulate_association_scene(SceneConfig(clutter_rate=1000, max_attempts=2), 1)


@pytest.mark.parametrize("factory", [
    lambda: window((10,)), lambda: window((0, 0, 0, 0, 0)),
    lambda: ObservationWindow((0, 0), ((), ())),
    lambda: ObservationWindow((0,), (((float("nan"), 0),),)),
    lambda: MotionModel(measurement_std=0), lambda: SurvivalModel((1,), (1,)),
    lambda: AssociationModel(discount=1), lambda: AssociationModel(target_probability=0),
    lambda: SceneConfig(population_probabilities=(0, 0)),
])
def test_invalid_inputs_fail(factory):
    with pytest.raises(ValueError):
        factory()
