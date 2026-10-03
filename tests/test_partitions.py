from math import prod

import pytest

from pitman_yor_dp import DirichletProcessCardinalityPrior, PitmanYorCardinalityPrior
from pitman_yor_dp.partitions import (
    conditional_partition_pmf,
    match_cluster_count_distribution,
    valid_partitions,
)


def test_two_frame_enumeration_has_each_partial_matching_once():
    partitions = valid_partitions((2, 2))
    assert len(partitions) == len(set(partitions)) == 7
    assert sum(len(set(partition)) == 2 for partition in partitions) == 2
    assert sum(len(set(partition)) == 3 for partition in partitions) == 4
    assert all(p[0] != p[1] and p[2] != p[3] for p in partitions)


def test_unrestricted_three_observation_probabilities_agree_with_crp():
    prior = PitmanYorCardinalityPrior(1.3, 0.4)
    pmf = conditional_partition_pmf(prior, (1, 1, 1))
    assert len(pmf) == 5
    for partition, probability in pmf.items():
        sizes = []
        predictive = []
        for label in partition:
            predictive.append(prior.predictive_probabilities(sizes)[label])
            if label == len(sizes):
                sizes.append(1)
            else:
                sizes[label] += 1
        assert probability == pytest.approx(prod(predictive))


@pytest.mark.parametrize("frame_counts", [(3,), (2, 2), (2, 2, 1)])
def test_zero_discount_is_dp_and_pmf_normalizes(frame_counts):
    dp = conditional_partition_pmf(DirichletProcessCardinalityPrior(2), frame_counts)
    pyp = conditional_partition_pmf(PitmanYorCardinalityPrior(2, 0), frame_counts)
    assert dp == pytest.approx(pyp)
    assert sum(dp.values()) == pytest.approx(1)


@pytest.mark.parametrize("theta,discount", [(1, 0), (2, 0.5), (-0.2, 0.7)])
def test_two_frame_partition_law_given_k_is_uniform(theta, discount):
    prior = PitmanYorCardinalityPrior(theta, discount)
    for k in range(3, 6):
        pmf = conditional_partition_pmf(prior, (3, 2), num_clusters=k)
        assert list(pmf.values()) == pytest.approx([1 / len(pmf)] * len(pmf))


def test_entire_cardinality_matching_removes_two_frame_discount_effect():
    dp = conditional_partition_pmf(DirichletProcessCardinalityPrior(1), (3, 3))
    pyp = conditional_partition_pmf(PitmanYorCardinalityPrior(1, 0.5), (3, 3))
    assert dp != pytest.approx(pyp)
    assert match_cluster_count_distribution(dp, pyp) == pytest.approx(pyp)


def test_three_frames_allow_a_discount_effect_at_fixed_k():
    # One particular (3,1,1) partition versus one (2,2,1) partition, both valid.
    long_track = (0, 1, 0, 2, 0)
    two_pairs = (0, 1, 0, 1, 2)
    for theta in (0.3, 3.0):
        for discount in (0.0, 0.5, 0.8):
            pmf = conditional_partition_pmf(PitmanYorCardinalityPrior(theta, discount), (2, 2, 1), num_clusters=3)
            assert pmf[long_track] / pmf[two_pairs] == pytest.approx((2 - discount) / (1 - discount))
    dp = conditional_partition_pmf(DirichletProcessCardinalityPrior(1), (2, 2, 1))
    pyp = conditional_partition_pmf(PitmanYorCardinalityPrior(1, 0.5), (2, 2, 1))
    assert match_cluster_count_distribution(dp, pyp) != pytest.approx(pyp)


def test_empty_exposure_and_invalid_requests():
    prior = PitmanYorCardinalityPrior()
    assert conditional_partition_pmf(prior, (0, 0)) == {(): 1.0}
    for counts in ((-1,), (10,), (True,)):
        with pytest.raises((TypeError, ValueError)):
            valid_partitions(counts)
    with pytest.raises(ValueError):
        conditional_partition_pmf(prior, (2, 1), num_clusters=1)
