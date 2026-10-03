"""Exact structural checks; these are not tracking-performance experiments."""

import argparse
import json
from pathlib import Path

from pitman_yor_dp import DirichletProcessCardinalityPrior, PitmanYorCardinalityPrior
from pitman_yor_dp.partitions import conditional_partition_pmf, match_cluster_count_distribution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = []
    for frame_counts in ((3,), (2, 2), (3, 3), (2, 2, 1), (2, 2, 2)):
        dp = conditional_partition_pmf(DirichletProcessCardinalityPrior(1), frame_counts)
        pyp = conditional_partition_pmf(PitmanYorCardinalityPrior(1, 0.5), frame_counts)
        matched = match_cluster_count_distribution(dp, pyp)
        rows.append({
            "frame_counts": frame_counts,
            "partitions": len(dp),
            "dp_expected_detected_identities": sum(len(set(p)) * q for p, q in dp.items()),
            "pyp_expected_detected_identities": sum(len(set(p)) * q for p, q in pyp.items()),
            "raw_total_variation": 0.5 * sum(abs(dp[p] - pyp[p]) for p in dp),
            "cardinality_matched_total_variation": 0.5 * sum(abs(matched[p] - pyp[p]) for p in dp),
        })
    result = {"kind": "exact_structural_prior_check", "strength": 1.0, "discount": 0.5, "rows": rows}
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
