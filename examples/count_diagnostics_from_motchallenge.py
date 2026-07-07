"""Run count diagnostics from a MOTChallenge-style annotation file.

Typical inputs are files such as ``gt/gt.txt`` with rows

    frame,id,bb_left,bb_top,bb_width,bb_height,conf,x,y,z

Example:

    python examples/count_diagnostics_from_motchallenge.py MOT17-02/gt/gt.txt \
        --dataset MOT17-02
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from pitman_yor_dp import (
    compare_count_models,
    comparison_to_dict,
    count_series_from_motchallenge_file,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mot_path", help="Path to a MOTChallenge-style gt.txt or det.txt file.")
    parser.add_argument("--dataset", default="MOTChallenge-sequence", help="Dataset/sequence label included in the JSON output.")
    parser.add_argument("--min-confidence", type=float, default=None, help="Optional minimum confidence filter.")
    parser.add_argument("--drop-empty-frames", action="store_true", help="Do not include empty frames between first and last observation.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    bundle = count_series_from_motchallenge_file(
        args.mot_path,
        min_confidence=args.min_confidence,
        include_empty_frames=not args.drop_empty_frames,
    )
    comparisons = (
        compare_count_models(bundle.live_counts, count_name="N_t_live_count"),
        compare_count_models(bundle.birth_counts, count_name="B_t_birth_count"),
        compare_count_models(bundle.lifetimes, count_name="L_k_lifetime"),
    )
    output = {
        "dataset": args.dataset,
        "num_frames": len(bundle.frames),
        "frames": list(bundle.frames),
        "live_counts": list(bundle.live_counts),
        "birth_counts": list(bundle.birth_counts),
        "lifetimes": list(bundle.lifetimes),
        "count_model_comparisons": [comparison_to_dict(comparison) for comparison in comparisons],
    }
    print(json.dumps(output, indent=2, default=asdict))


if __name__ == "__main__":
    main()
