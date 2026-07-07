"""Run count diagnostics from a simple tracking CSV.

Expected default columns:

    frame,track_id

Optional columns:

    confidence,is_ground_truth

Example:

    python examples/count_diagnostics_from_tracks.py path/to/tracks.csv \
        --frame-key frame --track-id-key track_id
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from pitman_yor_dp import (
    compare_count_models,
    comparison_to_dict,
    count_series_from_records,
    records_from_csv,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Path to a CSV file with frame-level track annotations.")
    parser.add_argument("--frame-key", default="frame", help="Frame-index column name.")
    parser.add_argument("--track-id-key", default="track_id", help="Track identity column name.")
    parser.add_argument("--confidence-key", default="confidence", help="Confidence column name; use an empty string to disable.")
    parser.add_argument("--is-ground-truth-key", default="", help="Optional ground-truth flag column name.")
    parser.add_argument("--min-confidence", type=float, default=None, help="Optional minimum confidence filter.")
    parser.add_argument("--require-ground-truth", action="store_true", help="Keep only rows whose ground-truth flag is true.")
    parser.add_argument("--drop-empty-frames", action="store_true", help="Do not include empty frames between first and last observation.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    confidence_key = args.confidence_key if args.confidence_key else None
    is_ground_truth_key = args.is_ground_truth_key if args.is_ground_truth_key else None
    require_ground_truth = True if args.require_ground_truth else None

    records = records_from_csv(
        args.csv_path,
        frame_key=args.frame_key,
        track_id_key=args.track_id_key,
        confidence_key=confidence_key,
        is_ground_truth_key=is_ground_truth_key,
        min_confidence=args.min_confidence,
        require_ground_truth=require_ground_truth,
    )
    bundle = count_series_from_records(records, include_empty_frames=not args.drop_empty_frames)
    comparisons = (
        compare_count_models(bundle.live_counts, count_name="N_t_live_count"),
        compare_count_models(bundle.birth_counts, count_name="B_t_birth_count"),
    )
    lifetime_comparison = None
    if bundle.lifetimes:
        lifetime_comparison = compare_count_models(bundle.lifetimes, count_name="L_k_lifetime")

    output = {
        "num_records": len(records),
        "num_frames": len(bundle.frames),
        "frames": list(bundle.frames),
        "live_counts": list(bundle.live_counts),
        "birth_counts": list(bundle.birth_counts),
        "lifetimes": list(bundle.lifetimes),
        "count_model_comparisons": [comparison_to_dict(comparison) for comparison in comparisons],
    }
    if lifetime_comparison is not None:
        output["count_model_comparisons"].append(comparison_to_dict(lifetime_comparison))
    print(json.dumps(output, indent=2, default=asdict))


if __name__ == "__main__":
    main()
