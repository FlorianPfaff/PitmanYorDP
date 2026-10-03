"""Run count diagnostics for every MOTChallenge sequence under a benchmark root.

Example:

    python examples/batch_motchallenge_diagnostics.py MOT17 --output-dir reports/mot17
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pitman_yor_dp import (
    reports_from_motchallenge_root,
    reports_to_dict,
    write_benchmark_report_outputs,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mot_root", help="Benchmark root containing sequence directories with gt/gt.txt files.")
    parser.add_argument("--output-dir", default=None, help="Optional directory for CSV outputs.")
    parser.add_argument("--min-confidence", type=float, default=None, help="Optional minimum confidence filter.")
    parser.add_argument(
        "--drop-empty-frames",
        action="store_true",
        help="Do not include empty frames between first and last observation.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress JSON output on stdout.")
    parser.add_argument("--mot17-ground-truth", action="store_true", help="Require seqinfo.ini; keep marked pedestrians and deduplicate detector variants.")
    parser.add_argument("--json-output", type=Path, help="Save the full audited report as JSON.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    reports = reports_from_motchallenge_root(
        args.mot_root,
        min_confidence=args.min_confidence,
        include_empty_frames=not args.drop_empty_frames,
        mot17_ground_truth=args.mot17_ground_truth,
    )
    output = reports_to_dict(reports)
    if args.output_dir is not None:
        output_paths = write_benchmark_report_outputs(reports, args.output_dir)
        output["output_paths"] = {name: str(path) for name, path in output_paths.items()}
    serialized = json.dumps(output, indent=2, allow_nan=False)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(serialized + "\n")
    if not args.quiet:
        print(serialized)


if __name__ == "__main__":
    main()
