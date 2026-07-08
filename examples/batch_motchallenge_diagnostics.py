"""Run count diagnostics for every MOTChallenge sequence under a benchmark root.

Example:

    python examples/batch_motchallenge_diagnostics.py MOT17 --output-dir reports/mot17
"""

from __future__ import annotations

import argparse
import json

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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    reports = reports_from_motchallenge_root(
        args.mot_root,
        min_confidence=args.min_confidence,
        include_empty_frames=not args.drop_empty_frames,
    )
    output = reports_to_dict(reports)
    if args.output_dir is not None:
        output_paths = write_benchmark_report_outputs(reports, args.output_dir)
        output["output_paths"] = {name: str(path) for name, path in output_paths.items()}
    if not args.quiet:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
