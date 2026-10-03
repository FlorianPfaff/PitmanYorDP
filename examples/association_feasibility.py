"""Run exact association development, pilot, lock, confirmation, and reporting."""

import argparse
import json
from pathlib import Path

from pitman_yor_dp.feasibility import FeasibilityConfig, run_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("development", "pilot", "lock", "confirmation", "report", "all"), default="all")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    config = FeasibilityConfig.from_dict(json.loads(args.config.read_text()))
    result = run_study(config, args.output_dir, stage=args.stage, workers=args.workers)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
