"""Batch reports for benchmark tracking count diagnostics."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adapters import CountSeriesBundle
from .model_selection import CountModelComparison, compare_count_models, comparison_to_dict
from .motchallenge import count_series_from_motchallenge_file

COUNT_TYPES = ("N_t", "B_t", "L_k")
COUNT_DIAGNOSTIC_FIELDNAMES = ("sequence", "N_t", "B_t", "L_k")
MODEL_RECOMMENDATION_FIELDNAMES = (
    "sequence",
    "count_type",
    "mean",
    "var",
    "dispersion",
    "best_model",
    "recommendation",
)


@dataclass(frozen=True)
class SequenceBenchmarkReport:
    """Count diagnostics and model comparisons for one benchmark sequence."""

    sequence: str
    gt_path: Path
    count_series: CountSeriesBundle
    comparisons: tuple[CountModelComparison, ...]

    @property
    def num_frames(self) -> int:
        """Number of frames included in the count series."""

        return len(self.count_series.frames)

    @property
    def num_tracks(self) -> int:
        """Number of unique track lifetimes summarized for this sequence."""

        return len(self.count_series.lifetimes)


def discover_motchallenge_gt_files(root: str | Path) -> tuple[Path, ...]:
    """Return sorted ``*/gt/gt.txt`` files under a MOTChallenge benchmark root."""

    root_path = Path(root)
    return tuple(sorted(root_path.glob("*/gt/gt.txt"), key=lambda path: sequence_name_from_gt_path(path)))


def sequence_name_from_gt_path(path: str | Path) -> str:
    """Infer a sequence name such as ``MOT17-02`` from ``MOT17-02/gt/gt.txt``."""

    gt_path = Path(path)
    return gt_path.parent.parent.name


def report_from_motchallenge_gt_file(
    path: str | Path,
    *,
    sequence: str | None = None,
    min_confidence: float | None = None,
    include_empty_frames: bool = True,
) -> SequenceBenchmarkReport:
    """Build count diagnostics and model comparisons for one MOTChallenge file."""

    gt_path = Path(path)
    count_series = count_series_from_motchallenge_file(
        gt_path,
        min_confidence=min_confidence,
        include_empty_frames=include_empty_frames,
    )
    comparisons = (
        compare_count_models(count_series.live_counts, count_name="N_t"),
        compare_count_models(count_series.birth_counts, count_name="B_t"),
        compare_count_models(count_series.lifetimes, count_name="L_k"),
    )
    return SequenceBenchmarkReport(
        sequence=sequence or sequence_name_from_gt_path(gt_path),
        gt_path=gt_path,
        count_series=count_series,
        comparisons=comparisons,
    )


def reports_from_motchallenge_root(
    root: str | Path,
    *,
    min_confidence: float | None = None,
    include_empty_frames: bool = True,
) -> tuple[SequenceBenchmarkReport, ...]:
    """Build reports for every ``*/gt/gt.txt`` file under a benchmark root."""

    gt_files = discover_motchallenge_gt_files(root)
    if not gt_files:
        raise FileNotFoundError(f"No MOTChallenge gt files found under {Path(root)!s}.")
    return tuple(
        report_from_motchallenge_gt_file(
            gt_file,
            min_confidence=min_confidence,
            include_empty_frames=include_empty_frames,
        )
        for gt_file in gt_files
    )


def count_diagnostic_rows(reports: Iterable[SequenceBenchmarkReport]) -> tuple[dict[str, object], ...]:
    """Return one row per sequence with compact ``N_t``, ``B_t``, and ``L_k`` samples."""

    return tuple(
        {
            "sequence": report.sequence,
            "N_t": _json_count_series(report.count_series.live_counts),
            "B_t": _json_count_series(report.count_series.birth_counts),
            "L_k": _json_count_series(report.count_series.lifetimes),
        }
        for report in reports
    )


def model_recommendation_rows(reports: Iterable[SequenceBenchmarkReport]) -> tuple[dict[str, object], ...]:
    """Return model-selection summaries by sequence and count type."""

    rows: list[dict[str, object]] = []
    for report in reports:
        for comparison in report.comparisons:
            diagnostics = comparison.diagnostics
            best_fit = comparison.best_fit
            rows.append(
                {
                    "sequence": report.sequence,
                    "count_type": comparison.count_name,
                    "mean": diagnostics.mean,
                    "var": diagnostics.variance,
                    "dispersion": diagnostics.dispersion_index,
                    "best_model": None if best_fit is None else best_fit.model_name,
                    "recommendation": comparison.recommended_family,
                }
            )
    return tuple(rows)


def count_series_for_type(report: SequenceBenchmarkReport, count_type: str) -> tuple[int, ...]:
    """Return the raw sample series for ``N_t``, ``B_t``, or ``L_k``."""

    if count_type == "N_t":
        return report.count_series.live_counts
    if count_type == "B_t":
        return report.count_series.birth_counts
    if count_type == "L_k":
        return report.count_series.lifetimes
    raise KeyError(f"Unsupported count type {count_type!r}; expected one of {COUNT_TYPES!r}.")


def report_to_dict(report: SequenceBenchmarkReport) -> dict[str, object]:
    """Serialize one sequence report to JSON-friendly primitives."""

    return {
        "sequence": report.sequence,
        "gt_path": str(report.gt_path),
        "num_frames": report.num_frames,
        "num_tracks": report.num_tracks,
        "frames": list(report.count_series.frames),
        "N_t": list(report.count_series.live_counts),
        "B_t": list(report.count_series.birth_counts),
        "L_k": list(report.count_series.lifetimes),
        "count_model_comparisons": [comparison_to_dict(comparison) for comparison in report.comparisons],
    }


def reports_to_dict(reports: Sequence[SequenceBenchmarkReport]) -> dict[str, object]:
    """Serialize a batch report to JSON-friendly primitives."""

    return {
        "sequences": [report_to_dict(report) for report in reports],
        "count_diagnostic_rows": list(count_diagnostic_rows(reports)),
        "model_recommendation_rows": list(model_recommendation_rows(reports)),
    }


def write_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> Path:
    """Write dictionaries to a CSV file and return the output path."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_tuple = tuple(rows)
    if fieldnames is None:
        if not row_tuple:
            raise ValueError("fieldnames must be provided when writing an empty row collection.")
        fieldnames = tuple(row_tuple[0])
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row_tuple)
    return output_path


def write_benchmark_report_outputs(
    reports: Sequence[SequenceBenchmarkReport],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write the standard batch MOTChallenge report CSV files."""

    output_path = Path(output_dir)
    return {
        "count_diagnostics": write_csv(
            output_path / "sequence_count_diagnostics.csv",
            count_diagnostic_rows(reports),
            fieldnames=COUNT_DIAGNOSTIC_FIELDNAMES,
        ),
        "model_recommendations": write_csv(
            output_path / "count_model_recommendations.csv",
            model_recommendation_rows(reports),
            fieldnames=MODEL_RECOMMENDATION_FIELDNAMES,
        ),
    }


def _json_count_series(values: Sequence[int]) -> str:
    return json.dumps(list(values), separators=(",", ":"))
