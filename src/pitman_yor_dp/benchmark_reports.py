"""Batch reports for benchmark tracking count diagnostics."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from configparser import ConfigParser
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from .adapters import CountSeriesBundle, count_series_from_records
from .model_selection import CountModelComparison, compare_count_models, comparison_to_dict
from .motchallenge import records_from_motchallenge_file

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
    frame_rate: float | None = None
    sequence_length: int | None = None
    mot17_ground_truth: bool = False
    min_confidence: float | None = None

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
    mot17_ground_truth: bool = False,
) -> SequenceBenchmarkReport:
    """Build count diagnostics and model comparisons for one MOTChallenge file."""

    gt_path = Path(path)
    records = records_from_motchallenge_file(
        gt_path,
        min_confidence=min_confidence,
        mot17_ground_truth=mot17_ground_truth,
    )
    frame_rate = None
    sequence_length = None
    if mot17_ground_truth:
        if not include_empty_frames:
            raise ValueError("MOT17 diagnostics require the complete frame exposure.")
        frame_rate, sequence_length = _sequence_exposure(gt_path)
        if any(record.frame < 1 or record.frame > sequence_length for record in records):
            raise ValueError("Annotation frame lies outside seqinfo.ini exposure.")
        keys = [(record.frame, record.track_id) for record in records]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate identity annotation within a frame.")
    count_series = count_series_from_records(
        records, include_empty_frames=include_empty_frames,
        start_frame=1 if mot17_ground_truth else None, end_frame=sequence_length,
    )
    comparisons = tuple(compare_count_models(values, count_name=name) for name, values in (
        ("N_t", count_series.live_counts),
        ("B_t", count_series.birth_counts[1:]),
        ("L_k", count_series.lifetimes),
    ) if values)
    return SequenceBenchmarkReport(
        sequence=sequence or sequence_name_from_gt_path(gt_path),
        gt_path=gt_path,
        count_series=count_series,
        comparisons=comparisons,
        frame_rate=frame_rate,
        sequence_length=sequence_length,
        mot17_ground_truth=mot17_ground_truth,
        min_confidence=min_confidence,
    )


def reports_from_motchallenge_root(
    root: str | Path,
    *,
    min_confidence: float | None = None,
    include_empty_frames: bool = True,
    mot17_ground_truth: bool = False,
) -> tuple[SequenceBenchmarkReport, ...]:
    """Build reports for every ``*/gt/gt.txt`` file under a benchmark root."""

    gt_files = discover_motchallenge_gt_files(root)
    if not gt_files:
        raise FileNotFoundError(f"No MOTChallenge gt files found under {Path(root)!s}.")
    if mot17_ground_truth:
        unique: dict[str, Path] = {}
        for path in gt_files:
            name = _canonical_sequence_name(sequence_name_from_gt_path(path))
            if name in unique:
                if _sha256(path) != _sha256(unique[name]):
                    raise ValueError(f"Detector variants of {name} contain different ground truth.")
                if _sequence_exposure(path) != _sequence_exposure(unique[name]):
                    raise ValueError(f"Detector variants of {name} have inconsistent exposure metadata.")
            else:
                unique[name] = path
        gt_files = tuple(unique.values())
    return tuple(
        report_from_motchallenge_gt_file(
            gt_file,
            sequence=_canonical_sequence_name(sequence_name_from_gt_path(gt_file)) if mot17_ground_truth else None,
            min_confidence=min_confidence,
            include_empty_frames=include_empty_frames,
            mot17_ground_truth=mot17_ground_truth,
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
                    "dispersion": diagnostics.dispersion_index if isfinite(diagnostics.dispersion_index) else None,
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
        "gt_path": str(Path(report.gt_path.parent.parent.name) / "gt" / report.gt_path.name),
        "num_frames": report.num_frames,
        "num_tracks": report.num_tracks,
        "frames": list(report.count_series.frames),
        "N_t": list(report.count_series.live_counts),
        "B_t": list(report.count_series.birth_counts),
        "L_k": list(report.count_series.lifetimes),
        "count_model_comparisons": [comparison_to_dict(comparison) for comparison in report.comparisons],
        "birth_comparison_omits_first_frame": True,
        "frame_rate": report.frame_rate,
        "sequence_length": report.sequence_length,
        "ground_truth_sha256": _sha256(report.gt_path),
        "sequence_metadata_sha256": _sha256(report.gt_path.parent.parent / "seqinfo.ini") if report.sequence_length is not None else None,
        "min_confidence": report.min_confidence,
        "annotation_filter": "mark=1,class=1; all visibility levels" if report.mot17_ground_truth else "generic; see supplied confidence filter",
        "track_spans": track_span_rows(report),
    }


def reports_to_dict(reports: Sequence[SequenceBenchmarkReport]) -> dict[str, object]:
    """Serialize a batch report to JSON-friendly primitives."""

    return {
        "schema_version": 2,
        "interpretation": "N_t=retained annotated identities; B_t=first observed identities; L_k=observed frame span. No tail or partition-prior test.",
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


def _canonical_sequence_name(name: str) -> str:
    for suffix in ("-DPM", "-FRCNN", "-SDP"):
        if name.startswith("MOT17-") and name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sequence_exposure(gt_path: Path) -> tuple[float, int]:
    config = ConfigParser()
    info_path = gt_path.parent.parent / "seqinfo.ini"
    if not config.read(info_path):
        raise FileNotFoundError(f"MOT17 diagnostics require {info_path}.")
    frame_rate = config.getfloat("Sequence", "frameRate")
    length = config.getint("Sequence", "seqLength")
    if not isfinite(frame_rate) or frame_rate <= 0 or length <= 0:
        raise ValueError("Sequence frame rate and length must be positive.")
    return frame_rate, length


def track_span_rows(report: SequenceBenchmarkReport) -> list[dict[str, object]]:
    """Preserve identity spans, gaps, and known clip boundaries without inferring deaths."""

    observations: dict[object, list[int]] = {}
    for frame, labels in zip(report.count_series.frames, report.count_series.label_sets):
        for label in labels:
            observations.setdefault(label, []).append(frame)
    rows = []
    for label in sorted(observations, key=str):
        times = observations[label]
        span = times[-1] - times[0] + 1
        rows.append({
            "track_id": label, "first_frame": times[0], "last_frame": times[-1],
            "span_frames": span, "observed_frames": len(times), "gap_frames": span - len(times),
            "touches_clip_start": times[0] == 1 if report.sequence_length is not None else None,
            "touches_clip_end": times[-1] == report.sequence_length if report.sequence_length is not None else None,
            "span_seconds": span / report.frame_rate if report.frame_rate else None,
        })
    return rows
