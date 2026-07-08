import csv
import json
from pathlib import Path

import pytest

from pitman_yor_dp import (
    count_diagnostic_rows,
    discover_motchallenge_gt_files,
    model_recommendation_rows,
    report_from_motchallenge_gt_file,
    reports_from_motchallenge_root,
    reports_to_dict,
    write_benchmark_report_outputs,
)


def test_discover_motchallenge_gt_files_is_sorted(tmp_path: Path):
    first = _write_mot_gt(tmp_path / "MOT17-02" / "gt" / "gt.txt", ["1,1,0,0,10,10,1,-1,-1,-1"])
    second = _write_mot_gt(tmp_path / "MOT17-04" / "gt" / "gt.txt", ["1,2,0,0,10,10,1,-1,-1,-1"])

    assert discover_motchallenge_gt_files(tmp_path) == (first, second)


def test_report_from_motchallenge_gt_file_contains_raw_count_series(tmp_path: Path):
    gt_file = _write_mot_gt(
        tmp_path / "MOT17-02" / "gt" / "gt.txt",
        [
            "1,1,0,0,10,10,1,-1,-1,-1",
            "1,2,0,0,10,10,1,-1,-1,-1",
            "2,1,0,0,10,10,1,-1,-1,-1",
            "4,3,0,0,10,10,1,-1,-1,-1",
        ],
    )

    report = report_from_motchallenge_gt_file(gt_file)

    assert report.sequence == "MOT17-02"
    assert report.num_frames == 4
    assert report.num_tracks == 3
    assert report.count_series.live_counts == (2, 1, 0, 1)
    assert report.count_series.birth_counts == (2, 0, 0, 1)
    assert report.count_series.lifetimes == (1, 1, 2)
    assert tuple(comparison.count_name for comparison in report.comparisons) == ("N_t", "B_t", "L_k")


def test_benchmark_rows_cover_each_sequence_and_count_type(tmp_path: Path):
    _write_mot_gt(
        tmp_path / "MOT17-02" / "gt" / "gt.txt",
        [
            "1,1,0,0,10,10,1,-1,-1,-1",
            "1,2,0,0,10,10,1,-1,-1,-1",
            "2,1,0,0,10,10,1,-1,-1,-1",
        ],
    )
    _write_mot_gt(
        tmp_path / "MOT17-04" / "gt" / "gt.txt",
        [
            "1,5,0,0,10,10,1,-1,-1,-1",
            "2,5,0,0,10,10,1,-1,-1,-1",
            "2,6,0,0,10,10,1,-1,-1,-1",
            "3,6,0,0,10,10,1,-1,-1,-1",
        ],
    )

    reports = reports_from_motchallenge_root(tmp_path)
    diagnostic_rows = count_diagnostic_rows(reports)
    recommendation_rows = model_recommendation_rows(reports)

    assert [row["sequence"] for row in diagnostic_rows] == ["MOT17-02", "MOT17-04"]
    assert json.loads(diagnostic_rows[0]["N_t"]) == [2, 1]
    assert json.loads(diagnostic_rows[1]["B_t"]) == [1, 1, 0]
    assert len(recommendation_rows) == 6
    assert {(row["sequence"], row["count_type"]) for row in recommendation_rows} == {
        ("MOT17-02", "N_t"),
        ("MOT17-02", "B_t"),
        ("MOT17-02", "L_k"),
        ("MOT17-04", "N_t"),
        ("MOT17-04", "B_t"),
        ("MOT17-04", "L_k"),
    }
    assert set(recommendation_rows[0]) == {
        "sequence",
        "count_type",
        "mean",
        "var",
        "dispersion",
        "best_model",
        "recommendation",
    }
    assert reports_to_dict(reports)["model_recommendation_rows"] == list(recommendation_rows)


def test_write_benchmark_report_outputs_writes_expected_csvs(tmp_path: Path):
    gt_file = _write_mot_gt(
        tmp_path / "MOT17-02" / "gt" / "gt.txt",
        [
            "1,1,0,0,10,10,1,-1,-1,-1",
            "2,1,0,0,10,10,1,-1,-1,-1",
        ],
    )
    reports = (report_from_motchallenge_gt_file(gt_file),)

    paths = write_benchmark_report_outputs(reports, tmp_path / "reports")

    assert set(paths) == {"count_diagnostics", "model_recommendations"}
    with paths["count_diagnostics"].open(newline="") as handle:
        diagnostic_rows = list(csv.DictReader(handle))
    with paths["model_recommendations"].open(newline="") as handle:
        recommendation_rows = list(csv.DictReader(handle))

    assert diagnostic_rows == [{"sequence": "MOT17-02", "N_t": "[1,1]", "B_t": "[1,0]", "L_k": "[2]"}]
    assert recommendation_rows[0].keys() == {
        "sequence",
        "count_type",
        "mean",
        "var",
        "dispersion",
        "best_model",
        "recommendation",
    }
    assert [row["count_type"] for row in recommendation_rows] == ["N_t", "B_t", "L_k"]


def test_reports_from_motchallenge_root_rejects_empty_root(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        reports_from_motchallenge_root(tmp_path)


def _write_mot_gt(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path
