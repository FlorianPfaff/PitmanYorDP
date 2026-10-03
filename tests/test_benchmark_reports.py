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


def _write_seqinfo(sequence_dir, length=5, fps=10):
    (sequence_dir / "seqinfo.ini").write_text(f"[Sequence]\nseqLength={length}\nframeRate={fps}\n")


def test_audited_mot17_report_preserves_exposure_gaps_and_boundaries(tmp_path):
    from pitman_yor_dp import report_to_dict

    path = _write_mot_gt(tmp_path / "MOT17-02-DPM" / "gt" / "gt.txt", [
        "2,1,0,0,10,10,1,1,0.5", "4,1,0,0,10,10,1,1,0.5",
        "5,2,0,0,10,10,1,1,1", "1,3,0,0,10,10,1,7,1",
    ])
    _write_seqinfo(path.parent.parent)
    report = report_from_motchallenge_gt_file(path, mot17_ground_truth=True)
    assert report.count_series.live_counts == (0, 1, 0, 1, 1)
    assert report.count_series.birth_counts == (0, 1, 0, 0, 1)
    assert report.comparisons[1].diagnostics.num_samples == 4
    rows = report_to_dict(report)["track_spans"]
    assert rows[0]["gap_frames"] == 1
    assert rows[0]["span_seconds"] == pytest.approx(0.3)
    assert rows[0]["touches_clip_start"] is False
    assert rows[1]["touches_clip_end"] is True
    json.dumps(reports_to_dict((report,)), allow_nan=False)


def test_duplicate_detector_variants_are_verified_and_collapsed(tmp_path):
    for detector in ("DPM", "FRCNN", "SDP"):
        path = _write_mot_gt(tmp_path / f"MOT17-02-{detector}" / "gt" / "gt.txt", ["1,1,0,0,10,10,1,1,1"])
        _write_seqinfo(path.parent.parent)
    reports = reports_from_motchallenge_root(tmp_path, mot17_ground_truth=True)
    assert len(reports) == 1
    assert reports[0].sequence == "MOT17-02"
    path.write_text("1,2,0,0,10,10,1,1,1\n")
    with pytest.raises(ValueError, match="different ground truth"):
        reports_from_motchallenge_root(tmp_path, mot17_ground_truth=True)


def test_empty_and_one_frame_sequences_do_not_invent_birth_samples(tmp_path):
    for rows in ([], ["1,1,0,0,10,10,1,1,1"]):
        path = _write_mot_gt(tmp_path / "MOT17-02" / "gt" / "gt.txt", rows)
        _write_seqinfo(path.parent.parent, length=1)
        report = report_from_motchallenge_gt_file(path, mot17_ground_truth=True)
        assert report.count_series.frames == (1,)
        assert "B_t" not in [comparison.count_name for comparison in report.comparisons]
        json.dumps(reports_to_dict((report,)), allow_nan=False)


def test_mot17_audit_requires_metadata(tmp_path):
    path = _write_mot_gt(tmp_path / "MOT17-02" / "gt" / "gt.txt", ["1,1,0,0,10,10,1,1,1"])
    with pytest.raises(FileNotFoundError):
        report_from_motchallenge_gt_file(path, mot17_ground_truth=True)


@pytest.mark.parametrize("rows,error", [
    (["6,1,0,0,10,10,1,1,1"], "outside"),
    (["1,1,0,0,10,10,1,1,1"] * 2, "Duplicate"),
])
def test_mot17_rejects_inconsistent_annotations(tmp_path, rows, error):
    path = _write_mot_gt(tmp_path / "MOT17-02" / "gt" / "gt.txt", rows)
    _write_seqinfo(path.parent.parent)
    with pytest.raises(ValueError, match=error):
        report_from_motchallenge_gt_file(path, mot17_ground_truth=True)


def test_duplicate_scene_metadata_must_match(tmp_path):
    for detector, fps in (("DPM", 10), ("FRCNN", 20)):
        path = _write_mot_gt(tmp_path / f"MOT17-02-{detector}" / "gt" / "gt.txt", ["1,1,0,0,10,10,1,1,1"])
        _write_seqinfo(path.parent.parent, fps=fps)
    with pytest.raises(ValueError, match="inconsistent exposure"):
        reports_from_motchallenge_root(tmp_path, mot17_ground_truth=True)


def test_birth_comparison_excludes_initial_population(tmp_path):
    path = _write_mot_gt(tmp_path / "seq" / "gt" / "gt.txt", [
        "1,1,0,0,10,10,1", "1,2,0,0,10,10,1", "3,1,0,0,10,10,1"
    ])
    report = report_from_motchallenge_gt_file(path)
    assert report.comparisons[1].diagnostics.mean == 0.0


def test_dropping_empty_frames_does_not_shorten_identity_span(tmp_path):
    path = _write_mot_gt(tmp_path / "seq" / "gt" / "gt.txt", [
        "1,1,0,0,10,10,1", "4,1,0,0,10,10,1"
    ])
    report = report_from_motchallenge_gt_file(path, include_empty_frames=False)
    assert report.count_series.lifetimes == (4,)
