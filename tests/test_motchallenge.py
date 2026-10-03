from pathlib import Path

import pytest

from pitman_yor_dp import (
    count_series_from_motchallenge_file,
    records_from_motchallenge_rows,
)


def test_records_from_motchallenge_rows_extract_track_records_and_metadata():
    rows = [
        ["1", "10", "0", "1", "20", "30", "1.0", "-1", "-1", "-1"],
        ["1", "-1", "5", "6", "20", "30", "0.9", "-1", "-1", "-1"],
        ["2", "10", "1", "1", "20", "30", "0.8", "-1", "-1", "-1"],
        ["3", "11", "2", "2", "20", "30", "0.2", "-1", "-1", "-1"],
    ]

    records = records_from_motchallenge_rows(rows, min_confidence=0.5)

    assert len(records) == 2
    assert records[0].frame == 1
    assert records[0].track_id == 10
    assert records[0].metadata["bb_width"] == 20.0
    assert records[1].frame == 2
    assert records[1].track_id == 10


def test_count_series_from_motchallenge_file(tmp_path: Path):
    mot_file = tmp_path / "gt.txt"
    mot_file.write_text(
        "1,1,0,0,10,10,1,-1,-1,-1\n"
        "1,2,0,0,10,10,1,-1,-1,-1\n"
        "2,1,0,0,10,10,1,-1,-1,-1\n"
        "4,3,0,0,10,10,1,-1,-1,-1\n"
    )

    bundle = count_series_from_motchallenge_file(mot_file)

    assert bundle.frames == (1, 2, 3, 4)
    assert bundle.live_counts == (2, 1, 0, 1)
    assert bundle.birth_counts == (2, 0, 0, 1)
    assert bundle.lifetimes == (1, 1, 2)


def test_mot17_ground_truth_uses_mark_and_class_and_preserves_visibility():
    rows = [
        [1, 1, 0, 0, 10, 10, 1, 1, 0.0],
        [1, 2, 0, 0, 10, 10, 1, 7, 1.0],
        [1, 3, 0, 0, 10, 10, 0, 1, 1.0],
    ]
    records = records_from_motchallenge_rows(rows, mot17_ground_truth=True)
    assert [record.track_id for record in records] == [1]
    assert records[0].is_ground_truth
    assert records[0].metadata["visibility"] == 0.0
    assert records[0].metadata["class_id"] == 1
    assert "world_x" not in records[0].metadata


@pytest.mark.parametrize("frame", ["1.5", "nan", "inf"])
def test_invalid_frame_is_not_truncated(frame):
    with pytest.raises(ValueError):
        records_from_motchallenge_rows([[frame, "1"]])
