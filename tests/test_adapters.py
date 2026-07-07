from pitman_yor_dp import (
    TrackRecord,
    count_series_from_records,
    label_sets_by_frame,
    records_from_rows,
)


def test_records_from_rows_filters_confidence_and_ground_truth():
    rows = [
        {"frame": "1", "track_id": "a", "confidence": "0.9", "kind": "gt"},
        {"frame": "1", "track_id": "b", "confidence": "0.2", "kind": "gt"},
        {"frame": "2", "track_id": "c", "confidence": "0.8", "kind": "det"},
        {"frame": "3", "track_id": "", "confidence": "1.0", "kind": "gt"},
    ]

    records = records_from_rows(
        rows,
        is_ground_truth_key="kind",
        min_confidence=0.5,
        require_ground_truth=True,
    )

    assert records == (TrackRecord(frame=1, track_id="a", confidence=0.9, is_ground_truth=True, metadata=None),)


def test_label_sets_by_frame_can_include_empty_frames():
    records = [
        TrackRecord(frame=2, track_id="a"),
        TrackRecord(frame=4, track_id="a"),
        TrackRecord(frame=4, track_id="b"),
    ]

    grouped = label_sets_by_frame(records, include_empty_frames=True, start_frame=1, end_frame=4)

    assert grouped == (
        (1, frozenset()),
        (2, frozenset({"a"})),
        (3, frozenset()),
        (4, frozenset({"a", "b"})),
    )


def test_count_series_from_records_extracts_live_birth_and_lifetime_counts():
    records = [
        TrackRecord(frame=1, track_id="a"),
        TrackRecord(frame=1, track_id="b"),
        TrackRecord(frame=2, track_id="a"),
        TrackRecord(frame=3, track_id="b"),
        TrackRecord(frame=3, track_id="c"),
    ]

    bundle = count_series_from_records(records, include_empty_frames=True)

    assert bundle.frames == (1, 2, 3)
    assert bundle.live_counts == (2, 1, 2)
    assert bundle.birth_counts == (2, 0, 1)
    assert bundle.lifetimes == (1, 3, 3)
