"""Adapters from tracking annotations to count diagnostics.

The functions in this module use a deliberately small internal schema so that
benchmark-specific readers can stay thin. A row only needs a frame index and a
track identity to produce live-count, birth-count, and lifetime diagnostics.
"""

from __future__ import annotations

import csv
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .diagnostics import (
    birth_counts_from_label_sets,
    lifetimes_from_label_sets,
    live_counts_from_label_sets,
)


@dataclass(frozen=True)
class TrackRecord:
    """One frame-level tracking annotation or detection record.

    Parameters
    ----------
    frame:
        Integer frame or scan index.
    track_id:
        Hashable identity label. For detector outputs without identity, use a
        unique detection identifier only if singleton lifetime diagnostics are
        intended.
    confidence:
        Optional detector confidence.
    is_ground_truth:
        Optional ground-truth flag. Leave as ``None`` for annotation tables that
        do not mix ground truth and detections.
    metadata:
        Additional fields retained for downstream experiment scripts.
    """

    frame: int
    track_id: Hashable
    confidence: float | None = None
    is_ground_truth: bool | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class CountSeriesBundle:
    """Count series extracted from frame-level label sets."""

    frames: tuple[int, ...]
    label_sets: tuple[frozenset[Hashable], ...]
    live_counts: tuple[int, ...]
    birth_counts: tuple[int, ...]
    lifetimes: tuple[int, ...]


def records_from_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    frame_key: str = "frame",
    track_id_key: str = "track_id",
    confidence_key: str | None = "confidence",
    is_ground_truth_key: str | None = None,
    min_confidence: float | None = None,
    require_ground_truth: bool | None = None,
    metadata_keys: Sequence[str] | None = None,
) -> tuple[TrackRecord, ...]:
    """Convert mapping rows to normalized :class:`TrackRecord` objects.

    Rows with missing track IDs are skipped. Confidence and ground-truth filters
    are applied before constructing records.
    """

    records: list[TrackRecord] = []
    metadata_keys = tuple(metadata_keys or ())
    for row in rows:
        if frame_key not in row or track_id_key not in row:
            raise KeyError(f"rows must contain {frame_key!r} and {track_id_key!r} keys.")

        raw_track_id = row[track_id_key]
        if raw_track_id is None or str(raw_track_id) == "":
            continue

        confidence = _optional_float(row.get(confidence_key)) if confidence_key is not None else None
        if min_confidence is not None:
            if confidence is None or confidence < float(min_confidence):
                continue

        is_ground_truth = _optional_bool(row.get(is_ground_truth_key)) if is_ground_truth_key is not None else None
        if require_ground_truth is not None and is_ground_truth != bool(require_ground_truth):
            continue

        metadata = {key: row.get(key) for key in metadata_keys} if metadata_keys else None
        records.append(
            TrackRecord(
                frame=_coerce_int(row[frame_key], "frame"),
                track_id=_coerce_hashable(raw_track_id),
                confidence=confidence,
                is_ground_truth=is_ground_truth,
                metadata=metadata,
            )
        )
    return tuple(records)


def records_from_csv(
    path: str | Path,
    *,
    frame_key: str = "frame",
    track_id_key: str = "track_id",
    confidence_key: str | None = "confidence",
    is_ground_truth_key: str | None = None,
    min_confidence: float | None = None,
    require_ground_truth: bool | None = None,
    metadata_keys: Sequence[str] | None = None,
) -> tuple[TrackRecord, ...]:
    """Load :class:`TrackRecord` objects from a CSV file."""

    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        return records_from_rows(
            reader,
            frame_key=frame_key,
            track_id_key=track_id_key,
            confidence_key=confidence_key,
            is_ground_truth_key=is_ground_truth_key,
            min_confidence=min_confidence,
            require_ground_truth=require_ground_truth,
            metadata_keys=metadata_keys,
        )


def label_sets_by_frame(
    records: Iterable[TrackRecord],
    *,
    include_empty_frames: bool = True,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> tuple[tuple[int, frozenset[Hashable]], ...]:
    """Group records into per-frame identity sets."""

    records = tuple(records)
    if not records:
        if start_frame is None or end_frame is None or not include_empty_frames:
            return ()
        if end_frame < start_frame:
            raise ValueError("end_frame must be greater than or equal to start_frame.")
        return tuple((frame, frozenset()) for frame in range(start_frame, end_frame + 1))

    observed_frames = [record.frame for record in records]
    first_frame = min(observed_frames) if start_frame is None else int(start_frame)
    last_frame = max(observed_frames) if end_frame is None else int(end_frame)
    if last_frame < first_frame:
        raise ValueError("end_frame must be greater than or equal to start_frame.")

    grouped: dict[int, set[Hashable]] = {}
    for record in records:
        if record.frame < first_frame or record.frame > last_frame:
            continue
        grouped.setdefault(record.frame, set()).add(record.track_id)

    if include_empty_frames:
        frames = range(first_frame, last_frame + 1)
    else:
        frames = sorted(grouped)
    return tuple((frame, frozenset(grouped.get(frame, set()))) for frame in frames)


def count_series_from_records(
    records: Iterable[TrackRecord],
    *,
    include_empty_frames: bool = True,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> CountSeriesBundle:
    """Extract live counts, birth counts, and lifetimes from track records."""

    grouped = label_sets_by_frame(
        records,
        include_empty_frames=include_empty_frames,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    frames = tuple(frame for frame, _ in grouped)
    label_sets = tuple(labels for _, labels in grouped)
    return CountSeriesBundle(
        frames=frames,
        label_sets=label_sets,
        live_counts=live_counts_from_label_sets(label_sets),
        birth_counts=birth_counts_from_label_sets(label_sets),
        lifetimes=lifetimes_from_label_sets(label_sets),
    )


def _coerce_int(value: Any, name: str) -> int:
    try:
        if isinstance(value, str) and value.strip() == "":
            raise ValueError
        return int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be convertible to an integer.") from exc


def _coerce_hashable(value: Any) -> Hashable:
    try:
        hash(value)
        return value
    except TypeError as exc:
        raise TypeError("track_id values must be hashable.") from exc


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return float(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "gt", "ground_truth"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "det", "detection"}:
        return False
    raise ValueError(f"Cannot convert {value!r} to bool.")
