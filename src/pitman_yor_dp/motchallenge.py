"""MOTChallenge-style annotation adapters.

MOTChallenge text files are comma-separated without a header. The common 2D MOT
layout is

``frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z``.

This module only uses the frame, identity, confidence, and bounding-box columns;
remaining fields are retained as metadata when present. The goal is to produce
the minimal :class:`pitman_yor_dp.adapters.TrackRecord` schema for count
diagnostics, not to implement a tracker or a full benchmark evaluator.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .adapters import TrackRecord, count_series_from_records


def records_from_motchallenge_file(
    path: str | Path,
    *,
    min_confidence: float | None = None,
    ignored_track_ids: Sequence[int | str] = (-1,),
    delimiter: str = ",",
) -> tuple[TrackRecord, ...]:
    """Read a MOTChallenge-style file into :class:`TrackRecord` objects.

    Parameters
    ----------
    path:
        Text file path. Typical examples are ``gt/gt.txt`` or ``det/det.txt``.
    min_confidence:
        Optional lower confidence bound. Rows with missing confidence are kept
        when this argument is ``None`` and removed otherwise.
    ignored_track_ids:
        Track identities to skip. The default skips ``-1``, commonly used for
        detections without persistent identity.
    delimiter:
        Field delimiter. MOTChallenge uses commas.
    """

    with Path(path).open(newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        return records_from_motchallenge_rows(
            reader,
            min_confidence=min_confidence,
            ignored_track_ids=ignored_track_ids,
        )


def records_from_motchallenge_rows(
    rows: Iterable[Sequence[Any]],
    *,
    min_confidence: float | None = None,
    ignored_track_ids: Sequence[int | str] = (-1,),
) -> tuple[TrackRecord, ...]:
    """Convert MOTChallenge-style rows to :class:`TrackRecord` objects."""

    ignored = {str(track_id) for track_id in ignored_track_ids}
    records: list[TrackRecord] = []
    for line_number, row in enumerate(rows, start=1):
        if not row:
            continue
        if len(row) < 2:
            raise ValueError(f"MOTChallenge row {line_number} has fewer than two columns.")
        raw_frame = str(row[0]).strip()
        raw_track_id = str(row[1]).strip()
        if raw_track_id in ignored or raw_track_id == "":
            continue

        confidence = _optional_float(row[6]) if len(row) > 6 else None
        if min_confidence is not None:
            if confidence is None or confidence < float(min_confidence):
                continue

        records.append(
            TrackRecord(
                frame=_mot_int(raw_frame, "frame", line_number),
                track_id=_mot_track_id(raw_track_id),
                confidence=confidence,
                metadata=_mot_metadata(row),
            )
        )
    return tuple(records)


def count_series_from_motchallenge_file(
    path: str | Path,
    *,
    min_confidence: float | None = None,
    ignored_track_ids: Sequence[int | str] = (-1,),
    include_empty_frames: bool = True,
) -> object:
    """Read a MOTChallenge-style file and return a count-series bundle."""

    records = records_from_motchallenge_file(
        path,
        min_confidence=min_confidence,
        ignored_track_ids=ignored_track_ids,
    )
    return count_series_from_records(records, include_empty_frames=include_empty_frames)


def _mot_int(value: str, name: str, line_number: int) -> int:
    try:
        return int(float(value))
    except ValueError as exc:
        raise ValueError(f"MOTChallenge row {line_number} has invalid {name}: {value!r}.") from exc


def _mot_track_id(value: str) -> int | str:
    try:
        as_float = float(value)
        as_int = int(as_float)
        if as_float == float(as_int):
            return as_int
    except ValueError:
        pass
    return value


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return float(text)


def _mot_metadata(row: Sequence[Any]) -> dict[str, float | int | None]:
    metadata: dict[str, float | int | None] = {}
    keys = ("bb_left", "bb_top", "bb_width", "bb_height", "world_x", "world_y", "world_z")
    positions = (2, 3, 4, 5, 7, 8, 9)
    for key, position in zip(keys, positions):
        if len(row) > position:
            metadata[key] = _optional_float(row[position])
    return metadata
