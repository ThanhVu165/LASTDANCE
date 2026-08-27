"""Deterministic temporal alignment from ASR segments to canonical keyframes."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from offline.asr_artifacts import AsrTranscriptRecord, TranscriptStatus
from offline.catalog import FRAME_COLUMNS, validate_frames_catalog
from shared.schemas.asr import AsrSegment
from shared.schemas.frame import FrameRecord


def load_frames_by_video(catalog_path: Path) -> dict[str, list[FrameRecord]]:
    source = Path(catalog_path)
    if not validate_frames_catalog(source):
        raise RuntimeError("frames.csv or its state is incomplete/invalid")
    grouped: dict[str, list[FrameRecord]] = defaultdict(list)
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FRAME_COLUMNS:
            raise RuntimeError("frames.csv columns do not match the canonical schema")
        for row in reader:
            frame = FrameRecord(**{**row, "window_id": row["window_id"] or None})
            grouped[frame.video_id].append(frame)
    for video_id, frames in grouped.items():
        frames.sort(key=lambda frame: (frame.pts_time, frame.keyframe_uid))
        if len({frame.keyframe_uid for frame in frames}) != len(frames):
            raise RuntimeError(f"duplicate keyframe_uid in frames.csv for {video_id}")
    return dict(grouped)


def nearest_frame_for_segment(
    frames: list[FrameRecord], *, start_time: float, end_time: float
) -> FrameRecord:
    """Apply BASELINE_SPEC §2A.2 with deterministic tie-breaking.

    When one or more keyframes are inside the segment, distance is measured from
    the segment midpoint. If none are inside, distance is measured from start_time.
    """

    if not frames:
        raise ValueError("frames must not be empty")
    if start_time < 0 or end_time < start_time:
        raise ValueError("invalid ASR segment time range")
    in_range = [frame for frame in frames if start_time <= frame.pts_time <= end_time]
    if in_range:
        target = (start_time + end_time) / 2.0
        candidates = in_range
    else:
        target = start_time
        candidates = frames
    return min(
        candidates,
        key=lambda frame: (
            abs(frame.pts_time - target),
            frame.pts_time,
            frame.keyframe_uid,
        ),
    )


def align_transcript_records(
    records: Iterable[AsrTranscriptRecord],
    *,
    frames_by_video: dict[str, list[FrameRecord]],
) -> list[AsrSegment]:
    rows = sorted(records, key=lambda record: record.video_id)
    video_ids = [record.video_id for record in rows]
    if len(set(video_ids)) != len(video_ids):
        raise RuntimeError("duplicate transcript record for one video_id")
    aligned: list[AsrSegment] = []
    seen_segments: set[tuple[str, str]] = set()
    for record in rows:
        if record.status == TranscriptStatus.NO_SPEECH:
            continue
        frames = frames_by_video.get(record.video_id)
        if not frames:
            raise RuntimeError(f"frames.csv has no keyframe for {record.video_id}")
        valid_uids = {frame.keyframe_uid for frame in frames}
        for segment in record.segments:
            key = (record.video_id, segment.segment_id)
            if key in seen_segments:
                raise RuntimeError(f"duplicate ASR segment: {key}")
            seen_segments.add(key)
            nearest = nearest_frame_for_segment(
                frames,
                start_time=segment.start_time,
                end_time=segment.end_time,
            )
            aligned_segment = AsrSegment(
                video_id=record.video_id,
                segment_id=segment.segment_id,
                start_time=segment.start_time,
                end_time=segment.end_time,
                transcribed_text=segment.transcribed_text,
                language=segment.language,
                keyframe_uid_nearest=nearest.keyframe_uid,
            )
            if aligned_segment.keyframe_uid_nearest not in valid_uids:
                raise RuntimeError("aligned keyframe_uid is not in the same video")
            aligned.append(aligned_segment)
    return aligned


def load_transcript_records(records_dir: Path) -> list[AsrTranscriptRecord]:
    root = Path(records_dir)
    paths = sorted(root.glob("*.json"))
    if not paths:
        raise RuntimeError(f"transcript record directory is empty: {root}")
    try:
        records = [
            AsrTranscriptRecord.model_validate_json(path.read_text(encoding="utf-8"))
            for path in paths
        ]
    except (OSError, ValueError) as error:
        raise RuntimeError("cannot load validated ASR transcript records") from error
    return records


def write_aligned_jsonl_atomic(path: Path, segments: Iterable[AsrSegment]) -> Path:
    rows = sorted(segments, key=lambda row: (row.video_id, row.start_time, row.segment_id))
    keys = [(row.video_id, row.segment_id) for row in rows]
    if len(set(keys)) != len(keys):
        raise RuntimeError("aligned ASR output contains duplicate segments")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
                + "\n"
            )
    temporary.replace(destination)
    return destination


def load_aligned_jsonl(path: Path) -> list[AsrSegment]:
    source = Path(path)
    rows: list[AsrSegment] = []
    try:
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise RuntimeError(f"blank aligned ASR line at {line_number}")
                rows.append(AsrSegment.model_validate_json(line))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"cannot load aligned ASR JSONL: {source}") from error
    keys = [(row.video_id, row.segment_id) for row in rows]
    if len(set(keys)) != len(keys):
        raise RuntimeError("aligned ASR JSONL contains duplicate segments")
    return rows
