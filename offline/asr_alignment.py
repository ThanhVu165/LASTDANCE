"""Temporal alignment of ASR segments to the canonical frames catalog."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any
import math

from shared.schemas.asr import AsrSegment
from shared.schemas.frame import FrameRecord


def _catalog_by_video(frames: Iterable[FrameRecord]) -> dict[str, list[FrameRecord]]:
    result: dict[str, list[FrameRecord]] = {}
    for frame in frames:
        result.setdefault(frame.video_id, []).append(frame)
    for values in result.values():
        values.sort(key=lambda row: (row.pts_time, row.keyframe_uid))
    return result


def nearest_keyframe_uid(
    video_id: str,
    start_time: float,
    end_time: float,
    frames: Iterable[FrameRecord] | Mapping[int, FrameRecord],
) -> int:
    if not math.isfinite(start_time) or not math.isfinite(end_time) or start_time < 0 or end_time < start_time:
        raise ValueError("invalid ASR segment timestamps")
    values = list(frames.values()) if isinstance(frames, Mapping) else list(frames)
    candidates = [row for row in values if row.video_id == video_id]
    if not candidates:
        raise ValueError(f"catalog has no keyframes for video_id={video_id}")
    midpoint = (start_time + end_time) / 2.0
    inside = [row for row in candidates if start_time <= row.pts_time <= end_time]
    chosen = min(inside or candidates, key=lambda row: (abs(row.pts_time - (midpoint if inside else start_time)), row.pts_time, row.keyframe_uid))
    return chosen.keyframe_uid


def build_asr_segments(
    video_id: str,
    raw_segments: Iterable[Mapping[str, Any] | Any],
    frames: Iterable[FrameRecord] | Mapping[int, FrameRecord],
    *,
    segment_prefix: str = "s",
) -> list[AsrSegment]:
    """Build canonical segments from Whisper-like dictionaries/objects."""

    if not video_id.strip():
        raise ValueError("video_id must not be empty")
    frame_values = list(frames.values()) if isinstance(frames, Mapping) else list(frames)
    result: list[AsrSegment] = []
    for index, raw in enumerate(raw_segments):
        if isinstance(raw, Mapping):
            start = raw.get("start", raw.get("start_time"))
            end = raw.get("end", raw.get("end_time"))
            text = raw.get("text", raw.get("transcribed_text"))
            language = raw.get("language", "vi")
        else:
            start = getattr(raw, "start", getattr(raw, "start_time", None))
            end = getattr(raw, "end", getattr(raw, "end_time", None))
            text = getattr(raw, "text", getattr(raw, "transcribed_text", None))
            language = getattr(raw, "language", "vi")
        if start is None or end is None or text is None:
            raise ValueError(f"segment {index} lacks start/end/text")
        segment_id = f"{segment_prefix}{index:06d}"
        result.append(
            AsrSegment(
                video_id=video_id,
                segment_id=segment_id,
                start_time=float(start),
                end_time=float(end),
                transcribed_text=str(text),
                language=str(language),
                keyframe_uid_nearest=nearest_keyframe_uid(
                    video_id, float(start), float(end), frame_values
                ),
            )
        )
    return result
