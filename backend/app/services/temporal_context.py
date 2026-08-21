"""Cheap temporal evidence utilities built from organizer keyframes.

The CLIP index treats keyframes as independent images.  These helpers restore a
small amount of video structure without loading another model: they map local
keyframe indices to PTS time, sample a bounded chronological window, and compose
one contact sheet for the existing VLM.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageOps

from app.config import KEYFRAMES_DIR, MAP_KEYFRAMES_DIR


@dataclass(frozen=True)
class KeyframeRecord:
    local_idx: int
    frame_id: int
    pts_time: float
    fps: float
    path: Path


def _keyframe_path(video_id: str, local_idx: int) -> Path:
    folder = KEYFRAMES_DIR / video_id
    for width in (3, 4):
        candidate = folder / f"{int(local_idx):0{width}d}.jpg"
        if candidate.exists():
            return candidate
    return folder / f"{int(local_idx):03d}.jpg"


@lru_cache(maxsize=256)
def keyframe_records(video_id: str) -> tuple[KeyframeRecord, ...]:
    map_path = MAP_KEYFRAMES_DIR / f"{video_id}.csv"
    if not map_path.exists():
        return ()
    records: list[KeyframeRecord] = []
    with map_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            local_idx = int(row["n"])
            path = _keyframe_path(video_id, local_idx)
            if not path.exists():
                continue
            fps = float(row.get("fps") or 0.0)
            frame_id = int(float(row["frame_idx"]))
            raw_pts = row.get("pts_time")
            pts_time = (
                float(raw_pts)
                if raw_pts not in (None, "")
                else frame_id / fps if fps > 0 else float(local_idx)
            )
            records.append(
                KeyframeRecord(
                    local_idx=local_idx,
                    frame_id=frame_id,
                    pts_time=pts_time,
                    fps=fps,
                    path=path,
                )
            )
    return tuple(sorted(records, key=lambda record: record.pts_time))


def record_for_local(video_id: str, local_idx: int) -> KeyframeRecord | None:
    return next(
        (
            record
            for record in keyframe_records(video_id)
            if record.local_idx == int(local_idx)
        ),
        None,
    )


def _even_positions(length: int, count: int) -> list[int]:
    if length <= 0:
        return []
    count = max(1, min(int(count), length))
    if count == 1:
        return [length // 2]
    return list(
        dict.fromkeys(
            int(round(index * (length - 1) / (count - 1)))
            for index in range(count)
        )
    )


def sample_temporal_context(
    video_id: str,
    center_local_idx: int,
    *,
    radius_seconds: float,
    count: int,
) -> list[KeyframeRecord]:
    """Sample a chronological PTS window and always retain the retrieved center."""
    records = keyframe_records(video_id)
    if not records:
        return []
    center_position = min(
        range(len(records)),
        key=lambda index: abs(records[index].local_idx - int(center_local_idx)),
    )
    center = records[center_position]
    bounded = [
        record
        for record in records
        if abs(record.pts_time - center.pts_time) <= max(radius_seconds, 0.0)
    ]
    if not bounded:
        bounded = [center]
    selected = [bounded[index] for index in _even_positions(len(bounded), count)]
    if center not in selected:
        if len(selected) >= max(1, count):
            replace_at = min(
                range(len(selected)),
                key=lambda index: abs(selected[index].pts_time - center.pts_time),
            )
            selected[replace_at] = center
        else:
            selected.append(center)
    return sorted(set(selected), key=lambda record: record.pts_time)


def temporal_triplet(video_id: str, local_idx: int) -> tuple[KeyframeRecord, ...]:
    """Return previous/current/next evidence, repeating an edge frame if needed."""
    records = keyframe_records(video_id)
    if not records:
        return ()
    position = min(
        range(len(records)),
        key=lambda index: abs(records[index].local_idx - int(local_idx)),
    )
    indices = (
        max(0, position - 1),
        position,
        min(len(records) - 1, position + 1),
    )
    return tuple(records[index] for index in indices)


def build_contact_sheet(
    records: Sequence[KeyframeRecord],
    output_path: Path,
    *,
    labels: Sequence[str] | None = None,
    columns: int = 4,
) -> Path:
    if not records:
        raise ValueError("Cannot build a contact sheet without frames.")
    columns = max(1, min(int(columns), len(records)))
    rows = math.ceil(len(records) / columns)
    cell_width, cell_height = 320, 180
    sheet = Image.new(
        "RGB", (columns * cell_width, rows * cell_height), color="black"
    )
    draw = ImageDraw.Draw(sheet)
    for index, record in enumerate(records):
        with Image.open(record.path) as source:
            cell = ImageOps.fit(source.convert("RGB"), (cell_width, cell_height))
        row, column = divmod(index, columns)
        x, y = column * cell_width, row * cell_height
        sheet.paste(cell, (x, y))
        timestamp = f"{record.pts_time / 60:05.2f}m"
        label = labels[index] if labels and index < len(labels) else str(index + 1)
        text = f"{label}  {timestamp}"
        draw.rectangle((x + 3, y + 3, x + 142, y + 31), fill="black")
        draw.text((x + 9, y + 9), text, fill="white")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=90)
    return output_path
