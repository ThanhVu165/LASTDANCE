"""Video inventory based on real ``ffprobe`` metadata."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .models import VideoInventoryRecord


_VIDEO_EXTENSIONS = frozenset({".mp4", ".mkv", ".mov", ".avi", ".webm"})
_Runner = Callable[..., subprocess.CompletedProcess[str]]


def discover_videos(videos_root: Path) -> list[Path]:
    root = Path(videos_root)
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in _VIDEO_EXTENSIONS
    )


def _parse_rate(value: object) -> float | None:
    if value in (None, "", "N/A", "0/0"):
        return None
    text = str(value)
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        denominator_value = float(denominator)
        if denominator_value == 0:
            return None
        return float(numerator) / denominator_value
    parsed = float(text)
    return parsed if parsed > 0 else None


def _parse_optional_float(value: object) -> float | None:
    if value in (None, "", "N/A"):
        return None
    parsed = float(str(value))
    return parsed if parsed >= 0 else None


def _parse_optional_int(value: object) -> int | None:
    if value in (None, "", "N/A"):
        return None
    parsed = int(str(value))
    return parsed if parsed >= 0 else None


def probe_video(
    video_path: Path,
    *,
    data_root: Path,
    ffprobe_binary: str = "ffprobe",
    runner: _Runner = subprocess.run,
) -> VideoInventoryRecord:
    """Probe one video and return only metadata read from the source file."""

    source = Path(video_path).resolve(strict=False)
    root = Path(data_root).resolve(strict=False)
    try:
        relative_path = source.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"video must be inside AIC_DATA: {source}") from exc

    command = [
        ffprobe_binary,
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(source),
    ]
    result = runner(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or "ffprobe failed").strip()
        raise RuntimeError(f"ffprobe failed for {source.name}: {detail}")

    try:
        payload: dict[str, Any] = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ffprobe returned invalid JSON for {source.name}") from exc

    streams = payload.get("streams") or []
    video_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "video"),
        None,
    )
    if video_stream is None:
        raise RuntimeError(f"no video stream found in {source.name}")

    fps = _parse_rate(video_stream.get("avg_frame_rate"))
    if fps is None:
        fps = _parse_rate(video_stream.get("r_frame_rate"))
    if fps is None:
        raise RuntimeError(f"missing valid FPS in {source.name}")

    duration = _parse_optional_float(video_stream.get("duration"))
    if duration is None:
        duration = _parse_optional_float((payload.get("format") or {}).get("duration"))
    if duration is None:
        raise RuntimeError(f"missing valid duration in {source.name}")

    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError(f"missing valid resolution in {source.name}")

    return VideoInventoryRecord(
        video_id=source.stem,
        relative_path=relative_path,
        width=width,
        height=height,
        fps=fps,
        duration=duration,
        frame_count=_parse_optional_int(video_stream.get("nb_frames")),
        has_audio=any(stream.get("codec_type") == "audio" for stream in streams),
    )


def build_inventory(
    video_paths: Iterable[Path],
    *,
    data_root: Path,
    ffprobe_binary: str = "ffprobe",
    runner: _Runner = subprocess.run,
) -> list[VideoInventoryRecord]:
    records = [
        probe_video(
            path,
            data_root=data_root,
            ffprobe_binary=ffprobe_binary,
            runner=runner,
        )
        for path in video_paths
    ]
    video_ids = [record.video_id for record in records]
    if len(video_ids) != len(set(video_ids)):
        raise ValueError("duplicate video_id detected in inventory")
    return sorted(records, key=lambda record: record.video_id)


def write_inventory_atomic(
    output_path: Path, records: Iterable[VideoInventoryRecord]
) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "videos": [record.as_dict() for record in records],
    }
    temporary = destination.with_name(f"{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
