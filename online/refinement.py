"""Decode an operator-requested source frame exactly, without average-FPS math."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from .artifacts import ArtifactRegistry
from .media import source_video


def exact_frame_command(ffmpeg: str, video: Path, frame_id: int, output: Path) -> list[str]:
    if frame_id < 0:
        raise ValueError("frame_id must be non-negative")
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-vf",
        f"select=eq(n\\,{frame_id})",
        "-vsync",
        "0",
        "-frames:v",
        "1",
        "-an",
        "-sn",
        "-dn",
        str(output),
    ]


class ExactFrameDecoder:
    def __init__(self, registry: ArtifactRegistry, *, ffmpeg: str | None = None) -> None:
        self.registry = registry
        self.ffmpeg = ffmpeg or os.environ.get("AIC_FFMPEG", "ffmpeg")

    def decode(self, video_id: str, frame_id: int) -> Path:
        video = source_video(self.registry, video_id)
        if video is None:
            raise RuntimeError(f"source video is unavailable: {video_id}")
        root = (self.registry.layout.data.root / "tmp" / "online-refinement").resolve()
        directory = (root / video_id).resolve()
        if root not in directory.parents:
            raise RuntimeError("unsafe exact-frame cache path")
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"frame-{frame_id}.jpg"
        if destination.is_file() and destination.stat().st_size > 0:
            return destination
        fd, temporary_name = tempfile.mkstemp(prefix=f"frame-{frame_id}-", suffix=".jpg", dir=directory)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            command = exact_frame_command(self.ffmpeg, video, frame_id, temporary)
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            if completed.returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
                detail = (completed.stderr or completed.stdout or "no decoded frame").strip()
                raise RuntimeError(f"FFmpeg exact-frame decode failed: {detail[-500:]}")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination
