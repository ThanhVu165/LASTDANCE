"""Decode an operator-requested source frame exactly, without average-FPS math."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from .artifacts import ArtifactRegistry
from .media import source_video
from .frame_references import SourceFrameVerifier


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
        self.verifier = SourceFrameVerifier(registry)

    def decode(self, video_id: str, frame_id: int) -> Path:
        reference = self.verifier.verify(video_id, frame_id)
        video = source_video(self.registry, video_id)
        if video is None:
            raise RuntimeError(f"source video is unavailable: {video_id}")
        root = (self.registry.layout.data.root / "tmp" / "online-refinement").resolve()
        directory = (root / video_id / reference.source_sha256).resolve()
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

    def strip(self, video_id: str, frame_id: int):
        """Decode up to 21 consecutive frames in one FFmpeg pass, preserving PTS."""
        center = self.verifier.verify(video_id, frame_id)
        count = len(self.verifier.timestamps(video_id))
        numbers = list(range(max(0, frame_id - 10), min(count, frame_id + 11)))
        directory = self.registry.layout.data.root / "tmp" / "online-refinement" / video_id / center.source_sha256
        directory.mkdir(parents=True, exist_ok=True)
        paths = [directory / f"frame-{number}.jpg" for number in numbers]
        if not all(path.is_file() and path.stat().st_size for path in paths):
            with tempfile.TemporaryDirectory(prefix="strip-", dir=directory) as temporary:
                pattern = Path(temporary) / "%06d.jpg"
                command = [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i",
                           str(source_video(self.registry, video_id)), "-vf",
                           f"select=between(n\\,{numbers[0]}\\,{numbers[-1]})",
                           "-vsync", "0", "-frames:v", str(len(numbers)), "-start_number", "0",
                           "-an", "-sn", "-dn", str(pattern)]
                result = subprocess.run(command, capture_output=True, text=True, check=False)
                files = [Path(temporary) / f"{index:06d}.jpg" for index in range(len(numbers))]
                if result.returncode or not all(path.is_file() and path.stat().st_size for path in files):
                    raise RuntimeError(f"FFmpeg strip decode failed: {result.stderr[-500:]}")
                self.verifier.validate(center)
                for source, target in zip(files, paths):
                    os.replace(source, target)
        return [(self.verifier.verify(video_id, number), path) for number, path in zip(numbers, paths)]
