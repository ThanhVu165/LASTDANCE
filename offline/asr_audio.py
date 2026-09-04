"""Safe, atomic extraction of ASR audio with FFmpeg/ffprobe."""

from __future__ import annotations

import json
import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


@dataclass(frozen=True, slots=True)
class AudioProbe:
    duration_seconds: float
    has_audio: bool


@dataclass(frozen=True, slots=True)
class AudioExtraction:
    source_video: Path
    output_audio: Path
    duration_seconds: float
    audio_sha256: str | None = None


Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _run_subprocess(
    command: Sequence[str], **kwargs: object
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, **kwargs)


def safe_video_path(video: str | Path, videos_root: str | Path) -> Path:
    """Resolve a video ID/path while refusing traversal outside ``videos_root``."""

    root = Path(videos_root).expanduser().resolve()
    candidate = Path(video)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("video path must remain under videos_root") from exc
    if resolved.suffix.lower() != ".mp4":
        raise ValueError("ASR input must be an MP4 video")
    return resolved


def probe_audio_duration(
    video: str | Path,
    *,
    videos_root: str | Path | None = None,
    ffprobe_binary: str = "ffprobe",
    runner: Runner | None = None,
) -> AudioProbe:
    source = safe_video_path(video, videos_root or Path(video).parent)
    run = runner or _run_subprocess
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
    try:
        result = run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)  # type: ignore[call-arg]
    except TypeError:
        result = run(command)
    if getattr(result, "returncode", 0) != 0:
        raise RuntimeError(f"ffprobe failed for {source.name}")
    try:
        payload = json.loads(result.stdout or "{}")
        duration = float((payload.get("format") or {}).get("duration", 0.0))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"ffprobe returned invalid metadata for {source.name}") from exc
    streams = payload.get("streams") or []
    has_audio = any(row.get("codec_type") == "audio" for row in streams)
    if duration <= 0:
        raise RuntimeError("ffprobe returned a missing or non-positive duration")
    return AudioProbe(duration_seconds=duration, has_audio=has_audio)


def extract_audio_flac(
    video: str | Path,
    output_path: str | Path,
    *,
    videos_root: str | Path | None = None,
    ffmpeg_binary: str = "ffmpeg",
    ffprobe_binary: str = "ffprobe",
    runner: Runner | None = None,
    overwrite: bool = False,
) -> AudioExtraction:
    source = safe_video_path(video, videos_root or Path(video).parent)
    destination = Path(output_path).expanduser()
    if destination.suffix.lower() != ".flac":
        raise ValueError("ASR audio output must be FLAC")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"audio already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    probe = probe_audio_duration(
        source,
        videos_root=source.parent,
        ffprobe_binary=ffprobe_binary,
        runner=runner,
    )
    if not probe.has_audio:
        raise RuntimeError(f"video has no audio stream: {source.name}")
    # Keep the final extension so FFmpeg can infer the FLAC muxer.
    temporary = destination.with_name(f".{destination.stem}.staging.flac")
    if temporary.exists():
        temporary.unlink()
    run = runner or _run_subprocess
    command = [
        ffmpeg_binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "flac",
        str(temporary),
    ]
    try:
        try:
            result = run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)  # type: ignore[call-arg]
        except TypeError:
            result = run(command)
        if getattr(result, "returncode", 0) != 0:
            raise RuntimeError(f"ffmpeg failed for {source.name}")
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError("FFmpeg did not produce a non-empty FLAC")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    digest_builder = hashlib.sha256()
    with destination.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest_builder.update(block)
    digest = digest_builder.hexdigest()
    return AudioExtraction(source, destination, probe.duration_seconds, digest)


def default_videos_root() -> Path:
    """Return the configured raw-video root without serializing it into artifacts.

    Production callers should set ``AIC_VIDEOS_ROOT`` (the local deployment uses
    ``F:\LASTDANCE-DATA\videos``).  The repository-local fallback keeps the
    library portable and avoids embedding a machine-specific path in artifacts.
    """

    return Path(os.environ.get("AIC_VIDEOS_ROOT", "data/videos"))


# Explicit aliases keep the public API readable in notebooks and tests.
extract_audio_16k_mono_flac = extract_audio_flac
probe_duration = probe_audio_duration
resolve_video_path = safe_video_path
