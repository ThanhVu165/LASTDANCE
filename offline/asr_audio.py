"""Atomic local extraction of canonical Whisper audio from the video inventory."""

from __future__ import annotations

import hashlib
import json
import subprocess
import wave
from collections.abc import Callable, Iterable
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from offline.artifacts import sha256_file
from offline.preprocessing.models import VideoInventoryRecord


SAMPLE_RATE_HZ = 16_000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2
CODEC = "pcm_s16le"
_Runner = Callable[..., subprocess.CompletedProcess[str]]


class AudioStatus(StrEnum):
    READY = "ready"
    NO_AUDIO = "no_audio"


def _relative_artifact_path(value: str) -> str:
    if "\\" in value:
        raise ValueError("artifact paths must use POSIX separators")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError("artifact path must be relative and cannot traverse parents")
    return value


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class AudioArtifact(BaseModel):
    """Hash-bound record for one inventory video and its optional PCM WAV."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    video_id: str
    status: AudioStatus
    source_video: str
    source_size_bytes: int = Field(ge=0)
    inventory_duration_seconds: float = Field(ge=0)
    extraction_signature: str
    ffmpeg_version: str
    wav_path: str | None = None
    wav_sha256: str | None = None
    wav_size_bytes: int | None = Field(default=None, ge=0)
    wav_duration_seconds: float | None = Field(default=None, ge=0)
    sample_rate_hz: int | None = Field(default=None, ge=1)
    channels: int | None = Field(default=None, ge=1)
    sample_width_bytes: int | None = Field(default=None, ge=1)
    codec: str | None = None
    megabytes_per_minute: float | None = Field(default=None, ge=0)
    wav_to_source_size_ratio: float | None = Field(default=None, ge=0)

    @field_validator("video_id")
    @classmethod
    def _validate_video_id(cls, value: str) -> str:
        if not value or value.strip() != value or any(char.isspace() for char in value):
            raise ValueError("video_id must be canonical and contain no whitespace")
        return value

    @field_validator("source_video")
    @classmethod
    def _validate_source_video(cls, value: str) -> str:
        return _relative_artifact_path(value)

    @field_validator("wav_path")
    @classmethod
    def _validate_wav_path(cls, value: str | None) -> str | None:
        return None if value is None else _relative_artifact_path(value)

    @field_validator("extraction_signature", "wav_sha256")
    @classmethod
    def _validate_digest(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("digest must be a lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def _validate_status_payload(self) -> "AudioArtifact":
        audio_fields = (
            self.wav_path,
            self.wav_sha256,
            self.wav_size_bytes,
            self.wav_duration_seconds,
            self.sample_rate_hz,
            self.channels,
            self.sample_width_bytes,
            self.codec,
            self.megabytes_per_minute,
            self.wav_to_source_size_ratio,
        )
        if self.status == AudioStatus.NO_AUDIO:
            if any(value is not None for value in audio_fields):
                raise ValueError("no_audio artifact cannot carry WAV metadata")
            return self
        if any(value is None for value in audio_fields):
            raise ValueError("ready audio artifact requires complete WAV metadata")
        if (
            self.sample_rate_hz != SAMPLE_RATE_HZ
            or self.channels != CHANNELS
            or self.sample_width_bytes != SAMPLE_WIDTH_BYTES
            or self.codec != CODEC
        ):
            raise ValueError("ready audio artifact is not PCM s16le 16 kHz mono")
        return self


def load_inventory(path: Path) -> list[VideoInventoryRecord]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read inventory: {source}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RuntimeError("unsupported inventory schema")
    rows = payload.get("videos")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("inventory contains no videos")
    try:
        records = [VideoInventoryRecord(**row) for row in rows]
    except (TypeError, ValueError) as error:
        raise RuntimeError("inventory contains an invalid video record") from error
    video_ids = [record.video_id for record in records]
    if len(set(video_ids)) != len(video_ids):
        raise RuntimeError("inventory contains duplicate video_id values")
    return sorted(records, key=lambda record: record.video_id)


def select_inventory_records(
    records: Iterable[VideoInventoryRecord], video_ids: set[str] | None
) -> list[VideoInventoryRecord]:
    rows = list(records)
    if video_ids is None:
        return rows
    available = {row.video_id for row in rows}
    missing = sorted(video_ids - available)
    if missing:
        raise RuntimeError(f"requested video IDs are absent from inventory: {missing[:5]}")
    return [row for row in rows if row.video_id in video_ids]


def ffmpeg_version(
    ffmpeg_binary: str, *, runner: _Runner = subprocess.run
) -> str:
    result = runner(
        [ffmpeg_binary, "-version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cannot execute FFmpeg: {(result.stderr or '').strip()}")
    lines = (result.stdout or result.stderr or "").splitlines()
    if not lines:
        raise RuntimeError("FFmpeg returned no version information")
    return lines[0].strip()


def build_extraction_signature(
    record: VideoInventoryRecord,
    *,
    source_size_bytes: int,
    ffmpeg_version_text: str,
) -> str:
    payload = {
        "schema_version": 1,
        "video_id": record.video_id,
        "source_video": record.relative_path,
        "source_size_bytes": source_size_bytes,
        "inventory_duration_seconds": record.duration,
        "inventory_has_audio": record.has_audio,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "channels": CHANNELS,
        "sample_width_bytes": SAMPLE_WIDTH_BYTES,
        "codec": CODEC,
        "ffmpeg_version": ffmpeg_version_text,
    }
    return _sha256_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def inspect_pcm_wav(path: Path) -> dict[str, int | float]:
    source = Path(path)
    try:
        with wave.open(str(source), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frame_count = handle.getnframes()
            compression = handle.getcomptype()
    except (OSError, EOFError, wave.Error) as error:
        raise RuntimeError(f"cannot read WAV artifact: {source}") from error
    if (
        channels != CHANNELS
        or sample_width != SAMPLE_WIDTH_BYTES
        or sample_rate != SAMPLE_RATE_HZ
        or compression != "NONE"
        or frame_count <= 0
    ):
        raise RuntimeError("WAV artifact is not non-empty PCM s16le 16 kHz mono")
    return {
        "channels": channels,
        "sample_width_bytes": sample_width,
        "sample_rate_hz": sample_rate,
        "frame_count": frame_count,
        "duration_seconds": frame_count / sample_rate,
    }


def _write_json_atomic(path: Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def validate_audio_artifact(manifest_path: Path, *, audio_root: Path) -> AudioArtifact:
    source = Path(manifest_path)
    try:
        artifact = AudioArtifact.model_validate_json(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"invalid audio manifest: {source}") from error
    if artifact.status == AudioStatus.NO_AUDIO:
        return artifact
    assert artifact.wav_path is not None and artifact.wav_sha256 is not None
    wav = (Path(audio_root) / artifact.wav_path).resolve(strict=False)
    root = Path(audio_root).resolve(strict=False)
    try:
        wav.relative_to(root)
    except ValueError as error:
        raise RuntimeError("audio manifest WAV path escapes its artifact root") from error
    if not wav.is_file() or sha256_file(wav) != artifact.wav_sha256:
        raise RuntimeError("audio WAV is missing or failed its manifest SHA-256")
    info = inspect_pcm_wav(wav)
    if (
        wav.stat().st_size != artifact.wav_size_bytes
        or abs(float(info["duration_seconds"]) - float(artifact.wav_duration_seconds))
        > 1e-6
    ):
        raise RuntimeError("audio WAV metadata changed after publication")
    return artifact


def extract_audio_artifact(
    record: VideoInventoryRecord,
    *,
    data_root: Path,
    audio_root: Path,
    ffmpeg_binary: str,
    ffmpeg_version_text: str,
    runner: _Runner = subprocess.run,
) -> AudioArtifact:
    """Extract one video atomically or validate and reuse its existing artifact."""

    root = Path(data_root).resolve(strict=False)
    source = (root / record.relative_path).resolve(strict=False)
    try:
        source.relative_to(root)
    except ValueError as error:
        raise RuntimeError("inventory video path escapes AIC_DATA") from error
    if not source.is_file():
        raise RuntimeError(f"inventory video is missing: {record.relative_path}")

    artifact_root = Path(audio_root).resolve(strict=False)
    manifest_path = artifact_root / "manifests" / f"{record.video_id}.json"
    wav_relative = PurePosixPath("wav") / f"{record.video_id}.wav"
    wav_path = artifact_root / Path(*wav_relative.parts)
    source_size = source.stat().st_size
    signature = build_extraction_signature(
        record,
        source_size_bytes=source_size,
        ffmpeg_version_text=ffmpeg_version_text,
    )

    if manifest_path.exists():
        existing = validate_audio_artifact(manifest_path, audio_root=artifact_root)
        if existing.video_id != record.video_id or existing.extraction_signature != signature:
            raise RuntimeError("audio checkpoint signature mismatch")
        return existing

    if not record.has_audio:
        artifact = AudioArtifact(
            video_id=record.video_id,
            status=AudioStatus.NO_AUDIO,
            source_video=record.relative_path,
            source_size_bytes=source_size,
            inventory_duration_seconds=record.duration,
            extraction_signature=signature,
            ffmpeg_version=ffmpeg_version_text,
        )
        _write_json_atomic(manifest_path, artifact.model_dump(mode="json"))
        return artifact

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = wav_path.with_name(f".{wav_path.name}.tmp.wav")
    if temporary.exists():
        temporary.unlink()
    command = [
        ffmpeg_binary,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        str(CHANNELS),
        "-ar",
        str(SAMPLE_RATE_HZ),
        "-c:a",
        CODEC,
        "-f",
        "wav",
        str(temporary),
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
        if temporary.exists():
            temporary.unlink()
        raise RuntimeError(
            f"FFmpeg audio extraction failed for {record.video_id}: "
            f"{(result.stderr or 'unknown error').strip()}"
        )
    info = inspect_pcm_wav(temporary)
    wav_duration = float(info["duration_seconds"])
    tolerance = max(2.0, record.duration * 0.01)
    if abs(wav_duration - record.duration) > tolerance:
        temporary.unlink()
        raise RuntimeError(
            f"WAV duration mismatch for {record.video_id}: "
            f"inventory={record.duration}, wav={wav_duration}"
        )
    if wav_path.exists():
        temporary.unlink()
        raise RuntimeError(f"refusing to overwrite WAV without a manifest: {wav_path}")
    temporary.replace(wav_path)

    wav_size = wav_path.stat().st_size
    duration_minutes = wav_duration / 60.0
    artifact = AudioArtifact(
        video_id=record.video_id,
        status=AudioStatus.READY,
        source_video=record.relative_path,
        source_size_bytes=source_size,
        inventory_duration_seconds=record.duration,
        extraction_signature=signature,
        ffmpeg_version=ffmpeg_version_text,
        wav_path=wav_relative.as_posix(),
        wav_sha256=sha256_file(wav_path),
        wav_size_bytes=wav_size,
        wav_duration_seconds=wav_duration,
        sample_rate_hz=int(info["sample_rate_hz"]),
        channels=int(info["channels"]),
        sample_width_bytes=int(info["sample_width_bytes"]),
        codec=CODEC,
        megabytes_per_minute=(wav_size / 1_000_000) / duration_minutes,
        wav_to_source_size_ratio=wav_size / source_size if source_size else 0.0,
    )
    _write_json_atomic(manifest_path, artifact.model_dump(mode="json"))
    return artifact
