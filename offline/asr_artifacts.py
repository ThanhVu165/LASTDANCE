"""Validated ASR envelopes, shard manifests, hashes, and coverage summaries."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator

from shared.schemas.asr import AsrSegment

_MAX_INT64 = (1 << 63) - 1


class AsrEngine(StrEnum):
    WHISPER_LARGE_V3 = "whisper_large_v3"
    PHOWHISPER = "phowhisper"


class AsrVideoStatus(StrEnum):
    SUCCESS = "success"
    SILENT = "silent"
    ERROR = "error"


AsrStatus = AsrVideoStatus


def _identifier(value: str) -> str:
    if not value or value != value.strip() or any(c.isspace() for c in value):
        raise ValueError("identifier must be non-empty and contain no whitespace")
    return value


def _sha256(value: str) -> str:
    value = value.lower()
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("value must be a lowercase SHA-256 digest")
    return value


def _relative_path(value: str) -> str:
    if not value or "\\" in value:
        raise ValueError("artifact path must be relative and use POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("artifact path cannot traverse parents")
    return value


class AsrRecordEnvelope(BaseModel):
    """One terminal result for one video (one JSONL line)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal[1] = 1
    batch_id: str
    video_id: str
    status: AsrVideoStatus
    engine: AsrEngine
    audio_path: str
    audio_sha256: str | None = None
    audio_duration_seconds: float = Field(ge=0, alias="duration_seconds")
    segments: list[AsrSegment] = Field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None
    error_status: str | None = None

    @field_validator("batch_id", "video_id")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        return _identifier(value)

    @field_validator("audio_path")
    @classmethod
    def validate_audio_path(cls, value: str) -> str:
        return _relative_path(value)

    @field_validator("audio_sha256")
    @classmethod
    def validate_audio_hash(cls, value: str | None) -> str | None:
        return None if value is None else _sha256(value)

    @field_validator("error_code", "error_message")
    @classmethod
    def validate_error_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("error metadata must be null or non-empty")
        return normalized

    @model_validator(mode="after")
    def validate_terminal_state(self) -> "AsrRecordEnvelope":
        if [s.segment_id for s in self.segments] != sorted(
            (s.segment_id for s in self.segments), key=lambda x: x
        ):
            raise ValueError("segments must be ordered by segment_id")
        if len({s.segment_id for s in self.segments}) != len(self.segments):
            raise ValueError("segment_id values must be unique")
        if any(s.video_id != self.video_id for s in self.segments):
            raise ValueError("segment video_id must match envelope")
        if self.status == AsrVideoStatus.SUCCESS:
            if not self.segments or self.error_code is not None:
                raise ValueError("success requires segments and no error_code")
        elif self.status == AsrVideoStatus.SILENT:
            if self.segments or self.error_code is not None:
                raise ValueError("silent videos cannot contain segments or error_code")
        else:
            if self.segments or not (self.error_code or self.error_status):
                raise ValueError("error videos require error metadata and no segments")
        if self.status != AsrVideoStatus.ERROR and (self.error_status or self.error_message):
            raise ValueError("non-error videos cannot carry error metadata")
        return self


# Friendly aliases used by callers and notebooks.
AsrVideoEnvelope = AsrRecordEnvelope
AsrEnvelope = AsrRecordEnvelope
AsrSegmentEnvelope = AsrRecordEnvelope


class AsrShardManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    batch_id: str
    worker_id: str
    engine: AsrEngine = AsrEngine.WHISPER_LARGE_V3
    catalog_path: str = "frames.csv"
    catalog_sha256: str
    config_sha256: str
    shard_path: str
    shard_sha256: str
    expected_video_sha256: str | None = None
    record_count: int | None = Field(default=None, ge=0)
    success_records: int | None = Field(default=None, ge=0)
    silent_records: int | None = Field(default=None, ge=0)
    error_records: int | None = Field(default=None, ge=0)
    expected_videos: int = Field(ge=0)
    processed_videos: int = Field(ge=0)
    success_videos: int = Field(ge=0)
    silent_videos: int = Field(default=0, ge=0)
    error_videos: int = Field(ge=0)
    duplicate_videos: int = Field(default=0, ge=0)
    missing_videos: int = Field(default=0, ge=0)
    foreign_videos: int = Field(default=0, ge=0)
    completion_gate_passed: bool

    @field_validator("batch_id", "worker_id")
    @classmethod
    def validate_ids(cls, value: str) -> str:
        return _identifier(value)

    @field_validator("catalog_sha256", "config_sha256", "shard_sha256")
    @classmethod
    def validate_hashes(cls, value: str) -> str:
        return _sha256(value)

    @field_validator("shard_path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _relative_path(value)

    @field_validator("catalog_path")
    @classmethod
    def validate_catalog_path(cls, value: str) -> str:
        return _relative_path(value)

    @field_validator("expected_video_sha256")
    @classmethod
    def validate_expected_hash(cls, value: str | None) -> str | None:
        return None if value is None else _sha256(value)

    @model_validator(mode="after")
    def validate_counts(self) -> "AsrShardManifest":
        if self.success_videos + self.silent_videos + self.error_videos != self.processed_videos:
            raise ValueError("video status counts do not sum to processed_videos")
        if self.record_count is not None and self.record_count != self.processed_videos:
            raise ValueError("record_count does not match processed_videos")
        if self.success_records is not None and self.success_records != self.success_videos:
            raise ValueError("success_records does not match success_videos")
        if self.silent_records is not None and self.silent_records != self.silent_videos:
            raise ValueError("silent_records does not match silent_videos")
        if self.error_records is not None and self.error_records != self.error_videos:
            raise ValueError("error_records does not match error_videos")
        gate = (
            self.processed_videos == self.expected_videos
            and self.error_videos == self.duplicate_videos == 0
            and self.missing_videos == self.foreign_videos == 0
        )
        if self.completion_gate_passed != gate:
            raise ValueError("completion_gate_passed does not match counts")
        return self


def video_set_sha256(video_ids: Iterable[str]) -> str:
    values = sorted(set(_identifier(str(value)) for value in video_ids))
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8") + b"\n")
    return digest.hexdigest()


def summarize_asr_coverage(
    records: Iterable[AsrRecordEnvelope],
    *,
    expected_video_ids: Iterable[str],
) -> dict[str, int | bool | str]:
    rows = list(records)
    expected_values = [str(value) for value in expected_video_ids]
    expected = set(expected_values)
    if not expected:
        raise ValueError("expected_video_ids must not be empty")
    if len(expected_values) != len(expected):
        raise ValueError("expected_video_ids contains duplicates")
    actual_ids = [row.video_id for row in rows]
    actual = set(actual_ids)
    counts = Counter(row.status for row in rows)
    duplicates = len(actual_ids) - len(actual)
    missing = len(expected - actual)
    foreign = len(actual - expected)
    errors = counts[AsrVideoStatus.ERROR]
    return {
        "expected_videos": len(expected),
        "processed_videos": len(rows),
        "success_videos": counts[AsrVideoStatus.SUCCESS],
        "silent_videos": counts[AsrVideoStatus.SILENT],
        "error_videos": errors,
        "duplicate_videos": duplicates,
        "missing_videos": missing,
        "foreign_videos": foreign,
        "expected_video_sha256": video_set_sha256(expected),
        "actual_video_sha256": video_set_sha256(actual),
        "completion_gate_passed": len(rows) == len(expected) and not errors and not duplicates and not missing and not foreign,
    }
