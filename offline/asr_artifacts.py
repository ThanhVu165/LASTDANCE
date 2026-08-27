"""Validated intermediate ASR records used before alignment to ``frames.csv``."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TranscriptStatus(StrEnum):
    SUCCESS = "success"
    NO_SPEECH = "no_speech"


def _canonical_identifier(value: str) -> str:
    if not value or value.strip() != value or any(char.isspace() for char in value):
        raise ValueError("identifier must be canonical and contain no whitespace")
    return value


def _relative_path(value: str) -> str:
    if "\\" in value:
        raise ValueError("artifact paths must use POSIX separators")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError("artifact path must be relative and cannot traverse parents")
    return value


def _digest(value: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError("value must be a SHA-256 digest")
    return normalized


class RawAsrSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segment_id: str
    start_time: float = Field(ge=0)
    end_time: float = Field(ge=0)
    transcribed_text: str
    language: Literal["vi", "en"]

    @field_validator("segment_id")
    @classmethod
    def _validate_segment_id(cls, value: str) -> str:
        return _canonical_identifier(value)

    @field_validator("transcribed_text")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("transcribed_text must not be empty")
        return normalized

    @model_validator(mode="after")
    def _validate_time_range(self) -> "RawAsrSegment":
        if self.end_time < self.start_time:
            raise ValueError("end_time must be greater than or equal to start_time")
        return self


class AsrTranscriptRecord(BaseModel):
    """One atomically published terminal transcription result for a video."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    batch_id: str
    video_id: str
    model_key: str
    model_id: str
    model_revision: str
    source_wav: str
    source_wav_sha256: str
    source_duration_seconds: float = Field(gt=0)
    status: TranscriptStatus
    elapsed_seconds: float = Field(ge=0)
    segments: list[RawAsrSegment]

    @field_validator("batch_id", "video_id", "model_key")
    @classmethod
    def _validate_ids(cls, value: str) -> str:
        return _canonical_identifier(value)

    @field_validator("model_id")
    @classmethod
    def _validate_model_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or any(char.isspace() for char in normalized):
            raise ValueError("model_id must not contain whitespace")
        return normalized

    @field_validator("model_revision")
    @classmethod
    def _validate_revision(cls, value: str) -> str:
        if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("model_revision must be an immutable commit SHA")
        return value

    @field_validator("source_wav")
    @classmethod
    def _validate_source_wav(cls, value: str) -> str:
        return _relative_path(value)

    @field_validator("source_wav_sha256")
    @classmethod
    def _validate_source_hash(cls, value: str) -> str:
        return _digest(value)

    @model_validator(mode="after")
    def _validate_terminal_status(self) -> "AsrTranscriptRecord":
        if self.status == TranscriptStatus.SUCCESS and not self.segments:
            raise ValueError("success transcript requires at least one segment")
        if self.status == TranscriptStatus.NO_SPEECH and self.segments:
            raise ValueError("no_speech transcript cannot carry segments")
        expected = [f"{self.video_id}:seg-{index:06d}" for index in range(len(self.segments))]
        if [segment.segment_id for segment in self.segments] != expected:
            raise ValueError("segment_id sequence is not canonical")
        return self


def transcript_selection_sha256(records: list[AsrTranscriptRecord]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda row: row.video_id):
        digest.update(
            json.dumps(
                record.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()
