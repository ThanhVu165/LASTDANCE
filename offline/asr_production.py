"""Hash-bound worker plans and safe Hugging Face ASR namespaces."""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


DEFAULT_ASR_BATCH_IDS = tuple(f"batch-{index:02d}" for index in range(1, 10))


def _identifier(value: str) -> str:
    if value != value.strip() or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) is None:
        raise ValueError("identifier must use only letters, digits, dot, underscore or dash")
    return value


def _sha256(value: str) -> str:
    value = value.lower()
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("value must be a lowercase SHA-256 digest")
    return value


def _relative_path(value: str) -> str:
    if not value or "\\" in value:
        raise ValueError("artifact paths must be relative POSIX paths")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("artifact path cannot traverse parents")
    return value


class AsrWorkerAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    worker_id: str
    enabled: bool = False
    batch_ids: list[str] = Field(min_length=1)

    @field_validator("worker_id")
    @classmethod
    def validate_worker(cls, value: str) -> str:
        return _identifier(value)

    @field_validator("batch_ids")
    @classmethod
    def validate_batches(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("worker batch_ids must be unique")
        return [_identifier(value) for value in values]


class AsrWorkerPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    catalog_sha256: str
    batch_mapping_sha256: str
    expected_batch_ids: list[str] = Field(default_factory=lambda: list(DEFAULT_ASR_BATCH_IDS))
    assignments: list[AsrWorkerAssignment] = Field(min_length=1, max_length=4)

    @field_validator("catalog_sha256", "batch_mapping_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _sha256(value)

    @field_validator("expected_batch_ids")
    @classmethod
    def validate_expected(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("expected_batch_ids must be unique")
        return [_identifier(value) for value in values]

    @model_validator(mode="after")
    def validate_partition(self) -> "AsrWorkerPlan":
        ids = [row.worker_id for row in self.assignments]
        if len(ids) != len(set(ids)):
            raise ValueError("worker_id values must be unique")
        enabled = [row for row in self.assignments if row.enabled]
        if not enabled:
            raise ValueError("at least one ASR worker must be enabled")
        assigned = [batch for row in enabled for batch in row.batch_ids]
        if len(assigned) != len(set(assigned)):
            raise ValueError("enabled worker batches overlap")
        if set(assigned) != set(self.expected_batch_ids):
            raise ValueError("enabled worker batches must be disjoint and exhaustive")
        return self


class AsrShardManifest(BaseModel):
    """Completion record for one UID/video-disjoint ASR shard."""
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    batch_id: str
    worker_id: str
    catalog_sha256: str
    config_sha256: str
    assigned_video_sha256: str
    audio_root: str
    output_jsonl_path: str
    output_jsonl_sha256: str
    expected_videos: int = Field(ge=0)
    processed_videos: int = Field(ge=0)
    error_videos: int = Field(ge=0)
    completion_gate_passed: bool

    @field_validator("batch_id", "worker_id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _identifier(value)

    @field_validator("catalog_sha256", "config_sha256", "assigned_video_sha256", "output_jsonl_sha256")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        return _sha256(value)

    @field_validator("audio_root", "output_jsonl_path")
    @classmethod
    def validate_artifact_path(cls, value: str) -> str:
        return _relative_path(value)

    @model_validator(mode="after")
    def validate_gate(self) -> "AsrShardManifest":
        gate = self.processed_videos == self.expected_videos and self.error_videos == 0
        if self.completion_gate_passed != gate:
            raise ValueError("completion_gate_passed does not match counts")
        return self


def asr_hf_audio_root(batch_id: str) -> str:
    """Audio is grouped by the same nine pinned production batches as ASR archives."""
    return f"asr/audio/{_identifier(batch_id)}"


def asr_hf_archive_root(batch_id: str) -> str:
    return f"asr/archives/{_identifier(batch_id)}"


def asr_hf_snapshot_root(snapshot_id: str) -> str:
    return f"asr/snapshots/{_identifier(snapshot_id)}"


def _under(path: str, root: str) -> str:
    normalized = _relative_path(path)
    candidate, parent = PurePosixPath(normalized), PurePosixPath(root)
    if candidate == parent or parent not in candidate.parents:
        raise ValueError(f"ASR artifact must stay under {root}/")
    return normalized


def validate_asr_audio_path(path: str, *, batch_id: str) -> str:
    return _under(path, asr_hf_audio_root(batch_id))


def validate_asr_archive_path(path: str, *, batch_id: str) -> str:
    return _under(path, asr_hf_archive_root(batch_id))


def validate_asr_snapshot_hf_path(path: str, *, snapshot_id: str) -> str:
    return _under(path, asr_hf_snapshot_root(snapshot_id))


def validate_asr_hf_path(
    path: str,
    *,
    namespace: Literal["audio", "archives", "snapshots"],
    identifier: str,
) -> str:
    """Validate a path against one of the three ASR namespaces."""

    roots = {
        "audio": asr_hf_audio_root,
        "archives": asr_hf_archive_root,
        "snapshots": asr_hf_snapshot_root,
    }
    return _under(path, roots[namespace](identifier))
