"""Fail-closed worker and layer manifests for distributed OCR production."""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class OcrLayer(StrEnum):
    CRAFT = "craft"
    EASYOCR = "easyocr"
    VINTERN = "vintern"
    GEMINI = "gemini"
    TERMINAL = "terminal"


class OcrWorkItemKind(StrEnum):
    KEYFRAME = "keyframe"
    REGION = "region"


def _identifier(value: str) -> str:
    if value != value.strip() or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) is None:
        raise ValueError("identifier must use only letters, digits, dot, underscore or dash")
    return value


def _sha256(value: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError("value must be a lowercase SHA-256 digest")
    return normalized


def _relative_path(value: str) -> str:
    if "\\" in value:
        raise ValueError("artifact paths must use POSIX separators")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError("artifact path must be relative and cannot traverse parents")
    return value


class OcrWorkerAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    worker_id: str
    enabled: bool = False
    batch_ids: list[str] = Field(min_length=1)

    @field_validator("worker_id")
    @classmethod
    def _validate_worker_id(cls, value: str) -> str:
        return _identifier(value)

    @field_validator("batch_ids")
    @classmethod
    def _validate_batch_ids(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("worker batch_ids must be unique")
        for value in values:
            _identifier(value)
        return values


class OcrWorkerPlan(BaseModel):
    """Hash-bound ownership for all production batches.

    A plan does not contain credentials. ``worker_id`` is an operational label;
    the Kaggle account mapping belongs in the private operator notes.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    catalog_sha256: str
    batch_mapping_sha256: str
    expected_batch_ids: list[str] = Field(min_length=1)
    assignments: list[OcrWorkerAssignment] = Field(min_length=1, max_length=4)

    @field_validator("catalog_sha256", "batch_mapping_sha256")
    @classmethod
    def _validate_hash(cls, value: str) -> str:
        return _sha256(value)

    @field_validator("expected_batch_ids")
    @classmethod
    def _validate_expected_batch_ids(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("expected_batch_ids must be unique")
        for value in values:
            _identifier(value)
        return values

    @model_validator(mode="after")
    def _validate_disjoint_and_exhaustive(self) -> "OcrWorkerPlan":
        worker_ids = [assignment.worker_id for assignment in self.assignments]
        if len(worker_ids) != len(set(worker_ids)):
            raise ValueError("worker_id values must be unique")
        enabled = [assignment for assignment in self.assignments if assignment.enabled]
        if not enabled:
            raise ValueError("at least one OCR worker must be enabled")
        assigned = [batch for assignment in enabled for batch in assignment.batch_ids]
        duplicates = sorted({batch for batch in assigned if assigned.count(batch) > 1})
        if duplicates:
            raise ValueError(f"enabled worker batches overlap: {duplicates}")
        missing = sorted(set(self.expected_batch_ids) - set(assigned))
        foreign = sorted(set(assigned) - set(self.expected_batch_ids))
        if missing or foreign:
            raise ValueError(
                f"enabled worker batches are not exhaustive: missing={missing}, foreign={foreign}"
            )
        return self


class OcrLayerShardManifest(BaseModel):
    """Completion gate for one layer of one UID-disjoint batch."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    batch_id: str
    worker_id: str
    layer: OcrLayer
    item_kind: OcrWorkItemKind
    catalog_sha256: str
    config_sha256: str
    assigned_uid_sha256: str
    input_artifact_path: str
    input_artifact_sha256: str
    output_jsonl_path: str
    output_jsonl_sha256: str
    expected_keyframes: int = Field(ge=0)
    processed_keyframes: int = Field(ge=0)
    expected_items: int = Field(ge=0)
    processed_items: int = Field(ge=0)
    error_items: int = Field(ge=0)
    duplicate_items: int = Field(ge=0)
    missing_keyframes: int = Field(ge=0)
    foreign_keyframes: int = Field(ge=0)
    completion_gate_passed: bool

    @field_validator("batch_id", "worker_id")
    @classmethod
    def _validate_identifier(cls, value: str) -> str:
        return _identifier(value)

    @field_validator(
        "catalog_sha256",
        "config_sha256",
        "assigned_uid_sha256",
        "input_artifact_sha256",
        "output_jsonl_sha256",
    )
    @classmethod
    def _validate_sha256(cls, value: str) -> str:
        return _sha256(value)

    @field_validator("input_artifact_path", "output_jsonl_path")
    @classmethod
    def _validate_relative_artifact_path(cls, value: str) -> str:
        return _relative_path(value)

    @model_validator(mode="after")
    def _validate_completion_gate(self) -> "OcrLayerShardManifest":
        gate = (
            self.processed_keyframes == self.expected_keyframes
            and self.processed_items == self.expected_items
            and self.error_items == 0
            and self.duplicate_items == 0
            and self.missing_keyframes == 0
            and self.foreign_keyframes == 0
        )
        if self.completion_gate_passed != gate:
            raise ValueError("completion_gate_passed does not match layer counts")
        return self


def ocr_hf_archive_root(batch_id: str) -> str:
    """Return the only allowed remote root in the shared HF Dataset."""

    return f"ocr/archives/{_identifier(batch_id)}"


def ocr_hf_snapshot_root(snapshot_id: str) -> str:
    """Return the development snapshot namespace in the same HF Dataset."""

    return f"ocr/snapshots/{_identifier(snapshot_id)}"


def validate_ocr_hf_path(path: str, *, batch_id: str) -> str:
    normalized = _relative_path(path)
    root = PurePosixPath(ocr_hf_archive_root(batch_id))
    candidate = PurePosixPath(normalized)
    if candidate == root or root not in candidate.parents:
        raise ValueError(f"OCR artifact must stay under {root.as_posix()}/")
    return normalized


def validate_ocr_snapshot_hf_path(path: str, *, snapshot_id: str) -> str:
    normalized = _relative_path(path)
    root = PurePosixPath(ocr_hf_snapshot_root(snapshot_id))
    candidate = PurePosixPath(normalized)
    if candidate == root or root not in candidate.parents:
        raise ValueError(f"OCR snapshot must stay under {root.as_posix()}/")
    return normalized
