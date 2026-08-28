"""OCR JSONL envelope and fail-closed shard completion contract."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from shared.schemas.ocr import OcrResult


_MAX_INT64 = (1 << 63) - 1


class OcrEngine(StrEnum):
    CRAFT = "craft"
    EASYOCR = "easyocr"
    VINTERN = "vintern"
    GEMINI = "gemini"


class OcrStatus(StrEnum):
    SUCCESS = "success"
    NO_TEXT = "no_text"
    ERROR = "error"


class OcrExecutionMode(StrEnum):
    LAYERED_ESCALATION = "craft_easyocr_vintern_gemini"

    # Superseded production/audit modes retained so old evidence remains readable.
    CRAFT_GATED_GEMINI = "craft_gated_gemini"
    GEMINI_PRIMARY = "gemini_primary"
    EASYOCR_OFFLINE = "easyocr_offline"


class OcrAttemptStage(StrEnum):
    DETECTION = "detection"
    RECOGNITION = "recognition"


class OcrAttemptOutcome(StrEnum):
    SUCCESS = "success"
    NO_TEXT = "no_text"
    RETRYABLE_ERROR = "retryable_error"
    INVALID_RESPONSE = "invalid_response"
    TERMINAL_ERROR = "terminal_error"


class OcrAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine: OcrEngine
    stage: OcrAttemptStage = OcrAttemptStage.RECOGNITION
    attempt_number: int = Field(ge=1)
    outcome: OcrAttemptOutcome
    latency_ms: int = Field(ge=0)
    error_code: str | None = None
    error_message: str | None = None

    @field_validator("error_code", "error_message")
    @classmethod
    def _normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("error metadata must be null or non-empty")
        return normalized

    @model_validator(mode="after")
    def _error_metadata_matches_outcome(self) -> "OcrAttempt":
        failed = self.outcome in {
            OcrAttemptOutcome.RETRYABLE_ERROR,
            OcrAttemptOutcome.INVALID_RESPONSE,
            OcrAttemptOutcome.TERMINAL_ERROR,
        }
        if failed and self.error_code is None:
            raise ValueError("failed OCR attempts require error_code")
        if not failed and (self.error_code is not None or self.error_message is not None):
            raise ValueError("successful/no_text attempts cannot carry error metadata")
        if self.engine == OcrEngine.CRAFT and self.stage != OcrAttemptStage.DETECTION:
            raise ValueError("CRAFT attempts must use detection stage")
        if self.engine != OcrEngine.CRAFT and self.stage != OcrAttemptStage.RECOGNITION:
            raise ValueError("recognizer attempts must use recognition stage")
        return self


def _validate_identifier(value: str) -> str:
    normalized = value.strip()
    if not normalized or normalized != value or any(character.isspace() for character in value):
        raise ValueError("identifier must be non-empty and contain no whitespace")
    return normalized


def _validate_sha256(value: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("value must be a lowercase SHA-256 digest")
    return normalized


def _validate_relative_path(value: str) -> str:
    if "\\" in value:
        raise ValueError("artifact paths must use POSIX separators")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError("artifact path must be relative and cannot traverse parents")
    return value


class OcrRecordEnvelope(BaseModel):
    """One terminal OCR decision written as one JSONL line.

    Bboxes are flattened normalized quadrilaterals in clockwise order, starting at
    the visual top-left: ``[x1,y1,x2,y2,x3,y3,x4,y4]`` in ``[0,1]``.
    ``no_text`` and ``error`` intentionally carry no synthetic ``OcrResult``.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    batch_id: str
    video_id: str
    keyframe_uid: int = Field(gt=0, le=_MAX_INT64)
    frame_id: int = Field(ge=0)
    source_image: str
    execution_mode: OcrExecutionMode
    status: OcrStatus
    engine: OcrEngine
    fallback_used: bool
    result: OcrResult | None
    attempts: list[OcrAttempt] = Field(min_length=1)

    @field_validator("batch_id", "video_id")
    @classmethod
    def _require_identifier(cls, value: str) -> str:
        return _validate_identifier(value)

    @field_validator("source_image")
    @classmethod
    def _require_relative_source(cls, value: str) -> str:
        return _validate_relative_path(value)

    @model_validator(mode="after")
    def _enforce_terminal_state(self) -> "OcrRecordEnvelope":
        expected_attempt_numbers = list(range(1, len(self.attempts) + 1))
        if [attempt.attempt_number for attempt in self.attempts] != expected_attempt_numbers:
            raise ValueError("OCR attempt_number values must be contiguous and ordered")
        final_attempt = self.attempts[-1]
        if final_attempt.engine != self.engine:
            raise ValueError("record engine must match the final attempt engine")

        easyocr_indexes = [
            index for index, attempt in enumerate(self.attempts)
            if attempt.engine == OcrEngine.EASYOCR
        ]
        if self.execution_mode == OcrExecutionMode.LAYERED_ESCALATION:
            if self.attempts[0].engine != OcrEngine.CRAFT:
                raise ValueError("layered escalation must start with CRAFT detection")
            recognition_attempts = self.attempts[1:]
            if not recognition_attempts:
                if self.attempts[0].outcome not in {
                    OcrAttemptOutcome.NO_TEXT,
                    OcrAttemptOutcome.RETRYABLE_ERROR,
                    OcrAttemptOutcome.INVALID_RESPONSE,
                    OcrAttemptOutcome.TERMINAL_ERROR,
                }:
                    raise ValueError("CRAFT text detection requires EasyOCR recognition")
                expected_fallback = False
            else:
                if self.attempts[0].outcome != OcrAttemptOutcome.SUCCESS:
                    raise ValueError("recognition requires successful CRAFT text detection")
                if recognition_attempts[0].engine != OcrEngine.EASYOCR:
                    raise ValueError("layered escalation must run EasyOCR before Vintern/Gemini")
                if any(
                    attempt.stage != OcrAttemptStage.RECOGNITION
                    or attempt.engine == OcrEngine.CRAFT
                    for attempt in recognition_attempts
                ):
                    raise ValueError("CRAFT can only appear as the first detection attempt")
                engine_rank = {
                    OcrEngine.EASYOCR: 0,
                    OcrEngine.VINTERN: 1,
                    OcrEngine.GEMINI: 2,
                }
                ranks = [engine_rank[attempt.engine] for attempt in recognition_attempts]
                if any(current < previous for previous, current in zip(ranks, ranks[1:])):
                    raise ValueError("recognition engines cannot resume after escalation")
                if any(current - previous > 1 for previous, current in zip(ranks, ranks[1:])):
                    raise ValueError("layered escalation cannot skip Vintern before Gemini")
                expected_fallback = recognition_attempts[-1].engine != OcrEngine.EASYOCR
        elif self.execution_mode == OcrExecutionMode.CRAFT_GATED_GEMINI:
            if self.attempts[0].engine != OcrEngine.CRAFT:
                raise ValueError("craft_gated_gemini must start with CRAFT detection")
            recognition_attempts = self.attempts[1:]
            if not recognition_attempts:
                if self.attempts[0].outcome not in {
                    OcrAttemptOutcome.NO_TEXT,
                    OcrAttemptOutcome.RETRYABLE_ERROR,
                    OcrAttemptOutcome.INVALID_RESPONSE,
                    OcrAttemptOutcome.TERMINAL_ERROR,
                }:
                    raise ValueError("CRAFT text detection requires a recognition attempt")
            else:
                if self.attempts[0].outcome != OcrAttemptOutcome.SUCCESS:
                    raise ValueError("recognition requires successful CRAFT text detection")
                if any(
                    attempt.stage != OcrAttemptStage.RECOGNITION
                    or attempt.engine == OcrEngine.CRAFT
                    for attempt in recognition_attempts
                ):
                    raise ValueError("CRAFT can only appear as the first detection attempt")
                first_easyocr = next(
                    (
                        index
                        for index, attempt in enumerate(recognition_attempts)
                        if attempt.engine == OcrEngine.EASYOCR
                    ),
                    None,
                )
                if first_easyocr is not None and any(
                    attempt.engine != OcrEngine.EASYOCR
                    for attempt in recognition_attempts[first_easyocr:]
                ):
                    raise ValueError("Gemini cannot resume after EasyOCR recognition fallback")
            expected_fallback = bool(recognition_attempts) and (
                recognition_attempts[-1].engine == OcrEngine.EASYOCR
            )
        elif self.execution_mode == OcrExecutionMode.EASYOCR_OFFLINE:
            if any(attempt.engine != OcrEngine.EASYOCR for attempt in self.attempts):
                raise ValueError("EasyOCR-only mode cannot contain Gemini attempts")
            expected_fallback = False
        else:
            if any(attempt.engine == OcrEngine.CRAFT for attempt in self.attempts):
                raise ValueError("legacy gemini_primary cannot contain CRAFT attempts")
            if easyocr_indexes:
                first_easyocr = easyocr_indexes[0]
                if any(
                    attempt.engine != OcrEngine.GEMINI
                    for attempt in self.attempts[:first_easyocr]
                ) or any(
                    attempt.engine != OcrEngine.EASYOCR
                    for attempt in self.attempts[first_easyocr:]
                ):
                    raise ValueError("Gemini attempts cannot resume after EasyOCR fallback")
                if not any(
                    attempt.outcome == OcrAttemptOutcome.INVALID_RESPONSE
                    for attempt in self.attempts[:first_easyocr]
                ):
                    raise ValueError(
                        "EasyOCR per-keyframe fallback requires a prior invalid Gemini response"
                    )
            expected_fallback = bool(easyocr_indexes)
        if self.fallback_used != expected_fallback:
            raise ValueError("fallback_used does not match the attempt history")

        if self.status == OcrStatus.SUCCESS:
            if final_attempt.outcome != OcrAttemptOutcome.SUCCESS or self.result is None:
                raise ValueError("success requires a successful final attempt and OcrResult")
            if self.result.frame_id != self.frame_id:
                raise ValueError("OcrResult.frame_id does not match the envelope")
            if not self.result.detected_text or any(
                not text.strip() for text in self.result.detected_text
            ):
                raise ValueError("success requires at least one non-empty detected_text")
            for bbox in self.result.bbox:
                if len(bbox) != 8 or any(coordinate < 0 or coordinate > 1 for coordinate in bbox):
                    raise ValueError(
                        "bbox must be an 8-value normalized clockwise quadrilateral"
                    )
        elif self.status == OcrStatus.NO_TEXT:
            if final_attempt.outcome != OcrAttemptOutcome.NO_TEXT or self.result is not None:
                raise ValueError("no_text requires a no_text final attempt and null result")
        else:
            if final_attempt.outcome not in {
                OcrAttemptOutcome.RETRYABLE_ERROR,
                OcrAttemptOutcome.INVALID_RESPONSE,
                OcrAttemptOutcome.TERMINAL_ERROR,
            } or self.result is not None:
                raise ValueError("error requires a failed final attempt and null result")
        return self


class OcrShardManifest(BaseModel):
    """Hash-bound summary used as the only batch completion gate."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    batch_id: str
    execution_mode: OcrExecutionMode
    catalog_path: str
    catalog_sha256: str
    config_sha256: str
    expected_uid_sha256: str
    shard_path: str
    shard_sha256: str
    expected_records: int = Field(ge=1)
    record_count: int = Field(ge=0)
    success_records: int = Field(ge=0)
    no_text_records: int = Field(ge=0)
    error_records: int = Field(ge=0)
    duplicate_keyframes: int = Field(ge=0)
    missing_keyframes: int = Field(ge=0)
    foreign_keyframes: int = Field(ge=0)
    completion_gate_passed: bool

    @field_validator("batch_id")
    @classmethod
    def _require_batch_id(cls, value: str) -> str:
        return _validate_identifier(value)

    @field_validator("catalog_path", "shard_path")
    @classmethod
    def _require_relative_path(cls, value: str) -> str:
        return _validate_relative_path(value)

    @field_validator(
        "catalog_sha256", "config_sha256", "expected_uid_sha256", "shard_sha256"
    )
    @classmethod
    def _require_sha256(cls, value: str) -> str:
        return _validate_sha256(value)

    @model_validator(mode="after")
    def _validate_counts_and_gate(self) -> "OcrShardManifest":
        if (
            self.success_records + self.no_text_records + self.error_records
            != self.record_count
        ):
            raise ValueError("OCR status counts do not sum to record_count")
        gate = (
            self.record_count == self.expected_records
            and self.error_records == 0
            and self.duplicate_keyframes == 0
            and self.missing_keyframes == 0
            and self.foreign_keyframes == 0
        )
        if self.completion_gate_passed != gate:
            raise ValueError("completion_gate_passed does not match coverage/error counts")
        return self


def uid_set_sha256(keyframe_uids: Iterable[int]) -> str:
    values = sorted(set(keyframe_uids))
    digest = hashlib.sha256()
    for value in values:
        if value <= 0 or value > _MAX_INT64:
            raise ValueError("keyframe_uid must be a positive signed-int64")
        digest.update(f"{value}\n".encode("ascii"))
    return digest.hexdigest()


def aggregate_easyocr_confidence(detections: Iterable[tuple[str, float]]) -> float:
    """Return the spec-defined text-length-weighted EasyOCR confidence."""

    weighted_sum = 0.0
    total_weight = 0
    for text, confidence in detections:
        normalized = text.strip()
        if not normalized:
            continue
        if confidence < 0 or confidence > 1:
            raise ValueError("EasyOCR confidence must be in [0,1]")
        weight = sum(not character.isspace() for character in normalized)
        if weight == 0:
            continue
        weighted_sum += confidence * weight
        total_weight += weight
    if total_weight == 0:
        raise ValueError("cannot aggregate confidence without detected text")
    return min(1.0, max(0.0, weighted_sum / total_weight))


def summarize_ocr_coverage(
    records: Iterable[OcrRecordEnvelope],
    *,
    expected_keyframe_uids: Iterable[int],
) -> dict[str, int | bool | str]:
    rows = list(records)
    expected = set(expected_keyframe_uids)
    if not expected:
        raise ValueError("expected_keyframe_uids must not be empty")
    actual_values = [row.keyframe_uid for row in rows]
    actual = set(actual_values)
    status_counts = Counter(row.status for row in rows)
    duplicates = len(actual_values) - len(actual)
    missing = len(expected - actual)
    foreign = len(actual - expected)
    errors = status_counts[OcrStatus.ERROR]
    gate = (
        len(rows) == len(expected)
        and errors == 0
        and duplicates == 0
        and missing == 0
        and foreign == 0
    )
    return {
        "expected_records": len(expected),
        "record_count": len(rows),
        "success_records": status_counts[OcrStatus.SUCCESS],
        "no_text_records": status_counts[OcrStatus.NO_TEXT],
        "error_records": errors,
        "duplicate_keyframes": duplicates,
        "missing_keyframes": missing,
        "foreign_keyframes": foreign,
        "expected_uid_sha256": uid_set_sha256(expected),
        "actual_uid_sha256": uid_set_sha256(actual),
        "completion_gate_passed": gate,
    }
