"""Legacy Gate 1 CRAFT-to-Gemini selection evidence and reusable budget guards.

The layered production design now creates Gemini candidates only from residual
regions after EasyOCR and Vintern.  ``select_gemini_escalations`` is retained to
read/reproduce older Gate 1 artifacts and must not schedule new API work.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_MAX_INT64 = (1 << 63) - 1


class CraftDetectionStatus(StrEnum):
    TEXT_DETECTED = "text_detected"
    NO_TEXT = "no_text"
    ERROR = "error"


class CraftFrameFeatures(BaseModel):
    """Hashable detector/reuse evidence for one keyframe before recognition."""

    model_config = ConfigDict(extra="forbid")

    keyframe_uid: int = Field(gt=0, le=_MAX_INT64)
    video_id: str = Field(min_length=1)
    shot_id: str = Field(min_length=1)
    frame_id: int = Field(ge=0)
    source_image: str = Field(min_length=1)
    status: CraftDetectionStatus
    detected_region_count: int = Field(ge=0)
    detector_confidence: float | None = Field(default=None, ge=0, le=1)
    min_region_area_ratio: float | None = Field(default=None, ge=0, le=1)
    perspective_score: float = Field(default=0, ge=0, le=1)
    visual_priority: float = Field(default=0, ge=0, le=1)
    reuse_source_uid: int | None = Field(default=None, gt=0, le=_MAX_INT64)
    embedding_cosine_to_source: float | None = Field(default=None, ge=-1, le=1)
    craft_layout_similarity_to_source: float | None = Field(default=None, ge=0, le=1)
    crop_ssim_to_source: float | None = Field(default=None, ge=0, le=1)
    crop_phash_distance_to_source: int | None = Field(default=None, ge=0)

    @field_validator("video_id", "shot_id")
    @classmethod
    def _validate_identifier(cls, value: str) -> str:
        if value != value.strip() or any(character.isspace() for character in value):
            raise ValueError("identifier must not contain whitespace")
        return value

    @field_validator("source_image")
    @classmethod
    def _validate_source_image(cls, value: str) -> str:
        if "\\" in value or value.startswith("/") or ".." in value.split("/"):
            raise ValueError("source_image must be a safe POSIX-relative path")
        return value

    @model_validator(mode="after")
    def _validate_detector_state(self) -> "CraftFrameFeatures":
        if self.status == CraftDetectionStatus.TEXT_DETECTED:
            if self.detected_region_count == 0 or self.detector_confidence is None:
                raise ValueError("text_detected requires regions and detector confidence")
            if self.min_region_area_ratio is None:
                raise ValueError("text_detected requires min_region_area_ratio")
        else:
            if self.detected_region_count != 0:
                raise ValueError("no_text/error cannot contain detector regions")
            if self.detector_confidence is not None or self.min_region_area_ratio is not None:
                raise ValueError("no_text/error cannot carry detector region metrics")
        if self.reuse_source_uid == self.keyframe_uid:
            raise ValueError("reuse_source_uid must identify a different keyframe")
        reuse_metrics = (
            self.embedding_cosine_to_source,
            self.craft_layout_similarity_to_source,
            self.crop_ssim_to_source,
            self.crop_phash_distance_to_source,
        )
        if self.reuse_source_uid is None and any(value is not None for value in reuse_metrics):
            raise ValueError("reuse metrics require reuse_source_uid")
        if self.reuse_source_uid is not None and any(value is None for value in reuse_metrics):
            raise ValueError("reuse_source_uid requires all four independent checks")
        return self


class OcrEscalationPolicy(BaseModel):
    """Versioned routing policy; real-image Gate 2 must validate all thresholds."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[2] = 2
    model_id: str = Field(
        default="gemini-3.1-flash-lite",
        pattern=r"^gemini-[a-z0-9][a-z0-9.-]*$",
    )
    media_resolution: Literal["MEDIA_RESOLUTION_MEDIUM"] = "MEDIA_RESOLUTION_MEDIUM"
    max_paid_frames: int = Field(default=20_000, ge=0)
    max_budget_vnd: int = Field(default=400_000, ge=0)
    retry_reserve_fraction: float = Field(default=0.15, ge=0, lt=1)
    usd_to_vnd: float = Field(default=26_300, gt=0)
    batch_input_usd_per_million_tokens: float = Field(default=0.125, ge=0)
    batch_output_usd_per_million_tokens: float = Field(default=0.75, ge=0)
    embedding_reuse_threshold: float = Field(default=0.98, ge=-1, le=1)
    craft_layout_reuse_threshold: float = Field(default=0.98, ge=0, le=1)
    crop_ssim_reuse_threshold: float = Field(default=0.98, ge=0, le=1)
    crop_phash_reuse_max_distance: int = Field(default=2, ge=0)


class OcrRecognitionCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keyframe_uid: int = Field(gt=0, le=_MAX_INT64)
    video_id: str
    shot_id: str
    frame_id: int = Field(ge=0)
    source_image: str
    priority_score: float = Field(ge=0)


class OcrReuseDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_keyframe_uid: int = Field(gt=0, le=_MAX_INT64)
    target_keyframe_uid: int = Field(gt=0, le=_MAX_INT64)
    shot_id: str


class OcrEscalationSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[2] = 2
    model_id: str
    media_resolution: Literal["MEDIA_RESOLUTION_MEDIUM"]
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    features_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_records: int = Field(ge=0)
    text_positive_records: int = Field(ge=0)
    local_no_text_records: int = Field(ge=0)
    detector_error_records: int = Field(ge=0)
    reuse_records: int = Field(ge=0)
    recognition_candidate_records: int = Field(ge=0)
    selected_paid_frames: int = Field(ge=0)
    selected_paid_requests: int = Field(ge=0)
    overflow_easyocr_frames: int = Field(ge=0)
    frame_cap: int = Field(ge=0)
    budget_request_cap: int = Field(ge=0)
    estimated_prompt_tokens_per_request: float = Field(gt=0)
    estimated_output_tokens_per_request: float = Field(gt=0)
    estimated_cost_vnd_with_reserve: float = Field(ge=0)
    candidates: list[OcrRecognitionCandidate]
    reuse: list[OcrReuseDecision]
    overflow_keyframe_uids: list[int]
    no_text_keyframe_uids: list[int]
    detector_error_keyframe_uids: list[int]

    @model_validator(mode="after")
    def _validate_counts(self) -> "OcrEscalationSelection":
        if self.selected_paid_frames != len(self.candidates):
            raise ValueError("selected_paid_frames does not match candidates")
        if self.reuse_records != len(self.reuse):
            raise ValueError("reuse_records does not match reuse")
        if self.overflow_easyocr_frames != len(self.overflow_keyframe_uids):
            raise ValueError("overflow count does not match overflow_keyframe_uids")
        if self.local_no_text_records != len(self.no_text_keyframe_uids):
            raise ValueError("no_text count does not match no_text_keyframe_uids")
        if self.detector_error_records != len(self.detector_error_keyframe_uids):
            raise ValueError("detector error count does not match detector_error_keyframe_uids")
        if self.selected_paid_frames > self.frame_cap:
            raise ValueError("selection exceeds paid frame cap")
        if self.selected_paid_requests > self.budget_request_cap:
            raise ValueError("selection exceeds budget request cap")
        return self


class OcrEscalationUsage(BaseModel):
    """Cumulative paid usage persisted by the Gemini recognition worker."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(pattern=r"^gemini-[a-z0-9][a-z0-9.-]*$")
    paid_frames: int = Field(default=0, ge=0)
    paid_requests: int = Field(default=0, ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


def estimate_batch_cost_vnd(
    *,
    prompt_tokens: float,
    output_tokens: float,
    policy: OcrEscalationPolicy,
    include_retry_reserve: bool,
) -> float:
    if prompt_tokens < 0 or output_tokens < 0:
        raise ValueError("token counts cannot be negative")
    cost_usd = (
        prompt_tokens * policy.batch_input_usd_per_million_tokens
        + output_tokens * policy.batch_output_usd_per_million_tokens
    ) / 1_000_000
    cost_vnd = cost_usd * policy.usd_to_vnd
    if include_retry_reserve:
        cost_vnd *= 1 + policy.retry_reserve_fraction
    return cost_vnd


def authorize_escalation_usage(
    current: OcrEscalationUsage,
    *,
    additional_frames: int,
    additional_requests: int,
    estimated_prompt_tokens: int,
    estimated_output_tokens: int,
    policy: OcrEscalationPolicy,
) -> OcrEscalationUsage:
    """Fail before scheduling a shot request that would exceed either paid cap."""

    if current.model_id != policy.model_id:
        raise ValueError("budget ledger model_id does not match policy")
    additions = (
        additional_frames,
        additional_requests,
        estimated_prompt_tokens,
        estimated_output_tokens,
    )
    if any(value < 0 for value in additions):
        raise ValueError("additional usage cannot be negative")
    projected = OcrEscalationUsage(
        model_id=current.model_id,
        paid_frames=current.paid_frames + additional_frames,
        paid_requests=current.paid_requests + additional_requests,
        prompt_tokens=current.prompt_tokens + estimated_prompt_tokens,
        output_tokens=current.output_tokens + estimated_output_tokens,
    )
    if projected.paid_frames > policy.max_paid_frames:
        raise RuntimeError("Gemini paid frame cap would be exceeded")
    projected_vnd = estimate_batch_cost_vnd(
        prompt_tokens=projected.prompt_tokens,
        output_tokens=projected.output_tokens,
        policy=policy,
        include_retry_reserve=True,
    )
    if projected_vnd > policy.max_budget_vnd + 1e-6:
        raise RuntimeError("Gemini VND budget cap would be exceeded")
    return projected


def _reuse_is_safe(row: CraftFrameFeatures, policy: OcrEscalationPolicy) -> bool:
    return bool(
        row.reuse_source_uid is not None
        and row.embedding_cosine_to_source >= policy.embedding_reuse_threshold  # type: ignore[operator]
        and row.craft_layout_similarity_to_source >= policy.craft_layout_reuse_threshold  # type: ignore[operator]
        and row.crop_ssim_to_source >= policy.crop_ssim_reuse_threshold  # type: ignore[operator]
        and row.crop_phash_distance_to_source <= policy.crop_phash_reuse_max_distance  # type: ignore[operator]
    )


def _candidate_for(row: CraftFrameFeatures) -> OcrRecognitionCandidate:
    score = row.visual_priority * 100
    score += min(20, row.detected_region_count * 2)
    score += row.perspective_score * 10
    if row.min_region_area_ratio is not None:
        score += min(15, 0.003 / max(row.min_region_area_ratio, 1e-9))
    return OcrRecognitionCandidate(
        keyframe_uid=row.keyframe_uid,
        video_id=row.video_id,
        shot_id=row.shot_id,
        frame_id=row.frame_id,
        source_image=row.source_image,
        priority_score=round(score, 6),
    )


def select_gemini_escalations(
    rows: Iterable[CraftFrameFeatures],
    *,
    policy: OcrEscalationPolicy,
    estimated_prompt_tokens_per_request: float,
    estimated_output_tokens_per_request: float,
) -> OcrEscalationSelection:
    """Route CRAFT-positive shots to Gemini; overflow uses the local recognizer."""

    if estimated_prompt_tokens_per_request <= 0:
        raise ValueError("estimated_prompt_tokens_per_request must be positive")
    if estimated_output_tokens_per_request <= 0:
        raise ValueError("estimated_output_tokens_per_request must be positive")

    source = list(rows)
    by_uid: dict[int, CraftFrameFeatures] = {}
    feature_digest = hashlib.sha256()
    for row in source:
        if row.keyframe_uid in by_uid:
            raise ValueError(f"duplicate feature UID: {row.keyframe_uid}")
        by_uid[row.keyframe_uid] = row
        feature_digest.update(
            json.dumps(
                row.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        feature_digest.update(b"\n")

    text_rows = [row for row in source if row.status == CraftDetectionStatus.TEXT_DETECTED]
    no_text_uids = sorted(
        row.keyframe_uid for row in source if row.status == CraftDetectionStatus.NO_TEXT
    )
    error_uids = sorted(
        row.keyframe_uid for row in source if row.status == CraftDetectionStatus.ERROR
    )
    reuse: list[OcrReuseDecision] = []
    recognition_rows: list[CraftFrameFeatures] = []
    for row in text_rows:
        if not _reuse_is_safe(row, policy):
            recognition_rows.append(row)
            continue
        source_row = by_uid.get(row.reuse_source_uid)
        if (
            source_row is None
            or source_row.status != CraftDetectionStatus.TEXT_DETECTED
            or source_row.video_id != row.video_id
            or source_row.shot_id != row.shot_id
        ):
            raise ValueError(f"invalid reuse source for UID: {row.keyframe_uid}")
        reuse.append(
            OcrReuseDecision(
                source_keyframe_uid=source_row.keyframe_uid,
                target_keyframe_uid=row.keyframe_uid,
                shot_id=row.shot_id,
            )
        )

    candidates = [_candidate_for(row) for row in recognition_rows]
    grouped: dict[tuple[str, str], list[OcrRecognitionCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[(candidate.video_id, candidate.shot_id)].append(candidate)
    shot_groups = sorted(
        grouped.values(),
        key=lambda group: (
            -max(candidate.priority_score for candidate in group),
            group[0].video_id,
            group[0].shot_id,
        ),
    )
    for group in shot_groups:
        group.sort(key=lambda candidate: (-candidate.priority_score, candidate.keyframe_uid))

    per_request_vnd = estimate_batch_cost_vnd(
        prompt_tokens=estimated_prompt_tokens_per_request,
        output_tokens=estimated_output_tokens_per_request,
        policy=policy,
        include_retry_reserve=True,
    )
    budget_request_cap = (
        math.floor(policy.max_budget_vnd / per_request_vnd) if per_request_vnd else 0
    )
    selected: list[OcrRecognitionCandidate] = []
    selected_requests = 0
    overflow: list[int] = []
    for group in shot_groups:
        remaining_frames = policy.max_paid_frames - len(selected)
        if selected_requests >= budget_request_cap or remaining_frames <= 0:
            overflow.extend(candidate.keyframe_uid for candidate in group)
            continue
        accepted = group[:remaining_frames]
        selected.extend(accepted)
        overflow.extend(candidate.keyframe_uid for candidate in group[remaining_frames:])
        selected_requests += 1

    estimated_cost = per_request_vnd * selected_requests
    if estimated_cost > policy.max_budget_vnd + 1e-6:
        raise RuntimeError("internal error: selection exceeded VND budget")
    policy_bytes = json.dumps(
        policy.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return OcrEscalationSelection(
        model_id=policy.model_id,
        media_resolution=policy.media_resolution,
        policy_sha256=hashlib.sha256(policy_bytes).hexdigest(),
        features_sha256=feature_digest.hexdigest(),
        input_records=len(source),
        text_positive_records=len(text_rows),
        local_no_text_records=len(no_text_uids),
        detector_error_records=len(error_uids),
        reuse_records=len(reuse),
        recognition_candidate_records=len(candidates),
        selected_paid_frames=len(selected),
        selected_paid_requests=selected_requests,
        overflow_easyocr_frames=len(overflow),
        frame_cap=policy.max_paid_frames,
        budget_request_cap=budget_request_cap,
        estimated_prompt_tokens_per_request=estimated_prompt_tokens_per_request,
        estimated_output_tokens_per_request=estimated_output_tokens_per_request,
        estimated_cost_vnd_with_reserve=round(estimated_cost, 2),
        candidates=selected,
        reuse=sorted(reuse, key=lambda row: row.target_keyframe_uid),
        overflow_keyframe_uids=sorted(overflow),
        no_text_keyframe_uids=no_text_uids,
        detector_error_keyframe_uids=error_uids,
    )
