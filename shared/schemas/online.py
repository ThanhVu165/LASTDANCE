"""Canonical contracts shared by the frame-level online runtime and UI."""

from __future__ import annotations

from enum import Enum
from pathlib import PurePath
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TaskType(str, Enum):
    KIS = "KIS"
    QA = "QA"
    TRAKE = "TRAKE"


_QUERY_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*-(kis|qa|trake)$", re.IGNORECASE)


class QuerySpec(BaseModel):
    """One official AIC26 qualifier query and its required submission filename."""

    model_config = ConfigDict(extra="forbid")

    query_name: str
    source_filename: str
    task_type: TaskType
    raw_query: str
    expected_event_count: int | None = Field(default=None, ge=2)

    @field_validator("query_name", "source_filename", "raw_query")
    @classmethod
    def _strip_query_text(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def _validate_query_spec(self) -> "QuerySpec":
        if not self.raw_query:
            raise ValueError("raw_query must not be empty")
        if PurePath(self.source_filename).name != self.source_filename:
            raise ValueError("source_filename must be a plain filename")
        if self.source_filename != f"{self.query_name}.txt":
            raise ValueError("source_filename must equal query_name with the .txt suffix")
        match = _QUERY_NAME.fullmatch(self.query_name)
        if match is None:
            raise ValueError("query_name must end in -kis, -qa or -trake")
        suffix_task = {
            "kis": TaskType.KIS,
            "qa": TaskType.QA,
            "trake": TaskType.TRAKE,
        }[match.group(1).lower()]
        if suffix_task != self.task_type:
            raise ValueError("query filename suffix does not match task_type")
        if self.task_type == TaskType.TRAKE and self.expected_event_count is None:
            raise ValueError("TRAKE query requires expected_event_count")
        if self.task_type != TaskType.TRAKE and self.expected_event_count is not None:
            raise ValueError("expected_event_count is only valid for TRAKE")
        return self

    @property
    def csv_filename(self) -> str:
        return f"{self.query_name}.csv"


class ArtifactAvailability(str, Enum):
    READY = "READY"
    UNAVAILABLE = "UNAVAILABLE"
    INVALID = "INVALID"


class ArtifactStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    availability: ArtifactAvailability
    path: str | None = None
    detail: str = ""
    record_count: int | None = Field(default=None, ge=0)


class UnifiedQueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_query: str
    caption_en: str
    retrieval_queries: list[str] = Field(min_length=1, max_length=2)
    scenes: list[str] = Field(default_factory=list)
    anchor_moment_index: int | None = Field(default=None, ge=0)
    must_have: list[str] = Field(default_factory=list)
    should_have: list[str] = Field(default_factory=list)
    negative_constraints: list[str] = Field(default_factory=list)
    visible_text: list[str] = Field(default_factory=list)
    spoken_text: list[str] = Field(default_factory=list)
    modality_weights: dict[str, float] = Field(default_factory=lambda: {"visual": 1.0})
    question: str | None = None
    answer_format: str | None = None
    answer_source: Literal["visual", "visible_text", "spoken_text", "mixed"] | None = None
    ordered_moments: list[str] = Field(default_factory=list)
    planner_provider: str = "rule"

    @field_validator(
        "raw_query",
        "caption_en",
        "question",
        "answer_format",
        mode="before",
    )
    @classmethod
    def _strip_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator(
        "retrieval_queries",
        "scenes",
        "must_have",
        "should_have",
        "negative_constraints",
        "visible_text",
        "spoken_text",
        "ordered_moments",
    )
    @classmethod
    def _clean_text_lists(cls, values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            text = value.strip()
            if text and text not in result:
                result.append(text)
        return result

    @model_validator(mode="after")
    def _validate_plan(self) -> "UnifiedQueryPlan":
        if not self.raw_query or not self.caption_en:
            raise ValueError("raw_query and caption_en must not be empty")
        if self.anchor_moment_index is not None:
            if not self.scenes or self.anchor_moment_index >= len(self.scenes):
                raise ValueError("anchor_moment_index must reference scenes")
        allowed = {"visual", "ocr", "asr"}
        if set(self.modality_weights) - allowed:
            raise ValueError("modality_weights contains an unsupported channel")
        if any(value < 0 for value in self.modality_weights.values()):
            raise ValueError("modality weights must be non-negative")
        if self.modality_weights.get("visual", 0.0) <= 0:
            raise ValueError("visual modality must have positive weight")
        return self


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_type: TaskType
    raw_query: str
    query_spec: QuerySpec | None = None
    max_results: int = Field(default=100, ge=1, le=100)
    mode: Literal["accurate"] = "accurate"

    @field_validator("raw_query")
    @classmethod
    def _query_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("raw_query must not be empty")
        return value

    @model_validator(mode="after")
    def _query_spec_matches_request(self) -> "SearchRequest":
        if self.query_spec is not None:
            if self.query_spec.task_type != self.task_type:
                raise ValueError("query_spec task_type must match SearchRequest.task_type")
            if self.query_spec.raw_query != self.raw_query:
                raise ValueError("query_spec raw_query must match SearchRequest.raw_query")
        return self


class FrameEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keyframe_uid: int = Field(gt=0)
    video_id: str
    frame_id: int = Field(ge=0)
    pts_time: float = Field(ge=0)
    shot_id: str
    query_part: str = ""
    score_clip: float | None = None
    score_siglip: float | None = None
    score_eva: float | None = None
    score_visual: float = 0.0
    score_ocr: float | None = None
    score_asr: float | None = None
    score_fused: float = 0.0
    neighbor_support: float = 0.0
    final_score: float = 0.0
    vlm_score: float | None = None
    model_consensus: float = Field(default=0.0, ge=0.0, le=1.0)


class VideoHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_id: str
    video_score: float
    base_video_score: float | None = None
    must_have_score: float | None = None
    should_have_score: float | None = None
    vlm_verified: bool = False
    coverage: float = Field(ge=0.0, le=1.0)
    model_consensus: float = Field(ge=0.0, le=1.0)
    best_frames: list[FrameEvidence]
    matched_scenes: list[str] = Field(default_factory=list)
    missing_scenes: list[str] = Field(default_factory=list)


class KISCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_type: Literal["KIS"] = "KIS"
    video_id: str
    frame_id: int = Field(ge=0)
    score: float
    evidence: FrameEvidence


class QACandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_type: Literal["QA"] = "QA"
    video_id: str
    frame_id: int = Field(ge=0)
    answer: str = Field(max_length=100)
    score: float
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: FrameEvidence


class TrakeCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_type: Literal["TRAKE"] = "TRAKE"
    video_id: str
    frame_ids: list[int] = Field(min_length=2)
    pts_times: list[float] = Field(min_length=2)
    score: float
    evidence: list[FrameEvidence]

    @model_validator(mode="after")
    def _validate_sequence(self) -> "TrakeCandidate":
        if len(self.frame_ids) != len(self.pts_times) or len(self.frame_ids) != len(self.evidence):
            raise ValueError("TRAKE frame_ids, pts_times and evidence must align")
        if len(set(self.frame_ids)) != len(self.frame_ids):
            raise ValueError("TRAKE must not reuse a frame")
        if any(right <= left for left, right in zip(self.pts_times, self.pts_times[1:])):
            raise ValueError("TRAKE pts_times must be strictly increasing")
        return self


TaskCandidate = Annotated[
    KISCandidate | QACandidate | TrakeCandidate,
    Field(discriminator="candidate_type"),
]


class SearchRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: SearchRequest
    query_plan: UnifiedQueryPlan
    artifact_status: dict[str, ArtifactStatus]
    video_hypotheses: list[VideoHypothesis]
    top_candidates: list[TaskCandidate] = Field(default_factory=list)
    task_candidates: list[TaskCandidate]
    timings_ms: dict[str, float] = Field(default_factory=dict)
    provenance: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
