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


class QueryRole(str, Enum):
    VIDEO_LOCATOR = "VIDEO_LOCATOR"
    TARGET_MOMENT = "TARGET_MOMENT"
    ANSWER_EVIDENCE = "ANSWER_EVIDENCE"
    ORDERED_EVENT = "ORDERED_EVENT"


class QueryUnit(BaseModel):
    """One grounded query span with one or more retrieval/task roles."""

    model_config = ConfigDict(extra="forbid")

    unit_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    description_original: str
    retrieval_query_en: str
    roles: list[QueryRole] = Field(min_length=1)
    requiredness: Literal["must", "should"] = "must"
    modalities: list[Literal["visual", "ocr", "asr"]] = Field(
        default_factory=lambda: ["visual"], min_length=1
    )
    temporal_group: int | None = Field(default=None, ge=0)
    temporal_order: int | None = Field(default=None, ge=0)
    known_text_literals: list[str] = Field(default_factory=list)
    visual_text_attributes: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("description_original", "retrieval_query_en")
    @classmethod
    def _unit_text_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query unit text must not be empty")
        return value

    @field_validator("roles", "modalities")
    @classmethod
    def _deduplicate_enums(cls, values: list[object]) -> list[object]:
        return list(dict.fromkeys(values))

    @field_validator("known_text_literals", "visual_text_attributes")
    @classmethod
    def _clean_unit_text_lists(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))


class AnswerTarget(BaseModel):
    """A value that QA must extract, never a known literal for FTS search."""

    model_config = ConfigDict(extra="forbid")

    question: str
    value_type: Literal["number", "color", "person", "place", "free_text"] = "free_text"
    source: Literal["visual", "ocr", "asr", "mixed"] = "visual"
    evidence_unit_ids: list[str] = Field(min_length=1)
    value_is_unknown: Literal[True] = True

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("answer target question must not be empty")
        return value

    @field_validator("evidence_unit_ids")
    @classmethod
    def _deduplicate_evidence_ids(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))


class UnifiedQueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_query: str
    global_context_en: str = ""
    query_units: list[QueryUnit] = Field(default_factory=list)
    answer_target: AnswerTarget | None = None
    ordered_event_ids: list[str] = Field(default_factory=list)
    submission_target_ids: list[str] = Field(default_factory=list)
    planner_warnings: list[str] = Field(default_factory=list)
    operator_reviewed: bool = False
    operator_edited: bool = False

    # Compatibility-only legacy fields. Online Core normalizes these into
    # query_units and never uses scenes to decide task output.
    caption_en: str = ""
    retrieval_queries: list[str] = Field(default_factory=list, max_length=2)
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

    @model_validator(mode="before")
    @classmethod
    def _adapt_legacy_plan(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        raw = str(data.get("raw_query", "")).strip()
        global_context = str(
            data.get("global_context_en") or data.get("caption_en") or raw
        ).strip()
        data["global_context_en"] = global_context
        data["caption_en"] = str(data.get("caption_en") or global_context).strip()
        data["retrieval_queries"] = data.get("retrieval_queries") or [global_context]

        if not data.get("query_units"):
            scenes = [
                str(item).strip()
                for item in (data.get("scenes") or [global_context])
                if str(item).strip()
            ]
            ordered = [
                str(item).strip()
                for item in (data.get("ordered_moments") or [])
                if str(item).strip()
            ]
            units: list[dict[str, object]] = []
            if ordered:
                for index, text in enumerate(ordered):
                    units.append(
                        {
                            "unit_id": f"event-{index + 1}",
                            "description_original": text,
                            "retrieval_query_en": text,
                            "roles": ["VIDEO_LOCATOR", "ORDERED_EVENT"],
                            "temporal_group": index,
                            "temporal_order": index,
                        }
                    )
                data["ordered_event_ids"] = data.get("ordered_event_ids") or [
                    str(item["unit_id"]) for item in units
                ]
            else:
                anchor = data.get("anchor_moment_index")
                try:
                    anchor_index = int(anchor) if anchor is not None else len(scenes) - 1
                except (TypeError, ValueError):
                    anchor_index = len(scenes) - 1
                anchor_index = min(max(anchor_index, 0), max(len(scenes) - 1, 0))
                is_qa = bool(data.get("question") or data.get("answer_source"))
                for index, text in enumerate(scenes):
                    roles = ["VIDEO_LOCATOR"]
                    if index == anchor_index:
                        roles.append("ANSWER_EVIDENCE" if is_qa else "TARGET_MOMENT")
                        if is_qa:
                            roles.append("TARGET_MOMENT")
                    units.append(
                        {
                            "unit_id": f"unit-{index + 1}",
                            "description_original": text,
                            "retrieval_query_en": text,
                            "roles": roles,
                            "temporal_group": index,
                            "temporal_order": index,
                        }
                    )
                target_id = str(units[anchor_index]["unit_id"]) if units else ""
                data["submission_target_ids"] = data.get("submission_target_ids") or (
                    [target_id] if target_id else []
                )
                if is_qa and data.get("answer_target") is None and target_id:
                    legacy_source = str(data.get("answer_source") or "visual")
                    source = {
                        "visible_text": "ocr",
                        "spoken_text": "asr",
                    }.get(legacy_source, legacy_source)
                    data["answer_target"] = {
                        "question": data.get("question") or raw,
                        "value_type": "free_text",
                        "source": source if source in {"visual", "ocr", "asr", "mixed"} else "visual",
                        "evidence_unit_ids": [target_id],
                        "value_is_unknown": True,
                    }
            data["query_units"] = units
        else:
            units = data["query_units"]
            if not data.get("scenes"):
                data["scenes"] = [
                    item.get("retrieval_query_en", "")
                    for item in units
                    if isinstance(item, dict) and item.get("retrieval_query_en")
                ]

        unit_values = data.get("query_units") or []
        if not data.get("ordered_event_ids"):
            data["ordered_event_ids"] = [
                str(item.get("unit_id"))
                for item in unit_values
                if isinstance(item, dict) and "ORDERED_EVENT" in item.get("roles", [])
            ]
        if not data.get("submission_target_ids"):
            data["submission_target_ids"] = [
                str(item.get("unit_id"))
                for item in unit_values
                if isinstance(item, dict)
                and any(
                    role in item.get("roles", [])
                    for role in ("TARGET_MOMENT", "ANSWER_EVIDENCE")
                )
            ]
        return data

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
        "ordered_event_ids",
        "submission_target_ids",
        "planner_warnings",
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
        if not self.raw_query or not self.caption_en or not self.global_context_en:
            raise ValueError("raw_query and global English context must not be empty")
        if not self.retrieval_queries:
            raise ValueError("at least one global retrieval query is required")
        if not self.query_units:
            raise ValueError("query plan must contain at least one query unit")
        units = {unit.unit_id: unit for unit in self.query_units}
        if len(units) != len(self.query_units):
            raise ValueError("query unit IDs must be unique")
        for identifier in self.ordered_event_ids + self.submission_target_ids:
            if identifier not in units:
                raise ValueError(f"query plan references unknown unit ID: {identifier}")
        for identifier in self.ordered_event_ids:
            if QueryRole.ORDERED_EVENT not in units[identifier].roles:
                raise ValueError("ordered_event_ids may only reference ORDERED_EVENT units")
        if self.answer_target is not None:
            for identifier in self.answer_target.evidence_unit_ids:
                if identifier not in units:
                    raise ValueError("answer target references an unknown evidence unit")
                if QueryRole.ANSWER_EVIDENCE not in units[identifier].roles:
                    raise ValueError("answer target must reference ANSWER_EVIDENCE units")
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

    def units_for_role(self, role: QueryRole) -> list[QueryUnit]:
        return [unit for unit in self.query_units if role in unit.roles]

    def units_by_id(self, identifiers: list[str]) -> list[QueryUnit]:
        lookup = {unit.unit_id: unit for unit in self.query_units}
        return [lookup[identifier] for identifier in identifiers if identifier in lookup]

    def validate_for_task(
        self,
        task_type: TaskType,
        *,
        expected_event_count: int | None = None,
    ) -> "UnifiedQueryPlan":
        if task_type == TaskType.KIS:
            targets = self.units_by_id(self.submission_target_ids)
            if not targets or any(QueryRole.TARGET_MOMENT not in unit.roles for unit in targets):
                raise ValueError("KIS requires at least one TARGET_MOMENT submission unit")
        elif task_type == TaskType.QA:
            if self.answer_target is None:
                raise ValueError("QA requires an unknown AnswerTarget")
            if not self.answer_target.evidence_unit_ids:
                raise ValueError("QA requires at least one ANSWER_EVIDENCE unit")
            targets = self.units_by_id(self.submission_target_ids)
            if not targets or any(QueryRole.TARGET_MOMENT not in unit.roles for unit in targets):
                raise ValueError("QA submission_target_ids must reference TARGET_MOMENT units")
            if not set(self.answer_target.evidence_unit_ids).issubset(self.submission_target_ids):
                raise ValueError("QA answer evidence must be included in submission targets")
        else:
            if len(self.ordered_event_ids) < 2:
                raise ValueError("TRAKE requires at least two ORDERED_EVENT units")
            if expected_event_count is not None and len(self.ordered_event_ids) != expected_event_count:
                raise ValueError(
                    f"TRAKE plan has {len(self.ordered_event_ids)} events; expected {expected_event_count}"
                )
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
    requires_review: bool = True
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
