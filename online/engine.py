"""Public orchestration boundary for the accuracy-first Online baseline."""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Callable

from shared.interfaces.query_planner import QueryPlanner
from shared.schemas.online import (
    ArtifactAvailability,
    FrameEvidence,
    QueryRole,
    QuerySpec,
    SearchRequest,
    SearchRun,
    TaskType,
    UnifiedQueryPlan,
    VideoHypothesis,
)

from .answering import FtsVideoAnswerer
from .artifacts import ArtifactRegistry, EXPECTED_VISUAL
from .config import OnlineConfig, OnlineLayout
from .encoders import TextEncoderRegistry, WorkerTextEncoderRegistry, get_text_encoder_registry
from .planners import PlannerChain, get_query_planner
from .gemini import get_gemini_quota_manager
from .qwen_runtime import DEFAULT_QWEN_MODEL_ID, resolve_qwen_revision
from .ranking import rank_trake_videos, rank_videos
from .retrieval import FrameRetriever, RetrievalResult
from .task_heads import (
    VideoAnswerer,
    build_kis_candidates,
    build_qa_candidates,
    build_trake_candidates,
)
from .vqa import FallbackVQAAnswerer, get_video_answerer
from .verification import VideoVerifier, get_video_verifier, rerank_with_verifier


Clock = Callable[[], float]


def merge_kis_anchor_frames(
    base_frames: list[FrameEvidence],
    anchor_frames: list[FrameEvidence],
    *,
    bonus: float,
    max_per_shot: int,
) -> list[FrameEvidence]:
    """Merge anchor evidence without ever reducing the original retrieval score."""

    if not 0.0 <= bonus <= 1.0:
        raise ValueError("KIS anchor bonus must be between 0 and 1")
    if max_per_shot <= 0:
        raise ValueError("KIS max frames per shot must be positive")
    anchor_by_uid = {frame.keyframe_uid: frame for frame in anchor_frames}
    anchor_by_shot: dict[str, FrameEvidence] = {}
    for frame in anchor_frames:
        current = anchor_by_shot.get(frame.shot_id)
        if current is None or frame.final_score > current.final_score:
            anchor_by_shot[frame.shot_id] = frame

    merged_by_uid: dict[int, FrameEvidence] = {}
    for frame in base_frames:
        support = anchor_by_uid.get(frame.keyframe_uid) or anchor_by_shot.get(frame.shot_id)
        if support is None:
            merged_by_uid[frame.keyframe_uid] = frame
            continue
        boosted = min(1.0, frame.final_score + bonus * support.final_score)
        merged_by_uid[frame.keyframe_uid] = frame.model_copy(
            update={"final_score": max(frame.final_score, boosted)}
        )
    for frame in anchor_frames:
        merged_by_uid.setdefault(frame.keyframe_uid, frame)

    ordered = sorted(merged_by_uid.values(), key=lambda item: item.final_score, reverse=True)
    shot_counts: dict[str, int] = {}
    selected: list[FrameEvidence] = []
    for frame in ordered:
        if shot_counts.get(frame.shot_id, 0) >= max_per_shot:
            continue
        selected.append(frame)
        shot_counts[frame.shot_id] = shot_counts.get(frame.shot_id, 0) + 1
        if len(selected) >= 100:
            break
    return selected


class OnlineEngine:
    """One in-process API used by Streamlit and diagnostic scripts.

    Artifact validation is fail-closed. Planner, visual text encoders and VQA are
    injectable so the deterministic core can be tested without model downloads.
    """

    def __init__(
        self,
        *,
        registry: ArtifactRegistry | None = None,
        layout: OnlineLayout | None = None,
        config: OnlineConfig | None = None,
        planner: QueryPlanner | None = None,
        encoders: TextEncoderRegistry | WorkerTextEncoderRegistry | None = None,
        answerer: VideoAnswerer | None = None,
        verifier: VideoVerifier | None = None,
        deep_preflight: bool = False,
        clock: Clock = time.perf_counter,
    ) -> None:
        self.config = config or OnlineConfig.load()
        self.registry = registry or ArtifactRegistry.load(layout, deep=deep_preflight)
        self.planner = planner or get_query_planner()
        self.encoders = encoders or get_text_encoder_registry(device="cpu")
        self.answerer = answerer or get_video_answerer(self.registry, self.config)
        self.verifier = verifier if verifier is not None else get_video_verifier(self.registry)
        self.retriever = FrameRetriever(self.registry, self.encoders, self.config)
        self._clock = clock

    @classmethod
    def from_environment(cls, *, deep_preflight: bool = False) -> "OnlineEngine":
        return cls(
            layout=OnlineLayout.from_environment(),
            deep_preflight=deep_preflight,
        )

    def plan(self, query_spec: QuerySpec) -> UnifiedQueryPlan:
        """Build and task-validate a reviewable plan without running retrieval."""

        if not isinstance(query_spec, QuerySpec):
            query_spec = QuerySpec.model_validate(query_spec)
        get_gemini_quota_manager().begin_search(
            max_calls=self.config.gemini_max_calls_per_search,
            timeout_seconds=self.config.gemini_search_timeout_seconds,
        )
        plan = self.planner.plan(query_spec.raw_query, query_spec.task_type)
        expected = query_spec.expected_event_count if query_spec.task_type == TaskType.TRAKE else None
        plan.validate_for_task(query_spec.task_type, expected_event_count=expected)
        fallbacks = self._planner_warnings()
        if fallbacks:
            plan = plan.model_copy(
                update={
                    "planner_warnings": list(
                        dict.fromkeys([*plan.planner_warnings, *fallbacks])
                    )
                }
            )
        return plan

    def search(
        self,
        request: SearchRequest,
        query_plan: UnifiedQueryPlan | None = None,
    ) -> SearchRun:
        """Retrieve and rank from an optional operator-reviewed role-aware plan."""

        if not isinstance(request, SearchRequest):
            request = SearchRequest.model_validate(request)
        if query_plan is not None and not isinstance(query_plan, UnifiedQueryPlan):
            query_plan = UnifiedQueryPlan.model_validate(query_plan)
        remaining_calls = self.config.gemini_max_calls_per_search
        if query_plan is not None and query_plan.planner_provider == "gemini":
            remaining_calls = max(1, remaining_calls - 1)
        get_gemini_quota_manager().begin_search(
            max_calls=remaining_calls,
            timeout_seconds=self.config.gemini_search_timeout_seconds,
        )
        started = self._clock()
        phase_started = started
        if query_plan is None:
            plan = self.planner.plan(request.raw_query, request.task_type)
            planner_ms = self._elapsed_ms(phase_started)
            warnings = self._planner_warnings()
        else:
            plan = query_plan
            planner_ms = 0.0
            warnings = []
        if plan.raw_query != request.raw_query:
            raise ValueError("query plan raw_query must match SearchRequest.raw_query")
        expected = (
            request.query_spec.expected_event_count
            if request.query_spec is not None and request.task_type == TaskType.TRAKE
            else None
        )
        plan.validate_for_task(request.task_type, expected_event_count=expected)
        timings = {"planner": planner_ms}
        warnings.extend(plan.planner_warnings)
        planned_visual_text = [
            plan.global_context_en,
            *plan.retrieval_queries,
            *(unit.retrieval_query_en for unit in plan.query_units),
            *plan.must_have,
            *plan.should_have,
        ]
        if plan.planner_provider == "rule" and any(
            not value.isascii() for value in planned_visual_text
        ):
            warnings.append(
                "Rule planner cannot guarantee English visual queries; "
                "accuracy is degraded until Gemini or Qwen is available"
            )

        if request.task_type == TaskType.TRAKE:
            hypotheses, candidates, task_warnings, retrieval_ms, ranking_ms, head_ms = self._search_trake(
                request, plan
            )
        else:
            phase_started = self._clock()
            retrieval = self.retriever.search(plan)
            retrieval_ms = self._elapsed_ms(phase_started)
            phase_started = self._clock()
            hypotheses = rank_videos(retrieval, plan, self.registry, self.config)
            ranking_ms = self._elapsed_ms(phase_started)
            phase_started = self._clock()
            hypotheses, target_warnings = self._apply_task_target_frames(
                plan,
                hypotheses,
                request.task_type,
            )
            retrieval_ms += self._elapsed_ms(phase_started)
            warnings.extend(target_warnings)
            hypotheses, verification_warnings = rerank_with_verifier(
                hypotheses,
                plan,
                self.verifier,
                self.config,
            )
            warnings.extend(verification_warnings)
            phase_started = self._clock()
            if request.task_type == TaskType.KIS:
                candidates = build_kis_candidates(
                    hypotheses,
                    max_results=request.max_results,
                    config=self.config,
                )
                task_warnings: list[str] = []
            else:
                qa_answerer = self._qa_answerer(plan)
                candidates, task_warnings = build_qa_candidates(
                    hypotheses,
                    plan,
                    answerer=qa_answerer,
                    max_results=request.max_results,
                    config=self.config,
                )
            head_ms = self._elapsed_ms(phase_started)
            warnings.extend(retrieval.warnings)
            if retrieval.degraded_to_clip:
                warnings.append("Visual retrieval is running in explicit CLIP rollback mode")

        warnings.extend(task_warnings)
        timings.update(
            {
                "retrieval": retrieval_ms,
                "video_ranking": ranking_ms,
                "task_head": head_ms,
                "total": self._elapsed_ms(started),
            }
        )
        return SearchRun(
            request=request,
            query_plan=plan,
            artifact_status=self.registry.statuses,
            video_hypotheses=hypotheses,
            top_candidates=candidates[: request.max_results],
            task_candidates=candidates[: request.max_results],
            timings_ms={key: round(value, 3) for key, value in timings.items()},
            provenance=self._provenance(plan),
            warnings=list(dict.fromkeys(warnings)),
        )

    def _search_trake(
        self,
        request: SearchRequest,
        plan: UnifiedQueryPlan,
    ) -> tuple[list, list, list[str], float, float, float]:
        event_units = plan.units_by_id(plan.ordered_event_ids)
        moments = self._ordered_moments(plan, request)
        phase_started = self._clock()
        moment_results: list[RetrievalResult] = []
        warnings: list[str] = []
        locator_result = self.retriever.search(
            plan,
            roles={QueryRole.VIDEO_LOCATOR},
            include_global=True,
        )
        warnings.extend(locator_result.warnings)
        for unit in event_units:
            literals = list(unit.known_text_literals)
            moment_plan = plan.model_copy(
                update={
                    "query_units": [unit],
                    "visible_text": literals if "ocr" in unit.modalities else [],
                    "spoken_text": literals if "asr" in unit.modalities else [],
                    "ordered_moments": [],
                }
            )
            result = self.retriever.search(
                moment_plan,
                roles={QueryRole.ORDERED_EVENT},
                include_global=False,
            )
            moment_results.append(result)
            warnings.extend(result.warnings)
            if result.degraded_to_clip:
                warnings.append(
                    f"TRAKE event used explicit CLIP rollback: {unit.retrieval_query_en}"
                )
        retrieval_ms = self._elapsed_ms(phase_started)

        phase_started = self._clock()
        hypotheses = rank_trake_videos(
            moment_results,
            moments,
            self.registry,
            self.config,
            locator_result=locator_result,
        )
        hypotheses, verification_warnings = rerank_with_verifier(
            hypotheses,
            plan,
            self.verifier,
            self.config,
        )
        warnings.extend(verification_warnings)
        ranking_ms = self._elapsed_ms(phase_started)
        phase_started = self._clock()
        candidates = build_trake_candidates(
            hypotheses,
            moment_results,
            max_results=request.max_results,
            config=self.config,
        )
        head_ms = self._elapsed_ms(phase_started)
        return hypotheses, candidates, warnings, retrieval_ms, ranking_ms, head_ms

    def _apply_task_target_frames(
        self,
        plan: UnifiedQueryPlan,
        hypotheses: list[VideoHypothesis],
        task_type: TaskType,
    ) -> tuple[list[VideoHypothesis], list[str]]:
        if task_type == TaskType.QA:
            assert plan.answer_target is not None
            target_ids = plan.answer_target.evidence_unit_ids
            target_role = QueryRole.ANSWER_EVIDENCE
        else:
            target_ids = plan.submission_target_ids
            target_role = QueryRole.TARGET_MOMENT
        units = plan.units_by_id(target_ids)
        if not units:
            raise ValueError(f"{task_type.value} has no target units for frame localization")
        literals = list(
            dict.fromkeys(
                literal for unit in units for literal in unit.known_text_literals
            )
        )
        target_plan = plan.model_copy(
            update={
                "query_units": units,
                "visible_text": [
                    literal
                    for literal in literals
                    if any("ocr" in unit.modalities for unit in units)
                ],
                "spoken_text": [
                    literal
                    for literal in literals
                    if any("asr" in unit.modalities for unit in units)
                ],
            }
        )
        target_result = self.retriever.search(
            target_plan,
            roles={target_role},
            include_global=False,
        )
        selected_videos = {hypothesis.video_id for hypothesis in hypotheses}
        by_video: dict[str, list[FrameEvidence]] = {}
        for frame in target_result.evidence:
            if frame.video_id in selected_videos:
                by_video.setdefault(frame.video_id, []).append(frame)
        updated = [
            hypothesis.model_copy(
                update={
                    # Do not shot-dedup here: adjacent keyframes can contain the
                    # exact KIS frame or a clearer OCR answer than the shot leader.
                    "best_frames": sorted(
                        by_video.get(hypothesis.video_id, []),
                        key=lambda frame: frame.final_score,
                        reverse=True,
                    )[:100]
                }
            )
            for hypothesis in hypotheses
        ]
        warnings = list(target_result.warnings)
        if target_result.degraded_to_clip:
            warnings.append(
                f"{task_type.value} target localization used explicit CLIP rollback"
            )
        return updated, warnings

    def _qa_answerer(self, plan: UnifiedQueryPlan) -> VideoAnswerer:
        if plan.answer_target is None:
            return self.answerer
        source = plan.answer_target.source
        units = plan.units_by_id(plan.answer_target.evidence_unit_ids)
        providers: list[VideoAnswerer] = []
        if source in {"ocr", "mixed"}:
            status = self.registry.statuses["ocr"].availability
            if status == ArtifactAvailability.READY:
                literals = list(
                    dict.fromkeys(
                        literal
                        for unit in units
                        if "ocr" in unit.modalities
                        for literal in unit.known_text_literals
                    )
                )
                providers.append(
                    FtsVideoAnswerer(
                        self.registry,
                        "ocr",
                        literals,
                        value_type=plan.answer_target.value_type,
                    )
                )
        if source in {"asr", "mixed"}:
            status = self.registry.statuses["asr"].availability
            if status == ArtifactAvailability.READY:
                literals = list(
                    dict.fromkeys(
                        literal
                        for unit in units
                        if "asr" in unit.modalities
                        for literal in unit.known_text_literals
                    )
                )
                providers.append(
                    FtsVideoAnswerer(
                        self.registry,
                        "asr",
                        literals,
                        value_type=plan.answer_target.value_type,
                    )
                )
        providers.append(self.answerer)
        return FallbackVQAAnswerer(providers)

    @staticmethod
    def _ordered_moments(
        plan: UnifiedQueryPlan,
        request: SearchRequest | None = None,
    ) -> list[str]:
        moments = [
            unit.retrieval_query_en
            for unit in plan.units_by_id(plan.ordered_event_ids)
        ]
        if len(moments) < 2:
            raise ValueError(
                "TRAKE requires at least two explicit ORDERED_EVENT units"
            )
        if request is not None and request.query_spec is not None:
            expected = request.query_spec.expected_event_count
            if expected is not None and len(moments) != expected:
                raise ValueError(
                    f"TRAKE planner returned {len(moments)} moments; official query requires {expected}"
                )
        return moments

    def _planner_warnings(self) -> list[str]:
        if isinstance(self.planner, PlannerChain):
            return [f"Planner fallback: {item}" for item in self.planner.last_errors]
        return []

    def _provenance(self, plan: UnifiedQueryPlan) -> dict[str, str]:
        provenance = {
            "catalog_sha256": self.registry.catalog.sha256,
            "planner_provider": plan.planner_provider,
            "online_config": "configs/online_baseline.json",
            "fusion": "SigLIP+EVA SRRF; CLIP rollback/tie evidence",
            "query_plan_schema": "role-aware-v1",
            "query_plan_reviewed": str(plan.operator_reviewed).lower(),
            "query_plan_edited": str(plan.operator_edited).lower(),
            "query_plan_sha256": hashlib.sha256(
                plan.model_dump_json(exclude={"planner_warnings"}).encode("utf-8")
            ).hexdigest(),
            "torch_execution": (
                "isolated-worker" if isinstance(self.encoders, WorkerTextEncoderRegistry) else "in-process"
            ),
        }
        if plan.planner_provider == "qwen-local":
            qwen_model = os.environ.get("AIC_QWEN_MODEL", DEFAULT_QWEN_MODEL_ID)
            provenance["qwen_model"] = qwen_model
            provenance["qwen_revision"] = resolve_qwen_revision(qwen_model)
        for modality, (_, model_id, revision) in EXPECTED_VISUAL.items():
            provenance[f"{modality}_model"] = model_id
            provenance[f"{modality}_revision"] = revision
            state = self.registry.visual[modality].state
            if state.get("index_sha256"):
                provenance[f"{modality}_index_sha256"] = str(state["index_sha256"])
        coverage_path = self.registry.layout.ocr_coverage
        ocr_status = self.registry.statuses.get("ocr")
        if (
            coverage_path is not None
            and coverage_path.is_file()
            and ocr_status is not None
            and ocr_status.availability == ArtifactAvailability.READY
        ):
            coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
            provenance["ocr_snapshot_id"] = str(coverage["snapshot_id"])
            provenance["ocr_intended_use"] = str(coverage["intended_use"])
            provenance["ocr_coverage_fraction"] = str(coverage["coverage_fraction"])
            provenance["ocr_error_keyframes"] = str(coverage["error_keyframes"])
            provenance["ocr_sqlite_sha256"] = str(coverage["sqlite_sha256"])
        return provenance

    def _elapsed_ms(self, started: float) -> float:
        return (self._clock() - started) * 1000.0
