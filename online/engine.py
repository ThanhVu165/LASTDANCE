"""Public orchestration boundary for the accuracy-first Online baseline."""

from __future__ import annotations

import json
import os
import time
from typing import Callable

from shared.interfaces.query_planner import QueryPlanner
from shared.schemas.online import (
    ArtifactAvailability,
    FrameEvidence,
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
from .vqa import get_video_answerer
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
        self.answerer = answerer or get_video_answerer(self.registry)
        self.verifier = verifier if verifier is not None else get_video_verifier(self.registry)
        self.retriever = FrameRetriever(self.registry, self.encoders, self.config)
        self._clock = clock

    @classmethod
    def from_environment(cls, *, deep_preflight: bool = False) -> "OnlineEngine":
        return cls(
            layout=OnlineLayout.from_environment(),
            deep_preflight=deep_preflight,
        )

    def search(self, request: SearchRequest) -> SearchRun:
        """Plan, retrieve, rank and build at most 100 task hypotheses."""

        if not isinstance(request, SearchRequest):
            request = SearchRequest.model_validate(request)
        get_gemini_quota_manager().begin_search(
            max_calls=self.config.gemini_max_calls_per_search,
            timeout_seconds=self.config.gemini_search_timeout_seconds,
        )
        started = self._clock()
        phase_started = started
        plan = self.planner.plan(request.raw_query, request.task_type)
        timings = {"planner": self._elapsed_ms(phase_started)}
        warnings = self._planner_warnings()
        planned_visual_text = [
            plan.caption_en,
            *plan.retrieval_queries,
            *plan.scenes,
            *plan.must_have,
            *plan.should_have,
            *plan.ordered_moments,
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
            if request.task_type == TaskType.KIS:
                phase_started = self._clock()
                hypotheses, anchor_warnings = self._apply_kis_anchor(
                    plan, hypotheses, retrieval
                )
                retrieval_ms += self._elapsed_ms(phase_started)
                warnings.extend(anchor_warnings)
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
        moments = self._ordered_moments(plan, request)
        phase_started = self._clock()
        moment_results: list[RetrievalResult] = []
        warnings: list[str] = []
        for moment in moments:
            moment_plan = plan.model_copy(
                update={
                    "caption_en": moment,
                    "retrieval_queries": [moment],
                    "scenes": [moment],
                    "anchor_moment_index": 0,
                    "ordered_moments": [],
                }
            )
            result = self.retriever.search(moment_plan)
            moment_results.append(result)
            warnings.extend(result.warnings)
            if result.degraded_to_clip:
                warnings.append(f"TRAKE moment used explicit CLIP rollback: {moment}")
        retrieval_ms = self._elapsed_ms(phase_started)

        phase_started = self._clock()
        hypotheses = rank_trake_videos(moment_results, moments, self.registry, self.config)
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

    def _apply_kis_anchor(
        self,
        plan: UnifiedQueryPlan,
        hypotheses: list[VideoHypothesis],
        base_result: RetrievalResult,
    ) -> tuple[list[VideoHypothesis], list[str]]:
        if plan.anchor_moment_index is None or len(plan.scenes) <= 1:
            return hypotheses, []
        anchor = plan.scenes[plan.anchor_moment_index]
        if anchor in plan.retrieval_queries:
            return hypotheses, []
        anchor_plan = plan.model_copy(
            update={
                "caption_en": anchor,
                "retrieval_queries": [anchor],
                "scenes": [anchor],
                "anchor_moment_index": 0,
            }
        )
        anchor_result = self.retriever.search(anchor_plan)
        by_video: dict[str, list[FrameEvidence]] = {}
        for frame in anchor_result.evidence:
            by_video.setdefault(frame.video_id, []).append(frame)
        selected_videos = {hypothesis.video_id for hypothesis in hypotheses}
        base_by_video: dict[str, list[FrameEvidence]] = {}
        for frame in base_result.evidence:
            if frame.video_id in selected_videos:
                base_by_video.setdefault(frame.video_id, []).append(frame)
        updated: list[VideoHypothesis] = []
        for hypothesis in hypotheses:
            frames = merge_kis_anchor_frames(
                base_by_video.get(hypothesis.video_id, hypothesis.best_frames),
                by_video.get(hypothesis.video_id, []),
                bonus=self.config.kis_anchor_bonus,
                max_per_shot=self.config.kis_max_frames_per_shot,
            )
            updated.append(hypothesis.model_copy(update={"best_frames": frames}))
        warnings = list(anchor_result.warnings)
        if anchor_result.degraded_to_clip:
            warnings.append("KIS anchor refinement used explicit CLIP rollback")
        return updated, warnings

    def _qa_answerer(self, plan: UnifiedQueryPlan) -> VideoAnswerer:
        source = plan.answer_source
        if source in {"visible_text", "mixed"}:
            status = self.registry.statuses["ocr"].availability
            if status == ArtifactAvailability.READY:
                return FtsVideoAnswerer(self.registry, "ocr", plan.visible_text)
        if source in {"spoken_text", "mixed"}:
            status = self.registry.statuses["asr"].availability
            if status == ArtifactAvailability.READY:
                return FtsVideoAnswerer(self.registry, "asr", plan.spoken_text)
        return self.answerer

    @staticmethod
    def _ordered_moments(
        plan: UnifiedQueryPlan,
        request: SearchRequest | None = None,
    ) -> list[str]:
        moments = list(plan.ordered_moments or plan.scenes)
        if len(moments) < 2:
            raise ValueError(
                "TRAKE requires at least two ordered moments; refine the query with explicit temporal steps"
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
