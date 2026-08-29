"""Aggregate independent frame evidence into ranked video hypotheses."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Iterable

from shared.schemas.online import FrameEvidence, QueryRole, UnifiedQueryPlan, VideoHypothesis

from .artifacts import ArtifactRegistry
from .config import OnlineConfig
from .retrieval import RetrievalResult


def _distinct_shot(frames: Iterable[FrameEvidence]) -> list[FrameEvidence]:
    best: dict[str, FrameEvidence] = {}
    for frame in frames:
        current = best.get(frame.shot_id)
        if current is None or frame.final_score > current.final_score:
            best[frame.shot_id] = frame
    return sorted(best.values(), key=lambda item: item.final_score, reverse=True)


def apply_clip_tie_break(
    hypotheses: list[VideoHypothesis],
    *,
    margin: float,
) -> list[VideoHypothesis]:
    """Use CLIP only to reorder close video scores; never add it to score_visual."""

    if margin <= 0 or len(hypotheses) < 2:
        return hypotheses
    result: list[VideoHypothesis] = []
    start = 0
    while start < len(hypotheses):
        leader = hypotheses[start].video_score
        stop = start + 1
        while stop < len(hypotheses) and leader - hypotheses[stop].video_score <= margin:
            stop += 1
        group = hypotheses[start:stop]

        def clip_support(item: VideoHypothesis) -> float:
            values = [frame.score_clip for frame in item.best_frames[:3] if frame.score_clip is not None]
            return max(values) if values else -float("inf")

        if any(clip_support(item) > -float("inf") for item in group):
            group = sorted(group, key=lambda item: (clip_support(item), item.video_score), reverse=True)
        result.extend(group)
        start = stop
    return result


def rank_videos(
    result: RetrievalResult,
    plan: UnifiedQueryPlan,
    registry: ArtifactRegistry,
    config: OnlineConfig,
    *,
    limit: int | None = None,
) -> list[VideoHypothesis]:
    by_video: dict[str, list[FrameEvidence]] = defaultdict(list)
    for frame in result.evidence:
        by_video[frame.video_id].append(frame)

    query_video_scores: dict[str, dict[str, float]] = {}
    query_top_videos: dict[str, set[str]] = {}
    for query_text, scores in result.per_query_scores.items():
        grouped: dict[str, dict[str, float]] = defaultdict(dict)
        for uid, score in scores.items():
            frame = registry.catalog.by_uid.get(uid)
            if frame is None:
                continue
            grouped[frame.video_id][frame.shot_id] = max(grouped[frame.video_id].get(frame.shot_id, 0.0), float(score))
        video_scores: dict[str, float] = {}
        for video_id, shot_scores in grouped.items():
            values = sorted(shot_scores.values(), reverse=True)[: config.video_frame_evidence_k]
            if not values:
                continue
            remainder_weight = 1.0 - config.video_evidence_max_weight
            video_scores[video_id] = config.video_evidence_max_weight * values[0] + remainder_weight * mean(values)
        query_video_scores[query_text] = video_scores
        ranked = sorted(video_scores, key=video_scores.get, reverse=True)[: config.video_part_top_k]
        query_top_videos[query_text] = set(ranked)

    locator_labels = list(
        dict.fromkeys(
            unit.retrieval_query_en
            for unit in plan.units_for_role(QueryRole.VIDEO_LOCATOR)
        )
    )
    target_labels = list(
        dict.fromkeys(
            unit.retrieval_query_en
            for unit in plan.query_units
            if QueryRole.TARGET_MOMENT in unit.roles
            or QueryRole.ANSWER_EVIDENCE in unit.roles
        )
    )
    if not target_labels:
        target_labels = locator_labels or [plan.global_context_en]
    if not locator_labels:
        locator_labels = target_labels
    global_queries = list(dict.fromkeys([plan.global_context_en, *plan.retrieval_queries]))
    hypotheses: list[VideoHypothesis] = []
    all_videos = set(by_video) | set().union(
        *(scores.keys() for scores in query_video_scores.values())
    )
    for video_id in all_videos:
        def role_evidence(labels: list[str]) -> tuple[float, list[bool]]:
            scores = [query_video_scores.get(label, {}).get(video_id, 0.0) for label in labels]
            hits = [video_id in query_top_videos.get(label, set()) for label in labels]
            coverage_value = sum(hits) / len(hits) if hits else 0.0
            mean_value = mean(scores) if scores else 0.0
            weakest_value = min(scores) if scores else 0.0
            return 0.50 * coverage_value + 0.30 * mean_value + 0.20 * weakest_value, hits

        locator_evidence, _locator_hits = role_evidence(locator_labels)
        target_evidence, coverage_hits = role_evidence(target_labels)
        coverage = sum(coverage_hits) / len(coverage_hits) if coverage_hits else 0.0
        global_values = [
            query_video_scores.get(query, {}).get(video_id, 0.0) for query in global_queries
        ]
        global_score = max(global_values) if global_values else target_evidence
        distinct = _distinct_shot(by_video.get(video_id, []))
        consensus = mean([frame.model_consensus for frame in distinct[:3]]) if distinct else 0.0
        base_score = (
            config.video_locator_weight * locator_evidence
            + config.video_target_weight * target_evidence
            + config.video_global_weight * global_score
            + config.video_consensus_weight * consensus
        )
        must_values = [
            query_video_scores.get(query, {}).get(video_id, 0.0) for query in plan.must_have
        ]
        should_values = [
            query_video_scores.get(query, {}).get(video_id, 0.0) for query in plan.should_have
        ]
        hypotheses.append(
            VideoHypothesis(
                video_id=video_id,
                video_score=base_score,
                base_video_score=base_score,
                must_have_score=mean(must_values) if must_values else base_score,
                should_have_score=mean(should_values) if should_values else base_score,
                coverage=coverage,
                model_consensus=consensus,
                best_frames=distinct[:100],
                matched_scenes=[
                    label for label, hit in zip(target_labels, coverage_hits) if hit
                ],
                missing_scenes=[
                    label for label, hit in zip(target_labels, coverage_hits) if not hit
                ],
            )
        )
    hypotheses.sort(key=lambda item: item.video_score, reverse=True)
    hypotheses = apply_clip_tie_break(hypotheses, margin=config.clip_tie_margin)
    return hypotheses[: (limit or config.video_top_k)]


def rank_trake_videos(
    moment_results: list[RetrievalResult],
    moments: list[str],
    registry: ArtifactRegistry,
    config: OnlineConfig,
    *,
    locator_result: RetrievalResult | None = None,
) -> list[VideoHypothesis]:
    by_moment_video: list[dict[str, list[FrameEvidence]]] = []
    for result in moment_results:
        grouped: dict[str, list[FrameEvidence]] = defaultdict(list)
        for frame in result.evidence:
            grouped[frame.video_id].append(frame)
        by_moment_video.append({video: _distinct_shot(frames) for video, frames in grouped.items()})
    locator_by_video: dict[str, float] = {}
    if locator_result is not None:
        for frame in locator_result.evidence:
            locator_by_video[frame.video_id] = max(
                locator_by_video.get(frame.video_id, 0.0), frame.final_score
            )
    all_videos = set(locator_by_video) | set().union(
        *(grouped.keys() for grouped in by_moment_video)
    )
    moment_top_videos = [
        {
            video_id
            for video_id, _frames in sorted(
                grouped.items(),
                key=lambda item: item[1][0].final_score if item[1] else 0.0,
                reverse=True,
            )[: config.video_part_top_k]
        }
        for grouped in by_moment_video
    ]
    hypotheses: list[VideoHypothesis] = []
    for video_id in all_videos:
        strengths = [frames[0].final_score if (frames := grouped.get(video_id, [])) else 0.0 for grouped in by_moment_video]
        hits = [video_id in top_videos for top_videos in moment_top_videos]
        coverage = sum(hits) / len(moment_results)
        weakest = min(strengths) if strengths else 0.0
        average = mean(strengths) if strengths else 0.0
        consensus_values = [frames[0].model_consensus for grouped in by_moment_video if (frames := grouped.get(video_id, []))]
        consensus = mean(consensus_values) if consensus_values else 0.0
        score = (
            config.trake_locator_weight * locator_by_video.get(video_id, 0.0)
            + config.trake_event_coverage_weight * coverage
            + config.trake_weakest_weight * weakest
            + config.trake_mean_weight * average
            + config.trake_consensus_weight * consensus
        )
        merged = _distinct_shot(frame for grouped in by_moment_video for frame in grouped.get(video_id, [])[:3])
        hypotheses.append(
            VideoHypothesis(
                video_id=video_id,
                video_score=score,
                coverage=coverage,
                model_consensus=consensus,
                best_frames=merged[:100],
                matched_scenes=[moment for moment, hit in zip(moments, hits) if hit],
                missing_scenes=[moment for moment, hit in zip(moments, hits) if not hit],
            )
        )
    hypotheses.sort(key=lambda item: item.video_score, reverse=True)
    hypotheses = apply_clip_tie_break(hypotheses, margin=config.clip_tie_margin)
    return hypotheses[: config.trake_video_top_k]
