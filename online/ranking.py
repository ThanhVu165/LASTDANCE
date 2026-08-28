"""Aggregate independent frame evidence into ranked video hypotheses."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Iterable

from shared.schemas.online import FrameEvidence, UnifiedQueryPlan, VideoHypothesis

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

    scene_labels = list(plan.scenes or [plan.caption_en])
    global_queries = list(plan.retrieval_queries)
    hypotheses: list[VideoHypothesis] = []
    all_videos = set(by_video) | set().union(
        *(scores.keys() for scores in query_video_scores.values())
    )
    for video_id in all_videos:
        scene_scores = [
            query_video_scores.get(scene, {}).get(video_id, 0.0) for scene in scene_labels
        ]
        coverage_hits = [
            video_id in query_top_videos.get(scene, set()) for scene in scene_labels
        ]
        coverage = sum(coverage_hits) / len(coverage_hits) if coverage_hits else 0.0
        mean_scene = mean(scene_scores) if scene_scores else 0.0
        weakest_scene = min(scene_scores) if scene_scores else 0.0
        global_values = [
            query_video_scores.get(query, {}).get(video_id, 0.0) for query in global_queries
        ]
        global_score = max(global_values) if global_values else mean_scene
        distinct = _distinct_shot(by_video.get(video_id, []))
        consensus = mean([frame.model_consensus for frame in distinct[:3]]) if distinct else 0.0
        base_score = (
            config.video_coverage_weight * coverage
            + config.video_evidence_weight * mean_scene
            + config.video_weakest_weight * weakest_scene
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
                    label for label, hit in zip(scene_labels, coverage_hits) if hit
                ],
                missing_scenes=[
                    label for label, hit in zip(scene_labels, coverage_hits) if not hit
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
) -> list[VideoHypothesis]:
    by_moment_video: list[dict[str, list[FrameEvidence]]] = []
    for result in moment_results:
        grouped: dict[str, list[FrameEvidence]] = defaultdict(list)
        for frame in result.evidence:
            grouped[frame.video_id].append(frame)
        by_moment_video.append({video: _distinct_shot(frames) for video, frames in grouped.items()})
    all_videos = set().union(*(grouped.keys() for grouped in by_moment_video))
    hypotheses: list[VideoHypothesis] = []
    for video_id in all_videos:
        strengths = [frames[0].final_score if (frames := grouped.get(video_id, [])) else 0.0 for grouped in by_moment_video]
        hits = [strength > 0.0 for strength in strengths]
        coverage = sum(hits) / len(moment_results)
        weakest = min(strengths) if strengths else 0.0
        average = mean(strengths) if strengths else 0.0
        consensus_values = [frames[0].model_consensus for grouped in by_moment_video if (frames := grouped.get(video_id, []))]
        consensus = mean(consensus_values) if consensus_values else 0.0
        score = 0.50 * coverage + 0.20 * weakest + 0.20 * average + 0.10 * consensus
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
