"""TRAKE retrieval and K-best monotonic alignment from one complete query."""
from __future__ import annotations

from dataclasses import dataclass

from app.config import (
    MAX_SUBMISSION_ROWS,
    TRAKE_GAP_PENALTY_WEIGHT,
    TRAKE_MAX_GAP_SECONDS,
)
from app.rerank.contest_ranking import TRAKE_VIDEO_CAPS, cutoff_aware_rank
from app.rerank.exact_frame_refinement import refine_top_trake_frames
from app.rerank.fusion_scoring import fuse_candidates
from app.rerank.visual_reranker import rerank_trake_hypotheses
from app.services.clip_search import search_text_batch
from app.services.query_processing import parse_semantic_query, split_trake_moments
from app.services.query_translation import with_english_visual_expansion
from app.services.temporal_context import record_for_local

GLOBAL_HITS_PER_PROMPT = 1200
MAX_VIDEO_HYPOTHESES = 60
PER_MOMENT_CANDIDATES = 20
BEAM_WIDTH = 60
ALIGNMENTS_PER_VIDEO = 12


@dataclass(frozen=True)
class MomentCandidate:
    local_idx: int
    frame_id: int
    score: float
    pts_time: float | None = None


@dataclass(frozen=True)
class Alignment:
    candidates: tuple[MomentCandidate, ...]
    score: float


def _moment_hits(moment_text: str) -> list[dict]:
    semantic = parse_semantic_query(moment_text)
    expansions = with_english_visual_expansion(
        semantic.retrieval_text, semantic.expansions
    )
    merged: dict[tuple[str, int], dict] = {}
    for hits in search_text_batch(
        expansions,
        top_k=GLOBAL_HITS_PER_PROMPT,
    ):
        if hits:
            high = float(hits[0]["score"])
            low = float(hits[-1]["score"])
        else:
            high = low = 0.0
        for rank, hit in enumerate(hits, 1):
            raw_score = float(hit["score"])
            minmax_score = (
                (raw_score - low) / (high - low)
                if high - low > 1e-8
                else 1.0
            )
            calibrated_score = 0.70 * minmax_score + 0.30 * (61.0 / (60.0 + rank))
            key = (hit["video_id"], hit["local_idx"])
            previous = merged.get(key)
            if previous is None or calibrated_score > previous["score"]:
                merged[key] = {
                    **hit,
                    "raw_clip_score": raw_score,
                    "score": calibrated_score,
                }
    return fuse_candidates(
        list(merged.values()),
        semantic.object_keywords,
        semantic.ocr_keywords,
    )


def _rank_videos(all_moment_hits: list[list[dict]]) -> list[tuple[str, float]]:
    number_of_moments = len(all_moment_hits)
    per_video: dict[str, list[float | None]] = {}
    for moment_index, hits in enumerate(all_moment_hits):
        for hit in hits:
            scores = per_video.setdefault(hit["video_id"], [None] * number_of_moments)
            previous = scores[moment_index]
            if previous is None or hit["score"] > previous:
                scores[moment_index] = float(hit["score"])

    ranked: list[tuple[str, float, int]] = []
    for video_id, scores in per_video.items():
        present = [score for score in scores if score is not None]
        coverage = len(present)
        if not present:
            continue
        mean_score = sum(present) / coverage
        coverage_ratio = coverage / number_of_moments
        # Coverage dominates: TRAKE gives zero for the wrong video and every moment
        # belongs to one video. Mean semantic score orders videos with equal coverage.
        video_score = 0.65 * coverage_ratio + 0.35 * mean_score
        ranked.append((video_id, video_score, coverage))

    ranked.sort(key=lambda row: (row[2], row[1]), reverse=True)
    return [(video_id, score) for video_id, score, _ in ranked[:MAX_VIDEO_HYPOTHESES]]


def _candidates_in_video(
    hits: list[dict], video_id: str
) -> list[MomentCandidate]:
    candidates: list[MomentCandidate] = []
    for hit in hits:
        if hit["video_id"] != video_id:
            continue
        local_idx = int(hit["local_idx"])
        record = record_for_local(video_id, local_idx)
        candidates.append(
            MomentCandidate(
                local_idx=local_idx,
                frame_id=int(hit["frame_id"]),
                score=float(hit["score"]),
                pts_time=record.pts_time if record is not None else None,
            )
        )
    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    return candidates[:PER_MOMENT_CANDIDATES]


def _k_best_monotonic_alignments(
    candidates_by_moment: list[list[MomentCandidate]],
    *,
    max_gap_seconds: float | None = None,
    gap_penalty_weight: float = 0.0,
) -> list[Alignment]:
    if not candidates_by_moment or any(not candidates for candidates in candidates_by_moment):
        return []

    beams: list[tuple[tuple[MomentCandidate, ...], float]] = [((), 0.0)]
    for moment_candidates in candidates_by_moment:
        expanded: list[tuple[tuple[MomentCandidate, ...], float]] = []
        for chosen, total_score in beams:
            last_local_idx = chosen[-1].local_idx if chosen else -1
            for candidate in moment_candidates:
                if candidate.local_idx <= last_local_idx:
                    continue
                if chosen and max_gap_seconds is not None and max_gap_seconds > 0:
                    previous_time = chosen[-1].pts_time
                    if (
                        previous_time is not None
                        and candidate.pts_time is not None
                        and candidate.pts_time - previous_time > max_gap_seconds
                    ):
                        continue
                expanded.append((chosen + (candidate,), total_score + candidate.score))
        if not expanded:
            return []

        deduplicated: dict[tuple[int, ...], tuple[tuple[MomentCandidate, ...], float]] = {}
        for chosen, total_score in expanded:
            key = tuple(candidate.local_idx for candidate in chosen)
            previous = deduplicated.get(key)
            if previous is None or total_score > previous[1]:
                deduplicated[key] = (chosen, total_score)
        beams = sorted(
            deduplicated.values(), key=lambda item: item[1], reverse=True
        )[:BEAM_WIDTH]

    number_of_moments = len(candidates_by_moment)
    alignments: list[Alignment] = []
    for chosen, total_score in beams[:ALIGNMENTS_PER_VIDEO]:
        score = total_score / number_of_moments
        timed = [candidate.pts_time for candidate in chosen if candidate.pts_time is not None]
        if max_gap_seconds and len(timed) >= 2:
            span_ratio = min(max((timed[-1] - timed[0]) / max_gap_seconds, 0.0), 1.0)
            score -= max(gap_penalty_weight, 0.0) * span_ratio
        alignments.append(Alignment(candidates=chosen, score=score))
    return sorted(alignments, key=lambda alignment: alignment.score, reverse=True)


def run_trake_query(
    text: str,
    top_k: int = MAX_SUBMISSION_ROWS,
) -> tuple[list[str], list[dict]]:
    moments = split_trake_moments(text)
    if len(moments) < 2:
        raise ValueError(
            "TRAKE query must describe at least two ordered moments. Use numbering "
            "such as (1), (2), ... when the sequence is ambiguous."
        )

    all_hits = [_moment_hits(moment) for moment in moments]
    ranked_videos = _rank_videos(all_hits)
    hypotheses: list[dict] = []
    for video_id, video_score in ranked_videos:
        candidates_by_moment = [
            _candidates_in_video(hits, video_id) for hits in all_hits
        ]
        for alignment in _k_best_monotonic_alignments(
            candidates_by_moment,
            max_gap_seconds=TRAKE_MAX_GAP_SECONDS,
            gap_penalty_weight=TRAKE_GAP_PENALTY_WEIGHT,
        ):
            hypotheses.append(
                {
                    "video_id": video_id,
                    "frame_ids": [candidate.frame_id for candidate in alignment.candidates],
                    "local_idxs": [candidate.local_idx for candidate in alignment.candidates],
                    "score": float(0.85 * alignment.score + 0.15 * video_score),
                }
            )

    deduplicated: dict[tuple[str, tuple[int, ...]], dict] = {}
    for hypothesis in hypotheses:
        key = (hypothesis["video_id"], tuple(hypothesis["frame_ids"]))
        previous = deduplicated.get(key)
        if previous is None or hypothesis["score"] > previous["score"]:
            deduplicated[key] = hypothesis
    ranked = sorted(deduplicated.values(), key=lambda row: row["score"], reverse=True)
    ranked = rerank_trake_hypotheses(text, ranked)

    requested = min(top_k, MAX_SUBMISSION_ROWS)
    selected = cutoff_aware_rank(
        ranked,
        top_k=requested,
        video_caps=TRAKE_VIDEO_CAPS,
    )
    selected = refine_top_trake_frames(moments, selected)
    selected = [
        {
            **row,
            "is_source_frames": "keyframe_frame_ids" in row,
        }
        for row in selected
    ]
    return moments, selected[:requested]
