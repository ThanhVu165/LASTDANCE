"""Textual KIS pipeline.

Flow: candidate generation (CLIP over query expansions) -> fusion scoring (CLIP +
object + OCR) -> temporal smoothing -> MMR diversity/dedup -> top-100 output.
"""
from app.config import (
    KIS_LONG_QUERY_CANDIDATES,
    MAX_SUBMISSION_ROWS,
    TOP_K_CANDIDATES,
)
from app.rerank.contest_ranking import KIS_VIDEO_CAPS, cutoff_aware_rank
from app.rerank.exact_frame_refinement import refine_top_kis_frames
from app.rerank.fusion_scoring import fuse_candidates
from app.rerank.storyboard_alignment import apply_storyboard_alignment
from app.rerank.temporal_smoothing import apply_temporal_smoothing
from app.rerank.visual_reranker import rerank_kis_candidates
from app.services.clip_search import search_text_batch
from app.services.query_processing import parse_semantic_query
from app.services.query_translation import (
    translate_visual_queries,
    translate_visual_scenes,
)


def _merge_clip_hits(
    expansions: list[str],
    top_k: int,
    *,
    consensus_weight: float | None = None,
    consensus_expansion_count: int | None = None,
    prompt_scene_indices: list[int | None] | None = None,
) -> list[dict]:
    merged: dict[tuple[str, int], dict] = {}
    per_scene_video_scores: dict[int, dict[str, float]] = {}
    if prompt_scene_indices is not None and len(prompt_scene_indices) != len(expansions):
        raise ValueError("prompt_scene_indices must match the expansion count")
    for expansion_index, hits in enumerate(search_text_batch(expansions, top_k=top_k)):
        scene_index = (
            prompt_scene_indices[expansion_index]
            if prompt_scene_indices is not None
            else expansion_index
            if (
                consensus_expansion_count is None
                or expansion_index < consensus_expansion_count
            )
            else None
        )
        video_scores: dict[str, float] = {}
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
            # Normalized RRF stabilizes near-ties and, unlike raw cosine values,
            # is comparable between the original and multilingual CLIP towers.
            rrf_score = 61.0 / (60.0 + rank)
            calibrated_score = 0.70 * minmax_score + 0.30 * rrf_score
            key = (hit["video_id"], hit["local_idx"])
            prev = merged.get(key)
            if prev is None:
                merged[key] = {
                    **hit,
                    "raw_clip_score": raw_score,
                    "calibrated_clip_score": calibrated_score,
                    "scene_scores": {},
                }
            elif calibrated_score > prev["calibrated_clip_score"]:
                scene_scores = prev["scene_scores"]
                merged[key] = {
                    **hit,
                    "raw_clip_score": raw_score,
                    "calibrated_clip_score": calibrated_score,
                    "scene_scores": scene_scores,
                }
            if scene_index is not None:
                scene_scores = merged[key]["scene_scores"]
                scene_scores[scene_index] = max(
                    float(scene_scores.get(scene_index, 0.0)),
                    calibrated_score,
                )
            video_id = hit["video_id"]
            video_scores[video_id] = max(
                video_scores.get(video_id, float("-inf")), calibrated_score
            )
        # English renderings are alternative encoders of the same source query,
        # not additional independent scene evidence.  They may contribute frame
        # candidates, but only original/base clauses vote in long-query video
        # consensus or the translated branch would be counted three extra times.
        if scene_index is not None:
            # Original Vietnamese and its English rendering are two encoders of
            # one evidence unit. Merge them with max-pooling so a translation can
            # improve recall without casting an extra semantic vote.
            aggregate = per_scene_video_scores.setdefault(scene_index, {})
            for video_id, score in video_scores.items():
                aggregate[video_id] = max(aggregate.get(video_id, 0.0), score)

    prompt_count = max(len(per_scene_video_scores), 1)
    if consensus_weight is None:
        consensus_weight = 0.45 if len(expansions) > 3 else 0.15
    for row in merged.values():
        video_id = row["video_id"]
        prompt_scores = [
            scores.get(video_id, 0.0)
            for _, scores in sorted(per_scene_video_scores.items())
        ]
        video_consensus = sum(prompt_scores) / prompt_count
        coverage = sum(score > 0.0 for score in prompt_scores) / prompt_count
        frame_score = float(row["calibrated_clip_score"])
        row["frame_clip_score"] = float(row["raw_clip_score"])
        row["query_coverage"] = coverage
        row["score"] = (
            (1.0 - consensus_weight) * frame_score
            + consensus_weight * video_consensus
        )
    return list(merged.values())


def run_kis_query(
    text: str,
    top_k: int = MAX_SUBMISSION_ROWS,
    *,
    refine_exact: bool = True,
) -> list[dict]:
    sem = parse_semantic_query(text)
    is_multiscene = len(sem.scenes) >= 2

    # Scene prompts are the evidence-bearing branches. The full description,
    # adjacent pairs and English paraphrases may recover candidates, but they do
    # not cast extra votes for scene coverage.
    expansions = list(sem.scenes)
    prompt_scene_indices: list[int | None] = list(range(len(sem.scenes)))
    if is_multiscene:
        for scene_index, translation in enumerate(
            translate_visual_scenes(tuple(sem.scenes))
        ):
            if translation and translation not in expansions:
                expansions.append(translation)
                prompt_scene_indices.append(scene_index)
    for expansion in sem.expansions:
        if expansion not in expansions:
            expansions.append(expansion)
            prompt_scene_indices.append(None)
    for translation in translate_visual_queries(sem.retrieval_text):
        if translation not in expansions:
            expansions.append(translation)
            prompt_scene_indices.append(None)

    candidates = _merge_clip_hits(
        expansions,
        top_k=(
            max(TOP_K_CANDIDATES, KIS_LONG_QUERY_CANDIDATES)
            if is_multiscene
            else TOP_K_CANDIDATES
        ),
        # Storyboard alignment below owns the video-level aggregation. Keep only
        # a small consensus prior here so one frame cannot dominate candidate
        # generation before all scenes have been inspected.
        consensus_weight=0.15 if is_multiscene else 0.0,
        consensus_expansion_count=len(sem.scenes),
        prompt_scene_indices=prompt_scene_indices,
    )
    candidates = fuse_candidates(candidates, sem.object_keywords, sem.ocr_keywords)
    apply_temporal_smoothing(candidates)
    apply_storyboard_alignment(
        candidates,
        scene_count=len(sem.scenes),
        temporal_edges=sem.temporal_edges,
    )
    candidates = rerank_kis_candidates(sem.original_text, candidates)

    final = cutoff_aware_rank(
        candidates,
        top_k=min(top_k, MAX_SUBMISSION_ROWS),
        video_caps=KIS_VIDEO_CAPS,
    )
    if refine_exact:
        final = refine_top_kis_frames(sem.original_text, final)
    return [
        {
            "video_id": c["video_id"],
            "frame_id": c["frame_id"],
            "local_idx": c["local_idx"],
            "score": float(c["score"]),
            "is_source_frame": "keyframe_frame_id" in c,
        }
        for c in final
    ]
