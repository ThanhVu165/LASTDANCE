"""Model-planned Textual KIS retrieval, verification, repair and ranking."""
from app.config import (
    KIS_LONG_QUERY_CANDIDATES,
    MAX_SUBMISSION_ROWS,
    MODEL_REPAIR_ENABLED,
    MODEL_REPAIR_MAX_ROUNDS,
    MODEL_RERANK_MIN_VERIFIED_VIDEOS,
    TOP_K_CANDIDATES,
)
from app.rerank.contest_ranking import KIS_VIDEO_CAPS, cutoff_aware_rank
from app.rerank.exact_frame_refinement import refine_top_kis_frames
from app.rerank.fusion_scoring import fuse_candidates
from app.rerank.model_reranker import (
    release_model_reranker,
    rerank_kis_with_generative_model,
    rerank_kis_with_model,
)
from app.rerank.storyboard_alignment import apply_storyboard_alignment
from app.rerank.temporal_smoothing import apply_temporal_smoothing
from app.rerank.visual_reranker import rerank_kis_candidates
from app.services.clip_search import search_text_batch
from app.services.query_planner import QueryPlan, plan_visual_query
from app.services.query_processing import parse_semantic_query
from app.services.query_translation import (
    translate_visual_queries,
    translate_visual_scenes,
)
from app.services.side_search import (
    search_siglip_batch,
    search_video_windows_batch,
)


def _merge_clip_hits(
    expansions: list[str],
    top_k: int,
    *,
    consensus_weight: float | None = None,
    consensus_expansion_count: int | None = None,
    prompt_scene_indices: list[int | None] | None = None,
    hit_batches: list[list[dict]] | None = None,
) -> list[dict]:
    merged: dict[tuple[str, int], dict] = {}
    per_scene_video_scores: dict[int, dict[str, float]] = {}
    if prompt_scene_indices is not None and len(prompt_scene_indices) != len(expansions):
        raise ValueError("prompt_scene_indices must match the expansion count")
    batches = (
        hit_batches
        if hit_batches is not None
        else search_text_batch(expansions, top_k=top_k)
    )
    if len(batches) != len(expansions):
        raise ValueError("hit_batches must match the expansion count")
    for expansion_index, hits in enumerate(batches):
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


def _merge_candidate_pools(*pools: list[dict]) -> list[dict]:
    """Union independently retrieved pools without losing per-scene evidence."""
    merged: dict[tuple[str, int], dict] = {}
    for pool in pools:
        for row in pool:
            key = (row["video_id"], int(row["local_idx"]))
            previous = merged.get(key)
            combined_scene_scores = dict((previous or {}).get("scene_scores") or {})
            for scene_index, score in (row.get("scene_scores") or {}).items():
                combined_scene_scores[int(scene_index)] = max(
                    float(combined_scene_scores.get(int(scene_index), 0.0)),
                    float(score),
                )
            if previous is None or float(row["score"]) > float(previous["score"]):
                merged[key] = {**row, "scene_scores": combined_scene_scores}
            else:
                previous["scene_scores"] = combined_scene_scores
    return list(merged.values())


def _score_candidates(clip_candidates: list[dict], semantic, plan: QueryPlan) -> list[dict]:
    visible_text = [
        text
        for scene in plan.scenes
        for text in scene.visible_text
    ]
    # Once the structured model plan succeeds, regex-derived object/OCR terms no
    # longer steer ranking. They remain solely as an outage-safe fallback.
    object_keywords = (
        [] if plan.source == "model" else list(semantic.object_keywords)
    )
    ocr_keywords = (
        list(dict.fromkeys(visible_text))
        if plan.source == "model"
        else list(dict.fromkeys([*semantic.ocr_keywords, *visible_text]))
    )
    candidates = fuse_candidates(
        clip_candidates,
        object_keywords,
        ocr_keywords,
    )
    apply_temporal_smoothing(candidates)
    apply_storyboard_alignment(
        candidates,
        scene_count=len(plan.scenes),
        temporal_edges=plan.temporal_edges,
    )
    return candidates


def _planned_prompts(plan: QueryPlan, semantic) -> tuple[list[str], list[int | None]]:
    if plan.source == "model":
        prompts = list(plan.retrieval_prompts)
        return [prompt for prompt, _ in prompts], [scene for _, scene in prompts]

    # A failed/disabled planner keeps the proven bilingual recall path intact.
    expansions = list(semantic.scenes)
    scene_indices: list[int | None] = list(range(len(semantic.scenes)))
    if len(semantic.scenes) >= 2:
        for scene_index, translation in enumerate(
            translate_visual_scenes(tuple(semantic.scenes))
        ):
            if translation and translation not in expansions:
                expansions.append(translation)
                scene_indices.append(scene_index)
    for expansion in semantic.expansions:
        if expansion not in expansions:
            expansions.append(expansion)
            scene_indices.append(None)
    for translation in translate_visual_queries(semantic.retrieval_text):
        if translation not in expansions:
            expansions.append(translation)
            scene_indices.append(None)
    return expansions, scene_indices


def run_kis_query(
    text: str,
    top_k: int = MAX_SUBMISSION_ROWS,
    *,
    refine_exact: bool = True,
) -> list[dict]:
    sem = parse_semantic_query(text)
    plan = plan_visual_query(sem.original_text)
    is_multiscene = len(plan.scenes) >= 2
    expansions, prompt_scene_indices = _planned_prompts(plan, sem)

    clip_candidates = _merge_clip_hits(
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
        consensus_expansion_count=len(plan.scenes),
        prompt_scene_indices=prompt_scene_indices,
    )
    # Side indexes are optional and atomically published. Missing/incomplete
    # indexes return empty batches, so organizer CLIP remains a complete fallback.
    side_top_k = (
        max(TOP_K_CANDIDATES, KIS_LONG_QUERY_CANDIDATES)
        if is_multiscene
        else TOP_K_CANDIDATES
    )
    for side_batches in (
        search_siglip_batch(expansions, top_k=side_top_k),
        search_video_windows_batch(expansions, top_k=side_top_k),
    ):
        if any(side_batches):
            clip_candidates = _merge_candidate_pools(
                clip_candidates,
                _merge_clip_hits(
                    expansions,
                    top_k=side_top_k,
                    consensus_weight=0.15 if is_multiscene else 0.0,
                    consensus_expansion_count=len(plan.scenes),
                    prompt_scene_indices=prompt_scene_indices,
                    hit_batches=side_batches,
                ),
            )
    candidates = _score_candidates(clip_candidates, sem, plan)
    candidates, verification = rerank_kis_with_model(
        plan.verification_query,
        candidates,
    )
    if not verification.available:
        candidates, verification = rerank_kis_with_generative_model(
            plan.verification_query,
            candidates,
        )

    # A low-confidence first pass triggers targeted recall rather than repeatedly
    # scoring the same incomplete pool. The dedicated reranker remains resident
    # across repair rounds, then is explicitly released before exact-frame VQA.
    repair_round = 0
    while (
        verification.available
        and MODEL_REPAIR_ENABLED
        and repair_round < max(0, MODEL_REPAIR_MAX_ROUNDS)
        and verification.high_confidence_videos < MODEL_RERANK_MIN_VERIFIED_VIDEOS
        and plan.repair_queries
    ):
        repair_round += 1
        repair_hits = _merge_clip_hits(
            list(plan.repair_queries),
            top_k=max(TOP_K_CANDIDATES, KIS_LONG_QUERY_CANDIDATES),
            consensus_weight=0.0,
            prompt_scene_indices=[None] * len(plan.repair_queries),
        )
        clip_candidates = _merge_candidate_pools(clip_candidates, repair_hits)
        candidates = _score_candidates(clip_candidates, sem, plan)
        candidates, verification = rerank_kis_with_model(
            plan.verification_query,
            candidates,
        )
        if not verification.available:
            candidates, verification = rerank_kis_with_generative_model(
                plan.verification_query,
                candidates,
            )

    if not verification.available:
        # Production-safe fallback while a model is unavailable or being built.
        candidates = rerank_kis_candidates(sem.original_text, candidates)
    release_model_reranker()

    verified_candidates = [
        row
        for row in candidates
        if row.get("model_verified")
        and row.get("model_relevance_score") is not None
    ]
    # When the verifier has scored enough rows, do not let an unchecked retrieval
    # hit re-enter the official Top 100 merely because its raw cosine scale is
    # larger. Sparse/error cases still retain the production-safe full pool.
    ranking_pool = (
        verified_candidates
        if len(verified_candidates) >= min(top_k, MAX_SUBMISSION_ROWS)
        else candidates
    )
    final = cutoff_aware_rank(
        ranking_pool,
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
            "model_relevance_score": (
                float(c["model_relevance_score"])
                if c.get("model_relevance_score") is not None
                else None
            ),
            "model_verified": bool(c.get("model_verified", False)),
            "is_source_frame": "keyframe_frame_id" in c,
        }
        for c in final
    ]
