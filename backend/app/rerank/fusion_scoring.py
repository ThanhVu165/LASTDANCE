"""Fusion scoring — combine multiple independent relevance signals into one score.

This is the core of the re-rank engine described in the project plan: instead of
trusting raw CLIP similarity alone, we fuse it with object-detection matches and
OCR matches, because the Final Score contest metric (R@1..R@100) rewards getting the
correct answer as close to rank #1 as possible, not merely somewhere in 100 rows.

Weights are currently fixed heuristics (documented below) but are intentionally
kept in one place so they can later be replaced by weights learned from a labeled
query/answer sample (see plan backlog: "learned fusion weights").
"""
from dataclasses import dataclass

from app.services.object_filter import object_match_score
from app.services.ocr_search import ocr_match_score


@dataclass(frozen=True)
class FusionWeights:
    clip: float = 0.80
    object_match: float = 0.13
    ocr_match: float = 0.07


DEFAULT_WEIGHTS = FusionWeights()


def fuse_score(
    video_id: str,
    local_idx: int,
    clip_score: float,
    object_keywords: list[str],
    ocr_keywords: list[str],
    weights: FusionWeights = DEFAULT_WEIGHTS,
) -> float:
    obj_s = object_match_score(video_id, local_idx, object_keywords)
    ocr_s = ocr_match_score(video_id, local_idx, ocr_keywords)
    return weights.clip * clip_score + weights.object_match * obj_s + weights.ocr_match * ocr_s


def fuse_candidates(candidates: list[dict], object_keywords: list[str], ocr_keywords: list[str]) -> list[dict]:
    """candidates: list of dicts with at least video_id, local_idx, frame_id, score
    (score == raw CLIP similarity here). Returns new list with `score` replaced by
    the fused score, plus a `clip_score` field kept for debugging/analysis."""
    out = []
    for c in candidates:
        fused = fuse_score(c["video_id"], c["local_idx"], c["score"], object_keywords, ocr_keywords)
        out.append({**c, "clip_score": c["score"], "score": fused})
    return out
