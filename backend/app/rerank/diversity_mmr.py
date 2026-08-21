"""Diversity-aware re-ranking: cap results per video and apply a simple Maximal
Marginal Relevance (MMR) style selection so the limited 100-row submission budget
isn't wasted on many near-duplicate keyframes from the same video/segment.
"""
from collections import defaultdict

TIME_CLOSE_THRESHOLD = 15  # local_idx distance considered "near duplicate" within a video


def _similarity_penalty(candidate: dict, selected: list[dict]) -> float:
    """Cheap proxy for visual/content similarity: same video + close in time =>
    likely near-duplicate content, so penalize to encourage diversity."""
    penalty = 0.0
    for s in selected:
        if s["video_id"] != candidate["video_id"]:
            continue
        dist = abs(s["local_idx"] - candidate["local_idx"])
        if dist == 0:
            penalty = max(penalty, 1.0)
        elif dist < TIME_CLOSE_THRESHOLD:
            penalty = max(penalty, 1.0 - dist / TIME_CLOSE_THRESHOLD)
    return penalty


def mmr_rerank(candidates: list[dict], top_k: int, lambda_param: float = 0.85, max_per_video: int = 5) -> list[dict]:
    """Greedy MMR selection: at each step pick the candidate maximizing
    lambda * relevance - (1 - lambda) * similarity_to_already_selected, while
    respecting a per-video cap for diversity across the whole result list."""
    pool = sorted(candidates, key=lambda x: x["score"], reverse=True)
    selected: list[dict] = []
    per_video_count: dict[str, int] = defaultdict(int)

    while pool and len(selected) < top_k:
        best_idx, best_val = -1, float("-inf")
        for i, cand in enumerate(pool):
            if per_video_count[cand["video_id"]] >= max_per_video:
                continue
            penalty = _similarity_penalty(cand, selected)
            val = lambda_param * cand["score"] - (1 - lambda_param) * penalty
            if val > best_val:
                best_val, best_idx = val, i
        if best_idx == -1:
            break
        chosen = pool.pop(best_idx)
        selected.append(chosen)
        per_video_count[chosen["video_id"]] += 1

    return selected
