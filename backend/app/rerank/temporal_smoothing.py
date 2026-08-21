"""Temporal smoothing — boost a keyframe's score if its immediate neighbours in the
same video (by local_idx, which is time-ordered) are also strongly scored.

Rationale: if several consecutive keyframes in one video score highly, that's a
strong signal the whole segment matches the query — even if the single highest-
scoring frame isn't the ideal one to submit. Smoothing spreads confidence across the
neighbourhood, which in practice nudges the best *representative* frame of the
correct segment towards the top instead of an isolated noisy peak elsewhere.
"""
from collections import defaultdict

NEIGHBOR_BOOST = 0.03


def apply_temporal_smoothing(candidates: list[dict]) -> None:
    """Mutates `score` in place."""
    by_video: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        by_video[c["video_id"]].append(c)

    for rows in by_video.values():
        rows.sort(key=lambda x: x["local_idx"])
        by_local_idx = {r["local_idx"]: r for r in rows}
        base_scores = {r["local_idx"]: r["score"] for r in rows}
        for r in rows:
            neighbours = [
                base_scores[r["local_idx"] - 1] if r["local_idx"] - 1 in by_local_idx else None,
                base_scores[r["local_idx"] + 1] if r["local_idx"] + 1 in by_local_idx else None,
            ]
            present = [n for n in neighbours if n is not None]
            if present:
                r["score"] += NEIGHBOR_BOOST * (sum(present) / len(present))
