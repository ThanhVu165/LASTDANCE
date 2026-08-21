"""Cutoff-aware portfolio ranking for the official AIC preliminary metric.

The organizer does not average relevance over all returned rows.  It computes the
best R-Score inside ranks 1, 5, 20, 50 and 100, then averages those five values.
This selector therefore increases the per-video quota at those exact boundaries:
the first ranks exploit the strongest hypothesis, while later ranks progressively
cover more alternative videos.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from app.config import RANKING_CUTOFFS


KIS_VIDEO_CAPS = {1: 1, 5: 2, 20: 3, 50: 4, 100: 5}
TRAKE_VIDEO_CAPS = {1: 1, 5: 2, 20: 4, 50: 5, 100: 5}


def _stage_cap(target: int, video_caps: Mapping[int, int]) -> int:
    for cutoff in sorted(video_caps):
        if target <= cutoff:
            return video_caps[cutoff]
    return video_caps[max(video_caps)]


def cutoff_aware_rank(
    candidates: Sequence[dict],
    *,
    top_k: int,
    video_caps: Mapping[int, int],
) -> list[dict]:
    """Select rows in metric-aligned stages while preserving score order.

    If a stage cannot be filled under its diversity quota, the quota is relaxed
    for the remaining slots.  This guarantees the requested result count whenever
    enough candidates exist and never discards recall merely to satisfy diversity.
    """
    if top_k <= 0:
        return []

    pool = sorted(candidates, key=lambda row: float(row["score"]), reverse=True)
    requested = min(top_k, len(pool))
    targets = [cutoff for cutoff in RANKING_CUTOFFS if cutoff < requested]
    targets.append(requested)

    selected: list[dict] = []
    selected_ids: set[int] = set()
    per_video: dict[str, int] = defaultdict(int)

    for target in targets:
        cap = _stage_cap(target, video_caps)
        for index, row in enumerate(pool):
            if len(selected) >= target:
                break
            if index in selected_ids or per_video[row["video_id"]] >= cap:
                continue
            selected.append(row)
            selected_ids.add(index)
            per_video[row["video_id"]] += 1

        # Sparse candidate sets should still fill every official submission slot.
        if len(selected) < target:
            for index, row in enumerate(pool):
                if len(selected) >= target:
                    break
                if index in selected_ids:
                    continue
                selected.append(row)
                selected_ids.add(index)
                per_video[row["video_id"]] += 1

    return selected
