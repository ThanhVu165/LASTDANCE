"""Official AIC2026 preliminary-round R-Score and Final Score helpers."""
from __future__ import annotations

from collections.abc import Callable, Sequence

from app.config import RANKING_CUTOFFS


def frame_in_interval(frame_id: int, interval: Sequence[int]) -> bool:
    if len(interval) != 2:
        raise ValueError("A frame interval must contain [start, end].")
    start, end = int(interval[0]), int(interval[1])
    return start <= int(frame_id) <= end


def kis_r_score(row: dict, ground_truth: dict) -> float:
    return float(
        row["video_id"] == ground_truth["video_id"]
        and frame_in_interval(row["frame_id"], ground_truth["frame_interval"])
    )


def qa_r_score(
    row: dict,
    ground_truth: dict,
    *,
    answer_matcher: Callable[[str, Sequence[str]], bool],
) -> float:
    return float(
        row["video_id"] == ground_truth["video_id"]
        and frame_in_interval(row["frame_id"], ground_truth["frame_interval"])
        and answer_matcher(row["answer"], ground_truth["answers"])
    )


def trake_r_score(row: dict, ground_truth: dict) -> float:
    intervals = ground_truth["frame_intervals"]
    frame_ids = row["frame_ids"]
    if row["video_id"] != ground_truth["video_id"] or len(frame_ids) != len(intervals):
        return 0.0
    matches = sum(
        frame_in_interval(frame_id, interval)
        for frame_id, interval in zip(frame_ids, intervals)
    )
    return matches / len(intervals) if intervals else 0.0


def final_score(r_scores: Sequence[float]) -> dict:
    """Return R@1/5/20/50/100 and their official arithmetic mean."""
    values = [float(score) for score in r_scores]
    r_at = {
        cutoff: max(values[:cutoff], default=0.0)
        for cutoff in RANKING_CUTOFFS
    }
    return {
        "r_at": r_at,
        "final_score": sum(r_at.values()) / len(RANKING_CUTOFFS),
    }
