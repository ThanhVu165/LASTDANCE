"""Video-level scene coverage and temporal alignment for long KIS narratives.

CLIP retrieves isolated keyframes, but final-round KIS descriptions commonly
describe several scenes from one video.  This module performs cheap late
interaction over those per-scene hits: it measures which clauses each video
covers, finds a high-scoring chronological alignment, and propagates that story
score back to the candidate frames before the expensive VLM reranker.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from app.config import (
    KIS_STORYBOARD_BEAM_SIZE,
    KIS_STORYBOARD_ENABLED,
    KIS_STORYBOARD_HITS_PER_SCENE,
    KIS_STORYBOARD_MIN_SCENE_SCORE,
    KIS_STORYBOARD_ORDER_SLACK,
    KIS_STORYBOARD_WEIGHT,
)


@dataclass(frozen=True)
class _BeamState:
    last_local_idx: int
    total_score: float
    covered: int
    selected: tuple[tuple[int, dict, float], ...]


def _scene_score(row: dict, scene_index: int) -> float:
    scores = row.get("scene_scores") or {}
    value = scores.get(scene_index, scores.get(str(scene_index), 0.0))
    return float(value)


def _best_per_scene(
    rows: Sequence[dict],
    scene_count: int,
) -> list[list[tuple[dict, float]]]:
    per_scene: list[list[tuple[dict, float]]] = []
    limit = max(1, KIS_STORYBOARD_HITS_PER_SCENE)
    for scene_index in range(scene_count):
        ranked = sorted(
            (
                (row, score)
                for row in rows
                if (score := _scene_score(row, scene_index)) > 0.0
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        per_scene.append(ranked[:limit])
    return per_scene


def _unordered_alignment(
    per_scene: Sequence[Sequence[tuple[dict, float]]],
) -> tuple[tuple[tuple[int, dict, float], ...], int, float]:
    selected: list[tuple[int, dict, float]] = []
    covered = 0
    total_score = 0.0
    for scene_index, hits in enumerate(per_scene):
        if not hits:
            continue
        row, score = hits[0]
        selected.append((scene_index, row, score))
        total_score += score
        covered += int(score >= KIS_STORYBOARD_MIN_SCENE_SCORE)
    return tuple(selected), covered, total_score


def _ordered_alignment(
    per_scene: Sequence[Sequence[tuple[dict, float]]],
    temporal_edges: Sequence[tuple[int, int]],
) -> tuple[tuple[tuple[int, dict, float], ...], int, float]:
    """Beam-search a partial-order alignment, allowing missing scene evidence."""
    beam = [_BeamState(-1, 0.0, 0, ())]
    beam_size = max(1, KIS_STORYBOARD_BEAM_SIZE)
    slack = max(0, KIS_STORYBOARD_ORDER_SLACK)
    threshold = KIS_STORYBOARD_MIN_SCENE_SCORE

    for scene_index, hits in enumerate(per_scene):
        expanded: list[_BeamState] = []
        for state in beam:
            # Missing one visual clause must not delete an otherwise correct video.
            expanded.append(state)
            selected_by_scene = {
                selected_scene: selected_row
                for selected_scene, selected_row, _ in state.selected
            }
            for row, score in hits:
                local_idx = int(row["local_idx"])
                predecessors = [
                    source
                    for source, target in temporal_edges
                    if target == scene_index and source in selected_by_scene
                ]
                if any(
                    local_idx + slack
                    < int(selected_by_scene[source]["local_idx"])
                    for source in predecessors
                ):
                    continue
                expanded.append(
                    _BeamState(
                        last_local_idx=max(state.last_local_idx, local_idx),
                        total_score=state.total_score + score,
                        covered=state.covered + int(score >= threshold),
                        selected=state.selected + ((scene_index, row, score),),
                    )
                )
        # Coverage is the primary objective; semantic strength breaks ties. A
        # tiny compactness term prefers a coherent clip when two alignments tie.
        expanded.sort(
            key=lambda state: (
                state.covered,
                state.total_score,
                -(
                    state.selected[-1][1]["local_idx"]
                    - state.selected[0][1]["local_idx"]
                )
                if len(state.selected) > 1
                else 0,
            ),
            reverse=True,
        )
        beam = expanded[:beam_size]

    best = max(
        beam,
        key=lambda state: (state.covered, state.total_score),
        default=_BeamState(-1, 0.0, 0, ()),
    )
    return best.selected, best.covered, best.total_score


def _story_score(
    per_scene: Sequence[Sequence[tuple[dict, float]]],
    selected: Sequence[tuple[int, dict, float]],
    aligned_covered: int,
    aligned_total: float,
    scene_count: int,
    ordered: bool,
) -> tuple[float, float, float]:
    threshold = KIS_STORYBOARD_MIN_SCENE_SCORE
    raw_covered = sum(
        bool(hits and hits[0][1] >= threshold) for hits in per_scene
    )
    raw_coverage = raw_covered / scene_count
    aligned_coverage = aligned_covered / scene_count
    semantic_quality = aligned_total / scene_count
    peak = max((score for _, _, score in selected), default=0.0)
    if ordered:
        score = (
            0.45 * aligned_coverage
            + 0.20 * raw_coverage
            + 0.30 * semantic_quality
            + 0.05 * peak
        )
    else:
        score = 0.65 * raw_coverage + 0.30 * semantic_quality + 0.05 * peak
    return min(max(score, 0.0), 1.0), raw_coverage, aligned_coverage


def apply_storyboard_alignment(
    candidates: list[dict],
    *,
    scene_count: int,
    temporal_edges: Sequence[tuple[int, int]] = (),
) -> None:
    """Mutate candidate scores using scene coverage from the same video.

    The operation is deliberately soft: missing one clause lowers the video score
    but never removes it from the Top-100 candidate pool.  This protects recall
    when organizer keyframes omit the exact moment described by one sentence.
    """
    if not KIS_STORYBOARD_ENABLED or scene_count < 2 or not candidates:
        return

    by_video: dict[str, list[dict]] = defaultdict(list)
    for row in candidates:
        by_video[row["video_id"]].append(row)

    weight = min(max(KIS_STORYBOARD_WEIGHT, 0.0), 1.0)
    ordered = bool(temporal_edges)
    for rows in by_video.values():
        per_scene = _best_per_scene(rows, scene_count)
        if ordered:
            selected, aligned_covered, aligned_total = _ordered_alignment(
                per_scene, temporal_edges
            )
        else:
            selected, aligned_covered, aligned_total = _unordered_alignment(per_scene)
        story_score, raw_coverage, aligned_coverage = _story_score(
            per_scene,
            selected,
            aligned_covered,
            aligned_total,
            scene_count,
            ordered,
        )
        evidence_by_local: dict[int, list[int]] = defaultdict(list)
        for scene_index, row, _ in selected:
            evidence_by_local[int(row["local_idx"])].append(scene_index)
        evidence_local_idxs = list(evidence_by_local)

        for row in rows:
            local_idx = int(row["local_idx"])
            base_score = float(row["score"])
            evidence_affinity = max(
                (float(value) for value in (row.get("scene_scores") or {}).values()),
                default=0.0,
            )
            row["retrieval_score_before_storyboard"] = base_score
            row["storyboard_score"] = story_score
            row["query_coverage"] = (
                aligned_coverage if ordered else raw_coverage
            )
            row["raw_query_coverage"] = raw_coverage
            row["temporal_alignment_coverage"] = aligned_coverage
            row["storyboard_local_idxs"] = evidence_local_idxs
            row["storyboard_scene_indices"] = evidence_by_local.get(local_idx, [])
            row["score"] = (
                (1.0 - weight) * base_score
                + weight * story_score
                + (0.02 * evidence_affinity if local_idx in evidence_by_local else 0.0)
            )
