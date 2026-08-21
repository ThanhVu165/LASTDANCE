"""Comparative visual reranking with the configured local Qwen3-VL model.

Only a small, retrieval-selected pool is inspected.  The model is used as visual
evidence, never as collection-wide retrieval and never as ground truth.  Any CUDA,
model, image, or parsing failure leaves the baseline ranking intact.
"""
from __future__ import annotations

import math
import re
import tempfile
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageOps

from app.config import (
    KEYFRAMES_DIR,
    ROOT_DIR,
    VLM_RERANK_BEST_FRAME_WEIGHT,
    VLM_RERANK_ENABLED,
    VLM_RERANK_FRAMES_PER_VIDEO,
    VLM_RERANK_GROUP_SIZE,
    VLM_RERANK_MAX_NEW_TOKENS,
    VLM_RERANK_TOP_SEQUENCES,
    VLM_RERANK_TOP_VIDEOS,
    VLM_RERANK_VISUAL_WEIGHT,
)
from app.services.visual_qa import generate_images
from app.services.temporal_context import temporal_triplet


_SCORE_PATTERN = re.compile(r"SCORE\s*[:=]\s*(100|[0-9]{1,2})", re.IGNORECASE)
_BEST_PATTERN = re.compile(r"BEST\s*[:=]\s*([0-9]+)", re.IGNORECASE)
_BEST_VIDEO_PATTERN = re.compile(
    r"BESTVIDEO\s*[:=]\s*(?:V|ROW\s*)?([0-9]+)", re.IGNORECASE
)
_BEST_PANEL_PATTERN = re.compile(
    r"BESTPANEL\s*[:=]\s*(?:V[0-9]+\s*[-/:]\s*)?"
    r"(?:P(?:ANEL)?\s*)?([0-9]+)",
    re.IGNORECASE,
)
_CHECKS_PATTERN = re.compile(r"CHECKS\s*[:=]\s*([^;\r\n]+)", re.IGNORECASE)


def _keyframe_path(row: dict) -> Path:
    supplied = row.get("keyframe_path")
    if supplied:
        supplied_path = Path(supplied)
        if supplied_path.exists():
            return supplied_path
    folder = KEYFRAMES_DIR / row["video_id"]
    for width in (3, 4):
        candidate = folder / f"{int(row['local_idx']):0{width}d}.jpg"
        if candidate.exists():
            return candidate
    return folder / f"{int(row['local_idx']):03d}.jpg"


def _parse_score_and_best(text: str, image_count: int) -> tuple[float, int | None] | None:
    score_match = _SCORE_PATTERN.search(text)
    if not score_match:
        return None
    score = min(max(int(score_match.group(1)), 0), 100) / 100.0
    best_match = _BEST_PATTERN.search(text)
    best_index = int(best_match.group(1)) - 1 if best_match else None
    if best_index is not None and not 0 <= best_index < image_count:
        best_index = None
    return score, best_index


def _parse_video_comparison(
    text: str,
    panel_counts: Sequence[int],
    criterion_count: int | None = None,
) -> tuple[float, int, int] | None:
    score_match = _SCORE_PATTERN.search(text)
    video_match = _BEST_VIDEO_PATTERN.search(text)
    panel_match = _BEST_PANEL_PATTERN.search(text)
    if not video_match or not panel_match:
        return None
    video_index = int(video_match.group(1)) - 1
    panel_index = int(panel_match.group(1)) - 1
    if not 0 <= video_index < len(panel_counts):
        return None
    if not 0 <= panel_index < panel_counts[video_index]:
        return None
    model_score = (
        min(max(int(score_match.group(1)), 0), 100) / 100.0
        if score_match
        else 0.5
    )
    score = model_score
    checks_match = _CHECKS_PATTERN.search(text)
    if checks_match and criterion_count and criterion_count > 0:
        coverage_by_video: dict[int, float] = {}
        for raw_video, bits in re.findall(
            r"V([0-9]+)\s*:\s*([01]+)",
            checks_match.group(1),
            flags=re.IGNORECASE,
        ):
            checked_video = int(raw_video) - 1
            if not 0 <= checked_video < len(panel_counts):
                continue
            normalized_bits = bits[:criterion_count].ljust(criterion_count, "0")
            coverage_by_video[checked_video] = (
                normalized_bits.count("1") / criterion_count
            )
        if coverage_by_video:
            best_coverage = max(coverage_by_video.values())
            coverage_winners = [
                index
                for index, coverage in coverage_by_video.items()
                if abs(coverage - best_coverage) < 1e-8
            ]
            if video_index not in coverage_winners:
                # Rows are retrieval-ranked before entering each group, so the
                # first coverage tie is the strongest calibrated candidate.
                video_index = min(coverage_winners)
                panel_index = min(panel_index, panel_counts[video_index] - 1)
            score = 0.70 * best_coverage + 0.30 * model_score
    return score, video_index, panel_index


def _contact_sheet(image_paths: tuple[str, ...], folder: Path) -> Path:
    columns = max(1, math.ceil(math.sqrt(len(image_paths))))
    rows = math.ceil(len(image_paths) / columns)
    cell_size = (320, 180)
    sheet = Image.new(
        "RGB",
        (columns * cell_size[0], rows * cell_size[1]),
        color="black",
    )
    draw = ImageDraw.Draw(sheet)
    for index, image_path in enumerate(image_paths):
        with Image.open(image_path) as source:
            cell = ImageOps.fit(source.convert("RGB"), cell_size)
        row, column = divmod(index, columns)
        x, y = column * cell_size[0], row * cell_size[1]
        sheet.paste(cell, (x, y))
        draw.rectangle((x + 4, y + 4, x + 58, y + 42), fill="black")
        draw.text((x + 18, y + 10), str(index + 1), fill="white")
    output = folder / "candidates.jpg"
    sheet.save(output, format="JPEG", quality=90)
    return output


def _temporal_sequence_sheet(image_paths: tuple[str, ...], folder: Path) -> Path:
    """Lay out before/current/after evidence as one row per TRAKE moment."""
    if len(image_paths) % 3:
        raise ValueError("TRAKE temporal evidence must contain frame triplets")
    cell_size = (320, 180)
    sheet = Image.new(
        "RGB",
        (3 * cell_size[0], (len(image_paths) // 3) * cell_size[1]),
        color="black",
    )
    draw = ImageDraw.Draw(sheet)
    states = ("BEFORE", "CANDIDATE", "AFTER")
    for index, image_path in enumerate(image_paths):
        with Image.open(image_path) as source:
            cell = ImageOps.fit(source.convert("RGB"), cell_size)
        event_index, state_index = divmod(index, 3)
        x, y = state_index * cell_size[0], event_index * cell_size[1]
        sheet.paste(cell, (x, y))
        label = f"E{event_index + 1}-{states[state_index]}"
        draw.rectangle((x + 4, y + 4, x + 174, y + 38), fill="black")
        draw.text((x + 12, y + 9), label, fill="white")
    output = folder / "temporal_sequence.jpg"
    sheet.save(output, format="JPEG", quality=90)
    return output


def _video_comparison_sheet(
    grouped_paths: tuple[tuple[str, ...], ...],
    folder: Path,
) -> Path:
    """One video per row so the VLM compares candidates on the same scale."""
    columns = max(len(paths) for paths in grouped_paths)
    cell_size = (320, 180)
    sheet = Image.new(
        "RGB",
        (columns * cell_size[0], len(grouped_paths) * cell_size[1]),
        color="black",
    )
    draw = ImageDraw.Draw(sheet)
    for video_index, paths in enumerate(grouped_paths):
        for panel_index, image_path in enumerate(paths):
            with Image.open(image_path) as source:
                cell = ImageOps.fit(source.convert("RGB"), cell_size)
            x, y = panel_index * cell_size[0], video_index * cell_size[1]
            sheet.paste(cell, (x, y))
            label = f"V{video_index + 1}-P{panel_index + 1}"
            draw.rectangle((x + 4, y + 4, x + 92, y + 38), fill="black")
            draw.text((x + 11, y + 9), label, fill="white")
    output = folder / "video_comparison.jpg"
    sheet.save(output, format="JPEG", quality=90)
    return output


@lru_cache(maxsize=256)
def _cached_video_comparison(
    query: str,
    grouped_paths: tuple[tuple[str, ...], ...],
) -> tuple[float, int, int] | None:
    if not grouped_paths or any(not paths for paths in grouped_paths):
        return None
    criteria = [
        clause.strip()
        for clause in re.split(r"(?<=[.!?])\s+|\s*;\s*", query)
        if len(clause.strip().split()) >= 3
    ][:8]
    if not criteria:
        criteria = [query.strip()]
    criteria_text = "\n".join(
        f"C{index}: {criterion}" for index, criterion in enumerate(criteria, 1)
    )
    prompt = (
        "Each ROW in the contact sheet is a different candidate video; panels in "
        "one row are chronological evidence from that video. Compare all rows and "
        "choose the ONE row that best matches the complete query. Every criterion "
        "below matters. Reject a row that only matches a generic theme or object but "
        "misses the distinctive combination of actions, attributes, setting, count, "
        "visible text or later scenes. Choose the least-wrong row if none is complete. "
        "Reply only with BESTVIDEO=<row number>;BESTPANEL=<panel number>. Do not copy "
        "the angle brackets and do not explain.\n"
        f"Required criteria:\n{criteria_text}"
    )
    temp_root = ROOT_DIR / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kis-tournament-", dir=temp_root) as temp:
        sheet = _video_comparison_sheet(grouped_paths, Path(temp))
        decoded = generate_images(
            (str(sheet),),
            prompt,
            max_new_tokens=max(VLM_RERANK_MAX_NEW_TOKENS, 32),
        )
    return _parse_video_comparison(
        decoded,
        [len(paths) for paths in grouped_paths],
        len(criteria),
    )


@lru_cache(maxsize=512)
def _cached_visual_judgment(
    query: str,
    image_paths: tuple[str, ...],
    sequence_mode: bool,
) -> tuple[float, int | None] | None:
    if sequence_mode:
        prompt = (
            "The contact sheet contains one chronological row per requested event. "
            "Each row has BEFORE, CANDIDATE and AFTER evidence from ONE video. "
            "Judge whether every CANDIDATE is the exact requested action boundary, "
            "whether the surrounding frames support that transition, and whether rows "
            "follow the ordered query. Do not score generic objects alone. Output "
            "exactly: SCORE=<integer 0-100>; BEST=1\n"
            f"Ordered query: {query}"
        )
    else:
        clauses = [
            clause.strip()
            for clause in re.split(r"(?<=[.!?])\s+|\s*;\s*", query)
            if len(clause.strip().split()) >= 3
        ]
        coverage_rule = (
            "Treat every sentence or scene clause as required evidence from the "
            "same video. A video that matches only a generic theme or object but "
            "misses any distinctive action, setting, named place, later scene, "
            "count, colour or product must score at most 35. "
            if len(clauses) >= 2
            else ""
        )
        prompt = (
            "The numbered panels in the contact sheet are candidate frames from ONE "
            "video in chronological order. "
            "Judge only visible evidence; do not reward a frame for merely containing a "
            "generic object. Score how well this video evidence matches the complete "
            "query, including actions, attributes, text and scene context. "
            f"{coverage_rule}"
            "BEST is the "
            "single panel most suitable to submit for the query. Output exactly: "
            "SCORE=<integer 0-100>; BEST=<panel number>\n"
            f"Query: {query}"
        )
    if len(image_paths) > 1:
        temp_root = ROOT_DIR / "tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        prefix = "trake-rerank-" if sequence_mode else "kis-rerank-"
        with tempfile.TemporaryDirectory(prefix=prefix, dir=temp_root) as temp:
            sheet_path = (
                _temporal_sequence_sheet(image_paths, Path(temp))
                if sequence_mode
                else _contact_sheet(image_paths, Path(temp))
            )
            decoded = generate_images(
                (str(sheet_path),),
                prompt,
                max_new_tokens=VLM_RERANK_MAX_NEW_TOKENS,
            )
    else:
        decoded = generate_images(
            image_paths,
            prompt,
            max_new_tokens=VLM_RERANK_MAX_NEW_TOKENS,
        )
    return _parse_score_and_best(decoded, len(image_paths))


def _minmax_scores(rows: Sequence[dict]) -> dict[int, float]:
    if not rows:
        return {}
    values = [float(row["score"]) for row in rows]
    low, high = min(values), max(values)
    if high - low < 1e-8:
        return {id(row): 1.0 for row in rows}
    return {id(row): (float(row["score"]) - low) / (high - low) for row in rows}


def _representative_rows(
    rows: Sequence[dict],
    limit: int,
    *,
    min_spacing: int,
) -> list[dict]:
    """Choose one evidence frame per story node before score-only fallbacks."""
    chosen: list[dict] = []
    storyboard_local_idxs = next(
        (
            list(row.get("storyboard_local_idxs") or [])
            for row in rows
            if row.get("storyboard_local_idxs")
        ),
        [],
    )
    storyboard_local_idxs = list(dict.fromkeys(map(int, storyboard_local_idxs)))
    if len(storyboard_local_idxs) > limit:
        positions = [
            int(round(index * (len(storyboard_local_idxs) - 1) / (limit - 1)))
            for index in range(limit)
        ] if limit > 1 else [len(storyboard_local_idxs) // 2]
        storyboard_local_idxs = [storyboard_local_idxs[index] for index in positions]
    for local_idx in storyboard_local_idxs:
        row = max(
            (
                candidate
                for candidate in rows
                if int(candidate["local_idx"]) == local_idx
            ),
            key=lambda candidate: float(candidate["score"]),
            default=None,
        )
        if row is not None and _keyframe_path(row).exists():
            chosen.append(row)

    for row in sorted(rows, key=lambda item: float(item["score"]), reverse=True):
        if len(chosen) >= limit:
            break
        if row in chosen:
            continue
        local_idx = int(row["local_idx"])
        if chosen and any(
            abs(local_idx - int(prev["local_idx"])) < min_spacing
            for prev in chosen
        ):
            continue
        if _keyframe_path(row).exists():
            chosen.append(row)
        if len(chosen) >= limit:
            break
    if not chosen:
        return []
    return sorted(chosen, key=lambda item: int(item["local_idx"]))


def _with_temporal_context(
    rows: Sequence[dict],
    *,
    limit: int,
) -> list[dict]:
    """Use before/current/after keyframes around the strongest short-query hit."""
    if not rows or limit < 3:
        return list(rows)[:limit]
    anchor = max(rows, key=lambda row: float(row["score"]))
    records = temporal_triplet(anchor["video_id"], int(anchor["local_idx"]))
    contextual: list[dict] = []
    seen: set[int] = set()
    for record in records:
        if record.local_idx in seen:
            continue
        seen.add(record.local_idx)
        contextual.append(
            {
                **anchor,
                "local_idx": record.local_idx,
                "frame_id": record.frame_id,
                "keyframe_path": str(record.path),
                "is_temporal_context": record.local_idx != int(anchor["local_idx"]),
            }
        )
    for row in rows:
        local_idx = int(row["local_idx"])
        if local_idx in seen:
            continue
        seen.add(local_idx)
        contextual.append(row)
        if len(contextual) >= limit:
            break
    return sorted(contextual[:limit], key=lambda row: int(row["local_idx"]))


def rerank_kis_candidates(query: str, candidates: list[dict]) -> list[dict]:
    if not VLM_RERANK_ENABLED or not candidates:
        return candidates

    by_video: dict[str, list[dict]] = defaultdict(list)
    for row in candidates:
        by_video[row["video_id"]].append(row)
    score_ranked_videos = sorted(
        by_video,
        key=lambda video_id: max(float(row["score"]) for row in by_video[video_id]),
        reverse=True,
    )
    ranked_videos = score_ranked_videos[:VLM_RERANK_TOP_VIDEOS]
    clauses = [
        clause.strip()
        for clause in re.split(r"(?<=[.!?])\s+|\s*;\s*", query)
        if len(clause.strip().split()) >= 3
    ]
    if len(clauses) >= 3 and VLM_RERANK_TOP_VIDEOS > 1:
        # Two thirds exploit best fused-frame score; the remaining third recovers
        # videos that appear across many independent scene prompts. The subsequent
        # tournament compares a small configured group per VLM call, so this wider
        # pool is still cheaper and more reliable than independent self-scoring.
        score_quota = max(1, math.ceil(2 * VLM_RERANK_TOP_VIDEOS / 3))
        ranked_videos = score_ranked_videos[:score_quota]
        coverage_ranked_videos = sorted(
            by_video,
            key=lambda video_id: (
                max(
                    float(row.get("query_coverage", 0.0))
                    for row in by_video[video_id]
                ),
                max(float(row["score"]) for row in by_video[video_id]),
            ),
            reverse=True,
        )
        for video_id in coverage_ranked_videos:
            if video_id not in ranked_videos:
                ranked_videos.append(video_id)
            if len(ranked_videos) >= VLM_RERANK_TOP_VIDEOS:
                break

    representatives_by_video: dict[str, list[dict]] = {}
    for video_id in ranked_videos:
        representatives = _representative_rows(
            by_video[video_id],
            max(1, VLM_RERANK_FRAMES_PER_VIDEO),
            min_spacing=5 if len(query) >= 220 else 1,
        )
        if len(clauses) <= 2:
            representatives = _with_temporal_context(
                representatives,
                limit=max(1, VLM_RERANK_FRAMES_PER_VIDEO),
            )
        if representatives:
            representatives_by_video[video_id] = representatives
    ranked_videos = [
        video_id for video_id in ranked_videos if video_id in representatives_by_video
    ]
    if not ranked_videos:
        return candidates

    # Qwen3-VL-2B is not calibrated as an absolute relevance scorer: it can emit
    # SCORE=100 for several visibly different videos. Relative selection is much
    # more reliable. A small knockout bracket also keeps each sheet readable.
    group_size = max(2, VLM_RERANK_GROUP_SIZE)
    remaining = list(ranked_videos)
    winner_best_index: int | None = None
    winner_visual_score = 0.5
    while len(remaining) > 1:
        next_round: list[str] = []
        for start in range(0, len(remaining), group_size):
            group = remaining[start : start + group_size]
            if len(group) == 1:
                next_round.append(group[0])
                continue
            grouped_paths = tuple(
                tuple(
                    str(_keyframe_path(row))
                    for row in representatives_by_video[video_id]
                )
                for video_id in group
            )
            try:
                judgment = _cached_video_comparison(query, grouped_paths)
            except (RuntimeError, OSError, ValueError):
                judgment = None
            if judgment is None:
                # The input order follows calibrated retrieval and is therefore
                # the safest fallback for a malformed model response.
                next_round.append(group[0])
                continue
            comparison_score, winner_index, panel_index = judgment
            next_round.append(group[winner_index])
            if len(remaining) <= group_size:
                winner_visual_score = comparison_score
                winner_best_index = panel_index
        remaining = next_round

    winner_video = remaining[0]
    winner_rows = representatives_by_video[winner_video]
    best_index = (
        winner_best_index
        if winner_best_index is not None and 0 <= winner_best_index < len(winner_rows)
        else max(
            range(len(winner_rows)),
            key=lambda index: float(winner_rows[index]["score"]),
        )
    )
    best_row = winner_rows[best_index]
    base_scores = _minmax_scores(candidates)
    retrieval_ceiling = max(float(row["score"]) for row in candidates)
    best_weight = min(max(VLM_RERANK_BEST_FRAME_WEIGHT, 0.0), 1.0)
    reranked: list[dict] = []
    visual_weight = min(max(VLM_RERANK_VISUAL_WEIGHT, 0.0), 1.0)
    for row in candidates:
        if row["video_id"] != winner_video:
            reranked.append(row)
            continue
        reranked.append(
            {
                **row,
                "retrieval_score": float(row["score"]),
                "visual_score": winner_visual_score,
                "score": (
                    retrieval_ceiling
                    + 0.01 * base_scores[id(row)]
                    + visual_weight * winner_visual_score
                    + best_weight * (1.0 if row is best_row else 0.0)
                ),
            }
        )
    return reranked


def rerank_trake_hypotheses(
    query: str,
    hypotheses: list[dict],
) -> list[dict]:
    if not VLM_RERANK_ENABLED or not hypotheses:
        return hypotheses

    # Judge at most two alignments per leading video. This samples video identity
    # as well as alignment uncertainty instead of spending all calls on one video.
    pool: list[dict] = []
    counts: dict[str, int] = defaultdict(int)
    for row in sorted(hypotheses, key=lambda item: float(item["score"]), reverse=True):
        if counts[row["video_id"]] >= 2:
            continue
        temporal_records = [
            temporal_triplet(row["video_id"], int(local_idx))
            for local_idx in row["local_idxs"]
        ]
        paths = tuple(
            str(record.path)
            for triplet in temporal_records
            for record in triplet
        )
        if not paths or any(not Path(path).exists() for path in paths):
            continue
        pool.append({**row, "_paths": paths})
        counts[row["video_id"]] += 1
        if len(pool) >= VLM_RERANK_TOP_SEQUENCES:
            break

    base_scores = _minmax_scores(hypotheses)
    judgments: dict[tuple[str, tuple[int, ...]], float] = {}
    try:
        for row in pool:
            judgment = _cached_visual_judgment(query, row["_paths"], True)
            if judgment is not None:
                judgments[(row["video_id"], tuple(row["local_idxs"]))] = judgment[0]
    except (RuntimeError, OSError, ValueError):
        return hypotheses

    if not judgments:
        return hypotheses
    visual_weight = min(max(VLM_RERANK_VISUAL_WEIGHT, 0.0), 1.0)
    reranked: list[dict] = []
    for row in hypotheses:
        key = (row["video_id"], tuple(row["local_idxs"]))
        visual_score = judgments.get(key)
        if visual_score is None:
            reranked.append(row)
            continue
        retrieval_score = float(row["score"])
        reranked.append(
            {
                **row,
                "retrieval_score": retrieval_score,
                "visual_score": visual_score,
                "score": (1.0 - visual_weight) * base_scores[id(row)]
                + visual_weight * visual_score,
            }
        )
    return reranked
