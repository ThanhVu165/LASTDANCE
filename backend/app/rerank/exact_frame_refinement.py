"""Coarse-to-fine source-video frame refinement for leading TRAKE results.

Official TRAKE semantic intervals are commonly shorter than ten source frames,
whereas supplied technical keyframes can be seconds apart.  This module decodes a
bounded neighborhood only for the leading ranked sequence and lets Qwen select the
frame whose visible action state best matches each moment description.
"""
from __future__ import annotations

import csv
import math
import re
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from app.config import (
    KIS_EXACT_FRAME_ENABLED,
    KIS_EXACT_FRAME_TOP_K,
    MAP_KEYFRAMES_DIR,
    ROOT_DIR,
    TRAKE_EXACT_FRAME_COARSE_SAMPLES,
    TRAKE_EXACT_FRAME_ENABLED,
    TRAKE_EXACT_FRAME_FINE_SAMPLES,
    TRAKE_EXACT_FRAME_MAX_RADIUS,
    TRAKE_EXACT_FRAME_MAX_NEW_TOKENS,
    TRAKE_EXACT_FRAME_TOP_K,
    VIDEOS_DIR,
)
from app.services.visual_qa import generate_images


_cv2_error = getattr(cv2, "error", None)
_REFINEMENT_ERRORS = (RuntimeError, OSError, ValueError) + (
    (_cv2_error,)
    if isinstance(_cv2_error, type) and issubclass(_cv2_error, BaseException)
    else ()
)
_BEST_PATTERN = re.compile(r"BEST\s*[:=]\s*([0-9]+)", re.IGNORECASE)
_BOUNDARY_PATTERN = re.compile(
    r"\b(?:first|begins?|starts?|initially|just as|touch(?:es|ing)?|contact|"
    r"leaves?|separates?|highest|lowest|enters?|cross(?:es|ing)?|fully|completely|"
    r"đầu tiên|bắt đầu|ngay khi|vừa|chạm|rời|tách|cao nhất|thấp nhất|đi vào|"
    r"vào lưới|vượt qua|cán qua|hoàn toàn)\b",
    re.IGNORECASE,
)


def _requires_exact_refinement(moment: str) -> bool:
    return _BOUNDARY_PATTERN.search(moment) is not None


@lru_cache(maxsize=256)
def _frame_map(video_id: str) -> tuple[tuple[int, int], ...]:
    path = MAP_KEYFRAMES_DIR / f"{video_id}.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return tuple(
            (int(row["n"]), int(float(row["frame_idx"])))
            for row in csv.DictReader(stream)
        )


def _candidate_window(video_id: str, local_idx: int, frame_id: int) -> tuple[int, int]:
    mapping = _frame_map(video_id)
    positions = {local: index for index, (local, _) in enumerate(mapping)}
    position = positions.get(int(local_idx))
    if position is None:
        return (
            max(0, frame_id - TRAKE_EXACT_FRAME_MAX_RADIUS),
            frame_id + TRAKE_EXACT_FRAME_MAX_RADIUS,
        )
    previous = mapping[position - 1][1] if position > 0 else None
    following = mapping[position + 1][1] if position + 1 < len(mapping) else None
    left = (previous + frame_id) // 2 if previous is not None else frame_id - 90
    right = (frame_id + following) // 2 if following is not None else frame_id + 90
    return (
        max(0, frame_id - TRAKE_EXACT_FRAME_MAX_RADIUS, left),
        min(frame_id + TRAKE_EXACT_FRAME_MAX_RADIUS, right),
    )


def _sample_frame_ids(start: int, end: int, count: int) -> list[int]:
    if end < start:
        start, end = end, start
    count = max(1, min(count, end - start + 1))
    return list(dict.fromkeys(int(round(value)) for value in np.linspace(start, end, count)))


def _decode_frames(video_id: str, frame_ids: Sequence[int], folder: Path) -> list[Path]:
    video_path = VIDEOS_DIR / f"{video_id}.mp4"
    if not video_path.exists() or not frame_ids:
        return []
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        return []
    paths: list[Path] = []
    try:
        for frame_id in frame_ids:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
            ok, frame = capture.read()
            if not ok:
                continue
            path = folder / f"{int(frame_id):09d}.jpg"
            if cv2.imwrite(str(path), frame):
                paths.append(path)
    finally:
        capture.release()
    return paths


def _contact_sheet(paths: Sequence[Path]) -> Path:
    columns = max(1, math.ceil(math.sqrt(len(paths))))
    rows = math.ceil(len(paths) / columns)
    cell_width, cell_height = 320, 180
    sheet = np.zeros((rows * cell_height, columns * cell_width, 3), dtype=np.uint8)
    for index, path in enumerate(paths):
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Cannot read decoded frame {path}.")
        cell = cv2.resize(frame, (cell_width, cell_height), interpolation=cv2.INTER_AREA)
        cv2.rectangle(cell, (4, 4), (58, 42), (0, 0, 0), thickness=-1)
        cv2.putText(
            cell,
            str(index + 1),
            (12, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        row, column = divmod(index, columns)
        sheet[
            row * cell_height : (row + 1) * cell_height,
            column * cell_width : (column + 1) * cell_width,
        ] = cell
    output = paths[0].parent / f"contact-{len(paths)}.jpg"
    if not cv2.imwrite(str(output), sheet):
        raise ValueError("Cannot write exact-frame contact sheet.")
    return output


def _select_frame(moment: str, frame_ids: Sequence[int], paths: Sequence[Path]) -> int:
    if not paths or len(paths) != len(frame_ids):
        raise ValueError("Decoded frame list is incomplete.")
    contact_sheet = _contact_sheet(paths)
    prompt = (
        "This contact sheet contains numbered consecutive video frames in chronological "
        "order. Select the numbered panel that most precisely satisfies the semantic "
        "moment, paying attention to first contact, first separation, peak position, "
        "object state and other boundary words. Judge only visible evidence. Output "
        "no explanation. Your response must begin with and contain only "
        "BEST=<image number>.\n"
        f"Semantic moment: {moment}"
    )
    decoded = generate_images(
        (str(contact_sheet),),
        prompt,
        max_new_tokens=TRAKE_EXACT_FRAME_MAX_NEW_TOKENS,
    )
    match = _BEST_PATTERN.search(decoded)
    if not match:
        raise ValueError("VLM did not return a valid BEST frame.")
    index = int(match.group(1)) - 1
    if not 0 <= index < len(frame_ids):
        raise ValueError("VLM BEST frame is outside the candidate list.")
    return int(frame_ids[index])


def _refine_one(video_id: str, local_idx: int, frame_id: int, moment: str) -> int:
    left, right = _candidate_window(video_id, local_idx, frame_id)
    temp_root = ROOT_DIR / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="trake-frame-", dir=temp_root) as temp:
        folder = Path(temp)
        coarse_ids = _sample_frame_ids(
            left, right, TRAKE_EXACT_FRAME_COARSE_SAMPLES
        )
        coarse_paths = _decode_frames(video_id, coarse_ids, folder)
        coarse_best = _select_frame(moment, coarse_ids, coarse_paths)
        coarse_step = max(2, (right - left) // max(len(coarse_ids) - 1, 1))

        fine_left = max(left, coarse_best - coarse_step)
        fine_right = min(right, coarse_best + coarse_step)
        fine_ids = _sample_frame_ids(
            fine_left, fine_right, TRAKE_EXACT_FRAME_FINE_SAMPLES
        )
        fine_paths = _decode_frames(video_id, fine_ids, folder)
        return _select_frame(moment, fine_ids, fine_paths)


def refine_top_trake_frames(moments: list[str], rows: list[dict]) -> list[dict]:
    if not TRAKE_EXACT_FRAME_ENABLED or not rows:
        return rows
    refined = list(rows)
    for rank in range(min(TRAKE_EXACT_FRAME_TOP_K, len(refined))):
        row = refined[rank]
        original_frame_ids = list(row["frame_ids"])
        try:
            exact_frame_ids = [
                (
                    _refine_one(row["video_id"], local_idx, frame_id, moment)
                    if _requires_exact_refinement(moment)
                    else frame_id
                )
                for local_idx, frame_id, moment in zip(
                    row["local_idxs"], original_frame_ids, moments
                )
            ]
        except _REFINEMENT_ERRORS:
            continue
        strictly_increasing = all(
            left < right
            for left, right in zip(exact_frame_ids, exact_frame_ids[1:])
        )
        if strictly_increasing and len(exact_frame_ids) == len(moments):
            refined[rank] = {
                **row,
                "keyframe_frame_ids": original_frame_ids,
                "frame_ids": exact_frame_ids,
            }
    return refined


def refine_top_kis_frames(query: str, rows: list[dict]) -> list[dict]:
    """Replace only leading KIS keyframes with bounded source-video evidence.

    Organizer keyframes are retrieval aids rather than guaranteed semantic-answer
    frames.  Refining the first result is affordable on the 6 GB GPU and directly
    targets the top-1 penalty without decoding the collection.
    """
    if not KIS_EXACT_FRAME_ENABLED or not rows:
        return rows
    refined = list(rows)
    for rank in range(min(KIS_EXACT_FRAME_TOP_K, len(refined))):
        row = refined[rank]
        original_frame_id = int(row["frame_id"])
        try:
            exact_frame_id = _refine_one(
                row["video_id"],
                int(row["local_idx"]),
                original_frame_id,
                query,
            )
        except _REFINEMENT_ERRORS:
            continue
        refined[rank] = {
            **row,
            "keyframe_frame_id": original_frame_id,
            "frame_id": exact_frame_id,
        }
    return refined
