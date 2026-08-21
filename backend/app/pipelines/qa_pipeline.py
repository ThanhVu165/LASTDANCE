"""Q&A: understand one complete query, retrieve the event, then answer visually."""
from __future__ import annotations

import re
import tempfile
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from app.config import (
    KEYFRAMES_DIR,
    MAX_ANSWER_LENGTH,
    MAX_SUBMISSION_ROWS,
    QA_JUDGMENT_MAX_NEW_TOKENS,
    QA_MATCH_WEIGHT,
    QA_CONTEXT_RADIUS,
    QA_MULTIFRAME_TOP_K,
    QA_TEMPORAL_RADIUS_SECONDS,
    QA_TEMPORAL_SAMPLES,
    QA_VLM_TOP_VIDEOS,
    ROOT_DIR,
)
from app.pipelines.kis_pipeline import run_kis_query
from app.rerank.contest_ranking import KIS_VIDEO_CAPS, cutoff_aware_rank
from app.services.query_processing import parse_qa_query
from app.services.temporal_context import (
    KeyframeRecord,
    build_contact_sheet,
    sample_temporal_context,
)
from app.services.visual_qa import generate_images


_QA_JUDGMENT_PATTERN = re.compile(
    r"MATCH\s*[:=]\s*(100|[0-9]{1,2})\s*;\s*"
    r"BEST\s*[:=]\s*([0-9]+)\s*;\s*"
    r"ANSWER\s*[:=]\s*([^\r\n]+)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class QaJudgment:
    match_score: float
    best_index: int
    answer: str
    records: tuple[KeyframeRecord, ...]


def _keyframe_path(video_id: str, local_idx: int) -> Path:
    for width in (3, 4):
        candidate = KEYFRAMES_DIR / video_id / f"{local_idx:0{width}d}.jpg"
        if candidate.exists():
            return candidate
    return KEYFRAMES_DIR / video_id / f"{local_idx:03d}.jpg"


def _keyframe_context(video_id: str, local_idx: int, radius: int) -> list[Path]:
    paths = [
        _keyframe_path(video_id, neighbor)
        for neighbor in range(max(1, local_idx - radius), local_idx + radius + 1)
    ]
    return [path for path in paths if path.exists()]


def _normalize_answer(
    answer: str,
    *,
    uppercase: bool = False,
    no_spaces: bool = False,
) -> str:
    normalized = " ".join(answer.strip().split())
    if uppercase:
        normalized = normalized.upper()
    if no_spaces:
        normalized = normalized.replace(" ", "")
    return (normalized or "unknown")[:MAX_ANSWER_LENGTH]


def _parse_qa_judgment(text: str, image_count: int) -> tuple[float, int, str] | None:
    match = _QA_JUDGMENT_PATTERN.search(text.strip())
    if not match:
        return None
    best_index = int(match.group(2)) - 1
    if not 0 <= best_index < image_count:
        return None
    answer = " ".join(match.group(3).strip().split()).strip(" \"'“”.")
    if not answer:
        return None
    return min(max(int(match.group(1)), 0), 100) / 100.0, best_index, answer


def _judge_video(task, row: dict) -> QaJudgment | None:
    records = sample_temporal_context(
        row["video_id"],
        int(row["local_idx"]),
        radius_seconds=QA_TEMPORAL_RADIUS_SECONDS,
        count=QA_TEMPORAL_SAMPLES,
    )
    if not records:
        return None
    temp_root = ROOT_DIR / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="qa-context-", dir=temp_root) as temp:
        sheet = build_contact_sheet(
            records,
            Path(temp) / "timeline.jpg",
            columns=4,
        )
        prompt = (
            "The numbered panels are sparse frames from ONE video in chronological "
            "order around a retrieved event. First verify whether the visible video "
            "evidence matches the event description. Then answer the question from "
            "the best visible moment. Track actions across panels and count on the "
            "single best panel when the question asks for a number. Common knowledge "
            "may be used only after the video identity is visually consistent. Never "
            "copy a claimed answer from the description as if it were visual evidence. "
            "If the video is wrong or the answer cannot be determined, use UNKNOWN. "
            "Answer in the language requested by the question. Output exactly one line: "
            "MATCH=<integer 0-100>; BEST=<panel number>; ANSWER=<short answer>\n"
            f"Event description: {task.retrieval_text}\n"
            f"Question: {task.question}"
        )
        decoded = generate_images(
            (str(sheet),),
            prompt,
            max_new_tokens=QA_JUDGMENT_MAX_NEW_TOKENS,
        )
    parsed = _parse_qa_judgment(decoded, len(records))
    if parsed is None:
        return None
    match_score, best_index, answer = parsed
    return QaJudgment(
        match_score=match_score,
        best_index=best_index,
        answer=answer,
        records=tuple(records),
    )


def _legacy_answer_candidates(
    candidates: list[dict],
    task,
    answerer: Callable[[Sequence[str | Path], str], str],
) -> list[dict]:
    """Dependency-injected path used by tests and lightweight custom answerers."""
    results: list[dict] = []
    for rank, row in enumerate(candidates):
        if rank < QA_MULTIFRAME_TOP_K:
            image_paths = _keyframe_context(
                row["video_id"], row["local_idx"], QA_CONTEXT_RADIUS
            )
        else:
            image_paths = [_keyframe_path(row["video_id"], row["local_idx"])]
        answer = answerer(image_paths, task.question)
        results.append(
            {
                "video_id": row["video_id"],
                "frame_id": row["frame_id"],
                "local_idx": row["local_idx"],
                "answer": _normalize_answer(
                    answer,
                    uppercase=task.answer_uppercase,
                    no_spaces=task.answer_no_spaces,
                ),
                "score": float(row["score"]),
            }
        )
    return results


def run_qa_query(
    text: str,
    top_k: int = MAX_SUBMISSION_ROWS,
    answerer: Callable[[Sequence[str | Path], str], str] | None = None,
) -> list[dict]:
    task = parse_qa_query(text)
    result_count = min(top_k, MAX_SUBMISSION_ROWS)
    # QA performs its own temporal frame selection below, so KIS source-frame
    # refinement here would spend two extra VLM calls without affecting retrieval.
    candidates = run_kis_query(
        task.retrieval_text,
        top_k=result_count,
        refine_exact=False,
    )
    if answerer is not None:
        return _legacy_answer_candidates(candidates, task, answerer)[:result_count]

    # Answer once per leading video, not once per returned frame.  This keeps the
    # required top-100 output while reducing the expensive VLM calls by roughly an
    # order of magnitude and gives every judged video a real temporal window.
    by_video: dict[str, list[dict]] = defaultdict(list)
    for row in candidates:
        by_video[row["video_id"]].append(row)
    leading_videos = list(by_video)[: max(0, QA_VLM_TOP_VIDEOS)]
    judgments: dict[str, QaJudgment] = {}
    for video_id in leading_videos:
        strongest = max(by_video[video_id], key=lambda row: float(row["score"]))
        try:
            judgment = _judge_video(task, strongest)
        except (RuntimeError, OSError, ValueError):
            judgment = None
        if judgment is not None:
            judgments[video_id] = judgment

    base_values = [float(row["score"]) for row in candidates]
    low = min(base_values, default=0.0)
    high = max(base_values, default=1.0)
    match_weight = min(max(QA_MATCH_WEIGHT, 0.0), 1.0)
    best_row_by_video: dict[str, dict] = {}
    for video_id, rows in by_video.items():
        judgment = judgments.get(video_id)
        if judgment is None:
            best_row_by_video[video_id] = max(
                rows, key=lambda row: float(row["score"])
            )
            continue
        target = judgment.records[judgment.best_index]
        # Prefer the candidate that already represents the selected panel.  This
        # avoids replacing the strongest row with a frame already present lower
        # in Top 100, which would waste one official submission slot.
        best_row_by_video[video_id] = min(
            rows,
            key=lambda row: (
                abs(int(row["local_idx"]) - target.local_idx),
                -float(row["score"]),
            ),
        )
    results: list[dict] = []
    for row in candidates:
        base_score = (
            (float(row["score"]) - low) / (high - low)
            if high - low > 1e-8
            else 1.0
        )
        judgment = judgments.get(row["video_id"])
        answer = "unknown"
        score = (1.0 - match_weight) * base_score
        frame_id = int(row["frame_id"])
        local_idx = int(row["local_idx"])
        if judgment is not None:
            answer = judgment.answer
            score += match_weight * judgment.match_score
            if row is best_row_by_video[row["video_id"]]:
                best_record = judgment.records[judgment.best_index]
                frame_id = best_record.frame_id
                local_idx = best_record.local_idx
                score += 1e-6
        results.append(
            {
                "video_id": row["video_id"],
                "frame_id": frame_id,
                "local_idx": local_idx,
                "answer": _normalize_answer(
                    answer,
                    uppercase=task.answer_uppercase,
                    no_spaces=task.answer_no_spaces,
                ),
                "score": float(score),
            }
        )
    ranked = cutoff_aware_rank(
        results,
        top_k=result_count,
        video_caps=KIS_VIDEO_CAPS,
    )
    return ranked[:result_count]
