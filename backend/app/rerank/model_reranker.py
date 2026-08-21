"""Dedicated multimodal cross-encoder verification for KIS video hypotheses."""
from __future__ import annotations

import gc
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from app.config import (
    KEYFRAMES_DIR,
    MODEL_RERANK_BATCH_SIZE,
    MODEL_RERANK_CONFIDENCE_THRESHOLD,
    MODEL_RERANK_DEVICE,
    MODEL_RERANK_ENABLED,
    MODEL_RERANK_FRAMES_PER_VIDEO,
    MODEL_GENERATIVE_VERIFY_GROUP_SIZE,
    MODEL_GENERATIVE_VERIFY_MAX_NEW_TOKENS,
    MODEL_RERANK_LOCAL_FILES_ONLY,
    MODEL_RERANK_NAME,
    MODEL_RERANK_TOP_VIDEOS,
    MODEL_RERANK_WEIGHT,
    ROOT_DIR,
)
from app.services.visual_qa import release_vqa
from app.services.visual_qa import generate_images


@dataclass(frozen=True)
class VerificationReport:
    available: bool
    verified_videos: int = 0
    verified_candidates: int = 0
    high_confidence_videos: int = 0
    max_score: float = 0.0
    error: str | None = None
    backend: str | None = None


_LAST_STATUS: dict[str, object] = {
    "loaded": False,
    "last_error": None,
    "last_verified_videos": 0,
}


def _normal_device(device: str) -> str:
    normalized = device.strip().lower()
    return "cuda" + normalized[3:] if normalized.startswith("gpu") else normalized


def _dedicated_weights_ready() -> bool:
    configured = Path(MODEL_RERANK_NAME)
    if configured.exists():
        return (configured / "model.safetensors").exists()
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(MODEL_RERANK_NAME, "model.safetensors")
    except (ImportError, OSError, ValueError):
        return False
    return isinstance(cached, str) and Path(cached).is_file()


@lru_cache(maxsize=1)
def _load_reranker():
    if MODEL_RERANK_LOCAL_FILES_ONLY and not _dedicated_weights_ready():
        raise FileNotFoundError(
            f"Dedicated reranker weights are not complete in the local cache: "
            f"{MODEL_RERANK_NAME}"
        )
    import torch
    from sentence_transformers import CrossEncoder

    device = _normal_device(MODEL_RERANK_DEVICE)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Model reranker requested {device}, but CUDA is unavailable.")
    release_vqa()
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = CrossEncoder(
        MODEL_RERANK_NAME,
        device=device,
        local_files_only=MODEL_RERANK_LOCAL_FILES_ONLY,
        model_kwargs={
            "dtype": dtype,
            "low_cpu_mem_usage": True,
            "attn_implementation": "sdpa",
        },
        trust_remote_code=True,
    )
    _LAST_STATUS.update(loaded=True, last_error=None)
    return model


def release_model_reranker() -> None:
    _load_reranker.cache_clear()
    _LAST_STATUS["loaded"] = False
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def model_reranker_status() -> dict[str, object]:
    return dict(_LAST_STATUS)


def _keyframe_path(row: dict) -> Path:
    provided = Path(str(row.get("keyframe_path", "")))
    if provided.exists():
        return provided
    local_idx = int(row["local_idx"])
    for width in (3, 4):
        candidate = KEYFRAMES_DIR / row["video_id"] / f"{local_idx:0{width}d}.jpg"
        if candidate.exists():
            return candidate
    return KEYFRAMES_DIR / row["video_id"] / f"{local_idx:03d}.jpg"


def _representative_rows(rows: Sequence[dict], limit: int) -> list[dict]:
    by_local = {int(row["local_idx"]): row for row in rows}
    chosen: list[dict] = []
    story = max(
        rows,
        key=lambda row: (float(row.get("query_coverage", 0.0)), float(row["score"])),
    ).get("storyboard_local_idxs", [])
    for local_idx in story:
        row = by_local.get(int(local_idx))
        if row is not None and row not in chosen:
            chosen.append(row)
    for row in sorted(rows, key=lambda item: float(item["score"]), reverse=True):
        if row not in chosen:
            chosen.append(row)
        if len(chosen) >= limit:
            break
    return sorted(chosen[:limit], key=lambda row: int(row["local_idx"]))


def _sheet(rows: Sequence[dict], output: Path) -> Path:
    cell_w, cell_h = 336, 224
    columns = min(2, max(1, len(rows)))
    lines = (len(rows) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * cell_w, lines * cell_h), "black")
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(rows):
        path = _keyframe_path(row)
        with Image.open(path) as source:
            image = ImageOps.fit(source.convert("RGB"), (cell_w, cell_h))
        left = (index % columns) * cell_w
        top = (index // columns) * cell_h
        canvas.paste(image, (left, top))
        draw.rectangle((left + 4, top + 4, left + 52, top + 38), fill="black")
        draw.text((left + 15, top + 10), str(index + 1), fill="white")
    canvas.save(output, quality=90)
    return output


def _comparison_sheet(
    grouped_rows: Sequence[Sequence[dict]],
    output: Path,
) -> Path:
    """Put one candidate video per row for bounded listwise verification."""
    cell_w, cell_h = 280, 176
    columns = max(len(rows) for rows in grouped_rows)
    canvas = Image.new("RGB", (columns * cell_w, len(grouped_rows) * cell_h), "black")
    draw = ImageDraw.Draw(canvas)
    for video_index, rows in enumerate(grouped_rows):
        for panel_index, row in enumerate(rows):
            with Image.open(_keyframe_path(row)) as source:
                image = ImageOps.fit(source.convert("RGB"), (cell_w, cell_h))
            left, top = panel_index * cell_w, video_index * cell_h
            canvas.paste(image, (left, top))
            label = f"V{video_index + 1}-P{panel_index + 1}"
            draw.rectangle((left + 4, top + 4, left + 94, top + 34), fill="black")
            draw.text((left + 10, top + 9), label, fill="white")
    canvas.save(output, quality=90)
    return output


_ROW_SCORE = re.compile(r"\bV([0-9]+)\s*[:=]\s*(100|[0-9]{1,2})\b", re.I)


def _parse_row_scores(text: str, count: int) -> list[float] | None:
    parsed: dict[int, float] = {}
    for raw_index, raw_score in _ROW_SCORE.findall(text):
        index = int(raw_index) - 1
        if 0 <= index < count:
            parsed[index] = min(max(int(raw_score), 0), 100) / 100.0
    if len(parsed) != count:
        return None
    return [parsed[index] for index in range(count)]


def _minmax(rows: Sequence[dict]) -> dict[int, float]:
    values = [float(row["score"]) for row in rows]
    low, high = min(values, default=0.0), max(values, default=1.0)
    return {
        id(row): ((float(row["score"]) - low) / (high - low) if high - low > 1e-8 else 1.0)
        for row in rows
    }


def rerank_kis_with_model(
    query: str,
    candidates: list[dict],
) -> tuple[list[dict], VerificationReport]:
    if not MODEL_RERANK_ENABLED or not candidates:
        return candidates, VerificationReport(
            available=False, error="disabled", backend="dedicated"
        )

    by_video: dict[str, list[dict]] = defaultdict(list)
    for row in candidates:
        by_video[row["video_id"]].append(row)
    video_ids = sorted(
        by_video,
        key=lambda video_id: (
            max(float(row.get("query_coverage", 0.0)) for row in by_video[video_id]),
            max(float(row["score"]) for row in by_video[video_id]),
        ),
        reverse=True,
    )[: max(1, MODEL_RERANK_TOP_VIDEOS)]

    temp_root = ROOT_DIR / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    try:
        model = _load_reranker()
        with tempfile.TemporaryDirectory(prefix="model-rerank-", dir=temp_root) as temp:
            folder = Path(temp)
            documents: list[str] = []
            scored_video_ids: list[str] = []
            for video_id in video_ids:
                rows = _representative_rows(
                    by_video[video_id], max(1, MODEL_RERANK_FRAMES_PER_VIDEO)
                )
                if not rows or any(not _keyframe_path(row).exists() for row in rows):
                    continue
                documents.append(str(_sheet(rows, folder / f"{video_id}.jpg")))
                scored_video_ids.append(video_id)
            if not documents:
                return candidates, VerificationReport(
                    available=False, error="no_images", backend="dedicated"
                )

            import torch

            scores = model.predict(
                [(query, document) for document in documents],
                prompt=(
                    "Retrieve video evidence that satisfies the complete query. All "
                    "described entities, actions, attributes, relations, visible text "
                    "and temporal scenes are jointly required; reject partial matches."
                ),
                batch_size=max(1, MODEL_RERANK_BATCH_SIZE),
                show_progress_bar=False,
                activation_fn=torch.nn.Sigmoid(),
            )
        score_values = np.asarray(scores, dtype=np.float32).reshape(-1).tolist()
    except (RuntimeError, OSError, ValueError, ImportError) as exc:
        _LAST_STATUS.update(loaded=False, last_error=f"{type(exc).__name__}: {exc}")
        release_model_reranker()
        return candidates, VerificationReport(
            available=False, error=str(exc), backend="dedicated"
        )

    relevance = dict(zip(scored_video_ids, score_values))
    base = _minmax(candidates)
    weight = min(max(MODEL_RERANK_WEIGHT, 0.0), 1.0)
    reranked: list[dict] = []
    for row in candidates:
        model_score = relevance.get(row["video_id"])
        if model_score is None:
            reranked.append({**row, "model_verified": False})
            continue
        reranked.append(
            {
                **row,
                "retrieval_score": float(row["score"]),
                "model_verified": True,
                "model_relevance_score": float(model_score),
                "score": (
                    (1.0 - weight) * base[id(row)]
                    + weight * float(model_score)
                    + 0.01 * base[id(row)]
                ),
            }
        )
    high = sum(score >= MODEL_RERANK_CONFIDENCE_THRESHOLD for score in relevance.values())
    _LAST_STATUS.update(
        loaded=True,
        last_error=None,
        last_verified_videos=len(relevance),
    )
    return reranked, VerificationReport(
        available=True,
        verified_videos=len(relevance),
        verified_candidates=sum(
            row["video_id"] in relevance for row in candidates
        ),
        high_confidence_videos=high,
        max_score=max(relevance.values(), default=0.0),
        backend="dedicated",
    )


def rerank_kis_with_generative_model(
    query: str,
    candidates: list[dict],
) -> tuple[list[dict], VerificationReport]:
    """Score a broad video pool with the cached instruction VLM.

    This is the immediate, no-download model-first verifier for a 6 GiB GPU. It
    examines every selected video, unlike the older knockout tournament which
    only promoted one winner and left the rest unverified.
    """
    if not MODEL_RERANK_ENABLED or not candidates:
        return candidates, VerificationReport(
            available=False, error="disabled", backend="generative"
        )

    by_video: dict[str, list[dict]] = defaultdict(list)
    for row in candidates:
        by_video[row["video_id"]].append(row)
    video_ids = sorted(
        by_video,
        key=lambda video_id: (
            max(float(row.get("query_coverage", 0.0)) for row in by_video[video_id]),
            max(float(row["score"]) for row in by_video[video_id]),
        ),
        reverse=True,
    )[: max(1, MODEL_RERANK_TOP_VIDEOS)]

    representatives: dict[str, list[dict]] = {}
    for video_id in video_ids:
        rows = _representative_rows(
            by_video[video_id], max(1, MODEL_RERANK_FRAMES_PER_VIDEO)
        )
        if rows and all(_keyframe_path(row).exists() for row in rows):
            representatives[video_id] = rows
    video_ids = [video_id for video_id in video_ids if video_id in representatives]
    if not video_ids:
        return candidates, VerificationReport(
            available=False, error="no_images", backend="generative"
        )

    group_size = max(1, MODEL_GENERATIVE_VERIFY_GROUP_SIZE)
    relevance: dict[str, float] = {}
    temp_root = ROOT_DIR / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    prompt = (
        "Each ROW V1, V2, ... is a different candidate video; panels in a row are "
        "chronological evidence from that same video. Independently score EVERY row "
        "from 0 to 100 for satisfying the COMPLETE query. Require the joint "
        "combination of entities, actions, counts, attributes, relations, setting, "
        "visible text and ordered scenes. A partial match missing a distinctive "
        "condition must score below 50. Use only visible evidence. Reply on one line "
        "with every row exactly as V1=<score>;V2=<score>;... and no explanation.\n"
        f"Complete query and required evidence:\n{query}"
    )
    try:
        with tempfile.TemporaryDirectory(prefix="generative-verify-", dir=temp_root) as temp:
            folder = Path(temp)
            for start in range(0, len(video_ids), group_size):
                group = video_ids[start : start + group_size]
                sheet = _comparison_sheet(
                    [representatives[video_id] for video_id in group],
                    folder / f"group-{start // group_size:03d}.jpg",
                )
                decoded = generate_images(
                    (str(sheet),),
                    prompt,
                    max_new_tokens=max(32, MODEL_GENERATIVE_VERIFY_MAX_NEW_TOKENS),
                )
                scores = _parse_row_scores(decoded, len(group))
                if scores is None:
                    continue
                relevance.update(zip(group, scores))
    except (RuntimeError, OSError, ValueError, ImportError) as exc:
        _LAST_STATUS.update(loaded=False, last_error=f"{type(exc).__name__}: {exc}")
        return candidates, VerificationReport(
            available=False, error=str(exc), backend="generative"
        )

    if not relevance:
        return candidates, VerificationReport(
            available=False,
            error="model returned no complete row-score groups",
            backend="generative",
        )
    base = _minmax(candidates)
    weight = min(max(MODEL_RERANK_WEIGHT, 0.0), 1.0)
    reranked: list[dict] = []
    for row in candidates:
        model_score = relevance.get(row["video_id"])
        if model_score is None:
            reranked.append({**row, "model_verified": False})
            continue
        reranked.append(
            {
                **row,
                "retrieval_score": float(row["score"]),
                "model_verified": True,
                "model_relevance_score": float(model_score),
                "score": (1.0 - weight) * base[id(row)] + weight * model_score,
            }
        )
    high = sum(score >= MODEL_RERANK_CONFIDENCE_THRESHOLD for score in relevance.values())
    _LAST_STATUS.update(
        loaded=True,
        last_error=None,
        last_verified_videos=len(relevance),
        backend="generative",
    )
    return reranked, VerificationReport(
        available=True,
        verified_videos=len(relevance),
        verified_candidates=sum(
            row["video_id"] in relevance for row in candidates
        ),
        high_confidence_videos=high,
        max_score=max(relevance.values(), default=0.0),
        backend="generative",
    )
