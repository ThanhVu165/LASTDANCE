"""Build a structured keyframe OCR cache with EasyOCR.

The production recognizer is EasyOCR ``latin_g2`` behind the CRAFT detector.
That recognizer can emit the complete precomposed Vietnamese alphabet.  Model
initialization verifies that invariant before an existing cache is modified.

Each successful cache row stores ordered text lines, confidence, and geometry.
Failures are kept in a separate state file and retried; they are never confused
with an image which was processed successfully and genuinely contains no text.

Run after ``build_index.py``:
    python -m app.indexing.ocr_index

Useful smoke test:
    python -m app.indexing.ocr_index --limit 20 --checkpoint-every 5
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from app.config import (
    KEYFRAME_INDEX_PATH,
    OCR_CACHE_PATH,
    OCR_CHECKPOINT_EVERY,
    OCR_DETECTION_MODEL_NAME,
    OCR_DEVICE,
    OCR_INPUT_BATCH_SIZE,
    OCR_LANGUAGES,
    OCR_LINK_THRESHOLD,
    OCR_LOW_TEXT,
    OCR_MAGNIFICATION,
    OCR_MAX_RETRIES,
    OCR_MAX_SIDE,
    OCR_MIN_CONFIDENCE,
    OCR_MIN_TEXT_SIZE,
    OCR_MODEL_DIR,
    OCR_RECOGNITION_BATCH_SIZE,
    OCR_RECOGNITION_MODEL_NAME,
    OCR_STATE_PATH,
    OCR_TEXT_THRESHOLD,
)

CACHE_SCHEMA_VERSION = 2
OCR_ENGINE_VERSION = "easyocr-1.7.2"
TERMINAL_STATUSES = {"success", "no_text"}

# The model must support every precomposed character, not merely ASCII text with
# accents removed at search time.  This is deliberately an executable invariant.
VIETNAMESE_LOWERCASE = (
    "aăâbcdđeêghiklmnoôơpqrstuưvxy"
    "àáạảãằắặẳẵầấậẩẫ"
    "èéẹẻẽềếệểễ"
    "ìíịỉĩ"
    "òóọỏõồốộổỗờớợởỡ"
    "ùúụủũừứựửữ"
    "ỳýỵỷỹ"
)
REQUIRED_VIETNAMESE_CHARACTERS = frozenset(
    VIETNAMESE_LOWERCASE + VIETNAMESE_LOWERCASE.upper()
)


@dataclass(frozen=True)
class OcrItem:
    key: str
    image_path: str


@dataclass(frozen=True)
class OcrLine:
    text: str
    confidence: float
    box: tuple[tuple[float, float], ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 6),
            "box": [[round(x, 2), round(y, 2)] for x, y in self.box],
        }


@dataclass(frozen=True)
class OcrOutcome:
    key: str
    lines: tuple[OcrLine, ...] = ()
    error: str = ""

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    def cache_entry(self) -> dict[str, Any]:
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "text": self.text,
            "lines": [line.to_json() for line in self.lines],
        }


def _model_signature() -> str:
    languages = ",".join(OCR_LANGUAGES)
    return (
        f"{OCR_ENGINE_VERSION}:{OCR_DETECTION_MODEL_NAME}:"
        f"{OCR_RECOGNITION_MODEL_NAME}:languages={languages}:"
        f"confidence={OCR_MIN_CONFIDENCE}:max_side={OCR_MAX_SIDE}:"
        f"mag={OCR_MAGNIFICATION}:min_size={OCR_MIN_TEXT_SIZE}:"
        f"text_threshold={OCR_TEXT_THRESHOLD}:low_text={OCR_LOW_TEXT}:"
        f"link_threshold={OCR_LINK_THRESHOLD}:schema={CACHE_SCHEMA_VERSION}"
    )


def _load_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}, got {type(value).__name__}.")
    return value


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    """Publish a checkpoint without leaving a half-written JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _normalize_line(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).strip().split())


def _coerce_box(raw_box: Any) -> tuple[tuple[float, float], ...]:
    try:
        points = tuple((float(point[0]), float(point[1])) for point in raw_box)
    except (TypeError, ValueError, IndexError):
        return ()
    return points if len(points) >= 4 else ()


def _extract_lines(
    result: Any,
    min_confidence: float = OCR_MIN_CONFIDENCE,
) -> tuple[OcrLine, ...]:
    """Convert EasyOCR rows to normalized, confidence-filtered structured lines."""
    if result is None:
        return ()
    if not isinstance(result, (list, tuple)):
        raise TypeError(f"Unexpected EasyOCR result type: {type(result).__name__}")

    lines: list[OcrLine] = []
    for row in result:
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            continue
        text = _normalize_line(str(row[1]))
        if not text:
            continue
        try:
            confidence = float(row[2])
        except (TypeError, ValueError):
            continue
        if confidence < min_confidence:
            continue
        lines.append(
            OcrLine(
                text=text,
                confidence=confidence,
                box=_coerce_box(row[0]),
            )
        )
    return tuple(lines)


def _validate_vietnamese_alphabet(reader: Any) -> None:
    available = unicodedata.normalize(
        "NFC",
        f"{getattr(reader, 'lang_char', '')}{getattr(reader, 'character', '')}",
    )
    missing = sorted(REQUIRED_VIETNAMESE_CHARACTERS.difference(available))
    if missing:
        sample = "".join(missing[:30])
        suffix = "…" if len(missing) > 30 else ""
        raise RuntimeError(
            f"OCR recognizer is missing {len(missing)} Vietnamese characters "
            f"({sample}{suffix}). Refusing to build a lossy OCR cache."
        )


def _torch_device(device: str) -> str:
    normalized = device.strip().lower()
    if normalized.startswith("gpu"):
        normalized = "cuda" + normalized[3:]
    return normalized


def _create_pipeline(device: str):
    try:
        import easyocr
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "EasyOCR/PyTorch is not installed. Follow the OCR installation section "
            "in backend/README.md."
        ) from exc

    torch_device = _torch_device(device)
    if torch_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested {torch_device}, but this PyTorch build cannot access CUDA."
        )

    OCR_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    reader = easyocr.Reader(
        list(OCR_LANGUAGES),
        gpu=torch_device,
        model_storage_directory=str(OCR_MODEL_DIR),
        download_enabled=True,
        detector=True,
        recognizer=True,
        detect_network=OCR_DETECTION_MODEL_NAME,
        recog_network=OCR_RECOGNITION_MODEL_NAME,
        verbose=True,
    )
    _validate_vietnamese_alphabet(reader)
    return reader


def _clear_cuda_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _predict_batch(pipeline: Any, items: Sequence[OcrItem]) -> list[OcrOutcome]:
    """Predict a batch and isolate a bad/OOM image if the batch path fails."""
    if not items:
        return []
    try:
        results = pipeline.readtext_batched(
            [item.image_path for item in items],
            decoder="greedy",
            batch_size=OCR_RECOGNITION_BATCH_SIZE,
            workers=0,
            detail=1,
            paragraph=False,
            min_size=OCR_MIN_TEXT_SIZE,
            text_threshold=OCR_TEXT_THRESHOLD,
            low_text=OCR_LOW_TEXT,
            link_threshold=OCR_LINK_THRESHOLD,
            canvas_size=OCR_MAX_SIDE,
            mag_ratio=OCR_MAGNIFICATION,
        )
        if len(results) != len(items):
            raise RuntimeError(
                f"EasyOCR returned {len(results)} results for {len(items)} images."
            )
        return [
            OcrOutcome(key=item.key, lines=_extract_lines(result))
            for item, result in zip(items, results)
        ]
    except Exception as exc:
        if len(items) == 1:
            message = f"{type(exc).__name__}: {exc}"[:1000]
            return [OcrOutcome(key=items[0].key, error=message)]

        _clear_cuda_cache()
        print(
            f"OCR batch size {len(items)} failed ({type(exc).__name__}: {exc}); "
            "retrying as smaller batches.",
            file=sys.stderr,
            flush=True,
        )
        midpoint = len(items) // 2
        return _predict_batch(pipeline, items[:midpoint]) + _predict_batch(
            pipeline, items[midpoint:]
        )


def _is_complete(state: dict[str, Any], key: str, signature: str) -> bool:
    row = state.get(key)
    return (
        isinstance(row, dict)
        and row.get("status") in TERMINAL_STATUSES
        and row.get("signature") == signature
    )


def _attempts(state: dict[str, Any], key: str, signature: str) -> int:
    row = state.get(key)
    if not isinstance(row, dict) or row.get("signature") != signature:
        return 0
    try:
        return int(row.get("attempts", 0))
    except (TypeError, ValueError):
        return 0


def _chunks(items: Sequence[OcrItem], size: int) -> Iterable[Sequence[OcrItem]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _save_checkpoint(cache: dict[str, Any], state: dict[str, Any]) -> None:
    _atomic_write_json(OCR_CACHE_PATH, cache)
    _atomic_write_json(OCR_STATE_PATH, state)


def _build_todo(
    keyframes: list[dict[str, Any]],
    state: dict[str, Any],
    signature: str,
    retry_failed: bool,
) -> tuple[list[OcrItem], int]:
    todo: list[OcrItem] = []
    exhausted = 0
    for row in keyframes:
        key = f"{row['video_id']}:{row['local_idx']}"
        if _is_complete(state, key, signature):
            continue
        if not retry_failed and _attempts(state, key, signature) >= OCR_MAX_RETRIES:
            exhausted += 1
            continue
        todo.append(OcrItem(key=key, image_path=str(row["keyframe_path"])))
    return todo, exhausted


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default=OCR_DEVICE,
        help="PyTorch inference device, e.g. cuda:0 or cpu (default: %(default)s).",
    )
    parser.add_argument(
        "--input-batch-size",
        type=int,
        default=OCR_INPUT_BATCH_SIZE,
        help="Number of keyframes passed to each detector call (default: %(default)s).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=OCR_CHECKPOINT_EVERY,
        help="Persist cache/state after this many results (default: %(default)s).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N pending images; useful for a smoke test.",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=1,
        help=(
            "Take every Nth pending image before --limit; useful for a smoke test "
            "spread across the collection (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry errors even after they reached AIC_OCR_MAX_RETRIES.",
    )
    args = parser.parse_args(argv)
    if args.input_batch_size < 1:
        parser.error("--input-batch-size must be at least 1")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.sample_stride < 1:
        parser.error("--sample-stride must be at least 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not KEYFRAME_INDEX_PATH.exists():
        raise FileNotFoundError("Missing keyframe_index.json. Run build_index.py first.")

    keyframes = json.loads(KEYFRAME_INDEX_PATH.read_text(encoding="utf-8"))
    if not isinstance(keyframes, list):
        raise RuntimeError(f"Expected a JSON array in {KEYFRAME_INDEX_PATH}.")
    cache = _load_json_dict(OCR_CACHE_PATH)
    state = _load_json_dict(OCR_STATE_PATH)
    signature = _model_signature()
    todo, exhausted_before_run = _build_todo(
        keyframes, state, signature, args.retry_failed
    )
    todo = todo[:: args.sample_stride]
    if args.limit is not None:
        todo = todo[: args.limit]

    completed = sum(
        _is_complete(state, f"{row['video_id']}:{row['local_idx']}", signature)
        for row in keyframes
    )
    print(
        f"OCR model: {OCR_DETECTION_MODEL_NAME} + {OCR_RECOGNITION_MODEL_NAME}",
        flush=True,
    )
    print(f"Device: {_torch_device(args.device)}; signature: {signature}", flush=True)
    print(
        f"Total: {len(keyframes)}, complete for this model: {completed}, "
        f"pending this run: {len(todo)}, exhausted errors: {exhausted_before_run}",
        flush=True,
    )
    if not todo:
        return 1 if exhausted_before_run else 0

    # Construction validates the dependencies, GPU and Vietnamese alphabet before
    # any cache rows from the preceding model are removed.
    try:
        pipeline = _create_pipeline(args.device)
    except Exception as exc:
        print(f"Cannot initialize EasyOCR: {exc}", file=sys.stderr, flush=True)
        return 2

    stale_keys = [key for key in cache if not _is_complete(state, key, signature)]
    for key in stale_keys:
        cache.pop(key, None)

    processed = success = no_text = failed = 0
    since_checkpoint = 0
    started_at = time.perf_counter()
    for batch in _chunks(todo, args.input_batch_size):
        for outcome in _predict_batch(pipeline, batch):
            attempts = _attempts(state, outcome.key, signature) + 1
            if outcome.error:
                failed += 1
                cache.pop(outcome.key, None)
                state[outcome.key] = {
                    "status": "error",
                    "attempts": attempts,
                    "signature": signature,
                    "error": outcome.error,
                }
            else:
                cache[outcome.key] = outcome.cache_entry()
                status = "success" if outcome.lines else "no_text"
                state[outcome.key] = {
                    "status": status,
                    "attempts": attempts,
                    "signature": signature,
                }
                if outcome.lines:
                    success += 1
                else:
                    no_text += 1

            processed += 1
            since_checkpoint += 1
            if since_checkpoint >= args.checkpoint_every:
                _save_checkpoint(cache, state)
                since_checkpoint = 0
                elapsed = max(time.perf_counter() - started_at, 1e-9)
                rate = processed / elapsed
                eta_seconds = (len(todo) - processed) / max(rate, 1e-9)
                print(
                    f"OCR progress: {processed}/{len(todo)}; text={success}, "
                    f"no_text={no_text}, errors={failed}; rate={rate:.2f} image/s, "
                    f"ETA={eta_seconds / 3600:.2f}h",
                    flush=True,
                )

    _save_checkpoint(cache, state)
    elapsed = max(time.perf_counter() - started_at, 1e-9)
    remaining, exhausted_after_run = _build_todo(
        keyframes, state, signature, retry_failed=False
    )
    print(
        f"OCR run complete: processed={processed}, text={success}, no_text={no_text}, "
        f"errors={failed}, retryable_remaining={len(remaining)}, "
        f"exhausted_errors={exhausted_after_run}, elapsed={elapsed:.1f}s, "
        f"rate={processed / elapsed:.2f} image/s",
        flush=True,
    )
    return 1 if failed or exhausted_after_run else 0


if __name__ == "__main__":
    raise SystemExit(main())
