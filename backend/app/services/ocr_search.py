"""Read the structured OCR cache and provide error-tolerant text matching."""
from __future__ import annotations

import difflib
import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

from app.config import OCR_CACHE_PATH


@dataclass(frozen=True)
class OcrSearchLine:
    text: str
    confidence: float = 1.0
    box: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class OcrDocument:
    text: str
    lines: tuple[OcrSearchLine, ...]


def _box_from_json(value: Any) -> tuple[tuple[float, float], ...]:
    try:
        points = tuple((float(point[0]), float(point[1])) for point in value)
    except (TypeError, ValueError, IndexError):
        return ()
    return points if len(points) >= 4 else ()


def _document_from_json(value: Any) -> OcrDocument | None:
    # Backward compatibility makes a cache transition safe; new runs always write
    # schema v2 and therefore retain confidence/geometry for downstream ranking.
    if isinstance(value, str):
        text = value.strip()
        lines = tuple(OcrSearchLine(line.strip()) for line in text.splitlines() if line.strip())
        return OcrDocument(text=text, lines=lines)
    if not isinstance(value, dict):
        return None

    raw_lines = value.get("lines", [])
    parsed_lines: list[OcrSearchLine] = []
    if isinstance(raw_lines, list):
        for raw_line in raw_lines:
            if not isinstance(raw_line, dict):
                continue
            text = str(raw_line.get("text", "")).strip()
            if not text:
                continue
            try:
                confidence = min(1.0, max(0.0, float(raw_line.get("confidence", 0.0))))
            except (TypeError, ValueError):
                confidence = 0.0
            parsed_lines.append(
                OcrSearchLine(
                    text=text,
                    confidence=confidence,
                    box=_box_from_json(raw_line.get("box", [])),
                )
            )

    stored_text = value.get("text", "")
    text = str(stored_text).strip() if isinstance(stored_text, str) else ""
    if not parsed_lines and text:
        parsed_lines = [
            OcrSearchLine(line.strip()) for line in text.splitlines() if line.strip()
        ]
    if parsed_lines:
        # Structured lines are authoritative: this avoids stale duplicated flat text.
        text = "\n".join(line.text for line in parsed_lines)
    return OcrDocument(text=text, lines=tuple(parsed_lines))


@lru_cache(maxsize=2)
def _load_ocr_cache_versioned(mtime_ns: int) -> dict[str, OcrDocument]:
    del mtime_ns  # cache-key only
    if not OCR_CACHE_PATH.exists():
        return {}
    try:
        raw = json.loads(OCR_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}

    documents: dict[str, OcrDocument] = {}
    for key, value in raw.items():
        document = _document_from_json(value)
        if document is not None:
            documents[str(key)] = document
    return documents


def _load_ocr_cache() -> dict[str, OcrDocument]:
    """Reload when the offline OCR builder atomically publishes a checkpoint."""
    if not OCR_CACHE_PATH.exists():
        return {}
    return _load_ocr_cache_versioned(OCR_CACHE_PATH.stat().st_mtime_ns)


def _normalize_for_match(text: str) -> str:
    folded = unicodedata.normalize("NFD", text.casefold())
    without_marks = "".join(
        character
        for character in folded
        if unicodedata.category(character) != "Mn"
    ).replace("đ", "d")
    return " ".join(re.findall(r"[a-z0-9]+", without_marks))


def _accepted_token_similarity(query: str, candidate: str) -> float:
    if query == candidate:
        return 1.0
    if min(len(query), len(candidate)) <= 3:
        return 0.0

    similarity = difflib.SequenceMatcher(
        None, query, candidate, autojunk=False
    ).ratio()
    shortest = min(len(query), len(candidate))
    threshold = 0.75 if shortest == 4 else 0.80 if shortest <= 7 else 0.84
    return similarity if similarity >= threshold else 0.0


def _token_match_score(query: str, document_tokens: Sequence[str]) -> float:
    return max(
        (_accepted_token_similarity(query, candidate) for candidate in document_tokens),
        default=0.0,
    )


def _phrase_match_score(query_tokens: Sequence[str], document_tokens: Sequence[str]) -> float:
    if not query_tokens or not document_tokens:
        return 0.0
    target = " ".join(query_tokens)
    widths = {
        width
        for width in (len(query_tokens) - 1, len(query_tokens), len(query_tokens) + 1)
        if 1 <= width <= len(document_tokens)
    }
    best = 0.0
    for width in widths:
        for start in range(0, len(document_tokens) - width + 1):
            candidate = " ".join(document_tokens[start : start + width])
            similarity = difflib.SequenceMatcher(
                None, target, candidate, autojunk=False
            ).ratio()
            best = max(best, similarity)
    # A long phrase has enough evidence to tolerate a few OCR substitutions or a
    # merged/split token, while still requiring strong alignment as a whole.
    return best if best >= 0.80 else 0.0


def _keyword_match_score(keyword: str, document_tokens: Sequence[str]) -> float:
    query_tokens = _normalize_for_match(keyword).split()
    if not query_tokens:
        return 0.0
    if len(query_tokens) == 1:
        return _token_match_score(query_tokens[0], document_tokens)
    return _phrase_match_score(query_tokens, document_tokens)


def _document_keyword_score(keyword: str, document: OcrDocument) -> float:
    query_token_count = len(_normalize_for_match(keyword).split())
    line_scores = [
        _keyword_match_score(keyword, _normalize_for_match(line.text).split())
        # A multi-word phrase already carries strong evidence and EasyOCR often
        # underestimates confidence for all-caps Vietnamese signs. For a lone word,
        # confidence still suppresses chance matches in noisy corner graphics.
        * (1.0 if query_token_count > 1 else 0.5 + 0.5 * line.confidence)
        for line in document.lines
    ]
    # A phrase can be split across adjacent OCR boxes. Keep a conservative fallback
    # over the flat document while line confidence remains the primary evidence.
    whole_document_score = 0.75 * _keyword_match_score(
        keyword, _normalize_for_match(document.text).split()
    )
    return max([whole_document_score, *line_scores], default=0.0)


def ocr_match_score(video_id: str, local_idx: int, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    document = _load_ocr_cache().get(f"{video_id}:{local_idx}")
    if document is None or not document.text:
        return 0.0
    normalized_keywords = list(
        dict.fromkeys(
            normalized
            for keyword in keywords
            if (normalized := _normalize_for_match(keyword))
        )
    )
    if not normalized_keywords:
        return 0.0
    return sum(
        _document_keyword_score(keyword, document)
        for keyword in normalized_keywords
    ) / len(normalized_keywords)
