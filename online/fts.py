"""Unified SQLite FTS5 search for OCR and ASR artifacts."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable


_TOKEN = re.compile(r"[^\W_]+", flags=re.UNICODE)


@dataclass(frozen=True, slots=True)
class FtsHit:
    keyframe_uid: int
    video_id: str
    text: str
    score: float


class FtsSearcher:
    _CONTRACT = {
        "ocr": ("ocr_fts", "detected_text", "keyframe_uid"),
        "asr": ("asr_fts", "transcribed_text", "keyframe_uid_nearest"),
    }

    def __init__(self, path: Path, modality: str) -> None:
        if modality not in self._CONTRACT:
            raise ValueError("FTS modality must be ocr or asr")
        self.path = Path(path)
        self.modality = modality
        self.table, self.content_column, self.uid_column = self._CONTRACT[modality]

    def _connection(self) -> sqlite3.Connection:
        if not self.path.is_file():
            raise RuntimeError(f"{self.modality} artifact is unavailable")
        return sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True)

    @staticmethod
    def _tokens(texts: Iterable[str]) -> list[str]:
        result: list[str] = []
        for text in texts:
            for token in _TOKEN.findall(text.lower()):
                if len(token) > 1 and token not in result:
                    result.append(token)
        return result[:20]

    def search(
        self,
        texts: Iterable[str],
        *,
        limit: int = 1000,
        restrict_videos: set[str] | None = None,
    ) -> dict[int, float]:
        return {
            hit.keyframe_uid: hit.score
            for hit in self.search_hits(texts, limit=limit, restrict_videos=restrict_videos)
        }

    def search_hits(
        self,
        texts: Iterable[str],
        *,
        limit: int = 1000,
        restrict_videos: set[str] | None = None,
    ) -> list[FtsHit]:
        phrases = [text.strip() for text in texts if text.strip()]
        tokens = self._tokens(phrases)
        if not phrases or not tokens:
            return []
        exact = " OR ".join(f'"{phrase.replace(chr(34), chr(34) * 2)}"' for phrase in phrases)
        full = " AND ".join(f'"{token}"' for token in tokens)
        prefix = " OR ".join(f'"{token}"*' for token in tokens)
        queries = [(exact, 1.0, "exact"), (full, 0.85, "full"), (prefix, 0.65, "prefix")]
        scores: dict[int, float] = {}
        fuzzy_rows: dict[int, str] = {}
        row_videos: dict[int, str] = {}
        connection = self._connection()
        try:
            for query, boost, stage in queries:
                restrictions = ""
                parameters: list[object] = [query]
                if restrict_videos:
                    placeholders = ",".join("?" for _ in restrict_videos)
                    restrictions = f" AND video_id IN ({placeholders})"
                    parameters.extend(sorted(restrict_videos))
                sql = (
                    f"SELECT {self.uid_column}, {self.content_column}, -bm25({self.table}) AS relevance, video_id "
                    f"FROM {self.table} WHERE {self.table} MATCH ?{restrictions} "
                    f"ORDER BY bm25({self.table}) LIMIT ?"
                )
                stage_limit = 5000 if stage == "prefix" else limit
                parameters.append(stage_limit)
                try:
                    rows = connection.execute(sql, parameters).fetchall()
                except sqlite3.OperationalError:
                    continue
                denominator = max(len(rows) - 1, 1)
                for rank, (uid, content, _relevance, video_id) in enumerate(rows):
                    if restrict_videos is not None and str(video_id) not in restrict_videos:
                        continue
                    identifier = int(uid)
                    # Scores from separate MATCH expressions are not numerically
                    # comparable. Preserve the baseline cascade (exact > AND >
                    # prefix) and use BM25 only through its within-stage rank.
                    rank_factor = 1.0 - 0.1 * (rank / denominator)
                    coverage_factor = 1.0
                    if stage == "prefix":
                        content_tokens = set(self._tokens([str(content or "")]))
                        matched = sum(
                            any(candidate.startswith(token) for candidate in content_tokens)
                            for token in tokens
                        )
                        coverage_factor = 0.5 + 0.5 * (matched / len(tokens))
                    value = boost * coverage_factor * rank_factor
                    scores[identifier] = max(scores.get(identifier, 0.0), value)
                    fuzzy_rows.setdefault(identifier, str(content or ""))
                    row_videos.setdefault(identifier, str(video_id))
        finally:
            connection.close()

        needle = " ".join(tokens)
        for uid, content in list(fuzzy_rows.items())[:500]:
            ratio = SequenceMatcher(None, needle, content.lower()).ratio()
            if ratio >= 0.55:
                scores[uid] = max(scores.get(uid, 0.0), 0.4 * ratio)
        if not scores:
            return []
        identifiers = sorted(scores, key=scores.get, reverse=True)[:limit]
        return [
            FtsHit(uid, row_videos[uid], fuzzy_rows[uid], scores[uid])
            for uid in identifiers
            if uid in row_videos
        ]
