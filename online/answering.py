"""Route QA answer extraction to candidate-video OCR/ASR evidence."""

from __future__ import annotations

import re
from typing import Sequence

from shared.schemas.online import FrameEvidence

from .artifacts import ArtifactRegistry
from .fts import FtsSearcher


class FtsVideoAnswerer:
    def __init__(
        self,
        registry: ArtifactRegistry,
        modality: str,
        intents: Sequence[str],
        *,
        value_type: str = "free_text",
    ) -> None:
        self.registry = registry
        self.modality = modality
        self.intents = [item for item in intents if item.strip()]
        self.value_type = value_type

    def _extract_unknown_value(self, hits: Sequence[object]) -> str:
        texts = [re.sub(r"\s+", " ", str(getattr(hit, "text", ""))).strip() for hit in hits]
        if self.value_type == "number":
            candidates: list[tuple[int, int, str]] = []
            for text_index, text in enumerate(texts):
                for value in re.findall(r"\d+(?:[.,]\d+)?", text):
                    digit_count = sum(character.isdigit() for character in value)
                    candidates.append((digit_count, -text_index, value))
            if candidates:
                return max(candidates)[2]
            return ""
        return next((text[:100] for text in texts if text), "")

    def answer(
        self,
        *,
        video_id: str,
        frames: Sequence[FrameEvidence],
        question: str,
    ) -> tuple[str, float, list[str]]:
        path = self.registry.layout.ocr if self.modality == "ocr" else self.registry.layout.asr
        searcher = FtsSearcher(path, self.modality)
        if self.intents:
            hits = searcher.search_hits(
                self.intents,
                limit=20,
                restrict_videos={video_id},
            )
        else:
            uids: list[int] = []
            for frame in frames:
                uids.append(frame.keyframe_uid)
                uids.extend(
                    item.keyframe_uid
                    for item in self.registry.catalog.neighbors(frame.keyframe_uid, 2)
                )
            hits = searcher.rows_for_uids(uids, video_id=video_id)
        if not hits:
            return "Uncertain", 0.0, [f"No {self.modality.upper()} answer evidence in candidate video {video_id}"]
        preferred = {frame.keyframe_uid for frame in frames}
        hits.sort(key=lambda item: (item.keyframe_uid in preferred, item.score), reverse=True)
        answer = self._extract_unknown_value(hits)
        if not answer:
            return "Uncertain", 0.0, [
                f"{self.modality.upper()} text exists but no {self.value_type} answer was extracted in {video_id}"
            ]
        confidence = 0.78 if self.value_type == "number" else 0.65
        return answer[:100], confidence, [
            f"{self.modality.upper()} frame-local answer requires operator verification"
        ]
