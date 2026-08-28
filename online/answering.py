"""Route QA answer extraction to candidate-video OCR/ASR evidence."""

from __future__ import annotations

import re
from typing import Sequence

from shared.schemas.online import FrameEvidence

from .artifacts import ArtifactRegistry
from .fts import FtsSearcher


class FtsVideoAnswerer:
    def __init__(self, registry: ArtifactRegistry, modality: str, intents: Sequence[str]) -> None:
        self.registry = registry
        self.modality = modality
        self.intents = [item for item in intents if item.strip()]

    def answer(
        self,
        *,
        video_id: str,
        frames: Sequence[FrameEvidence],
        question: str,
    ) -> tuple[str, float, list[str]]:
        path = self.registry.layout.ocr if self.modality == "ocr" else self.registry.layout.asr
        queries = self.intents or [question]
        hits = FtsSearcher(path, self.modality).search_hits(
            queries,
            limit=20,
            restrict_videos={video_id},
        )
        if not hits:
            return "Uncertain", 0.0, [f"No {self.modality.upper()} answer evidence in candidate video {video_id}"]
        preferred = {frame.keyframe_uid for frame in frames}
        hits.sort(key=lambda item: (item.keyframe_uid in preferred, item.score), reverse=True)
        answer = re.sub(r"\s+", " ", hits[0].text).strip()[:100]
        if not answer:
            return "Uncertain", 0.0, [f"Empty {self.modality.upper()} answer evidence in {video_id}"]
        return answer, 0.8, [f"{self.modality.upper()} answer requires operator verification"]
