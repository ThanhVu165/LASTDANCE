"""Question-conditioned extraction from frame-local OCR/ASR evidence."""
from __future__ import annotations
import re
from typing import Sequence
from shared.schemas.online import FrameEvidence, AnswerResult
from .fts import FtsSearcher

_NUMBER = re.compile(r"[-+]?\d+(?:[.,]\d+)*")


class FtsVideoAnswerer:
    def __init__(self, registry, modality: str, intents: Sequence[str], *, value_type="free_text"):
        self.registry, self.modality = registry, modality
        self.intents = [x for x in intents if x.strip()]
        self.value_type = value_type

    def _number_candidates(self, hits, question: str):
        # A noun supplied by the question is a constraint, not a video-specific rule.
        noun = re.search(r"(?:bao nhiêu|how many)\s+([^\W\d_]+)", question.casefold())
        known = set(_NUMBER.findall(question))
        candidates = []
        for hit in hits:
            text = str(hit.text)
            for match in _NUMBER.finditer(text):
                if match.group() in known:
                    continue
                if noun and re.match(r"\s*" + re.escape(noun.group(1)) + r"\b", text[match.end():].casefold()) is None:
                    continue
                candidates.append((match.group(), hit))
        return candidates

    def _extract_unknown_value(self, hits, question: str = "") -> str:
        if self.value_type != "number":
            return ""
        candidates = self._number_candidates(hits, question)
        values = {value for value, _hit in candidates}
        return next(iter(values)) if len(values) == 1 else ""

    def answer(self, *, video_id: str, frames: Sequence[FrameEvidence], question: str) -> AnswerResult:
        path = self.registry.layout.ocr if self.modality == "ocr" else self.registry.layout.asr
        uids = []
        for frame in frames:
            uids.append(frame.keyframe_uid)
            uids.extend(f.keyframe_uid for f in self.registry.catalog.neighbors(frame.keyframe_uid, 2))
        # Known literals may locate videos but must not substitute for answer evidence.
        hits = FtsSearcher(path, self.modality).rows_for_uids(uids, video_id=video_id)
        answer = self._extract_unknown_value(hits, question)
        if not answer:
            return AnswerResult(provider=self.modality, warnings=[
                f"No unambiguous {self.value_type} answer in frame-local {self.modality.upper()}; continue verification"])
        candidates = [(value, hit) for value, hit in self._number_candidates(hits, question) if value == answer]
        supported = []
        for _value, hit in candidates:
            known = self.registry.catalog.by_uid.get(hit.keyframe_uid)
            if known is None or known.video_id != video_id:
                continue
            if any(f.keyframe_uid == hit.keyframe_uid for f in supported):
                continue
            supported.append(FrameEvidence(keyframe_uid=known.keyframe_uid, video_id=known.video_id,
                                           frame_id=known.frame_id, pts_time=known.pts_time, shot_id=known.shot_id))
        if not supported:
            return AnswerResult(warnings=["Answer text has no valid catalog evidence"])
        return AnswerResult(answer=answer, value_type=self.value_type, evidence=supported,
                            confidence=0.65, requires_review=True, provider=self.modality,
                            warnings=["Text extraction requires operator verification"])
