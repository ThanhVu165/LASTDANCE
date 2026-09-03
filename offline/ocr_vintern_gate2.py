"""Pinned Vintern router and output guards for the layered OCR pilot.

The architecture now includes this stage after EasyOCR, but router v2 remains
pilot-only until end-to-end accuracy and resume gates pass.  This module routes
regions and rejects obviously invalid model output; it does not publish final
``OcrResult`` records by itself.
"""

from __future__ import annotations

import unicodedata
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class VinternGate2Policy(BaseModel):
    """Versioned thresholds derived from the dev-subset-5 visual audit."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    model_id: Literal["5CD-AI/Vintern-1B-v3_5"] = "5CD-AI/Vintern-1B-v3_5"
    model_revision: Literal["b98f263eab246eb5269ade64edbdca8a887dc44d"] = (
        "b98f263eab246eb5269ade64edbdca8a887dc44d"
    )
    model_weight_filename: Literal["model.safetensors"] = "model.safetensors"
    model_weight_bytes: Literal[3752849256] = 3752849256
    model_weight_sha256: Literal[
        "296a16a6bf28e6d3f0fb9298deba70b3cfa1d7519f4aa326e2f862bf2e63be05"
    ] = "296a16a6bf28e6d3f0fb9298deba70b3cfa1d7519f4aa326e2f862bf2e63be05"
    dtype: Literal["float16"] = "float16"
    max_num: Literal[1] = 1
    hard_confidence_threshold: float = Field(default=0.40, ge=0, le=1)
    selective_confidence_ceiling: float = Field(default=0.60, ge=0, le=1)
    escalate_region_mixed: bool = True
    escalate_ambiguous_glyphs: bool = True
    noisy_character_ratio_threshold: float = Field(default=0.34, ge=0, le=1)
    noisy_character_min_length: int = Field(default=3, ge=1)
    max_candidate_fraction: float = Field(default=0.40, gt=0, le=1)


class VinternRegionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate: bool
    reasons: tuple[str, ...]


_ALLOWED_PUNCTUATION = frozenset(".,:;!?%+-/()[]{}'\"&@#_\\|~`=<>₫$€£¥…–—")
_PROMPT_LEAK_MARKERS = (
    "i don't understand",
    "i do not understand",
    "this is a blurry image",
    "the image is blurry",
    "the text is blurry",
    "chữ bị che khuất",
    "chữ trong ảnh",
    "nội dung trong ảnh",
    "không thể đọc",
    "không đọc được",
    "từ đó tôi sẽ",
    "hãy chép lại",
    "chép lại nguyên văn",
)


def _is_noisy_character(character: str) -> bool:
    if character.isspace() or character.isalnum() or character in _ALLOWED_PUNCTUATION:
        return False
    category = unicodedata.category(character)
    return not category.startswith(("L", "N", "P", "S", "M"))


def _noise_ratio(text: str) -> float:
    visible = [character for character in text if not character.isspace()]
    if not visible:
        return 0.0
    return sum(_is_noisy_character(character) for character in visible) / len(visible)


def route_vintern_region(
    region: dict[str, Any],
    *,
    policy: VinternGate2Policy,
) -> VinternRegionDecision:
    """Route one region without consulting any other region in its frame."""

    text = str(region.get("easyocr_text") or "").strip()
    raw_confidence = region.get("easyocr_confidence")
    if raw_confidence is None:
        confidence = 0.0
    elif isinstance(raw_confidence, bool) or not isinstance(raw_confidence, (int, float)):
        raise ValueError("easyocr_confidence must be numeric or null")
    else:
        confidence = float(raw_confidence)
    if not 0 <= confidence <= 1:
        raise ValueError("easyocr_confidence must be in [0, 1]")

    reasons: list[str] = []
    if not text:
        reasons.append("empty_text")
    elif confidence < policy.hard_confidence_threshold:
        reasons.append("confidence_lt_0_40")
    elif confidence < policy.selective_confidence_ceiling:
        if (
            policy.escalate_region_mixed
            and bool(region.get("has_vi_marks"))
            and bool(region.get("has_ascii_word"))
        ):
            reasons.append("region_mixed_0_40_to_0_60")
        if policy.escalate_ambiguous_glyphs and ("?" in text or "�" in text):
            reasons.append("ambiguous_glyph_0_40_to_0_60")
        visible_length = len("".join(text.split()))
        if (
            visible_length >= policy.noisy_character_min_length
            and _noise_ratio(text) >= policy.noisy_character_ratio_threshold
        ):
            reasons.append("noisy_text_0_40_to_0_60")

    return VinternRegionDecision(candidate=bool(reasons), reasons=tuple(reasons))


def vintern_output_rejection_reasons(
    *,
    easyocr_text: str,
    vintern_text: str,
) -> tuple[str, ...]:
    """Reject obvious non-OCR answers; this is a safety guard, not accuracy scoring."""

    candidate = vintern_text.strip()
    if not candidate:
        return ("empty_output",)

    reasons: list[str] = []
    normalized = " ".join(candidate.casefold().split())
    if any(marker in normalized for marker in _PROMPT_LEAK_MARKERS):
        reasons.append("prompt_or_explanation_leak")

    easy_length = len("".join(easyocr_text.split()))
    candidate_length = len("".join(candidate.split()))
    if candidate_length > max(96, easy_length * 8 + 48):
        reasons.append("gross_length_expansion")
    if _noise_ratio(candidate) >= 0.50 and candidate_length >= 4:
        reasons.append("noisy_output")
    return tuple(reasons)
