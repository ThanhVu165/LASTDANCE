"""Cost/quality canary helpers for shot-grouped Gemini OCR requests.

This module is deliberately separate from the production runner.  It compares
request packaging strategies while keeping the final per-frame ``OcrResult``
contract intact.  Synthetic shots are only suitable for API/schema/token
preflight; production recall still requires real keyframes.
"""

from __future__ import annotations

import base64
import io
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from PIL import Image, ImageDraw, ImageFont

from shared.schemas.ocr import OcrResult


CanaryStrategy = Literal["separate", "multi_image", "crop_sheet", "middle_only"]
CANARY_STRATEGIES: tuple[CanaryStrategy, ...] = (
    "separate",
    "multi_image",
    "crop_sheet",
    "middle_only",
)
_MODEL_ID_PATTERN = re.compile(r"^gemini-[a-z0-9][a-z0-9.-]*$")


@dataclass(frozen=True)
class SyntheticFrame:
    keyframe_uid: int
    frame_id: int
    image_bytes: bytes
    expected_text: tuple[str, ...]
    regions: tuple["SyntheticRegion", ...]


@dataclass(frozen=True)
class SyntheticRegion:
    region_id: int
    keyframe_uid: int
    frame_id: int
    image_bytes: bytes
    source_bbox: tuple[float, float, float, float, float, float, float, float]
    expected_text: str


@dataclass(frozen=True)
class SyntheticShot:
    shot_id: str
    frames: tuple[SyntheticFrame, SyntheticFrame, SyntheticFrame]


def validate_model_id(model_id: str) -> str:
    if not _MODEL_ID_PATTERN.fullmatch(model_id):
        raise ValueError(f"invalid Gemini model id: {model_id!r}")
    return model_id


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\tahoma.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_synthetic_shot(index: int) -> SyntheticShot:
    """Create a three-frame shot whose subtitle changes across time."""

    scenes = (
        (
            "CỬA HÀNG SÀI GÒN",
            ("GIẢM GIÁ 10%", "GIẢM GIÁ 30%", "GIẢM GIÁ 50%"),
            ("MỞ CỬA", "OPEN NOW", "MỞ CỬA"),
        ),
        (
            "BỆNH VIỆN QUẬN 1",
            ("PHÒNG KHÁM A", "PHÒNG KHÁM B", "PHÒNG KHÁM C"),
            ("LẦU 1", "TẦNG 2", "EXIT"),
        ),
        (
            "ĐẠI HỌC THÀNH PHỐ",
            ("CỔNG SỐ 1", "CỔNG SỐ 2", "CỔNG SỐ 3"),
            ("ENTRANCE", "LỐI VÀO", "ENTRANCE"),
        ),
        (
            "ĐƯỜNG NGUYỄN HUỆ",
            ("CẤM RẼ TRÁI", "ĐI THẲNG", "CẤM RẼ PHẢI"),
            ("07:00", "12:30", "18:45"),
        ),
    )
    common, changing, ticker = scenes[(index - 1) % len(scenes)]
    frames: list[SyntheticFrame] = []
    for position in range(3):
        width, height = 1280, 720
        image = Image.new("RGB", (width, height), (20 + position * 5, 45, 64))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle(
            (90, 150, width - 90, height - 120),
            radius=30,
            fill="#fffdf7",
            outline="#f4b400",
            width=10,
        )
        title_font = _font(68)
        subtitle_font = _font(54)
        ticker_font = _font(30)
        text_rows = (
            (common, (150, 220), title_font, "#18212a"),
            (changing[position], (220, 360), subtitle_font, "#b3261e"),
            (ticker[position], (900, 535), ticker_font, "#165d3a"),
        )
        for text, position_xy, text_font, color in text_rows:
            draw.text(position_xy, text, fill=color, font=text_font)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90, optimize=True)
        regions: list[SyntheticRegion] = []
        for row_index, (text, position_xy, text_font, _) in enumerate(text_rows):
            left, top, right, bottom = draw.textbbox(position_xy, text, font=text_font)
            padding = 12
            left = max(0, left - padding)
            top = max(0, top - padding)
            right = min(width, right + padding)
            bottom = min(height, bottom + padding)
            crop = image.crop((left, top, right, bottom))
            crop_buffer = io.BytesIO()
            crop.save(crop_buffer, format="JPEG", quality=92, optimize=True)
            regions.append(
                SyntheticRegion(
                    region_id=(index * 100) + (position * 10) + row_index,
                    keyframe_uid=(index * 1000) + position + 1,
                    frame_id=index * 10 + position,
                    image_bytes=crop_buffer.getvalue(),
                    source_bbox=(
                        left / width,
                        top / height,
                        right / width,
                        top / height,
                        right / width,
                        bottom / height,
                        left / width,
                        bottom / height,
                    ),
                    expected_text=text,
                )
            )
        frames.append(
            SyntheticFrame(
                keyframe_uid=(index * 1000) + position + 1,
                frame_id=index * 10 + position,
                image_bytes=buffer.getvalue(),
                expected_text=(common, changing[position], ticker[position]),
                regions=tuple(regions),
            )
        )
    return SyntheticShot(shot_id=f"synthetic-{index:04d}", frames=tuple(frames))  # type: ignore[arg-type]


def make_crop_sheet(
    frames: Sequence[SyntheticFrame],
) -> tuple[bytes, tuple[SyntheticRegion, ...]]:
    """Mosaic detector crops while preserving each source-frame quadrilateral."""

    if len(frames) != 3:
        raise ValueError("crop_sheet requires exactly three frames")
    regions = tuple(region for frame in frames for region in frame.regions)
    crops: list[Image.Image] = []
    for region in regions:
        with Image.open(io.BytesIO(region.image_bytes)) as image:
            crop = image.convert("RGB")
            crop.thumbnail((960, 96), Image.Resampling.LANCZOS)
            crops.append(crop.copy())
    gap = 8
    sheet = Image.new(
        "RGB",
        (
            max(crop.width for crop in crops),
            sum(crop.height for crop in crops) + (len(crops) - 1) * gap,
        ),
        "white",
    )
    y = 0
    for crop in crops:
        sheet.paste(crop, (0, y))
        y += crop.height + gap
    buffer = io.BytesIO()
    sheet.save(buffer, format="JPEG", quality=90, optimize=True)
    return buffer.getvalue(), regions


def strict_crop_region_results_schema(region_ids: Sequence[int]) -> dict[str, Any]:
    expected = list(region_ids)
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("region_ids must be non-empty and unique")
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "regions": {
                "type": "array",
                "minItems": len(expected),
                "maxItems": len(expected),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "region_id": {"type": "integer", "enum": expected},
                        "text": {"type": "string", "minLength": 1},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "language": {"type": "string", "enum": ["vi", "en", "mixed"]},
                    },
                    "required": ["region_id", "text", "language", "confidence"],
                    "propertyOrdering": [
                        "region_id",
                        "text",
                        "language",
                        "confidence",
                    ],
                },
            }
        },
        "required": ["regions"],
        "propertyOrdering": ["regions"],
    }


def strict_results_schema(frame_ids: Sequence[int]) -> dict[str, Any]:
    expected = list(frame_ids)
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("frame_ids must be non-empty and unique")
    result_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "frame_id": {"type": "integer", "enum": expected},
            "detected_text": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "minLength": 1},
            },
            "bbox": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "array",
                    "minItems": 8,
                    "maxItems": 8,
                    "items": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "language": {"type": "string", "enum": ["vi", "en", "mixed"]},
        },
        "required": ["frame_id", "detected_text", "bbox", "confidence", "language"],
        "propertyOrdering": [
            "frame_id",
            "detected_text",
            "bbox",
            "confidence",
            "language",
        ],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "results": {
                "type": "array",
                "minItems": len(expected),
                "maxItems": len(expected),
                "items": result_schema,
            }
        },
        "required": ["results"],
        "propertyOrdering": ["results"],
    }


def _inline_image(image_bytes: bytes) -> dict[str, Any]:
    return {
        "inlineData": {
            "mimeType": "image/jpeg",
            "data": base64.b64encode(image_bytes).decode("ascii"),
        }
    }


def build_request_payload(
    shot: SyntheticShot,
    strategy: CanaryStrategy,
) -> tuple[bytes, tuple[int, ...]]:
    if strategy not in CANARY_STRATEGIES:
        raise ValueError(f"unsupported OCR canary strategy: {strategy}")
    if strategy == "separate":
        raise ValueError("separate strategy must build one payload per frame")

    if strategy == "middle_only":
        selected = (shot.frames[1],)
        image_parts = [_inline_image(selected[0].image_bytes)]
        mapping = f"The only image is frame_id {selected[0].frame_id}."
    elif strategy == "multi_image":
        selected = shot.frames
        image_parts = [_inline_image(frame.image_bytes) for frame in selected]
        mapping = (
            "Images are ordered left-to-right in the request and correspond to frame_ids "
            f"{[frame.frame_id for frame in selected]}."
        )
    else:
        raise ValueError("crop_sheet uses build_crop_sheet_request_payload")

    frame_ids = tuple(frame.frame_id for frame in selected)
    prompt = (
        "OCR every requested frame. Return only JSON matching the supplied schema. "
        "Preserve Vietnamese diacritics and original casing. detected_text and bbox must "
        "have equal lengths. Every bbox is exactly 8 normalized numbers in [0,1], "
        "clockwise from visual top-left. language is vi, en, or mixed. "
        f"{mapping} Return exactly one result for every listed frame_id and no others."
    )
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}, *image_parts],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 2048,
            "responseMimeType": "application/json",
            "responseJsonSchema": strict_results_schema(frame_ids),
        },
    }
    return json.dumps(payload, ensure_ascii=False).encode("utf-8"), frame_ids


def build_crop_sheet_request_payload(
    shot: SyntheticShot,
) -> tuple[bytes, tuple[int, ...], tuple[SyntheticRegion, ...]]:
    image_bytes, regions = make_crop_sheet(shot.frames)
    region_ids = tuple(region.region_id for region in regions)
    keyframe_uids = tuple(frame.keyframe_uid for frame in shot.frames)
    region_map = [
        {"region_id": region.region_id, "keyframe_uid": region.keyframe_uid}
        for region in regions
    ]
    prompt = (
        "Recognize every text crop in the single vertical sheet. Crops are ordered "
        f"top-to-bottom with this local mapping: {region_map}. The three source "
        f"keyframe_uids are {list(keyframe_uids)}. Return exactly one row per region_id "
        "with only region_id, text, language and confidence. Preserve Vietnamese "
        "diacritics and original casing. Do not return frame_id, keyframe_uid or bounding "
        "boxes; the adapter owns identity and source-frame geometry."
    )
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": prompt}, _inline_image(image_bytes)]}
        ],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 1024,
            "mediaResolution": "MEDIA_RESOLUTION_MEDIUM",
            "responseMimeType": "application/json",
            "responseJsonSchema": strict_crop_region_results_schema(region_ids),
        },
    }
    return (
        json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        keyframe_uids,
        regions,
    )


def build_separate_payloads(shot: SyntheticShot) -> list[tuple[bytes, tuple[int, ...]]]:
    payloads = []
    for position, frame in enumerate(shot.frames):
        single_shot = SyntheticShot(
            shot_id=f"{shot.shot_id}-frame-{position}",
            frames=(frame, frame, frame),
        )
        payload, frame_ids = build_request_payload(single_shot, "middle_only")
        payloads.append((payload, frame_ids))
    return payloads


def parse_strict_results(
    response: dict[str, Any],
    expected_frame_ids: Sequence[int],
) -> tuple[list[OcrResult], dict[str, Any]]:
    text = response["candidates"][0]["content"]["parts"][0]["text"]
    payload = json.loads(text)
    rows = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("structured response is missing results")
    results = [OcrResult.model_validate(row) for row in rows]
    actual_ids = [result.frame_id for result in results]
    if len(actual_ids) != len(set(actual_ids)):
        raise ValueError("structured response contains duplicate frame_id")
    if set(actual_ids) != set(expected_frame_ids):
        raise ValueError("structured response frame_id set does not match request")
    for result in results:
        if not result.detected_text or any(not text.strip() for text in result.detected_text):
            raise ValueError("canary text frame returned empty detected_text")
        for bbox in result.bbox:
            if len(bbox) != 8 or any(coordinate < 0 or coordinate > 1 for coordinate in bbox):
                raise ValueError("bbox must be an 8-value normalized quadrilateral")
    usage = response.get("usageMetadata", {})
    return results, {
        "model_version": response.get("modelVersion"),
        "prompt_tokens": int(usage.get("promptTokenCount", 0)),
        "output_tokens": int(usage.get("candidatesTokenCount", 0)),
        "total_tokens": int(usage.get("totalTokenCount", 0)),
    }


def parse_crop_sheet_results(
    response: dict[str, Any],
    regions: Sequence[SyntheticRegion],
) -> tuple[list[OcrResult], dict[str, Any]]:
    text = response["candidates"][0]["content"]["parts"][0]["text"]
    payload = json.loads(text)
    rows = payload.get("regions") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("structured response is missing regions")
    region_by_id = {region.region_id: region for region in regions}
    if len(region_by_id) != len(regions):
        raise ValueError("local detector contains duplicate region_id")
    actual_ids = [row.get("region_id") for row in rows if isinstance(row, dict)]
    if len(actual_ids) != len(rows) or len(actual_ids) != len(set(actual_ids)):
        raise ValueError("structured response contains invalid/duplicate region_id")
    if set(actual_ids) != set(region_by_id):
        raise ValueError("structured response region_id set does not match detector crops")

    response_by_id = {row["region_id"]: row for row in rows}
    regions_by_frame: dict[int, list[SyntheticRegion]] = {}
    for region in regions:
        regions_by_frame.setdefault(region.frame_id, []).append(region)
    results: list[OcrResult] = []
    for frame_id, frame_regions in regions_by_frame.items():
        detected_text: list[str] = []
        confidences: list[float] = []
        languages: set[str] = set()
        for region in frame_regions:
            row = response_by_id[region.region_id]
            value = row.get("text")
            confidence = row.get("confidence")
            language = row.get("language")
            if not isinstance(value, str) or not value.strip():
                raise ValueError("crop recognition returned empty text")
            if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                raise ValueError("crop recognition confidence must be in [0,1]")
            if language not in {"vi", "en", "mixed"}:
                raise ValueError("crop recognition language is invalid")
            detected_text.append(value.strip())
            confidences.append(float(confidence))
            languages.add(language)
        weights = [max(1, sum(not char.isspace() for char in value)) for value in detected_text]
        aggregate_confidence = sum(
            confidence * weight for confidence, weight in zip(confidences, weights, strict=True)
        ) / sum(weights)
        aggregate_language = next(iter(languages)) if len(languages) == 1 else "mixed"
        results.append(
            OcrResult(
                frame_id=frame_id,
                detected_text=detected_text,
                bbox=[list(region.source_bbox) for region in frame_regions],
                confidence=aggregate_confidence,
                language=aggregate_language,
            )
        )
    usage = response.get("usageMetadata", {})
    return results, {
        "model_version": response.get("modelVersion"),
        "prompt_tokens": int(usage.get("promptTokenCount", 0)),
        "output_tokens": int(usage.get("candidatesTokenCount", 0)),
        "total_tokens": int(usage.get("totalTokenCount", 0)),
    }


def normalize_ocr_text(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value).casefold()
    return " ".join(re.sub(r"[^\w%]+", " ", normalized, flags=re.UNICODE).split())


def expected_line_recall(
    shot: SyntheticShot,
    results: Sequence[OcrResult],
    *,
    propagate_single_result: bool = False,
) -> dict[str, float | int]:
    by_frame = {result.frame_id: result for result in results}
    if propagate_single_result:
        if len(results) != 1:
            raise ValueError("propagation requires exactly one OCR result")
        by_frame = {frame.frame_id: results[0] for frame in shot.frames}
    expected_count = 0
    matched_count = 0
    for frame in shot.frames:
        result = by_frame.get(frame.frame_id)
        detected = " ".join(
            normalize_ocr_text(text) for text in result.detected_text
        ) if result is not None else ""
        for expected in frame.expected_text:
            expected_count += 1
            if normalize_ocr_text(expected) in detected:
                matched_count += 1
    return {
        "expected_lines": expected_count,
        "matched_lines": matched_count,
        "recall": matched_count / expected_count if expected_count else 0.0,
    }
