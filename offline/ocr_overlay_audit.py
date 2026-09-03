"""Audit-only filtering for static, low-information OCR overlays.

The functions in this module never delete OCR text.  They identify residual regions that
may be withheld from Gemini because the same small overlay is repeated at a stable screen
position.  Production routing must opt in separately after the audit is reviewed.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, model_validator


_TIME_PATTERN = re.compile(r"^\s*\d{1,2}\s*[:.]\s*\d{2}(?:\s*[:.]\s*\d{2})?\s*$")
_BROADCASTER_PATTERN = re.compile(
    r"^(?:(?:htv|vtv|tv|vtc)\d*(?:hd)?\d*|hd|hdtv)$",
    re.IGNORECASE,
)


class OverlayAuditPolicy(BaseModel):
    """Conservative policy for an audit simulation, not a production mutation."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    mode: str = "audit_only"
    min_distinct_frames: int = Field(default=20, ge=2)
    min_video_frame_fraction: float = Field(default=0.05, gt=0.0, le=1.0)
    bbox_bin_size: float = Field(default=0.05, gt=0.0, le=0.25)
    max_short_text_characters: int = Field(default=16, ge=1)
    top_band_max_center_y: float = Field(default=0.30, gt=0.0, lt=1.0)
    top_right_min_center_x: float = Field(default=0.75, gt=0.5, lt=1.0)
    max_corner_width: float = Field(default=0.25, gt=0.0, le=1.0)
    max_corner_height: float = Field(default=0.20, gt=0.0, le=1.0)
    max_tiny_text_characters: int = Field(default=3, ge=1)
    max_tiny_area_fraction: float = Field(default=0.015, gt=0.0, le=1.0)
    protect_bottom_min_y: float = Field(default=0.62, gt=0.0, lt=1.0)
    protect_bottom_min_width: float = Field(default=0.18, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_audit_only(self) -> "OverlayAuditPolicy":
        if self.schema_version != 1:
            raise ValueError("unsupported overlay audit policy schema")
        if self.mode != "audit_only":
            raise ValueError("overlay policy must remain audit_only until explicitly approved")
        return self


def normalize_overlay_text(value: Any) -> str:
    """Return a stable Unicode signature while retaining Vietnamese letters."""

    text = unicodedata.normalize("NFC", str(value or "")).casefold()
    return "".join(character for character in text if character.isalnum())


def _bbox_geometry(region: dict[str, Any]) -> dict[str, float]:
    values = region.get("bbox_normalized")
    if not isinstance(values, list) or len(values) != 8:
        raise ValueError(f"region {region.get('region_id')!r} lacks normalized 8-number bbox")
    numbers = [float(value) for value in values]
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in numbers):
        raise ValueError(f"region {region.get('region_id')!r} has invalid normalized bbox")
    xs = numbers[0::2]
    ys = numbers[1::2]
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    width = max(0.0, right - left)
    height = max(0.0, bottom - top)
    return {
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
        "center_x": (left + right) / 2.0,
        "center_y": (top + bottom) / 2.0,
        "width": width,
        "height": height,
        "area": width * height,
    }


def _bin(value: float, size: float) -> int:
    return int(math.floor((value + 1e-12) / size))


def _position_signature(geometry: dict[str, float], policy: OverlayAuditPolicy) -> str:
    values = (
        geometry["center_x"],
        geometry["center_y"],
        geometry["width"],
        geometry["height"],
    )
    return ":".join(str(_bin(value, policy.bbox_bin_size)) for value in values)


def _region_group(
    region: dict[str, Any], policy: OverlayAuditPolicy
) -> tuple[str | None, str | None, dict[str, float]]:
    geometry = _bbox_geometry(region)
    raw_text = str(region.get("final_text") or region.get("easyocr_text") or "").strip()
    normalized = normalize_overlay_text(raw_text)
    if not normalized:
        return None, None, geometry

    bottom_wide = (
        geometry["top"] >= policy.protect_bottom_min_y
        and geometry["width"] >= policy.protect_bottom_min_width
    )
    if bottom_wide:
        return None, "protected_bottom_subtitle_or_ticker", geometry

    top_corner = (
        geometry["center_x"] >= policy.top_right_min_center_x
        and geometry["center_y"] <= policy.top_band_max_center_y
        and geometry["width"] <= policy.max_corner_width
        and geometry["height"] <= policy.max_corner_height
        and len(normalized) <= policy.max_short_text_characters
    )
    position = _position_signature(geometry, policy)
    if top_corner:
        # Position is deliberately stronger than OCR spelling here: broadcaster logos are
        # frequently recognized as a different one-to-three glyph string on every frame.
        return f"top_corner:{position}", "stable_top_corner_short_overlay", geometry

    tiny = (
        len(normalized) <= policy.max_tiny_text_characters
        and geometry["area"] <= policy.max_tiny_area_fraction
    )
    if tiny:
        return f"tiny:{normalized}:{position}", "stable_tiny_glyph_or_noise", geometry
    return None, None, geometry


def is_known_low_information_ground_truth(value: Any) -> bool:
    """Narrow helper used only to summarize labeled audit collisions."""

    raw = unicodedata.normalize("NFC", str(value or "")).strip()
    if not raw:
        return True
    compact = normalize_overlay_text(raw)
    if _TIME_PATTERN.fullmatch(raw):
        return True
    if _BROADCASTER_PATTERN.fullmatch(compact):
        return True
    without_times = re.sub(
        r"\d{1,2}\s*[:.]\s*\d{1,2}(?:\s*[:.]\s*\d{1,2})?",
        " ",
        raw,
    )
    if _BROADCASTER_PATTERN.fullmatch(normalize_overlay_text(without_times)):
        return True
    parts = re.split(r"\s+", raw)
    return bool(parts) and all(
        _TIME_PATTERN.fullmatch(part) or _BROADCASTER_PATTERN.fullmatch(normalize_overlay_text(part))
        for part in parts
    )


def audit_overlay_residuals(
    frames: Iterable[dict[str, Any]],
    *,
    policy: OverlayAuditPolicy,
    ground_truth_rows: Iterable[dict[str, Any]] = (),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return an audit report and a non-destructive Gemini residual manifest."""

    frame_rows = list(frames)
    video_frame_counts = Counter(str(row["video_id"]) for row in frame_rows)
    observed_uids: set[int] = set()
    entries: list[dict[str, Any]] = []
    group_frames: dict[tuple[str, str], set[int]] = defaultdict(set)
    group_texts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    group_reasons: dict[tuple[str, str], str] = {}

    for frame in frame_rows:
        video_id = str(frame["video_id"])
        keyframe_uid = int(frame["keyframe_uid"])
        shot_id = str(frame["shot_id"])
        if keyframe_uid in observed_uids:
            raise ValueError(f"duplicate keyframe_uid in calibrated frames: {keyframe_uid}")
        observed_uids.add(keyframe_uid)
        for region in frame.get("regions") or []:
            group_id, reason, geometry = _region_group(region, policy)
            entry = {
                "schema_version": 1,
                "video_id": video_id,
                "shot_id": shot_id,
                "keyframe_uid": keyframe_uid,
                "region_id": str(region["region_id"]),
                "bbox_normalized": region["bbox_normalized"],
                "final_text": str(region.get("final_text") or ""),
                "final_confidence": float(region.get("final_confidence") or 0.0),
                "final_engine": str(region.get("final_engine") or ""),
                "gemini_residual_before_overlay_audit": bool(region.get("gemini_residual")),
                "overlay_group_id": group_id,
                "overlay_reason": reason,
                "geometry": geometry,
            }
            entries.append(entry)
            if group_id is not None:
                key = (video_id, group_id)
                group_frames[key].add(keyframe_uid)
                group_texts[key][entry["final_text"]] += 1
                group_reasons[key] = str(reason)

    qualified: dict[tuple[str, str], dict[str, Any]] = {}
    for key, uids in group_frames.items():
        video_id, group_id = key
        required = max(
            policy.min_distinct_frames,
            math.ceil(video_frame_counts[video_id] * policy.min_video_frame_fraction),
        )
        if len(uids) < required:
            continue
        qualified[key] = {
            "video_id": video_id,
            "group_id": group_id,
            "reason": group_reasons[key],
            "distinct_frames": len(uids),
            "required_frames": required,
            "fraction_of_video_frames": len(uids) / video_frame_counts[video_id],
            "common_texts": [
                {"text": text, "regions": count}
                for text, count in group_texts[key].most_common(8)
            ],
        }

    residual_manifest: list[dict[str, Any]] = []
    for entry in entries:
        if not entry["gemini_residual_before_overlay_audit"]:
            continue
        key = (entry["video_id"], entry["overlay_group_id"])
        suppressed = entry["overlay_group_id"] is not None and key in qualified
        row = dict(entry)
        row["overlay_audit_suppression_candidate"] = suppressed
        row["gemini_residual_after_overlay_audit"] = not suppressed
        row["overlay_group_evidence"] = qualified.get(key)
        residual_manifest.append(row)

    before = residual_manifest
    after = [row for row in residual_manifest if row["gemini_residual_after_overlay_audit"]]
    suppressed_rows = [
        row for row in residual_manifest if row["overlay_audit_suppression_candidate"]
    ]

    def cardinality(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
        values = list(rows)
        return {
            "regions": len(values),
            "frames": len({int(row["keyframe_uid"]) for row in values}),
            "shots_or_grouped_requests": len(
                {(str(row["video_id"]), str(row["shot_id"])) for row in values}
            ),
        }

    label_by_candidate = {
        str(row.get("candidate_id") or row.get("region_id") or ""): row
        for row in ground_truth_rows
        if str(row.get("label_status") or "").strip().casefold() == "labeled"
    }
    labeled_residuals = [
        row for row in residual_manifest if row["region_id"] in label_by_candidate
    ]
    labeled_suppressed = [
        row for row in labeled_residuals if row["overlay_audit_suppression_candidate"]
    ]
    collisions: list[dict[str, Any]] = []
    for row in labeled_suppressed:
        label = label_by_candidate[row["region_id"]]
        is_empty = str(label.get("ground_truth_is_empty") or "").strip().casefold() == "yes"
        human_text = str(label.get("human_text") or "").strip()
        known_low_information = is_empty or is_known_low_information_ground_truth(human_text)
        collisions.append(
            {
                "candidate_id": row["region_id"],
                "video_id": row["video_id"],
                "human_text": human_text,
                "ground_truth_is_empty": is_empty,
                "known_low_information": known_low_information,
                "needs_manual_policy_review": not known_low_information,
                "group_id": row["overlay_group_id"],
            }
        )

    report = {
        "schema_version": 1,
        "decision": "AUDIT_ONLY_DO_NOT_APPLY_TO_PRODUCTION",
        "policy": policy.model_dump(mode="json"),
        "scope": {
            "frames": len(frame_rows),
            "videos": dict(sorted(video_frame_counts.items())),
            "all_regions": len(entries),
        },
        "gemini_residual": {
            "before_overlay_audit": cardinality(before),
            "suppression_candidates": cardinality(suppressed_rows),
            "after_overlay_audit": cardinality(after),
            "region_reduction_fraction": (
                len(suppressed_rows) / len(before) if before else 0.0
            ),
        },
        "qualified_overlay_groups": sorted(
            qualified.values(),
            key=lambda row: (-int(row["distinct_frames"]), str(row["video_id"]), str(row["group_id"])),
        ),
        "ground_truth_collision_audit": {
            "labeled_rows_total": len(label_by_candidate),
            "labeled_rows_in_residual": len(labeled_residuals),
            "labeled_suppression_candidates": len(labeled_suppressed),
            "potential_semantic_false_suppressions": sum(
                bool(row["needs_manual_policy_review"]) for row in collisions
            ),
            "rows": collisions,
        },
        "invariants": {
            "raw_ocr_text_deleted": False,
            "bbox_modified": False,
            "gemini_called": False,
            "production_router_modified": False,
            "bottom_subtitle_ticker_protected": True,
        },
    }
    return report, residual_manifest
