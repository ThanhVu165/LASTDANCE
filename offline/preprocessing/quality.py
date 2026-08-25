"""Measure blur and perceptual similarity without deleting source keyframes."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from .keyframes import KeyframePlanItem


@dataclass(frozen=True, slots=True)
class QualityMetrics:
    laplacian_variance: float
    phash: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.laplacian_variance) or self.laplacian_variance < 0:
            raise ValueError("laplacian_variance must be finite and non-negative")
        try:
            int(self.phash, 16)
        except (TypeError, ValueError) as exc:
            raise ValueError("phash must be a hexadecimal string") from exc
        if not self.phash:
            raise ValueError("phash must not be empty")


@dataclass(frozen=True, slots=True)
class QualityDecision:
    keyframe_uid: int
    video_id: str
    shot_id: str
    local_idx: int
    frame_id: int
    relative_image_path: str
    laplacian_variance: float
    phash: str
    kept: bool
    reason: str
    duplicate_of_keyframe_uid: int | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "keyframe_uid": self.keyframe_uid,
            "video_id": self.video_id,
            "shot_id": self.shot_id,
            "local_idx": self.local_idx,
            "frame_id": self.frame_id,
            "relative_image_path": self.relative_image_path,
            "laplacian_variance": self.laplacian_variance,
            "phash": self.phash,
            "kept": self.kept,
            "reason": self.reason,
            "duplicate_of_keyframe_uid": self.duplicate_of_keyframe_uid,
        }


_MetricReader = Callable[[Path], QualityMetrics]


def read_quality_metrics(image_path: Path) -> QualityMetrics:
    """Read one JPEG and calculate Laplacian variance plus 64-bit pHash."""

    try:
        import cv2
        import imagehash
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "quality filtering requires requirements/offline-local.txt"
        ) from exc

    source = Path(image_path)
    if not source.is_file():
        raise RuntimeError(f"keyframe image not found: {source}")
    try:
        with Image.open(source) as opened:
            rgb = opened.convert("RGB")
            pixels = np.asarray(rgb)
            phash = str(imagehash.phash(rgb))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"cannot decode keyframe image: {source}") from exc
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return QualityMetrics(laplacian_variance=variance, phash=phash)


def phash_distance(left: str, right: str) -> int:
    if len(left) != len(right):
        raise ValueError("pHash values must use the same bit length")
    try:
        return (int(left, 16) ^ int(right, 16)).bit_count()
    except ValueError as exc:
        raise ValueError("pHash values must be hexadecimal") from exc


def assess_keyframes(
    items: Iterable[KeyframePlanItem],
    *,
    data_root: Path,
    blur_threshold: float | None = None,
    phash_max_distance: int | None = None,
    metric_reader: _MetricReader = read_quality_metrics,
) -> list[QualityDecision]:
    """Assess all items, deduplicating only within a shot and preserving coverage."""

    if blur_threshold is not None and blur_threshold < 0:
        raise ValueError("blur_threshold must be non-negative")
    if phash_max_distance is not None and phash_max_distance < 0:
        raise ValueError("phash_max_distance must be non-negative")
    rows = list(items)
    if not rows:
        raise ValueError("items must not be empty")
    root = Path(data_root).resolve(strict=False)
    measured: dict[int, QualityMetrics] = {}
    grouped: dict[tuple[str, str], list[KeyframePlanItem]] = defaultdict(list)
    for item in rows:
        relative = Path(item.relative_image_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("relative_image_path must stay inside AIC_DATA")
        image_path = (root / relative).resolve(strict=False)
        try:
            image_path.relative_to(root)
        except ValueError as exc:
            raise ValueError("keyframe image must stay inside AIC_DATA") from exc
        uid = item.frame.keyframe_uid
        if uid in measured:
            raise RuntimeError(f"duplicate keyframe_uid in quality input: {uid}")
        measured[uid] = metric_reader(image_path)
        grouped[(item.frame.video_id, item.frame.shot_id)].append(item)

    decisions_by_uid: dict[int, QualityDecision] = {}
    for shot_items in grouped.values():
        eligible = list(shot_items)
        fallback_uid: int | None = None
        blurred_uids: set[int] = set()
        if blur_threshold is not None:
            eligible = [
                item
                for item in shot_items
                if measured[item.frame.keyframe_uid].laplacian_variance
                >= blur_threshold
            ]
            blurred_uids = {
                item.frame.keyframe_uid for item in shot_items if item not in eligible
            }
            if not eligible:
                fallback = max(
                    shot_items,
                    key=lambda item: measured[
                        item.frame.keyframe_uid
                    ].laplacian_variance,
                )
                eligible = [fallback]
                fallback_uid = fallback.frame.keyframe_uid
                blurred_uids.remove(fallback_uid)

        kept_items: list[KeyframePlanItem] = []
        duplicate_of: dict[int, int] = {}
        for item in eligible:
            uid = item.frame.keyframe_uid
            if phash_max_distance is not None and kept_items:
                nearest = min(
                    kept_items,
                    key=lambda kept: phash_distance(
                        measured[uid].phash,
                        measured[kept.frame.keyframe_uid].phash,
                    ),
                )
                distance = phash_distance(
                    measured[uid].phash,
                    measured[nearest.frame.keyframe_uid].phash,
                )
                if distance <= phash_max_distance:
                    duplicate_of[uid] = nearest.frame.keyframe_uid
                    continue
            kept_items.append(item)

        kept_uids = {item.frame.keyframe_uid for item in kept_items}
        for item in shot_items:
            uid = item.frame.keyframe_uid
            metrics = measured[uid]
            if uid in blurred_uids:
                kept = False
                reason = "blur"
            elif uid in duplicate_of:
                kept = False
                reason = "near_duplicate"
            elif uid == fallback_uid:
                kept = True
                reason = "kept_best_blur_fallback"
            else:
                kept = uid in kept_uids
                reason = "kept"
            decisions_by_uid[uid] = QualityDecision(
                keyframe_uid=uid,
                video_id=item.frame.video_id,
                shot_id=item.frame.shot_id,
                local_idx=item.frame.local_idx,
                frame_id=item.frame.frame_id,
                relative_image_path=item.relative_image_path,
                laplacian_variance=metrics.laplacian_variance,
                phash=metrics.phash,
                kept=kept,
                reason=reason,
                duplicate_of_keyframe_uid=duplicate_of.get(uid),
            )

    decisions = [decisions_by_uid[item.frame.keyframe_uid] for item in rows]
    kept_shots = {
        (decision.video_id, decision.shot_id)
        for decision in decisions
        if decision.kept
    }
    if kept_shots != set(grouped):
        raise RuntimeError("quality filtering removed every keyframe from a shot")
    return decisions


def write_quality_manifest_atomic(
    output_path: Path,
    *,
    video_id: str,
    source_plan_sha256: str,
    blur_threshold: float | None,
    phash_max_distance: int | None,
    decisions: Iterable[QualityDecision],
) -> None:
    rows = list(decisions)
    if not rows:
        raise ValueError("quality manifest must contain decisions")
    if any(row.video_id != video_id for row in rows):
        raise ValueError("quality decisions contain a mismatched video_id")
    config = {
        "blur_threshold": blur_threshold,
        "phash_max_distance": phash_max_distance,
        "preserve_at_least_one_per_shot": True,
    }
    signature_payload = json.dumps(
        {"source_plan_sha256": source_plan_sha256, "config": config},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    signature = hashlib.sha256(signature_payload).hexdigest()
    payload = {
        "schema_version": 1,
        "video_id": video_id,
        "source_plan_sha256": source_plan_sha256,
        "config": config,
        "config_signature": signature,
        "counts": {
            "input": len(rows),
            "kept": sum(row.kept for row in rows),
            "blurred": sum(row.reason == "blur" for row in rows),
            "near_duplicates": sum(row.reason == "near_duplicate" for row in rows),
            "blur_fallbacks": sum(
                row.reason == "kept_best_blur_fallback" for row in rows
            ),
        },
        "items": [row.as_dict() for row in rows],
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
