"""Resolve review media without exposing positional catalog identifiers."""

from __future__ import annotations

import json
from pathlib import Path

from .artifacts import ArtifactRegistry


def evidence_image(registry: ArtifactRegistry, keyframe_uid: int) -> Path | None:
    frame = registry.catalog.by_uid.get(keyframe_uid)
    if frame is None:
        return None
    path = frame.image_path(registry.layout.data.keyframes)
    return path if path.is_file() else None


def source_video(registry: ArtifactRegistry, video_id: str) -> Path | None:
    """Resolve the inventory path; never infer FPS or source frame numbers."""

    inventory = registry.layout.data.index / "inventory.json"
    if inventory.is_file():
        payload = json.loads(inventory.read_text(encoding="utf-8"))
        for row in payload.get("videos", []):
            if row.get("video_id") != video_id:
                continue
            value = str(row.get("relative_path", "")).strip()
            candidate = registry.layout.data.root / value
            if value and candidate.is_file():
                return candidate
    candidates = list(registry.layout.data.videos.rglob(f"{video_id}.*"))
    return next((path for path in candidates if path.suffix.lower() in {".mp4", ".mkv", ".webm"}), None)
