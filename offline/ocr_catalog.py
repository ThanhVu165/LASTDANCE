"""Evidence-grade catalog audit used before OCR extraction."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from offline.artifacts import sha256_file
from offline.catalog import FRAME_COLUMNS, validate_frames_catalog
from offline.identifiers import make_keyframe_uid
from offline.ocr_artifacts import uid_set_sha256
from shared.schemas.frame import FrameRecord


def audit_ocr_catalog(
    csv_path: Path,
    *,
    state_path: Path | None = None,
) -> dict[str, object]:
    """Validate the complete catalog and recompute every spec-defined UID."""

    catalog = Path(csv_path)
    state = state_path or catalog.with_name(f"{catalog.name}.state.json")
    if not validate_frames_catalog(catalog, state):
        raise RuntimeError(
            f"frames.csv/state validation failed: catalog={catalog}, state={state}"
        )

    with catalog.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FRAME_COLUMNS:
            raise RuntimeError("frames.csv columns do not match the canonical schema")
        records = [
            FrameRecord(**{**row, "window_id": row["window_id"] or None})
            for row in reader
        ]
    if not records:
        raise RuntimeError("frames.csv contains no records")

    mismatched_uids = [
        record.keyframe_uid
        for record in records
        if record.keyframe_uid
        != make_keyframe_uid(record.video_id, record.shot_id, record.local_idx)
    ]
    if mismatched_uids:
        raise RuntimeError(
            "frames.csv contains keyframe_uid values that do not match BASELINE_SPEC: "
            f"sample={mismatched_uids[:5]}"
        )

    state_payload = json.loads(state.read_text(encoding="utf-8"))
    catalog_sha256 = sha256_file(catalog)
    if catalog_sha256 != state_payload["csv_sha256"]:
        raise RuntimeError("catalog hash changed after validation")
    per_video = Counter(record.video_id for record in records)
    return {
        "schema_version": 1,
        "catalog_path": catalog.name,
        "catalog_state_path": state.name,
        "catalog_sha256": catalog_sha256,
        "uid_set_sha256": uid_set_sha256(record.keyframe_uid for record in records),
        "record_count": len(records),
        "video_count": len(per_video),
        "min_keyframes_per_video": min(per_video.values()),
        "max_keyframes_per_video": max(per_video.values()),
        "keyframe_uid_formula_mismatches": 0,
        "duplicate_keyframe_uids": 0,
        "state_complete": True,
    }
