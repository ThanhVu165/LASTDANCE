"""Build and validate the canonical frame catalog from verified per-video stages."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from offline.artifacts import sha256_file
from offline.preprocessing.keyframes import KeyframePlanItem
from offline.preprocessing.quality import QualityDecision
from shared.schemas.frame import FrameRecord


FRAME_COLUMNS = [
    "video_id",
    "local_idx",
    "frame_id",
    "pts_time",
    "shot_id",
    "window_id",
    "keyframe_uid",
]


def load_inventory_video_ids(path: Path) -> list[str]:
    """Load the canonical collection membership from inventory schema v1."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read inventory: {source}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("inventory top-level value must be an object")
    if payload.get("schema_version") != 1:
        raise RuntimeError("unsupported inventory schema version")
    videos = payload.get("videos")
    if not isinstance(videos, list) or not videos:
        raise RuntimeError("inventory contains no videos")
    video_ids: list[str] = []
    for index, row in enumerate(videos):
        if not isinstance(row, dict):
            raise RuntimeError(f"inventory video[{index}] is not an object")
        video_id = str(row.get("video_id", ""))
        if not video_id.strip() or video_id != video_id.strip():
            raise RuntimeError(f"inventory video[{index}] has an invalid video_id")
        video_ids.append(video_id)
    if len(set(video_ids)) != len(video_ids):
        raise RuntimeError("inventory contains duplicate video_id values")
    return sorted(video_ids)


def discover_catalog_inputs(
    plans_dir: Path,
    quality_dir: Path,
    *,
    expected_video_ids: Iterable[str],
) -> list[tuple[Path, Path]]:
    """Match per-video plan/quality manifests for an exact collection."""

    expected = list(expected_video_ids)
    if not expected:
        raise ValueError("expected_video_ids must not be empty")
    if len(set(expected)) != len(expected):
        raise ValueError("expected_video_ids contains duplicates")
    expected_set = set(expected)

    def manifests_by_stem(folder: Path, label: str) -> dict[str, Path]:
        root = Path(folder)
        if not root.is_dir():
            raise RuntimeError(f"{label} directory not found: {root}")
        paths = sorted(root.glob("*.json"))
        mapping = {path.stem: path for path in paths if path.is_file()}
        if len(mapping) != len(paths):
            raise RuntimeError(f"{label} directory contains duplicate manifest stems")
        return mapping

    plans = manifests_by_stem(plans_dir, "keyframe plan")
    qualities = manifests_by_stem(quality_dir, "keyframe quality")
    plan_ids = set(plans)
    quality_ids = set(qualities)
    if plan_ids != expected_set or quality_ids != expected_set:
        raise RuntimeError(
            "catalog collection inputs do not match inventory: "
            f"missing_plans={sorted(expected_set - plan_ids)[:10]}, "
            f"unexpected_plans={sorted(plan_ids - expected_set)[:10]}, "
            f"missing_quality={sorted(expected_set - quality_ids)[:10]}, "
            f"unexpected_quality={sorted(quality_ids - expected_set)[:10]}"
        )
    return [(plans[video_id], qualities[video_id]) for video_id in sorted(expected_set)]


def load_quality_manifest(
    path: Path,
) -> tuple[str, str, str, list[QualityDecision]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    video_id = str(payload["video_id"])
    source_plan_sha256 = str(payload["source_plan_sha256"])
    config_signature = str(payload["config_signature"])
    decisions = [QualityDecision(**row) for row in payload.get("items", [])]
    if not decisions:
        raise RuntimeError("quality manifest contains no decisions")
    if any(decision.video_id != video_id for decision in decisions):
        raise RuntimeError("quality manifest contains a mismatched video_id")
    if len({decision.keyframe_uid for decision in decisions}) != len(decisions):
        raise RuntimeError("quality manifest contains duplicate keyframe_uid values")
    counts = payload.get("counts") or {}
    if counts.get("input") != len(decisions):
        raise RuntimeError("quality manifest input count does not match its items")
    if counts.get("kept") != sum(decision.kept for decision in decisions):
        raise RuntimeError("quality manifest kept count does not match its items")
    return video_id, source_plan_sha256, config_signature, decisions


def select_catalog_records(
    items: Iterable[KeyframePlanItem],
    decisions: Iterable[QualityDecision],
) -> list[FrameRecord]:
    plan = list(items)
    quality = list(decisions)
    plan_by_uid = {item.frame.keyframe_uid: item for item in plan}
    quality_by_uid = {decision.keyframe_uid: decision for decision in quality}
    if len(plan_by_uid) != len(plan):
        raise RuntimeError("keyframe plan contains duplicate keyframe_uid values")
    if len(quality_by_uid) != len(quality):
        raise RuntimeError("quality decisions contain duplicate keyframe_uid values")
    if set(plan_by_uid) != set(quality_by_uid):
        missing_quality = sorted(set(plan_by_uid) - set(quality_by_uid))
        unexpected_quality = sorted(set(quality_by_uid) - set(plan_by_uid))
        raise RuntimeError(
            "quality/keyframe UID mismatch: "
            f"missing_quality={missing_quality[:5]}, "
            f"unexpected_quality={unexpected_quality[:5]}"
        )
    records: list[FrameRecord] = []
    for item in plan:
        decision = quality_by_uid[item.frame.keyframe_uid]
        if (
            decision.video_id != item.frame.video_id
            or decision.shot_id != item.frame.shot_id
            or decision.local_idx != item.frame.local_idx
            or decision.frame_id != item.frame.frame_id
            or decision.relative_image_path != item.relative_image_path
        ):
            raise RuntimeError(
                f"quality metadata mismatch for keyframe_uid={item.frame.keyframe_uid}"
            )
        if decision.kept:
            records.append(item.frame)
    if not records:
        raise RuntimeError("quality filtering kept no catalog records")
    return records


def _validate_catalog_records(records: list[FrameRecord]) -> None:
    if not records:
        raise ValueError("frames catalog must contain at least one record")
    uids = [record.keyframe_uid for record in records]
    if len(set(uids)) != len(uids):
        raise RuntimeError("frames catalog contains duplicate keyframe_uid values")
    submission_keys = [(record.video_id, record.frame_id) for record in records]
    if len(set(submission_keys)) != len(submission_keys):
        raise RuntimeError("frames catalog contains duplicate (video_id, frame_id) values")
    local_keys = [(record.video_id, record.local_idx) for record in records]
    if len(set(local_keys)) != len(local_keys):
        raise RuntimeError("frames catalog contains duplicate (video_id, local_idx) values")


def write_frames_catalog_atomic(
    output_path: Path,
    *,
    records: Iterable[FrameRecord],
    sources: Iterable[Mapping[str, str]],
) -> Path:
    """Publish CSV first and its hash-bound complete state last."""

    rows = sorted(records, key=lambda record: (record.video_id, record.local_idx))
    _validate_catalog_records(rows)
    source_rows = sorted(
        (dict(source) for source in sources),
        key=lambda source: source["video_id"],
    )
    if not source_rows:
        raise ValueError("frames catalog sources must not be empty")
    required_source_fields = {
        "video_id",
        "plan_sha256",
        "quality_sha256",
        "quality_config_signature",
    }
    if any(
        not required_source_fields.issubset(source)
        or any(not str(source[field]).strip() for field in required_source_fields)
        for source in source_rows
    ):
        raise RuntimeError("frames catalog source provenance is incomplete")
    if {source["video_id"] for source in source_rows} != {
        record.video_id for record in rows
    }:
        raise RuntimeError("frames catalog sources do not match record video IDs")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FRAME_COLUMNS)
        writer.writeheader()
        for record in rows:
            payload = record.model_dump()
            if payload["window_id"] is None:
                payload["window_id"] = ""
            writer.writerow(payload)

    csv_sha256 = sha256_file(temporary)
    state_path = destination.with_name(f"{destination.name}.state.json")
    state_temporary = state_path.with_name(f"{state_path.name}.tmp")
    state = {
        "schema_version": 1,
        "complete": True,
        "record_count": len(rows),
        "video_count": len({record.video_id for record in rows}),
        "csv_sha256": csv_sha256,
        "sources": source_rows,
    }
    state_temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    state_temporary.replace(state_path)
    return state_path


def validate_frames_catalog(csv_path: Path, state_path: Path | None = None) -> bool:
    destination = Path(csv_path)
    state_file = state_path or destination.with_name(f"{destination.name}.state.json")
    if not destination.is_file() or not state_file.is_file():
        return False
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            return False
        if state.get("schema_version") != 1:
            return False
        if state.get("complete") is not True:
            return False
        if sha256_file(destination) != state.get("csv_sha256"):
            return False
        with destination.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != FRAME_COLUMNS:
                return False
            records = [
                FrameRecord(
                    **{
                        **row,
                        "window_id": row["window_id"] or None,
                    }
                )
                for row in reader
            ]
        _validate_catalog_records(records)
        record_video_ids = {record.video_id for record in records}
        sources = state.get("sources")
        if not isinstance(sources, list) or any(
            not isinstance(source, dict) for source in sources
        ):
            return False
        source_video_ids = [str(source.get("video_id", "")) for source in sources]
        if len(set(source_video_ids)) != len(source_video_ids):
            return False
        if set(source_video_ids) != record_video_ids:
            return False
        return (
            len(records) == state.get("record_count")
            and len(record_video_ids) == state.get("video_count")
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError):
        return False
