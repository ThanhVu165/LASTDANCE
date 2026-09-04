"""Validate, normalize, and merge immutable ASR archives/checkpoints for handoff."""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from offline.artifacts import sha256_file
from offline.asr_alignment import nearest_keyframe_uid
from offline.asr_artifacts import AsrRecordEnvelope, video_set_sha256
from offline.asr_snapshot import _load_catalog
from shared.schemas.frame import FrameRecord


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, text in enumerate(stream, 1):
            if not text.strip():
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row at {path}:{line_number} must be an object")
            rows.append(value)
    return rows


def _load_inventory_durations(path: Path) -> dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    videos = payload.get("videos") if isinstance(payload, dict) else None
    if not isinstance(videos, list):
        raise ValueError("inventory must contain a videos list")
    result: dict[str, float] = {}
    for row in videos:
        video_id = str(row.get("video_id", "")).strip()
        duration = float(row.get("duration", 0.0))
        if not video_id or not math.isfinite(duration) or duration <= 0:
            raise ValueError("inventory contains an invalid video_id/duration")
        if video_id in result:
            raise ValueError(f"duplicate inventory video: {video_id}")
        result[video_id] = duration
    return result


def _validate_archive(
    jsonl_path: Path, manifest_path: Path, *, catalog_sha256: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _read_jsonl(jsonl_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"archive manifest must be an object: {manifest_path}")
    batch_id = str(manifest.get("batch_id", ""))
    ids = [str(row.get("video_id", "")) for row in rows]
    counts = Counter(str(row.get("status", "")) for row in rows)
    checks = {
        "completion gate": manifest.get("completion_gate_passed") is True,
        "catalog SHA": manifest.get("catalog_sha256") == catalog_sha256,
        "shard SHA": manifest.get("shard_sha256") == sha256_file(jsonl_path),
        "record count": manifest.get("record_count") == len(rows),
        "processed count": manifest.get("processed_videos") == len(rows),
        "success count": manifest.get("success_videos") == counts["success"],
        "silent count": manifest.get("silent_videos") == counts["silent"],
        "error count": manifest.get("error_videos") == counts["error"],
        "unique videos": len(ids) == len(set(ids)),
        "video-set SHA": manifest.get("expected_video_sha256") == video_set_sha256(ids),
        "row batch IDs": all(row.get("batch_id") == batch_id for row in rows),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"invalid completed ASR archive {batch_id}: {', '.join(failed)}")
    return rows, {
        "kind": "completed_archive", "batch_id": batch_id,
        "jsonl": str(jsonl_path), "jsonl_sha256": sha256_file(jsonl_path),
        "manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path),
        "records": len(rows),
    }


def _validate_checkpoint(
    jsonl_path: Path, state_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _read_jsonl(jsonl_path)
    state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    if not isinstance(state, dict) or not isinstance(state.get("completed"), list):
        raise ValueError("checkpoint state must contain completed video IDs")
    batch_id = str(state.get("batch_id", ""))
    ids = [str(row.get("video_id", "")) for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(map(str, state["completed"])):
        raise ValueError("checkpoint JSONL video set does not match atomic checkpoint state")
    if not batch_id or any(row.get("batch_id") != batch_id for row in rows):
        raise ValueError("checkpoint row batch IDs do not match checkpoint state")
    return rows, {
        "kind": "partial_checkpoint", "batch_id": batch_id,
        "jsonl": str(jsonl_path), "jsonl_sha256": sha256_file(jsonl_path),
        "state": str(state_path), "state_sha256": sha256_file(state_path),
        "records": len(rows),
    }


def _content_identity(envelope: AsrRecordEnvelope) -> str:
    payload = envelope.model_dump(mode="json")
    payload.pop("batch_id", None)
    payload.pop("audio_path", None)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalize_row(
    raw: Mapping[str, Any], *, frames: Sequence[FrameRecord], media_duration: float
) -> tuple[AsrRecordEnvelope, list[dict[str, Any]]]:
    value = dict(raw)
    reported_duration = float(value.get("audio_duration_seconds", value.get("duration_seconds", 0.0)))
    if not math.isfinite(reported_duration) or reported_duration <= 0:
        return AsrRecordEnvelope.model_validate(value), []
    if abs(reported_duration - media_duration) > 0.25:
        raise ValueError(
            f"audio/video duration mismatch for {value.get('video_id')}: "
            f"audio={reported_duration}, video={media_duration}"
        )
    corrections: list[dict[str, Any]] = []
    normalized_segments: list[dict[str, Any]] = []
    for segment in value.get("segments", []):
        current = dict(segment)
        start = float(current["start_time"])
        end = float(current["end_time"])
        if end > reported_duration:
            if start >= reported_duration:
                corrections.append({
                    "video_id": value.get("video_id"), "segment_id": current.get("segment_id"),
                    "action": "drop_outside_audio", "start_time": start, "end_time": end,
                    "verified_duration": reported_duration,
                })
                continue
            current["end_time"] = reported_duration
            current["keyframe_uid_nearest"] = nearest_keyframe_uid(
                str(value["video_id"]), start, reported_duration, frames
            )
            corrections.append({
                "video_id": value.get("video_id"), "segment_id": current.get("segment_id"),
                "action": "clamp_to_audio_duration", "start_time": start, "old_end_time": end,
                "new_end_time": reported_duration,
                "keyframe_uid_nearest": current["keyframe_uid_nearest"],
            })
        normalized_segments.append(current)
    value["segments"] = normalized_segments
    if value.get("status") == "success" and not normalized_segments:
        raise ValueError(f"normalization removed every success segment for {value.get('video_id')}")
    return AsrRecordEnvelope.model_validate(value), corrections


def materialize_asr_handoff(
    *, archive_pairs: Sequence[tuple[Path, Path]], checkpoint_pair: tuple[Path, Path] | None,
    catalog_path: Path, inventory_path: Path, output_jsonl: Path, audit_path: Path,
    source_revision: str,
) -> tuple[list[AsrRecordEnvelope], dict[str, Any]]:
    """Create one deterministic, strict input JSONL for the ASR snapshot builder."""
    if not source_revision.strip():
        raise ValueError("source_revision is required for an immutable handoff")
    catalog, per_video = _load_catalog(catalog_path, None)
    catalog_sha = sha256_file(catalog_path)
    frames_by_video: dict[str, list[FrameRecord]] = defaultdict(list)
    for frame in catalog.values():
        frames_by_video[frame.video_id].append(frame)
    durations = _load_inventory_durations(inventory_path)
    if set(durations) != set(per_video):
        raise ValueError("inventory and canonical catalog video sets differ")

    sources: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    for jsonl_path, manifest_path in archive_pairs:
        rows, provenance = _validate_archive(
            Path(jsonl_path), Path(manifest_path), catalog_sha256=catalog_sha
        )
        raw_rows.extend(rows)
        sources.append(provenance)
    if checkpoint_pair is not None:
        rows, provenance = _validate_checkpoint(*map(Path, checkpoint_pair))
        raw_rows.extend(rows)
        sources.append(provenance)

    normalized: dict[str, list[AsrRecordEnvelope]] = defaultdict(list)
    corrections: list[dict[str, Any]] = []
    for raw in raw_rows:
        video_id = str(raw.get("video_id", ""))
        if video_id not in frames_by_video:
            raise ValueError(f"foreign ASR video: {video_id}")
        envelope, changes = _normalize_row(
            raw, frames=frames_by_video[video_id], media_duration=durations[video_id]
        )
        normalized[video_id].append(envelope)
        corrections.extend(changes)

    accepted: list[AsrRecordEnvelope] = []
    equivalent_duplicates: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for video_id, candidates in sorted(normalized.items()):
        if len({_content_identity(row) for row in candidates}) != 1:
            conflicts.append({
                "video_id": video_id, "batch_ids": sorted(row.batch_id for row in candidates),
                "candidate_count": len(candidates), "action": "quarantine",
            })
            continue
        accepted.append(candidates[0])
        if len(candidates) > 1:
            equivalent_duplicates.append({
                "video_id": video_id, "batch_ids": sorted(row.batch_id for row in candidates),
                "discarded_equivalent_records": len(candidates) - 1,
            })

    output_jsonl, audit_path = Path(output_jsonl), Path(audit_path)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_jsonl.with_name(f".{output_jsonl.name}.staging")
    temporary.write_text(
        "".join(row.model_dump_json(by_alias=False) + "\n" for row in accepted), encoding="utf-8"
    )
    os.replace(temporary, output_jsonl)
    statuses = Counter(row.status.value for row in accepted)
    covered = sum(row.verified_complete for row in accepted)
    audit = {
        "schema_version": 1, "artifact_kind": "asr_handoff_union",
        "source_revision": source_revision, "catalog_sha256": catalog_sha,
        "inventory_sha256": sha256_file(inventory_path), "sources": sources,
        "raw_records": len(raw_rows), "raw_unique_videos": len(normalized),
        "accepted_records": len(accepted), "accepted_jsonl": str(output_jsonl),
        "accepted_jsonl_sha256": sha256_file(output_jsonl),
        "equivalent_duplicate_videos": len(equivalent_duplicates),
        "equivalent_duplicates": equivalent_duplicates,
        "conflict_videos": len(conflicts), "conflicts": conflicts,
        "timestamp_corrections": corrections,
        "success_videos": statuses["success"], "silent_videos": statuses["silent"],
        "error_videos": statuses["error"], "covered_videos": covered,
        "catalog_videos": len(per_video), "coverage_fraction": covered / len(per_video),
        "missing_or_quarantined_videos": len(per_video) - len(accepted),
        "complete": False, "production_ready": False,
    }
    temporary_audit = audit_path.with_name(f".{audit_path.name}.staging")
    temporary_audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary_audit, audit_path)
    return accepted, audit
