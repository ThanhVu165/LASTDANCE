"""Fail-closed union of UID-disjoint OCR batch shards into a development snapshot."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from offline.artifacts import sha256_file
from offline.ocr_artifacts import OcrStatus, uid_set_sha256
from offline.ocr_snapshot import (
    SnapshotBatchCoverage,
    SnapshotRecord,
    _load_catalog,
    _read_jsonl,
    load_envelope_snapshot_records,
    load_gate2_calibrated_snapshot_records,
    load_gate2_easyocr_snapshot_records,
)


SnapshotTier = Literal["craft_only", "easyocr", "vintern_calibrated", "gemini_final"]
BatchSourceFormat = Literal[
    "craft_jsonl_v1",
    "ocr_envelope_v1",
    "gate2_easyocr_dev_v1",
    "gate2_calibrated_dev_v1",
]


class IncrementalBatchSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    tier: SnapshotTier
    video_ids: tuple[str, ...]
    source_format: BatchSourceFormat
    source_jsonl: str | None = None
    updated_utc: str

    @field_validator("video_ids")
    @classmethod
    def validate_video_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values or len(values) != len(set(values)) or any(not value for value in values):
            raise ValueError("batch video_ids must be non-empty and unique")
        return values

    @field_validator("updated_utc")
    @classmethod
    def validate_updated_utc(cls, value: str) -> str:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("updated_utc must include timezone")
        return value

    @model_validator(mode="after")
    def validate_tier_source(self) -> "IncrementalBatchSource":
        allowed = {
            "craft_only": {"craft_jsonl_v1"},
            "easyocr": {"ocr_envelope_v1", "gate2_easyocr_dev_v1"},
            "vintern_calibrated": {
                "ocr_envelope_v1",
                "gate2_calibrated_dev_v1",
            },
            "gemini_final": {"ocr_envelope_v1"},
        }
        if self.source_format not in allowed[self.tier]:
            raise ValueError("batch tier/source_format combination is invalid")
        if self.source_jsonl is None and self.tier != "craft_only":
            raise ValueError("recognition tier requires source_jsonl")
        return self


class IncrementalSnapshotPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_batch_ids: tuple[str, ...] = Field(min_length=1)
    batches: tuple[IncrementalBatchSource, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_partition(self) -> "IncrementalSnapshotPlan":
        expected = list(self.expected_batch_ids)
        actual = [batch.batch_id for batch in self.batches]
        if len(expected) != len(set(expected)):
            raise ValueError("expected_batch_ids must be unique")
        if len(actual) != len(set(actual)):
            raise ValueError("batch_id values must be unique")
        if set(expected) != set(actual):
            raise ValueError("plan batches must exactly match expected_batch_ids")
        videos = [video for batch in self.batches for video in batch.video_ids]
        if len(videos) != len(set(videos)):
            raise ValueError("video_ids overlap across OCR batches")
        return self


def load_incremental_snapshot_plan(path: Path) -> IncrementalSnapshotPlan:
    return IncrementalSnapshotPlan.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _resolve_source(plan_path: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    source = Path(value)
    if not source.is_absolute():
        source = plan_path.parent / source
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"batch source JSONL does not exist: {source}")
    return source


def _load_searchable_records(
    source: Path, source_format: BatchSourceFormat
) -> list[SnapshotRecord]:
    if source_format == "ocr_envelope_v1":
        return load_envelope_snapshot_records([source])
    if source_format == "gate2_easyocr_dev_v1":
        return load_gate2_easyocr_snapshot_records(source)
    if source_format == "gate2_calibrated_dev_v1":
        return load_gate2_calibrated_snapshot_records(source)
    raise ValueError("craft JSONL does not materialize searchable OCR records")


def prepare_incremental_snapshot_union(
    *,
    plan_path: Path,
    catalog_path: Path,
    catalog_state_path: Path | None = None,
) -> tuple[list[SnapshotRecord], dict[str, SnapshotBatchCoverage], list[Path]]:
    """Validate the batch partition and return records/coverage for one SQLite build."""

    plan_path = Path(plan_path).resolve()
    plan = load_incremental_snapshot_plan(plan_path)
    if sha256_file(catalog_path) != plan.catalog_sha256:
        raise ValueError("incremental plan catalog_sha256 mismatch")
    catalog, _ = _load_catalog(catalog_path, catalog_state_path)
    catalog_videos = {record.video_id for record in catalog.values()}
    plan_videos = {video for batch in plan.batches for video in batch.video_ids}
    missing_videos = sorted(catalog_videos - plan_videos)
    foreign_videos = sorted(plan_videos - catalog_videos)
    if missing_videos or foreign_videos:
        raise ValueError(
            f"batch video partition is not exhaustive: missing={missing_videos}, "
            f"foreign={foreign_videos}"
        )

    records: list[SnapshotRecord] = []
    coverage: dict[str, SnapshotBatchCoverage] = {}
    source_paths: list[Path] = [plan_path]
    union_observed: set[int] = set()
    for batch in sorted(plan.batches, key=lambda value: value.batch_id):
        expected_uids = {
            uid for uid, frame in catalog.items() if frame.video_id in set(batch.video_ids)
        }
        source = _resolve_source(plan_path, batch.source_jsonl)
        raw_rows = _read_jsonl(source) if source is not None else []
        raw_uids: list[int] = []
        for row in raw_rows:
            if "keyframe_uid" not in row:
                raise ValueError(f"{batch.batch_id} row lacks keyframe_uid")
            raw_uids.append(int(row["keyframe_uid"]))
        if len(raw_uids) != len(set(raw_uids)):
            raise ValueError(f"duplicate keyframe_uid inside {batch.batch_id}")
        observed_uids = set(raw_uids)
        foreign_uids = sorted(observed_uids - expected_uids)
        if foreign_uids:
            raise ValueError(f"foreign UID in {batch.batch_id}: {foreign_uids[0]}")
        overlap = sorted(union_observed & observed_uids)
        if overlap:
            raise ValueError(f"UID overlap across batches: {overlap[0]}")
        union_observed.update(observed_uids)

        batch_records: list[SnapshotRecord] = []
        if batch.tier != "craft_only" and source is not None:
            batch_records = _load_searchable_records(source, batch.source_format)
            if {record.keyframe_uid for record in batch_records} != observed_uids:
                raise ValueError(f"record loader changed UID set for {batch.batch_id}")
            records.extend(batch_records)
        status_counts = Counter(record.status for record in batch_records)
        raw_errors = sum(str(row.get("status") or "") == "error" for row in raw_rows)
        raw_no_text = sum(str(row.get("status") or "") == "no_text" for row in raw_rows)
        error_count = status_counts[OcrStatus.ERROR] if batch_records else raw_errors
        processed = len(observed_uids)
        expected = len(expected_uids)
        complete = processed == expected and error_count == 0
        artifact = None
        if source is not None:
            source_paths.append(source)
            artifact = {
                "path": source.name,
                "sha256": sha256_file(source),
                "bytes": source.stat().st_size,
            }
        coverage[batch.batch_id] = SnapshotBatchCoverage(
            batch_id=batch.batch_id,
            tier=batch.tier,
            complete=complete,
            searchable=batch.tier != "craft_only" and processed > 0,
            video_ids=tuple(sorted(batch.video_ids)),
            expected_keyframes=expected,
            frames=processed,
            processed_keyframes=processed,
            success_keyframes=(
                status_counts[OcrStatus.SUCCESS] if batch.tier != "craft_only" else 0
            ),
            no_text_keyframes=(status_counts[OcrStatus.NO_TEXT] if batch_records else raw_no_text),
            error_keyframes=error_count,
            pending_recognition_keyframes=(
                processed - raw_no_text - raw_errors if batch.tier == "craft_only" else 0
            ),
            missing_keyframes=expected - processed,
            coverage_fraction=processed / expected if expected else 0.0,
            updated_utc=batch.updated_utc,
            assigned_uid_sha256=uid_set_sha256(expected_uids),
            observed_uid_sha256=uid_set_sha256(observed_uids),
            source_format=batch.source_format,
            source_artifact=artifact,
        )
    return records, coverage, source_paths
