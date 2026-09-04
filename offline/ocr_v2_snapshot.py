"""Fail-closed union of OCR v2 production shards into an immutable dev snapshot.

This module deliberately does not coerce OCR v2 records into the legacy
``OcrRecordEnvelope``.  VietOCR/Paddle provenance remains region-level while the
unchanged five-column FTS projection contains only selected terminal text.
"""

from __future__ import annotations

import csv
import hashlib
import heapq
import io
import json
import math
import os
import re
import shutil
import sqlite3
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from offline.artifacts import sha256_file
from offline.catalog import FRAME_COLUMNS, validate_frames_catalog
from shared.schemas.frame import FrameRecord
from shared.schemas.ocr import OcrResult


CONTRACT = "ocr-v2-recognition-worker-v1"
BATCH_IDS = tuple(f"batch-{index:02d}" for index in range(1, 10))
MODEL_NAMES = ("vietocr", "paddle")
RESULT_MEMBERS = {
    "report.json",
    "run-signature.json",
    "predictions.jsonl",
    "frame-selections.jsonl",
    "residual.jsonl",
    "SHA256SUMS",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SNAPSHOT_ID = re.compile(r"^ocr-snapshot-\d{8}T\d{6}Z-[0-9a-f]{12}$")
_RESULT_PATH = re.compile(
    r"^ocr/archives/(batch-\d{2})/ocr-v2/([0-9a-f]{64})/production/"
    r"results-([0-9a-f]{64})\.zip$"
)
_REPORT_PATH = re.compile(
    r"^ocr/archives/(batch-\d{2})/ocr-v2/([0-9a-f]{64})/production/reports/"
    r"summary-([0-9a-f]{64})\.json$"
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def uid_set_sha256(values: set[int] | list[int]) -> str:
    return sha256_bytes("".join(f"{value}\n" for value in sorted(values)).encode())


def _require_relative_posix(value: str) -> str:
    if "\\" in value:
        raise ValueError("artifact paths must use POSIX separators")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or ":" in value:
        raise ValueError("artifact path must be relative and cannot traverse parents")
    return value


class OcrV2SourceArtifact(BaseModel):
    """One content-addressed production ZIP and its external summary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_id: str = Field(pattern=r"^batch-\d{2}$")
    worker: str = Field(pattern=r"^[1-4]$")
    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    signature: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_path: str
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_bytes: int = Field(gt=0)
    report_path: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    report_bytes: int = Field(gt=0)
    source_commit: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    source_committed_utc: str | None = None
    equivalent_result_sha256: tuple[str, ...] = ()
    equivalent_report_sha256: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_paths(self) -> "OcrV2SourceArtifact":
        _require_relative_posix(self.result_path)
        _require_relative_posix(self.report_path)
        result_match = _RESULT_PATH.fullmatch(self.result_path)
        report_match = _REPORT_PATH.fullmatch(self.report_path)
        if result_match is None or report_match is None:
            raise ValueError("source paths do not use the OCR v2 production namespace")
        for match, digest in (
            (result_match, self.result_sha256),
            (report_match, self.report_sha256),
        ):
            if match.group(1) != self.batch_id or match.group(2) != self.run_id:
                raise ValueError("source path batch/run_id mismatch")
            if match.group(3) != digest:
                raise ValueError("source path content digest mismatch")
        if (self.source_commit is None) != (self.source_committed_utc is None):
            raise ValueError("source commit and timestamp must be recorded together")
        if self.source_committed_utc is not None:
            timestamp = datetime.fromisoformat(
                self.source_committed_utc.replace("Z", "+00:00")
            )
            if timestamp.tzinfo is None:
                raise ValueError("source_committed_utc must include a timezone")
        for digests, selected in (
            (self.equivalent_result_sha256, self.result_sha256),
            (self.equivalent_report_sha256, self.report_sha256),
        ):
            if (
                len(digests) != len(set(digests))
                or selected in digests
                or any(_SHA256.fullmatch(digest) is None for digest in digests)
            ):
                raise ValueError("invalid equivalent artifact digest evidence")
        if len(self.equivalent_result_sha256) != len(
            self.equivalent_report_sha256
        ):
            raise ValueError("equivalent result/report evidence must be paired")
        return self


class OcrV2SourceManifest(BaseModel):
    """Pinned HF revision plus exactly nine selected production artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact_kind: Literal["ocr_v2_production_source_set"] = (
        "ocr_v2_production_source_set"
    )
    created_utc: str
    repository: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    worker_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifacts: tuple[OcrV2SourceArtifact, ...]

    @model_validator(mode="after")
    def validate_complete_source_set(self) -> "OcrV2SourceManifest":
        batches = [artifact.batch_id for artifact in self.artifacts]
        paths = [artifact.result_path for artifact in self.artifacts]
        if len(self.artifacts) != len(BATCH_IDS) or set(batches) != set(BATCH_IDS):
            raise ValueError("source manifest must contain exactly batch-01 through batch-09")
        if len(batches) != len(set(batches)) or len(paths) != len(set(paths)):
            raise ValueError("source manifest contains duplicate batches or result paths")
        if datetime.fromisoformat(self.created_utc.replace("Z", "+00:00")).tzinfo is None:
            raise ValueError("created_utc must include a timezone")
        return self


class OcrV2CoverageUnit(BaseModel):
    """Recognition coverage; it never claims full Offline publishing readiness."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    expected_keyframes: int = Field(ge=0)
    processed_keyframes: int = Field(ge=0)
    success_keyframes: int = Field(ge=0)
    no_text_keyframes: int = Field(ge=0)
    error_keyframes: int = Field(ge=0)
    missing_keyframes: int = Field(ge=0)
    residual_frames: int = Field(ge=0)
    residual_regions: int = Field(ge=0)
    selected_region_engine_counts: dict[str, int]
    assigned_uid_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observed_uid_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    recognition_coverage_complete: bool
    complete: Literal[False] = False
    production_ready: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> "OcrV2CoverageUnit":
        terminal = self.success_keyframes + self.no_text_keyframes + self.error_keyframes
        if terminal != self.processed_keyframes:
            raise ValueError("terminal status counts do not sum to processed_keyframes")
        if self.missing_keyframes != self.expected_keyframes - self.processed_keyframes:
            raise ValueError("missing_keyframes does not match expected minus processed")
        gate = self.missing_keyframes == 0 and (
            self.assigned_uid_sha256 == self.observed_uid_sha256
        )
        if self.recognition_coverage_complete != gate:
            raise ValueError("recognition_coverage_complete does not match UID coverage")
        if set(self.selected_region_engine_counts) - {"vietocr", "paddle", "unresolved"}:
            raise ValueError("unknown OCR v2 selected engine")
        if any(value < 0 for value in self.selected_region_engine_counts.values()):
            raise ValueError("selected engine counts must be non-negative")
        return self


class OcrV2BatchCoverage(OcrV2CoverageUnit):
    batch_id: str = Field(pattern=r"^batch-\d{2}$")
    worker: str = Field(pattern=r"^[1-4]$")
    video_ids: tuple[str, ...]
    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    signature: str = Field(pattern=r"^[0-9a-f]{64}$")
    tasks_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    regions: int = Field(ge=0)
    predictions: int = Field(ge=0)
    source_recognition_complete: Literal[True] = True
    resume_with_new_work: bool

    @model_validator(mode="after")
    def validate_batch_details(self) -> "OcrV2BatchCoverage":
        if not self.video_ids or len(self.video_ids) != len(set(self.video_ids)):
            raise ValueError("batch video_ids must be non-empty and unique")
        if sum(self.selected_region_engine_counts.values()) != self.regions:
            raise ValueError("batch selected engine counts do not sum to regions")
        return self


class OcrV2VideoCoverage(OcrV2CoverageUnit):
    video_id: str = Field(min_length=1)
    batch_id: str = Field(pattern=r"^batch-\d{2}$")


class OcrV2SnapshotManifest(BaseModel):
    """Immutable OCR v2 development snapshot coverage sidecar."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[3] = 3
    artifact_kind: Literal["ocr_sqlite_snapshot"] = "ocr_sqlite_snapshot"
    snapshot_id: str = Field(pattern=_SNAPSHOT_ID.pattern)
    created_utc: str
    immutable: Literal[True] = True
    complete: Literal[False] = False
    production_ready: Literal[False] = False
    intended_use: Literal["online_development_only"] = "online_development_only"
    source_format: Literal["ocr_v2_batch_union_v1"] = "ocr_v2_batch_union_v1"
    materialized_text_policy: str
    builder_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalog_path: str
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalog_state_path: str
    catalog_state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalog_records: int = Field(gt=0)
    catalog_videos: int = Field(gt=0)
    worker_plan_path: str
    worker_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    worker_plan_semantic_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_manifest_path: str
    source_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_repository: str
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_artifacts: tuple[OcrV2SourceArtifact, ...]
    totals: OcrV2CoverageUnit
    batches: dict[str, OcrV2BatchCoverage]
    videos: dict[str, OcrV2VideoCoverage]
    sqlite_path: Literal["ocr.sqlite"] = "ocr.sqlite"
    sqlite_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sqlite_bytes: int = Field(gt=0)
    fts_rows: int = Field(ge=0)
    fts_probe: dict[str, Any]

    @model_validator(mode="after")
    def validate_maps_and_totals(self) -> "OcrV2SnapshotManifest":
        if set(self.batches) != set(BATCH_IDS):
            raise ValueError("coverage must contain exactly nine batches")
        if any(key != value.batch_id for key, value in self.batches.items()):
            raise ValueError("batch coverage map key mismatch")
        if any(key != value.video_id for key, value in self.videos.items()):
            raise ValueError("video coverage map key mismatch")
        if self.fts_rows != self.totals.success_keyframes:
            raise ValueError("FTS row count must equal successful keyframes")
        if datetime.fromisoformat(self.created_utc.replace("Z", "+00:00")).tzinfo is None:
            raise ValueError("created_utc must include a timezone")
        return self


def allocate_batches(batches: dict[str, Any]) -> tuple[dict[str, list[str]], dict[str, int]]:
    if set(batches) != set(BATCH_IDS):
        raise ValueError("worker plan needs exactly batch-01 through batch-09")
    loads = {str(index): 0 for index in range(1, 5)}
    assignments = {str(index): [] for index in range(1, 5)}
    for batch_id in sorted(batches, key=lambda value: (-int(batches[value]["regions"]), value)):
        worker = min(loads, key=lambda value: (loads[value], int(value)))
        assignments[worker].append(batch_id)
        loads[worker] += int(batches[batch_id]["regions"])
    return assignments, loads


def load_worker_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise ValueError("worker plan must be a JSON object")
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan.get("contract") != CONTRACT or sha256_bytes(canonical_json(body)) != plan.get(
        "plan_sha256"
    ):
        raise ValueError("worker plan signature mismatch")
    if plan.get("catalog_source") != "kaggle_input":
        raise ValueError("worker plan does not pin the attached Kaggle catalog")
    if not isinstance(plan.get("repo"), str) or not plan["repo"]:
        raise ValueError("worker plan repo is missing")
    if _REVISION.fullmatch(str(plan.get("input_revision"))) is None:
        raise ValueError("worker plan input revision must be an immutable commit")
    assignments, loads = allocate_batches(plan.get("batches", {}))
    if plan.get("assignments") != assignments or plan.get("worker_regions") != loads:
        raise ValueError("worker assignment drift")
    assigned = [batch for values in assignments.values() for batch in values]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(BATCH_IDS):
        raise ValueError("worker assignments are not disjoint/exhaustive")
    return plan


def _load_catalog(
    catalog_path: Path,
    catalog_state_path: Path | None,
) -> tuple[dict[int, FrameRecord], dict[str, set[int]], Path]:
    catalog_path = Path(catalog_path)
    state_path = catalog_state_path or catalog_path.with_name(catalog_path.name + ".state.json")
    if not validate_frames_catalog(catalog_path, state_path):
        raise RuntimeError("frames.csv/state validation failed")
    records: dict[int, FrameRecord] = {}
    by_video: dict[str, set[int]] = defaultdict(set)
    with catalog_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FRAME_COLUMNS:
            raise RuntimeError("frames.csv columns do not match canonical schema")
        for raw in reader:
            record = FrameRecord(**{**raw, "window_id": raw["window_id"] or None})
            if record.keyframe_uid in records:
                raise ValueError(f"duplicate catalog UID: {record.keyframe_uid}")
            records[record.keyframe_uid] = record
            by_video[record.video_id].add(record.keyframe_uid)
    return records, dict(by_video), state_path


def validate_plan_against_catalog(
    plan: dict[str, Any],
    catalog_path: Path,
    catalog_state_path: Path | None = None,
) -> tuple[dict[int, FrameRecord], dict[str, set[int]], dict[str, set[int]], Path]:
    catalog, by_video, state_path = _load_catalog(catalog_path, catalog_state_path)
    if plan.get("catalog_sha256") != sha256_file(catalog_path):
        raise ValueError("worker plan catalog SHA-256 differs from frames.csv")
    if plan.get("catalog_state_sha256") != sha256_file(state_path):
        raise ValueError("worker plan catalog state SHA-256 differs from local state")
    batches: dict[str, set[int]] = {}
    seen_videos: set[str] = set()
    for batch_id in BATCH_IDS:
        evidence = plan["batches"][batch_id]
        video_ids = evidence.get("video_ids")
        if not isinstance(video_ids, list) or not video_ids or len(video_ids) != len(set(video_ids)):
            raise ValueError(f"{batch_id} has invalid video_ids")
        if seen_videos.intersection(video_ids):
            raise ValueError("worker plan video partitions overlap")
        if any(video_id not in by_video for video_id in video_ids):
            raise ValueError(f"{batch_id} contains a foreign video")
        seen_videos.update(video_ids)
        uids = set().union(*(by_video[video_id] for video_id in video_ids))
        if len(uids) != int(evidence.get("frames", -1)):
            raise ValueError(f"{batch_id} frame count differs from catalog partition")
        if uid_set_sha256(uids) != evidence.get("uid_sha256"):
            raise ValueError(f"{batch_id} assigned UID hash differs from catalog partition")
        batches[batch_id] = uids
    if seen_videos != set(by_video) or set().union(*batches.values()) != set(catalog):
        raise ValueError("worker plan does not exhaust the catalog")
    return catalog, by_video, batches, state_path


def worker_for_batch(plan: dict[str, Any]) -> dict[str, str]:
    return {
        batch_id: worker
        for worker, batches in plan["assignments"].items()
        for batch_id in batches
    }


def _validate_zip_layout(
    archive: zipfile.ZipFile, *, verify_members: bool
) -> dict[str, str]:
    names = archive.namelist()
    if len(names) != len(set(names)) or set(names) != RESULT_MEMBERS:
        raise ValueError("OCR v2 result ZIP has unexpected or duplicate members")
    checks: dict[str, str] = {}
    for line in archive.read("SHA256SUMS").decode("ascii").splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError("invalid result ZIP SHA256SUMS row")
        digest, name = fields
        if _SHA256.fullmatch(digest) is None or name in checks:
            raise ValueError("invalid or duplicate result ZIP checksum")
        _require_relative_posix(name)
        checks[name] = digest
    if set(checks) != RESULT_MEMBERS - {"SHA256SUMS"}:
        raise ValueError("result ZIP checksum coverage mismatch")
    if verify_members:
        for name, expected in checks.items():
            digest = hashlib.sha256()
            with archive.open(name) as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError(f"result ZIP checksum mismatch: {name}")
    return checks


def result_member_hashes(path: Path, *, verify_members: bool = True) -> dict[str, str]:
    """Return declared member hashes after optionally verifying every member."""

    with zipfile.ZipFile(path) as archive:
        return _validate_zip_layout(archive, verify_members=verify_members)


def read_result_identity(path: Path, *, verify_members: bool = False) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    """Read small identity members, optionally hashing every decompressed member."""

    with zipfile.ZipFile(path) as archive:
        _validate_zip_layout(archive, verify_members=verify_members)
        report_bytes = archive.read("report.json")
        report = json.loads(report_bytes)
        run_signature = json.loads(archive.read("run-signature.json"))
    if not isinstance(report, dict) or not isinstance(run_signature, dict):
        raise ValueError("result report/signature must be JSON objects")
    return report, run_signature, report_bytes


def load_source_manifest(path: Path) -> OcrV2SourceManifest:
    return OcrV2SourceManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))


@dataclass
class _MutableCoverage:
    expected: set[int]
    observed: set[int]
    statuses: Counter[str]
    engines: Counter[str]
    residual_frames: int = 0
    residual_regions: int = 0


def _new_coverage(expected: set[int]) -> _MutableCoverage:
    return _MutableCoverage(set(expected), set(), Counter(), Counter())


def _coverage_model(values: _MutableCoverage) -> dict[str, Any]:
    missing = len(values.expected - values.observed)
    assigned_hash = uid_set_sha256(values.expected)
    observed_hash = uid_set_sha256(values.observed)
    return {
        "expected_keyframes": len(values.expected),
        "processed_keyframes": len(values.observed),
        "success_keyframes": values.statuses["success"],
        "no_text_keyframes": values.statuses["no_text"],
        "error_keyframes": values.statuses["error"],
        "missing_keyframes": missing,
        "residual_frames": values.residual_frames,
        "residual_regions": values.residual_regions,
        "selected_region_engine_counts": dict(sorted(values.engines.items())),
        "assigned_uid_sha256": assigned_hash,
        "observed_uid_sha256": observed_hash,
        "recognition_coverage_complete": missing == 0 and assigned_hash == observed_hash,
        "complete": False,
        "production_ready": False,
    }


def _jsonl_rows(archive: zipfile.ZipFile, member: str):
    with archive.open(member) as binary:
        with io.TextIOWrapper(binary, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL in {member}:{line_number}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL row is not an object in {member}:{line_number}")
                yield line_number, row


def _finite_score(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    score = float(value)
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return score


def _validate_report_identity(
    report: dict[str, Any],
    run_signature: dict[str, Any],
    source: OcrV2SourceArtifact,
    plan: dict[str, Any],
) -> None:
    if (
        report.get("contract") != CONTRACT
        or report.get("mode") != "production"
        or report.get("batch") != source.batch_id
        or str(report.get("worker")) != source.worker
        or report.get("run_id") != source.run_id
        or report.get("signature") != source.signature
        or report.get("recognition_complete") is not True
        or report.get("complete") is not False
        or report.get("production_ready") is not False
    ):
        raise ValueError(f"{source.batch_id} report identity/readiness mismatch")
    resources = run_signature.get("resources")
    if not isinstance(resources, dict) or sha256_bytes(canonical_json(resources)) != source.run_id:
        raise ValueError(f"{source.batch_id} run_id does not match run-signature resources")
    if (
        resources.get("contract") != CONTRACT
        or str(resources.get("worker")) != source.worker
        or resources.get("plan_sha256") != plan["plan_sha256"]
        or run_signature.get("signature") != source.signature
        or run_signature.get("tasks_sha256") != report.get("tasks_sha256")
        or run_signature.get("batch") != source.batch_id
        or run_signature.get("mode") != "production"
    ):
        raise ValueError(f"{source.batch_id} run-signature drift")
    expected_signature = sha256_bytes(
        canonical_json(
            {
                "run_id": source.run_id,
                "batch": source.batch_id,
                "mode": "production",
                "tasks": report.get("tasks_sha256"),
            }
        )
    )
    if expected_signature != source.signature:
        raise ValueError(f"{source.batch_id} signature does not match tasks/run identity")


_FRAME_KEYS = {
    "artifact_kind",
    "batch_id",
    "signature",
    "keyframe_uid",
    "video_id",
    "frame_id",
    "source_image",
    "status",
    "result",
    "regions",
    "source_status",
    "source_error",
    "complete",
    "production_ready",
}
_TASK_KEYS = {
    "region_id",
    "keyframe_uid",
    "video_id",
    "frame_id",
    "shot_id",
    "source_image",
    "image_key",
    "source_sha256",
    "bbox_px",
    "easyocr_text",
    "task_sha256",
}
_REGION_KEYS = _TASK_KEYS | {
    "selected_text",
    "selected_confidence",
    "selected_engine",
    "selection",
    "residual_reasons",
}
_PREDICTION_KEYS = {"model", "region_id", "task_sha256", "signature", "text", "confidence"}


def _validate_batch_archive(
    *,
    archive_path: Path,
    external_report_path: Path,
    source: OcrV2SourceArtifact,
    plan: dict[str, Any],
    catalog: dict[int, FrameRecord],
    coverage: _MutableCoverage,
    video_coverage: dict[str, _MutableCoverage],
    connection: sqlite3.Connection,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    if archive_path.stat().st_size != source.result_bytes or sha256_file(archive_path) != source.result_sha256:
        raise ValueError(f"{source.batch_id} result ZIP size/SHA-256 mismatch")
    if external_report_path.stat().st_size != source.report_bytes or sha256_file(external_report_path) != source.report_sha256:
        raise ValueError(f"{source.batch_id} external report size/SHA-256 mismatch")

    if progress is not None:
        progress(f"{source.batch_id} verify ZIP members/checksums")
    report, run_signature, internal_report = read_result_identity(
        archive_path, verify_members=True
    )
    if external_report_path.read_bytes() != internal_report:
        raise ValueError(f"{source.batch_id} external report differs from ZIP report")
    _validate_report_identity(report, run_signature, source, plan)

    region_tasks: dict[str, str] = {}
    expected_residuals: dict[str, str] = {}
    task_digest = hashlib.sha256()
    statuses: Counter[str] = Counter()
    engines: Counter[str] = Counter()
    residual_frame_uids: set[int] = set()
    residual_shots: set[tuple[str, str]] = set()
    observed: set[int] = set()
    regions_total = 0

    with zipfile.ZipFile(archive_path) as archive:
        for _, row in _jsonl_rows(archive, "frame-selections.jsonl"):
            if set(row) != _FRAME_KEYS:
                raise ValueError(f"{source.batch_id} frame selection schema drift")
            uid = row.get("keyframe_uid")
            if isinstance(uid, bool) or not isinstance(uid, int) or uid not in coverage.expected:
                raise ValueError(f"{source.batch_id} has a foreign/non-integer UID")
            if uid in observed:
                raise ValueError(f"{source.batch_id} has duplicate UID {uid}")
            observed.add(uid)
            canonical = catalog[uid]
            if row.get("video_id") != canonical.video_id or row.get("frame_id") != canonical.frame_id:
                raise ValueError(f"{source.batch_id} catalog mapping drift for UID {uid}")
            if (
                row.get("artifact_kind") != "ocr_v2_frame_selection_v1"
                or row.get("batch_id") != source.batch_id
                or row.get("signature") != source.signature
                or row.get("complete") is not False
                or row.get("production_ready") is not False
            ):
                raise ValueError(f"{source.batch_id} frame identity/readiness drift")
            source_image = str(row.get("source_image") or "")
            _require_relative_posix(source_image)
            source_status = row.get("source_status")
            if source_status not in {"text_detected", "no_text", "error"}:
                raise ValueError(f"{source.batch_id} frame has invalid source_status")
            regions = row.get("regions")
            if not isinstance(regions, list):
                raise ValueError(f"{source.batch_id} frame regions must be a list")

            accepted_text: list[str] = []
            accepted_confidence: list[float] = []
            frame_residuals = 0
            for region in regions:
                if not isinstance(region, dict) or set(region) != _REGION_KEYS:
                    raise ValueError(f"{source.batch_id} region selection schema drift")
                if (
                    region.get("keyframe_uid") != uid
                    or region.get("video_id") != canonical.video_id
                    or region.get("frame_id") != canonical.frame_id
                    or region.get("shot_id") != canonical.shot_id
                    or region.get("source_image") != source_image
                ):
                    raise ValueError(f"{source.batch_id} region/catalog mapping drift")
                region_id = region.get("region_id")
                if not isinstance(region_id, str) or not region_id or region_id in region_tasks:
                    raise ValueError(f"{source.batch_id} duplicate/invalid region_id")
                if region.get("image_key") != f"{canonical.video_id}/{PurePosixPath(source_image).name}":
                    raise ValueError(f"{source.batch_id} region image_key drift")
                if _SHA256.fullmatch(str(region.get("source_sha256"))) is None:
                    raise ValueError(f"{source.batch_id} region source SHA-256 is invalid")
                bbox = region.get("bbox_px")
                if not isinstance(bbox, list) or len(bbox) != 8 or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in bbox
                ):
                    raise ValueError(f"{source.batch_id} region bbox_px is invalid")
                if not isinstance(region.get("easyocr_text"), str):
                    raise ValueError(f"{source.batch_id} region cache text is invalid")
                task = {key: region[key] for key in _TASK_KEYS if key != "task_sha256"}
                task_sha = sha256_bytes(canonical_json(task))
                if task_sha != region.get("task_sha256"):
                    raise ValueError(f"{source.batch_id} region task SHA-256 drift")
                region_tasks[region_id] = task_sha
                task_digest.update(canonical_json({**task, "task_sha256": task_sha}) + b"\n")
                regions_total += 1

                reasons = region.get("residual_reasons")
                if not isinstance(reasons, list) or any(
                    not isinstance(reason, str) or not reason for reason in reasons
                ) or len(reasons) != len(set(reasons)):
                    raise ValueError(f"{source.batch_id} residual reasons are invalid")
                engine = region.get("selected_engine")
                text = region.get("selected_text")
                confidence = region.get("selected_confidence")
                selection = region.get("selection")
                if engine is None:
                    if text is not None or confidence is not None or selection != "unresolved" or not reasons:
                        raise ValueError(f"{source.batch_id} unresolved selection is inconsistent")
                    engines["unresolved"] += 1
                else:
                    if engine not in MODEL_NAMES or not isinstance(text, str) or not text.strip():
                        raise ValueError(f"{source.batch_id} selected OCR result is invalid")
                    selected_score = _finite_score(confidence, "selected_confidence")
                    if engine == "vietocr" and selection != "vietocr_default":
                        raise ValueError(f"{source.batch_id} VietOCR selection provenance drift")
                    if engine == "paddle" and selection not in {
                        "numeric_cache_or_viet_guard",
                        "ascii_cache_guard",
                    }:
                        raise ValueError(f"{source.batch_id} Paddle selection provenance drift")
                    engines[engine] += 1
                    accepted_text.append(text)
                    accepted_confidence.append(selected_score)
                if reasons:
                    frame_residuals += 1
                    expected_residuals[region_id] = sha256_bytes(canonical_json(region))
                    residual_shots.add((canonical.video_id, canonical.shot_id))

            if source_status == "no_text" and regions:
                raise ValueError(f"{source.batch_id} CRAFT no_text frame contains regions")
            status = row.get("status")
            expected_status = "success" if accepted_text else (
                "no_text" if source_status == "no_text" else "error"
            )
            if status != expected_status:
                raise ValueError(f"{source.batch_id} terminal frame status is inconsistent")
            result = row.get("result")
            if status == "success":
                parsed = OcrResult.model_validate(result)
                if (
                    parsed.frame_id != canonical.frame_id
                    or parsed.detected_text != accepted_text
                    or parsed.language != "mixed"
                    or any(
                        len(box) != 8 or any(not math.isfinite(value) or not 0 <= value <= 1 for value in box)
                        for box in parsed.bbox
                    )
                ):
                    raise ValueError(f"{source.batch_id} OcrResult content/mapping drift")
                weights = [max(1, len("".join(text.split()))) for text in accepted_text]
                expected_confidence = sum(
                    weight * score for weight, score in zip(weights, accepted_confidence)
                ) / sum(weights)
                if not math.isclose(parsed.confidence, expected_confidence, abs_tol=1e-12):
                    raise ValueError(f"{source.batch_id} frame confidence aggregation drift")
                connection.execute(
                    "INSERT INTO ocr_fts(video_id,keyframe_uid,detected_text,language,confidence) "
                    "VALUES(?,?,?,?,?)",
                    (
                        canonical.video_id,
                        uid,
                        "\n".join(parsed.detected_text),
                        parsed.language,
                        parsed.confidence,
                    ),
                )
            elif result is not None:
                raise ValueError(f"{source.batch_id} no_text/error frame carries OcrResult")

            statuses[status] += 1
            coverage.observed.add(uid)
            coverage.statuses[status] += 1
            video = video_coverage[canonical.video_id]
            video.observed.add(uid)
            video.statuses[status] += 1
            if frame_residuals:
                residual_frame_uids.add(uid)
                coverage.residual_frames += 1
                video.residual_frames += 1
            coverage.residual_regions += frame_residuals
            video.residual_regions += frame_residuals
            # Per-frame engine counts are accumulated below from the region list.
            frame_engines = Counter(
                (region["selected_engine"] or "unresolved") for region in regions
            )
            coverage.engines.update(frame_engines)
            video.engines.update(frame_engines)
            if progress is not None and len(observed) % 10_000 == 0:
                progress(
                    f"{source.batch_id} frames={len(observed)}/{len(coverage.expected)} "
                    f"regions={regions_total}"
                )

        if observed != coverage.expected:
            missing = sorted(coverage.expected - observed)[:5]
            foreign = sorted(observed - coverage.expected)[:5]
            raise ValueError(
                f"{source.batch_id} UID coverage mismatch; missing={missing}, foreign={foreign}"
            )
        if task_digest.hexdigest() != report.get("tasks_sha256"):
            raise ValueError(f"{source.batch_id} task manifest SHA-256 drift")
        sample = heapq.nsmallest(
            256, region_tasks, key=lambda region_id: sha256_bytes(region_id.encode())
        )
        sample_sha = sha256_bytes(canonical_json([region_tasks[region_id] for region_id in sample]))
        if sample_sha != report.get("sample_task_sha256"):
            raise ValueError(f"{source.batch_id} sample task SHA-256 drift")

        seen_predictions = {model: set() for model in MODEL_NAMES}
        prediction_counts: Counter[str] = Counter()
        for _, row in _jsonl_rows(archive, "predictions.jsonl"):
            if set(row) != _PREDICTION_KEYS:
                raise ValueError(f"{source.batch_id} prediction schema drift")
            model = row.get("model")
            region_id = row.get("region_id")
            if model not in MODEL_NAMES or region_id not in region_tasks:
                raise ValueError(f"{source.batch_id} prediction references foreign model/region")
            if region_id in seen_predictions[model]:
                raise ValueError(f"{source.batch_id} duplicate {model} prediction")
            if (
                row.get("task_sha256") != region_tasks[region_id]
                or row.get("signature") != source.signature
                or not isinstance(row.get("text"), str)
            ):
                raise ValueError(f"{source.batch_id} prediction provenance drift")
            if row.get("confidence") is not None:
                _finite_score(row["confidence"], "prediction confidence")
            seen_predictions[model].add(region_id)
            prediction_counts[model] += 1
            if progress is not None and sum(prediction_counts.values()) % 100_000 == 0:
                progress(
                    f"{source.batch_id} predictions={sum(prediction_counts.values())}/"
                    f"{report.get('predictions')}"
                )
        if len(seen_predictions["vietocr"]) != len(region_tasks):
            raise ValueError(f"{source.batch_id} VietOCR prediction coverage is incomplete")

        seen_residuals: set[str] = set()
        for _, row in _jsonl_rows(archive, "residual.jsonl"):
            region_id = row.get("region_id")
            if region_id in seen_residuals or region_id not in expected_residuals:
                raise ValueError(f"{source.batch_id} duplicate/foreign residual region")
            if sha256_bytes(canonical_json(row)) != expected_residuals[region_id]:
                raise ValueError(f"{source.batch_id} residual sidecar content drift")
            seen_residuals.add(region_id)
        if len(seen_residuals) != len(expected_residuals):
            raise ValueError(f"{source.batch_id} residual sidecar coverage is incomplete")

    report_status = report.get("status")
    if not isinstance(report_status, dict) or {str(k): int(v) for k, v in report_status.items()} != dict(statuses):
        raise ValueError(f"{source.batch_id} report status counts mismatch")
    expected_counts = {
        "frames": len(observed),
        "regions": regions_total,
        "predictions": sum(prediction_counts.values()),
        "residual_regions": len(expected_residuals),
        "residual_frames": len(residual_frame_uids),
        "residual_shots": len(residual_shots),
    }
    for field, actual in expected_counts.items():
        if report.get(field) != actual:
            raise ValueError(f"{source.batch_id} report {field} mismatch")
    if report.get("model_calls_saved") != dict(prediction_counts):
        raise ValueError(f"{source.batch_id} report model call counts mismatch")
    other_calls = report.get("other_model_calls")
    if (
        not isinstance(other_calls, dict)
        or set(other_calls) != {"easyocr", "vintern", "gemini"}
        or any(value != 0 for value in other_calls.values())
    ):
        raise ValueError(f"{source.batch_id} report claims forbidden model calls")
    phases = report.get("phases")
    if not isinstance(phases, dict) or set(phases) != set(MODEL_NAMES):
        raise ValueError(f"{source.batch_id} report phase set mismatch")
    for model in MODEL_NAMES:
        phase = phases[model]
        if (
            not isinstance(phase, dict)
            or phase.get("model") != model
            or phase.get("completed") != prediction_counts[model]
            or phase.get("expected") != prediction_counts[model]
            or int(phase.get("new_predictions", -1)) + int(phase.get("resumed_predictions", -1))
            != prediction_counts[model]
        ):
            raise ValueError(f"{source.batch_id} {model} phase coverage mismatch")
    plan_evidence = plan["batches"][source.batch_id]
    if regions_total != plan_evidence["regions"] or len(observed) != plan_evidence["frames"]:
        raise ValueError(f"{source.batch_id} result counts differ from worker plan")
    return report, dict(engines)


def _create_fts(connection: sqlite3.Connection) -> None:
    connection.execute(
        """CREATE VIRTUAL TABLE ocr_fts USING fts5(
            video_id UNINDEXED,
            keyframe_uid UNINDEXED,
            detected_text,
            language UNINDEXED,
            confidence UNINDEXED
        )"""
    )


def _fts_probe(connection: sqlite3.Connection) -> dict[str, Any]:
    row = connection.execute(
        "SELECT keyframe_uid, detected_text FROM ocr_fts ORDER BY video_id, keyframe_uid LIMIT 1"
    ).fetchone()
    if row is None:
        return {"executed": False, "reason": "no_success_rows"}
    uid, text = row
    tokens = [token for token in re.findall(r"[^\W_]+", str(text), flags=re.UNICODE) if len(token) >= 2]
    if not tokens:
        return {"executed": False, "reason": "no_probe_token"}
    token = max(tokens, key=len)
    query = f'"{token.replace(chr(34), chr(34) * 2)}"'
    matched = connection.execute(
        "SELECT 1 FROM ocr_fts WHERE ocr_fts MATCH ? AND keyframe_uid=? LIMIT 1",
        (query, uid),
    ).fetchone()
    if matched is None:
        raise RuntimeError("FTS5 probe did not return its source keyframe")
    return {"executed": True, "token": token, "keyframe_uid": int(uid)}


def _source_local_path(root: Path, relative: str) -> Path:
    _require_relative_posix(relative)
    root = root.resolve()
    candidate = root.joinpath(*PurePosixPath(relative).parts).resolve()
    if root != candidate and root not in candidate.parents:
        raise ValueError("source manifest path escapes its root")
    if os.name == "nt" and candidate.is_absolute():
        # Content-addressed HF namespaces can exceed the legacy Win32 MAX_PATH.
        # Prefix only after the ordinary resolved-path containment check above.
        return Path("\\\\?\\" + str(candidate))
    return candidate


def build_ocr_v2_snapshot(
    *,
    catalog_path: Path,
    catalog_state_path: Path | None,
    worker_plan_path: Path,
    source_manifest_path: Path,
    source_root: Path | None,
    output_root: Path,
    created_utc: datetime | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[Path, OcrV2SnapshotManifest]:
    """Validate all nine shards and atomically create a development-only SQLite."""

    plan = load_worker_plan(worker_plan_path)
    catalog, catalog_by_video, expected_batches, state_path = validate_plan_against_catalog(
        plan, catalog_path, catalog_state_path
    )
    sources = load_source_manifest(source_manifest_path)
    if sources.repository != plan["repo"] or sources.worker_plan_sha256 != plan["plan_sha256"]:
        raise ValueError("source manifest repository/worker-plan mismatch")
    batch_workers = worker_for_batch(plan)
    for source in sources.artifacts:
        if source.worker != batch_workers[source.batch_id]:
            raise ValueError(f"{source.batch_id} source worker differs from worker plan")

    root = Path(source_root) if source_root is not None else Path(source_manifest_path).parent
    artifacts_by_batch = {artifact.batch_id: artifact for artifact in sources.artifacts}
    builder_sha = sha256_file(Path(__file__))
    semantic = sha256_bytes(
        canonical_json(
            {
                "catalog_sha256": sha256_file(catalog_path),
                "catalog_state_sha256": sha256_file(state_path),
                "worker_plan_sha256": plan["plan_sha256"],
                "source_revision": sources.revision,
                "sources": [
                    {
                        "batch_id": artifact.batch_id,
                        "result_sha256": artifact.result_sha256,
                        "report_sha256": artifact.report_sha256,
                    }
                    for artifact in sorted(sources.artifacts, key=lambda value: value.batch_id)
                ],
                "builder_sha256": builder_sha,
            }
        )
    )
    created = created_utc or datetime.now(UTC)
    if created.tzinfo is None:
        raise ValueError("created_utc must be timezone-aware")
    snapshot_id = (
        f"ocr-snapshot-{created.astimezone(UTC).strftime('%Y%m%dT%H%M%SZ')}-{semantic[:12]}"
    )
    output_root = Path(output_root)
    destination = output_root / snapshot_id
    if destination.exists():
        raise FileExistsError(f"immutable snapshot already exists: {destination}")
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{snapshot_id}.", dir=output_root))

    batches = {batch_id: _new_coverage(expected_batches[batch_id]) for batch_id in BATCH_IDS}
    videos = {video_id: _new_coverage(uids) for video_id, uids in catalog_by_video.items()}
    reports: dict[str, dict[str, Any]] = {}
    try:
        sqlite_path = temporary / "ocr.sqlite"
        connection = sqlite3.connect(sqlite_path)
        try:
            connection.execute("PRAGMA synchronous=FULL")
            _create_fts(connection)
            for batch_id in BATCH_IDS:
                if progress is not None:
                    progress(f"{batch_id} start")
                source = artifacts_by_batch[batch_id]
                result_path = _source_local_path(root, source.result_path)
                report_path = _source_local_path(root, source.report_path)
                if not result_path.is_file() or not report_path.is_file():
                    raise FileNotFoundError(f"missing local source files for {batch_id}")
                batch_videos = {
                    video_id: videos[video_id]
                    for video_id in plan["batches"][batch_id]["video_ids"]
                }
                report, _ = _validate_batch_archive(
                    archive_path=result_path,
                    external_report_path=report_path,
                    source=source,
                    plan=plan,
                    catalog=catalog,
                    coverage=batches[batch_id],
                    video_coverage=batch_videos,
                    connection=connection,
                    progress=progress,
                )
                reports[batch_id] = report
                if progress is not None:
                    progress(
                        f"{batch_id} validated frames={report['frames']} "
                        f"regions={report['regions']} residual={report['residual_regions']}"
                    )
            all_observed = set().union(*(coverage.observed for coverage in batches.values()))
            if all_observed != set(catalog) or sum(len(value.observed) for value in batches.values()) != len(catalog):
                raise ValueError("nine OCR v2 result shards are not UID-disjoint/exhaustive")
            connection.commit()
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            if integrity != ("ok",):
                raise RuntimeError(f"SQLite integrity_check failed: {integrity}")
            fts_rows = int(connection.execute("SELECT count(*) FROM ocr_fts").fetchone()[0])
            expected_fts = sum(values.statuses["success"] for values in batches.values())
            if fts_rows != expected_fts:
                raise RuntimeError("FTS5 row count differs from successful keyframes")
            probe = _fts_probe(connection)
            connection.execute("VACUUM")
        finally:
            connection.close()

        batch_models: dict[str, OcrV2BatchCoverage] = {}
        for batch_id in BATCH_IDS:
            source = artifacts_by_batch[batch_id]
            report = reports[batch_id]
            batch_models[batch_id] = OcrV2BatchCoverage(
                **_coverage_model(batches[batch_id]),
                batch_id=batch_id,
                worker=source.worker,
                video_ids=tuple(plan["batches"][batch_id]["video_ids"]),
                run_id=source.run_id,
                signature=source.signature,
                tasks_sha256=report["tasks_sha256"],
                source_result_sha256=source.result_sha256,
                source_report_sha256=source.report_sha256,
                regions=report["regions"],
                predictions=report["predictions"],
                source_recognition_complete=True,
                resume_with_new_work=bool(report.get("resume_with_new_work")),
            )
        video_to_batch = {
            video_id: batch_id
            for batch_id in BATCH_IDS
            for video_id in plan["batches"][batch_id]["video_ids"]
        }
        video_models = {
            video_id: OcrV2VideoCoverage(
                **_coverage_model(values), video_id=video_id, batch_id=video_to_batch[video_id]
            )
            for video_id, values in sorted(videos.items())
        }
        total_values = _new_coverage(set(catalog))
        for values in batches.values():
            total_values.observed.update(values.observed)
            total_values.statuses.update(values.statuses)
            total_values.engines.update(values.engines)
            total_values.residual_frames += values.residual_frames
            total_values.residual_regions += values.residual_regions
        totals = OcrV2CoverageUnit(**_coverage_model(total_values))
        manifest = OcrV2SnapshotManifest(
            snapshot_id=snapshot_id,
            created_utc=created.astimezone(UTC).isoformat(),
            materialized_text_policy=(
                "Only OCR v2 guard/selection-approved VietOCR or conditional Paddle text; "
                "unresolved residual text is excluded from FTS"
            ),
            builder_sha256=builder_sha,
            catalog_path=Path(catalog_path).name,
            catalog_sha256=sha256_file(catalog_path),
            catalog_state_path=state_path.name,
            catalog_state_sha256=sha256_file(state_path),
            catalog_records=len(catalog),
            catalog_videos=len(catalog_by_video),
            worker_plan_path=Path(worker_plan_path).name,
            worker_plan_sha256=sha256_file(worker_plan_path),
            worker_plan_semantic_sha256=plan["plan_sha256"],
            source_manifest_path=Path(source_manifest_path).name,
            source_manifest_sha256=sha256_file(source_manifest_path),
            source_repository=sources.repository,
            source_revision=sources.revision,
            source_artifacts=tuple(sorted(sources.artifacts, key=lambda value: value.batch_id)),
            totals=totals,
            batches=batch_models,
            videos=video_models,
            sqlite_sha256=sha256_file(sqlite_path),
            sqlite_bytes=sqlite_path.stat().st_size,
            fts_rows=fts_rows,
            fts_probe=probe,
        )
        coverage_path = temporary / "coverage.json"
        coverage_path.write_text(
            json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (temporary / "SHA256SUMS").write_text(
            f"{sha256_file(sqlite_path)}  ocr.sqlite\n"
            f"{sha256_file(coverage_path)}  coverage.json\n",
            encoding="ascii",
        )
        os.replace(temporary, destination)
        return destination, manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_ocr_v2_snapshot(
    *,
    snapshot_dir: Path,
    catalog_path: Path,
    catalog_state_path: Path | None = None,
) -> OcrV2SnapshotManifest:
    """Independently validate a built snapshot, its FTS rows and catalog joins."""

    snapshot_dir = Path(snapshot_dir)
    manifest_path = snapshot_dir / "coverage.json"
    sqlite_path = snapshot_dir / "ocr.sqlite"
    checksums_path = snapshot_dir / "SHA256SUMS"
    if not all(path.is_file() for path in (manifest_path, sqlite_path, checksums_path)):
        raise FileNotFoundError("snapshot needs coverage.json, ocr.sqlite and SHA256SUMS")
    manifest = OcrV2SnapshotManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if snapshot_dir.name != manifest.snapshot_id:
        raise ValueError("snapshot directory name differs from coverage snapshot_id")
    checks: dict[str, str] = {}
    for line in checksums_path.read_text(encoding="ascii").splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or _SHA256.fullmatch(fields[0]) is None or fields[1] in checks:
            raise ValueError("invalid snapshot SHA256SUMS")
        checks[fields[1]] = fields[0]
    if set(checks) != {"ocr.sqlite", "coverage.json"}:
        raise ValueError("snapshot SHA256SUMS has unexpected coverage")
    if checks["ocr.sqlite"] != sha256_file(sqlite_path) or checks["coverage.json"] != sha256_file(manifest_path):
        raise ValueError("snapshot checksum mismatch")
    if (
        manifest.sqlite_sha256 != checks["ocr.sqlite"]
        or manifest.sqlite_bytes != sqlite_path.stat().st_size
        or manifest.catalog_sha256 != sha256_file(catalog_path)
    ):
        raise ValueError("snapshot SQLite/catalog identity mismatch")
    catalog, _, state_path = _load_catalog(catalog_path, catalog_state_path)
    if manifest.catalog_state_sha256 != sha256_file(state_path):
        raise ValueError("snapshot catalog state identity mismatch")

    connection = sqlite3.connect(f"file:{sqlite_path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity != ("ok",):
            raise RuntimeError(f"SQLite integrity_check failed: {integrity}")
        columns = [row[1] for row in connection.execute("PRAGMA table_info(ocr_fts)")]
        if columns != ["video_id", "keyframe_uid", "detected_text", "language", "confidence"]:
            raise ValueError("ocr_fts does not have the canonical five-column schema")
        rows = connection.execute(
            "SELECT video_id,keyframe_uid,detected_text,language,confidence FROM ocr_fts"
        )
        seen: set[int] = set()
        per_video: Counter[str] = Counter()
        count = 0
        for video_id, uid_value, text, language, confidence in rows:
            try:
                uid = int(uid_value)
            except (TypeError, ValueError) as exc:
                raise ValueError("FTS row has a non-integer keyframe_uid") from exc
            expected = catalog.get(uid)
            if uid in seen or expected is None or video_id != expected.video_id:
                raise ValueError("FTS row has duplicate/foreign/mismatched keyframe_uid")
            if not isinstance(text, str) or not text.strip() or language not in {"vi", "en", "mixed"}:
                raise ValueError("FTS row has invalid searchable text/language")
            _finite_score(confidence, "FTS confidence")
            seen.add(uid)
            per_video[str(video_id)] += 1
            count += 1
        if count != manifest.fts_rows or count != manifest.totals.success_keyframes:
            raise ValueError("FTS row count differs from coverage")
        if any(
            per_video[video_id] != coverage.success_keyframes
            for video_id, coverage in manifest.videos.items()
        ):
            raise ValueError("per-video FTS rows differ from coverage")
        _fts_probe(connection)
    finally:
        connection.close()

    for field in (
        "expected_keyframes",
        "processed_keyframes",
        "success_keyframes",
        "no_text_keyframes",
        "error_keyframes",
        "missing_keyframes",
        "residual_frames",
        "residual_regions",
    ):
        if sum(getattr(value, field) for value in manifest.batches.values()) != getattr(
            manifest.totals, field
        ):
            raise ValueError(f"batch coverage does not sum to totals: {field}")
        if sum(getattr(value, field) for value in manifest.videos.values()) != getattr(
            manifest.totals, field
        ):
            raise ValueError(f"video coverage does not sum to totals: {field}")
    return manifest
