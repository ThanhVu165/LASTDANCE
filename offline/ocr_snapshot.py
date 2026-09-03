"""Build immutable development-only OCR SQLite snapshots from validated JSONL."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from offline.artifacts import sha256_file
from offline.catalog import FRAME_COLUMNS, validate_frames_catalog
from offline.ocr_artifacts import (
    OcrEngine,
    OcrRecordEnvelope,
    OcrStatus,
    aggregate_easyocr_confidence,
    uid_set_sha256,
)
from offline.ocr_vintern_gate2 import (
    VinternGate2Policy,
    route_vintern_region,
    vintern_output_rejection_reasons,
)
from shared.schemas.frame import FrameRecord


_MAX_INT64 = (1 << 63) - 1
_SNAPSHOT_ID = re.compile(r"^ocr-snapshot-\d{8}T\d{6}Z-[0-9a-f]{12}$")


class SnapshotRecord(BaseModel):
    """FTS projection plus provenance; not a replacement for ``OcrResult``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    video_id: str = Field(min_length=1)
    keyframe_uid: int = Field(gt=0, le=_MAX_INT64)
    status: OcrStatus
    engine: OcrEngine
    detected_text: tuple[str, ...] = ()
    language: Literal["vi", "en", "mixed"] | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)


class VinternCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    required_regions: int = Field(ge=0)
    completed_regions: int = Field(ge=0)
    accepted_regions: int = Field(ge=0)
    residual_regions: int = Field(ge=0)
    pending_regions: int = Field(ge=0)
    state: Literal[
        "not_required",
        "not_run",
        "partial",
        "complete",
        "complete_with_residual",
    ]


class SnapshotVideoCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    expected_keyframes: int = Field(ge=0)
    observed_keyframes: int = Field(ge=0)
    success_keyframes: int = Field(ge=0)
    no_text_keyframes: int = Field(ge=0)
    error_keyframes: int = Field(ge=0)
    missing_keyframes: int = Field(ge=0)
    coverage_fraction: float = Field(ge=0, le=1)
    final_engine_counts: dict[str, int]
    materialized_text_tier: Literal[
        "unavailable",
        "easyocr_only",
        "easyocr_vintern_calibrated",
        "vintern_or_later",
    ]
    vintern: VinternCoverage
    snapshot_uid_coverage_full: bool


class SnapshotBatchCoverage(BaseModel):
    """Exact per-batch state exposed to Online consumers of a development snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    tier: Literal["craft_only", "easyocr", "vintern_calibrated", "gemini_final"]
    complete: bool
    searchable: bool
    video_ids: tuple[str, ...]
    expected_keyframes: int = Field(ge=0)
    frames: int = Field(ge=0)
    processed_keyframes: int = Field(ge=0)
    success_keyframes: int = Field(ge=0)
    no_text_keyframes: int = Field(ge=0)
    error_keyframes: int = Field(ge=0)
    pending_recognition_keyframes: int = Field(ge=0)
    missing_keyframes: int = Field(ge=0)
    coverage_fraction: float = Field(ge=0, le=1)
    updated_utc: str
    assigned_uid_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observed_uid_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_format: str | None = None
    source_artifact: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_counts_and_tier(self) -> "SnapshotBatchCoverage":
        if len(self.video_ids) != len(set(self.video_ids)) or not self.video_ids:
            raise ValueError("batch video_ids must be non-empty and unique")
        if self.processed_keyframes > self.expected_keyframes:
            raise ValueError("batch processed_keyframes exceeds expected_keyframes")
        if self.frames != self.processed_keyframes:
            raise ValueError("batch frames must equal processed_keyframes")
        if (
            self.success_keyframes
            + self.no_text_keyframes
            + self.error_keyframes
            + self.pending_recognition_keyframes
            != self.processed_keyframes
        ):
            raise ValueError("batch status counts do not sum to processed_keyframes")
        if self.missing_keyframes != self.expected_keyframes - self.processed_keyframes:
            raise ValueError("batch missing_keyframes does not match expected/processed")
        expected_fraction = (
            self.processed_keyframes / self.expected_keyframes
            if self.expected_keyframes
            else 0.0
        )
        if not math.isclose(self.coverage_fraction, expected_fraction, abs_tol=1e-12):
            raise ValueError("batch coverage_fraction does not match counts")
        if self.tier == "craft_only":
            if self.searchable:
                raise ValueError("craft_only batch cannot be searchable")
            if self.success_keyframes:
                raise ValueError("craft_only batch cannot claim OCR success")
        elif not self.searchable and self.processed_keyframes:
            raise ValueError("recognition tier with records must be searchable")
        elif self.pending_recognition_keyframes:
            raise ValueError("recognition tier cannot contain pending recognition records")
        if datetime.fromisoformat(self.updated_utc.replace("Z", "+00:00")).tzinfo is None:
            raise ValueError("batch updated_utc must include timezone")
        gate = (
            self.processed_keyframes == self.expected_keyframes
            and self.error_keyframes == 0
        )
        if self.complete != gate:
            raise ValueError("batch complete does not match coverage/error counts")
        return self


class OcrSnapshotManifest(BaseModel):
    """Sidecar that prevents a partial snapshot from masquerading as production."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1, 2] = 2
    artifact_kind: Literal["ocr_sqlite_snapshot"] = "ocr_sqlite_snapshot"
    snapshot_id: str = Field(pattern=_SNAPSHOT_ID.pattern)
    created_utc: str
    immutable: Literal[True] = True
    complete: Literal[False] = False
    production_ready: Literal[False] = False
    intended_use: Literal["online_development_only"] = "online_development_only"
    source_format: Literal[
        "ocr_envelope_v1",
        "gate2_easyocr_dev_v1",
        "gate2_calibrated_dev_v1",
        "incremental_batch_union_v1",
    ]
    materialized_text_policy: str
    builder_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_snapshot_id: str | None = None
    catalog_path: str
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalog_records: int = Field(ge=1)
    catalog_videos: int = Field(ge=1)
    source_artifacts: list[dict[str, Any]]
    source_records: int = Field(ge=0)
    observed_uid_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    success_keyframes: int = Field(ge=0)
    no_text_keyframes: int = Field(ge=0)
    error_keyframes: int = Field(ge=0)
    missing_keyframes: int = Field(ge=0)
    covered_videos: int = Field(ge=0)
    coverage_fraction: float = Field(ge=0, le=1)
    sqlite_path: Literal["ocr.sqlite"] = "ocr.sqlite"
    sqlite_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sqlite_bytes: int = Field(gt=0)
    fts_rows: int = Field(ge=0)
    fts_probe: dict[str, Any]
    videos: dict[str, SnapshotVideoCoverage]
    batches: dict[str, SnapshotBatchCoverage] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_batch_keys(self) -> "OcrSnapshotManifest":
        if any(key != value.batch_id for key, value in self.batches.items()):
            raise ValueError("coverage batch map key must equal batch_id")
        return self


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _load_catalog(
    catalog_path: Path,
    catalog_state_path: Path | None,
) -> tuple[dict[int, FrameRecord], dict[str, int]]:
    catalog = Path(catalog_path)
    state = catalog_state_path or catalog.with_name(f"{catalog.name}.state.json")
    if not validate_frames_catalog(catalog, state):
        raise RuntimeError("frames.csv/state validation failed")
    by_uid: dict[int, FrameRecord] = {}
    per_video: Counter[str] = Counter()
    with catalog.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FRAME_COLUMNS:
            raise RuntimeError("frames.csv columns do not match canonical schema")
        for raw in reader:
            record = FrameRecord(**{**raw, "window_id": raw["window_id"] or None})
            if record.keyframe_uid in by_uid:
                raise RuntimeError(f"duplicate catalog UID: {record.keyframe_uid}")
            by_uid[record.keyframe_uid] = record
            per_video[record.video_id] += 1
    return by_uid, dict(per_video)


def _language_for_gate2(row: dict[str, Any], texts: Sequence[str]) -> str:
    has_vi = any(bool(region.get("has_vi_marks")) for region in row.get("regions", []))
    has_ascii = any(bool(region.get("has_ascii_word")) for region in row.get("regions", []))
    if bool(row.get("frame_mixed_candidate")) or (has_vi and has_ascii):
        return "mixed"
    if has_vi:
        return "vi"
    return "en"


def load_envelope_snapshot_records(paths: Sequence[Path]) -> list[SnapshotRecord]:
    records: list[SnapshotRecord] = []
    for path in paths:
        for raw in _read_jsonl(path):
            envelope = OcrRecordEnvelope.model_validate(raw)
            if envelope.status == OcrStatus.SUCCESS:
                assert envelope.result is not None
                records.append(
                    SnapshotRecord(
                        video_id=envelope.video_id,
                        keyframe_uid=envelope.keyframe_uid,
                        status=envelope.status,
                        engine=envelope.engine,
                        detected_text=tuple(envelope.result.detected_text),
                        language=envelope.result.language,
                        confidence=envelope.result.confidence,
                    )
                )
            else:
                records.append(
                    SnapshotRecord(
                        video_id=envelope.video_id,
                        keyframe_uid=envelope.keyframe_uid,
                        status=envelope.status,
                        engine=envelope.engine,
                    )
                )
    return records


def load_gate2_easyocr_snapshot_records(path: Path) -> list[SnapshotRecord]:
    """Project recovered dev evidence without claiming Vintern text was materialized."""

    records: list[SnapshotRecord] = []
    for raw in _read_jsonl(path):
        status = str(raw.get("status"))
        if status == "error":
            records.append(
                SnapshotRecord(
                    video_id=str(raw["video_id"]),
                    keyframe_uid=int(raw["keyframe_uid"]),
                    status=OcrStatus.ERROR,
                    engine=OcrEngine.EASYOCR,
                )
            )
            continue
        if status == "no_text":
            records.append(
                SnapshotRecord(
                    video_id=str(raw["video_id"]),
                    keyframe_uid=int(raw["keyframe_uid"]),
                    status=OcrStatus.NO_TEXT,
                    engine=OcrEngine.CRAFT,
                )
            )
            continue
        if status != "text_detected":
            raise ValueError(f"unsupported Gate 2 frame status: {status}")
        detections = [
            (str(region.get("easyocr_text") or "").strip(), float(region["easyocr_confidence"]))
            for region in raw.get("regions", [])
            if str(region.get("easyocr_text") or "").strip()
        ]
        if not detections:
            records.append(
                SnapshotRecord(
                    video_id=str(raw["video_id"]),
                    keyframe_uid=int(raw["keyframe_uid"]),
                    status=OcrStatus.ERROR,
                    engine=OcrEngine.EASYOCR,
                )
            )
            continue
        texts = [text for text, _ in detections]
        records.append(
            SnapshotRecord(
                video_id=str(raw["video_id"]),
                keyframe_uid=int(raw["keyframe_uid"]),
                status=OcrStatus.SUCCESS,
                engine=OcrEngine.EASYOCR,
                detected_text=tuple(texts),
                language=_language_for_gate2(raw, texts),
                confidence=aggregate_easyocr_confidence(detections),
            )
        )
    return records


def load_gate2_calibrated_snapshot_records(path: Path) -> list[SnapshotRecord]:
    """Project Gate B final region decisions after empirical Vintern calibration."""

    records: list[SnapshotRecord] = []
    for raw in _read_jsonl(path):
        status = str(raw.get("status"))
        if status == "error":
            records.append(
                SnapshotRecord(
                    video_id=str(raw["video_id"]),
                    keyframe_uid=int(raw["keyframe_uid"]),
                    status=OcrStatus.ERROR,
                    engine=OcrEngine.EASYOCR,
                )
            )
            continue
        if status == "no_text":
            records.append(
                SnapshotRecord(
                    video_id=str(raw["video_id"]),
                    keyframe_uid=int(raw["keyframe_uid"]),
                    status=OcrStatus.NO_TEXT,
                    engine=OcrEngine.CRAFT,
                )
            )
            continue
        if status != "text_detected":
            raise ValueError(f"unsupported calibrated Gate 2 frame status: {status}")

        detections: list[tuple[str, float]] = []
        has_vintern_override = False
        for region in raw.get("regions", []):
            final_text = str(region.get("final_text") or "").strip()
            final_confidence = region.get("final_confidence")
            final_engine = str(region.get("final_engine") or "")
            override = bool(region.get("vintern_override"))
            if final_engine not in {"easyocr", "vintern"}:
                raise ValueError("calibrated region has invalid final_engine")
            if override != (final_engine == "vintern"):
                raise ValueError("Vintern override/final_engine mismatch")
            if override and not region.get("calibration_bucket_id"):
                raise ValueError("Vintern override lacks calibration bucket evidence")
            has_vintern_override = has_vintern_override or override
            if not final_text:
                continue
            if isinstance(final_confidence, bool) or not isinstance(
                final_confidence, (int, float)
            ):
                raise ValueError("final_confidence must be numeric")
            confidence = float(final_confidence)
            if not 0 <= confidence <= 1:
                raise ValueError("final_confidence must be in [0, 1]")
            detections.append((final_text, confidence))
        if not detections:
            records.append(
                SnapshotRecord(
                    video_id=str(raw["video_id"]),
                    keyframe_uid=int(raw["keyframe_uid"]),
                    status=OcrStatus.ERROR,
                    engine=OcrEngine.VINTERN if has_vintern_override else OcrEngine.EASYOCR,
                )
            )
            continue
        texts = [text for text, _ in detections]
        records.append(
            SnapshotRecord(
                video_id=str(raw["video_id"]),
                keyframe_uid=int(raw["keyframe_uid"]),
                status=OcrStatus.SUCCESS,
                engine=OcrEngine.VINTERN if has_vintern_override else OcrEngine.EASYOCR,
                detected_text=tuple(texts),
                language=_language_for_gate2(raw, texts),
                confidence=aggregate_easyocr_confidence(detections),
            )
        )
    return records


def gate2_calibrated_vintern_coverage(path: Path) -> dict[str, VinternCoverage]:
    """Derive per-video Vintern coverage from a calibrated Gate B frame artifact."""

    counters: dict[str, Counter[str]] = defaultdict(Counter)
    videos: set[str] = set()
    for frame in _read_jsonl(path):
        video_id = str(frame["video_id"])
        videos.add(video_id)
        for region in frame.get("regions", []):
            if not bool(region.get("vintern_candidate")):
                continue
            counters[video_id]["required"] += 1
            status = region.get("vintern_result_status")
            if status is None:
                continue
            counters[video_id]["completed"] += 1
            rejected = list(region.get("vintern_guard_rejection_reasons") or [])
            if status != "success" or rejected:
                counters[video_id]["residual"] += 1
            else:
                counters[video_id]["accepted"] += 1

    coverage: dict[str, VinternCoverage] = {}
    for video_id in videos:
        values = counters[video_id]
        required = values["required"]
        completed = values["completed"]
        pending = required - completed
        residual = values["residual"]
        if required == 0:
            state = "not_required"
        elif completed == 0:
            state = "not_run"
        elif pending:
            state = "partial"
        elif residual:
            state = "complete_with_residual"
        else:
            state = "complete"
        coverage[video_id] = VinternCoverage(
            required_regions=required,
            completed_regions=completed,
            accepted_regions=values["accepted"],
            residual_regions=residual,
            pending_regions=pending,
            state=state,  # type: ignore[arg-type]
        )
    return coverage


def gate2_vintern_coverage(
    easyocr_path: Path,
    vintern_results_path: Path | None,
    *,
    policy: VinternGate2Policy,
) -> dict[str, VinternCoverage]:
    candidates: dict[str, tuple[str, str]] = {}
    videos: set[str] = set()
    for frame in _read_jsonl(easyocr_path):
        video_id = str(frame["video_id"])
        videos.add(video_id)
        for region in frame.get("regions", []):
            decision = route_vintern_region(region, policy=policy)
            if decision.candidate:
                candidate_id = str(region["region_id"])
                if candidate_id in candidates:
                    raise ValueError(f"duplicate Vintern candidate ID: {candidate_id}")
                candidates[candidate_id] = (video_id, str(region.get("easyocr_text") or ""))

    results: dict[str, dict[str, Any]] = {}
    if vintern_results_path is not None:
        for raw in _read_jsonl(vintern_results_path):
            candidate_id = str(raw.get("candidate_id") or "")
            if candidate_id in results:
                raise ValueError(f"duplicate Vintern result ID: {candidate_id}")
            if candidate_id in candidates:
                results[candidate_id] = raw

    counters: dict[str, Counter[str]] = defaultdict(Counter)
    for candidate_id, (video_id, easyocr_text) in candidates.items():
        counters[video_id]["required"] += 1
        result = results.get(candidate_id)
        if result is None:
            continue
        counters[video_id]["completed"] += 1
        rejection = ("runtime_error",)
        if result.get("status") == "success":
            rejection = vintern_output_rejection_reasons(
                easyocr_text=easyocr_text,
                vintern_text=str(result.get("vintern_text") or ""),
            )
        if rejection:
            counters[video_id]["residual"] += 1
        else:
            counters[video_id]["accepted"] += 1

    coverage: dict[str, VinternCoverage] = {}
    for video_id in videos:
        values = counters[video_id]
        required = values["required"]
        completed = values["completed"]
        residual = values["residual"]
        pending = required - completed
        if required == 0:
            state = "not_required"
        elif completed == 0:
            state = "not_run"
        elif pending:
            state = "partial"
        elif residual:
            state = "complete_with_residual"
        else:
            state = "complete"
        coverage[video_id] = VinternCoverage(
            required_regions=required,
            completed_regions=completed,
            accepted_regions=values["accepted"],
            residual_regions=residual,
            pending_regions=pending,
            state=state,  # type: ignore[arg-type]
        )
    return coverage


def _default_vintern_coverage() -> VinternCoverage:
    return VinternCoverage(
        required_regions=0,
        completed_regions=0,
        accepted_regions=0,
        residual_regions=0,
        pending_regions=0,
        state="not_run",
    )


def _validate_records(
    records: Sequence[SnapshotRecord],
    catalog: dict[int, FrameRecord],
) -> None:
    seen: set[int] = set()
    for record in records:
        if record.keyframe_uid in seen:
            raise ValueError(f"duplicate snapshot UID: {record.keyframe_uid}")
        seen.add(record.keyframe_uid)
        expected = catalog.get(record.keyframe_uid)
        if expected is None:
            raise ValueError(f"foreign snapshot UID: {record.keyframe_uid}")
        if expected.video_id != record.video_id:
            raise ValueError(f"video_id mismatch for UID: {record.keyframe_uid}")
        if record.status == OcrStatus.SUCCESS:
            if not record.detected_text or record.language is None or record.confidence is None:
                raise ValueError("success snapshot record lacks searchable OCR fields")
        elif record.detected_text or record.language is not None or record.confidence is not None:
            raise ValueError("no_text/error snapshot record cannot carry searchable OCR fields")


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
    matched = connection.execute(
        "SELECT 1 FROM ocr_fts WHERE ocr_fts MATCH ? AND keyframe_uid = ? LIMIT 1",
        (f'"{token.replace(chr(34), chr(34) * 2)}"', uid),
    ).fetchone()
    if matched is None:
        raise RuntimeError("FTS5 probe did not return its source keyframe")
    return {"executed": True, "token": token, "keyframe_uid": int(uid)}


def _write_sqlite(path: Path, records: Sequence[SnapshotRecord]) -> tuple[int, dict[str, Any]]:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """CREATE VIRTUAL TABLE ocr_fts USING fts5(
                video_id UNINDEXED,
                keyframe_uid UNINDEXED,
                detected_text,
                language UNINDEXED,
                confidence UNINDEXED
            )"""
        )
        searchable = sorted(
            (record for record in records if record.status == OcrStatus.SUCCESS),
            key=lambda record: (record.video_id, record.keyframe_uid),
        )
        connection.executemany(
            "INSERT INTO ocr_fts(video_id,keyframe_uid,detected_text,language,confidence) VALUES(?,?,?,?,?)",
            [
                (
                    record.video_id,
                    record.keyframe_uid,
                    "\n".join(record.detected_text),
                    record.language,
                    record.confidence,
                )
                for record in searchable
            ],
        )
        connection.commit()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity != ("ok",):
            raise RuntimeError(f"SQLite integrity_check failed: {integrity}")
        count = int(connection.execute("SELECT count(*) FROM ocr_fts").fetchone()[0])
        if count != len(searchable):
            raise RuntimeError("FTS5 row count changed during build")
        probe = _fts_probe(connection)
        connection.execute("VACUUM")
        return count, probe
    finally:
        connection.close()


def _snapshot_id(created: datetime, semantic_digest: str) -> str:
    return f"ocr-snapshot-{created.astimezone(UTC).strftime('%Y%m%dT%H%M%SZ')}-{semantic_digest[:12]}"


def build_ocr_snapshot(
    *,
    catalog_path: Path,
    catalog_state_path: Path | None,
    records: Sequence[SnapshotRecord],
    source_paths: Sequence[Path],
    source_format: Literal[
        "ocr_envelope_v1",
        "gate2_easyocr_dev_v1",
        "gate2_calibrated_dev_v1",
        "incremental_batch_union_v1",
    ],
    materialized_text_policy: str,
    output_root: Path,
    vintern_by_video: dict[str, VinternCoverage] | None = None,
    parent_snapshot_id: str | None = None,
    created_utc: datetime | None = None,
    batch_coverage: dict[str, SnapshotBatchCoverage] | None = None,
) -> tuple[Path, OcrSnapshotManifest]:
    catalog, catalog_per_video = _load_catalog(catalog_path, catalog_state_path)
    _validate_records(records, catalog)
    if parent_snapshot_id is not None and _SNAPSHOT_ID.fullmatch(parent_snapshot_id) is None:
        raise ValueError("parent_snapshot_id has invalid format")

    source_artifacts = [
        {
            "path": Path(path).name,
            "sha256": sha256_file(path),
            "bytes": Path(path).stat().st_size,
        }
        for path in source_paths
    ]
    builder_sha256 = sha256_file(Path(__file__))
    semantic = hashlib.sha256(
        json.dumps(
            {
                "catalog_sha256": sha256_file(catalog_path),
                "sources": source_artifacts,
                "source_format": source_format,
                "materialized_text_policy": materialized_text_policy,
                "builder_sha256": builder_sha256,
                "batches": {
                    key: value.model_dump(mode="json")
                    for key, value in sorted((batch_coverage or {}).items())
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    created = created_utc or datetime.now(UTC)
    if created.tzinfo is None:
        raise ValueError("created_utc must be timezone-aware")
    snapshot_id = _snapshot_id(created, semantic)
    destination = Path(output_root) / snapshot_id
    if destination.exists():
        raise FileExistsError(f"immutable snapshot already exists: {destination}")
    Path(output_root).mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{snapshot_id}.", dir=output_root))
    try:
        sqlite_path = temporary / "ocr.sqlite"
        fts_rows, probe = _write_sqlite(sqlite_path, records)
        by_video: dict[str, list[SnapshotRecord]] = defaultdict(list)
        for record in records:
            by_video[record.video_id].append(record)
        vintern_evidence = vintern_by_video or {}
        videos: dict[str, SnapshotVideoCoverage] = {}
        for video_id in sorted(catalog_per_video):
            expected = catalog_per_video[video_id]
            observed = by_video.get(video_id, [])
            status_counts = Counter(record.status for record in observed)
            engine_counts = Counter(record.engine for record in observed)
            materialized_tier = "unavailable"
            if observed:
                if source_format == "gate2_calibrated_dev_v1":
                    materialized_tier = "easyocr_vintern_calibrated"
                elif engine_counts[OcrEngine.VINTERN] or engine_counts[OcrEngine.GEMINI]:
                    materialized_tier = "vintern_or_later"
                else:
                    materialized_tier = "easyocr_only"
            missing = expected - len(observed)
            error_count = status_counts[OcrStatus.ERROR]
            videos[video_id] = SnapshotVideoCoverage(
                expected_keyframes=expected,
                observed_keyframes=len(observed),
                success_keyframes=status_counts[OcrStatus.SUCCESS],
                no_text_keyframes=status_counts[OcrStatus.NO_TEXT],
                error_keyframes=error_count,
                missing_keyframes=missing,
                coverage_fraction=len(observed) / expected,
                final_engine_counts={engine.value: count for engine, count in sorted(engine_counts.items(), key=lambda item: item[0].value)},
                materialized_text_tier=materialized_tier,  # type: ignore[arg-type]
                vintern=vintern_evidence.get(video_id, _default_vintern_coverage()),
                snapshot_uid_coverage_full=missing == 0 and error_count == 0,
            )

        status_counts = Counter(record.status for record in records)
        observed_uids = [record.keyframe_uid for record in records]
        manifest = OcrSnapshotManifest(
            snapshot_id=snapshot_id,
            created_utc=created.astimezone(UTC).isoformat(),
            parent_snapshot_id=parent_snapshot_id,
            source_format=source_format,
            materialized_text_policy=materialized_text_policy,
            builder_sha256=builder_sha256,
            catalog_path=Path(catalog_path).name,
            catalog_sha256=sha256_file(catalog_path),
            catalog_records=len(catalog),
            catalog_videos=len(catalog_per_video),
            source_artifacts=source_artifacts,
            source_records=len(records),
            observed_uid_sha256=uid_set_sha256(observed_uids),
            success_keyframes=status_counts[OcrStatus.SUCCESS],
            no_text_keyframes=status_counts[OcrStatus.NO_TEXT],
            error_keyframes=status_counts[OcrStatus.ERROR],
            missing_keyframes=len(catalog) - len(records),
            covered_videos=sum(bool(rows) for rows in by_video.values()),
            coverage_fraction=len(records) / len(catalog),
            sqlite_sha256=sha256_file(sqlite_path),
            sqlite_bytes=sqlite_path.stat().st_size,
            fts_rows=fts_rows,
            fts_probe=probe,
            videos=videos,
            batches=batch_coverage or {},
        )
        coverage_path = temporary / "coverage.json"
        coverage_path.write_text(
            json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        checksums = temporary / "SHA256SUMS"
        checksums.write_text(
            f"{sha256_file(sqlite_path)}  ocr.sqlite\n{sha256_file(coverage_path)}  coverage.json\n",
            encoding="ascii",
        )
        os.replace(temporary, destination)
        return destination, manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
