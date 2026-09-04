"""Build immutable, development-only ASR SQLite snapshots."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from offline.artifacts import sha256_file
from offline.asr_artifacts import AsrRecordEnvelope, AsrVideoStatus, video_set_sha256
from offline.catalog import FRAME_COLUMNS, validate_frames_catalog
from shared.schemas.frame import FrameRecord

_SNAPSHOT_ID = re.compile(r"^asr-snapshot-\d{8}T\d{6}Z-[0-9a-f]{12}$")


class AsrVideoCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    expected_keyframes: int = Field(ge=0)
    observed_keyframes: int = Field(ge=0)
    segment_count: int = Field(ge=0)
    status: AsrVideoStatus
    error: bool
    coverage_fraction: float = Field(ge=0, le=1)


class AsrSnapshotManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    artifact_kind: Literal["asr_sqlite_snapshot"] = "asr_sqlite_snapshot"
    snapshot_id: str = Field(pattern=_SNAPSHOT_ID.pattern)
    created_utc: str
    immutable: Literal[True] = True
    complete: Literal[False] = False
    production_ready: Literal[False] = False
    source_format: Literal["asr_envelope_v1"] = "asr_envelope_v1"
    catalog_path: str
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalog_records: int = Field(ge=1)
    catalog_videos: int = Field(ge=1)
    source_artifacts: list[dict[str, Any]]
    source_records: int = Field(ge=0)
    observed_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    success_videos: int = Field(ge=0)
    silent_videos: int = Field(ge=0)
    error_videos: int = Field(ge=0)
    missing_videos: int = Field(ge=0)
    covered_videos: int = Field(ge=0)
    coverage_fraction: float = Field(ge=0, le=1)
    sqlite_path: Literal["asr.sqlite"] = "asr.sqlite"
    sqlite_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sqlite_bytes: int = Field(gt=0)
    fts_rows: int = Field(ge=0)
    fts_probe: dict[str, Any]
    videos: dict[str, AsrVideoCoverage]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line, text in enumerate(stream, 1):
            if not text.strip():
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row at {path}:{line} must be an object")
            rows.append(value)
    return rows


def load_envelope_records(paths: Sequence[Path]) -> list[AsrRecordEnvelope]:
    records: list[AsrRecordEnvelope] = []
    for path in paths:
        records.extend(AsrRecordEnvelope.model_validate(row) for row in _read_jsonl(Path(path)))
    return records


# Naming parallel to the OCR implementation.
load_asr_envelope_records = load_envelope_records


def _load_catalog(path: Path, state_path: Path | None) -> tuple[dict[int, FrameRecord], Counter[str]]:
    catalog = Path(path)
    state = state_path or catalog.with_name(f"{catalog.name}.state.json")
    if not validate_frames_catalog(catalog, state):
        raise RuntimeError("frames.csv/state validation failed")
    records: dict[int, FrameRecord] = {}
    per_video: Counter[str] = Counter()
    with catalog.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FRAME_COLUMNS:
            raise RuntimeError("frames.csv columns do not match canonical schema")
        for raw in reader:
            frame = FrameRecord(**{**raw, "window_id": raw["window_id"] or None})
            if frame.keyframe_uid in records:
                raise RuntimeError("duplicate catalog keyframe_uid")
            records[frame.keyframe_uid] = frame
            per_video[frame.video_id] += 1
    return records, per_video


def _validate_envelopes(
    records: Sequence[AsrRecordEnvelope], catalog: dict[int, FrameRecord]
) -> None:
    catalog_videos = {frame.video_id for frame in catalog.values()}
    seen: set[str] = set()
    for envelope in records:
        if envelope.video_id not in catalog_videos:
            raise ValueError(f"foreign ASR video: {envelope.video_id}")
        if envelope.video_id in seen:
            raise ValueError(f"duplicate ASR video: {envelope.video_id}")
        seen.add(envelope.video_id)
        for segment in envelope.segments:
            frame = catalog.get(segment.keyframe_uid_nearest)
            if frame is None:
                raise ValueError(f"foreign ASR keyframe UID: {segment.keyframe_uid_nearest}")
            if frame.video_id != envelope.video_id:
                raise ValueError("ASR segment keyframe belongs to another video")


def _fts_probe(connection: sqlite3.Connection) -> dict[str, Any]:
    row = connection.execute(
        "SELECT video_id,segment_id,transcribed_text FROM asr_fts ORDER BY video_id,segment_id LIMIT 1"
    ).fetchone()
    if row is None:
        return {"executed": False, "reason": "no_segments"}
    token = next((part for part in re.findall(r"[^\W_]+", row[2], flags=re.UNICODE) if len(part) >= 2), None)
    if not token:
        return {"executed": False, "reason": "no_probe_token"}
    matched = connection.execute(
        "SELECT 1 FROM asr_fts WHERE asr_fts MATCH ? AND segment_id=? LIMIT 1",
        (f'"{token.replace(chr(34), chr(34) * 2)}"', row[1]),
    ).fetchone()
    if matched is None:
        raise RuntimeError("ASR FTS5 probe did not return its source segment")
    return {"executed": True, "token": token, "segment_id": row[1]}


def _write_sqlite(path: Path, envelopes: Sequence[AsrRecordEnvelope]) -> tuple[int, dict[str, Any]]:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """CREATE VIRTUAL TABLE asr_fts USING fts5(
                video_id UNINDEXED,
                segment_id UNINDEXED,
                transcribed_text,
                language UNINDEXED,
                keyframe_uid_nearest UNINDEXED,
                start_time UNINDEXED,
                end_time UNINDEXED
            )"""
        )
        segments = sorted(
            (segment for envelope in envelopes for segment in envelope.segments),
            key=lambda row: (row.video_id, row.segment_id),
        )
        connection.executemany(
            "INSERT INTO asr_fts(video_id,segment_id,transcribed_text,language,keyframe_uid_nearest,start_time,end_time) VALUES(?,?,?,?,?,?,?)",
            [
                (s.video_id, s.segment_id, s.transcribed_text, s.language,
                 s.keyframe_uid_nearest, s.start_time, s.end_time)
                for s in segments
            ],
        )
        connection.commit()
        if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise RuntimeError("SQLite integrity_check failed")
        count = int(connection.execute("SELECT count(*) FROM asr_fts").fetchone()[0])
        probe = _fts_probe(connection)
        connection.execute("VACUUM")
        return count, probe
    finally:
        connection.close()


def build_asr_snapshot(
    *,
    catalog_path: Path,
    records: Sequence[AsrRecordEnvelope],
    source_paths: Sequence[Path] = (),
    output_root: Path,
    catalog_state_path: Path | None = None,
    created_utc: datetime | None = None,
    source_format: str = "asr_envelope_v1",
) -> tuple[Path, AsrSnapshotManifest]:
    catalog, per_video = _load_catalog(Path(catalog_path), catalog_state_path)
    _validate_envelopes(records, catalog)
    source_artifacts = [
        {"path": Path(path).name, "sha256": sha256_file(Path(path)), "bytes": Path(path).stat().st_size}
        for path in source_paths
    ]
    created = created_utc or datetime.now(UTC)
    if created.tzinfo is None:
        raise ValueError("created_utc must be timezone-aware")
    semantic = hashlib.sha256(json.dumps({
        "catalog": sha256_file(Path(catalog_path)),
        "sources": source_artifacts,
        "source_format": source_format,
        "records": [row.model_dump(mode="json") for row in records],
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    snapshot_id = f"asr-snapshot-{created.astimezone(UTC).strftime('%Y%m%dT%H%M%SZ')}-{semantic[:12]}"
    destination = Path(output_root) / snapshot_id
    if destination.exists():
        raise FileExistsError(f"immutable snapshot already exists: {destination}")
    Path(output_root).mkdir(parents=True, exist_ok=True)
    temporary = Path(output_root) / f".{snapshot_id}.staging"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    published = False
    try:
        sqlite_path = temporary / "asr.sqlite"
        fts_rows, probe = _write_sqlite(sqlite_path, records)
        by_video = {row.video_id: row for row in records}
        videos: dict[str, AsrVideoCoverage] = {}
        for video_id, expected in sorted(per_video.items()):
            row = by_video.get(video_id)
            status = row.status if row else AsrVideoStatus.ERROR
            observed = len(row.segments) if row else 0
            videos[video_id] = AsrVideoCoverage(
                expected_keyframes=expected,
                observed_keyframes=len({s.keyframe_uid_nearest for s in row.segments}) if row else 0,
                segment_count=observed,
                status=status,
                error=status == AsrVideoStatus.ERROR,
                coverage_fraction=(1.0 if row and status != AsrVideoStatus.ERROR else 0.0),
            )
        counts = Counter(row.status for row in records)
        missing = len(set(per_video) - set(by_video))
        manifest = AsrSnapshotManifest(
            snapshot_id=snapshot_id,
            created_utc=created.astimezone(UTC).isoformat(),
            catalog_path=Path(catalog_path).name,
            source_format=source_format,
            catalog_sha256=sha256_file(Path(catalog_path)),
            catalog_records=len(catalog),
            catalog_videos=len(per_video),
            source_artifacts=source_artifacts,
            source_records=len(records),
            observed_video_sha256=video_set_sha256(by_video),
            success_videos=counts[AsrVideoStatus.SUCCESS],
            silent_videos=counts[AsrVideoStatus.SILENT],
            error_videos=counts[AsrVideoStatus.ERROR],
            missing_videos=missing,
            covered_videos=sum(1 for row in records if row.status != AsrVideoStatus.ERROR),
            coverage_fraction=(len(records) - counts[AsrVideoStatus.ERROR]) / len(per_video),
            sqlite_sha256=sha256_file(sqlite_path),
            sqlite_bytes=sqlite_path.stat().st_size,
            fts_rows=fts_rows,
            fts_probe=probe,
            videos=videos,
        )
        coverage = temporary / "coverage.json"
        coverage.write_text(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (temporary / "SHA256SUMS").write_text(
            f"{sha256_file(sqlite_path)}  asr.sqlite\n{sha256_file(coverage)}  coverage.json\n",
            encoding="ascii",
        )
        os.replace(temporary, destination)
        published = True
        return destination, manifest
    finally:
        if not published:
            shutil.rmtree(temporary, ignore_errors=True)
