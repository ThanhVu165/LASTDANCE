"""Build and validate the canonical SQLite FTS5 ASR artifact and coverage report."""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from collections import Counter
from collections.abc import Iterable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from offline.artifacts import sha256_file
from offline.asr_artifacts import AsrTranscriptRecord, TranscriptStatus
from offline.asr_audio import AudioArtifact, AudioStatus
from shared.schemas.asr import AsrSegment


ASR_COLUMNS = [
    "video_id",
    "segment_id",
    "transcribed_text",
    "language",
    "keyframe_uid_nearest",
    "start_time",
    "end_time",
]
_CREATE_ASR_FTS = """CREATE VIRTUAL TABLE asr_fts USING fts5(
    video_id UNINDEXED,
    segment_id UNINDEXED,
    transcribed_text,
    language UNINDEXED,
    keyframe_uid_nearest UNINDEXED,
    start_time UNINDEXED,
    end_time UNINDEXED
)"""


class AsrCoverageStatus(StrEnum):
    COMPLETE = "complete"
    NO_AUDIO = "no_audio"
    NO_SPEECH_VERIFIED = "no_speech_verified"
    NO_SPEECH_UNVERIFIED = "no_speech_unverified"
    PENDING = "pending"
    FAILED_ALIGNMENT = "failed_alignment"


class AsrCoverageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    video_id: str
    status: AsrCoverageStatus
    has_audio: bool
    transcript_segments: int = Field(ge=0)
    aligned_segments: int = Field(ge=0)
    indexed_segments: int = Field(ge=0)
    complete: bool
    note: str

    @field_validator("video_id", "note")
    @classmethod
    def _require_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("coverage identifiers/notes must not be empty")
        return normalized

    @model_validator(mode="after")
    def _validate_completion(self) -> "AsrCoverageRecord":
        terminal_complete = {
            AsrCoverageStatus.COMPLETE,
            AsrCoverageStatus.NO_AUDIO,
            AsrCoverageStatus.NO_SPEECH_VERIFIED,
        }
        if self.complete != (self.status in terminal_complete):
            raise ValueError("coverage complete flag does not match status")
        if self.status == AsrCoverageStatus.COMPLETE:
            if not (
                self.transcript_segments
                == self.aligned_segments
                == self.indexed_segments
                and self.transcript_segments > 0
            ):
                raise ValueError("complete coverage requires equal positive segment counts")
        return self


def derive_asr_coverage(
    *,
    expected_video_ids: Iterable[str],
    inventory_has_audio: Mapping[str, bool],
    audio_artifacts: Mapping[str, AudioArtifact],
    transcripts: Mapping[str, AsrTranscriptRecord],
    aligned_segments: Iterable[AsrSegment],
    verified_no_speech: set[str] | None = None,
) -> list[AsrCoverageRecord]:
    expected = sorted(expected_video_ids)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("expected_video_ids must be non-empty and unique")
    if set(expected) - set(inventory_has_audio):
        raise RuntimeError("inventory_has_audio does not cover every expected video")
    verified = verified_no_speech or set()
    foreign_verified = verified - set(expected)
    if foreign_verified:
        raise RuntimeError(f"verified-no-speech contains foreign videos: {sorted(foreign_verified)[:5]}")
    aligned_counts = Counter(segment.video_id for segment in aligned_segments)
    rows: list[AsrCoverageRecord] = []
    for video_id in expected:
        has_audio = bool(inventory_has_audio[video_id])
        audio = audio_artifacts.get(video_id)
        transcript = transcripts.get(video_id)
        aligned = aligned_counts[video_id]
        if not has_audio:
            status = AsrCoverageStatus.NO_AUDIO
            note = "inventory ffprobe reports no audio stream"
            transcript_count = 0
        elif audio is None or audio.status != AudioStatus.READY or transcript is None:
            status = AsrCoverageStatus.PENDING
            note = "audio extraction or transcription is incomplete"
            transcript_count = 0 if transcript is None else len(transcript.segments)
        elif transcript.status == TranscriptStatus.NO_SPEECH:
            transcript_count = 0
            if video_id in verified:
                status = AsrCoverageStatus.NO_SPEECH_VERIFIED
                note = "no-speech result verified by human review"
            else:
                status = AsrCoverageStatus.NO_SPEECH_UNVERIFIED
                note = "no-speech result still requires human review"
        else:
            transcript_count = len(transcript.segments)
            if aligned == transcript_count and transcript_count > 0:
                status = AsrCoverageStatus.COMPLETE
                note = "all transcript segments aligned and indexed"
            else:
                status = AsrCoverageStatus.FAILED_ALIGNMENT
                note = "transcript/alignment segment counts differ"
        complete = status in {
            AsrCoverageStatus.COMPLETE,
            AsrCoverageStatus.NO_AUDIO,
            AsrCoverageStatus.NO_SPEECH_VERIFIED,
        }
        rows.append(
            AsrCoverageRecord(
                video_id=video_id,
                status=status,
                has_audio=has_audio,
                transcript_segments=transcript_count,
                aligned_segments=aligned,
                indexed_segments=aligned,
                complete=complete,
                note=note,
            )
        )
    foreign_aligned = set(aligned_counts) - set(expected)
    if foreign_aligned:
        raise RuntimeError(f"aligned ASR contains foreign videos: {sorted(foreign_aligned)[:5]}")
    return rows


def _query_smoke(connection: sqlite3.Connection, segments: list[AsrSegment]) -> None:
    count = connection.execute("SELECT count(*) FROM asr_fts").fetchone()[0]
    if count != len(segments):
        raise RuntimeError("asr_fts row count does not match aligned segments")
    if not segments:
        return
    tokens = re.findall(r"\w+", segments[0].transcribed_text, flags=re.UNICODE)
    if not tokens:
        raise RuntimeError("cannot construct an ASR FTS5 smoke query")
    query = '"' + tokens[0].replace('"', '""') + '"'
    result = connection.execute(
        "SELECT video_id, segment_id FROM asr_fts WHERE asr_fts MATCH ? LIMIT 1",
        (query,),
    ).fetchone()
    if result is None:
        raise RuntimeError("ASR FTS5 smoke query returned no result")


def build_asr_sqlite_atomic(path: Path, segments: Iterable[AsrSegment]) -> Path:
    rows = sorted(segments, key=lambda row: (row.video_id, row.start_time, row.segment_id))
    keys = [(row.video_id, row.segment_id) for row in rows]
    if len(set(keys)) != len(keys):
        raise RuntimeError("cannot index duplicate ASR segments")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    try:
        connection.execute(_CREATE_ASR_FTS)
        connection.executemany(
            "INSERT INTO asr_fts "
            "(video_id, segment_id, transcribed_text, language, "
            "keyframe_uid_nearest, start_time, end_time) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    row.video_id,
                    row.segment_id,
                    row.transcribed_text,
                    row.language,
                    row.keyframe_uid_nearest,
                    row.start_time,
                    row.end_time,
                )
                for row in rows
            ],
        )
        connection.commit()
        _query_smoke(connection, rows)
    finally:
        connection.close()
    if destination.exists():
        temporary.unlink()
        raise RuntimeError(f"refusing to overwrite ASR SQLite artifact: {destination}")
    temporary.replace(destination)
    return destination


def write_coverage_csv_atomic(path: Path, rows: Iterable[AsrCoverageRecord]) -> Path:
    records = sorted(rows, key=lambda row: row.video_id)
    if not records or len({row.video_id for row in records}) != len(records):
        raise RuntimeError("ASR coverage must contain unique video rows")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    fieldnames = [
        "video_id",
        "status",
        "has_audio",
        "transcript_segments",
        "aligned_segments",
        "indexed_segments",
        "complete",
        "note",
    ]
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            payload = record.model_dump(mode="json")
            payload.pop("schema_version")
            writer.writerow(payload)
    temporary.replace(destination)
    return destination


def write_asr_state_atomic(
    path: Path,
    *,
    sqlite_path: Path,
    coverage_path: Path,
    segments: Iterable[AsrSegment],
    coverage: Iterable[AsrCoverageRecord],
) -> Path:
    segment_rows = list(segments)
    coverage_rows = list(coverage)
    payload = {
        "schema_version": 1,
        "complete": all(row.complete for row in coverage_rows),
        "segment_count": len(segment_rows),
        "video_count": len(coverage_rows),
        "complete_video_count": sum(row.complete for row in coverage_rows),
        "asr_sqlite_sha256": sha256_file(sqlite_path),
        "coverage_csv_sha256": sha256_file(coverage_path),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    return destination


def validate_asr_sqlite(path: Path, *, expected_segments: Iterable[AsrSegment]) -> bool:
    expected = list(expected_segments)
    source = Path(path)
    if not source.is_file():
        return False
    try:
        connection = sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True)
        try:
            columns = [row[1] for row in connection.execute("PRAGMA table_info(asr_fts)")]
            if columns != ASR_COLUMNS:
                return False
            _query_smoke(connection, expected)
        finally:
            connection.close()
        return True
    except (OSError, RuntimeError, sqlite3.Error):
        return False
