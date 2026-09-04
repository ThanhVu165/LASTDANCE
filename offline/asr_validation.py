"""Validate ASR bytes and per-video coverage against the consuming catalog."""
from __future__ import annotations
import math
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from offline.asr_alignment import nearest_keyframe_uid
from offline.artifacts import sha256_file
from offline.asr_snapshot import AsrSnapshotManifest

COLUMNS = ["video_id", "segment_id", "transcribed_text", "language", "keyframe_uid_nearest", "start_time", "end_time"]


def validate_asr_bundle(sqlite_path: Path, coverage_path: Path, *, catalog_sha256: str,
                        frames: dict[int, object]) -> AsrSnapshotManifest:
    manifest = AsrSnapshotManifest.model_validate_json(coverage_path.read_text(encoding="utf-8"))
    expected = Counter(frame.video_id for frame in frames.values())
    if (manifest.catalog_sha256 != catalog_sha256 or manifest.catalog_records != len(frames)
            or manifest.catalog_videos != len(expected) or set(manifest.videos) != set(expected)):
        raise ValueError("ASR coverage catalog identity or video set mismatch")
    if sha256_file(sqlite_path) != manifest.sqlite_sha256 or sqlite_path.stat().st_size != manifest.sqlite_bytes:
        raise ValueError("ASR SQLite checksum/size does not match coverage; publication may be interrupted")
    counts, observed, seen = Counter(), defaultdict(set), set()
    by_video = defaultdict(list)
    for frame in frames.values():
        by_video[frame.video_id].append(frame)
    connection = sqlite3.connect(f"{sqlite_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        table = connection.execute("SELECT sql FROM sqlite_master WHERE name='asr_fts'").fetchone()
        if table is None or not re.search(r"CREATE\s+VIRTUAL\s+TABLE.*USING\s+fts5", table[0], re.I | re.S):
            raise ValueError("ASR requires an FTS5 virtual table")
        if [row[1] for row in connection.execute("PRAGMA table_info(asr_fts)")] != COLUMNS:
            raise ValueError("ASR FTS schema mismatch")
        if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise ValueError("ASR SQLite integrity failed")
        for video, segment, text, language, uid, start, end in connection.execute("SELECT " + ",".join(COLUMNS) + " FROM asr_fts"):
            frame = frames.get(uid)
            if frame is None or frame.video_id != video:
                raise ValueError("ASR foreign UID or cross-video join")
            if (video, segment) in seen or not str(text).strip() or language not in {"vi", "en"}:
                raise ValueError("ASR duplicate/invalid segment")
            if not all(isinstance(t, (float, int)) and math.isfinite(t) for t in (start, end)) or not 0 <= start <= end:
                raise ValueError("ASR invalid segment timestamps")
            if nearest_keyframe_uid(video, start, end, by_video[video]) != uid:
                raise ValueError("ASR segment nearest UID does not match actual PTS alignment")
            seen.add((video, segment)); counts[video] += 1; observed[video].add(uid)
    finally:
        connection.close()
    if sum(counts.values()) != manifest.fts_rows:
        raise ValueError("ASR FTS row count differs from coverage")
    covered = 0
    unverified = 0
    for video, coverage in manifest.videos.items():
        if (coverage.expected_keyframes != expected[video] or coverage.segment_count != counts[video]
                or coverage.observed_keyframes != len(observed[video])):
            raise ValueError(f"ASR per-video counts mismatch: {video}")
        if (coverage.status == "success") != (counts[video] > 0):
            raise ValueError(f"ASR status inconsistent with actual segments: {video}")
        valid = coverage.status == "success" or (coverage.status == "silent" and coverage.silence_verified)
        unverified += coverage.status == "silent" and not coverage.silence_verified
        if coverage.error != (coverage.status == "error") or coverage.coverage_fraction != float(valid):
            raise ValueError(f"ASR false complete per-video coverage: {video}")
        covered += valid
    statuses = Counter(value.status for value in manifest.videos.values())
    if (covered != manifest.covered_videos or not math.isclose(manifest.coverage_fraction, covered / len(expected))
            or unverified != manifest.unverified_silent_videos
            or manifest.success_videos != statuses["success"] or manifest.silent_videos != statuses["silent"]
            or manifest.error_videos + manifest.missing_videos != statuses["error"]
            or manifest.source_records + manifest.missing_videos != len(expected)):
        raise ValueError("ASR aggregate coverage differs from actual per-video coverage")
    return manifest
