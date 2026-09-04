import sqlite3
import json
from offline.asr_validation import validate_asr_bundle
from offline.artifacts import sha256_file
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from offline.asr_artifacts import AsrRecordEnvelope
from offline.asr_snapshot import build_asr_snapshot
from offline.catalog import write_frames_catalog_atomic
from offline.identifiers import make_keyframe_uid
from shared.schemas.frame import FrameRecord


class AsrSnapshotTests(unittest.TestCase):
    def _catalog(self, root: Path) -> tuple[Path, FrameRecord]:
        frame = FrameRecord(
            video_id="v1",
            local_idx=0,
            frame_id=0,
            pts_time=1.0,
            shot_id="s1",
            keyframe_uid=make_keyframe_uid("v1", "s1", 0),
        )
        catalog = root / "frames.csv"
        write_frames_catalog_atomic(
            catalog,
            records=[frame],
            sources=[
                {
                    "video_id": "v1",
                    "plan_sha256": "a" * 64,
                    "quality_sha256": "b" * 64,
                    "quality_config_signature": "c" * 64,
                }
            ],
        )
        return catalog, frame

    def test_builds_queryable_immutable_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog, frame = self._catalog(root)
            envelope = AsrRecordEnvelope(
                batch_id="batch-01",
                video_id="v1",
                status="success",
                engine="whisper_large_v3",
                audio_path="asr/audio/batch-01/v1.flac",
                duration_seconds=3.0,
                segments=[
                    {
                        "video_id": "v1",
                        "segment_id": "s000000",
                        "start_time": 0.5,
                        "end_time": 1.5,
                        "transcribed_text": "hello world",
                        "language": "en",
                        "keyframe_uid_nearest": frame.keyframe_uid,
                    }
                ],
            )
            destination, manifest = build_asr_snapshot(
                catalog_path=catalog,
                records=[envelope],
                output_root=root / "snapshots",
                created_utc=datetime(2026, 9, 4, tzinfo=UTC),
            )
            self.assertEqual(manifest.fts_rows, 1)
            self.assertEqual(manifest.success_videos, 1)
            connection = sqlite3.connect(destination / "asr.sqlite")
            try:
                row = connection.execute(
                    'SELECT video_id,transcribed_text,keyframe_uid_nearest '
                    'FROM asr_fts WHERE asr_fts MATCH ?',
                    ('"hello"',),
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(row, ("v1", "hello world", frame.keyframe_uid))
            with self.assertRaises(FileExistsError):
                build_asr_snapshot(
                    catalog_path=catalog,
                    records=[envelope],
                    output_root=root / "snapshots",
                    created_utc=datetime(2026, 9, 4, tzinfo=UTC),
                )

    def test_legacy_silence_is_readable_but_forged_coverage_and_wrong_catalog_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog, frame = self._catalog(root)
            envelope = AsrRecordEnvelope(batch_id="batch-01", video_id="v1", status="silent",
                engine="whisper_large_v3", audio_path="audio/v1.flac", duration_seconds=3)
            destination, manifest = build_asr_snapshot(catalog_path=catalog, records=[envelope], output_root=root / "snapshots")
            self.assertEqual(manifest.covered_videos, 0)
            self.assertEqual(manifest.unverified_silent_videos, 1)
            arguments = {"catalog_sha256": sha256_file(catalog), "frames": {frame.keyframe_uid: frame}}
            validate_asr_bundle(destination / "asr.sqlite", destination / "coverage.json", **arguments)
            with self.assertRaisesRegex(ValueError, "catalog"):
                validate_asr_bundle(destination / "asr.sqlite", destination / "coverage.json",
                                    catalog_sha256="0" * 64, frames=arguments["frames"])
            forged = manifest.model_dump(mode="json")
            forged["videos"]["v1"]["coverage_fraction"] = 1.0
            (destination / "coverage.json").write_text(json.dumps(forged), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "false complete"):
                validate_asr_bundle(destination / "asr.sqlite", destination / "coverage.json", **arguments)

    def test_rejects_foreign_video_before_snapshot_publish(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog, _ = self._catalog(root)
            envelope = AsrRecordEnvelope(
                batch_id="batch-01",
                video_id="foreign",
                status="error",
                engine="whisper_large_v3",
                audio_path="asr/audio/batch-01/foreign.flac",
                duration_seconds=0,
                error_code="transcription_failed",
            )
            with self.assertRaisesRegex(ValueError, "foreign ASR video"):
                build_asr_snapshot(
                    catalog_path=catalog,
                    records=[envelope],
                    output_root=root / "snapshots",
                )


if __name__ == "__main__":
    unittest.main()
