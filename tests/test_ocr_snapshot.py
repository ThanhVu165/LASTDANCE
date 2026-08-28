import json
import sqlite3
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from offline.catalog import write_frames_catalog_atomic
from offline.ocr_snapshot import (
    build_ocr_snapshot,
    gate2_calibrated_vintern_coverage,
    gate2_vintern_coverage,
    load_gate2_calibrated_snapshot_records,
    load_gate2_easyocr_snapshot_records,
)
from offline.ocr_vintern_gate2 import VinternGate2Policy
from shared.schemas.frame import FrameRecord


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class OcrSnapshotTests(unittest.TestCase):
    def _catalog(self, root: Path) -> Path:
        catalog = root / "frames.csv"
        records = [
            FrameRecord(
                video_id="L01_V001",
                local_idx=0,
                frame_id=10,
                pts_time=1.0,
                shot_id="s000001",
                keyframe_uid=101,
            ),
            FrameRecord(
                video_id="L01_V001",
                local_idx=1,
                frame_id=20,
                pts_time=2.0,
                shot_id="s000001",
                keyframe_uid=102,
            ),
            FrameRecord(
                video_id="L01_V002",
                local_idx=0,
                frame_id=30,
                pts_time=3.0,
                shot_id="s000001",
                keyframe_uid=201,
            ),
        ]
        write_frames_catalog_atomic(
            catalog,
            records=records,
            sources=[
                {
                    "video_id": video_id,
                    "plan_sha256": "a" * 64,
                    "quality_sha256": "b" * 64,
                    "quality_config_signature": "c" * 64,
                }
                for video_id in ("L01_V001", "L01_V002")
            ],
        )
        return catalog

    def test_gate2_snapshot_is_immutable_partial_and_queryable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = self._catalog(root)
            easy = root / "easyocr.jsonl"
            vintern = root / "vintern.jsonl"
            _write_jsonl(
                easy,
                [
                    {
                        "video_id": "L01_V001",
                        "keyframe_uid": 101,
                        "status": "text_detected",
                        "frame_mixed_candidate": False,
                        "regions": [
                            {
                                "region_id": "r1",
                                "easyocr_text": "Xin chào",
                                "easyocr_confidence": 0.2,
                                "has_vi_marks": True,
                                "has_ascii_word": False,
                            }
                        ],
                    },
                    {
                        "video_id": "L01_V001",
                        "keyframe_uid": 102,
                        "status": "no_text",
                        "regions": [],
                    },
                ],
            )
            _write_jsonl(
                vintern,
                [
                    {
                        "candidate_id": "r1",
                        "status": "success",
                        "vintern_text": "Xin chào",
                    }
                ],
            )
            records = load_gate2_easyocr_snapshot_records(easy)
            coverage = gate2_vintern_coverage(
                easy, vintern, policy=VinternGate2Policy()
            )
            output = root / "snapshots"
            destination, manifest = build_ocr_snapshot(
                catalog_path=catalog,
                catalog_state_path=None,
                records=records,
                source_paths=[easy, vintern],
                source_format="gate2_easyocr_dev_v1",
                materialized_text_policy="EasyOCR text only",
                output_root=output,
                vintern_by_video=coverage,
                created_utc=datetime(2026, 8, 28, 12, 0, tzinfo=UTC),
            )
            self.assertFalse(manifest.complete)
            self.assertFalse(manifest.production_ready)
            self.assertTrue(manifest.immutable)
            self.assertEqual(manifest.fts_rows, 1)
            self.assertEqual(manifest.missing_keyframes, 1)
            self.assertEqual(manifest.videos["L01_V001"].vintern.state, "complete")
            self.assertEqual(
                manifest.videos["L01_V001"].materialized_text_tier,
                "easyocr_only",
            )
            self.assertEqual(
                manifest.videos["L01_V002"].materialized_text_tier,
                "unavailable",
            )
            connection = sqlite3.connect(destination / "ocr.sqlite")
            try:
                row = connection.execute(
                    "SELECT video_id,keyframe_uid,detected_text,language,confidence "
                    "FROM ocr_fts WHERE ocr_fts MATCH ?",
                    ('"chào"',),
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(row[0:4], ("L01_V001", 101, "Xin chào", "vi"))
            self.assertAlmostEqual(row[4], 0.2)
            self.assertTrue((destination / "coverage.json").is_file())
            self.assertTrue((destination / "SHA256SUMS").is_file())

            with self.assertRaises(FileExistsError):
                build_ocr_snapshot(
                    catalog_path=catalog,
                    catalog_state_path=None,
                    records=records,
                    source_paths=[easy, vintern],
                    source_format="gate2_easyocr_dev_v1",
                    materialized_text_policy="EasyOCR text only",
                    output_root=output,
                    vintern_by_video=coverage,
                    created_utc=datetime(2026, 8, 28, 12, 0, tzinfo=UTC),
                )

    def test_duplicate_snapshot_uid_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = self._catalog(root)
            easy = root / "easyocr.jsonl"
            row = {
                "video_id": "L01_V001",
                "keyframe_uid": 101,
                "status": "no_text",
                "regions": [],
            }
            _write_jsonl(easy, [row, row])
            records = load_gate2_easyocr_snapshot_records(easy)
            with self.assertRaisesRegex(ValueError, "duplicate snapshot UID"):
                build_ocr_snapshot(
                    catalog_path=catalog,
                    catalog_state_path=None,
                    records=records,
                    source_paths=[easy],
                    source_format="gate2_easyocr_dev_v1",
                    materialized_text_policy="EasyOCR text only",
                    output_root=root / "snapshots",
                )

    def test_calibrated_snapshot_materializes_vintern_text_and_tier(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = self._catalog(root)
            calibrated = root / "gate2-calibrated.jsonl"
            _write_jsonl(
                calibrated,
                [
                    {
                        "video_id": "L01_V001",
                        "keyframe_uid": 101,
                        "status": "text_detected",
                        "regions": [
                            {
                                "region_id": "r1",
                                "has_vi_marks": True,
                                "has_ascii_word": False,
                                "final_text": "Việt Nam",
                                "final_confidence": 0.8,
                                "final_engine": "vintern",
                                "vintern_override": True,
                                "calibration_bucket_id": "global",
                                "vintern_candidate": True,
                                "vintern_result_status": "success",
                                "vintern_guard_rejection_reasons": [],
                            }
                        ],
                    }
                ],
            )
            records = load_gate2_calibrated_snapshot_records(calibrated)
            coverage = gate2_calibrated_vintern_coverage(calibrated)
            destination, manifest = build_ocr_snapshot(
                catalog_path=catalog,
                catalog_state_path=None,
                records=records,
                source_paths=[calibrated],
                source_format="gate2_calibrated_dev_v1",
                materialized_text_policy="EasyOCR+Vintern calibrated",
                output_root=root / "snapshots",
                vintern_by_video=coverage,
                created_utc=datetime(2026, 8, 28, 13, 0, tzinfo=UTC),
            )
            self.assertEqual(
                manifest.videos["L01_V001"].materialized_text_tier,
                "easyocr_vintern_calibrated",
            )
            self.assertEqual(manifest.videos["L01_V001"].final_engine_counts, {"vintern": 1})
            connection = sqlite3.connect(destination / "ocr.sqlite")
            try:
                row = connection.execute(
                    "SELECT detected_text,confidence FROM ocr_fts WHERE keyframe_uid=101"
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(row[0], "Việt Nam")
            self.assertAlmostEqual(row[1], 0.8)


if __name__ == "__main__":
    unittest.main()
