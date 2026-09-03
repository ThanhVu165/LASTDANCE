import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from offline.artifacts import sha256_file
from offline.catalog import write_frames_catalog_atomic
from offline.ocr_incremental_snapshot import prepare_incremental_snapshot_union
from offline.ocr_snapshot import build_ocr_snapshot
from shared.schemas.frame import FrameRecord


class OcrIncrementalSnapshotTests(unittest.TestCase):
    def _catalog(self, root: Path) -> Path:
        path = root / "frames.csv"
        write_frames_catalog_atomic(
            path,
            records=[
                FrameRecord(video_id="V1", local_idx=0, frame_id=10, pts_time=1, shot_id="s1", keyframe_uid=101),
                FrameRecord(video_id="V1", local_idx=1, frame_id=20, pts_time=2, shot_id="s1", keyframe_uid=102),
                FrameRecord(video_id="V2", local_idx=0, frame_id=30, pts_time=3, shot_id="s2", keyframe_uid=201),
            ],
            sources=[
                {"video_id": video, "plan_sha256": "a" * 64, "quality_sha256": "b" * 64, "quality_config_signature": "c" * 64}
                for video in ("V1", "V2")
            ],
        )
        return path

    def _easyocr(self, path: Path) -> None:
        rows = [
            {
                "video_id": "V1",
                "keyframe_uid": 101,
                "status": "text_detected",
                "regions": [{"region_id": "r1", "easyocr_text": "Việt Nam", "easyocr_confidence": 0.8, "has_vi_marks": True, "has_ascii_word": False}],
            },
            {"video_id": "V1", "keyframe_uid": 102, "status": "no_text", "regions": []},
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_union_exposes_batch_tiers_and_keeps_craft_out_of_fts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = self._catalog(root)
            easy = root / "batch-01.easyocr.jsonl"
            self._easyocr(easy)
            plan = root / "plan.json"
            plan.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "catalog_sha256": sha256_file(catalog),
                        "expected_batch_ids": ["batch-01", "batch-02"],
                        "batches": [
                            {
                                "batch_id": "batch-01",
                                "tier": "easyocr",
                                "video_ids": ["V1"],
                                "source_format": "gate2_easyocr_dev_v1",
                                "source_jsonl": easy.name,
                                "updated_utc": "2026-08-28T10:00:00+00:00",
                            },
                            {
                                "batch_id": "batch-02",
                                "tier": "craft_only",
                                "video_ids": ["V2"],
                                "source_format": "craft_jsonl_v1",
                                "source_jsonl": None,
                                "updated_utc": "2026-08-28T10:00:00+00:00",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            records, batches, sources = prepare_incremental_snapshot_union(
                plan_path=plan, catalog_path=catalog
            )
            self.assertEqual(len(records), 2)
            self.assertTrue(batches["batch-01"].complete)
            self.assertEqual(batches["batch-01"].tier, "easyocr")
            self.assertFalse(batches["batch-02"].complete)
            self.assertFalse(batches["batch-02"].searchable)
            destination, manifest = build_ocr_snapshot(
                catalog_path=catalog,
                catalog_state_path=None,
                records=records,
                source_paths=sources,
                source_format="incremental_batch_union_v1",
                materialized_text_policy="test",
                output_root=root / "snapshots",
                batch_coverage=batches,
            )
            self.assertEqual(set(manifest.batches), {"batch-01", "batch-02"})
            coverage = json.loads((destination / "coverage.json").read_text(encoding="utf-8"))
            self.assertEqual(coverage["schema_version"], 2)
            self.assertEqual(coverage["batches"]["batch-02"]["tier"], "craft_only")
            connection = sqlite3.connect(destination / "ocr.sqlite")
            try:
                self.assertEqual(connection.execute("SELECT count(*) FROM ocr_fts").fetchone()[0], 1)
            finally:
                connection.close()

    def test_catalog_hash_and_video_overlap_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = self._catalog(root)
            plan = root / "plan.json"
            plan.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "catalog_sha256": "0" * 64,
                        "expected_batch_ids": ["batch-01", "batch-02"],
                        "batches": [
                            {"batch_id": "batch-01", "tier": "craft_only", "video_ids": ["V1"], "source_format": "craft_jsonl_v1", "updated_utc": "2026-08-28T10:00:00Z"},
                            {"batch_id": "batch-02", "tier": "craft_only", "video_ids": ["V1", "V2"], "source_format": "craft_jsonl_v1", "updated_utc": "2026-08-28T10:00:00Z"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "overlap"):
                prepare_incremental_snapshot_union(plan_path=plan, catalog_path=catalog)


if __name__ == "__main__":
    unittest.main()
