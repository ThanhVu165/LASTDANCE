import tempfile
import unittest
from pathlib import Path

from offline.catalog import write_frames_catalog_atomic
from offline.identifiers import make_keyframe_uid
from offline.ocr_catalog import audit_ocr_catalog
from shared.schemas.frame import FrameRecord


class OcrCatalogAuditTests(unittest.TestCase):
    def test_audit_recomputes_uid_and_hashes_complete_catalog(self):
        with tempfile.TemporaryDirectory() as temporary:
            catalog = Path(temporary) / "frames.csv"
            record = FrameRecord(
                video_id="L21_V001",
                local_idx=0,
                frame_id=15,
                pts_time=0.5,
                shot_id="s000001",
                keyframe_uid=make_keyframe_uid("L21_V001", "s000001", 0),
            )
            write_frames_catalog_atomic(
                catalog,
                records=[record],
                sources=[
                    {
                        "video_id": "L21_V001",
                        "plan_sha256": "a" * 64,
                        "quality_sha256": "b" * 64,
                        "quality_config_signature": "c" * 64,
                    }
                ],
            )
            report = audit_ocr_catalog(catalog)
            self.assertEqual(report["record_count"], 1)
            self.assertEqual(report["video_count"], 1)
            self.assertEqual(report["keyframe_uid_formula_mismatches"], 0)

    def test_audit_rejects_hash_valid_catalog_with_wrong_uid_formula(self):
        with tempfile.TemporaryDirectory() as temporary:
            catalog = Path(temporary) / "frames.csv"
            record = FrameRecord(
                video_id="L21_V001",
                local_idx=0,
                frame_id=15,
                pts_time=0.5,
                shot_id="s000001",
                keyframe_uid=123,
            )
            write_frames_catalog_atomic(
                catalog,
                records=[record],
                sources=[
                    {
                        "video_id": "L21_V001",
                        "plan_sha256": "a" * 64,
                        "quality_sha256": "b" * 64,
                        "quality_config_signature": "c" * 64,
                    }
                ],
            )
            with self.assertRaisesRegex(RuntimeError, "BASELINE_SPEC"):
                audit_ocr_catalog(catalog)


if __name__ == "__main__":
    unittest.main()
