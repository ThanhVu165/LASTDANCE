import csv
import json
import tempfile
import unittest
from pathlib import Path

from offline.catalog import (
    FRAME_COLUMNS,
    load_quality_manifest,
    select_catalog_records,
    validate_frames_catalog,
    write_frames_catalog_atomic,
)
from offline.preprocessing.keyframes import select_keyframes
from offline.preprocessing.models import ShotBoundary
from offline.preprocessing.quality import QualityDecision
from shared.schemas.frame import FrameRecord


class CatalogTests(unittest.TestCase):
    def _items(self):
        return select_keyframes(
            video_id="V1",
            shots=[ShotBoundary("s0", 0, 2)],
            frame_timestamps=[0.0, 0.1, 0.2],
        )

    @staticmethod
    def _decision(item, kept=True):
        return QualityDecision(
            keyframe_uid=item.frame.keyframe_uid,
            video_id=item.frame.video_id,
            shot_id=item.frame.shot_id,
            local_idx=item.frame.local_idx,
            frame_id=item.frame.frame_id,
            relative_image_path=item.relative_image_path,
            laplacian_variance=10.0,
            phash="00",
            kept=kept,
            reason="kept" if kept else "blur",
        )

    def test_selection_requires_exact_uid_set_and_metadata(self):
        items = self._items()
        decisions = [self._decision(item) for item in items[:-1]]
        with self.assertRaisesRegex(RuntimeError, "UID mismatch"):
            select_catalog_records(items, decisions)

        decisions = [self._decision(item) for item in items]
        decisions[0] = QualityDecision(
            **{
                **decisions[0].as_dict(),
                "frame_id": 999,
            }
        )
        with self.assertRaisesRegex(RuntimeError, "metadata mismatch"):
            select_catalog_records(items, decisions)

    def test_selection_keeps_only_reversible_quality_choices(self):
        items = self._items()
        decisions = [
            self._decision(item, kept=index != 1)
            for index, item in enumerate(items)
        ]
        records = select_catalog_records(items, decisions)
        self.assertEqual([record.local_idx for record in records], [0, 2])

    def test_catalog_publish_and_hash_bound_validation(self):
        items = self._items()
        records = [item.frame for item in items]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "frames.csv"
            state = write_frames_catalog_atomic(
                output,
                records=records,
                sources=[
                    {
                        "video_id": "V1",
                        "plan_sha256": "a" * 64,
                        "quality_sha256": "b" * 64,
                        "quality_config_signature": "c" * 64,
                    }
                ],
            )
            self.assertTrue(validate_frames_catalog(output, state))
            with output.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, FRAME_COLUMNS)
                rows = list(reader)
            self.assertEqual(rows[0]["window_id"], "")
            self.assertEqual(int(rows[0]["keyframe_uid"]), records[0].keyframe_uid)

            output.write_text(output.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            self.assertFalse(validate_frames_catalog(output, state))

    def test_catalog_rejects_duplicate_submission_key(self):
        first = self._items()[0].frame
        collision = FrameRecord(
            video_id=first.video_id,
            local_idx=99,
            frame_id=first.frame_id,
            pts_time=first.pts_time,
            shot_id="other",
            keyframe_uid=first.keyframe_uid + 1,
        )
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, "duplicate.*frame_id"):
                write_frames_catalog_atomic(
                    Path(folder) / "frames.csv",
                    records=[first, collision],
                    sources=[{"video_id": "V1"}],
                )

    def test_quality_manifest_counts_are_validated(self):
        item = self._items()[0]
        decision = self._decision(item)
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / "quality.json"
            manifest.write_text(
                json.dumps(
                    {
                        "video_id": "V1",
                        "source_plan_sha256": "a" * 64,
                        "config_signature": "b" * 64,
                        "counts": {"input": 2, "kept": 1},
                        "items": [decision.as_dict()],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "input count"):
                load_quality_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
