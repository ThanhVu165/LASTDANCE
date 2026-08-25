import csv
import json
import tempfile
import unittest
from pathlib import Path

from offline.catalog import (
    FRAME_COLUMNS,
    discover_catalog_inputs,
    load_inventory_video_ids,
    load_quality_manifest,
    select_catalog_records,
    validate_frames_catalog,
    write_frames_catalog_atomic,
)
from offline.config import DataLayout
from offline.preprocessing.keyframes import select_keyframes
from offline.preprocessing.models import ShotBoundary
from offline.preprocessing.quality import QualityDecision
from scripts.build_frames_catalog import build_parser, resolve_catalog_inputs
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

    def test_collection_discovery_requires_exact_inventory_membership(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inventory = root / "inventory.json"
            inventory.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "videos": [
                            {"video_id": "V2"},
                            {"video_id": "V1"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            plans = root / "plans"
            qualities = root / "qualities"
            plans.mkdir()
            qualities.mkdir()
            for video_id in ("V1", "V2"):
                (plans / f"{video_id}.json").write_text("{}", encoding="utf-8")
                (qualities / f"{video_id}.json").write_text("{}", encoding="utf-8")

            video_ids = load_inventory_video_ids(inventory)
            self.assertEqual(video_ids, ["V1", "V2"])
            pairs = discover_catalog_inputs(
                plans,
                qualities,
                expected_video_ids=video_ids,
            )
            self.assertEqual(
                [(plan.stem, quality.stem) for plan, quality in pairs],
                [("V1", "V1"), ("V2", "V2")],
            )

            (qualities / "V2.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "missing_quality=.*V2"):
                discover_catalog_inputs(
                    plans,
                    qualities,
                    expected_video_ids=video_ids,
                )

    def test_collection_cli_uses_layout_defaults(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            index = root / "index"
            plans = index / "keyframe-plans"
            qualities = index / "keyframe-quality"
            plans.mkdir(parents=True)
            qualities.mkdir(parents=True)
            (index / "inventory.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "videos": [{"video_id": "V1"}],
                    }
                ),
                encoding="utf-8",
            )
            plan = plans / "V1.json"
            quality = qualities / "V1.json"
            plan.write_text("{}", encoding="utf-8")
            quality.write_text("{}", encoding="utf-8")

            args = build_parser().parse_args(
                ["--collection", "--data-root", str(root)]
            )
            pairs = resolve_catalog_inputs(args, DataLayout(root))
            self.assertEqual(pairs, [(plan.resolve(), quality.resolve())])

            conflicting = build_parser().parse_args(
                [
                    "--collection",
                    "--plan",
                    str(plan),
                    "--quality",
                    str(quality),
                ]
            )
            with self.assertRaisesRegex(ValueError, "cannot be combined"):
                resolve_catalog_inputs(conflicting, DataLayout(root))

    def test_inventory_membership_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            inventory = Path(folder) / "inventory.json"
            inventory.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "videos": [
                            {"video_id": "V1"},
                            {"video_id": "V1"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "duplicate video_id"):
                load_inventory_video_ids(inventory)
            inventory.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "top-level"):
                load_inventory_video_ids(inventory)

    def test_catalog_validation_rejects_tampered_collection_state(self):
        items = self._items()
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "frames.csv"
            state_path = write_frames_catalog_atomic(
                output,
                records=[item.frame for item in items],
                sources=[
                    {
                        "video_id": "V1",
                        "plan_sha256": "a" * 64,
                        "quality_sha256": "b" * 64,
                        "quality_config_signature": "c" * 64,
                    }
                ],
            )
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["video_count"] = 2
            state_path.write_text(json.dumps(state), encoding="utf-8")
            self.assertFalse(validate_frames_catalog(output, state_path))
            state_path.write_text("[]", encoding="utf-8")
            self.assertFalse(validate_frames_catalog(output, state_path))


if __name__ == "__main__":
    unittest.main()
