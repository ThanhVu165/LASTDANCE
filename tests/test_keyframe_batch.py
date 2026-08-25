import json
import tempfile
import unittest
from pathlib import Path

from offline.artifacts import sha256_file
from offline.checkpoints import CheckpointStore
from offline.config import DataLayout
from offline.preprocessing.keyframes import (
    select_keyframes,
    write_keyframe_plan_atomic,
)
from offline.preprocessing.models import ShotBoundary
from offline.preprocessing.quality import (
    QualityDecision,
    write_quality_manifest_atomic,
)
from scripts.finalize_keyframe_collection import validate_completed_shard_reports
from scripts.run_keyframe_batch import (
    BatchConfig,
    InventoryVideo,
    load_inventory_videos,
    pipeline_signature,
    process_video,
    select_inventory_videos,
    shard_inventory_videos,
    validate_shot_membership,
)


class KeyframeBatchTests(unittest.TestCase):
    @staticmethod
    def _config(root: Path) -> BatchConfig:
        layout = DataLayout(root)
        return BatchConfig(
            layout=layout,
            shots_dir=layout.shots,
            plans_dir=layout.index / "keyframe-plans",
            quality_dir=layout.index / "keyframe-quality",
            extraction_state=layout.index / "keyframe_extraction_state.json",
            ffprobe="ffprobe",
            ffmpeg="ffmpeg",
            jpeg_quality=3,
            checkpoint_every=25,
            blur_threshold=None,
            phash_max_distance=None,
        )

    def test_inventory_loader_and_subset_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            inventory_path = Path(folder) / "inventory.json"
            inventory_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "videos": [
                            {
                                "video_id": "V2",
                                "relative_path": "videos/V2.mp4",
                            },
                            {
                                "video_id": "V1",
                                "relative_path": "videos/V1.mp4",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            inventory = load_inventory_videos(inventory_path)
            self.assertEqual([row.video_id for row in inventory], ["V1", "V2"])
            selected = select_inventory_videos(inventory, ["V2"])
            self.assertEqual([row.video_id for row in selected], ["V2"])
            with self.assertRaisesRegex(RuntimeError, "outside inventory"):
                select_inventory_videos(inventory, ["V3"])

            first_shard = shard_inventory_videos(
                inventory,
                shard_count=2,
                shard_index=0,
            )
            second_shard = shard_inventory_videos(
                inventory,
                shard_count=2,
                shard_index=1,
            )
            self.assertEqual([row.video_id for row in first_shard], ["V1"])
            self.assertEqual([row.video_id for row in second_shard], ["V2"])
            with self.assertRaisesRegex(ValueError, "0 <= index"):
                shard_inventory_videos(inventory, shard_count=2, shard_index=2)

            shots = Path(folder) / "shots"
            shots.mkdir()
            (shots / "V1.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "missing=.*V2"):
                validate_shot_membership(shots, inventory)
            (shots / "V2.json").write_text("{}", encoding="utf-8")
            validate_shot_membership(shots, inventory)
            (shots / "unexpected.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "unexpected=.*unexpected"):
                validate_shot_membership(shots, inventory)

    def test_finalizer_requires_disjoint_exact_inventory_coverage(self):
        reports = [
            {
                "scope": "collection-shard-01-of-02",
                "complete": True,
                "requested": 2,
                "completed": ["V1"],
                "skipped": ["V3"],
                "failed": {},
            },
            {
                "scope": "collection-shard-02-of-02",
                "complete": True,
                "requested": 1,
                "completed": ["V2"],
                "skipped": [],
                "failed": {},
            },
        ]
        validate_completed_shard_reports(
            reports,
            expected_video_ids=["V1", "V2", "V3"],
        )
        reports[1]["completed"] = ["V1"]
        with self.assertRaisesRegex(RuntimeError, "overlap"):
            validate_completed_shard_reports(
                reports,
                expected_video_ids=["V1", "V2", "V3"],
            )
        reports[1]["completed"] = ["V4"]
        with self.assertRaisesRegex(RuntimeError, "do not cover inventory"):
            validate_completed_shard_reports(
                reports,
                expected_video_ids=["V1", "V2", "V3"],
            )

    def test_pipeline_signature_changes_with_shot_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = self._config(root)
            video = InventoryVideo("V1", "videos/V1.mp4")
            source = root / video.relative_path
            source.parent.mkdir(parents=True)
            source.write_bytes(b"video")
            config.shots_dir.mkdir(parents=True)
            shot = config.shots_dir / "V1.json"
            shot.write_bytes(b"first")
            first = pipeline_signature(config, video)
            shot.write_bytes(b"second")
            second = pipeline_signature(config, video)
            self.assertNotEqual(first, second)

    def test_completed_video_is_validated_and_skipped(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = self._config(root)
            video = InventoryVideo("V1", "videos/V1.mp4")
            source = root / video.relative_path
            source.parent.mkdir(parents=True)
            source.write_bytes(b"video")
            config.shots_dir.mkdir(parents=True)
            (config.shots_dir / "V1.json").write_text("{}", encoding="utf-8")

            items = select_keyframes(
                video_id="V1",
                shots=[ShotBoundary("s0", 0, 2)],
                frame_timestamps=[0.0, 0.1, 0.2],
            )
            plan_path = config.plans_dir / "V1.json"
            write_keyframe_plan_atomic(
                plan_path,
                video_id="V1",
                relative_video_path=video.relative_path,
                items=items,
            )
            for item in items:
                image = root / item.relative_image_path
                image.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(b"jpeg")

            decisions = [
                QualityDecision(
                    keyframe_uid=item.frame.keyframe_uid,
                    video_id=item.frame.video_id,
                    shot_id=item.frame.shot_id,
                    local_idx=item.frame.local_idx,
                    frame_id=item.frame.frame_id,
                    relative_image_path=item.relative_image_path,
                    laplacian_variance=10.0,
                    phash="00",
                    kept=True,
                    reason="kept",
                )
                for item in items
            ]
            quality_path = config.quality_dir / "V1.json"
            write_quality_manifest_atomic(
                quality_path,
                video_id="V1",
                source_plan_sha256=sha256_file(plan_path),
                blur_threshold=None,
                phash_max_distance=None,
                decisions=decisions,
            )

            checkpoint = CheckpointStore(
                config.layout.index / "keyframe-batch.checkpoint.json"
            )
            signature = pipeline_signature(config, video)
            checkpoint.update(
                video_id="V1",
                stage="keyframe-pipeline",
                signature=signature,
                next_index=3,
                total=3,
            )

            def unexpected_runner(*args, **kwargs):
                del args, kwargs
                raise AssertionError("completed video must not run a subprocess")

            status, kept = process_video(
                config,
                video,
                checkpoint,
                runner=unexpected_runner,
            )
            self.assertEqual(status, "skipped")
            self.assertEqual(kept, 3)


if __name__ == "__main__":
    unittest.main()
