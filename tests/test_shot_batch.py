import tempfile
import unittest
from pathlib import Path

from offline.checkpoints import CheckpointStore
from offline.config import DataLayout
from offline.preprocessing.models import ShotBoundary
from offline.preprocessing.shot_detection import ShotDetectionResult
from scripts.run_shot_batch import (
    build_video_checkpoint_signature,
    read_video_ids,
    resolve_checkpoint_path,
    resolve_shots_directory,
    run_checkpointed_video,
)


class FakeDetector:
    name = "transnetv2"

    def __init__(self, *, interrupt: bool = False) -> None:
        self.interrupt = interrupt
        self.calls = 0
        self.signature = {"name": self.name, "device": "cuda"}

    def detect(self, source: Path) -> ShotDetectionResult:
        self.calls += 1
        if self.interrupt:
            raise KeyboardInterrupt
        return ShotDetectionResult(
            shots=(ShotBoundary("s000000", 0, 1),),
            total_frame_count=2,
        )


class CrashAfterPublishStore(CheckpointStore):
    def update(self, **kwargs):
        if kwargs["next_index"] == 1:
            raise KeyboardInterrupt
        return super().update(**kwargs)


class ShotBatchTests(unittest.TestCase):
    def test_video_list_is_ordered_and_canonical(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "worker.txt"
            path.write_text("L21_V001\nL21_V006\n", encoding="utf-8")
            self.assertEqual(read_video_ids(path), ["L21_V001", "L21_V006"])

    def test_video_list_rejects_whitespace_and_duplicates(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "worker.txt"
            path.write_text("L21_V001 \n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "whitespace"):
                read_video_ids(path)

            path.write_text("L21_V001\nL21_V001\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_video_ids(path)

    def test_shots_directory_must_stay_inside_data_root(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            layout = DataLayout(root)
            self.assertEqual(
                resolve_shots_directory(layout, root / "index" / "gpu-parity"),
                root / "index" / "gpu-parity",
            )
            with self.assertRaisesRegex(ValueError, "inside AIC_DATA"):
                resolve_shots_directory(layout, root.parent / "outside")

    def test_checkpoint_resumes_video_interrupted_during_gpu_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            layout = DataLayout(root)
            source = layout.videos / "V1.mp4"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"video")
            shots_directory = layout.shots
            output = shots_directory / "V1.json"
            checkpoint_path = layout.index / "shot-batches" / "worker.checkpoint.json"
            store = CheckpointStore(checkpoint_path)
            interrupted = FakeDetector(interrupt=True)

            with self.assertRaises(KeyboardInterrupt):
                run_checkpointed_video(
                    video_id="V1",
                    source=source,
                    relative_video_path="videos/V1.mp4",
                    output=output,
                    detector=interrupted,
                    expected_signature=interrupted.signature,
                    shots_directory=shots_directory,
                    data_root=root,
                    checkpoint_store=store,
                )

            progress = store.get("V1", "shot_detection")
            self.assertEqual(progress.next_index, 0)
            self.assertFalse(output.exists())

            resumed = FakeDetector()
            outcome = run_checkpointed_video(
                video_id="V1",
                source=source,
                relative_video_path="videos/V1.mp4",
                output=output,
                detector=resumed,
                expected_signature=resumed.signature,
                shots_directory=shots_directory,
                data_root=root,
                checkpoint_store=CheckpointStore(checkpoint_path),
            )

            self.assertEqual(outcome, "completed")
            self.assertEqual(resumed.calls, 1)
            self.assertTrue(
                CheckpointStore(checkpoint_path)
                .get("V1", "shot_detection")
                .finished
            )
            self.assertTrue(output.is_file())

    def test_resume_adopts_manifest_published_before_checkpoint_advance(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            layout = DataLayout(root)
            source = layout.videos / "V1.mp4"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"video")
            output = layout.shots / "V1.json"
            checkpoint_path = layout.index / "shot-batches" / "worker.checkpoint.json"
            detector = FakeDetector()

            with self.assertRaises(KeyboardInterrupt):
                run_checkpointed_video(
                    video_id="V1",
                    source=source,
                    relative_video_path="videos/V1.mp4",
                    output=output,
                    detector=detector,
                    expected_signature=detector.signature,
                    shots_directory=layout.shots,
                    data_root=root,
                    checkpoint_store=CrashAfterPublishStore(checkpoint_path),
                )

            self.assertTrue(output.is_file())
            self.assertEqual(
                CheckpointStore(checkpoint_path)
                .get("V1", "shot_detection")
                .next_index,
                0,
            )

            resumed = FakeDetector(interrupt=True)
            outcome = run_checkpointed_video(
                video_id="V1",
                source=source,
                relative_video_path="videos/V1.mp4",
                output=output,
                detector=resumed,
                expected_signature=resumed.signature,
                shots_directory=layout.shots,
                data_root=root,
                checkpoint_store=CheckpointStore(checkpoint_path),
            )

            self.assertEqual(outcome, "skipped")
            self.assertEqual(resumed.calls, 0)
            self.assertTrue(
                CheckpointStore(checkpoint_path)
                .get("V1", "shot_detection")
                .finished
            )

    def test_complete_checkpoint_without_manifest_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            layout = DataLayout(root)
            source = layout.videos / "V1.mp4"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"video")
            detector = FakeDetector()
            checkpoint_path = layout.index / "shot-batches" / "worker.checkpoint.json"
            store = CheckpointStore(checkpoint_path)
            signature = build_video_checkpoint_signature(
                source=source,
                relative_video_path="videos/V1.mp4",
                expected_signature=detector.signature,
                shots_directory=layout.shots,
                data_root=root,
            )
            store.update(
                video_id="V1",
                stage="shot_detection",
                signature=signature,
                next_index=1,
                total=1,
            )

            with self.assertRaisesRegex(RuntimeError, "manifest is missing"):
                run_checkpointed_video(
                    video_id="V1",
                    source=source,
                    relative_video_path="videos/V1.mp4",
                    output=layout.shots / "V1.json",
                    detector=detector,
                    expected_signature=detector.signature,
                    shots_directory=layout.shots,
                    data_root=root,
                    checkpoint_store=store,
                )
            self.assertEqual(detector.calls, 0)

    def test_default_checkpoint_isolated_by_device_and_output_namespace(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            layout = DataLayout(root)
            video_list = root / "worker-01.txt"
            cpu = resolve_checkpoint_path(
                layout,
                None,
                video_list=video_list,
                device="cpu",
                shots_directory=layout.shots,
            )
            cuda = resolve_checkpoint_path(
                layout,
                None,
                video_list=video_list,
                device="cuda",
                shots_directory=layout.shots,
            )
            parity = resolve_checkpoint_path(
                layout,
                None,
                video_list=video_list,
                device="cuda",
                shots_directory=layout.index / "shot-parity",
            )

            self.assertNotEqual(cpu, cuda)
            self.assertNotEqual(cuda, parity)
            self.assertTrue(cuda.is_relative_to(layout.root))


if __name__ == "__main__":
    unittest.main()
