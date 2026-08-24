import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from offline.identifiers import make_keyframe_uid
from offline.preprocessing.keyframes import (
    extract_keyframe_exact,
    extract_keyframes_exact_batch,
    load_keyframe_plan,
    probe_frame_timestamps,
    select_keyframes,
    write_keyframe_plan_atomic,
)
from offline.preprocessing.models import ShotBoundary
from offline.preprocessing.shot_detection import load_shot_manifest


class KeyframePlanTests(unittest.TestCase):
    def test_manifest_v2_exclusions_cannot_enter_keyframe_plan(self):
        total_frame_count = 31_064
        excluded_fraction = 2 / total_frame_count
        payload = {
            "schema_version": 2,
            "video_id": "L21_V006",
            "relative_video_path": "videos/L21_V006.mp4",
            "detector": "transnetv2",
            "detector_signature": {"threshold": 0.5},
            "excluded_transition_ranges": [
                {
                    "start_frame": 0,
                    "end_frame": 0,
                    "reason": "transition_score_above_threshold",
                },
                {
                    "start_frame": 1063,
                    "end_frame": 1063,
                    "reason": "transition_score_above_threshold",
                },
            ],
            "transition_exclusion_validation": {
                "total_frame_count": total_frame_count,
                "excluded_frame_count": 2,
                "excluded_frame_fraction": excluded_fraction,
                "warning_threshold": 0.01,
                "exceeds_warning_threshold": False,
            },
            "shots": [
                {"shot_id": "s000000", "start_frame": 1, "end_frame": 1062},
                {
                    "shot_id": "s000001",
                    "start_frame": 1064,
                    "end_frame": total_frame_count - 1,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / "L21_V006.json"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            video_id, _, detection = load_shot_manifest(manifest)

        items = select_keyframes(
            video_id=video_id,
            shots=detection.shots,
            frame_timestamps=[index / 30 for index in range(total_frame_count)],
            excluded_transition_ranges=detection.excluded_transition_ranges,
        )
        selected_frame_ids = {item.frame.frame_id for item in items}
        self.assertEqual(selected_frame_ids, {1, 531, 1062, 1064, 16063, 31063})
        self.assertTrue(
            all(
                not any(
                    excluded.start_frame <= frame_id <= excluded.end_frame
                    for excluded in detection.excluded_transition_ranges
                )
                for frame_id in selected_frame_ids
            )
        )

        with self.assertRaisesRegex(RuntimeError, "excluded transition frame 0"):
            select_keyframes(
                video_id=video_id,
                shots=[ShotBoundary("broken-contract", 0, 1)],
                frame_timestamps=[0.0, 1 / 30],
                excluded_transition_ranges=detection.excluded_transition_ranges,
            )

    def test_plan_loader_rejects_noncanonical_frame_order(self):
        items = select_keyframes(
            video_id="V1",
            shots=[ShotBoundary("s0", 0, 4)],
            frame_timestamps=[0.0, 0.1, 0.2, 0.3, 0.4],
        )
        payload = {
            "schema_version": 1,
            "video_id": "V1",
            "relative_video_path": "videos/V1.mp4",
            "items": [item.as_dict() for item in reversed(items)],
        }
        with tempfile.TemporaryDirectory() as folder:
            plan = Path(folder) / "V1.json"
            plan.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "strictly increasing"):
                load_keyframe_plan(plan)

    def test_selection_uses_exact_timestamp_index_and_deduplicates_short_shot(self):
        timestamps = [0.0, 0.04, 0.09, 0.15, 0.24, 0.51]
        shots = [
            ShotBoundary("s000000", 0, 4),
            ShotBoundary("s000001", 5, 5),
        ]
        items = select_keyframes(
            video_id="L01_V001",
            shots=shots,
            frame_timestamps=timestamps,
        )

        self.assertEqual([item.frame.frame_id for item in items], [0, 2, 4, 5])
        self.assertEqual([item.frame.pts_time for item in items], [0.0, 0.09, 0.24, 0.51])
        self.assertEqual([item.frame.local_idx for item in items], [0, 1, 2, 3])
        self.assertEqual(
            items[0].frame.keyframe_uid,
            make_keyframe_uid("L01_V001", "s000000", 0),
        )
        self.assertEqual(
            items[-1].relative_image_path,
            "keyframes/L01_V001/s000001_3.jpg",
        )

    def test_selection_fails_when_shot_exceeds_probed_frames(self):
        with self.assertRaisesRegex(RuntimeError, "only 2 timestamps"):
            select_keyframes(
                video_id="V1",
                shots=[ShotBoundary("s0", 0, 2)],
                frame_timestamps=[0.0, 0.1],
            )

    def test_probe_reads_monotonic_best_effort_timestamps(self):
        payload = json.dumps(
            {
                "frames": [
                    {"best_effort_timestamp_time": "0.000"},
                    {"best_effort_timestamp_time": "0.041"},
                ]
            }
        )

        def runner(command, **kwargs):
            self.assertIn("frame=best_effort_timestamp_time", command)
            return SimpleNamespace(returncode=0, stdout=payload, stderr="")

        self.assertEqual(
            probe_frame_timestamps(Path("video.mp4"), runner=runner),
            [0.0, 0.041],
        )

    def test_plan_write_is_atomic_and_contains_no_absolute_image_path(self):
        item = select_keyframes(
            video_id="V1",
            shots=[ShotBoundary("s0", 0, 0)],
            frame_timestamps=[0.0],
        )
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "plan.json"
            write_keyframe_plan_atomic(
                output,
                video_id="V1",
                relative_video_path="videos/V1.mp4",
                items=item,
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["items"][0]["relative_image_path"], "keyframes/V1/s0_0.jpg")
            self.assertFalse(output.with_name("plan.json.tmp").exists())

    def test_exact_extraction_uses_frame_selector_and_atomic_replace(self):
        item = select_keyframes(
            video_id="V1",
            shots=[ShotBoundary("s0", 0, 0)],
            frame_timestamps=[0.0],
        )[0]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root / "videos" / "V1.mp4"
            video.parent.mkdir()
            video.write_bytes(b"video")

            def runner(command, **kwargs):
                del kwargs
                self.assertIn("select=eq(n\\,0)", command)
                Path(command[-1]).write_bytes(b"jpeg")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            output = extract_keyframe_exact(
                video,
                item,
                data_root=root,
                runner=runner,
            )
            self.assertEqual(output.read_bytes(), b"jpeg")
            self.assertFalse(output.with_name(f"{output.stem}.tmp{output.suffix}").exists())

    def test_failed_extraction_does_not_publish_partial_image(self):
        item = select_keyframes(
            video_id="V1",
            shots=[ShotBoundary("s0", 0, 0)],
            frame_timestamps=[0.0],
        )[0]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root / "videos" / "V1.mp4"
            video.parent.mkdir()
            video.write_bytes(b"video")

            def runner(command, **kwargs):
                del kwargs
                Path(command[-1]).write_bytes(b"partial")
                return SimpleNamespace(returncode=1, stdout="", stderr="boom")

            with self.assertRaisesRegex(RuntimeError, "boom"):
                extract_keyframe_exact(video, item, data_root=root, runner=runner)
            output = root / item.relative_image_path
            self.assertFalse(output.exists())
            self.assertFalse(output.with_name(f"{output.stem}.tmp{output.suffix}").exists())

    def test_batch_extraction_decodes_once_and_maps_outputs_in_frame_order(self):
        items = select_keyframes(
            video_id="V1",
            shots=[ShotBoundary("s0", 0, 4)],
            frame_timestamps=[0.0, 0.1, 0.2, 0.3, 0.4],
        )
        progress = []
        runner_calls = []
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root / "videos" / "V1.mp4"
            video.parent.mkdir()
            video.write_bytes(b"video")

            def runner(command, **kwargs):
                del kwargs
                runner_calls.append(command)
                self.assertIn("select=eq(n\\,0)+eq(n\\,2)+eq(n\\,4)", command)
                pattern = str(command[-1])
                for index in range(3):
                    Path(pattern.replace("%08d", f"{index:08d}")).write_bytes(
                        f"jpeg-{index}".encode("ascii")
                    )
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            outputs = extract_keyframes_exact_batch(
                video,
                items,
                data_root=root,
                runner=runner,
                on_progress=progress.append,
            )

            self.assertEqual(len(runner_calls), 1)
            self.assertEqual(progress, [1, 2, 3])
            self.assertEqual(
                [output.read_bytes() for output in outputs],
                [b"jpeg-0", b"jpeg-1", b"jpeg-2"],
            )
            self.assertEqual(
                list((root / "keyframes").glob(".*-extract-*")),
                [],
            )

    def test_failed_batch_extraction_publishes_no_keyframes(self):
        items = select_keyframes(
            video_id="V1",
            shots=[ShotBoundary("s0", 0, 2)],
            frame_timestamps=[0.0, 0.1, 0.2],
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root / "videos" / "V1.mp4"
            video.parent.mkdir()
            video.write_bytes(b"video")

            def runner(command, **kwargs):
                del kwargs
                pattern = str(command[-1])
                Path(pattern.replace("%08d", "00000000")).write_bytes(b"partial")
                return SimpleNamespace(returncode=1, stdout="", stderr="boom")

            with self.assertRaisesRegex(RuntimeError, "boom"):
                extract_keyframes_exact_batch(
                    video,
                    items,
                    data_root=root,
                    runner=runner,
                )

            self.assertTrue(
                all(not (root / item.relative_image_path).exists() for item in items)
            )
            self.assertEqual(
                list((root / "keyframes").glob(".*-extract-*")),
                [],
            )


if __name__ == "__main__":
    unittest.main()
