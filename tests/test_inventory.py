import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from offline.preprocessing.inventory import (
    build_inventory,
    discover_videos,
    probe_video,
    write_inventory_atomic,
)


def _ffprobe_payload(*, video_id: str = "unused") -> str:
    del video_id
    return json.dumps(
        {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1920,
                    "height": 1080,
                    "avg_frame_rate": "30000/1001",
                    "duration": "12.5",
                    "nb_frames": "375",
                },
                {"codec_type": "audio"},
            ],
            "format": {"duration": "12.5"},
        }
    )


class InventoryTests(unittest.TestCase):
    def test_probe_reads_real_metadata_and_stores_relative_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root / "videos" / "L01_V001.mp4"
            video.parent.mkdir()
            video.write_bytes(b"")

            calls = []

            def runner(command, **kwargs):
                calls.append((command, kwargs))
                return SimpleNamespace(
                    returncode=0,
                    stdout=_ffprobe_payload(),
                    stderr="",
                )

            record = probe_video(video, data_root=root, runner=runner)

            self.assertEqual(record.video_id, "L01_V001")
            self.assertEqual(record.relative_path, "videos/L01_V001.mp4")
            self.assertAlmostEqual(record.fps, 30000 / 1001)
            self.assertEqual(record.frame_count, 375)
            self.assertTrue(record.has_audio)
            self.assertEqual(calls[0][0][-1], str(video.resolve()))

    def test_probe_fails_closed_on_invalid_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root / "videos" / "broken.mp4"
            video.parent.mkdir()
            video.write_bytes(b"")

            def runner(command, **kwargs):
                del command, kwargs
                return SimpleNamespace(returncode=0, stdout='{"streams": []}', stderr="")

            with self.assertRaisesRegex(RuntimeError, "no video stream"):
                probe_video(video, data_root=root, runner=runner)

    def test_discovery_sorting_duplicate_detection_and_atomic_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            videos = root / "videos"
            videos.mkdir()
            (videos / "B.mp4").write_bytes(b"")
            (videos / "A.MP4").write_bytes(b"")
            (videos / "ignore.txt").write_text("x", encoding="utf-8")
            paths = discover_videos(videos)
            self.assertEqual([path.stem for path in paths], ["A", "B"])

            def runner(command, **kwargs):
                del command, kwargs
                return SimpleNamespace(
                    returncode=0,
                    stdout=_ffprobe_payload(),
                    stderr="",
                )

            records = build_inventory(paths, data_root=root, runner=runner)
            output = root / "index" / "inventory.json"
            write_inventory_atomic(output, records)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual([row["video_id"] for row in payload["videos"]], ["A", "B"])
            self.assertFalse(output.with_name("inventory.json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
