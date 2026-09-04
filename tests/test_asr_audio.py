import json
import shutil
import unittest
from pathlib import Path
from subprocess import CompletedProcess

from offline.asr_audio import extract_audio_flac, safe_video_path


class AsrAudioTests(unittest.TestCase):
    def test_path_traversal_is_rejected(self):
        with self.assertRaises(ValueError):
            safe_video_path("../outside.mp4", "asr_audio_test/videos")

    def test_extraction_uses_mono_16k_and_atomic_publish(self):
        root = Path("asr_audio_test")
        shutil.rmtree(root, ignore_errors=True)
        try:
            videos = root / "videos"
            videos.mkdir(parents=True)
            source = videos / "clip.mp4"
            source.write_bytes(b"video")

            def runner(command, **kwargs):
                if command[0] == "ffprobe":
                    return CompletedProcess(command, 0, json.dumps({
                        "format": {"duration": "3.5"},
                        "streams": [{"codec_type": "audio"}],
                    }), "")
                self.assertTrue(command[-1].endswith(".staging.flac"))
                Path(command[-1]).write_bytes(b"FLAC")
                return CompletedProcess(command, 0, "", "")

            result = extract_audio_flac(source, root / "audio.flac", videos_root=videos, runner=runner)
            self.assertEqual(result.duration_seconds, 3.5)
            self.assertTrue((root / "audio.flac").is_file())
            self.assertNotIn(".staging", [path.name for path in root.iterdir()])
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
