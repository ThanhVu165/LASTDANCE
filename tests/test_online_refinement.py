import unittest
from pathlib import Path

from online.refinement import exact_frame_command


class OnlineRefinementTests(unittest.TestCase):
    def test_exact_decode_selects_frame_index_without_fps_or_time_seek(self):
        command = exact_frame_command("ffmpeg", Path("video.mp4"), 42, Path("frame.jpg"))
        self.assertIn("select=eq(n\\,42)", command)
        self.assertNotIn("-ss", command)
        self.assertNotIn("-r", command)
        self.assertEqual(command[-1], "frame.jpg")

    def test_negative_frame_is_rejected(self):
        with self.assertRaises(ValueError):
            exact_frame_command("ffmpeg", Path("video.mp4"), -1, Path("frame.jpg"))


if __name__ == "__main__":
    unittest.main()
