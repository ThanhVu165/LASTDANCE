import tempfile
import unittest
from pathlib import Path

from scripts.run_shot_batch import read_video_ids


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


if __name__ == "__main__":
    unittest.main()
