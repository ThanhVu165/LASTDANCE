import tempfile
import unittest
from pathlib import Path

from offline.checkpoints import CheckpointStore


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_resumes_monotonically_and_writes_atomically(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.json"
            store = CheckpointStore(path)
            store.update(
                video_id="L01_V001",
                stage="clip",
                signature="model-a",
                next_index=2,
                total=10,
            )
            resumed = CheckpointStore(path).get("L01_V001", "clip")
            self.assertEqual(resumed.next_index, 2)
            self.assertFalse(resumed.finished)

            completed = store.update(
                video_id="L01_V001",
                stage="clip",
                signature="model-a",
                next_index=10,
                total=10,
            )
            self.assertTrue(completed.finished)
            self.assertFalse(path.with_name("state.json.tmp").exists())

    def test_checkpoint_rejects_signature_mismatch_and_rewind(self):
        with tempfile.TemporaryDirectory() as folder:
            store = CheckpointStore(Path(folder) / "state.json")
            store.update(
                video_id="V1",
                stage="siglip",
                signature="a",
                next_index=3,
                total=5,
            )
            with self.assertRaisesRegex(RuntimeError, "signature mismatch"):
                store.update(
                    video_id="V1",
                    stage="siglip",
                    signature="b",
                    next_index=3,
                    total=5,
                )
            with self.assertRaisesRegex(RuntimeError, "move backwards"):
                store.update(
                    video_id="V1",
                    stage="siglip",
                    signature="a",
                    next_index=2,
                    total=5,
                )


if __name__ == "__main__":
    unittest.main()
