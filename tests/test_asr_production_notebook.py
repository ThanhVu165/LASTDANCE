import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_kaggle_asr_production_notebook import build_notebook


class AsrProductionNotebookTests(unittest.TestCase):
    def test_notebook_embeds_runtime_and_invocation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = build_notebook(Path(temporary) / "asr.ipynb", worker_count=4)
            notebook = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(notebook["nbformat"], 4)
            self.assertEqual(len(notebook["cells"]), 5)
            source = "".join(
                line for cell in notebook["cells"] for line in cell["source"]
            )
            self.assertIn("condition_on_previous_text=False", source)
            self.assertIn("run_production(", source)
            self.assertIn("run_batches(", source)
            self.assertIn("WORKER_BATCHES", source)
            self.assertIn("GPU_COUNT", source)
            self.assertIn("CHECKPOINT_EVERY", source)
            self.assertIn("_push_checkpoint(", source)
            self.assertIn("_restore_checkpoint(", source)
            self.assertIn("%%writefile kaggle_asr_runtime.py", source)
            self.assertIn("CUDA_VISIBLE_DEVICES", source)
            self.assertIn("subprocess.Popen", source)


if __name__ == "__main__":
    unittest.main()
