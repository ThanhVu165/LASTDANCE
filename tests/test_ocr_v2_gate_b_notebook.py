import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "kaggle_ocr_v2_gate_b.ipynb"
RUNTIME = ROOT / "scripts" / "kaggle_ocr_v2_gate_b_runtime.py"


class OcrV2GateBNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell.get("cell_type") == "code"
        )

    def test_code_cells_compile(self):
        for index, cell in enumerate(self.notebook["cells"]):
            if cell.get("cell_type") == "code":
                compile("".join(cell["source"]), f"{NOTEBOOK}:cell-{index}", "exec")

    def test_sample_and_speed_canary_are_locked(self):
        self.assertIn("SAMPLE_SIZE = 120", self.code)
        self.assertIn("BENCHMARK_REGIONS = 5000", self.code)
        self.assertIn("same-crop", self.code)
        self.assertIn("final recognizer result coverage is not exact", self.code)

    def test_models_and_artifacts_are_pinned(self):
        self.assertIn("latin_PP-OCRv5_mobile_rec", self.code)
        self.assertIn("vietocr_vgg_seq2seq", self.code)
        self.assertIn("paddlepaddle-gpu==3.2.2", self.code)
        self.assertIn("paddleocr==3.7.0", self.code)
        self.assertIn("07b3777e5176b0d733cb056b68bd817371605f4b3514795fbf91ad4e181b8ccf", self.code)
        self.assertIn("b23105a6a1ea38e32a97c5a0ddc7e8a9bbf541d8e47421e2c99e9ccabe29509c", self.code)
        self.assertIn("0921503a41375a0584268e23ef3d414ea478a8fe8777865c7745d38f2d0bc5db", self.code)

    def test_exact_nine_manifest_eta_and_no_paid_api(self):
        self.assertIn("Attach all nine EasyOCR archives for an exact full-catalog ETA", self.code)
        self.assertIn('"gemini": 0', self.code)
        self.assertIn('"vintern": 0', self.code)
        self.assertNotIn("generate_content", self.code)
        self.assertNotIn("HfApi", self.code)

    def test_notebook_is_generated_from_runtime(self):
        self.assertIn(RUNTIME.read_text(encoding="utf-8"), self.code)


if __name__ == "__main__":
    unittest.main()
