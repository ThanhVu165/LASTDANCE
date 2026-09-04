import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "kaggle_ocr_v2_review.ipynb"
RUNTIME = ROOT / "scripts" / "ocr_v2_review_bundle.py"


class OcrV2ReviewNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code_cells = [
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell.get("cell_type") == "code"
        ]
        cls.code = "\n".join(cls.code_cells)

    def test_every_code_cell_compiles(self):
        for index, code in enumerate(self.code_cells):
            compile(code, f"{NOTEBOOK}:cell-{index}", "exec")

    def test_balanced_five_video_contract(self):
        self.assertIn("SAMPLE_FRAMES = 100", self.code)
        self.assertIn("SAMPLE_REGIONS = 120", self.code)
        self.assertIn("len(VIDEO_IDS) == 5", self.code)
        self.assertIn("--video-ids", self.code)

    def test_embeds_current_runtime_and_does_not_call_models(self):
        self.assertIn(RUNTIME.read_text(encoding="utf-8"), self.code)
        self.assertNotIn("easyocr.Reader", self.code)
        self.assertNotIn("TextRecognition", self.code)
        self.assertNotIn("generate_content", self.code)


if __name__ == "__main__":
    unittest.main()
