import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "scripts" / "kaggle_ocr_production_easyocr.ipynb"


class OcrProductionNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell.get("cell_type") == "code"
        )

    def test_every_code_cell_compiles(self):
        for index, cell in enumerate(self.notebook["cells"]):
            if cell.get("cell_type") == "code":
                compile("".join(cell["source"]), f"{NOTEBOOK}:cell-{index}", "exec")

    def test_worker_partition_is_disjoint_exhaustive_and_balanced(self):
        match = re.search(r"WORKER_BATCHES = (\{.*?\n\})", self.code, re.DOTALL)
        self.assertIsNotNone(match)
        self.assertIn('1: ("batch-01", "batch-09")', self.code)
        self.assertIn('2: ("batch-02", "batch-03", "batch-04")', self.code)
        self.assertIn('3: ("batch-05", "batch-08")', self.code)
        self.assertIn('4: ("batch-06", "batch-07")', self.code)
        assigned = re.findall(r'"batch-(?:0[1-9])"', match.group(1))
        self.assertEqual(len(assigned), 9)
        self.assertEqual(len(set(assigned)), 9)
        self.assertIn('"keyframe_count": 59836', self.code)
        self.assertIn('"keyframe_count": 9948', self.code)

    def test_pins_catalog_weights_and_uid_sets(self):
        self.assertIn(
            'CATALOG_SHA256 = "ee9693e75580527a0a257e9ba003984e105b059b716922c03c7a0b72b1508a37"',
            self.code,
        )
        self.assertIn(
            'BATCH_MAPPING_SHA256 = "e7e519e5fe3e47c3e487bfe0522c09c3f0bae6c7f67dff2d31168aead0b911d2"',
            self.code,
        )
        self.assertEqual(self.code.count('"uid_set_sha256":'), 9)
        self.assertIn("download_enabled=False", self.code)
        self.assertIn("weight checksum mismatch", self.code)

    def test_resume_archive_and_hf_are_fail_closed(self):
        self.assertIn("checkpoint contains foreign UID", self.code)
        self.assertIn("final UID set mismatch", self.code)
        self.assertIn("completion gate failed", self.code)
        self.assertIn("media file leaked into OCR archive", self.code)
        self.assertIn('remote_path = f"ocr/archives/{batch_id}/easyocr/', self.code)
        self.assertIn("HF round-trip archive checksum mismatch", self.code)
        self.assertIn("HF Dataset must be private", self.code)

    def test_phase_one_never_calls_vintern_or_gemini(self):
        self.assertNotIn("AutoModel.from_pretrained", self.code)
        self.assertNotIn("model.chat", self.code)
        self.assertNotIn("generate_content", self.code)
        self.assertIn('"model_calls": {"gemini": 0, "vintern": 0}', self.code)

    def test_runtime_reports_each_long_phase(self):
        self.assertIn('PROGRESS_EVERY = 25', self.code)
        self.assertIn('"EASYOCR_MODEL_INIT_START"', self.code)
        self.assertIn('"EASYOCR_MODEL_INIT_DONE"', self.code)
        self.assertIn('"CATALOG_SCAN_START"', self.code)
        self.assertIn('"CATALOG_SCAN_DONE"', self.code)

    def test_deep_kaggle_dataset_mount_is_supported(self):
        self.assertIn('f"*/*/*/{directory_name}"', self.code)


if __name__ == "__main__":
    unittest.main()
