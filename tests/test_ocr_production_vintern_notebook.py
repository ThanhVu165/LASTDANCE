import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "scripts" / "kaggle_ocr_production_vintern.ipynb"


class OcrProductionVinternNotebookTests(unittest.TestCase):
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

    def test_official_fp16_provenance_is_pinned(self):
        self.assertIn('MODEL_ID = "5CD-AI/Vintern-1B-v3_5"', self.code)
        self.assertIn('MODEL_REVISION = "b98f263eab246eb5269ade64edbdca8a887dc44d"', self.code)
        self.assertIn('MODEL_WEIGHT_BYTES = 3_752_849_256', self.code)
        self.assertIn('MODEL_WEIGHT_SHA256 = "296a16a6bf28e6d3f0fb9298deba70b3cfa1d7519f4aa326e2f862bf2e63be05"', self.code)
        self.assertIn("torch_dtype=torch.float16", self.code)
        self.assertIn("use_flash_attn=False", self.code)

    def test_partition_and_deep_kaggle_mount_are_supported(self):
        match = re.search(r"WORKER_BATCHES = (\{.*?\n\})", self.code, re.DOTALL)
        self.assertIsNotNone(match)
        assigned = re.findall(r'"batch-(?:0[1-9])"', match.group(1))
        self.assertEqual(len(assigned), 9)
        self.assertEqual(len(set(assigned)), 9)
        self.assertIn('f"*/*/*/{directory_name}"', self.code)

    def test_source_resume_and_publish_are_fail_closed(self):
        self.assertIn("WAIT_EASYOCR_ARCHIVE", self.code)
        self.assertIn("EasyOCR source provenance mismatch", self.code)
        self.assertIn("stale Vintern checkpoint signature", self.code)
        self.assertIn("final candidate set mismatch", self.code)
        self.assertIn("Vintern completion gate failed", self.code)
        self.assertIn('f"ocr/archives/{batch_id}/vintern/', self.code)
        self.assertIn("HF round-trip Vintern archive checksum mismatch", self.code)

    def test_raw_layer_does_not_claim_calibration_or_sqlite(self):
        self.assertIn('"calibrated": False', self.code)
        self.assertIn('"searchable": False', self.code)
        self.assertNotIn("generate_content", self.code)
        self.assertNotIn("sqlite3", self.code)
        self.assertIn('"model_calls": {"vintern": len(rows), "gemini": 0}', self.code)

    def test_progress_reports_long_phases(self):
        for marker in (
            "EASYOCR_ARCHIVE_DOWNLOAD_START",
            "EASYOCR_ARCHIVE_VERIFIED",
            "VINTERN_MODEL_DOWNLOAD_START",
            "VINTERN_MODEL_LOAD_START",
            "VINTERN_MODEL_READY",
            "VINTERN_BATCH_RESUME",
            "VINTERN_PRODUCTION_PROGRESS",
        ):
            self.assertIn(marker, self.code)


if __name__ == "__main__":
    unittest.main()
