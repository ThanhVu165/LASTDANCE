import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "scripts/kaggle_ocr_production_gemini.ipynb"


class OcrProductionGeminiNotebookTests(unittest.TestCase):
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

    def test_default_mode_cannot_call_api(self):
        self.assertIn("EXECUTION_MODE = 'preflight'", self.code)
        self.assertIn("APPROVE_PAID_CANARY = False", self.code)
        self.assertIn("APPROVE_GEMINI_PRODUCTION = False", self.code)
        preflight_position = self.code.index('if EXECUTION_MODE == "preflight"')
        secret_position = self.code.index('api_key = get_secret("GEMINI_API_KEY")')
        self.assertLess(preflight_position, secret_position)

    def test_model_schema_media_and_identity_are_pinned(self):
        self.assertIn('MODEL_ID = "gemini-2.5-flash-lite"', self.code)
        self.assertIn('MEDIA_RESOLUTION = "MEDIA_RESOLUTION_MEDIUM"', self.code)
        self.assertIn('"minItems": len(region_ids)', self.code)
        self.assertIn('"maxItems": len(region_ids)', self.code)
        self.assertIn('"region_id": {"type": "string", "enum": region_ids}', self.code)
        self.assertIn("Gemini region_id set is duplicate/missing/foreign", self.code)

    def test_user_and_canary_gates_are_fail_closed(self):
        self.assertIn("APPROVED_REPORT_SHA256 must equal", self.code)
        self.assertIn("APPROVED_MODEL_VERSION from the paid canary", self.code)
        self.assertIn("actual Gemini usage would exceed approved VND cap", self.code)
        self.assertIn("Gemini auth failure HTTP", self.code)
        self.assertIn("runtime model version drift", self.code)
        self.assertIn('RUNNER_BILLING_MODE = "standard"', self.code)
        self.assertIn("this runner uses Standard synchronous API", self.code)
        self.assertIn('report["cost"][RUNNER_BILLING_MODE]["within_budget"]', self.code)

    def test_resume_archive_and_hf_are_isolated(self):
        self.assertIn("stale Gemini checkpoint signature", self.code)
        self.assertIn('f"ocr/archives/{batch_id}/gemini/', self.code)
        self.assertIn("HF round-trip Gemini archive checksum mismatch", self.code)
        self.assertIn("GEMINI_HF_BATCH_RESTORED", self.code)
        self.assertNotIn("sqlite3", self.code)

    def test_gemini_never_owns_bbox(self):
        schema_start = self.code.index("def response_schema")
        schema_end = self.code.index("def request_payload")
        self.assertNotIn('"bbox"', self.code[schema_start:schema_end])
        self.assertIn("never return bbox", self.code)


if __name__ == "__main__":
    unittest.main()
