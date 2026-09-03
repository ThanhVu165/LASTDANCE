import json
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPOSITORY_ROOT / "notebooks" / "kaggle_eva_clip_production.ipynb"


class EvaClipProductionNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.code = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell.get("cell_type") == "code"
        )

    def test_every_code_cell_compiles(self) -> None:
        for index, cell in enumerate(self.notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            compile(
                "".join(cell.get("source", [])),
                f"{NOTEBOOK_PATH}:cell-{index}",
                "exec",
            )

    def test_pins_the_passed_eva_gate_and_runtime_contract(self) -> None:
        self.assertIn(
            'EXPECTED_COMMIT = "63cdf244449e3eb8bcffd8d47a417ef2a07d7927"',
            self.code.replace("(\n    ", "").replace("\n)", ""),
        )
        self.assertIn("BATCH_SIZE = 32", self.code)
        self.assertIn('EVA_CLIP_EXPECTED_VECTOR_DIM = 768', self.code)
        self.assertIn('runtime["open_clip_torch"] == "3.3.0"', self.code)
        self.assertIn('runtime["timm"] == "1.0.28"', self.code)
        self.assertIn('os.environ["HF_TOKEN"] = HF_TOKEN', self.code)
        self.assertIn(
            'os.environ["HF_HOME"] = "/kaggle/working/huggingface-cache"',
            self.code,
        )
        self.assertIn(
            "0a0949293c78a3c902e4174418e22c23e9cf853350d80f274b59aea5512a89b4",
            self.code,
        )
        self.assertIn('"dev_gate_evidence": {', self.code)
        self.assertIn('"weights": {', self.code)
        self.assertIn('model_report["weights_sha256"]', self.code)

    def test_isolates_eva_remote_and_uses_uid_bound_builder(self) -> None:
        self.assertIn('remote_root = f"eva_clip/archives/{batch_id}"', self.code)
        self.assertIn('assert not remote_root.startswith("clip/")', self.code)
        self.assertIn('assert not remote_root.startswith("siglip/")', self.code)
        self.assertIn('"--modality",\n        "eva_clip",', self.code)
        self.assertIn('"--catalog",', self.code)
        self.assertIn('"--video-id-file",', self.code)
        self.assertIn('"--require-resume-verified",', self.code)
        self.assertNotIn('"--modality",\n        "clip",', self.code)
        self.assertNotIn('"--modality",\n        "siglip",', self.code)
        self.assertNotIn("eva_clip dev gate pending", self.code)

    def test_requires_the_full_nine_batch_mapping(self) -> None:
        self.assertIn('mapping["batch_count"] == 9', self.code)
        self.assertIn('mapping["video_count"] == 873', self.code)
        self.assertIn('mapping["keyframe_count"] == 293336', self.code)
        self.assertIn('len(completed_rows) == 9', self.code)
        self.assertIn('sum(\n    row["record_count"]', self.code)
        self.assertIn("RESTORED CONTROL FROM INPUT", self.code)
        self.assertIn("HF CONTROL UPLOAD PASS", self.code)
        self.assertIn("CONTROL FILE RESOLUTION START", self.code)
        self.assertNotIn("INPUT_ROOT.rglob(local_path.name)", self.code)


if __name__ == "__main__":
    unittest.main()
