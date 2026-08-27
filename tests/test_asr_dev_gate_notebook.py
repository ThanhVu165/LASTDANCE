import json
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPOSITORY_ROOT / "notebooks" / "kaggle_asr_dev_gate.ipynb"


class AsrDevGateNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.code = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell.get("cell_type") == "code"
        )

    def test_every_code_cell_compiles(self):
        for index, cell in enumerate(self.notebook["cells"]):
            if cell.get("cell_type") == "code":
                compile(
                    "".join(cell.get("source", [])),
                    f"{NOTEBOOK_PATH}:cell-{index}",
                    "exec",
                )

    def test_follows_pinned_git_checkout_structure(self):
        self.assertIn("BRANCH = 'codex/offline-asr'", self.code)
        self.assertIn(
            "EXPECTED_COMMIT = 'a6a81bcf2dfb8785668eb307923c5ae75a3f83ab'",
            self.code,
        )
        self.assertNotIn("PIN_AFTER_ASR_CODE_COMMIT", self.code)
        self.assertIn("github.com/ThanhVu165/LASTDANCE.git", self.code)
        self.assertIn("'checkout', '--detach', EXPECTED_COMMIT", self.code)
        self.assertIn("actual_commit == EXPECTED_COMMIT", self.code)
        self.assertIn("scripts/run_asr_dev_gate.py", self.code)
        self.assertNotIn("repo_markers", self.code)
        self.assertNotIn("shutil.copytree", self.code)

    def test_requires_t4_and_exact_five_video_subset(self):
        for video_id in (
            "L21_V001",
            "L21_V002",
            "L21_V003",
            "L21_V005",
            "L21_V006",
        ):
            self.assertIn(repr(video_id), self.code)
        self.assertIn("'T4' in gpu_name", self.code)
        self.assertIn("ready_count'] == 5", self.code)
        self.assertIn("expected one audio input", self.code)
        self.assertIn("manifest['codec'] == 'pcm_s16le'", self.code)

    def test_runs_each_model_in_new_process_with_real_resume_gate(self):
        self.assertIn("MODELS = ['whisper_large_v3', 'phowhisper_large']", self.code)
        self.assertIn("'--stop-after-videos', '2'", self.code)
        self.assertIn("interrupted.returncode == 75", self.code)
        self.assertIn("'--require-resume-verified'", self.code)
        self.assertIn("peak_cuda_memory_bytes", self.code)
        self.assertIn("manifest['runtime']['device'] == 'cuda'", self.code)
        self.assertIn("manifest['record_count'] == 5", self.code)
        self.assertNotIn("faster-whisper", self.code)

    def test_uploads_dev_evidence_to_private_hf_namespace(self):
        self.assertIn("HF_REPO_NAME = 'lastdance-asr-artifacts'", self.code)
        self.assertIn("api.whoami()['name']", self.code)
        self.assertIn("asr/dev-gate/dev-subset-5", self.code)
        self.assertIn("UserSecretsClient().get_secret('HF_TOKEN')", self.code)
        self.assertIn("os.environ['HF_TOKEN'] = HF_TOKEN", self.code)
        self.assertIn("private=True", self.code)
        self.assertIn("CommitOperationAdd", self.code)
        self.assertIn("remote_sha256 == archive_sha256", self.code)
        self.assertIn("production_model_selected'] is False", self.code)
        self.assertIn("actual_commit[:7]", self.code)
        for forbidden in ("'.wav'", "'.mp4'", "'.bin'", "'.safetensors'"):
            self.assertIn(forbidden, self.code)

    def test_notebook_is_clean_and_sectioned_like_gpu_production_notebooks(self):
        headings = [
            "".join(cell.get("source", []))
            for cell in self.notebook["cells"]
            if cell.get("cell_type") == "markdown"
        ]
        for expected in (
            "## Chuẩn bị trên Kaggle",
            "## 1. Clone và khóa đúng commit ASR",
            "## 2. Cài dependency Kaggle GPU",
            "## 3. Xác thực HF, T4 và model registry",
            "## 4. Resolve và verify đúng Dataset audio 5 video",
            "## 5. Chạy hai model",
            "## 6. So sánh và tạo gate report",
            "## 7. Archive và upload evidence",
        ):
            self.assertTrue(any(expected in heading for heading in headings), expected)
        for cell in self.notebook["cells"]:
            if cell.get("cell_type") == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs"), [])


if __name__ == "__main__":
    unittest.main()
