import unittest

from scripts.run_eva_clip_dev_gate import _assert_final_manifest


class EvaClipDevGateTests(unittest.TestCase):
    def setUp(self):
        self.model_config = {
            "model_id": "test/eva",
            "revision": "a" * 40,
            "expected_vector_dim": 768,
        }
        self.manifest = {
            "modality": "eva_clip",
            "record_count": 4164,
            "vector_dim": 768,
            "vector_dtype": "float16",
            "checkpoint_resume_verified": True,
            "model": {"id": "test/eva", "revision": "a" * 40},
            "runtime": {
                "device": "cuda",
                "gpu_name": "Tesla T4",
                "open_clip_torch": "3.3.0",
                "timm": "1.0.28",
            },
        }

    def test_accepts_only_the_complete_cuda_dev_contract(self):
        _assert_final_manifest(
            self.manifest,
            model_config=self.model_config,
            expected_record_count=4164,
            expected_gpu_name="Tesla T4",
        )

    def test_rejects_wrong_runtime_vector_dimension(self):
        self.manifest["vector_dim"] = 1024
        with self.assertRaisesRegex(RuntimeError, "vector_dim"):
            _assert_final_manifest(
                self.manifest,
                model_config=self.model_config,
                expected_record_count=4164,
                expected_gpu_name="Tesla T4",
            )


if __name__ == "__main__":
    unittest.main()
