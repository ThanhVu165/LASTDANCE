import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from offline.visual_models import (
    OpenClipEVAImageEncoder,
    TransformersImageEncoder,
    _verify_eva_clip_snapshot,
    create_visual_encoder,
)


class VisualModelTests(unittest.TestCase):
    def test_eva_clip_factory_uses_the_pinned_safe_adapter(self):
        encoder = create_visual_encoder("eva_clip")
        self.assertIsInstance(encoder, OpenClipEVAImageEncoder)
        self.assertEqual(encoder.modality, "eva_clip")
        self.assertEqual(encoder.expected_vector_dim, 768)
        self.assertEqual(encoder.weights_filename, "open_clip_model.safetensors")

    def test_eva_clip_snapshot_is_verified_before_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "open_clip_config.json"
            config.write_text(
                json.dumps({"model_cfg": {"embed_dim": 768}}), encoding="utf-8"
            )
            weights = root / "open_clip_model.safetensors"
            weights.write_bytes(b"safe-tensor-test")
            expected_sha256 = hashlib.sha256(weights.read_bytes()).hexdigest()
            _verify_eva_clip_snapshot(
                config_path=config,
                weights_path=weights,
                expected_filename=weights.name,
                expected_sha256=expected_sha256,
                expected_size_bytes=weights.stat().st_size,
                expected_vector_dim=768,
            )

            unsafe = root / "open_clip_pytorch_model.bin"
            unsafe.write_bytes(b"pickle")
            with self.assertRaisesRegex(RuntimeError, "safe config/weights"):
                _verify_eva_clip_snapshot(
                    config_path=config,
                    weights_path=unsafe,
                    expected_filename=unsafe.name,
                    expected_sha256=hashlib.sha256(unsafe.read_bytes()).hexdigest(),
                    expected_size_bytes=unsafe.stat().st_size,
                    expected_vector_dim=768,
                )

    def test_runtime_metadata_records_portable_environment_provenance(self):
        encoder = TransformersImageEncoder(
            modality="clip",
            model_id="test/clip",
            model_revision="a" * 40,
        )
        versions = {"transformers": "5.15.1", "torch": "2.10.0+cu128"}
        with patch(
            "offline.visual_models.importlib.metadata.version",
            side_effect=lambda package: versions[package],
        ), patch(
            "offline.visual_models.platform.python_version", return_value="3.12.13"
        ), patch(
            "offline.visual_models.platform.system", return_value="Linux"
        ), patch(
            "offline.visual_models.platform.machine", return_value="x86_64"
        ):
            metadata = encoder.runtime_metadata

        self.assertEqual(
            metadata,
            {
                "device": "cuda",
                "python": "3.12.13",
                "system": "Linux",
                "machine": "x86_64",
                "transformers": "5.15.1",
                "torch": "2.10.0+cu128",
            },
        )


if __name__ == "__main__":
    unittest.main()
