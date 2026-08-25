import unittest
from unittest.mock import patch

from offline.visual_models import TransformersImageEncoder


class VisualModelTests(unittest.TestCase):
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
