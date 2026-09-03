import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from offline.ocr_models import (
    DEFAULT_EASYOCR_REGISTRY,
    load_easyocr_registry,
    verify_easyocr_offline_files,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _md5(value: bytes) -> str:
    return hashlib.md5(value, usedforsecurity=False).hexdigest()


class EasyOcrModelRegistryTests(unittest.TestCase):
    def test_official_registry_pins_download_disabled_craft_and_latin_g2(self):
        payload = load_easyocr_registry(DEFAULT_EASYOCR_REGISTRY)
        self.assertEqual(payload["package"]["version"], "1.7.2")
        self.assertFalse(payload["runtime"]["download_enabled"])
        self.assertEqual(set(payload["models"]), {"craft", "latin_g2"})

    def test_offline_verifier_checks_sha_size_and_upstream_md5(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            storage = root / "models"
            archives = root / "archives"
            storage.mkdir()
            archives.mkdir()
            craft_weights = b"craft-weights"
            latin_weights = b"latin-weights"
            craft_archive = b"craft-archive"
            latin_archive = b"latin-archive"
            wheel = b"wheel"
            (storage / "craft_mlt_25k.pth").write_bytes(craft_weights)
            (storage / "latin_g2.pth").write_bytes(latin_weights)
            (archives / "craft_mlt_25k.zip").write_bytes(craft_archive)
            (archives / "latin_g2.zip").write_bytes(latin_archive)
            wheel_path = root / "easyocr-1.7.2-py3-none-any.whl"
            wheel_path.write_bytes(wheel)
            registry = {
                "schema_version": 1,
                "package": {
                    "name": "easyocr",
                    "version": "1.7.2",
                    "wheel_filename": wheel_path.name,
                    "wheel_size_bytes": len(wheel),
                    "wheel_sha256": _sha256(wheel),
                    "pypi_source": "https://pypi.org/project/easyocr/1.7.2/",
                    "upstream_repository": "https://github.com/JaidedAI/EasyOCR",
                    "upstream_revision": "a" * 40,
                },
                "runtime": {
                    "languages": ["vi", "en"],
                    "detect_network": "craft",
                    "recognition_network": "latin_g2",
                    "download_enabled": False,
                },
                "models": {
                    "craft": {
                        "role": "detector",
                        "archive_filename": "craft_mlt_25k.zip",
                        "archive_url": "https://github.com/JaidedAI/EasyOCR/releases/download/x/craft_mlt_25k.zip",
                        "archive_size_bytes": len(craft_archive),
                        "archive_sha256": _sha256(craft_archive),
                        "weights_filename": "craft_mlt_25k.pth",
                        "weights_size_bytes": len(craft_weights),
                        "weights_sha256": _sha256(craft_weights),
                        "weights_md5_upstream": _md5(craft_weights),
                    },
                    "latin_g2": {
                        "role": "recognizer",
                        "archive_filename": "latin_g2.zip",
                        "archive_url": "https://github.com/JaidedAI/EasyOCR/releases/download/x/latin_g2.zip",
                        "archive_size_bytes": len(latin_archive),
                        "archive_sha256": _sha256(latin_archive),
                        "weights_filename": "latin_g2.pth",
                        "weights_size_bytes": len(latin_weights),
                        "weights_sha256": _sha256(latin_weights),
                        "weights_md5_upstream": _md5(latin_weights),
                    },
                },
            }
            registry_path = root / "registry.json"
            registry_path.write_text(json.dumps(registry), encoding="utf-8")

            report = verify_easyocr_offline_files(
                storage,
                registry_path=registry_path,
                archive_directory=archives,
                wheel_path=wheel_path,
                verify_package_version=False,
            )
            self.assertEqual(len(report["weights"]), 2)
            self.assertEqual(len(report["archives"]), 2)
            self.assertFalse(report["download_enabled"])

            (storage / "latin_g2.pth").write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                verify_easyocr_offline_files(
                    storage,
                    registry_path=registry_path,
                    verify_package_version=False,
                )


if __name__ == "__main__":
    unittest.main()
