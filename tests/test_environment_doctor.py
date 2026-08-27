import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.environment_doctor import (
    check_python,
    check_transnet_weights,
    collect_checks,
)


class EnvironmentDoctorTests(unittest.TestCase):
    def test_python_check_requires_311(self):
        self.assertTrue(check_python((3, 11, 9)).ok)
        self.assertFalse(check_python((3, 12, 0)).ok)

    def test_python_check_supports_kaggle_312_contract(self):
        self.assertTrue(check_python((3, 12, 13), required_minor=(3, 12)).ok)
        self.assertFalse(check_python((3, 11, 9), required_minor=(3, 12)).ok)

    def test_weight_check_requires_existing_verified_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "weights.pth"
            path.write_bytes(b"weight")
            digest = hashlib.sha256(b"weight").hexdigest()
            passed = check_transnet_weights(
                {
                    "AIC_TRANSNETV2_WEIGHTS": str(path),
                    "AIC_TRANSNETV2_WEIGHTS_SHA256": digest,
                }
            )
            self.assertTrue(passed[0].ok)

            failed = check_transnet_weights(
                {
                    "AIC_TRANSNETV2_WEIGHTS": str(path),
                    "AIC_TRANSNETV2_WEIGHTS_SHA256": "0" * 64,
                }
            )
            self.assertFalse(failed[0].ok)

    def test_external_weight_check_does_not_accept_placeholder(self):
        checks = check_transnet_weights(
            {
                "AIC_TRANSNETV2_WEIGHTS": "weights.pth",
                "AIC_TRANSNETV2_WEIGHTS_SHA256": "replace_with_verified_sha256",
            }
        )
        self.assertFalse(checks[0].ok)

    def test_bundled_weight_is_accepted_after_integrity_check(self):
        with patch(
            "scripts.environment_doctor.resolve_and_verify_transnetv2_weights",
            return_value=(Path("bundled.pth"), "a" * 64, "bundled"),
        ):
            checks = check_transnet_weights({})
        self.assertTrue(checks[0].ok)
        self.assertIn("bundled", checks[0].detail)

    def test_toolchain_only_omits_data_layout_check(self):
        with patch("scripts.environment_doctor.check_python") as python_check, patch(
            "scripts.environment_doctor.check_package"
        ) as package_check, patch(
            "scripts.environment_doctor.check_executable"
        ) as executable_check, patch(
            "scripts.environment_doctor.check_transnet_weights",
            return_value=[],
        ):
            python_check.return_value.ok = True
            package_check.return_value.ok = True
            executable_check.return_value.ok = True
            checks = collect_checks("offline-local", {}, check_data=False)
        python_check.assert_called_once_with(required_minor=(3, 11))
        self.assertNotIn("AIC_DATA", [check.name for check in checks])

    def test_kaggle_profile_requires_python_312(self):
        with patch("scripts.environment_doctor.check_python") as python_check, patch(
            "scripts.environment_doctor.check_package"
        ) as package_check, patch(
            "torch.cuda.is_available", return_value=True
        ), patch(
            "torch.cuda.get_device_name", return_value="Tesla T4"
        ):
            python_check.return_value.ok = True
            package_check.return_value.ok = True
            checks = collect_checks("kaggle-gpu", {}, check_data=False)

        python_check.assert_called_once_with(required_minor=(3, 12))
        cuda = next(check for check in checks if check.name == "cuda")
        self.assertTrue(cuda.ok)

    def test_asr_kaggle_profile_requires_python_312_and_cuda(self):
        with patch("scripts.environment_doctor.check_python") as python_check, patch(
            "scripts.environment_doctor.check_package"
        ) as package_check, patch(
            "torch.cuda.is_available", return_value=True
        ), patch(
            "torch.cuda.get_device_name", return_value="Tesla T4"
        ):
            python_check.return_value.ok = True
            package_check.return_value.ok = True
            checks = collect_checks("asr-kaggle-gpu", {}, check_data=False)

        python_check.assert_called_once_with(required_minor=(3, 12))
        cuda = next(check for check in checks if check.name == "cuda")
        self.assertTrue(cuda.ok)
        self.assertEqual(cuda.detail, "Tesla T4")

    def test_colab_shot_profile_requires_cuda_without_requiring_data(self):
        with patch("scripts.environment_doctor.check_python") as python_check, patch(
            "scripts.environment_doctor.check_package"
        ) as package_check, patch(
            "scripts.environment_doctor.check_executable"
        ) as executable_check, patch(
            "scripts.environment_doctor.check_transnet_weights",
            return_value=[],
        ), patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.get_device_name",
            return_value="Tesla T4",
        ):
            python_check.return_value.ok = True
            package_check.return_value.ok = True
            executable_check.return_value.ok = True
            checks = collect_checks("shot-colab-gpu", {}, check_data=False)

        self.assertNotIn("AIC_DATA", [check.name for check in checks])
        cuda = next(check for check in checks if check.name == "cuda")
        self.assertTrue(cuda.ok)
        self.assertEqual(cuda.detail, "Tesla T4")

    def test_windows_shot_profile_requires_cuda(self):
        with patch("scripts.environment_doctor.check_python") as python_check, patch(
            "scripts.environment_doctor.check_package"
        ) as package_check, patch(
            "scripts.environment_doctor.check_executable"
        ) as executable_check, patch(
            "scripts.environment_doctor.check_transnet_weights",
            return_value=[],
        ), patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.get_device_name",
            return_value="NVIDIA GeForce RTX 4050 Laptop GPU",
        ):
            python_check.return_value.ok = True
            package_check.return_value.ok = True
            executable_check.return_value.ok = True
            checks = collect_checks("shot-windows-gpu", {}, check_data=False)

        cuda = next(check for check in checks if check.name == "cuda")
        self.assertTrue(cuda.ok)
        self.assertIn("RTX 4050", cuda.detail)


if __name__ == "__main__":
    unittest.main()
