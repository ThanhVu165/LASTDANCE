import hashlib
import tempfile
import unittest
from pathlib import Path

from offline.artifacts import sha256_file, verify_sha256


class ArtifactIntegrityTests(unittest.TestCase):
    def test_sha256_verification_passes_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "weights.pth"
            path.write_bytes(b"known-weight-bytes")
            expected = hashlib.sha256(b"known-weight-bytes").hexdigest()

            self.assertEqual(sha256_file(path), expected)
            self.assertEqual(verify_sha256(path, expected.upper()), expected)
            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                verify_sha256(path, "0" * 64)

    def test_invalid_expected_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "weights.pth"
            path.write_bytes(b"x")
            with self.assertRaises(ValueError):
                verify_sha256(path, "not-a-sha")


if __name__ == "__main__":
    unittest.main()
