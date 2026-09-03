import tempfile
import unittest
from pathlib import Path

from offline.artifacts import sha256_file
from offline.ocr_snapshot_hf import (
    SNAPSHOT_FILENAMES,
    classify_remote_snapshot,
    validate_local_snapshot_for_publish,
)
from offline.ocr_snapshot import OcrSnapshotManifest


class OcrSnapshotHfTests(unittest.TestCase):
    def _snapshot(self, destination: Path) -> Path:
        snapshot_id = "ocr-snapshot-20260828T120000Z-abcdef123456"
        target = destination / snapshot_id
        target.mkdir()
        sqlite_path = target / "ocr.sqlite"
        sqlite_path.write_bytes(b"test-sqlite")
        sqlite_sha = sha256_file(sqlite_path)
        manifest = OcrSnapshotManifest(
            snapshot_id=snapshot_id,
            created_utc="2026-08-28T12:00:00+00:00",
            source_format="gate2_easyocr_dev_v1",
            materialized_text_policy="EasyOCR text only",
            builder_sha256="a" * 64,
            catalog_path="frames.csv",
            catalog_sha256="b" * 64,
            catalog_records=1,
            catalog_videos=1,
            source_artifacts=[],
            source_records=0,
            observed_uid_sha256="c" * 64,
            success_keyframes=0,
            no_text_keyframes=0,
            error_keyframes=0,
            missing_keyframes=1,
            covered_videos=0,
            coverage_fraction=0,
            sqlite_sha256=sqlite_sha,
            sqlite_bytes=sqlite_path.stat().st_size,
            fts_rows=0,
            fts_probe={"executed": False, "reason": "no_success_rows"},
            videos={},
        )
        coverage_path = target / "coverage.json"
        coverage_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
        (target / "SHA256SUMS").write_text(
            f"{sqlite_sha}  ocr.sqlite\n{sha256_file(coverage_path)}  coverage.json\n",
            encoding="ascii",
        )
        return target

    def test_local_snapshot_preflight_and_remote_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = self._snapshot(Path(temporary))
            plan = validate_local_snapshot_for_publish(snapshot)
            self.assertEqual(plan.snapshot_id, snapshot.name)
            self.assertEqual(classify_remote_snapshot(plan=plan, repo_files=[]), "missing")
            self.assertEqual(
                classify_remote_snapshot(plan=plan, repo_files=list(plan.remote_paths)),
                "complete",
            )
            with self.assertRaisesRegex(RuntimeError, "partial"):
                classify_remote_snapshot(plan=plan, repo_files=[plan.remote_paths[0]])

    def test_preflight_rejects_modified_sqlite(self):
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = self._snapshot(Path(temporary))
            with (snapshot / "ocr.sqlite").open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                validate_local_snapshot_for_publish(snapshot)


if __name__ == "__main__":
    unittest.main()
