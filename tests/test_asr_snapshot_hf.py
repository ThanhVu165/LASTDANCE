import tempfile
import unittest
from pathlib import Path

from offline.artifacts import sha256_file
from offline.asr_snapshot_hf import (
    SNAPSHOT_FILENAMES,
    classify_remote_snapshot,
    validate_local_snapshot_for_publish,
)
from offline.asr_snapshot import AsrSnapshotManifest


class AsrSnapshotHfTests(unittest.TestCase):
    def _snapshot(self, root: Path) -> Path:
        snapshot_id = "asr-snapshot-20260904T001200Z-abcdef123456"
        target = root / snapshot_id
        target.mkdir()
        sqlite_path = target / "asr.sqlite"
        sqlite_path.write_bytes(b"test-sqlite")
        sqlite_sha = sha256_file(sqlite_path)
        manifest = AsrSnapshotManifest(
            snapshot_id=snapshot_id,
            created_utc="2026-09-04T00:12:00+00:00",
            catalog_path="frames.csv",
            catalog_sha256="b" * 64,
            catalog_records=1,
            catalog_videos=1,
            source_artifacts=[],
            source_records=0,
            observed_video_sha256="c" * 64,
            success_videos=0,
            silent_videos=0,
            error_videos=0,
            missing_videos=1,
            covered_videos=0,
            coverage_fraction=0,
            sqlite_sha256=sqlite_sha,
            sqlite_bytes=sqlite_path.stat().st_size,
            fts_rows=0,
            fts_probe={"executed": False, "reason": "no_segments"},
            videos={},
        )
        coverage = target / "coverage.json"
        coverage.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
        (target / "SHA256SUMS").write_text(
            f"{sqlite_sha}  asr.sqlite\n"
            f"{sha256_file(coverage)}  coverage.json\n",
            encoding="ascii",
        )
        return target

    def test_preflight_and_remote_classification(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = validate_local_snapshot_for_publish(self._snapshot(Path(temporary)))
            self.assertEqual(set(plan.remote_paths), {
                f"asr/snapshots/{plan.snapshot_id}/{name}"
                for name in SNAPSHOT_FILENAMES
            })
            self.assertEqual(classify_remote_snapshot(plan=plan, repo_files=[]), "missing")
            self.assertEqual(
                classify_remote_snapshot(plan=plan, repo_files=list(plan.remote_paths)),
                "complete",
            )
            with self.assertRaisesRegex(RuntimeError, "partial"):
                classify_remote_snapshot(plan=plan, repo_files=[plan.remote_paths[0]])


if __name__ == "__main__":
    unittest.main()
