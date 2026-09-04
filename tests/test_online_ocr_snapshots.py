import json
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

from offline.artifacts import sha256_file
from offline.ocr_snapshot import OcrSnapshotManifest
from offline.ocr_v2_snapshot import build_ocr_v2_snapshot
from online.artifacts import (
    CatalogFrame,
    FrameCatalog,
    _inspect_ocr,
    _load_catalog,
    load_ocr_snapshot_summary,
)
from online.config import OnlineLayout
from shared.schemas.online import ArtifactAvailability
from tests import test_ocr_v2_snapshot as ocr_v2_fixtures


class OnlineOcrSnapshotCompatibilityTests(unittest.TestCase):
    @staticmethod
    def _write_fts(path: Path) -> None:
        connection = sqlite3.connect(path)
        try:
            connection.execute(
                "CREATE VIRTUAL TABLE ocr_fts USING fts5("
                "video_id UNINDEXED, keyframe_uid UNINDEXED, detected_text, "
                "language UNINDEXED, confidence UNINDEXED)"
            )
            connection.execute(
                "INSERT INTO ocr_fts VALUES (?, ?, ?, ?, ?)",
                ("v1", 101, "Việt Nam", "vi", 0.9),
            )
            connection.commit()
        finally:
            connection.close()

    def _write_legacy_snapshot(self, root: Path, schema_version: int):
        snapshot = root / "ocr" / "snapshots" / (
            f"ocr-snapshot-20260904T14000{schema_version}Z-123456789abc"
        )
        snapshot.mkdir(parents=True)
        sqlite_path = snapshot / "ocr.sqlite"
        self._write_fts(sqlite_path)
        catalog = FrameCatalog(
            [CatalogFrame(101, "v1", 0, 10, 1.0, "s1")],
            sha256="a" * 64,
        )
        manifest = OcrSnapshotManifest(
            schema_version=schema_version,
            snapshot_id=snapshot.name,
            created_utc="2026-09-04T14:00:00+00:00",
            source_format="ocr_envelope_v1",
            materialized_text_policy="test",
            builder_sha256="b" * 64,
            catalog_path="frames.csv",
            catalog_sha256=catalog.sha256,
            catalog_records=1,
            catalog_videos=1,
            source_artifacts=[],
            source_records=1,
            observed_uid_sha256=catalog.uid_set_sha256,
            success_keyframes=1,
            no_text_keyframes=0,
            error_keyframes=0,
            missing_keyframes=0,
            covered_videos=1,
            coverage_fraction=1.0,
            sqlite_sha256=sha256_file(sqlite_path),
            sqlite_bytes=sqlite_path.stat().st_size,
            fts_rows=1,
            fts_probe={},
            videos={
                "v1": {
                    "expected_keyframes": 1,
                    "observed_keyframes": 1,
                    "success_keyframes": 1,
                    "no_text_keyframes": 0,
                    "error_keyframes": 0,
                    "missing_keyframes": 0,
                    "coverage_fraction": 1.0,
                    "final_engine_counts": {"easyocr": 1},
                    "materialized_text_tier": "easyocr_only",
                    "vintern": {
                        "required_regions": 0,
                        "completed_regions": 0,
                        "accepted_regions": 0,
                        "residual_regions": 0,
                        "pending_regions": 0,
                        "state": "not_required",
                    },
                    "snapshot_uid_coverage_full": True,
                }
            },
        )
        coverage = snapshot / "coverage.json"
        coverage.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
        (snapshot / "SHA256SUMS").write_text(
            f"{sha256_file(sqlite_path)}  ocr.sqlite\n"
            f"{sha256_file(coverage)}  coverage.json\n",
            encoding="ascii",
        )
        layout = OnlineLayout.from_environment(
            {"AIC_DATA": str(root), "AIC_OCR_SNAPSHOT_DIR": str(snapshot)}
        )
        return layout, catalog

    def _build_v3_snapshot(self, root: Path):
        fixture_root = root / "fixture"
        fixture_root.mkdir()
        fixture = ocr_v2_fixtures.OcrV2SnapshotTests(
            methodName="test_builds_immutable_v2_snapshot_with_exact_fts_and_coverage"
        )
        source_catalog, source_state, plan, sources = fixture._fixture(fixture_root)
        data_root = root / "data"
        index = data_root / "index"
        index.mkdir(parents=True)
        shutil.copy2(source_catalog, index / "frames.csv")
        shutil.copy2(source_state, index / "frames.csv.state.json")
        destination, _ = build_ocr_v2_snapshot(
            catalog_path=source_catalog,
            catalog_state_path=source_state,
            worker_plan_path=plan,
            source_manifest_path=sources,
            source_root=fixture_root,
            output_root=data_root / "ocr" / "snapshots",
        )
        layout = OnlineLayout.from_environment(
            {"AIC_DATA": str(data_root), "AIC_OCR_SNAPSHOT_DIR": str(destination)}
        )
        return layout, _load_catalog(layout), destination

    def test_legacy_schema_1_and_2_remain_readable(self):
        for schema_version in (1, 2):
            with (
                self.subTest(schema_version=schema_version),
                tempfile.TemporaryDirectory() as temporary,
            ):
                layout, catalog = self._write_legacy_snapshot(
                    Path(temporary), schema_version
                )
                status = _inspect_ocr(layout, catalog)
                self.assertEqual(status.availability, ArtifactAvailability.READY)
                self.assertEqual(status.record_count, 1)

    def test_valid_schema_3_is_ready_with_real_engine_and_residual_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            layout, catalog, destination = self._build_v3_snapshot(Path(temporary))
            status = _inspect_ocr(layout, catalog)
            summary = load_ocr_snapshot_summary(destination / "coverage.json")
            self.assertEqual(status.availability, ArtifactAvailability.READY)
            self.assertEqual(status.record_count, 7)
            self.assertEqual(summary.source_format, "ocr_v2_batch_union_v1")
            self.assertEqual(summary.residual_regions, 1)
            self.assertIn("paddle=1", status.detail)
            self.assertIn("vietocr=6", status.detail)
            self.assertIn("residual_regions=1", status.detail)

    def test_schema_3_tamper_is_invalid(self):
        with tempfile.TemporaryDirectory() as temporary:
            layout, catalog, destination = self._build_v3_snapshot(Path(temporary))
            with (destination / "ocr.sqlite").open("ab") as handle:
                handle.write(b"tamper")
            status = _inspect_ocr(layout, catalog)
            self.assertEqual(status.availability, ArtifactAvailability.INVALID)
            self.assertIn("checksum", status.detail)

    def test_schema_3_catalog_mismatch_is_invalid(self):
        with tempfile.TemporaryDirectory() as temporary:
            layout, catalog, _ = self._build_v3_snapshot(Path(temporary))
            mismatched = FrameCatalog(catalog.frames, sha256="0" * 64)
            status = _inspect_ocr(layout, mismatched)
            self.assertEqual(status.availability, ArtifactAvailability.INVALID)
            self.assertIn("catalog_sha256", status.detail)

    def test_unknown_schema_is_invalid_without_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            layout, catalog, destination = self._build_v3_snapshot(Path(temporary))
            coverage = destination / "coverage.json"
            payload = json.loads(coverage.read_text(encoding="utf-8"))
            payload["schema_version"] = 4
            coverage.write_text(json.dumps(payload), encoding="utf-8")
            (destination / "SHA256SUMS").write_text(
                f"{sha256_file(destination / 'ocr.sqlite')}  ocr.sqlite\n"
                f"{sha256_file(coverage)}  coverage.json\n",
                encoding="ascii",
            )
            status = _inspect_ocr(layout, catalog)
            self.assertEqual(status.availability, ArtifactAvailability.INVALID)
            self.assertIn("unsupported OCR snapshot schema", status.detail)


if __name__ == "__main__":
    unittest.main()
