import sqlite3
import tempfile
import unittest
from pathlib import Path

from online.artifacts import CatalogFrame, FrameCatalog, _inspect_fts
from online.config import OnlineLayout
from shared.schemas.online import ArtifactAvailability


class OnlineArtifactFtsTests(unittest.TestCase):
    def setUp(self):
        self.catalog = FrameCatalog(
            [CatalogFrame(101, "v1", 0, 10, 1.0, "s1")],
            sha256="catalog",
        )

    @staticmethod
    def _create_ocr(path: Path, *, virtual: bool = True, video_id: str = "v1") -> None:
        connection = sqlite3.connect(path)
        try:
            prefix = "CREATE VIRTUAL TABLE ocr_fts USING fts5" if virtual else "CREATE TABLE ocr_fts"
            connection.execute(
                prefix
                + "(video_id UNINDEXED, keyframe_uid UNINDEXED, detected_text, "
                "language UNINDEXED, confidence UNINDEXED)"
            )
            connection.execute("INSERT INTO ocr_fts VALUES (?, ?, ?, ?, ?)", (video_id, 101, "hello", "en", 1.0))
            connection.commit()
        finally:
            connection.close()

    def test_valid_fts5_with_uid_join_is_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ocr.sqlite"
            self._create_ocr(path)
            status = _inspect_fts(path, "ocr", self.catalog)
            self.assertEqual(status.availability, ArtifactAvailability.READY)
            self.assertEqual(status.record_count, 1)

    def test_regular_table_is_invalid_even_when_columns_match(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ocr.sqlite"
            self._create_ocr(path, virtual=False)
            status = _inspect_fts(path, "ocr", self.catalog)
            self.assertEqual(status.availability, ArtifactAvailability.INVALID)
            self.assertIn("FTS5 virtual table", status.detail)

    def test_cross_video_uid_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ocr.sqlite"
            self._create_ocr(path, video_id="v2")
            status = _inspect_fts(path, "ocr", self.catalog)
            self.assertEqual(status.availability, ArtifactAvailability.INVALID)
            self.assertIn("cross-video", status.detail)

    def test_explicit_ocr_snapshot_path_does_not_alias_production_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "ocr" / "snapshots" / "ocr-snapshot-20260828T000000Z-123456789abc"
            layout = OnlineLayout.from_environment(
                {"AIC_DATA": str(root), "AIC_OCR_SNAPSHOT_DIR": str(snapshot)}
            )
            self.assertEqual(layout.ocr, snapshot / "ocr.sqlite")
            self.assertEqual(layout.ocr_coverage, snapshot / "coverage.json")
            self.assertNotEqual(layout.ocr, root / "index" / "ocr.sqlite")


if __name__ == "__main__":
    unittest.main()
