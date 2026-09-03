import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from online.answering import FtsVideoAnswerer
from online.fts import FtsSearcher
from shared.schemas.online import FrameEvidence


class OnlineFtsTests(unittest.TestCase):
    def test_video_restriction_is_applied_in_sql_before_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ocr.sqlite"
            connection = sqlite3.connect(path)
            try:
                connection.execute(
                    "CREATE VIRTUAL TABLE ocr_fts USING fts5("
                    "video_id UNINDEXED, keyframe_uid UNINDEXED, detected_text, "
                    "language UNINDEXED, confidence UNINDEXED)"
                )
                connection.executemany(
                    "INSERT INTO ocr_fts VALUES (?, ?, ?, ?, ?)",
                    [
                        ("v1", 1, "hello hello hello", "en", 1.0),
                        ("v2", 2, "hello", "en", 0.8),
                    ],
                )
                connection.commit()
            finally:
                connection.close()
            hits = FtsSearcher(path, "ocr").search_hits(["hello"], limit=1, restrict_videos={"v2"})
            self.assertEqual([(item.video_id, item.keyframe_uid) for item in hits], [("v2", 2)])

    def test_full_term_match_outranks_prefix_only_bm25(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ocr.sqlite"
            connection = sqlite3.connect(path)
            try:
                connection.execute(
                    "CREATE VIRTUAL TABLE ocr_fts USING fts5("
                    "video_id UNINDEXED, keyframe_uid UNINDEXED, detected_text, "
                    "language UNINDEXED, confidence UNINDEXED)"
                )
                connection.executemany(
                    "INSERT INTO ocr_fts VALUES (?, ?, ?, ?, ?)",
                    [
                        ("target", 1, "gia dau mazut", "vi", 0.8),
                        ("noise", 2, "dau " * 100, "vi", 0.9),
                    ],
                )
                connection.commit()
            finally:
                connection.close()
            hits = FtsSearcher(path, "ocr").search_hits(["dau mazut"], limit=10)
            self.assertEqual(hits[0].keyframe_uid, 1)
            self.assertGreater(hits[0].score, next(hit.score for hit in hits if hit.keyframe_uid == 2))

    def test_unknown_number_is_read_from_neighbor_uid_without_fts_query(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ocr.sqlite"
            connection = sqlite3.connect(path)
            try:
                connection.execute(
                    "CREATE VIRTUAL TABLE ocr_fts USING fts5("
                    "video_id UNINDEXED, keyframe_uid UNINDEXED, detected_text, "
                    "language UNINDEXED, confidence UNINDEXED)"
                )
                connection.executemany(
                    "INSERT INTO ocr_fts VALUES (?, ?, ?, ?, ?)",
                    [("L21_V009", 1, "教练", "mixed", 0.8), ("L21_V009", 2, "教练6549", "mixed", 0.9)],
                )
                connection.commit()
            finally:
                connection.close()
            neighbor = SimpleNamespace(keyframe_uid=2)
            registry = SimpleNamespace(
                layout=SimpleNamespace(ocr=path, asr=Path(directory) / "asr.sqlite"),
                catalog=SimpleNamespace(neighbors=lambda uid, radius: [neighbor]),
            )
            evidence = FrameEvidence(
                keyframe_uid=1,
                video_id="L21_V009",
                frame_id=21402,
                pts_time=856.0,
                shot_id="s1",
            )
            answer, confidence, _warnings = FtsVideoAnswerer(
                registry,
                "ocr",
                [],
                value_type="number",
            ).answer(video_id="L21_V009", frames=[evidence], question="Số mấy?")
            self.assertEqual(answer, "6549")
            self.assertGreater(confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
