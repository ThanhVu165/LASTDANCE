import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services import ocr_search


class OcrSearchTests(unittest.TestCase):
    def tearDown(self):
        ocr_search._load_ocr_cache_versioned.cache_clear()

    def _with_cache(self, payload):
        temporary = tempfile.TemporaryDirectory()
        path = Path(temporary.name) / "ocr_cache.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return temporary, path

    def test_phrase_match_tolerates_real_ocr_substitutions_without_patch_table(self):
        temporary, path = self._with_cache(
            {
                "L21_V001:17": {
                    "schema_version": 2,
                    "text": "SẠT LỎ MGUY HIỂM\nSụt lún ở ĐBSCL đang dižn ra rất nhanh",
                    "lines": [
                        {"text": "SẠT LỎ MGUY HIỂM", "confidence": 0.71, "box": []},
                        {
                            "text": "Sụt lún ở ĐBSCL đang dižn ra rất nhanh",
                            "confidence": 0.89,
                            "box": [],
                        },
                    ],
                }
            }
        )
        try:
            with patch.object(ocr_search, "OCR_CACHE_PATH", path):
                self.assertGreaterEqual(
                    ocr_search.ocr_match_score(
                        "L21_V001", 17, ["sạt lở nguy hiểm"]
                    ),
                    0.80,
                )
                self.assertGreaterEqual(
                    ocr_search.ocr_match_score(
                        "L21_V001", 17, ["đang diễn ra rất nhanh"]
                    ),
                    0.80,
                )
        finally:
            temporary.cleanup()

    def test_short_words_and_substrings_do_not_create_false_matches(self):
        temporary, path = self._with_cache({"V:1": "cartoon on a sign"})
        try:
            with patch.object(ocr_search, "OCR_CACHE_PATH", path):
                self.assertEqual(ocr_search.ocr_match_score("V", 1, ["car"]), 0.0)
                self.assertEqual(ocr_search.ocr_match_score("V", 1, ["an"]), 0.0)
                self.assertEqual(ocr_search.ocr_match_score("V", 1, ["on"]), 1.0)
        finally:
            temporary.cleanup()

    def test_structured_cache_answer_prefers_informative_line_over_logo_and_clock(self):
        temporary, path = self._with_cache(
            {
                "V:7": {
                    "schema_version": 2,
                    "text": "stale text must not win",
                    "lines": [
                        {"text": "HTV9", "confidence": 0.99, "box": []},
                        {"text": "06:30:34", "confidence": 0.99, "box": []},
                        {
                            "text": "TRÁI TIM ĐƯỢC VẬN CHUYỂN CẤP TỐC VỀ HUẾ",
                            "confidence": 0.91,
                            "box": [[0, 0], [10, 0], [10, 2], [0, 2]],
                        },
                    ],
                }
            }
        )
        try:
            with patch.object(ocr_search, "OCR_CACHE_PATH", path):
                self.assertEqual(
                    ocr_search.extract_answer_from_ocr("V", 7),
                    "TRÁI TIM ĐƯỢC VẬN CHUYỂN CẤP TỐC VỀ HUẾ",
                )
                self.assertNotIn("stale", ocr_search.ocr_text("V", 7))
        finally:
            temporary.cleanup()

    def test_legacy_string_cache_remains_readable_during_migration(self):
        temporary, path = self._with_cache({"V:3": "Tình trạng sụt lún"})
        try:
            with patch.object(ocr_search, "OCR_CACHE_PATH", path):
                self.assertEqual(ocr_search.ocr_text("V", 3), "Tình trạng sụt lún")
                self.assertEqual(
                    ocr_search.ocr_match_score("V", 3, ["sut", "lun"]), 1.0
                )
        finally:
            temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
