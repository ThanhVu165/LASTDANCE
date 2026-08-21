import json
import tempfile
import unittest
from pathlib import Path

from app.indexing.ocr_index import (
    REQUIRED_VIETNAMESE_CHARACTERS,
    OcrItem,
    _atomic_write_json,
    _extract_lines,
    _is_complete,
    _predict_batch,
    _validate_vietnamese_alphabet,
)


class FakePipeline:
    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error

    def readtext_batched(self, inputs, **kwargs):
        if self.error:
            raise self.error
        return self.results


class FakeReader:
    def __init__(self, characters: str):
        self.lang_char = characters
        self.character = ""


class OcrIndexTests(unittest.TestCase):
    def test_extract_lines_filters_confidence_normalizes_nfc_and_keeps_geometry(self):
        result = [
            ([[0, 0], [10, 0], [10, 5], [0, 5]], "  TI\u0301NH   CHÍNH  ", 0.98),
            ([[0, 6], [10, 6], [10, 8], [0, 8]], "noise", 0.10),
            ([[0, 9], [12, 9], [12, 12], [0, 12]], "Xin  chào", 0.83),
        ]
        lines = _extract_lines(result, 0.20)
        self.assertEqual([line.text for line in lines], ["TÍNH CHÍNH", "Xin chào"])
        self.assertEqual(lines[0].box[2], (10.0, 5.0))
        self.assertEqual(lines[1].confidence, 0.83)

    def test_recognizer_must_contain_complete_vietnamese_alphabet(self):
        _validate_vietnamese_alphabet(
            FakeReader("".join(REQUIRED_VIETNAMESE_CHARACTERS))
        )
        with self.assertRaisesRegex(RuntimeError, "Vietnamese characters"):
            _validate_vietnamese_alphabet(FakeReader("abcdefghijklmnopqrstuvwxyz"))

    def test_empty_legacy_cache_is_not_complete_without_state(self):
        self.assertFalse(_is_complete({}, "L01_V001:1", "current-model"))

    def test_no_text_is_complete_only_for_same_model_signature(self):
        state = {
            "L01_V001:1": {
                "status": "no_text",
                "attempts": 1,
                "signature": "current-model",
            }
        }
        self.assertTrue(_is_complete(state, "L01_V001:1", "current-model"))
        self.assertFalse(_is_complete(state, "L01_V001:1", "different-model"))

    def test_batch_results_are_structured(self):
        items = [OcrItem("a", "a.jpg")]
        rows = [
            [
                ([[0, 0], [1, 0], [1, 1], [0, 1]], "Cảnh báo", 0.91),
            ]
        ]
        outcomes = _predict_batch(FakePipeline(results=rows), items)
        self.assertEqual(outcomes[0].text, "Cảnh báo")
        self.assertEqual(outcomes[0].cache_entry()["schema_version"], 2)
        self.assertEqual(outcomes[0].cache_entry()["lines"][0]["confidence"], 0.91)

    def test_batch_error_is_reported_per_image(self):
        items = [OcrItem("a", "a.jpg"), OcrItem("b", "b.jpg")]
        outcomes = _predict_batch(FakePipeline(error=RuntimeError("boom")), items)
        self.assertEqual([outcome.key for outcome in outcomes], ["a", "b"])
        self.assertTrue(all("boom" in outcome.error for outcome in outcomes))

    def test_atomic_json_checkpoint_is_readable(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cache.json"
            _atomic_write_json(path, {"x": {"text": "Tiếng Việt"}})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"x": {"text": "Tiếng Việt"}},
            )
            self.assertFalse(path.with_name("cache.json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
