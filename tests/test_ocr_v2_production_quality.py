import unittest

from scripts.ocr_v2_production_quality import _metrics, _stratum


class OcrV2ProductionQualityTests(unittest.TestCase):
    def test_stratum_tracks_the_terminal_production_policy(self):
        self.assertEqual(
            _stratum(
                {
                    "selection": "unresolved",
                    "selected_engine": None,
                    "selected_text": None,
                    "easyocr_text": "cache",
                }
            ),
            "unresolved",
        )
        self.assertEqual(
            _stratum(
                {
                    "selection": "numeric_cache_or_viet_guard",
                    "selected_engine": "paddle",
                    "selected_text": "2026",
                    "easyocr_text": "202G",
                }
            ),
            "paddle",
        )
        self.assertEqual(
            _stratum(
                {
                    "selection": "vietocr_default",
                    "selected_engine": "vietocr",
                    "selected_text": "Việt Nam",
                    "easyocr_text": "viet nam",
                }
            ),
            "vietocr_changed",
        )
        self.assertEqual(
            _stratum(
                {
                    "selection": "vietocr_default",
                    "selected_engine": "vietocr",
                    "selected_text": "Hà Nội",
                    "easyocr_text": "HÀ NỘI",
                }
            ),
            "vietocr_agree",
        )

    def test_metrics_penalize_unresolved_selected_text(self):
        rows = [
            {
                "label_status": "labeled",
                "human_text": "Hà Nội 2026",
                "text_type": "numeric_or_name",
                "old": "Ha Noi 202G",
                "new": "Hà Nội 2026",
            },
            {
                "label_status": "labeled",
                "human_text": "Việt Nam",
                "text_type": "ordinary",
                "old": "Viet Nam",
                "new": "",
            },
            {
                "label_status": "false_positive",
                "human_text": "",
                "text_type": "",
                "old": "logo",
                "new": "",
            },
            {
                "label_status": "exclude_unreadable",
                "human_text": "",
                "text_type": "",
                "old": "ignored",
                "new": "ignored",
            },
        ]
        old = _metrics(rows, "old")
        new = _metrics(rows, "new")
        self.assertEqual(old["evaluated_regions"], 3)
        self.assertEqual(new["readable_regions"], 2)
        self.assertGreater(new["exact_token_recall"], old["exact_token_recall"])
        self.assertEqual(new["empty_prediction_rate_on_readable"], 0.5)
        self.assertEqual(new["numeric_name_exact_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
