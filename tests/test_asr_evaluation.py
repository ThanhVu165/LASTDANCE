import unittest

from offline.asr_evaluation import word_error_rate


class AsrEvaluationTests(unittest.TestCase):
    def test_optional_wer_uses_explicit_normalization(self):
        report = word_error_rate(
            ["Xin chào, thế giới!"],
            ["xin chào thế gioi"],
        )
        self.assertEqual(report["reference_words"], 4)
        self.assertEqual(report["word_errors"], 1)
        self.assertEqual(report["wer"], 0.25)


if __name__ == "__main__":
    unittest.main()
