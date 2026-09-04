import unittest

from online.config import OnlineConfig


class OnlineConfigTests(unittest.TestCase):
    def test_loads_vqa_agreement_and_search_budget(self):
        config = OnlineConfig.load()
        self.assertEqual(config.qa_vqa_agreement_similarity, 0.6)
        self.assertEqual(config.gemini_max_calls_per_search, 14)

    def test_rejects_invalid_vqa_agreement_similarity(self):
        with self.assertRaisesRegex(ValueError, "qa_vqa_agreement_similarity"):
            OnlineConfig(qa_vqa_agreement_similarity=1.1)


if __name__ == "__main__":
    unittest.main()
