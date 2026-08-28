import unittest

from online.fusion import combine_query_scores, fuse_modalities, fuse_visual_channels


class OnlineFusionTests(unittest.TestCase):
    def test_srrf_rewards_cross_model_agreement(self):
        result = fuse_visual_channels(
            {1: 0.9, 2: 0.7, 3: 0.1},
            {1: 0.8, 2: 0.2, 3: 0.7},
            siglip_reference_scores=[0.9, 0.7, 0.1],
            eva_reference_scores=[0.8, 0.7, 0.2],
        )
        self.assertGreater(result.scores[1], result.scores[2])
        self.assertGreater(result.scores[1], result.scores[3])

    def test_query_expansions_use_max_plus_small_support(self):
        result = combine_query_scores([{1: 1.0, 2: 0.5}, {1: 0.5, 2: 1.0}], consensus_bonus=0.1)
        self.assertEqual(result[1], result[2])

    def test_missing_ocr_is_renormalized_not_zero_penalty(self):
        visual_only = fuse_modalities(
            {1: 0.9, 2: 0.2},
            ocr_scores={},
            modality_weights={"visual": 0.55, "ocr": 0.45},
        )
        plain = fuse_modalities({1: 0.9, 2: 0.2}, modality_weights={"visual": 1.0})
        self.assertEqual(visual_only, plain)


if __name__ == "__main__":
    unittest.main()
