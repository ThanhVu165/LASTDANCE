import unittest

from offline.ocr_vintern_gate2 import (
    VinternGate2Policy,
    route_vintern_region,
    vintern_output_rejection_reasons,
)


class VinternGate2RouterTests(unittest.TestCase):
    def setUp(self):
        self.policy = VinternGate2Policy()

    def test_empty_and_low_confidence_escalate(self):
        empty = route_vintern_region(
            {"easyocr_text": "", "easyocr_confidence": 0.9}, policy=self.policy
        )
        low = route_vintern_region(
            {"easyocr_text": "xin chao", "easyocr_confidence": 0.399},
            policy=self.policy,
        )
        self.assertEqual(empty.reasons, ("empty_text",))
        self.assertEqual(low.reasons, ("confidence_lt_0_40",))

    def test_policy_pins_official_fp16_checkpoint(self):
        self.assertEqual(self.policy.model_id, "5CD-AI/Vintern-1B-v3_5")
        self.assertEqual(
            self.policy.model_revision,
            "b98f263eab246eb5269ade64edbdca8a887dc44d",
        )
        self.assertEqual(self.policy.dtype, "float16")
        self.assertEqual(self.policy.max_num, 1)

    def test_frame_wide_mixed_state_cannot_propagate(self):
        unrelated_region = route_vintern_region(
            {
                "easyocr_text": "06:30:11",
                "easyocr_confidence": 0.9,
                "has_vi_marks": False,
                "has_ascii_word": False,
                "frame_mixed_candidate": True,
            },
            policy=self.policy,
        )
        mixed_region = route_vintern_region(
            {
                "easyocr_text": "HTV Việt Nam",
                "easyocr_confidence": 0.5,
                "has_vi_marks": True,
                "has_ascii_word": True,
            },
            policy=self.policy,
        )
        self.assertFalse(unrelated_region.candidate)
        self.assertEqual(mixed_region.reasons, ("region_mixed_0_40_to_0_60",))

    def test_selective_band_uses_ambiguous_glyph_only_below_ceiling(self):
        selected = route_vintern_region(
            {"easyocr_text": "FI?D", "easyocr_confidence": 0.5}, policy=self.policy
        )
        passed = route_vintern_region(
            {"easyocr_text": "FI?D", "easyocr_confidence": 0.8}, policy=self.policy
        )
        self.assertIn("ambiguous_glyph_0_40_to_0_60", selected.reasons)
        self.assertFalse(passed.candidate)

    def test_invalid_confidence_fails_closed(self):
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            route_vintern_region(
                {"easyocr_text": "text", "easyocr_confidence": 1.1},
                policy=self.policy,
            )

    def test_output_guard_rejects_explanation_and_allows_ocr_text(self):
        rejected = vintern_output_rejection_reasons(
            easyocr_text="", vintern_text="This is a blurry image, I cannot read it."
        )
        accepted = vintern_output_rejection_reasons(
            easyocr_text="06.30:11", vintern_text="06:30:11"
        )
        self.assertIn("prompt_or_explanation_leak", rejected)
        self.assertEqual(accepted, ())

    def test_output_guard_rejects_empty_output(self):
        self.assertEqual(
            vintern_output_rejection_reasons(easyocr_text="abc", vintern_text="  "),
            ("empty_output",),
        )


if __name__ == "__main__":
    unittest.main()
