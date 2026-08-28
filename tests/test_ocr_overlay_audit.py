import unittest

from offline.ocr_overlay_audit import OverlayAuditPolicy, audit_overlay_residuals


def _region(region_id, text, bbox, *, residual=True):
    return {
        "region_id": region_id,
        "bbox_normalized": bbox,
        "easyocr_text": text,
        "final_text": text,
        "final_confidence": 0.2,
        "final_engine": "easyocr",
        "gemini_residual": residual,
    }


class OverlayAuditTests(unittest.TestCase):
    def setUp(self):
        self.policy = OverlayAuditPolicy(
            min_distinct_frames=2,
            min_video_frame_fraction=0.1,
        )

    def test_repeated_top_corner_is_candidate_but_text_is_preserved(self):
        logo = [0.82, 0.05, 0.92, 0.05, 0.92, 0.14, 0.82, 0.14]
        frames = [
            {
                "video_id": "L01_V001",
                "shot_id": f"s{index:06d}",
                "keyframe_uid": index,
                "regions": [_region(f"r{index}", "VTV" if index == 1 else "7?", logo)],
            }
            for index in range(1, 4)
        ]
        report, rows = audit_overlay_residuals(frames, policy=self.policy)
        self.assertEqual(report["gemini_residual"]["suppression_candidates"]["regions"], 3)
        self.assertTrue(all(row["overlay_audit_suppression_candidate"] for row in rows))
        self.assertEqual([row["final_text"] for row in rows], ["VTV", "7?", "7?"])
        self.assertFalse(report["invariants"]["raw_ocr_text_deleted"])

    def test_bottom_ticker_is_never_suppressed(self):
        ticker = [0.05, 0.80, 0.95, 0.80, 0.95, 0.90, 0.05, 0.90]
        frames = [
            {
                "video_id": "L01_V001",
                "shot_id": "s000001",
                "keyframe_uid": index,
                "regions": [_region(f"r{index}", "tin tức quan trọng", ticker)],
            }
            for index in range(1, 4)
        ]
        report, rows = audit_overlay_residuals(frames, policy=self.policy)
        self.assertEqual(report["gemini_residual"]["suppression_candidates"]["regions"], 0)
        self.assertTrue(all(row["gemini_residual_after_overlay_audit"] for row in rows))

    def test_policy_cannot_silently_become_production(self):
        with self.assertRaisesRegex(ValueError, "audit_only"):
            OverlayAuditPolicy(mode="production")


if __name__ == "__main__":
    unittest.main()
