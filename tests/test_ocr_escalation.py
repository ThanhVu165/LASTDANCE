import unittest
from pathlib import Path

from pydantic import ValidationError

from offline.ocr_escalation import (
    CraftDetectionStatus,
    CraftFrameFeatures,
    OcrEscalationPolicy,
    OcrEscalationUsage,
    authorize_escalation_usage,
    select_gemini_escalations,
)


def _row(uid: int, **overrides) -> CraftFrameFeatures:
    values = {
        "keyframe_uid": uid,
        "video_id": "L21_V001",
        "shot_id": "s000001",
        "frame_id": 100 + uid,
        "source_image": f"L21_V001/s000001_{uid}.jpg",
        "status": CraftDetectionStatus.TEXT_DETECTED,
        "detected_region_count": 1,
        "detector_confidence": 0.9,
        "min_region_area_ratio": 0.01,
    }
    values.update(overrides)
    return CraftFrameFeatures(**values)


class OcrEscalationTests(unittest.TestCase):
    def test_committed_policy_is_valid_and_medium_resolution(self):
        policy_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "ocr_escalation_policy.json"
        )
        policy = OcrEscalationPolicy.model_validate_json(
            policy_path.read_text(encoding="utf-8")
        )
        self.assertEqual(policy.max_paid_frames, 20_000)
        self.assertEqual(policy.max_budget_vnd, 400_000)
        self.assertEqual(policy.media_resolution, "MEDIA_RESOLUTION_MEDIUM")

    def test_craft_no_text_never_enters_gemini(self):
        no_text = _row(
            1,
            status=CraftDetectionStatus.NO_TEXT,
            detected_region_count=0,
            detector_confidence=None,
            min_region_area_ratio=None,
            visual_priority=1.0,
        )
        selection = select_gemini_escalations(
            [no_text],
            policy=OcrEscalationPolicy(),
            estimated_prompt_tokens_per_request=1000,
            estimated_output_tokens_per_request=200,
        )
        self.assertEqual(selection.local_no_text_records, 1)
        self.assertEqual(selection.selected_paid_frames, 0)

    def test_three_frames_in_one_shot_count_as_one_request(self):
        selection = select_gemini_escalations(
            [_row(1), _row(2), _row(3)],
            policy=OcrEscalationPolicy(),
            estimated_prompt_tokens_per_request=1000,
            estimated_output_tokens_per_request=200,
        )
        self.assertEqual(selection.selected_paid_frames, 3)
        self.assertEqual(selection.selected_paid_requests, 1)

    def test_reuse_requires_embedding_craft_ssim_and_phash(self):
        source = _row(1)
        target = _row(
            2,
            reuse_source_uid=1,
            embedding_cosine_to_source=0.99,
            craft_layout_similarity_to_source=0.99,
            crop_ssim_to_source=0.99,
            crop_phash_distance_to_source=1,
        )
        unsafe = _row(
            3,
            reuse_source_uid=1,
            embedding_cosine_to_source=0.99,
            craft_layout_similarity_to_source=0.99,
            crop_ssim_to_source=0.90,
            crop_phash_distance_to_source=1,
        )
        selection = select_gemini_escalations(
            [source, target, unsafe],
            policy=OcrEscalationPolicy(),
            estimated_prompt_tokens_per_request=1000,
            estimated_output_tokens_per_request=200,
        )
        self.assertEqual(selection.reuse_records, 1)
        self.assertEqual(selection.reuse[0].target_keyframe_uid, 2)
        self.assertEqual(
            {candidate.keyframe_uid for candidate in selection.candidates},
            {1, 3},
        )

    def test_budget_overflow_routes_whole_later_shot_to_easyocr(self):
        policy = OcrEscalationPolicy(
            max_paid_frames=10,
            max_budget_vnd=10,
            retry_reserve_fraction=0,
            usd_to_vnd=1,
            batch_input_usd_per_million_tokens=1_000_000,
            batch_output_usd_per_million_tokens=0,
        )
        rows = [
            _row(1, shot_id="s000001", visual_priority=1.0),
            _row(2, shot_id="s000001", visual_priority=1.0),
            _row(3, shot_id="s000002", visual_priority=0.0),
        ]
        selection = select_gemini_escalations(
            rows,
            policy=policy,
            estimated_prompt_tokens_per_request=6,
            estimated_output_tokens_per_request=1,
        )
        self.assertEqual(selection.budget_request_cap, 1)
        self.assertEqual(selection.selected_paid_requests, 1)
        self.assertEqual(
            {candidate.keyframe_uid for candidate in selection.candidates},
            {1, 2},
        )
        self.assertEqual(selection.overflow_keyframe_uids, [3])

    def test_usage_ledger_fails_before_frame_or_vnd_cap(self):
        frame_policy = OcrEscalationPolicy(max_paid_frames=1, max_budget_vnd=400_000)
        initial = OcrEscalationUsage(model_id=frame_policy.model_id)
        used = authorize_escalation_usage(
            initial,
            additional_frames=1,
            additional_requests=1,
            estimated_prompt_tokens=100,
            estimated_output_tokens=10,
            policy=frame_policy,
        )
        with self.assertRaisesRegex(RuntimeError, "frame cap"):
            authorize_escalation_usage(
                used,
                additional_frames=1,
                additional_requests=1,
                estimated_prompt_tokens=100,
                estimated_output_tokens=10,
                policy=frame_policy,
            )

        cost_policy = OcrEscalationPolicy(
            max_paid_frames=10,
            max_budget_vnd=1,
            retry_reserve_fraction=0,
            usd_to_vnd=1,
            batch_input_usd_per_million_tokens=1_000_000,
            batch_output_usd_per_million_tokens=0,
        )
        with self.assertRaisesRegex(RuntimeError, "VND budget"):
            authorize_escalation_usage(
                OcrEscalationUsage(model_id=cost_policy.model_id),
                additional_frames=1,
                additional_requests=1,
                estimated_prompt_tokens=2,
                estimated_output_tokens=0,
                policy=cost_policy,
            )

    def test_incomplete_reuse_evidence_is_rejected(self):
        with self.assertRaises(ValidationError):
            _row(4, reuse_source_uid=1, embedding_cosine_to_source=0.99)

    def test_duplicate_feature_uid_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "duplicate feature UID"):
            select_gemini_escalations(
                [_row(1), _row(1)],
                policy=OcrEscalationPolicy(),
                estimated_prompt_tokens_per_request=1000,
                estimated_output_tokens_per_request=200,
            )


if __name__ == "__main__":
    unittest.main()
