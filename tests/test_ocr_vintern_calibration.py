import unittest

from offline.ocr_vintern_calibration import (
    VinternCalibrationPolicy,
    calibration_examples_from_rows,
    calibrated_vintern_confidence,
    derive_vintern_signals,
    fit_vintern_calibration,
    make_calibration_example,
    materialize_calibrated_gate2_frames,
    validate_materialized_calibration_audit,
)
from offline.ocr_vintern_gate2 import VinternGate2Policy


class VinternCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.policy = VinternCalibrationPolicy(
            min_ground_truth_frames=1,
            min_total_labeled_regions=4,
            min_bucket_samples=3,
        )

    def _examples(self):
        values = [
            ("r1", "abc", "abc"),
            ("r2", "abd", "abd"),
            ("r3", "abe", "abf"),
            ("r4", "a much longer source", "a much longer source"),
        ]
        return [
            make_calibration_example(
                candidate_id=candidate_id,
                easyocr_text="bad",
                easyocr_confidence=0.2,
                result={"status": "success", "vintern_text": vintern_text},
                ground_truth_text=truth,
            )
            for candidate_id, truth, vintern_text in values
        ]

    def test_confidence_is_empirical_bucket_accuracy_with_backoff(self):
        table = fit_vintern_calibration(
            self._examples(), policy=self.policy, ground_truth_frame_count=4
        )
        short = derive_vintern_signals(
            easyocr_text="bad",
            vintern_text="abc",
            result={"vintern_text": "abc"},
        )
        decision = calibrated_vintern_confidence(
            short, table=table, policy=self.policy
        )
        self.assertEqual(decision.bucket_support, 3)
        self.assertEqual(decision.bucket_correct, 2)
        self.assertAlmostEqual(decision.calibrated_confidence, 2 / 3)

        sparse = derive_vintern_signals(
            easyocr_text="bad",
            vintern_text="a much longer source",
            result={"vintern_text": "a much longer source"},
        )
        fallback = calibrated_vintern_confidence(
            sparse, table=table, policy=self.policy
        )
        self.assertEqual(fallback.bucket_id, "global")
        self.assertAlmostEqual(fallback.calibrated_confidence, 3 / 4)

    def test_materialization_overrides_only_when_strictly_better_and_guard_passes(self):
        table = fit_vintern_calibration(
            self._examples(), policy=self.policy, ground_truth_frame_count=4
        )
        frames = [
            {
                "video_id": "L01_V001",
                "keyframe_uid": 101,
                "status": "text_detected",
                "regions": [
                    {
                        "region_id": "a",
                        "easyocr_text": "abd",
                        "easyocr_confidence": 0.2,
                    },
                    {
                        "region_id": "b",
                        "easyocr_text": "high",
                        "easyocr_confidence": 0.9,
                    },
                    {
                        "region_id": "c",
                        "easyocr_text": "bad",
                        "easyocr_confidence": 0.1,
                    },
                ],
            }
        ]
        results = [
            {"candidate_id": "a", "status": "success", "vintern_text": "abc"},
            {"candidate_id": "c", "status": "success", "vintern_text": ""},
        ]
        materialized, audit = materialize_calibrated_gate2_frames(
            frames,
            results,
            table=table,
            calibration_policy=self.policy,
            gate_policy=VinternGate2Policy(),
        )
        regions = materialized[0]["regions"]
        self.assertTrue(regions[0]["vintern_override"])
        self.assertEqual(regions[0]["final_text"], "abc")
        self.assertEqual(regions[0]["final_engine"], "vintern")
        self.assertFalse(regions[1]["vintern_override"])
        self.assertEqual(regions[1]["final_engine"], "easyocr")
        self.assertFalse(regions[2]["vintern_override"])
        self.assertIn("empty_output", audit[1]["guard_rejection_reasons"])
        self.assertEqual(audit[1]["decision_reason"], "vintern_output_guard_rejected")
        validate_materialized_calibration_audit(materialized, audit, table=table)
        tampered = [dict(row) for row in audit]
        tampered[0]["final_text"] = "tampered"
        with self.assertRaisesRegex(ValueError, "final_text mismatch"):
            validate_materialized_calibration_audit(materialized, tampered, table=table)

    def test_insufficient_ground_truth_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "insufficient labeled"):
            fit_vintern_calibration(
                self._examples()[:3], policy=self.policy, ground_truth_frame_count=3
            )

    def test_deadline_policy_accepts_exactly_two_unreadable_exclusions(self):
        policy = VinternCalibrationPolicy(
            schema_version=3,
            evidence_tier="emergency_single_annotator_98_of_100",
            review_rows=100,
            review_rows_per_video=20,
            review_rows_per_stratum_per_video=4,
            review_selection_seed="vintern-gate-b-emergency-100-v1",
            min_ground_truth_frames=98,
            min_total_labeled_regions=98,
            min_bucket_samples=20,
            max_excluded_unreadable=2,
            allow_global_bucket_override=False,
        )
        self.assertEqual(policy.max_excluded_unreadable, 2)
        self.assertEqual(policy.min_ground_truth_frames, 98)

    def test_emergency_policy_never_uses_sparse_global_bucket_for_override(self):
        policy = VinternCalibrationPolicy(
            schema_version=2,
            evidence_tier="emergency_single_annotator_100",
            review_rows=100,
            review_rows_per_video=20,
            review_rows_per_stratum_per_video=4,
            review_selection_seed="vintern-gate-b-emergency-100-v1",
            min_ground_truth_frames=100,
            min_total_labeled_regions=100,
            min_bucket_samples=20,
            allow_global_bucket_override=False,
        )
        examples = []
        for index in range(100):
            text = "short" if index < 10 else "a much longer calibration phrase"
            examples.append(
                make_calibration_example(
                    candidate_id=f"r{index}",
                    easyocr_text="bad",
                    easyocr_confidence=0.1,
                    result={"status": "success", "vintern_text": text},
                    ground_truth_text=text,
                )
            )
        table = fit_vintern_calibration(
            examples, policy=policy, ground_truth_frame_count=100
        )
        signals = derive_vintern_signals(
            easyocr_text="bad",
            vintern_text="short",
            result={"status": "success", "vintern_text": "short"},
        )
        self.assertIsNone(
            calibrated_vintern_confidence(signals, table=table, policy=policy)
        )
        frames = [
            {
                "video_id": "V1",
                "keyframe_uid": 101,
                "regions": [
                    {
                        "region_id": "candidate",
                        "easyocr_text": "bad",
                        "easyocr_confidence": 0.1,
                    }
                ],
            }
        ]
        results = [
            {
                "candidate_id": "candidate",
                "status": "success",
                "vintern_text": "short",
            }
        ]
        materialized, audit = materialize_calibrated_gate2_frames(
            frames,
            results,
            table=table,
            calibration_policy=policy,
            gate_policy=VinternGate2Policy(),
        )
        region = materialized[0]["regions"][0]
        self.assertFalse(region["vintern_override"])
        self.assertTrue(region["gemini_residual"])
        self.assertEqual(
            audit[0]["decision_reason"],
            "insufficient_calibration_bucket_support",
        )
        validate_materialized_calibration_audit(materialized, audit, table=table)

    def test_empty_ground_truth_and_unreadable_exclusion_are_explicit(self):
        frames = [
            {
                "video_id": "V1",
                "keyframe_uid": 101,
                "regions": [
                    {"region_id": "noise", "easyocr_text": "", "easyocr_confidence": 0.1},
                    {"region_id": "skip", "easyocr_text": "bad", "easyocr_confidence": 0.1},
                ],
            }
        ]
        results = [
            {"candidate_id": "noise", "status": "empty", "vintern_text": ""},
            {"candidate_id": "skip", "status": "success", "vintern_text": "guess"},
        ]
        labels = [
            {
                "candidate_id": "noise",
                "keyframe_uid": 101,
                "label_status": "labeled",
                "ground_truth_is_empty": "yes",
                "human_text": "",
            },
            {
                "candidate_id": "skip",
                "keyframe_uid": 101,
                "label_status": "exclude_unreadable",
                "ground_truth_is_empty": "no",
                "human_text": "",
            },
        ]
        examples = calibration_examples_from_rows(
            frames, results, labels, gate_policy=VinternGate2Policy()
        )
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].candidate_id, "noise")
        self.assertTrue(examples[0].correct)


if __name__ == "__main__":
    unittest.main()
