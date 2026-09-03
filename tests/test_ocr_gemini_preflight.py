import json
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from offline.ocr_gemini_preflight import (
    GeminiProductionPolicy,
    build_preflight_report,
    build_shot_requests,
    residual_regions_from_materialized,
)
from offline.ocr_vintern_calibration import (
    VinternCalibrationPolicy,
    VinternCalibrationTable,
    VinternInternalSignals,
    calibrated_vintern_confidence,
    materialize_calibrated_gate2_frames,
)
from offline.ocr_vintern_gate2 import VinternGate2Policy
from scripts.prepare_ocr_gemini_production import resolve_hf_artifacts


ROOT = Path(__file__).resolve().parents[1]


def frame(uid, shot_id, regions):
    return {
        "video_id": "L21_V001",
        "shot_id": shot_id,
        "local_idx": uid,
        "keyframe_uid": uid,
        "source_image": f"keyframes-batch-01/L21_V001/{shot_id}_{uid}.jpg",
        "regions": regions,
    }


def region(region_id, residual=True):
    return {
        "region_id": region_id,
        "bbox_px": [0, 0, 100, 0, 100, 30, 0, 30],
        "easyocr_text": "abc",
        "easyocr_confidence": 0.2,
        "final_text": "abc",
        "final_confidence": 0.2,
        "final_engine": "easyocr",
        "gemini_residual": residual,
        "gemini_residual_reasons": ["calibrated_confidence_not_greater"] if residual else [],
    }


class GeminiPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = GeminiProductionPolicy.model_validate_json(
            (ROOT / "configs/ocr_gemini_production_policy.json").read_text(encoding="utf-8")
        )

    def test_pinned_calibration_table_matches_policy(self):
        policy = VinternCalibrationPolicy.model_validate_json(
            (ROOT / "configs/ocr_vintern_calibration_policy_emergency_98.json").read_text(encoding="utf-8")
        )
        table = VinternCalibrationTable.model_validate_json(
            (ROOT / "configs/ocr_vintern_calibration_table_emergency_98.json").read_text(encoding="utf-8")
        )
        self.assertEqual(policy.evidence_tier, "emergency_single_annotator_98_of_100")
        self.assertEqual(table.ground_truth_frames, 98)
        self.assertEqual(table.policy_sha256, "344ed662992bd663522dc69ace26c56782437fff637f2e74c6984e7943293989")
        decision = calibrated_vintern_confidence(
            VinternInternalSignals(
                output_length=3,
                guard_length_limit=96,
                guard_margin_ratio=93 / 96,
                mean_token_logprob=None,
            ),
            table=table,
            policy=policy,
        )
        self.assertIsNotNone(decision)
        self.assertAlmostEqual(decision.calibrated_confidence, 0.71875)

    def test_residual_manifest_excludes_terminal_regions(self):
        rows = residual_regions_from_materialized(
            [frame(1, "s001", [region("r1"), region("r2", False)])], batch_id="batch-01"
        )
        self.assertEqual([row["region_id"] for row in rows], ["r1"])

    def test_one_request_per_shot_and_dense_shot_uses_multiple_sheets(self):
        frames = [frame(index + 1, "s001", [region(f"r{index:02d}")]) for index in range(19)]
        residuals = residual_regions_from_materialized(frames, batch_id="batch-01")
        requests = build_shot_requests(residuals, policy=self.policy)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["region_count"], 19)
        self.assertEqual(requests[0]["contact_sheet_count"], 2)
        self.assertEqual([len(page) for page in requests[0]["region_pages"]], [18, 1])

    def test_report_exactly_partitions_residuals_and_never_authorizes_api(self):
        residuals = residual_regions_from_materialized(
            [frame(1, "s001", [region("r1")]), frame(2, "s002", [region("r2")])],
            batch_id="batch-01",
        )
        requests = build_shot_requests(residuals, policy=self.policy)
        report = build_preflight_report(
            residuals,
            requests,
            policy=self.policy,
            batch_summaries={"batch-01": {"frames": 2}},
        )
        self.assertEqual(report["exact_counts"]["regions"], 2)
        self.assertEqual(report["exact_counts"]["requests"], 2)
        self.assertFalse(report["api_called"])
        self.assertFalse(report["gate"]["gemini_execution_authorized"])
        self.assertEqual(report["cost"]["kind"], "planning_estimate_not_billing_fact")

    def test_duplicate_region_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate materialized region_id"):
            residual_regions_from_materialized(
                [frame(1, "s001", [region("r1")]), frame(2, "s002", [region("r1")])],
                batch_id="batch-01",
            )

    def test_vintern_is_optional_and_missing_result_routes_directly_to_gemini(self):
        calibration_policy = VinternCalibrationPolicy.model_validate_json(
            (ROOT / "configs/ocr_vintern_calibration_policy_emergency_98.json").read_text(
                encoding="utf-8"
            )
        )
        table = VinternCalibrationTable.model_validate_json(
            (ROOT / "configs/ocr_vintern_calibration_table_emergency_98.json").read_text(
                encoding="utf-8"
            )
        )
        gate_policy = VinternGate2Policy.model_validate_json(
            (ROOT / "configs/ocr_vintern_gate2_policy.json").read_text(encoding="utf-8")
        )
        easy_frame = frame(1, "s001", [region("r1")])
        easy_frame["regions"][0].update(
            {"has_vi_marks": False, "has_ascii_word": True}
        )
        materialized, audit = materialize_calibrated_gate2_frames(
            [easy_frame],
            [],
            table=table,
            calibration_policy=calibration_policy,
            gate_policy=gate_policy,
        )
        self.assertTrue(materialized[0]["regions"][0]["gemini_residual"])
        self.assertEqual(audit[0]["decision_reason"], "missing_vintern_result")
        residuals = residual_regions_from_materialized(
            materialized, batch_id="batch-01"
        )
        self.assertEqual([row["region_id"] for row in residuals], ["r1"])

    def test_hf_download_accepts_cached_cli_login_without_env_token(self):
        calls = {}

        class FakeInfo:
            private = True
            sha = "pinned-revision"

        class FakeApi:
            def __init__(self, token):
                calls["api_token"] = token

            def repo_info(self, **kwargs):
                calls["repo_info"] = kwargs
                return FakeInfo()

        def fake_snapshot_download(**kwargs):
            calls["snapshot"] = kwargs
            return kwargs["local_dir"]

        fake_hub = types.ModuleType("huggingface_hub")
        fake_hub.HfApi = FakeApi
        fake_hub.get_token = lambda: "cached-cli-token"
        fake_hub.snapshot_download = fake_snapshot_download

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            sys.modules, {"huggingface_hub": fake_hub}
        ), patch.dict("os.environ", {"HF_TOKEN": ""}):
            root, revision = resolve_hf_artifacts(
                Namespace(
                    artifact_root=None,
                    download_dir=Path(directory),
                    hf_repo_id="owner/private-dataset",
                    hf_revision=None,
                )
            )

        self.assertEqual(revision, "pinned-revision")
        self.assertEqual(calls["api_token"], "cached-cli-token")
        self.assertEqual(calls["snapshot"]["token"], "cached-cli-token")
        self.assertEqual(root, Path(directory).resolve())


if __name__ == "__main__":
    unittest.main()
