import csv
import json
import tempfile
import unittest
from pathlib import Path

from offline.ocr_v2_gate import (
    EASYOCR_MODEL_ID,
    MODEL_IDS,
    PADDLE_MODEL_ID,
    VIETOCR_MODEL_ID,
    OcrV2GatePolicy,
    evaluate_gate_a,
    evaluate_gate_b,
    levenshtein,
    normalize_text,
)


def policy(**overrides):
    values = {
        "schema_version": 1,
        "detector_sample_frames": 20,
        "detector_video_ids": ["V1", "V2"],
        "recognizer_sample_regions": 20,
        "recognizer_min_usable_regions": 20,
        "numeric_name_min_regions": 5,
        "detector_failure_trigger": 0.2,
        "token_recall_min_absolute_gain": 0.05,
        "cer_min_relative_reduction": 0.1,
        "numeric_name_max_absolute_regression": 0.02,
        "token_recall_tie_margin": 0.02,
        "throughput_canary_regions": 20,
        "catalog_archive_count": 1,
        "production_workers": 4,
        "production_max_hours": 18.0,
        "model_ids": list(MODEL_IDS),
    }
    values.update(overrides)
    return OcrV2GatePolicy.model_validate(values)


class OcrV2GateTests(unittest.TestCase):
    def test_unicode_normalization_and_edit_distance(self):
        self.assertEqual(normalize_text("  GIA\u0301   DẦU "), "giá dầu")
        self.assertEqual(levenshtein("kitten", "sitting"), 3)

    def test_gate_a_triggers_dbnet_at_twenty_percent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["keyframe_uid", "video_id", "gt_has_text", "bbox_quality"])
                writer.writeheader()
                for index in range(20):
                    writer.writerow(
                        {
                            "keyframe_uid": index,
                            "video_id": "V1" if index < 10 else "V2",
                            "gt_has_text": "yes",
                            "bbox_quality": "miss" if index < 4 else "correct",
                        }
                    )
            report = evaluate_gate_a(path, policy())
        self.assertEqual(report["decision"], "RUN_DBNET_CHALLENGER")
        self.assertEqual(report["bbox_issue_rate"], 0.2)

    def test_gate_b_selects_clear_winner_and_preserves_exact_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = []
            labels = []
            results = []
            for index in range(20):
                region_id = f"region-{index}"
                sample_hash = f"hash-{index}"
                samples.append(
                    {
                        "region_id": region_id,
                        "sample_row_sha256": sample_hash,
                        "video_id": "V1" if index < 10 else "V2",
                    }
                )
                truth = f"giá dầu {index}" if index < 5 else f"thành phố {index}"
                labels.append(
                    {
                        "region_id": region_id,
                        "sample_row_sha256": sample_hash,
                        "label_status": "labeled",
                        "human_text": truth,
                        "text_type": "numeric_or_name" if index < 5 else "ordinary",
                    }
                )
                for model_id in MODEL_IDS:
                    if model_id == EASYOCR_MODEL_ID:
                        text = "sai"
                    elif model_id == PADDLE_MODEL_ID:
                        text = truth
                    else:
                        text = truth if index < 18 else "sai"
                    results.append(
                        {
                            "region_id": region_id,
                            "sample_row_sha256": sample_hash,
                            "model_id": model_id,
                            "status": "success",
                            "text": text,
                        }
                    )
            sample_path = root / "sample.jsonl"
            sample_path.write_text("".join(json.dumps(row) + "\n" for row in samples), encoding="utf-8")
            label_path = root / "labels.csv"
            with label_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(labels[0]))
                writer.writeheader()
                writer.writerows(labels)
            result_path = root / "results.jsonl"
            result_path.write_text("".join(json.dumps(row) + "\n" for row in results), encoding="utf-8")
            runtime_path = root / "runtime.json"
            runtime_path.write_text(
                json.dumps(
                    {
                        "catalog_evidence": {
                            "archive_count": 1,
                            "catalog_regions": 2000,
                            "manifest_sha256s": ["a" * 64]
                        },
                        "models": {
                            PADDLE_MODEL_ID: {
                                "benchmark_regions": 20,
                                "elapsed_seconds": 2.0,
                                "regions_per_second": 10.0,
                                "error_count": 0,
                                "peak_vram_mb": 1000,
                            },
                            VIETOCR_MODEL_ID: {
                                "benchmark_regions": 20,
                                "elapsed_seconds": 1.0,
                                "regions_per_second": 20.0,
                                "error_count": 0,
                                "peak_vram_mb": 1200,
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            report = evaluate_gate_b(sample_path, label_path, result_path, runtime_path, policy())
        self.assertEqual(report["decision"], "SELECT_RECOGNIZER_CHALLENGER")
        self.assertEqual(report["selected_model_id"], PADDLE_MODEL_ID)
        self.assertEqual(report["metrics"][PADDLE_MODEL_ID]["normalized_exact_line_accuracy"], 1.0)

    def test_gate_b_fails_on_missing_model_result(self):
        self.assertIn(VIETOCR_MODEL_ID, MODEL_IDS)


if __name__ == "__main__":
    unittest.main()
