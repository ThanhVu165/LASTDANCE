import csv
import json
import tempfile
import unittest
from pathlib import Path

from offline.ocr_craft_gate_a import (
    CraftGateAPolicy,
    evaluate_craft_gate_a,
    load_gate_a_selected_config,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class CraftGateATests(unittest.TestCase):
    def setUp(self):
        self.policy = CraftGateAPolicy(
            sample_frames=2,
            video_ids=("V1", "V2"),
            frames_per_video=1,
            min_region_recall=0.5,
            min_text_frame_recall=0.5,
            configs=(
                {
                    "config_id": "recall_current",
                    "text_threshold": 0.6,
                    "low_text": 0.3,
                    "link_threshold": 0.3,
                },
                {
                    "config_id": "strict",
                    "text_threshold": 0.8,
                    "low_text": 0.5,
                    "link_threshold": 0.5,
                },
            ),
        )

    def _artifacts(self, root: Path, *, blank_label: bool = False) -> tuple[Path, Path]:
        results = root / "results.jsonl"
        rows = []
        for uid in (101, 201):
            for config in self.policy.configs:
                rows.append(
                    {
                        "keyframe_uid": uid,
                        "video_id": "V1" if uid == 101 else "V2",
                        "shot_id": "s1" if uid == 101 else "s2",
                        "config_id": config.config_id,
                        "thresholds": {
                            "text_threshold": config.text_threshold,
                            "low_text": config.low_text,
                            "link_threshold": config.link_threshold,
                        },
                        "status": "success",
                        "error": None,
                        "region_count": 3 if config.config_id == "recall_current" else 1,
                        "latency_seconds": 0.2,
                    }
                )
        _write_jsonl(results, rows)
        review = root / "review.csv"
        fields = [
            "video_id",
            "keyframe_uid",
            "gt_has_text",
            "gt_region_count",
            "missed_gt_regions__recall_current",
            "missed_gt_regions__strict",
            "annotator",
        ]
        with review.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow(
                {
                    "video_id": "V1",
                    "keyframe_uid": 101,
                    "gt_has_text": "yes",
                    "gt_region_count": 2,
                    "missed_gt_regions__recall_current": 0,
                    "missed_gt_regions__strict": 1,
                    "annotator": "" if blank_label else "human-a",
                }
            )
            writer.writerow(
                {
                    "video_id": "V2",
                    "keyframe_uid": 201,
                    "gt_has_text": "no",
                    "gt_region_count": 0,
                    "missed_gt_regions__recall_current": 0,
                    "missed_gt_regions__strict": 0,
                    "annotator": "human-a",
                }
            )
        return results, review

    def test_selects_lightest_eligible_threshold(self):
        with tempfile.TemporaryDirectory() as temporary:
            results, review = self._artifacts(Path(temporary))
            report = evaluate_craft_gate_a(
                results_path=results, review_csv_path=review, policy=self.policy
            )
            self.assertEqual(report["decision"], "PASS_THRESHOLD_SELECTED")
            self.assertEqual(report["selected_config_id"], "strict")
            self.assertTrue(report["gate_b_allowed"])

    def test_blank_human_label_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            results, review = self._artifacts(Path(temporary), blank_label=True)
            with self.assertRaisesRegex(ValueError, "annotator is required"):
                evaluate_craft_gate_a(
                    results_path=results, review_csv_path=review, policy=self.policy
                )

    def test_gate_b_preflight_rejects_tampered_pass_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results, review = self._artifacts(root)
            report = evaluate_craft_gate_a(
                results_path=results, review_csv_path=review, policy=self.policy
            )
            report_path = root / "report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            selected = load_gate_a_selected_config(report_path, policy=self.policy)
            self.assertEqual(selected.config_id, "strict")
            report["selected_thresholds"]["text_threshold"] = 0.99
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "drifted"):
                load_gate_a_selected_config(report_path, policy=self.policy)

    def test_missing_threshold_result_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results, review = self._artifacts(root)
            rows = results.read_text(encoding="utf-8").splitlines()
            results.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "coverage is not exact"):
                evaluate_craft_gate_a(
                    results_path=results, review_csv_path=review, policy=self.policy
                )

    def test_deadline_override_keeps_current_and_is_preflight_verifiable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy = CraftGateAPolicy(
                schema_version=2,
                sample_frames=2,
                video_ids=("V1", "V2"),
                frames_per_video=None,
                sample_video_counts={"V1": 1, "V2": 1},
                allow_current_fallback_for_gate_b=True,
                evidence_limitations=("single annotator emergency sample",),
                min_region_recall=0.99,
                min_text_frame_recall=0.99,
                configs=self.policy.configs,
            )
            results, review = self._artifacts(root)
            with review.open("r", encoding="utf-8", newline="") as handle:
                review_rows = list(csv.DictReader(handle))
            review_rows[0]["missed_gt_regions__recall_current"] = "1"
            with review.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(review_rows[0]))
                writer.writeheader()
                writer.writerows(review_rows)
            report = evaluate_craft_gate_a(
                results_path=results, review_csv_path=review, policy=policy
            )
            self.assertEqual(report["decision"], "DEADLINE_OVERRIDE_KEEP_CURRENT")
            self.assertEqual(report["selected_config_id"], "recall_current")
            self.assertTrue(report["gate_b_allowed"])
            self.assertFalse(report["metrics"]["recall_current"]["eligible"])
            report_path = root / "report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            selected = load_gate_a_selected_config(report_path, policy=policy)
            self.assertEqual(selected.config_id, "recall_current")


if __name__ == "__main__":
    unittest.main()
