import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from scripts.freeze_qualifier_config import freeze
from scripts.prepare_qualifier_labels import prepare
from shared.evaluation import EvaluationCase, Prediction, diagnostic_metrics
from shared.schemas.online import QuerySpec, TaskType


class QualifierWorkflowTests(unittest.TestCase):
    def test_freeze_rejects_heldout_and_preserves_immutable_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = {"split": "held_out", "config_sha256": "a" * 64, "labels_sha256": "b" * 64,
                      "mean_final_score": .7}
            source = root / "report.json"; source.write_text(json.dumps(report))
            with self.assertRaises(ValueError): freeze([source], root / "lock.json")
            report["split"] = "development"; source.write_text(json.dumps(report))
            lock = freeze([source], root / "lock.json")
            self.assertEqual(lock["selected_on"], "development")
            with self.assertRaises(FileExistsError): freeze([source], root / "lock.json")

    def test_annotation_assignment_is_video_disjoint_and_cannot_score_as_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); catalog = root / "frames.csv"
            catalog.write_text("video_id,frame_id,pts_time,shot_id,local_idx\n" +
                               "".join(f"v{i},10,.33,s0,0\n" for i in range(65)), encoding="utf-8")
            path = prepare(catalog, root / "labels")
            rows = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(rows), 60)
            self.assertFalse({r["video_id"] for r in rows[:30]} & {r["video_id"] for r in rows[30:]})
            with self.assertRaises(ValueError): EvaluationCase.model_validate(rows[0])
            with self.assertRaises(FileExistsError): prepare(catalog, root / "labels")

    def test_diagnostics_separate_catalog_support_and_prediction_alignment(self):
        name = "query-p1-1-kis"
        case = EvaluationCase(query=QuerySpec(query_name=name, source_filename=name + ".txt", task_type=TaskType.KIS,
                                             raw_query="target"), split="development", video_id="v1",
                              intervals=[{"start": 11, "end": 12}], verified_by="test reviewer")
        predictions = {name: [Prediction(video_id="v1", frame_ids=[10])]}
        result = diagnostic_metrics([case], predictions, split="development",
                                    catalog_frames=[SimpleNamespace(video_id="v1", frame_id=10)])
        self.assertEqual(result["catalog_event_support_fraction"], 0)
        self.assertEqual(result["video_recall"]["R@1"], 1)
        self.assertEqual(result["queries"][0]["nearest_prediction_distance_frames"], [1])
        self.assertIsNone(result["latency_p50_ms"])


if __name__ == "__main__": unittest.main()
