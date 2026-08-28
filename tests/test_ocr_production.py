import unittest

from pydantic import ValidationError

from offline.ocr_production import (
    OcrLayer,
    OcrLayerShardManifest,
    OcrWorkItemKind,
    OcrWorkerAssignment,
    OcrWorkerPlan,
    ocr_hf_archive_root,
    ocr_hf_snapshot_root,
    validate_ocr_hf_path,
    validate_ocr_snapshot_hf_path,
)


SHA = "a" * 64


def _plan(assignments):
    return OcrWorkerPlan(
        catalog_sha256=SHA,
        batch_mapping_sha256=SHA,
        expected_batch_ids=[f"batch-{index:02d}" for index in range(1, 10)],
        assignments=assignments,
    )


class OcrProductionContractTests(unittest.TestCase):
    def test_four_workers_must_be_disjoint_and_exhaustive(self):
        plan = _plan(
            [
                OcrWorkerAssignment(worker_id="ocr-01", enabled=True, batch_ids=["batch-01", "batch-05", "batch-09"]),
                OcrWorkerAssignment(worker_id="ocr-02", enabled=True, batch_ids=["batch-02", "batch-06"]),
                OcrWorkerAssignment(worker_id="ocr-03", enabled=True, batch_ids=["batch-03", "batch-07"]),
                OcrWorkerAssignment(worker_id="ocr-04", enabled=True, batch_ids=["batch-04", "batch-08"]),
            ]
        )
        self.assertEqual(len(plan.assignments), 4)

    def test_overlap_or_missing_batch_fails_closed(self):
        with self.assertRaisesRegex(ValidationError, "overlap"):
            _plan(
                [
                    OcrWorkerAssignment(worker_id="ocr-01", enabled=True, batch_ids=["batch-01", "batch-02"]),
                    OcrWorkerAssignment(worker_id="ocr-02", enabled=True, batch_ids=["batch-02", "batch-03", "batch-04", "batch-05", "batch-06", "batch-07", "batch-08", "batch-09"]),
                ]
            )
        with self.assertRaisesRegex(ValidationError, "not exhaustive"):
            _plan(
                [OcrWorkerAssignment(worker_id="ocr-01", enabled=True, batch_ids=["batch-01"])]
            )

    def test_layer_manifest_gate_is_derived_from_counts(self):
        manifest = OcrLayerShardManifest(
            batch_id="batch-01",
            worker_id="ocr-01",
            layer=OcrLayer.EASYOCR,
            item_kind=OcrWorkItemKind.REGION,
            catalog_sha256=SHA,
            config_sha256=SHA,
            assigned_uid_sha256=SHA,
            input_artifact_path="ocr/layers/craft/batch-01.jsonl",
            input_artifact_sha256=SHA,
            output_jsonl_path="ocr/layers/easyocr/batch-01.jsonl",
            output_jsonl_sha256=SHA,
            expected_keyframes=100,
            processed_keyframes=100,
            expected_items=850,
            processed_items=850,
            error_items=0,
            duplicate_items=0,
            missing_keyframes=0,
            foreign_keyframes=0,
            completion_gate_passed=True,
        )
        self.assertTrue(manifest.completion_gate_passed)
        with self.assertRaisesRegex(ValidationError, "does not match"):
            OcrLayerShardManifest(**{**manifest.model_dump(), "error_items": 1})

    def test_hf_paths_are_confined_to_shared_ocr_namespace(self):
        self.assertEqual(ocr_hf_archive_root("batch-01"), "ocr/archives/batch-01")
        path = "ocr/archives/batch-01/layers-vintern.tar.gz"
        self.assertEqual(validate_ocr_hf_path(path, batch_id="batch-01"), path)
        with self.assertRaisesRegex(ValueError, "must stay under"):
            validate_ocr_hf_path("clip/archives/batch-01/a.tar.gz", batch_id="batch-01")
        with self.assertRaisesRegex(ValueError, "identifier"):
            ocr_hf_archive_root("../clip")

        snapshot_id = "ocr-snapshot-20260828T120000Z-abcdef123456"
        snapshot_path = f"ocr/snapshots/{snapshot_id}/ocr.sqlite"
        self.assertEqual(ocr_hf_snapshot_root(snapshot_id), f"ocr/snapshots/{snapshot_id}")
        self.assertEqual(
            validate_ocr_snapshot_hf_path(snapshot_path, snapshot_id=snapshot_id),
            snapshot_path,
        )
        with self.assertRaisesRegex(ValueError, "must stay under"):
            validate_ocr_snapshot_hf_path(
                "ocr/archives/batch-01/a.tar.gz", snapshot_id=snapshot_id
            )


if __name__ == "__main__":
    unittest.main()
