import unittest

from pydantic import ValidationError

from offline.asr_production import AsrWorkerPlan, DEFAULT_ASR_BATCH_IDS


class AsrProductionTests(unittest.TestCase):
    def test_default_nine_batches_are_exhaustive(self):
        digest = "a" * 64
        plan = AsrWorkerPlan(
            catalog_sha256=digest, batch_mapping_sha256=digest,
            assignments=[
                {"worker_id": "worker-1", "enabled": True, "batch_ids": list(DEFAULT_ASR_BATCH_IDS[:5])},
                {"worker_id": "worker-2", "enabled": True, "batch_ids": list(DEFAULT_ASR_BATCH_IDS[5:])},
            ],
        )
        self.assertEqual(len(plan.expected_batch_ids), 9)

    def test_overlap_is_rejected(self):
        digest = "a" * 64
        with self.assertRaises(ValidationError):
            AsrWorkerPlan(
                catalog_sha256=digest, batch_mapping_sha256=digest,
                assignments=[
                    {"worker_id": "w1", "enabled": True, "batch_ids": list(DEFAULT_ASR_BATCH_IDS[:2])},
                    {"worker_id": "w2", "enabled": True, "batch_ids": list(DEFAULT_ASR_BATCH_IDS[1:])},
                ],
            )


if __name__ == "__main__":
    unittest.main()
