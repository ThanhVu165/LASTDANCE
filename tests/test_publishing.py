import unittest

from offline.publishing import (
    REQUIRED_VISUAL_INDEXES,
    VectorHealth,
    assess_publishing_readiness,
)
from offline.visual_embeddings import SUPPORTED_MODALITIES


class PublishingCriteriaTests(unittest.TestCase):
    def test_embedding_and_publishing_modality_contracts_match(self):
        self.assertEqual(REQUIRED_VISUAL_INDEXES, SUPPORTED_MODALITIES)

    def test_video_is_complete_only_when_all_criteria_pass(self):
        healthy = VectorHealth(finite=True, normalized=True)
        report = assess_publishing_readiness(
            video_id="L01_V001",
            frame_uids={1, 2, 3},
            index_uids={
                "clip": {1, 2, 3},
                "siglip": {1, 2, 3},
                "eva_clip": {1, 2, 3},
            },
            vector_health={"clip": healthy, "siglip": healthy, "eva_clip": healthy},
            mapping_verified=True,
            checkpoint_resume_verified=True,
        )
        self.assertTrue(report.complete)
        self.assertTrue(report.as_state()["complete"])

    def test_missing_index_id_and_bad_norm_fail_closed(self):
        report = assess_publishing_readiness(
            video_id="L01_V001",
            frame_uids={1, 2, 3},
            index_uids={
                "clip": {1, 2, 3},
                "siglip": {1, 2},
                "eva_clip": {1, 2, 3},
            },
            vector_health={
                "clip": VectorHealth(True, True),
                "siglip": VectorHealth(True, True),
                "eva_clip": VectorHealth(True, False),
            },
            mapping_verified=True,
            checkpoint_resume_verified=True,
        )
        self.assertFalse(report.complete)
        self.assertEqual(report.missing_ids["siglip"], frozenset({3}))

    def test_empty_video_cannot_be_published(self):
        healthy = VectorHealth(True, True)
        report = assess_publishing_readiness(
            video_id="empty",
            frame_uids=set(),
            index_uids={"clip": set(), "siglip": set(), "eva_clip": set()},
            vector_health={"clip": healthy, "siglip": healthy, "eva_clip": healthy},
            mapping_verified=True,
            checkpoint_resume_verified=True,
        )
        self.assertFalse(report.complete)


if __name__ == "__main__":
    unittest.main()
