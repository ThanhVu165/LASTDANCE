import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from online.artifacts import CatalogFrame, FrameCatalog
from online.config import OnlineConfig
from online.retrieval import FrameRetriever, visual_query_texts
from shared.schemas.online import ArtifactAvailability, ArtifactStatus, UnifiedQueryPlan


class _FailPrimaryEncoder:
    def encode(self, modality, texts):
        if modality == "siglip":
            raise RuntimeError("forced primary failure")
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


class _FakeClipIndex:
    def search(self, query, top_k):
        return np.asarray([1, 2], dtype=np.int64), np.asarray([0.9, 0.4], dtype=np.float32)

    def scores_for(self, query, uids):
        return {int(uid): 1.0 - 0.25 * index for index, uid in enumerate(uids)}


class OnlineRetrievalTests(unittest.TestCase):
    def test_caption_is_always_the_first_faithful_visual_query(self):
        plan = UnifiedQueryPlan(
            raw_query="raw Vietnamese query",
            caption_en="faithful English caption",
            retrieval_queries=["discriminative English query"],
            scenes=["English scene"],
            must_have=["rare detail"],
        )
        self.assertEqual(
            visual_query_texts(plan),
            [
                "faithful English caption",
                "discriminative English query",
                "English scene",
                "rare detail",
            ],
        )

    def test_primary_failure_uses_explicit_clip_rollback(self):
        catalog = FrameCatalog(
            [
                CatalogFrame(1, "v1", 0, 10, 1.0, "s1"),
                CatalogFrame(2, "v2", 0, 20, 2.0, "s2"),
            ],
            sha256="catalog",
        )
        def unavailable(name):
            return ArtifactStatus(
                name=name,
                availability=ArtifactAvailability.UNAVAILABLE,
                detail="absent",
            )
        with tempfile.TemporaryDirectory() as directory:
            registry = SimpleNamespace(
                catalog=catalog,
                visual={"clip": _FakeClipIndex()},
                statuses={"ocr": unavailable("ocr"), "asr": unavailable("asr")},
                layout=SimpleNamespace(
                    ocr=Path(directory) / "ocr.sqlite",
                    asr=Path(directory) / "asr.sqlite",
                ),
            )
            plan = UnifiedQueryPlan(
                raw_query="person opening a door",
                caption_en="person opening a door",
                retrieval_queries=["person opening a door"],
                scenes=["person opening a door"],
                anchor_moment_index=0,
            )
            result = FrameRetriever(registry, _FailPrimaryEncoder(), OnlineConfig()).search(plan)
        self.assertTrue(result.degraded_to_clip)
        self.assertEqual(result.candidate_uids, {1, 2})
        self.assertTrue(any("CLIP degraded rollback" in warning for warning in result.warnings))
        self.assertTrue(all(item.score_clip is not None for item in result.evidence))
        self.assertTrue(all(item.score_siglip is None and item.score_eva is None for item in result.evidence))


if __name__ == "__main__":
    unittest.main()
