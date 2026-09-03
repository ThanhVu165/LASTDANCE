import unittest

from online.config import OnlineConfig
from online.verification import VideoVerification, rerank_with_verifier
from shared.schemas.online import FrameEvidence, UnifiedQueryPlan, VideoHypothesis


def frame(frame_id: int, score: float) -> FrameEvidence:
    return FrameEvidence(
        keyframe_uid=frame_id + 1,
        video_id="v1",
        frame_id=frame_id,
        pts_time=float(frame_id),
        shot_id=f"s{frame_id}",
        final_score=score,
    )


class FakeVerifier:
    def verify(self, **_kwargs):
        return VideoVerification(
            must_have_score=1.0,
            should_have_score=0.5,
            scene_matches=["scene one"],
            ranked_frame_ids=[20, 10],
            provider="fake",
        )


class OnlineVerificationTests(unittest.TestCase):
    def test_verifier_applies_declared_video_and_frame_weights(self):
        hypothesis = VideoHypothesis(
            video_id="v1",
            video_score=0.6,
            base_video_score=0.6,
            coverage=1.0,
            model_consensus=1.0,
            best_frames=[frame(10, 0.9), frame(20, 0.8)],
        )
        plan = UnifiedQueryPlan(
            raw_query="raw",
            caption_en="caption",
            retrieval_queries=["caption"],
            scenes=["scene one", "scene two"],
            must_have=["must"],
            should_have=["should"],
        )
        result, warnings = rerank_with_verifier(
            [hypothesis],
            plan,
            FakeVerifier(),
            OnlineConfig(),
        )
        self.assertEqual(warnings, [])
        self.assertTrue(result[0].vlm_verified)
        self.assertAlmostEqual(result[0].video_score, 0.7 * 0.6 + 0.2 * 1.0 + 0.1 * 0.5)
        self.assertEqual([item.frame_id for item in result[0].best_frames], [20, 10])
        self.assertAlmostEqual(result[0].best_frames[0].final_score, 0.7 * 0.8 + 0.3)
        self.assertEqual(result[0].missing_scenes, ["scene two"])

    def test_verifier_failure_keeps_retrieval_scores(self):
        class FailingVerifier:
            def verify(self, **_kwargs):
                raise RuntimeError("offline")

        hypothesis = VideoHypothesis(
            video_id="v1",
            video_score=0.6,
            coverage=1.0,
            model_consensus=1.0,
            best_frames=[frame(10, 0.9)],
        )
        plan = UnifiedQueryPlan(
            raw_query="raw",
            caption_en="caption",
            retrieval_queries=["caption"],
        )
        result, warnings = rerank_with_verifier(
            [hypothesis],
            plan,
            FailingVerifier(),
            OnlineConfig(),
        )
        self.assertEqual(result[0].video_score, 0.6)
        self.assertEqual(result[0].best_frames[0].final_score, 0.9)
        self.assertTrue(warnings)

    def test_verifier_omission_does_not_zero_or_reduce_frame(self):
        class PartialVerifier:
            def verify(self, **_kwargs):
                return VideoVerification(
                    must_have_score=1.0,
                    should_have_score=1.0,
                    scene_matches=[],
                    ranked_frame_ids=[20],
                    provider="partial",
                )

        hypothesis = VideoHypothesis(
            video_id="v1",
            video_score=0.6,
            coverage=1.0,
            model_consensus=1.0,
            best_frames=[frame(10, 0.9), frame(20, 0.8)],
        )
        plan = UnifiedQueryPlan(
            raw_query="raw",
            caption_en="caption",
            retrieval_queries=["caption"],
        )
        result, warnings = rerank_with_verifier(
            [hypothesis], plan, PartialVerifier(), OnlineConfig()
        )
        self.assertEqual(warnings, [])
        unmentioned = next(item for item in result[0].best_frames if item.frame_id == 10)
        self.assertEqual(unmentioned.final_score, 0.9)
        self.assertIsNone(unmentioned.vlm_score)


if __name__ == "__main__":
    unittest.main()
