import unittest
from types import SimpleNamespace

from online.config import OnlineConfig
from online.ranking import apply_clip_tie_break, rank_videos
from online.engine import merge_kis_anchor_frames
from online.task_heads import build_kis_candidates, build_qa_candidates, build_trake_candidates
from online.retrieval import RetrievalResult
from shared.schemas.online import FrameEvidence, UnifiedQueryPlan, VideoHypothesis


def frame(video: str, frame_id: int, pts_time: float, score: float, shot: str) -> FrameEvidence:
    return FrameEvidence(
        keyframe_uid=frame_id + (10000 if video == "v2" else 20000 if video == "v3" else 1),
        video_id=video,
        frame_id=frame_id,
        pts_time=pts_time,
        shot_id=shot,
        final_score=score,
        model_consensus=1.0,
    )


class OnlineTaskHeadTests(unittest.TestCase):
    def test_clip_only_reorders_videos_inside_tie_margin(self):
        low_clip = frame("v1", 1, 1.0, 0.9, "s1").model_copy(update={"score_clip": 0.2})
        high_clip = frame("v2", 2, 1.0, 0.9, "s1").model_copy(update={"score_clip": 0.8})
        hypotheses = [
            VideoHypothesis(video_id="v1", video_score=0.90, coverage=1, model_consensus=1, best_frames=[low_clip]),
            VideoHypothesis(video_id="v2", video_score=0.89, coverage=1, model_consensus=1, best_frames=[high_clip]),
        ]
        result = apply_clip_tie_break(hypotheses, margin=0.02)
        self.assertEqual([item.video_id for item in result], ["v2", "v1"])
        self.assertEqual([item.video_score for item in result], [0.89, 0.90])

    def test_kis_top_five_uses_baseline_interleave(self):
        hypotheses = []
        for rank, video in enumerate(("v1", "v2", "v3")):
            frames = [frame(video, rank * 100 + index, float(index), 1.0 - index * 0.05, f"s{index}") for index in range(4)]
            hypotheses.append(
                VideoHypothesis(
                    video_id=video,
                    video_score=1.0 - rank * 0.1,
                    coverage=1.0,
                    model_consensus=1.0,
                    best_frames=frames,
                )
            )
        result = build_kis_candidates(hypotheses, max_results=5, config=OnlineConfig())
        self.assertEqual([item.video_id for item in result], ["v1", "v1", "v2", "v1", "v3"])

    def test_kis_portfolio_reserves_thirty_rows_for_primary_video(self):
        hypotheses = []
        for rank, video in enumerate(("v1", "v2", "v3")):
            frames = [
                frame(video, rank * 1000 + index, float(index), 1.0 - index * 0.001, f"s{index}")
                for index in range(40)
            ]
            hypotheses.append(
                VideoHypothesis(
                    video_id=video,
                    video_score=1.0 - rank * 0.1,
                    coverage=1.0,
                    model_consensus=1.0,
                    best_frames=frames,
                )
            )
        result = build_kis_candidates(hypotheses, max_results=100, config=OnlineConfig())
        self.assertEqual(len(result), 100)
        self.assertGreaterEqual(sum(item.video_id == "v1" for item in result), 30)
        self.assertLessEqual(sum(item.video_id == "v1" for item in result), 40)

    def test_kis_portfolio_uses_weighted_round_robin_after_seed(self):
        hypotheses = []
        for rank in range(12):
            video = f"v{rank + 1}"
            frames = [
                frame(video, rank * 1000 + index, float(index), 1.0 - index * 0.001, f"s{index}")
                for index in range(40)
            ]
            hypotheses.append(
                VideoHypothesis(
                    video_id=video,
                    video_score=1.0 - rank * 0.04,
                    coverage=1.0,
                    model_consensus=1.0,
                    best_frames=frames,
                )
            )
        result = build_kis_candidates(hypotheses, max_results=100, config=OnlineConfig())
        self.assertEqual([item.video_id for item in result[:5]], ["v1", "v1", "v2", "v1", "v3"])
        self.assertLessEqual(next(index for index, item in enumerate(result, 1) if item.video_id == "v12"), 16)
        for video in ("v1", "v2", "v3", "v4", "v5"):
            self.assertGreaterEqual(sum(item.video_id == video for item in result), 2)

    def test_kis_anchor_merge_is_boost_only_and_keeps_new_shots(self):
        target = frame("v1", 6471, 258.84, 0.95, "s16")
        base_other = frame("v1", 7000, 280.0, 0.80, "s17")
        weak_anchor = target.model_copy(update={"final_score": 0.20})
        new_anchor = frame("v1", 7100, 284.0, 0.75, "s18")
        merged = merge_kis_anchor_frames(
            [target, base_other],
            [weak_anchor, new_anchor],
            bonus=0.15,
            max_per_shot=3,
        )
        merged_target = next(item for item in merged if item.frame_id == 6471)
        self.assertGreaterEqual(merged_target.final_score, target.final_score)
        self.assertIn(7100, [item.frame_id for item in merged])

    def test_kis_keeps_nearby_frames_but_seeds_distinct_shots(self):
        frames = [
            frame("v1", 6480, 259.20, 1.00, "s16"),
            frame("v1", 6471, 258.84, 0.99, "s16"),
            frame("v1", 6430, 257.20, 0.98, "s14"),
            frame("v1", 6462, 258.48, 0.97, "s15"),
        ]
        hypothesis = VideoHypothesis(
            video_id="v1",
            video_score=1.0,
            coverage=1.0,
            model_consensus=1.0,
            best_frames=frames,
        )
        result = build_kis_candidates([hypothesis], max_results=4, config=OnlineConfig())
        self.assertEqual([item.frame_id for item in result], [6480, 6430, 6462, 6471])
        self.assertEqual(len({item.evidence.shot_id for item in result[:3]}), 3)

    def test_kis_secondary_video_keeps_same_shot_score_order(self):
        primary = VideoHypothesis(
            video_id="v1",
            video_score=1.0,
            coverage=1.0,
            model_consensus=1.0,
            best_frames=[frame("v1", index, float(index), 1.0 - index * 0.01, f"s{index}") for index in range(6)],
        )
        secondary = VideoHypothesis(
            video_id="v2",
            video_score=0.9,
            coverage=1.0,
            model_consensus=1.0,
            best_frames=[
                frame("v2", 6480, 259.20, 1.00, "s16"),
                frame("v2", 6471, 258.84, 0.99, "s16"),
                frame("v2", 6430, 257.20, 0.98, "s14"),
            ],
        )
        result = build_kis_candidates(
            [primary, secondary], max_results=9, config=OnlineConfig()
        )
        secondary_frames = [item.frame_id for item in result if item.video_id == "v2"]
        self.assertEqual(secondary_frames[:2], [6480, 6471])

    def test_video_ranking_reports_target_labels_not_locator_or_global_labels(self):
        first = frame("v1", 1, 1.0, 0.9, "s1")
        second = frame("v2", 2, 1.0, 0.8, "s1")
        registry = SimpleNamespace(
            catalog=SimpleNamespace(
                by_uid={
                    first.keyframe_uid: SimpleNamespace(video_id="v1", shot_id="s1"),
                    second.keyframe_uid: SimpleNamespace(video_id="v2", shot_id="s1"),
                }
            )
        )
        plan = UnifiedQueryPlan(
            raw_query="raw",
            caption_en="global",
            retrieval_queries=["global"],
            scenes=["map scene", "door scene"],
            must_have=["door scene"],
        )
        result = RetrievalResult(
            evidence=[first, second],
            per_query_scores={
                "global": {first.keyframe_uid: 0.9, second.keyframe_uid: 0.8},
                "map scene": {first.keyframe_uid: 0.9, second.keyframe_uid: 0.1},
                "door scene": {first.keyframe_uid: 0.8, second.keyframe_uid: 0.2},
            },
            candidate_uids={first.keyframe_uid, second.keyframe_uid},
        )
        ranked = rank_videos(result, plan, registry, OnlineConfig())
        self.assertEqual(ranked[0].video_id, "v1")
        self.assertEqual(ranked[0].matched_scenes, ["door scene"])
        self.assertNotIn("global", ranked[0].matched_scenes)

    def test_qa_attempts_top_three_below_locator_threshold_without_uncertain_rows(self):
        class Answerer:
            def __init__(self):
                self.calls = []

            def answer(self, *, video_id, frames, question):
                self.calls.append(video_id)
                return "42", 0.9, []

        hypotheses = [
            VideoHypothesis(
                video_id=f"v{index}",
                video_score=0.2,
                coverage=0.2,
                model_consensus=0.2,
                best_frames=[frame(f"v{index}", index, float(index), 0.2, f"s{index}")],
            )
            for index in range(1, 5)
        ]
        plan = UnifiedQueryPlan(
            raw_query="Con số là bao nhiêu?",
            caption_en="What number is visible?",
            retrieval_queries=["visible number"],
            scenes=["visible number"],
            question="Con số là bao nhiêu?",
            answer_source="visible_text",
        )
        answerer = Answerer()
        candidates, _warnings = build_qa_candidates(
            hypotheses,
            plan,
            answerer=answerer,
            max_results=100,
            config=OnlineConfig(),
        )
        self.assertEqual(answerer.calls, ["v1", "v2", "v3"])
        self.assertTrue(candidates)
        self.assertTrue(all(item.answer == "42" for item in candidates))
        self.assertTrue(all(item.requires_review for item in candidates))

    def test_trake_beam_enforces_same_video_and_increasing_time(self):
        first = RetrievalResult(
            evidence=[frame("v1", 1, 1.0, 0.9, "s1"), frame("v2", 2, 1.0, 0.8, "s1")],
            per_query_scores={},
            candidate_uids=set(),
        )
        second = RetrievalResult(
            evidence=[frame("v1", 3, 4.0, 0.9, "s2"), frame("v1", 4, 0.5, 1.0, "s3")],
            per_query_scores={},
            candidate_uids=set(),
        )
        hypothesis = VideoHypothesis(
            video_id="v1",
            video_score=1.0,
            coverage=1.0,
            model_consensus=1.0,
            best_frames=first.evidence + second.evidence,
        )
        result = build_trake_candidates(
            [hypothesis], [first, second], max_results=10, config=OnlineConfig()
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].frame_ids, [1, 3])
        self.assertLess(result[0].pts_times[0], result[0].pts_times[1])


if __name__ == "__main__":
    unittest.main()
