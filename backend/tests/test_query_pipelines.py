import unittest
from unittest.mock import patch

from app.evaluation.official_metric import final_score, trake_r_score
from app.rerank.contest_ranking import (
    KIS_VIDEO_CAPS,
    TRAKE_VIDEO_CAPS,
    cutoff_aware_rank,
)
from app.rerank.visual_reranker import (
    _parse_score_and_best,
    _parse_video_comparison,
    _with_temporal_context,
    rerank_kis_candidates,
)
from app.rerank.exact_frame_refinement import _requires_exact_refinement
from app.rerank.storyboard_alignment import apply_storyboard_alignment
from app.pipelines.kis_pipeline import _merge_clip_hits
from app.pipelines.qa_pipeline import (
    _normalize_answer,
    _parse_qa_judgment,
    run_qa_query,
)
from app.pipelines.trake_pipeline import (
    MomentCandidate,
    _k_best_monotonic_alignments,
    run_trake_query,
)


class KisPipelineTests(unittest.TestCase):
    def test_long_query_video_consensus_rewards_cross_scene_evidence(self):
        batches = [
            [
                {"video_id": "single", "local_idx": 1, "frame_id": 10, "score": 0.9},
                {"video_id": "multi", "local_idx": 1, "frame_id": 10, "score": 0.6},
            ],
            [{"video_id": "multi", "local_idx": 2, "frame_id": 20, "score": 0.6}],
            [{"video_id": "multi", "local_idx": 3, "frame_id": 30, "score": 0.6}],
            [{"video_id": "multi", "local_idx": 4, "frame_id": 40, "score": 0.6}],
        ]
        with patch("app.pipelines.kis_pipeline.search_text_batch", return_value=batches):
            candidates = _merge_clip_hits(["scene 1", "scene 2", "scene 3", "scene 4"], 20)

        best_by_video = {}
        for candidate in candidates:
            best_by_video[candidate["video_id"]] = max(
                best_by_video.get(candidate["video_id"], 0.0), candidate["score"]
            )
        self.assertGreater(best_by_video["multi"], best_by_video["single"])

    def test_translated_alternatives_do_not_cast_extra_consensus_votes(self):
        batches = [
            [{"video_id": "complete", "local_idx": 1, "frame_id": 10, "score": 0.8}],
            [{"video_id": "complete", "local_idx": 2, "frame_id": 20, "score": 0.8}],
            [{"video_id": "generic", "local_idx": 1, "frame_id": 10, "score": 0.9}],
            [{"video_id": "generic", "local_idx": 2, "frame_id": 20, "score": 0.9}],
            [{"video_id": "generic", "local_idx": 3, "frame_id": 30, "score": 0.9}],
        ]
        with patch("app.pipelines.kis_pipeline.search_text_batch", return_value=batches):
            candidates = _merge_clip_hits(
                ["scene one", "scene two", "translation 1", "translation 2", "translation 3"],
                20,
                consensus_weight=1.0,
                consensus_expansion_count=2,
            )

        best_by_video = {}
        for candidate in candidates:
            best_by_video[candidate["video_id"]] = max(
                best_by_video.get(candidate["video_id"], 0.0), candidate["score"]
            )
        self.assertGreater(best_by_video["complete"], best_by_video["generic"])

    def test_merge_retains_per_scene_evidence_without_auxiliary_votes(self):
        batches = [
            [{"video_id": "target", "local_idx": 10, "frame_id": 100, "score": 0.9}],
            [{"video_id": "target", "local_idx": 20, "frame_id": 200, "score": 0.8}],
            [{"video_id": "generic", "local_idx": 1, "frame_id": 10, "score": 0.99}],
        ]
        with patch("app.pipelines.kis_pipeline.search_text_batch", return_value=batches):
            candidates = _merge_clip_hits(
                ["scene one", "scene two", "full-query paraphrase"],
                20,
                prompt_scene_indices=[0, 1, None],
            )

        target_rows = [row for row in candidates if row["video_id"] == "target"]
        generic = next(row for row in candidates if row["video_id"] == "generic")
        self.assertEqual(target_rows[0]["scene_scores"], {0: 1.0})
        self.assertEqual(target_rows[1]["scene_scores"], {1: 1.0})
        self.assertEqual(generic["scene_scores"], {})

    def test_translation_of_same_scene_does_not_double_consensus_vote(self):
        batches = [
            [{"video_id": "target", "local_idx": 1, "frame_id": 10, "score": 0.8}],
            [{"video_id": "target", "local_idx": 2, "frame_id": 20, "score": 0.8}],
            [{"video_id": "target", "local_idx": 1, "frame_id": 10, "score": 0.9}],
        ]
        with patch("app.pipelines.kis_pipeline.search_text_batch", return_value=batches):
            candidates = _merge_clip_hits(
                ["scene one", "scene two", "scene one in English"],
                20,
                consensus_weight=1.0,
                prompt_scene_indices=[0, 1, 0],
            )

        self.assertTrue(candidates)
        self.assertTrue(all(row["query_coverage"] == 1.0 for row in candidates))

    def test_short_query_temporal_context_keeps_chronological_triplet(self):
        rows = [{"video_id": "V1", "local_idx": 5, "frame_id": 50, "score": 0.9}]
        records = [
            type("Record", (), {"local_idx": 4, "frame_id": 40, "path": "4.jpg"})(),
            type("Record", (), {"local_idx": 5, "frame_id": 50, "path": "5.jpg"})(),
            type("Record", (), {"local_idx": 6, "frame_id": 60, "path": "6.jpg"})(),
        ]
        with patch("app.rerank.visual_reranker.temporal_triplet", return_value=records):
            contextual = _with_temporal_context(rows, limit=4)

        self.assertEqual([row["local_idx"] for row in contextual], [4, 5, 6])
        self.assertTrue(contextual[0]["is_temporal_context"])
        self.assertFalse(contextual[1]["is_temporal_context"])

    def test_storyboard_alignment_rewards_complete_ordered_video(self):
        candidates = []
        for video_id, locals_by_scene, strength in (
            ("complete", [10, 20, 30], 0.75),
            ("reversed", [30, 20, 10], 0.95),
        ):
            for scene_index, local_idx in enumerate(locals_by_scene):
                candidates.append(
                    {
                        "video_id": video_id,
                        "local_idx": local_idx,
                        "frame_id": local_idx * 10,
                        "score": 0.5,
                        "scene_scores": {scene_index: strength},
                    }
                )

        apply_storyboard_alignment(
            candidates,
            scene_count=3,
            temporal_edges=[(0, 1), (1, 2)],
        )

        complete = [row for row in candidates if row["video_id"] == "complete"]
        reversed_rows = [row for row in candidates if row["video_id"] == "reversed"]
        self.assertEqual(complete[0]["query_coverage"], 1.0)
        self.assertLess(reversed_rows[0]["query_coverage"], 1.0)
        self.assertGreater(complete[0]["score"], reversed_rows[0]["score"])
        self.assertEqual(complete[0]["storyboard_local_idxs"], [10, 20, 30])


class ContestRankingTests(unittest.TestCase):
    @staticmethod
    def _candidates() -> list[dict]:
        rows = []
        score = 1.0
        for video_index in range(30):
            for local_idx in range(1, 9):
                rows.append(
                    {
                        "video_id": f"V{video_index:02d}",
                        "local_idx": local_idx,
                        "score": score,
                    }
                )
                score -= 0.001
        return rows

    def test_kis_portfolio_uses_official_cutoff_quotas(self):
        ranked = cutoff_aware_rank(
            self._candidates(), top_k=100, video_caps=KIS_VIDEO_CAPS
        )

        self.assertEqual(ranked[0]["score"], 1.0)
        for cutoff, cap in KIS_VIDEO_CAPS.items():
            counts = {}
            for row in ranked[:cutoff]:
                counts[row["video_id"]] = counts.get(row["video_id"], 0) + 1
            self.assertLessEqual(max(counts.values()), cap)

    def test_trake_top_five_covers_video_uncertainty(self):
        ranked = cutoff_aware_rank(
            self._candidates(), top_k=20, video_caps=TRAKE_VIDEO_CAPS
        )
        self.assertGreaterEqual(len({row["video_id"] for row in ranked[:5]}), 3)

    def test_visual_judgment_parser_is_strict_and_bounded(self):
        self.assertEqual(_parse_score_and_best("SCORE=87; BEST=2", 3), (0.87, 1))
        self.assertEqual(_parse_score_and_best("SCORE=100; BEST=9", 3), (1.0, None))
        self.assertIsNone(_parse_score_and_best("looks relevant", 3))

    def test_cross_video_judgment_parser_is_strict_and_bounded(self):
        self.assertEqual(
            _parse_video_comparison(
                "SCORE=92; BESTVIDEO=3; BESTPANEL=2",
                [4, 4, 3, 4, 4],
            ),
            (0.92, 2, 1),
        )
        self.assertIsNone(
            _parse_video_comparison(
                "SCORE=92; BESTVIDEO=3; BESTPANEL=4",
                [4, 4, 3],
            )
        )
        self.assertEqual(
            _parse_video_comparison(
                "BESTVIDEO=V3;BESTPANEL=V3-P2",
                [4, 4, 3],
            ),
            (0.5, 2, 1),
        )

    def test_cross_video_checklist_overrides_generic_model_shortcut(self):
        parsed = _parse_video_comparison(
            "CHECKS=V1:1000,V2:1111,V3:1100; "
            "SCORE=90; BESTVIDEO=1; BESTPANEL=2",
            [4, 4, 4],
            criterion_count=4,
        )

        self.assertEqual(parsed, (0.97, 1, 1))

    def test_tournament_winner_is_promoted_above_retrieval_leader(self):
        candidates = [
            {
                "video_id": "generic",
                "local_idx": 1,
                "frame_id": 10,
                "score": 0.95,
                "query_coverage": 0.5,
            },
            {
                "video_id": "complete",
                "local_idx": 1,
                "frame_id": 20,
                "score": 0.75,
                "query_coverage": 1.0,
            },
        ]
        with patch(
            "app.rerank.visual_reranker.VLM_RERANK_TOP_VIDEOS", 2
        ), patch(
            "app.rerank.visual_reranker.VLM_RERANK_FRAMES_PER_VIDEO", 1
        ), patch(
            "app.rerank.visual_reranker._representative_rows",
            side_effect=lambda rows, *_args, **_kwargs: rows,
        ), patch(
            "app.rerank.visual_reranker._cached_video_comparison",
            return_value=(0.9, 1, 0),
        ):
            reranked = rerank_kis_candidates(
                "Cảnh một có sản phẩm. Cảnh hai có hành động sau đó. Cảnh ba ở địa điểm.",
                candidates,
            )

        best = max(reranked, key=lambda row: row["score"])
        self.assertEqual(best["video_id"], "complete")

    def test_exact_frame_refinement_only_runs_for_semantic_boundaries(self):
        self.assertTrue(
            _requires_exact_refinement(
                "khoảnh khắc đầu tiên bàn chân rời hoàn toàn khỏi mặt đất"
            )
        )
        self.assertFalse(
            _requires_exact_refinement("một con lừa và con non đứng trong chuồng")
        )

    def test_official_final_score_matches_organizer_example(self):
        r_scores = [0.5, 0.2, 0.8] + [0.6] * 97
        metric = final_score(r_scores)

        self.assertEqual(metric["r_at"], {1: 0.5, 5: 0.8, 20: 0.8, 50: 0.8, 100: 0.8})
        self.assertAlmostEqual(metric["final_score"], 0.74)

    def test_official_trake_partial_credit_matches_organizer_example(self):
        score = trake_r_score(
            {"video_id": "L10_V010", "frame_ids": [101, 156, 203, 251]},
            {
                "video_id": "L10_V010",
                "frame_intervals": [[95, 105], [145, 155], [195, 205], [245, 255]],
            },
        )
        self.assertEqual(score, 0.75)


class QaPipelineTests(unittest.TestCase):
    def test_answer_format_is_applied_generically(self):
        self.assertEqual(
            _normalize_answer("  Walt Disney  ", uppercase=True, no_spaces=True),
            "WALTDISNEY",
        )

    def test_temporal_qa_judgment_parser_rejects_invalid_panel(self):
        self.assertEqual(
            _parse_qa_judgment(
                "MATCH=91; BEST=3; ANSWER=hai người",
                6,
            ),
            (0.91, 2, "hai người"),
        )
        self.assertIsNone(
            _parse_qa_judgment("MATCH=91; BEST=8; ANSWER=hai người", 6)
        )

    def test_qa_retrieves_and_answers_every_requested_candidate(self):
        candidates = [
            {
                "video_id": f"V{index:03d}",
                "frame_id": index * 10,
                "local_idx": index,
                "score": 1.0 - index / 1000,
            }
            for index in range(1, 101)
        ]
        answered_questions = []

        def fake_answerer(_path, question):
            answered_questions.append(question)
            return "màu đỏ"

        with patch("app.pipelines.qa_pipeline.run_kis_query", return_value=candidates) as retrieval:
            results = run_qa_query(
                "Một chiếc xe chạy trên đường. Câu hỏi: Xe có màu gì?",
                top_k=100,
                answerer=fake_answerer,
            )

        retrieval.assert_called_once_with(
            "Một chiếc xe chạy trên đường",
            top_k=100,
            refine_exact=False,
        )
        self.assertEqual(len(results), 100)
        self.assertEqual(len(answered_questions), 100)
        self.assertTrue(
            all(
                question == "Xe có màu gì?"
                for question in answered_questions
            )
        )
        self.assertTrue(all(row["answer"] == "màu đỏ" for row in results))


class TrakePipelineTests(unittest.TestCase):
    def test_alignment_enforces_strict_temporal_order(self):
        alignments = _k_best_monotonic_alignments(
            [
                [MomentCandidate(5, 50, 0.9), MomentCandidate(15, 150, 0.8)],
                [MomentCandidate(10, 100, 0.9), MomentCandidate(20, 200, 0.8)],
                [MomentCandidate(25, 250, 0.9)],
            ]
        )

        self.assertGreater(len(alignments), 0)
        for alignment in alignments:
            local_indices = [candidate.local_idx for candidate in alignment.candidates]
            self.assertEqual(local_indices, sorted(local_indices))
            self.assertEqual(len(local_indices), len(set(local_indices)))

    def test_alignment_rejects_implausibly_distant_moments(self):
        alignments = _k_best_monotonic_alignments(
            [
                [MomentCandidate(1, 10, 0.9, pts_time=10.0)],
                [MomentCandidate(2, 20, 0.9, pts_time=420.0)],
            ],
            max_gap_seconds=300.0,
            gap_penalty_weight=0.15,
        )

        self.assertEqual(alignments, [])

    def test_alignment_compactness_penalty_prefers_nearby_sequence(self):
        alignments = _k_best_monotonic_alignments(
            [
                [MomentCandidate(1, 10, 0.9, pts_time=10.0)],
                [
                    MomentCandidate(2, 20, 0.8, pts_time=20.0),
                    MomentCandidate(3, 30, 0.9, pts_time=250.0),
                ],
            ],
            max_gap_seconds=300.0,
            gap_penalty_weight=0.15,
        )

        self.assertGreater(len(alignments), 1)
        self.assertEqual(alignments[0].candidates[-1].local_idx, 2)

    def test_trake_returns_100_ranked_sequences_across_video_hypotheses(self):
        moments = ["bắt đầu chạy", "nhảy qua xà", "tiếp đất"]
        hits_by_moment = []
        for moment_index in range(len(moments)):
            hits = []
            for video_index in range(25):
                video_id = f"V{video_index:03d}"
                base = moment_index * 100
                for offset in range(5):
                    local_idx = base + offset + 1
                    hits.append(
                        {
                            "video_id": video_id,
                            "local_idx": local_idx,
                            "frame_id": local_idx * 10,
                            "score": 0.95 - video_index / 1000 - offset / 10000,
                        }
                    )
            hits_by_moment.append(hits)

        with patch(
            "app.pipelines.trake_pipeline.split_trake_moments",
            return_value=moments,
        ), patch(
            "app.pipelines.trake_pipeline._moment_hits",
            side_effect=hits_by_moment,
        ):
            parsed_moments, results = run_trake_query("complete organizer query", top_k=100)

        self.assertEqual(parsed_moments, moments)
        self.assertEqual(len(results), 100)
        self.assertTrue(all(len(row["frame_ids"]) == len(moments) for row in results))
        self.assertTrue(
            all(row["local_idxs"] == sorted(row["local_idxs"]) for row in results)
        )

    def test_trake_requires_two_ordered_moments(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            run_trake_query("một người đang chạy", top_k=100)


if __name__ == "__main__":
    unittest.main()
