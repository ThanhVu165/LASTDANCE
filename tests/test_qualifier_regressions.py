import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from offline.config import DataLayout
from online.config import OnlineLayout, OnlineConfig
from online.frame_references import SourceFrameVerifier
from online.workspace import SubmissionWorkspace
from online.task_heads import build_trake_candidates
from online.vqa import _answers_agree, _parse_result, _verified_pair
from online.answering import FtsVideoAnswerer
from shared.schemas.frame import VerifiedFrameRef
from shared.schemas.online import FrameEvidence, QuerySpec, TaskType, KISCandidate, TrakeCandidate, VideoHypothesis


def frame(number, score=.9):
    return FrameEvidence(keyframe_uid=number + 100, video_id="v1", frame_id=number,
                         pts_time=number / 10, shot_id="same-shot", final_score=score)


def query(task=TaskType.KIS):
    return QuerySpec(query_name="query-p1-1-" + task.value.lower(), source_filename="query-p1-1-" + task.value.lower() + ".txt",
                     raw_query="test query", task_type=task, expected_event_count=2 if task == TaskType.TRAKE else None)


class QualifierRegressionTests(unittest.TestCase):
    def test_source_pts_and_fingerprint_invalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.mp4"
            path.write_bytes(b"first source")
            verifier = SourceFrameVerifier(SimpleNamespace())
            with patch("online.frame_references.source_video", return_value=path), patch(
                    "online.frame_references.probe_frame_timestamps", return_value=[0., .033, .081, .17]):
                ref = verifier.verify("v1", 2)
                self.assertEqual(ref.pts_time, .081)
                self.assertEqual(verifier.validate(ref), .081)
                for bad in [-1, 4, 1.2, True]:
                    with self.assertRaises(ValueError): verifier.verify("v1", bad)
                path.write_bytes(b"changed source with different bytes")
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    verifier.validate(ref)

    def test_missing_source_and_unsafe_id_fail(self):
        verifier = SourceFrameVerifier(SimpleNamespace())
        with patch("online.frame_references.source_video", return_value=None):
            with self.assertRaisesRegex(ValueError, "unavailable"): verifier.verify("v1", 0)
            with self.assertRaisesRegex(ValueError, "unsafe"): verifier.verify("../v1", 0)

    def test_source_frame_outside_catalog_exports_without_fake_uid(self):
        with tempfile.TemporaryDirectory() as directory:
            layout = OnlineLayout(DataLayout(Path(directory)))
            original = frame(10)
            catalog = SimpleNamespace(frames=[original])
            ref = VerifiedFrameRef(video_id="v1", frame_id=11, pts_time=1.13, source_sha256="a" * 64)
            verifier = SimpleNamespace(validate=lambda r: r.pts_time, clear=lambda: None)
            spec = query()
            item = KISCandidate(video_id="v1", frame_id=11, verified_frame=ref, score=.9, evidence=original)
            workspace = SubmissionWorkspace(folder_name="raw", expected_queries=[spec], layout=layout,
                                            catalog=catalog, frame_verifier=verifier,
                                            query_drafts={spec.query_name: [item]})
            self.assertEqual(workspace.csv_bytes(spec.query_name), b"v1,11\n")
            self.assertTrue(workspace.export_zip().zip_path.is_file())
            self.assertEqual(item.evidence.keyframe_uid, original.keyframe_uid)
            with self.assertRaisesRegex(ValueError, "VerifiedFrameRef"):
                workspace.replace_query_draft(spec.query_name, [item.model_copy(update={"verified_frame": None})])

    def test_full_merge_is_idempotent_and_preserves_first_hundred(self):
        with tempfile.TemporaryDirectory() as directory:
            frames = [frame(i) for i in range(101)]
            candidates = [KISCandidate(video_id="v1", frame_id=f.frame_id, score=.9, evidence=f) for f in frames]
            spec = query()
            workspace = SubmissionWorkspace(folder_name="merge", expected_queries=[spec],
                layout=OnlineLayout(DataLayout(Path(directory))), catalog=SimpleNamespace(frames=frames),
                query_drafts={spec.query_name: candidates[:99]})
            workspace.merge_ranked(spec.query_name, candidates[99:])
            workspace.merge_ranked(spec.query_name, candidates[:100])
            self.assertEqual([r.frame_id for r in workspace.query_drafts[spec.query_name]], list(range(100)))

    def test_trake_same_shot_dead_ends_do_not_hide_valid_path(self):
        # More dead ends than the old per-moment cap, and a lower-scored valid
        # predecessor in the same shot. Changing UID/order cannot remove it.
        valid, end = frame(10, .2), frame(20, .8)
        dead = [frame(i, .99) for i in range(30, 80)]
        hypothesis = VideoHypothesis(video_id="v1", video_score=.9, coverage=1., model_consensus=1., best_frames=[valid])
        for inputs in ([*dead, valid], [valid, *reversed(dead)]):
            rows = build_trake_candidates([hypothesis], [SimpleNamespace(evidence=inputs), SimpleNamespace(evidence=[end])],
                                         max_results=100, config=OnlineConfig(trake_beam_width=1))
            self.assertEqual(rows[0].frame_ids, [10, 20])

    def test_lexical_similarity_cannot_validate_semantics(self):
        for left, right in [("red car", "red cat"), ("12 kg", "12 g"), ("2 red cars", "2 blue cars"),
                            ("has helmet", "has no helmet"), ("5", "50")]:
            self.assertFalse(_answers_agree(left, right))
        self.assertTrue(_answers_agree(" ĐỎ ", "đỏ"))

    def test_answer_requires_actual_panel_and_shared_evidence(self):
        frames = [frame(10), frame(20)]
        with self.assertRaises(ValueError):
            _parse_result({"answer": "5", "confidence": .99}, frames, "test")
        first = _parse_result({"answer": "5", "value_type": "number", "confidence": .99, "evidence_panel": "A"}, frames, "test")
        second = _parse_result({"answer": "5", "value_type": "number", "confidence": .99, "evidence_panel": "B"}, frames, "test")
        self.assertTrue(_verified_pair(first, second).requires_review)
        self.assertEqual(first.evidence[0].frame_id, 10)

    def test_numeric_unknown_is_contextual_and_ambiguity_abstains(self):
        answerer = FtsVideoAnswerer(None, "ocr", [], value_type="number")
        self.assertEqual(answerer._extract_unknown_value([SimpleNamespace(text="Năm 2026 có 5 người")], "Có bao nhiêu người?"), "5")
        self.assertEqual(answerer._extract_unknown_value([SimpleNamespace(text="Có 5 người và 7 người")], "Có bao nhiêu người?"), "")

    def test_candidate_rejects_boolean_and_fractional_frame_ids(self):
        for bad in (True, 1.0, "1", -1):
            with self.assertRaises(ValueError):
                KISCandidate(video_id="v1", frame_id=bad, score=.9, evidence=frame(1))
        with self.assertRaises(ValueError):
            TrakeCandidate(video_id="v1", frame_ids=[1, True], pts_times=[.1, .2], score=.9, evidence=[frame(1), frame(2)])

    def test_reference_rejects_nonfinite_pts(self):
        for value in [float("inf"), float("nan")]:
            with self.assertRaises(ValueError):
                VerifiedFrameRef(video_id="v1", frame_id=0, pts_time=value, source_sha256="a" * 64)


if __name__ == "__main__":
    unittest.main()
