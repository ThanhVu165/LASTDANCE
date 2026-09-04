import unittest

from shared.evaluation import EvaluationCase, Prediction, score_case, validate_suite


def case(task="KIS", **updates):
    count = 4 if task == "TRAKE" else 1
    row = dict(query=dict(query_name=f"q-{task.lower()}", source_filename=f"q-{task.lower()}.txt",
                         task_type=task, raw_query="reviewed query", expected_event_count=count if task == "TRAKE" else None),
               split="development", video_id="video", intervals=[dict(start=100*i, end=100*i+9) for i in range(1,count+1)],
               accepted_answers=["5", "Năm"] if task == "QA" else [], verified_by="human-reviewer")
    row.update(updates)
    return EvaluationCase.model_validate(row)


class QualifierScoringTests(unittest.TestCase):
    def test_kis_inclusive_interval_and_rank_weight(self):
        rows = [Prediction(video_id="other",frame_ids=[100])]*0
        rows.extend(Prediction(video_id="other", frame_ids=[i]) for i in range(5))
        rows.append(Prediction(video_id="video",frame_ids=[109]))
        r=score_case(case(),rows)
        self.assertEqual(r["final_score"],.6)
        self.assertEqual(r["R@5"],0)

    def test_trake_partial_credit_and_wrong_video(self):
        c=case("TRAKE")
        rows=[Prediction(video_id="video",frame_ids=[100,210,300,400])]
        self.assertEqual(score_case(c,rows)["final_score"],.75)
        self.assertEqual(score_case(c,[rows[0].model_copy(update={"video_id":"other"})])["final_score"],0)

    def test_qa_human_aliases_and_evidence_both_required(self):
        c=case("QA")
        self.assertEqual(score_case(c,[Prediction(video_id="video",frame_ids=[100],answer=" NĂM ")])["final_score"],1)
        self.assertEqual(score_case(c,[Prediction(video_id="video",frame_ids=[99],answer="5")])["final_score"],0)
        self.assertEqual(score_case(c,[Prediction(video_id="video",frame_ids=[100],answer="5 kg")])["final_score"],0)

    def test_empty_result_is_zero_not_a_diagnostic_pass(self):
        self.assertEqual(score_case(case(),[])["final_score"],0)

    def test_held_out_video_leakage_rejected(self):
        c=case(); other=case(split="held_out")
        other.query=other.query.model_copy(update={"query_name":"other-kis","source_filename":"other-kis.txt"})
        with self.assertRaises(ValueError): validate_suite([c,other])

    def test_frame_numbers_and_query_names_do_not_change_score(self):
        c=case(); row=Prediction(video_id="video",frame_ids=[105])
        original=score_case(c,[row])["final_score"]
        c.video_id="renamed"; c.intervals[0].start+=731; c.intervals[0].end+=731
        self.assertEqual(score_case(c,[Prediction(video_id="renamed",frame_ids=[836])])["final_score"],original)

    def test_acceptance_requires_all_sixty_reviewed_cases(self):
        with self.assertRaises(ValueError): validate_suite([case()],acceptance=True)

    def test_invalid_or_duplicate_predictions_fail(self):
        with self.assertRaises(ValueError): Prediction(video_id="x",frame_ids=[1.5])
        r=Prediction(video_id="video",frame_ids=[100])
        with self.assertRaises(ValueError): score_case(case(),[r,r])
