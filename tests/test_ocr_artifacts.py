import unittest

from pydantic import ValidationError

from offline.ocr_artifacts import (
    OcrAttempt,
    OcrAttemptOutcome,
    OcrEngine,
    OcrExecutionMode,
    OcrRecordEnvelope,
    OcrStatus,
    aggregate_easyocr_confidence,
    summarize_ocr_coverage,
)
from shared.schemas.ocr import OcrResult


def _attempt(
    engine: OcrEngine,
    number: int,
    outcome: OcrAttemptOutcome,
) -> OcrAttempt:
    failed = outcome in {
        OcrAttemptOutcome.INVALID_RESPONSE,
        OcrAttemptOutcome.RETRYABLE_ERROR,
        OcrAttemptOutcome.TERMINAL_ERROR,
    }
    return OcrAttempt(
        engine=engine,
        attempt_number=number,
        outcome=outcome,
        latency_ms=10,
        error_code="invalid_json" if failed else None,
    )


def _success(uid: int) -> OcrRecordEnvelope:
    return OcrRecordEnvelope(
        batch_id="batch-01",
        video_id="L21_V001",
        keyframe_uid=uid,
        frame_id=100 + uid,
        source_image=f"L21_V001/s000001_{uid}.jpg",
        execution_mode=OcrExecutionMode.GEMINI_PRIMARY,
        status=OcrStatus.SUCCESS,
        engine=OcrEngine.GEMINI,
        fallback_used=False,
        result=OcrResult(
            frame_id=100 + uid,
            detected_text=["Xin chao"],
            bbox=[[0.1, 0.1, 0.9, 0.1, 0.9, 0.3, 0.1, 0.3]],
            confidence=0.9,
            language="vi",
        ),
        attempts=[_attempt(OcrEngine.GEMINI, 1, OcrAttemptOutcome.SUCCESS)],
    )


class OcrArtifactContractTests(unittest.TestCase):
    def test_success_enforces_normalized_quadrilateral(self):
        record = _success(1)
        self.assertEqual(record.status, OcrStatus.SUCCESS)
        with self.assertRaisesRegex(ValidationError, "quadrilateral"):
            OcrRecordEnvelope(
                **{
                    **record.model_dump(),
                    "result": {
                        **record.result.model_dump(),
                        "bbox": [[0.1, 0.1, 0.9, 0.3]],
                    },
                }
            )

    def test_easyocr_fallback_requires_invalid_gemini_response(self):
        record = OcrRecordEnvelope(
            batch_id="batch-01",
            video_id="L21_V001",
            keyframe_uid=2,
            frame_id=102,
            source_image="L21_V001/s000001_2.jpg",
            execution_mode=OcrExecutionMode.GEMINI_PRIMARY,
            status=OcrStatus.NO_TEXT,
            engine=OcrEngine.EASYOCR,
            fallback_used=True,
            result=None,
            attempts=[
                _attempt(OcrEngine.GEMINI, 1, OcrAttemptOutcome.INVALID_RESPONSE),
                _attempt(OcrEngine.EASYOCR, 2, OcrAttemptOutcome.NO_TEXT),
            ],
        )
        self.assertTrue(record.fallback_used)

        with self.assertRaisesRegex(ValidationError, "invalid Gemini response"):
            OcrRecordEnvelope(
                **{
                    **record.model_dump(),
                    "attempts": [
                        _attempt(
                            OcrEngine.GEMINI,
                            1,
                            OcrAttemptOutcome.RETRYABLE_ERROR,
                        ),
                        _attempt(OcrEngine.EASYOCR, 2, OcrAttemptOutcome.NO_TEXT),
                    ],
                }
            )

    def test_completion_gate_accepts_success_plus_no_text_but_not_error(self):
        no_text = OcrRecordEnvelope(
            batch_id="batch-01",
            video_id="L21_V001",
            keyframe_uid=2,
            frame_id=102,
            source_image="L21_V001/s000001_2.jpg",
            execution_mode=OcrExecutionMode.GEMINI_PRIMARY,
            status=OcrStatus.NO_TEXT,
            engine=OcrEngine.GEMINI,
            fallback_used=False,
            result=None,
            attempts=[_attempt(OcrEngine.GEMINI, 1, OcrAttemptOutcome.NO_TEXT)],
        )
        report = summarize_ocr_coverage(
            [_success(1), no_text], expected_keyframe_uids=[1, 2]
        )
        self.assertTrue(report["completion_gate_passed"])
        self.assertEqual(report["no_text_records"], 1)

        error = OcrRecordEnvelope(
            batch_id="batch-01",
            video_id="L21_V001",
            keyframe_uid=2,
            frame_id=102,
            source_image="L21_V001/s000001_2.jpg",
            execution_mode=OcrExecutionMode.GEMINI_PRIMARY,
            status=OcrStatus.ERROR,
            engine=OcrEngine.GEMINI,
            fallback_used=False,
            result=None,
            attempts=[
                _attempt(OcrEngine.GEMINI, 1, OcrAttemptOutcome.TERMINAL_ERROR)
            ],
        )
        failed = summarize_ocr_coverage(
            [_success(1), error], expected_keyframe_uids=[1, 2]
        )
        self.assertFalse(failed["completion_gate_passed"])
        self.assertEqual(failed["error_records"], 1)

    def test_explicit_easyocr_offline_mode_is_not_silent_fallback(self):
        record = OcrRecordEnvelope(
            batch_id="batch-01",
            video_id="L21_V001",
            keyframe_uid=3,
            frame_id=103,
            source_image="L21_V001/s000001_3.jpg",
            execution_mode=OcrExecutionMode.EASYOCR_OFFLINE,
            status=OcrStatus.NO_TEXT,
            engine=OcrEngine.EASYOCR,
            fallback_used=False,
            result=None,
            attempts=[_attempt(OcrEngine.EASYOCR, 1, OcrAttemptOutcome.NO_TEXT)],
        )
        self.assertFalse(record.fallback_used)

    def test_easyocr_confidence_is_weighted_by_non_whitespace_text_length(self):
        confidence = aggregate_easyocr_confidence([("AB", 1.0), (" C ", 0.4)])
        self.assertAlmostEqual(confidence, 0.8)


if __name__ == "__main__":
    unittest.main()
