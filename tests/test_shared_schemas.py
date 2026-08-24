import unittest

from pydantic import ValidationError

from shared.schemas import AsrSegment, FrameRecord, OcrResult


class SharedSchemaTests(unittest.TestCase):
    def test_frame_record_keeps_internal_and_submission_identifiers_distinct(self):
        record = FrameRecord(
            video_id="L01_V001",
            local_idx=7,
            frame_id=1500,
            pts_time=50.0,
            shot_id="s000003",
            keyframe_uid=12345,
        )
        self.assertEqual(record.local_idx, 7)
        self.assertEqual(record.frame_id, 1500)
        self.assertIsNone(record.window_id)

    def test_frame_record_rejects_unknown_contract_fields(self):
        with self.assertRaises(ValidationError):
            FrameRecord(
                video_id="L01_V001",
                local_idx=0,
                frame_id=0,
                pts_time=0,
                shot_id="s000000",
                keyframe_uid=1,
                faiss_row_id=0,
            )

    def test_ocr_text_and_bbox_must_align(self):
        with self.assertRaisesRegex(ValidationError, "same length"):
            OcrResult(
                frame_id=12,
                detected_text=["Xin chào"],
                bbox=[],
                confidence=0.9,
                language="vi",
            )

    def test_asr_segment_requires_monotonic_timestamps(self):
        with self.assertRaisesRegex(ValidationError, "end_time"):
            AsrSegment(
                video_id="L01_V001",
                segment_id="seg-1",
                start_time=8.0,
                end_time=7.5,
                transcribed_text="Xin chào",
                language="vi",
                keyframe_uid_nearest=123,
            )


if __name__ == "__main__":
    unittest.main()
