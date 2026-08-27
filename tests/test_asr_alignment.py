import unittest

from offline.asr_alignment import align_transcript_records, nearest_frame_for_segment
from offline.asr_artifacts import (
    AsrTranscriptRecord,
    RawAsrSegment,
    TranscriptStatus,
)
from shared.schemas.frame import FrameRecord


def _frame(uid: int, pts: float) -> FrameRecord:
    return FrameRecord(
        video_id="V001",
        local_idx=uid,
        frame_id=uid * 10,
        pts_time=pts,
        shot_id=f"s{uid:06d}",
        keyframe_uid=uid,
    )


class AsrAlignmentTests(unittest.TestCase):
    def test_prefers_keyframe_in_range_nearest_segment_midpoint(self):
        frames = [_frame(1, 1.0), _frame(2, 4.0), _frame(3, 8.0)]
        nearest = nearest_frame_for_segment(frames, start_time=2.0, end_time=7.0)
        self.assertEqual(nearest.keyframe_uid, 2)

    def test_falls_back_to_keyframe_nearest_start(self):
        frames = [_frame(1, 1.0), _frame(2, 4.0)]
        nearest = nearest_frame_for_segment(frames, start_time=8.0, end_time=9.0)
        self.assertEqual(nearest.keyframe_uid, 2)

    def test_alignment_preserves_submission_independent_uid_contract(self):
        transcript = AsrTranscriptRecord(
            batch_id="dev-subset-5",
            video_id="V001",
            model_key="whisper_large_v3",
            model_id="openai/whisper-large-v3",
            model_revision="a" * 40,
            source_wav="wav/V001.wav",
            source_wav_sha256="b" * 64,
            source_duration_seconds=10.0,
            status=TranscriptStatus.SUCCESS,
            elapsed_seconds=1.0,
            segments=[
                RawAsrSegment(
                    segment_id="V001:seg-000000",
                    start_time=2.0,
                    end_time=7.0,
                    transcribed_text="xin chao",
                    language="vi",
                )
            ],
        )
        aligned = align_transcript_records(
            [transcript], frames_by_video={"V001": [_frame(1, 1.0), _frame(2, 4.0)]}
        )
        self.assertEqual(aligned[0].keyframe_uid_nearest, 2)


if __name__ == "__main__":
    unittest.main()
