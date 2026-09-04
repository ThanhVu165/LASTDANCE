import unittest

from offline.asr_alignment import build_asr_segments, nearest_keyframe_uid
from offline.identifiers import make_keyframe_uid
from shared.schemas.frame import FrameRecord


class AsrAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.frames = [
            FrameRecord(video_id="v1", local_idx=i, frame_id=i, pts_time=float(i),
                        shot_id=f"s{i}", keyframe_uid=make_keyframe_uid("v1", f"s{i}", i))
            for i in range(3)
        ]

    def test_prefers_keyframe_inside_segment(self):
        self.assertEqual(nearest_keyframe_uid("v1", 0.8, 1.2, self.frames), self.frames[1].keyframe_uid)

    def test_fallback_uses_start_time(self):
        self.assertEqual(nearest_keyframe_uid("v1", 1.6, 1.8, self.frames), self.frames[2].keyframe_uid)

    def test_builds_canonical_segment(self):
        rows = build_asr_segments("v1", [{"start": 0, "end": 1, "text": "hello", "language": "en"}], self.frames)
        self.assertEqual(rows[0].segment_id, "s000000")
        self.assertEqual(rows[0].keyframe_uid_nearest, self.frames[0].keyframe_uid)


if __name__ == "__main__":
    unittest.main()
