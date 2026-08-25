import hashlib
import unittest

from offline.identifiers import make_keyframe_uid


class KeyframeUidTests(unittest.TestCase):
    def test_uid_is_deterministic_positive_int64(self):
        first = make_keyframe_uid("L01_V001", "s000001", 2)
        second = make_keyframe_uid("L01_V001", "s000001", 2)
        self.assertEqual(first, second)
        self.assertGreater(first, 0)
        self.assertLessEqual(first, (1 << 63) - 1)

    def test_uid_matches_spec_formula_and_golden_vector(self):
        video_id = "L01_V001"
        shot_id = "s000001"
        local_idx = 2

        raw = f"{video_id}:{shot_id}:{local_idx}"
        spec_digest = hashlib.blake2b(
            raw.encode("utf-8"), digest_size=8
        ).digest()
        spec_uid = int.from_bytes(spec_digest, "big", signed=False) >> 1

        self.assertEqual(spec_uid, 8984422734592370359)
        self.assertEqual(
            make_keyframe_uid(video_id, shot_id, local_idx),
            spec_uid,
        )

    def test_uid_changes_with_every_identity_component(self):
        baseline = make_keyframe_uid("L01_V001", "s000001", 2)
        variants = {
            make_keyframe_uid("L02_V001", "s000001", 2),
            make_keyframe_uid("L01_V001", "s000002", 2),
            make_keyframe_uid("L01_V001", "s000001", 3),
        }
        self.assertNotIn(baseline, variants)
        self.assertEqual(len(variants), 3)

    def test_uid_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            make_keyframe_uid("", "s1", 0)
        with self.assertRaises(ValueError):
            make_keyframe_uid("v1", "", 0)
        with self.assertRaises(ValueError):
            make_keyframe_uid("v1", "s1", -1)

    def test_uid_rejects_whitespace_instead_of_normalizing(self):
        invalid_inputs = (
            (" L01_V001", "s000001"),
            ("L01_V001 ", "s000001"),
            ("L01 V001", "s000001"),
            ("L01_V001", "\ts000001"),
            ("L01_V001", "s000001\n"),
            ("L01_V001", "s00 0001"),
        )
        for video_id, shot_id in invalid_inputs:
            with self.subTest(video_id=video_id, shot_id=shot_id):
                with self.assertRaises(ValueError):
                    make_keyframe_uid(video_id, shot_id, 2)


if __name__ == "__main__":
    unittest.main()
