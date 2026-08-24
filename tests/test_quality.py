import json
import tempfile
import unittest
from pathlib import Path

from offline.preprocessing.keyframes import select_keyframes
from offline.preprocessing.models import ShotBoundary
from offline.preprocessing.quality import (
    QualityMetrics,
    assess_keyframes,
    phash_distance,
    write_quality_manifest_atomic,
)


class QualityFilteringTests(unittest.TestCase):
    def _items(self):
        return select_keyframes(
            video_id="V1",
            shots=[
                ShotBoundary("s0", 0, 2),
                ShotBoundary("s1", 3, 5),
            ],
            frame_timestamps=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        )

    @staticmethod
    def _reader(variances, hashes):
        def read(path):
            local_idx = int(path.stem.rsplit("_", 1)[1])
            return QualityMetrics(variances[local_idx], hashes[local_idx])

        return read

    def test_report_only_measures_without_filtering(self):
        items = self._items()
        decisions = assess_keyframes(
            items,
            data_root=Path("data"),
            metric_reader=self._reader([1, 2, 3, 4, 5, 6], ["0"] * 6),
        )
        self.assertTrue(all(decision.kept for decision in decisions))
        self.assertEqual({decision.reason for decision in decisions}, {"kept"})

    def test_blur_filter_preserves_best_frame_when_entire_shot_is_below_threshold(self):
        items = self._items()
        decisions = assess_keyframes(
            items,
            data_root=Path("data"),
            blur_threshold=100.0,
            metric_reader=self._reader([1, 3, 2, 6, 4, 5], ["0"] * 6),
        )
        kept = [decision for decision in decisions if decision.kept]
        self.assertEqual([(row.shot_id, row.local_idx) for row in kept], [("s0", 1), ("s1", 3)])
        self.assertTrue(all(row.reason == "kept_best_blur_fallback" for row in kept))

    def test_phash_dedup_is_limited_to_each_shot(self):
        items = self._items()
        decisions = assess_keyframes(
            items,
            data_root=Path("data"),
            phash_max_distance=0,
            metric_reader=self._reader([10] * 6, ["00"] * 6),
        )
        kept = [decision for decision in decisions if decision.kept]
        self.assertEqual([(row.shot_id, row.local_idx) for row in kept], [("s0", 0), ("s1", 3)])
        duplicates = [row for row in decisions if row.reason == "near_duplicate"]
        self.assertEqual(len(duplicates), 4)
        self.assertEqual(duplicates[0].duplicate_of_keyframe_uid, kept[0].keyframe_uid)

    def test_phash_distance_uses_hamming_bits(self):
        self.assertEqual(phash_distance("00", "03"), 2)
        with self.assertRaisesRegex(ValueError, "same bit length"):
            phash_distance("0", "00")

    def test_manifest_is_atomic_and_records_reversible_decisions(self):
        items = self._items()
        decisions = assess_keyframes(
            items,
            data_root=Path("data"),
            phash_max_distance=0,
            metric_reader=self._reader([10] * 6, ["00"] * 6),
        )
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "quality.json"
            write_quality_manifest_atomic(
                output,
                video_id="V1",
                source_plan_sha256="a" * 64,
                blur_threshold=None,
                phash_max_distance=0,
                decisions=decisions,
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["counts"]["input"], 6)
            self.assertEqual(payload["counts"]["kept"], 2)
            self.assertEqual(payload["counts"]["near_duplicates"], 4)
            self.assertEqual(len(payload["config_signature"]), 64)
            self.assertFalse(output.with_name("quality.json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
