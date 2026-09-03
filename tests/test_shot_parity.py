import tempfile
import unittest
from pathlib import Path

from offline.preprocessing.models import ShotBoundary
from offline.preprocessing.shot_detection import (
    ExcludedTransitionRange,
    ShotDetectionResult,
    write_shot_manifest_atomic,
)
from offline.preprocessing.shot_parity import compare_shot_manifests


class _Detector:
    name = "transnetv2"

    def __init__(self, device: str) -> None:
        self.signature = {
            "name": "transnetv2",
            "implementation": "transnetv2-pytorch",
            "package_version": "1.0.5",
            "device": device,
            "threshold": 0.5,
            "weights_source": "bundled",
            "weights_sha256": "a" * 64,
        }


class ShotManifestParityTests(unittest.TestCase):
    def test_device_may_differ_when_every_semantic_field_matches(self):
        detection = ShotDetectionResult(
            shots=(
                ShotBoundary("s000000", 0, 49),
                ShotBoundary("s000001", 51, 99),
            ),
            total_frame_count=100,
            excluded_transition_ranges=(ExcludedTransitionRange(50, 50),),
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            reference = root / "cpu.json"
            candidate = root / "cuda.json"
            for path, device in ((reference, "cpu"), (candidate, "cuda")):
                write_shot_manifest_atomic(
                    path,
                    video_id="V1",
                    relative_video_path="videos/V1.mp4",
                    detector=_Detector(device),
                    detection=detection,
                )

            mismatches = compare_shot_manifests(reference, candidate)

        self.assertEqual(mismatches, [])

    def test_first_boundary_mismatch_is_reported(self):
        reference_detection = ShotDetectionResult(
            shots=(
                ShotBoundary("s000000", 0, 49),
                ShotBoundary("s000001", 51, 99),
            ),
            total_frame_count=100,
            excluded_transition_ranges=(ExcludedTransitionRange(50, 50),),
        )
        candidate_detection = ShotDetectionResult(
            shots=(
                ShotBoundary("s000000", 0, 48),
                ShotBoundary("s000001", 50, 99),
            ),
            total_frame_count=100,
            excluded_transition_ranges=(ExcludedTransitionRange(49, 49),),
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            reference = root / "cpu.json"
            candidate = root / "cuda.json"
            write_shot_manifest_atomic(
                reference,
                video_id="V1",
                relative_video_path="videos/V1.mp4",
                detector=_Detector("cpu"),
                detection=reference_detection,
            )
            write_shot_manifest_atomic(
                candidate,
                video_id="V1",
                relative_video_path="videos/V1.mp4",
                detector=_Detector("cuda"),
                detection=candidate_detection,
            )

            mismatches = compare_shot_manifests(reference, candidate)

        self.assertTrue(any("shot[0] differs" in item for item in mismatches))
        self.assertTrue(
            any("excluded_transition_ranges differ" in item for item in mismatches)
        )


if __name__ == "__main__":
    unittest.main()
