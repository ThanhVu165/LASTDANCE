import tempfile
import unittest
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from offline.preprocessing.shot_detection import (
    ExcludedTransitionRange,
    ShotDetectionResult,
    TransNetV2ShotDetector,
    resolve_and_verify_transnetv2_weights,
    write_shot_manifest_atomic,
)
from offline.preprocessing.models import ShotBoundary


class FakeTransNetModel:
    def __init__(self):
        self.video_paths = []

    def predict_video(self, path, quiet=False):
        self.video_paths.append(path)
        self.quiet = quiet
        predictions = [0.0] * 20
        return object(), predictions, predictions

    def predictions_to_scenes(self, predictions, threshold=0.5):
        self.predictions = predictions
        self.threshold = threshold
        return [[0, 9], [10, 19]]


class ShotDetectionTests(unittest.TestCase):
    def test_transnet_adapter_is_lazy_and_normalizes_boundaries(self):
        model = FakeTransNetModel()
        factory_calls = []

        def factory():
            factory_calls.append(True)
            return model

        detector = TransNetV2ShotDetector(model_factory=factory)
        self.assertEqual(factory_calls, [])

        detection = detector.detect(Path("video.mp4"))

        self.assertEqual(len(factory_calls), 1)
        self.assertEqual(
            [shot.shot_id for shot in detection.shots],
            ["s000000", "s000001"],
        )
        self.assertEqual(
            (detection.shots[1].start_frame, detection.shots[1].end_frame),
            (10, 19),
        )
        self.assertEqual(detection.total_frame_count, 20)
        self.assertEqual(detection.excluded_transition_ranges, ())
        self.assertTrue(model.quiet)
        self.assertEqual(model.threshold, 0.5)
        self.assertEqual(detector.signature["weights_source"], "injected-model-factory")
        self.assertEqual(detector.signature["device"], "cpu")

    def test_cuda_device_is_explicit_and_recorded(self):
        detector = TransNetV2ShotDetector(
            model_factory=FakeTransNetModel,
            device="cuda",
        )
        with patch("torch.cuda.is_available", return_value=True):
            detection = detector.detect(Path("video.mp4"))

        self.assertEqual(len(detection.shots), 2)
        self.assertEqual(detector.signature["device"], "cuda")

    def test_cuda_device_never_falls_back_when_unavailable(self):
        detector = TransNetV2ShotDetector(
            model_factory=FakeTransNetModel,
            device="cuda",
        )
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                detector.detect(Path("video.mp4"))

    def test_unknown_device_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be one of"):
            TransNetV2ShotDetector(
                model_factory=FakeTransNetModel,
                device="auto",
            )

    def test_overlapping_boundaries_fail_closed(self):
        class OverlappingModel(FakeTransNetModel):
            def predictions_to_scenes(self, predictions, threshold=0.5):
                return [[0, 10], [10, 20]]

        detector = TransNetV2ShotDetector(model_factory=OverlappingModel)
        with self.assertRaisesRegex(RuntimeError, "non-overlapping"):
            detector.detect(Path("video.mp4"))

    def test_transition_gaps_are_explicit_in_detection_result(self):
        class TransitionGapModel(FakeTransNetModel):
            def predict_video(self, path, quiet=False):
                predictions = [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
                return object(), predictions, predictions

            def predictions_to_scenes(self, predictions, threshold=0.5):
                return [[1, 4], [6, 7]]

        detection = TransNetV2ShotDetector(
            model_factory=TransitionGapModel
        ).detect(Path("video.mp4"))

        self.assertEqual(
            detection.excluded_transition_ranges,
            (
                ExcludedTransitionRange(0, 0),
                ExcludedTransitionRange(5, 5),
            ),
        )

    def test_manifest_warns_but_publishes_when_exclusion_exceeds_one_percent(self):
        detector = TransNetV2ShotDetector(model_factory=FakeTransNetModel)
        detection = ShotDetectionResult(
            shots=(
                ShotBoundary("s000000", 0, 48),
                ShotBoundary("s000001", 51, 99),
            ),
            total_frame_count=100,
            excluded_transition_ranges=(ExcludedTransitionRange(49, 50),),
        )

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "shots.json"
            with self.assertWarnsRegex(RuntimeWarning, "2.000% > 1.000%"):
                write_shot_manifest_atomic(
                    output,
                    video_id="V1",
                    relative_video_path="videos/V1.mp4",
                    detector=detector,
                    detection=detection,
                )
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(
            payload["excluded_transition_ranges"],
            [
                {
                    "start_frame": 49,
                    "end_frame": 50,
                    "reason": "transition_score_above_threshold",
                }
            ],
        )
        self.assertEqual(
            payload["transition_exclusion_validation"]["excluded_frame_count"],
            2,
        )
        self.assertTrue(
            payload["transition_exclusion_validation"][
                "exceeds_warning_threshold"
            ]
        )

    def test_manifest_rejects_unaccounted_transition_gap(self):
        detector = TransNetV2ShotDetector(model_factory=FakeTransNetModel)
        detection = ShotDetectionResult(
            shots=(
                ShotBoundary("s000000", 0, 4),
                ShotBoundary("s000001", 6, 9),
            ),
            total_frame_count=10,
            excluded_transition_ranges=(),
        )

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "shots.json"
            with self.assertRaisesRegex(RuntimeError, "do not match gaps"):
                write_shot_manifest_atomic(
                    output,
                    video_id="V1",
                    relative_video_path="videos/V1.mp4",
                    detector=detector,
                    detection=detection,
                )
            self.assertFalse(output.exists())

    def test_external_weight_requires_and_verifies_checksum(self):
        with tempfile.TemporaryDirectory() as temporary:
            weight = Path(temporary) / "weights.pth"
            weight.write_bytes(b"verified-weight")
            with self.assertRaisesRegex(RuntimeError, "require a SHA-256"):
                resolve_and_verify_transnetv2_weights(weight, None)

            expected = hashlib.sha256(b"verified-weight").hexdigest()
            resolved, actual, source = resolve_and_verify_transnetv2_weights(
                weight,
                expected,
            )

        self.assertEqual(resolved, weight.resolve())
        self.assertEqual(actual, expected)
        self.assertEqual(source, "external")


if __name__ == "__main__":
    unittest.main()
