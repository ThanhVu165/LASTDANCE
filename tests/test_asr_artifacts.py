import unittest

from pydantic import ValidationError

from offline.asr_artifacts import (
    AsrRecordEnvelope,
    AsrVideoStatus,
    summarize_asr_coverage,
    video_set_sha256,
)


class AsrArtifactTests(unittest.TestCase):
    def test_video_hash_is_order_independent(self):
        self.assertEqual(video_set_sha256(["b", "a"]), video_set_sha256(["a", "b", "a"]))

    def test_silent_envelope_and_coverage(self):
        row = AsrRecordEnvelope(
            batch_id="batch-01",
            video_id="v1",
            status=AsrVideoStatus.SILENT,
            engine="whisper_large_v3",
            audio_path="asr/audio/batch-01/v1.flac",
            duration_seconds=2,
        )
        report = summarize_asr_coverage([row], expected_video_ids=["v1"])
        self.assertFalse(report["completion_gate_passed"])
        self.assertEqual(report["unverified_silent_videos"], 1)
        verified = AsrRecordEnvelope.model_validate({**row.model_dump(), "audio_sha256": "a" * 64,
            "silence_verification": {"audio_sha256": "a" * 64, "reviewed_by": "test reviewer",
                                     "evidence_path": "reviews/v1.json"}})
        self.assertTrue(summarize_asr_coverage([verified], expected_video_ids=["v1"])["completion_gate_passed"])

    def test_error_requires_code(self):
        with self.assertRaises(ValidationError):
            AsrRecordEnvelope(
                batch_id="batch-01", video_id="v1", status="error",
                engine="whisper_large_v3", audio_path="audio.flac",
                duration_seconds=1,
            )


if __name__ == "__main__":
    unittest.main()
