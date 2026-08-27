import sqlite3
import tempfile
import unittest
from pathlib import Path

from offline.asr_artifacts import (
    AsrTranscriptRecord,
    RawAsrSegment,
    TranscriptStatus,
)
from offline.asr_audio import AudioArtifact, AudioStatus
from offline.asr_catalog import (
    AsrCoverageStatus,
    build_asr_sqlite_atomic,
    derive_asr_coverage,
    validate_asr_sqlite,
)
from shared.schemas.asr import AsrSegment


def _audio(video_id: str) -> AudioArtifact:
    return AudioArtifact(
        video_id=video_id,
        status=AudioStatus.READY,
        source_video=f"videos/{video_id}.mp4",
        source_size_bytes=100,
        inventory_duration_seconds=1.0,
        extraction_signature="a" * 64,
        ffmpeg_version="ffmpeg version test",
        wav_path=f"wav/{video_id}.wav",
        wav_sha256="b" * 64,
        wav_size_bytes=100,
        wav_duration_seconds=1.0,
        sample_rate_hz=16_000,
        channels=1,
        sample_width_bytes=2,
        codec="pcm_s16le",
        megabytes_per_minute=1.0,
        wav_to_source_size_ratio=1.0,
    )


def _transcript(video_id: str, no_speech: bool = False) -> AsrTranscriptRecord:
    segments = [] if no_speech else [
        RawAsrSegment(
            segment_id=f"{video_id}:seg-000000",
            start_time=0.0,
            end_time=1.0,
            transcribed_text="xin chào thế giới",
            language="vi",
        )
    ]
    return AsrTranscriptRecord(
        batch_id="dev-subset-5",
        video_id=video_id,
        model_key="whisper_large_v3",
        model_id="openai/whisper-large-v3",
        model_revision="c" * 40,
        source_wav=f"wav/{video_id}.wav",
        source_wav_sha256="b" * 64,
        source_duration_seconds=1.0,
        status=TranscriptStatus.NO_SPEECH if no_speech else TranscriptStatus.SUCCESS,
        elapsed_seconds=1.0,
        segments=segments,
    )


class AsrCatalogTests(unittest.TestCase):
    def test_builds_exact_fts5_schema_and_queryable_text(self):
        segment = AsrSegment(
            video_id="V001",
            segment_id="V001:seg-000000",
            start_time=0.0,
            end_time=1.0,
            transcribed_text="xin chào thế giới",
            language="vi",
            keyframe_uid_nearest=123,
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "asr.sqlite"
            build_asr_sqlite_atomic(path, [segment])
            self.assertTrue(validate_asr_sqlite(path, expected_segments=[segment]))
            connection = sqlite3.connect(path)
            try:
                row = connection.execute(
                    "SELECT video_id, keyframe_uid_nearest FROM asr_fts "
                    "WHERE asr_fts MATCH 'xin'"
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(row, ("V001", 123))

    def test_no_speech_never_completes_without_explicit_verification(self):
        rows = derive_asr_coverage(
            expected_video_ids=["V001"],
            inventory_has_audio={"V001": True},
            audio_artifacts={"V001": _audio("V001")},
            transcripts={"V001": _transcript("V001", no_speech=True)},
            aligned_segments=[],
        )
        self.assertEqual(rows[0].status, AsrCoverageStatus.NO_SPEECH_UNVERIFIED)
        self.assertFalse(rows[0].complete)

        verified = derive_asr_coverage(
            expected_video_ids=["V001"],
            inventory_has_audio={"V001": True},
            audio_artifacts={"V001": _audio("V001")},
            transcripts={"V001": _transcript("V001", no_speech=True)},
            aligned_segments=[],
            verified_no_speech={"V001"},
        )
        self.assertEqual(verified[0].status, AsrCoverageStatus.NO_SPEECH_VERIFIED)
        self.assertTrue(verified[0].complete)


if __name__ == "__main__":
    unittest.main()
