import hashlib
import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from offline.asr_artifacts import RawAsrSegment
from offline.asr_audio import AudioArtifact, AudioStatus
from offline.asr_models import TranscriptionOutput
from offline.asr_transcription import (
    IntentionalAsrInterruption,
    run_asr_transcription,
)


class FakeTranscriber:
    model_key = "whisper_large_v3"
    model_id = "openai/whisper-large-v3"
    model_revision = "a" * 40
    weight_sha256 = "b" * 64

    def __init__(self):
        self.calls = []

    def prepare(self):
        return None

    @property
    def runtime_metadata(self):
        return {
            "runner": "transformers_pytorch",
            "gpu_name": "Tesla T4",
            "peak_cuda_memory_bytes": 123,
        }

    def transcribe(self, audio_path: Path, *, video_id: str):
        self.calls.append(video_id)
        return TranscriptionOutput(
            segments=(
                RawAsrSegment(
                    segment_id=f"{video_id}:seg-000000",
                    start_time=0.0,
                    end_time=0.5,
                    transcribed_text="xin chao",
                    language="vi",
                ),
            )
        )


def _write_audio(audio_root: Path, video_id: str) -> None:
    wav = audio_root / "wav" / f"{video_id}.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(wav), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\0\0" * 16_000)
    digest = hashlib.sha256(wav.read_bytes()).hexdigest()
    artifact = AudioArtifact(
        video_id=video_id,
        status=AudioStatus.READY,
        source_video=f"videos/{video_id}.mp4",
        source_size_bytes=100,
        inventory_duration_seconds=1.0,
        extraction_signature="c" * 64,
        ffmpeg_version="ffmpeg version test",
        wav_path=f"wav/{video_id}.wav",
        wav_sha256=digest,
        wav_size_bytes=wav.stat().st_size,
        wav_duration_seconds=1.0,
        sample_rate_hz=16_000,
        channels=1,
        sample_width_bytes=2,
        codec="pcm_s16le",
        megabytes_per_minute=1.92264,
        wav_to_source_size_ratio=wav.stat().st_size / 100,
    )
    manifest = audio_root / "manifests" / f"{video_id}.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(artifact.model_dump(mode="json")), encoding="utf-8")


class AsrTranscriptionResumeTests(unittest.TestCase):
    def test_interrupt_then_new_process_resume_without_duplicates(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            audio_root = root / "audio"
            for video_id in ("V001", "V002"):
                _write_audio(audio_root, video_id)
            transcriber = FakeTranscriber()
            with patch("offline.asr_transcription._PROCESS_TOKEN", "process-a"):
                with self.assertRaises(IntentionalAsrInterruption):
                    run_asr_transcription(
                        transcriber=transcriber,
                        batch_id="dev-subset-5",
                        audio_root=audio_root,
                        output_root=root / "output",
                        video_ids=["V001", "V002"],
                        stop_after_videos=1,
                    )
            self.assertEqual(transcriber.calls, ["V001"])

            with patch("offline.asr_transcription._PROCESS_TOKEN", "process-b"):
                result = run_asr_transcription(
                    transcriber=transcriber,
                    batch_id="dev-subset-5",
                    audio_root=audio_root,
                    output_root=root / "output",
                    video_ids=["V001", "V002"],
                )
            self.assertTrue(result.complete)
            self.assertTrue(result.checkpoint_resume_verified)
            self.assertEqual(transcriber.calls, ["V001", "V002"])
            records = list((result.output_dir / "records").glob("*.json"))
            self.assertEqual(len(records), 2)


if __name__ == "__main__":
    unittest.main()
