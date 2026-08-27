import json
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import Mock

from offline.asr_audio import (
    AudioStatus,
    extract_audio_artifact,
    validate_audio_artifact,
)
from offline.preprocessing.models import VideoInventoryRecord


class AsrAudioTests(unittest.TestCase):
    def test_extracts_pcm16k_mono_atomically_and_resumes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "data"
            video = data / "videos" / "L21_V001.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"source-video")
            audio_root = data / "asr" / "audio"
            record = VideoInventoryRecord(
                video_id="L21_V001",
                relative_path="videos/L21_V001.mp4",
                width=1280,
                height=720,
                fps=30.0,
                duration=1.0,
                frame_count=30,
                has_audio=True,
            )

            def fake_ffmpeg(command, **kwargs):
                output = Path(command[-1])
                with wave.open(str(output), "wb") as handle:
                    handle.setnchannels(1)
                    handle.setsampwidth(2)
                    handle.setframerate(16_000)
                    handle.writeframes(b"\0\0" * 16_000)
                return subprocess.CompletedProcess(command, 0, "", "")

            runner = Mock(side_effect=fake_ffmpeg)
            artifact = extract_audio_artifact(
                record,
                data_root=data,
                audio_root=audio_root,
                ffmpeg_binary="ffmpeg",
                ffmpeg_version_text="ffmpeg version test",
                runner=runner,
            )
            self.assertEqual(artifact.status, AudioStatus.READY)
            self.assertEqual(artifact.sample_rate_hz, 16_000)
            self.assertEqual(artifact.channels, 1)
            self.assertEqual(artifact.codec, "pcm_s16le")
            self.assertAlmostEqual(artifact.megabytes_per_minute, 1.92264, places=4)
            validate_audio_artifact(
                audio_root / "manifests" / "L21_V001.json", audio_root=audio_root
            )

            resumed = extract_audio_artifact(
                record,
                data_root=data,
                audio_root=audio_root,
                ffmpeg_binary="ffmpeg",
                ffmpeg_version_text="ffmpeg version test",
                runner=runner,
            )
            self.assertEqual(resumed.wav_sha256, artifact.wav_sha256)
            self.assertEqual(runner.call_count, 1)

    def test_no_audio_is_explicit_without_fake_wav(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "data"
            video = data / "videos" / "silent.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"source")
            record = VideoInventoryRecord(
                video_id="silent",
                relative_path="videos/silent.mp4",
                width=1,
                height=1,
                fps=1.0,
                duration=1.0,
                frame_count=1,
                has_audio=False,
            )
            artifact = extract_audio_artifact(
                record,
                data_root=data,
                audio_root=data / "asr" / "audio",
                ffmpeg_binary="ffmpeg",
                ffmpeg_version_text="ffmpeg version test",
                runner=Mock(),
            )
            self.assertEqual(artifact.status, AudioStatus.NO_AUDIO)
            self.assertIsNone(artifact.wav_path)
            payload = json.loads(
                (data / "asr" / "audio" / "manifests" / "silent.json").read_text()
            )
            self.assertEqual(payload["status"], "no_audio")


if __name__ == "__main__":
    unittest.main()
