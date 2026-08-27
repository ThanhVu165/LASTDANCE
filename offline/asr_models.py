"""Pinned PyTorch/Transformers Whisper adapters for the ASR Dev Gate."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from offline.artifacts import sha256_file
from offline.asr_artifacts import RawAsrSegment
from offline.asr_audio import CHANNELS, SAMPLE_RATE_HZ, SAMPLE_WIDTH_BYTES


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASR_MODEL_CONFIG = _REPOSITORY_ROOT / "configs" / "asr_models.json"
SUPPORTED_ASR_MODELS = ("whisper_large_v3", "phowhisper_large")


def _require_revision(value: object) -> str:
    revision = str(value)
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise RuntimeError("ASR model revision must be an immutable commit SHA")
    return revision


def _require_sha256(value: object) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise RuntimeError("ASR model weight SHA-256 is invalid")
    return digest


def load_asr_model_config(
    model_key: str, *, path: Path = DEFAULT_ASR_MODEL_CONFIG
) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read ASR model registry: {path}") from error
    if payload.get("schema_version") != 1 or payload.get("runner") != "transformers_pytorch":
        raise RuntimeError("unsupported ASR model registry schema/runner")
    row = payload.get(model_key)
    if model_key not in SUPPORTED_ASR_MODELS or not isinstance(row, dict):
        raise ValueError(f"unknown ASR model: {model_key}")
    required_strings = ("model_id", "weight_filename", "weight_format")
    if any(not isinstance(row.get(key), str) or not row[key].strip() for key in required_strings):
        raise RuntimeError(f"ASR model registry row is incomplete: {model_key}")
    row = dict(row)
    row["revision"] = _require_revision(row.get("revision"))
    if row.get("weight_sha256") is not None:
        row["weight_sha256"] = _require_sha256(row["weight_sha256"])
    if row["weight_format"] == "safetensors":
        if not row["weight_filename"].endswith(".safetensors") or row["weight_sha256"] is None:
            raise RuntimeError("safetensors model must pin its exact weight SHA-256")
    elif row["weight_format"] == "pytorch_bin_weights_only":
        if not row["weight_filename"].endswith(".bin"):
            raise RuntimeError("weights-only PyTorch model must name a .bin checkpoint")
        if row.get("production_allowed") is not False:
            raise RuntimeError("unconverted PyTorch .bin ASR weights cannot be production-ready")
    else:
        raise RuntimeError(f"unsupported ASR weight format: {row['weight_format']}")
    if row.get("forced_language") not in {None, "vi", "en"}:
        raise RuntimeError("forced_language must be null, vi, or en")
    if type(row.get("dev_gate_allowed")) is not bool or type(row.get("production_allowed")) is not bool:
        raise RuntimeError("ASR model allow flags must be boolean")
    return row


def _torch_version_tuple(version: str) -> tuple[int, int]:
    base = version.split("+", 1)[0]
    parts = base.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError) as error:
        raise RuntimeError(f"cannot parse Torch version: {version}") from error


def _load_pcm16_mono(path: Path) -> np.ndarray:
    source = Path(path)
    try:
        with wave.open(str(source), "rb") as handle:
            if (
                handle.getnchannels() != CHANNELS
                or handle.getsampwidth() != SAMPLE_WIDTH_BYTES
                or handle.getframerate() != SAMPLE_RATE_HZ
                or handle.getcomptype() != "NONE"
            ):
                raise RuntimeError("ASR input must be PCM s16le 16 kHz mono")
            frames = handle.readframes(handle.getnframes())
    except (OSError, EOFError, wave.Error) as error:
        raise RuntimeError(f"cannot read ASR WAV: {source}") from error
    values = np.frombuffer(frames, dtype="<i2")
    if not values.size:
        raise RuntimeError("ASR WAV contains no samples")
    return values.astype(np.float32) / 32768.0


def normalize_whisper_language(value: object) -> str:
    normalized = str(value).strip().lower()
    mapping = {
        "vi": "vi",
        "<|vi|>": "vi",
        "vietnamese": "vi",
        "en": "en",
        "<|en|>": "en",
        "english": "en",
    }
    if normalized not in mapping:
        raise RuntimeError(f"Whisper returned unsupported language: {value!r}")
    return mapping[normalized]


def _initialize_cuda_memory_measurement(torch_module: Any) -> tuple[int, Any]:
    """Bind to Kaggle's active CUDA device before resetting allocator statistics."""

    device_index = int(torch_module.cuda.current_device())
    cuda_device = torch_module.device(f"cuda:{device_index}")
    torch_module.cuda.set_device(cuda_device)
    probe = torch_module.empty(1, device=cuda_device)
    del probe
    torch_module.cuda.empty_cache()
    # Use the active device. Passing a hard-coded integer fails on some Kaggle
    # Torch/CUDA builds even though CUDA inference itself is available.
    torch_module.cuda.reset_peak_memory_stats()
    return device_index, cuda_device


@dataclass(frozen=True, slots=True)
class TranscriptionOutput:
    segments: tuple[RawAsrSegment, ...]


class TransformersWhisperTranscriber:
    """One-model-per-process Dev Gate runner with exact CUDA provenance."""

    def __init__(
        self,
        *,
        model_key: str,
        config_path: Path = DEFAULT_ASR_MODEL_CONFIG,
        purpose: str = "dev_gate",
        device: str = "cuda",
    ) -> None:
        if purpose not in {"dev_gate", "production"}:
            raise ValueError("purpose must be dev_gate or production")
        if device != "cuda":
            raise ValueError("ASR model runner requires explicit device='cuda'")
        row = load_asr_model_config(model_key, path=config_path)
        if purpose == "dev_gate" and not row["dev_gate_allowed"]:
            raise RuntimeError(f"ASR model is not approved for Dev Gate: {model_key}")
        if purpose == "production" and not row["production_allowed"]:
            raise RuntimeError(f"ASR model is not approved for production: {model_key}")
        self.model_key = model_key
        self.model_id = str(row["model_id"])
        self.model_revision = str(row["revision"])
        self.weight_filename = str(row["weight_filename"])
        self.weight_format = str(row["weight_format"])
        self.expected_weight_sha256 = row.get("weight_sha256")
        self.forced_language = row.get("forced_language")
        self.device = device
        self._torch: Any | None = None
        self._pipeline: Any | None = None
        self._weight_sha256: str | None = None
        self._snapshot_path: Path | None = None
        self._cuda_device_index: int | None = None

    def prepare(self) -> None:
        if self._pipeline is not None:
            return
        try:
            import torch
            from huggingface_hub import snapshot_download
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        except ImportError as error:
            raise RuntimeError(
                "ASR Kaggle dependencies are missing; install requirements/asr-kaggle-gpu.txt"
            ) from error
        if not torch.cuda.is_available():
            raise RuntimeError("ASR Dev Gate requires CUDA; CPU fallback is forbidden")
        torch_version = importlib.metadata.version("torch")
        if self.weight_format == "pytorch_bin_weights_only" and _torch_version_tuple(
            torch_version
        ) < (2, 6):
            raise RuntimeError("weights-only .bin loading requires torch>=2.6")

        snapshot = Path(
            snapshot_download(
                repo_id=self.model_id,
                revision=self.model_revision,
                allow_patterns=[
                    self.weight_filename,
                    "*.json",
                    "*.txt",
                    "merges.txt",
                    "vocab.json",
                ],
            )
        )
        weight_path = snapshot / self.weight_filename
        if not weight_path.is_file():
            raise RuntimeError(f"pinned ASR weight is missing: {self.weight_filename}")
        actual_sha256 = sha256_file(weight_path)
        if (
            self.expected_weight_sha256 is not None
            and actual_sha256 != self.expected_weight_sha256
        ):
            raise RuntimeError(
                f"ASR weight SHA-256 mismatch: expected={self.expected_weight_sha256}, "
                f"actual={actual_sha256}"
            )

        cuda_device_index, cuda_device = _initialize_cuda_memory_measurement(torch)
        use_safetensors = self.weight_format == "safetensors"
        processor = AutoProcessor.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False
        )
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=use_safetensors,
            weights_only=True,
            dtype=torch.float16,
        )
        model.eval().to(cuda_device)
        self._pipeline = pipeline(
            task="automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=cuda_device_index,
            dtype=torch.float16,
        )
        self._torch = torch
        self._weight_sha256 = actual_sha256
        self._snapshot_path = snapshot
        self._cuda_device_index = cuda_device_index

    @property
    def weight_sha256(self) -> str:
        self.prepare()
        assert self._weight_sha256 is not None
        return self._weight_sha256

    @property
    def runtime_metadata(self) -> dict[str, object]:
        self.prepare()
        assert self._torch is not None
        assert self._cuda_device_index is not None
        return {
            "runner": "transformers_pytorch",
            "device": self.device,
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "transformers": importlib.metadata.version("transformers"),
            "torch": importlib.metadata.version("torch"),
            "huggingface_hub": importlib.metadata.version("huggingface-hub"),
            "cuda_runtime": str(self._torch.version.cuda),
            "cuda_device_index": self._cuda_device_index,
            "gpu_name": self._torch.cuda.get_device_name(),
            "peak_cuda_memory_bytes": self._torch.cuda.max_memory_allocated(),
            "weight_filename": self.weight_filename,
            "weight_format": self.weight_format,
            "weight_sha256": self.weight_sha256,
        }

    def transcribe(self, audio_path: Path, *, video_id: str) -> TranscriptionOutput:
        self.prepare()
        assert self._pipeline is not None
        audio = _load_pcm16_mono(audio_path)
        call_kwargs: dict[str, object] = {
            "return_timestamps": True,
            "return_language": True,
            "task": "transcribe",
        }
        if self.forced_language is not None:
            call_kwargs["language"] = self.forced_language
        result = self._pipeline(
            {"raw": audio, "sampling_rate": SAMPLE_RATE_HZ}, **call_kwargs
        )
        if not isinstance(result, dict):
            raise RuntimeError("Whisper pipeline returned a non-object result")
        chunks = result.get("chunks")
        if not isinstance(chunks, list):
            raise RuntimeError("Whisper did not return segment timestamps")
        top_language = result.get("language")
        segments: list[RawAsrSegment] = []
        duration = len(audio) / SAMPLE_RATE_HZ
        for chunk in chunks:
            if not isinstance(chunk, dict):
                raise RuntimeError("Whisper returned an invalid segment object")
            text = " ".join(str(chunk.get("text", "")).split())
            if not text:
                continue
            timestamp = chunk.get("timestamp")
            if (
                not isinstance(timestamp, (tuple, list))
                or len(timestamp) != 2
                or timestamp[0] is None
                or timestamp[1] is None
            ):
                raise RuntimeError("Whisper returned a segment without a closed timestamp")
            start = float(timestamp[0])
            end = float(timestamp[1])
            if start < 0 or end < start or end > duration + 2.0:
                raise RuntimeError("Whisper returned an invalid/out-of-range timestamp")
            language_value = (
                chunk.get("language")
                or top_language
                or self.forced_language
            )
            if language_value is None:
                raise RuntimeError("Whisper did not return a language for the segment")
            segments.append(
                RawAsrSegment(
                    segment_id=f"{video_id}:seg-{len(segments):06d}",
                    start_time=start,
                    end_time=end,
                    transcribed_text=text,
                    language=normalize_whisper_language(language_value),
                )
            )
        if not segments and str(result.get("text", "")).strip():
            raise RuntimeError("Whisper returned text without usable timestamped segments")
        return TranscriptionOutput(segments=tuple(segments))

    def release(self) -> None:
        self._pipeline = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


def create_asr_transcriber(
    model_key: str,
    *,
    config_path: Path = DEFAULT_ASR_MODEL_CONFIG,
    purpose: str = "dev_gate",
    device: str = "cuda",
) -> TransformersWhisperTranscriber:
    return TransformersWhisperTranscriber(
        model_key=model_key,
        config_path=config_path,
        purpose=purpose,
        device=device,
    )
