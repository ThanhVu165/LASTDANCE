"""Pinned text encoders matching the three production visual indexes."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from .artifacts import EXPECTED_VISUAL


def _normalize(values: np.ndarray) -> np.ndarray:
    vectors = np.asarray(values, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    if not np.isfinite(vectors).all():
        raise RuntimeError("text encoder produced NaN or Inf")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise RuntimeError("text encoder produced a zero vector")
    return np.ascontiguousarray(vectors / norms, dtype=np.float32)


class TextEncoder(Protocol):
    dimension: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class TransformersTextEncoder:
    def __init__(self, modality: str, *, device: str = "cpu") -> None:
        if modality not in {"clip", "siglip"}:
            raise ValueError("TransformersTextEncoder supports clip or siglip")
        self.modality = modality
        self.dimension, self.model_id, self.revision = EXPECTED_VISUAL[modality]
        self.device = device
        self._torch: Any = None
        self._tokenizer: Any = None
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, CLIPModel, SiglipModel

        local_only = os.environ.get("AIC_ALLOW_MODEL_DOWNLOAD") != "1"
        model_class = CLIPModel if self.modality == "clip" else SiglipModel
        # Query-time retrieval is text-only. AutoProcessor also resolves the image
        # preprocessor and can fail in an otherwise complete offline text cache.
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=self.revision,
            local_files_only=local_only,
        )
        self._model = model_class.from_pretrained(self.model_id, revision=self.revision, local_files_only=local_only)
        self._model.eval().to(self.device)
        self._torch = torch

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self._load()
        inputs = self._tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            output = self._model.get_text_features(**inputs)
            features = output.pooler_output if hasattr(output, "pooler_output") else output
        if not isinstance(features, self._torch.Tensor):
            raise RuntimeError(f"{self.modality} text encoder returned a non-tensor")
        result = _normalize(features.detach().float().cpu().numpy())
        if result.shape[1] != self.dimension:
            raise RuntimeError(f"{self.modality} text dimension mismatch")
        return result


class OpenClipEVATextEncoder:
    def __init__(self, *, device: str = "cpu") -> None:
        self.modality = "eva_clip"
        self.dimension, self.model_id, self.revision = EXPECTED_VISUAL["eva_clip"]
        self.device = device
        self._torch: Any = None
        self._model: Any = None
        self._tokenizer: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import open_clip
            import torch
            from huggingface_hub import hf_hub_download
            from open_clip.factory import load_checkpoint
        except ImportError as error:
            raise RuntimeError("open-clip-torch and huggingface-hub are required for EVA text search") from error
        local_only = os.environ.get("AIC_ALLOW_MODEL_DOWNLOAD") != "1"
        config_path = Path(
            hf_hub_download(
                repo_id=self.model_id,
                filename="open_clip_config.json",
                revision=self.revision,
                local_files_only=local_only,
            )
        )
        weights_path = Path(
            hf_hub_download(
                repo_id=self.model_id,
                filename="open_clip_model.safetensors",
                revision=self.revision,
                local_files_only=local_only,
            )
        )
        if config_path.parent != weights_path.parent:
            raise RuntimeError("EVA config and weights must resolve to one immutable snapshot")
        from offline.visual_models import _verify_eva_clip_snapshot

        _verify_eva_clip_snapshot(
            config_path=config_path,
            weights_path=weights_path,
            expected_filename="open_clip_model.safetensors",
            expected_sha256="00af04296f09f24dcc69559440a80b7a44daf4855a72827e016067e6e571b851",
            expected_size_bytes=855584096,
            expected_vector_dim=self.dimension,
        )
        # open_clip warns that a load_weights=False model starts randomly initialized.
        # This is intentional for the next strict safetensors load, already SHA-verified.
        root_logger = logging.getLogger()
        previous_disabled = root_logger.disabled
        root_logger.disabled = True
        try:
            model, _, _ = open_clip.create_model_and_transforms(
                f"local-dir:{config_path.parent}", load_weights=False, precision="fp32", device="cpu"
            )
        finally:
            root_logger.disabled = previous_disabled
        load_checkpoint(model, str(weights_path), strict=True, weights_only=True, device="cpu")
        self._model = model.eval().to(self.device)
        self._tokenizer = open_clip.get_tokenizer(f"local-dir:{config_path.parent}")
        self._torch = torch

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self._load()
        tokens = self._tokenizer(list(texts)).to(self.device)
        with self._torch.inference_mode():
            features = self._model.encode_text(tokens, normalize=False)
        result = _normalize(features.detach().float().cpu().numpy())
        if result.shape[1] != self.dimension:
            raise RuntimeError("eva_clip text dimension mismatch")
        return result


class TextEncoderRegistry:
    def __init__(self, encoders: dict[str, TextEncoder] | None = None, *, device: str = "cpu") -> None:
        self.encoders: dict[str, TextEncoder] = encoders or {
            "clip": TransformersTextEncoder("clip", device=device),
            "siglip": TransformersTextEncoder("siglip", device=device),
            "eva_clip": OpenClipEVATextEncoder(device=device),
        }
        self._cache: dict[tuple[str, str], np.ndarray] = {}

    def encode(self, modality: str, texts: Sequence[str]) -> np.ndarray:
        if modality not in self.encoders:
            raise KeyError(f"no text encoder registered for {modality}")
        revision = EXPECTED_VISUAL[modality][2]
        missing: list[str] = []
        for text in texts:
            key = (modality, hashlib.sha256(f"{revision}\n{text}".encode("utf-8")).hexdigest())
            if key not in self._cache:
                missing.append(text)
        if missing:
            encoded = self.encoders[modality].encode(missing)
            for text, vector in zip(missing, encoded):
                key = (modality, hashlib.sha256(f"{revision}\n{text}".encode("utf-8")).hexdigest())
                self._cache[key] = vector
        return np.vstack(
            [
                self._cache[(modality, hashlib.sha256(f"{revision}\n{text}".encode("utf-8")).hexdigest())]
                for text in texts
            ]
        ).astype(np.float32, copy=False)


class WorkerTextEncoderRegistry:
    """Text encoder proxy for the isolated Torch process."""

    def __init__(self, *, device: str = "cpu", client: Any = None) -> None:
        if client is None:
            from .torch_worker_client import get_torch_worker_client

            client = get_torch_worker_client()
        self.client = client
        self.device = device
        self._cache: dict[tuple[str, str], np.ndarray] = {}

    def encode(self, modality: str, texts: Sequence[str]) -> np.ndarray:
        if modality not in EXPECTED_VISUAL:
            raise KeyError(f"no text encoder registered for {modality}")
        revision = EXPECTED_VISUAL[modality][2]
        keys = [
            (modality, hashlib.sha256(f"{revision}\n{text}".encode("utf-8")).hexdigest())
            for text in texts
        ]
        missing_positions = [index for index, key in enumerate(keys) if key not in self._cache]
        if missing_positions:
            missing = [str(texts[index]) for index in missing_positions]
            response = self.client.request("encode", modality=modality, texts=missing, device=self.device)
            encoded = _normalize(np.asarray(response["vectors"], dtype=np.float32))
            if encoded.shape != (len(missing), EXPECTED_VISUAL[modality][0]):
                raise RuntimeError(f"{modality} worker text dimension mismatch")
            for index, vector in zip(missing_positions, encoded):
                self._cache[keys[index]] = vector
        return np.vstack([self._cache[key] for key in keys]).astype(np.float32, copy=False)


def get_text_encoder_registry(
    *,
    device: str = "cpu",
    environment: dict[str, str] | None = None,
) -> TextEncoderRegistry | WorkerTextEncoderRegistry:
    values = os.environ if environment is None else environment
    default_worker = "1" if os.name == "nt" else "0"
    if values.get("AIC_TORCH_WORKER", default_worker) != "0":
        return WorkerTextEncoderRegistry(device=device)
    return TextEncoderRegistry(device=device)
