"""Lazy Kaggle GPU adapters for the pinned visual embedding candidates."""

from __future__ import annotations

import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_CONFIG = _REPOSITORY_ROOT / "configs" / "visual_embedding_models.json"


def load_model_config(modality: str, *, path: Path = DEFAULT_MODEL_CONFIG) -> dict[str, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise RuntimeError("unsupported visual embedding model config schema")
    row = payload.get(modality)
    if not isinstance(row, dict):
        raise ValueError(f"unknown visual embedding modality: {modality}")
    if row.get("status") == "blocked_model_selection":
        raise RuntimeError(
            f"{modality} is fail-closed: {row.get('reason', 'model selection is unresolved')}"
        )
    required = ("backend", "model_id", "revision")
    if any(not isinstance(row.get(key), str) or not row[key].strip() for key in required):
        raise RuntimeError(f"visual model config for {modality} is incomplete")
    return {key: str(row[key]) for key in required}


class TransformersImageEncoder:
    """Load one CLIP/SigLIP model lazily and expose image-only embeddings."""

    def __init__(
        self,
        *,
        modality: str,
        model_id: str,
        model_revision: str,
        device: str = "cuda",
    ) -> None:
        if modality not in {"clip", "siglip"}:
            raise ValueError("TransformersImageEncoder only supports clip or siglip")
        if device != "cuda":
            raise ValueError("visual embedding production adapter requires explicit device='cuda'")
        self.modality = modality
        self.model_id = model_id
        self.model_revision = model_revision
        self.device = device
        self._torch: Any | None = None
        self._processor: Any | None = None
        self._model: Any | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, CLIPModel, SiglipModel
        except ImportError as error:
            raise RuntimeError(
                "Kaggle visual dependencies are missing; install requirements/kaggle-gpu.txt"
            ) from error
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        model_class = CLIPModel if self.modality == "clip" else SiglipModel
        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            revision=self.model_revision,
        )
        self._model = model_class.from_pretrained(
            self.model_id,
            revision=self.model_revision,
        )
        self._model.eval().to(self.device)
        self._torch = torch

    @property
    def runtime_metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {
            "device": self.device,
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "transformers": importlib.metadata.version("transformers"),
            "torch": importlib.metadata.version("torch"),
        }
        if self._torch is not None and self._torch.cuda.is_available():
            metadata.update(
                {
                    "cuda_runtime": self._torch.version.cuda,
                    "gpu_name": self._torch.cuda.get_device_name(0),
                    "peak_cuda_memory_bytes": self._torch.cuda.max_memory_allocated(0),
                }
            )
        return metadata

    def encode(self, image_paths: Sequence[Path]) -> np.ndarray:
        self._load()
        assert self._processor is not None
        assert self._model is not None
        assert self._torch is not None
        images: list[Image.Image] = []
        try:
            for path in image_paths:
                with Image.open(path) as source:
                    images.append(source.convert("RGB"))
            inputs = self._processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            with self._torch.inference_mode(), self._torch.autocast(
                device_type="cuda", dtype=self._torch.float16
            ):
                model_output = self._model.get_image_features(pixel_values=pixel_values)
                # Transformers 5.x returns BaseModelOutputWithPooling for both
                # CLIP and SigLIP.  Older versions returned the tensor directly;
                # keep the explicit branch so a future API change fails closed.
                features = (
                    model_output.pooler_output
                    if hasattr(model_output, "pooler_output")
                    else model_output
                )
            if not isinstance(features, self._torch.Tensor):
                raise RuntimeError(
                    "model get_image_features() did not expose a Tensor pooler_output"
                )
            return features.detach().float().cpu().numpy()
        finally:
            for image in images:
                image.close()


def create_visual_encoder(
    modality: str,
    *,
    config_path: Path = DEFAULT_MODEL_CONFIG,
    device: str = "cuda",
) -> TransformersImageEncoder:
    row = load_model_config(modality, path=config_path)
    backend = row["backend"]
    expected_backend = {
        "clip": "transformers_clip",
        "siglip": "transformers_siglip",
    }.get(modality)
    if backend != expected_backend:
        raise RuntimeError(
            f"visual encoder backend mismatch for {modality}: {backend}"
        )
    return TransformersImageEncoder(
        modality=modality,
        model_id=row["model_id"],
        model_revision=row["revision"],
        device=device,
    )
