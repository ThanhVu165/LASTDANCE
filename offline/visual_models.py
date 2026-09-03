"""Lazy Kaggle GPU adapters for the pinned visual embedding candidates."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_CONFIG = _REPOSITORY_ROOT / "configs" / "visual_embedding_models.json"


def load_model_config(modality: str, *, path: Path = DEFAULT_MODEL_CONFIG) -> dict[str, Any]:
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
    if modality == "eva_clip":
        string_fields = ("weights_filename", "weights_sha256", "license")
        integer_fields = ("weights_size_bytes", "expected_vector_dim")
        if any(
            not isinstance(row.get(key), str) or not row[key].strip()
            for key in string_fields
        ) or any(type(row.get(key)) is not int or int(row[key]) <= 0 for key in integer_fields):
            raise RuntimeError("visual model config for eva_clip is incomplete")
        if (
            row["weights_filename"] != "open_clip_model.safetensors"
            or len(row["weights_sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in row["weights_sha256"].lower()
            )
        ):
            raise RuntimeError("eva_clip must pin one valid official safetensors checkpoint")
    return dict(row)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_eva_clip_snapshot(
    *,
    config_path: Path,
    weights_path: Path,
    expected_filename: str,
    expected_sha256: str,
    expected_size_bytes: int,
    expected_vector_dim: int,
) -> None:
    """Verify the pinned OpenCLIP config and safetensors before model construction."""

    config_path = Path(config_path)
    weights_path = Path(weights_path)
    if (
        not config_path.is_file()
        or config_path.name != "open_clip_config.json"
        or not weights_path.is_file()
        or weights_path.name != expected_filename
        or weights_path.suffix != ".safetensors"
    ):
        raise RuntimeError("eva_clip snapshot is missing its pinned safe config/weights")
    if weights_path.stat().st_size != expected_size_bytes:
        raise RuntimeError("eva_clip safetensors size does not match the pinned registry")
    if _sha256_file(weights_path) != expected_sha256.lower():
        raise RuntimeError("eva_clip safetensors SHA-256 does not match the pinned registry")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    model_cfg = payload.get("model_cfg")
    if (
        not isinstance(model_cfg, dict)
        or type(model_cfg.get("embed_dim")) is not int
        or int(model_cfg["embed_dim"]) != expected_vector_dim
    ):
        raise RuntimeError("eva_clip OpenCLIP config embed_dim does not match the registry")


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


class OpenClipEVAImageEncoder:
    """Pinned EVA02-CLIP adapter that never loads a pickle checkpoint."""

    def __init__(
        self,
        *,
        model_id: str,
        model_revision: str,
        weights_filename: str,
        weights_sha256: str,
        weights_size_bytes: int,
        expected_vector_dim: int,
        device: str = "cuda",
    ) -> None:
        if device != "cuda":
            raise ValueError("visual embedding production adapter requires explicit device='cuda'")
        if weights_filename != "open_clip_model.safetensors":
            raise ValueError("eva_clip adapter accepts only open_clip_model.safetensors")
        self.modality = "eva_clip"
        self.model_id = model_id
        self.model_revision = model_revision
        self.weights_filename = weights_filename
        self.weights_sha256 = weights_sha256.lower()
        self.weights_size_bytes = int(weights_size_bytes)
        self.expected_vector_dim = int(expected_vector_dim)
        self.device = device
        self._torch: Any | None = None
        self._preprocess: Any | None = None
        self._model: Any | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import open_clip
            import torch
            from huggingface_hub import hf_hub_download
            from open_clip.factory import load_checkpoint
        except ImportError as error:
            raise RuntimeError(
                "Kaggle EVA-CLIP dependencies are missing; install requirements/kaggle-gpu.txt"
            ) from error
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

        config_path = Path(
            hf_hub_download(
                repo_id=self.model_id,
                filename="open_clip_config.json",
                revision=self.model_revision,
            )
        )
        weights_path = Path(
            hf_hub_download(
                repo_id=self.model_id,
                filename=self.weights_filename,
                revision=self.model_revision,
            )
        )
        _verify_eva_clip_snapshot(
            config_path=config_path,
            weights_path=weights_path,
            expected_filename=self.weights_filename,
            expected_sha256=self.weights_sha256,
            expected_size_bytes=self.weights_size_bytes,
            expected_vector_dim=self.expected_vector_dim,
        )
        if config_path.parent != weights_path.parent:
            raise RuntimeError("eva_clip config and safetensors did not resolve to one snapshot")

        # Build only the architecture and preprocessing from the pinned local snapshot.
        # Loading is then explicit from the already-verified .safetensors path, so OpenCLIP
        # cannot fall back to the pickle .bin that also exists in the upstream repository.
        model, _, preprocess = open_clip.create_model_and_transforms(
            f"local-dir:{config_path.parent}",
            load_weights=False,
            precision="fp32",
            device="cpu",
        )
        load_checkpoint(
            model,
            str(weights_path),
            strict=True,
            weights_only=True,
            device="cpu",
        )
        self._model = model.eval().to(self.device)
        self._preprocess = preprocess
        self._torch = torch

    @property
    def runtime_metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {
            "device": self.device,
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "transformers": importlib.metadata.version("transformers"),
            "open_clip_torch": importlib.metadata.version("open_clip_torch"),
            "timm": importlib.metadata.version("timm"),
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
        assert self._preprocess is not None
        assert self._model is not None
        assert self._torch is not None
        images: list[Image.Image] = []
        try:
            for path in image_paths:
                with Image.open(path) as source:
                    images.append(source.convert("RGB"))
            pixel_values = self._torch.stack(
                [self._preprocess(image) for image in images]
            ).to(self.device)
            with self._torch.inference_mode(), self._torch.autocast(
                device_type="cuda", dtype=self._torch.float16
            ):
                features = self._model.encode_image(pixel_values, normalize=False)
            if (
                not isinstance(features, self._torch.Tensor)
                or features.ndim != 2
                or int(features.shape[1]) != self.expected_vector_dim
            ):
                raise RuntimeError(
                    "eva_clip runtime vector dimension does not match the pinned config"
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
) -> TransformersImageEncoder | OpenClipEVAImageEncoder:
    row = load_model_config(modality, path=config_path)
    backend = row["backend"]
    expected_backend = {
        "clip": "transformers_clip",
        "siglip": "transformers_siglip",
        "eva_clip": "open_clip_eva",
    }.get(modality)
    if backend != expected_backend:
        raise RuntimeError(
            f"visual encoder backend mismatch for {modality}: {backend}"
        )
    if modality == "eva_clip":
        return OpenClipEVAImageEncoder(
            model_id=str(row["model_id"]),
            model_revision=str(row["revision"]),
            weights_filename=str(row["weights_filename"]),
            weights_sha256=str(row["weights_sha256"]),
            weights_size_bytes=int(row["weights_size_bytes"]),
            expected_vector_dim=int(row["expected_vector_dim"]),
            device=device,
        )
    return TransformersImageEncoder(
        modality=modality,
        model_id=str(row["model_id"]),
        model_revision=str(row["revision"]),
        device=device,
    )
