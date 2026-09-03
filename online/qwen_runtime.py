"""One lazy Qwen runtime shared by query planning and visual QA.

The local reference machine only has 6 GiB VRAM.  Keeping separate planner and
VQA model objects would load the same 2B checkpoint twice and deterministically
OOM, so all in-process Qwen consumers go through this cache.
"""

from __future__ import annotations

import os
import threading
from typing import Any


DEFAULT_QWEN_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_QWEN_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"


_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}
_LOCK = threading.RLock()


def resolve_qwen_revision(model_id: str) -> str:
    configured = os.environ.get("AIC_QWEN_REVISION", "").strip()
    if configured:
        return configured
    if model_id == DEFAULT_QWEN_MODEL_ID:
        return DEFAULT_QWEN_REVISION
    raise RuntimeError("custom AIC_QWEN_MODEL requires immutable AIC_QWEN_REVISION")


def get_qwen_components(model_id: str) -> tuple[Any, Any]:
    """Return one processor/model pair per model id for the current process."""

    revision = resolve_qwen_revision(model_id)
    cache_key = (model_id, revision)
    with _LOCK:
        cached = _CACHE.get(cache_key)
        if cached is not None:
            return cached

        import torch

        if not torch.cuda.is_available() and os.environ.get("AIC_ALLOW_QWEN_CPU") != "1":
            raise RuntimeError("Qwen local runtime requires CUDA or AIC_ALLOW_QWEN_CPU=1")
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        local_only = os.environ.get("AIC_ALLOW_MODEL_DOWNLOAD") != "1"
        processor = AutoProcessor.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_only,
        )
        if hasattr(processor, "image_processor"):
            processor.image_processor.max_pixels = int(os.environ.get("AIC_QWEN_MAX_PIXELS", "200704"))
        kwargs: dict[str, Any] = {
            "local_files_only": local_only,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }
        if torch.cuda.is_available():
            kwargs["dtype"] = torch.float16
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            revision=revision,
            **kwargs,
        ).eval()
        _CACHE[cache_key] = (model, processor)
        return model, processor


def clear_qwen_runtime_cache() -> None:
    """Testing/operational escape hatch; normal Online runs never call this."""

    with _LOCK:
        _CACHE.clear()
