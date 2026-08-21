"""Lazy local visual question answering with the configured Qwen vision model."""
from __future__ import annotations

import sys
import gc
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from app.config import (
    VQA_DEVICE,
    VQA_MAX_NEW_TOKENS,
    VQA_MAX_PIXELS,
    VQA_MIN_PIXELS,
    VQA_MODEL_NAME,
)


def _normalized_device(device: str) -> str:
    normalized = device.strip().lower()
    if normalized.startswith("gpu"):
        return "cuda" + normalized[3:]
    return normalized


@lru_cache(maxsize=1)
def _load_vqa() -> tuple[Any, Any, str]:
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Vision-language model dependencies are missing. Install "
            "backend/requirements.txt."
        ) from exc

    device = _normalized_device(VQA_DEVICE)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"VQA requested {device}, but PyTorch {torch.__version__} in "
            f"{sys.executable} cannot access CUDA (torch CUDA build: "
            f"{torch.version.cuda}). Start the backend with "
            r".\.venv\Scripts\python.exe -m uvicorn app.main:app --host "
            "127.0.0.1 --port 8000 --reload."
        )
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    processor = AutoProcessor.from_pretrained(
        VQA_MODEL_NAME,
        min_pixels=VQA_MIN_PIXELS,
        max_pixels=VQA_MAX_PIXELS,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        VQA_MODEL_NAME,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()
    return model, processor, device


def release_vqa() -> None:
    """Release the instruction VLM before another 2B vision model uses the GPU."""
    _load_vqa.cache_clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def generate_images(
    image_paths: Sequence[str | Path],
    prompt: str,
    *,
    max_new_tokens: int = VQA_MAX_NEW_TOKENS,
) -> str:
    """Run one deterministic multimodal generation using the shared lazy model."""
    import torch

    model, processor, device = _load_vqa()
    paths = [Path(path) for path in image_paths if Path(path).exists()]
    if not paths:
        return "unknown"

    messages = [
        {
            "role": "user",
            "content": [
                *({"type": "image"} for _ in paths),
                {"type": "text", "text": prompt},
            ],
        }
    ]
    chat_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images = []
    for path in paths:
        with Image.open(path) as source:
            images.append(source.convert("RGB"))
    inputs = processor(
        text=[chat_text],
        images=images,
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    prompt_length = inputs["input_ids"].shape[1]
    decoded = processor.batch_decode(
        generated[:, prompt_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return decoded.strip()


def generate_text(prompt: str, *, max_new_tokens: int) -> str:
    """Run deterministic text-only generation through the shared Qwen model."""
    import torch

    model, processor, device = _load_vqa()
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ]
    chat_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[chat_text],
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    prompt_length = inputs["input_ids"].shape[1]
    return processor.batch_decode(
        generated[:, prompt_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
