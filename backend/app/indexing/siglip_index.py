"""Checkpointable SigLIP2 side-index builder over all organizer keyframes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from PIL import Image

from app.config import (
    INDEX_DIR,
    KEYFRAME_INDEX_PATH,
    SIGLIP_DEVICE,
    SIGLIP_FEATURES_PATH,
    SIGLIP_INDEX_PATH,
    SIGLIP_MODEL_NAME,
    SIGLIP_STATE_PATH,
)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _signature(total: int) -> str:
    return f"siglip2:{SIGLIP_MODEL_NAME}:keyframes={total}:normalized=1"


def _device(value: str) -> str:
    normalized = value.strip().lower()
    return "cuda" + normalized[3:] if normalized.startswith("gpu") else normalized


def _load_model(device: str):
    import torch
    from transformers import AutoModel, AutoProcessor

    normalized = _device(device)
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"SigLIP2 requested {normalized}, but CUDA is unavailable.")
    dtype = torch.float16 if normalized.startswith("cuda") else torch.float32
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME)
    model = AutoModel.from_pretrained(SIGLIP_MODEL_NAME, dtype=dtype)
    model.to(normalized).eval()
    return model, processor, normalized


def _image_features(model, processor, paths: list[str], device: str) -> np.ndarray:
    import torch

    images = []
    for path in paths:
        with Image.open(path) as source:
            images.append(source.convert("RGB"))
    inputs = processor(images=images, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        features = model.get_image_features(**inputs)
    if hasattr(features, "pooler_output"):
        features = features.pooler_output
    features = torch.nn.functional.normalize(features.float(), dim=-1)
    return features.cpu().numpy().astype(np.float16)


def _publish_faiss(features_path: Path, index_path: Path) -> None:
    features = np.load(features_path, mmap_mode="r")
    index = faiss.IndexFlatIP(int(features.shape[1]))
    for start in range(0, len(features), 4096):
        chunk = np.asarray(features[start : start + 4096], dtype=np.float32)
        faiss.normalize_L2(chunk)
        index.add(chunk)
    temporary = index_path.with_name(f"{index_path.name}.tmp")
    faiss.write_index(index, str(temporary))
    temporary.replace(index_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default=SIGLIP_DEVICE)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be a positive integer")

    rows = json.loads(KEYFRAME_INDEX_PATH.read_text(encoding="utf-8"))
    total = len(rows)
    signature = _signature(total)
    state = {}
    if SIGLIP_STATE_PATH.exists():
        state = json.loads(SIGLIP_STATE_PATH.read_text(encoding="utf-8"))
    if state and state.get("signature") != signature:
        raise RuntimeError(
            "SigLIP checkpoint belongs to another model/index. Move the old "
            "siglip2 state/features aside before rebuilding."
        )

    model, processor, device = _load_model(args.device)
    start = int(state.get("next_index", 0))
    stop = min(total, start + args.limit) if args.limit is not None else total
    if start >= total:
        if not SIGLIP_INDEX_PATH.exists() and SIGLIP_FEATURES_PATH.exists():
            _publish_faiss(SIGLIP_FEATURES_PATH, SIGLIP_INDEX_PATH)
        return 0

    first_stop = min(stop, start + max(1, args.batch_size))
    first = _image_features(
        model,
        processor,
        [str(rows[index]["keyframe_path"]) for index in range(start, first_stop)],
        device,
    )
    dimension = int(first.shape[1])
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    if start == 0 or not SIGLIP_FEATURES_PATH.exists():
        features = np.lib.format.open_memmap(
            SIGLIP_FEATURES_PATH,
            mode="w+",
            dtype=np.float16,
            shape=(total, dimension),
        )
    else:
        features = np.load(SIGLIP_FEATURES_PATH, mmap_mode="r+")
        if features.shape != (total, dimension):
            raise RuntimeError("SigLIP feature checkpoint shape does not match the model.")
    features[start:first_stop] = first
    next_index = first_stop

    while next_index < stop:
        batch_stop = min(stop, next_index + max(1, args.batch_size))
        paths = [str(rows[index]["keyframe_path"]) for index in range(next_index, batch_stop)]
        features[next_index:batch_stop] = _image_features(
            model, processor, paths, device
        )
        next_index = batch_stop
        if next_index % max(1, args.checkpoint_every) < max(1, args.batch_size):
            features.flush()
            _atomic_json(
                SIGLIP_STATE_PATH,
                {
                    "signature": signature,
                    "next_index": next_index,
                    "total": total,
                    "dimension": dimension,
                    "complete": False,
                },
            )
            print(f"SigLIP2 progress: {next_index}/{total}", flush=True)

    features.flush()
    complete = next_index >= total
    _atomic_json(
        SIGLIP_STATE_PATH,
        {
            "signature": signature,
            "next_index": next_index,
            "total": total,
            "dimension": dimension,
            "complete": complete,
        },
    )
    if complete:
        _publish_faiss(SIGLIP_FEATURES_PATH, SIGLIP_INDEX_PATH)
        print(f"Published {SIGLIP_INDEX_PATH} with {total} vectors.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
