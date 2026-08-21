"""Optional side-index retrieval; unavailable/incomplete indexes are ignored."""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Sequence

import faiss
import numpy as np

from app.config import (
    KEYFRAME_INDEX_PATH,
    SIDE_RETRIEVAL_TOP_K,
    SIGLIP_INDEX_PATH,
    SIGLIP_MODEL_NAME,
    SIGLIP_QUERY_DEVICE,
    SIGLIP_STATE_PATH,
    VIDEO_WINDOW_INDEX_PATH,
    VIDEO_WINDOW_METADATA_PATH,
    VIDEO_WINDOW_STATE_PATH,
)


def _complete(state_path) -> bool:
    if not state_path.exists():
        return False
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(state.get("complete"))


@lru_cache(maxsize=1)
def _keyframes() -> list[dict]:
    return json.loads(KEYFRAME_INDEX_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _siglip_runtime():
    import torch
    from transformers import AutoModel, AutoProcessor

    device = SIGLIP_QUERY_DEVICE
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModel.from_pretrained(SIGLIP_MODEL_NAME, dtype=dtype)
    model.to(device).eval()
    return model, AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME), faiss.read_index(
        str(SIGLIP_INDEX_PATH)
    )


def search_siglip_batch(
    queries: Sequence[str], top_k: int = SIDE_RETRIEVAL_TOP_K
) -> list[list[dict]]:
    if not queries or not SIGLIP_INDEX_PATH.exists() or not _complete(SIGLIP_STATE_PATH):
        return [[] for _ in queries]
    try:
        import torch

        model, processor, index = _siglip_runtime()
        inputs = processor(text=list(queries), padding=True, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            features = model.get_text_features(**inputs)
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        embeddings = torch.nn.functional.normalize(features.float(), dim=-1).cpu().numpy()
        scores, indices = index.search(embeddings.astype(np.float32), min(top_k, index.ntotal))
        metadata = _keyframes()
        return [
            [
                {
                    "video_id": metadata[position]["video_id"],
                    "local_idx": metadata[position]["local_idx"],
                    "frame_id": metadata[position]["frame_id"],
                    "keyframe_path": metadata[position]["keyframe_path"],
                    "score": float(score),
                    "retriever": "siglip2",
                }
                for score, position in zip(row_scores.tolist(), row_indices.tolist())
                if 0 <= position < len(metadata)
            ]
            for row_scores, row_indices in zip(scores, indices)
        ]
    except (RuntimeError, OSError, ValueError, ImportError):
        return [[] for _ in queries]


@lru_cache(maxsize=1)
def _window_runtime():
    from sentence_transformers import SentenceTransformer

    from app.config import (
        VIDEO_WINDOW_DEVICE,
        VIDEO_WINDOW_EMBEDDING_DIM,
        VIDEO_WINDOW_EMBEDDING_MODEL_NAME,
        VIDEO_WINDOW_LOCAL_FILES_ONLY,
    )

    model = SentenceTransformer(
        VIDEO_WINDOW_EMBEDDING_MODEL_NAME,
        device=VIDEO_WINDOW_DEVICE,
        truncate_dim=VIDEO_WINDOW_EMBEDDING_DIM,
        trust_remote_code=True,
        local_files_only=VIDEO_WINDOW_LOCAL_FILES_ONLY,
    )
    index = faiss.read_index(str(VIDEO_WINDOW_INDEX_PATH))
    metadata = json.loads(VIDEO_WINDOW_METADATA_PATH.read_text(encoding="utf-8"))
    return model, index, metadata


def search_video_windows_batch(
    queries: Sequence[str], top_k: int = SIDE_RETRIEVAL_TOP_K
) -> list[list[dict]]:
    if (
        not queries
        or not VIDEO_WINDOW_INDEX_PATH.exists()
        or not VIDEO_WINDOW_METADATA_PATH.exists()
        or not _complete(VIDEO_WINDOW_STATE_PATH)
    ):
        return [[] for _ in queries]
    try:
        # The planner/instruction VLM and the 2B embedding model cannot coexist
        # safely on a 6 GiB card. Query the window index in a bounded phase.
        from app.services.visual_qa import release_vqa

        release_vqa()
        model, index, metadata = _window_runtime()
        embeddings = model.encode(
            list(queries),
            prompt="Retrieve video windows relevant to the complete visual query.",
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        scores, indices = index.search(embeddings, min(top_k, index.ntotal))
        results = [
            [
                {
                    "video_id": metadata[position]["video_id"],
                    "local_idx": metadata[position]["center_local_idx"],
                    "frame_id": metadata[position]["center_frame_id"],
                    "keyframe_path": metadata[position]["center_keyframe_path"],
                    "window_local_idxs": metadata[position]["local_idxs"],
                    "score": float(score),
                    "retriever": "qwen_video_window",
                }
                for score, position in zip(row_scores.tolist(), row_indices.tolist())
                if 0 <= position < len(metadata)
            ]
            for row_scores, row_indices in zip(scores, indices)
        ]
        release_side_search_models()
        return results
    except (RuntimeError, OSError, ValueError, ImportError):
        release_side_search_models()
        return [[] for _ in queries]


def release_side_search_models() -> None:
    import gc

    _window_runtime.cache_clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
