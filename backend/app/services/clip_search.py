"""CLIP text -> keyframe image retrieval.

English uses the original CLIP text tower. Vietnamese uses the multilingual CLIP
student distilled into the same 512-dimensional image space. Image features remain
the organizer-provided clip-ViT-B-32 vectors; no image re-index is required.
"""
import json
from functools import lru_cache
from typing import Sequence

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import (
    CLIP_DEVICE,
    CLIP_MODEL_NAME,
    FAISS_INDEX_PATH,
    KEYFRAME_INDEX_PATH,
    MULTILINGUAL_CLIP_MODEL_NAME,
    TOP_K_CANDIDATES,
)
from app.services.query_processing import is_vietnamese_text


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name, device=CLIP_DEVICE)


@lru_cache(maxsize=1)
def _load_index() -> faiss.Index:
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError("Missing clip.faiss. Run: python -m app.indexing.build_index")
    return faiss.read_index(str(FAISS_INDEX_PATH))


@lru_cache(maxsize=1)
def _load_keyframe_index() -> list[dict]:
    if not KEYFRAME_INDEX_PATH.exists():
        raise FileNotFoundError("Missing keyframe_index.json. Run: python -m app.indexing.build_index")
    return json.loads(KEYFRAME_INDEX_PATH.read_text(encoding="utf-8"))


def search_text(
    query: str,
    top_k: int = TOP_K_CANDIDATES,
    multilingual: bool | None = None,
) -> list[dict]:
    return search_text_batch(
        [query], top_k=top_k, multilingual=multilingual
    )[0]


def search_text_batch(
    queries: Sequence[str],
    top_k: int = TOP_K_CANDIDATES,
    multilingual: bool | None = None,
) -> list[list[dict]]:
    if not queries:
        return []
    index = _load_index()
    keyframes = _load_keyframe_index()

    if multilingual is None:
        language_flags = {is_vietnamese_text(query) for query in queries}
        if len(language_flags) != 1:
            return [search_text(query, top_k=top_k) for query in queries]
        multilingual = language_flags.pop()

    model_name = MULTILINGUAL_CLIP_MODEL_NAME if multilingual else CLIP_MODEL_NAME
    model = _load_model(model_name)
    embeddings = model.encode(
        list(queries), normalize_embeddings=True
    ).astype(np.float32)
    scores, indices = index.search(embeddings, min(top_k, index.ntotal))

    batches: list[list[dict]] = []
    for query_scores, query_indices in zip(scores, indices):
        rows: list[dict] = []
        for score, index_position in zip(query_scores.tolist(), query_indices.tolist()):
            if index_position < 0 or index_position >= len(keyframes):
                continue
            row = keyframes[index_position]
            rows.append(
                {
                    "video_id": row["video_id"],
                    "local_idx": row["local_idx"],
                    "frame_id": row["frame_id"],
                    "keyframe_path": row["keyframe_path"],
                    "score": float(score),
                }
            )
        batches.append(rows)
    return batches
