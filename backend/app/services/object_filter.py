import json
from functools import lru_cache

from app.config import OBJECTS_CACHE_PATH


@lru_cache(maxsize=1)
def _load_objects_cache() -> dict:
    if not OBJECTS_CACHE_PATH.exists():
        return {}
    return json.loads(OBJECTS_CACHE_PATH.read_text(encoding="utf-8"))


def object_match_score(video_id: str, local_idx: int, keywords: list[str]) -> float:
    """Fraction of query object keywords found among the detected labels for this
    keyframe. Used as a re-rank boost signal on top of the CLIP similarity score."""
    if not keywords:
        return 0.0
    cache = _load_objects_cache()
    row = cache.get(f"{video_id}:{local_idx}", {})
    labels = row.get("labels", [])
    if not labels:
        return 0.0
    joined = " ".join(labels)
    matched = sum(1 for kw in keywords if kw.lower() in joined)
    return matched / max(len(keywords), 1)


def object_count(video_id: str, local_idx: int, label_hint: str) -> int:
    """Number of detected instances whose label contains `label_hint` — used by the
    Q&A pipeline to answer "how many ..." style questions."""
    cache = _load_objects_cache()
    row = cache.get(f"{video_id}:{local_idx}", {})
    counts = row.get("counts", {})
    total = 0
    for label, cnt in counts.items():
        if label_hint.lower() in label.lower():
            total += int(cnt)
    return total
