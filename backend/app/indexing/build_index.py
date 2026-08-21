"""Offline preprocessing: build the global keyframe index + FAISS index.

Matches the REAL on-disk layout (per-video .npy features + map-keyframes CSV giving
the true frame number), not the generic description in the contest PDF. Run after
placing the organizer dataset under data/ :

    python -m app.indexing.build_index
"""
import csv
import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from app.config import (
    FEATURES_DIR,
    FAISS_INDEX_PATH,
    INDEX_DIR,
    KEYFRAME_INDEX_PATH,
    KEYFRAMES_DIR,
    MAP_KEYFRAMES_DIR,
    OBJECTS_CACHE_PATH,
    OBJECTS_DIR,
)


def _read_map_keyframes(video_id: str) -> dict[int, dict[str, float]]:
    """Return {local_idx (n): {frame_id, pts_time, fps}} for one video."""
    csv_path = MAP_KEYFRAMES_DIR / f"{video_id}.csv"
    mapping: dict[int, dict[str, float]] = {}
    if not csv_path.exists():
        return mapping
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                n = int(row["n"])
                mapping[n] = {
                    "frame_id": int(float(row["frame_idx"])),
                    "pts_time": float(row["pts_time"]),
                    "fps": float(row["fps"]),
                }
            except (KeyError, ValueError):
                continue
    return mapping


def _keyframe_path(video_id: str, local_idx: int) -> Path | None:
    folder = KEYFRAMES_DIR / video_id
    for width in (3, 4):
        candidate = folder / f"{local_idx:0{width}d}.jpg"
        if candidate.exists():
            return candidate
    return None


def build_keyframe_index_and_features() -> tuple[list[dict[str, Any]], np.ndarray]:
    """Iterate every video's .npy feature file, align it with map-keyframes, and
    return (flat keyframe rows, stacked feature matrix) in matching order."""
    rows: list[dict[str, Any]] = []
    feature_chunks: list[np.ndarray] = []

    feature_files = sorted(FEATURES_DIR.glob("*.npy"))
    global_idx = 0
    for feat_path in feature_files:
        video_id = feat_path.stem
        feats = np.load(feat_path)
        n_kf = feats.shape[0]

        frame_map = _read_map_keyframes(video_id)

        valid_local_idxs: list[int] = []
        for local_idx in range(1, n_kf + 1):
            kf_path = _keyframe_path(video_id, local_idx)
            if kf_path is None:
                # No matching jpg on disk — skip this vector to keep index consistent.
                continue
            info = frame_map.get(local_idx, {})
            rows.append(
                {
                    "global_idx": global_idx,
                    "video_id": video_id,
                    "local_idx": local_idx,
                    "frame_id": info.get("frame_id", local_idx),
                    "pts_time": info.get("pts_time", 0.0),
                    "keyframe_path": str(kf_path),
                }
            )
            valid_local_idxs.append(local_idx - 1)
            global_idx += 1

        if valid_local_idxs:
            feature_chunks.append(feats[np.asarray(valid_local_idxs, dtype=np.int64)])

    if not feature_chunks:
        return rows, np.zeros((0, 512), dtype=np.float32)

    matrix = np.concatenate(feature_chunks, axis=0).astype(np.float32)
    return rows, matrix


def _extract_labels(obj_json: dict[str, Any]) -> list[str]:
    """The provided object JSON uses TensorFlow Object Detection API style keys:
    detection_class_entities (human-readable label per detection), detection_scores.
    """
    labels: list[str] = []
    entities = obj_json.get("detection_class_entities")
    if isinstance(entities, list):
        labels = [str(e).lower() for e in entities]
    return labels


def _extract_scores(obj_json: dict[str, Any]) -> list[float]:
    scores = obj_json.get("detection_scores")
    if not isinstance(scores, list):
        return []
    out = []
    for s in scores:
        try:
            out.append(float(s))
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def build_objects_cache(keyframe_index: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not OBJECTS_DIR.exists():
        return cache

    for row in keyframe_index:
        video_id = row["video_id"]
        local_idx = row["local_idx"]
        json_path = None
        for width in (3, 4):
            candidate = OBJECTS_DIR / video_id / f"{local_idx:0{width}d}.json"
            if candidate.exists():
                json_path = candidate
                break
        if json_path is None:
            continue
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        labels = _extract_labels(raw)
        scores = _extract_scores(raw)
        # Keep only reasonably confident detections to reduce noise in re-rank.
        kept = [lbl for lbl, sc in zip(labels, scores) if sc >= 0.3] or labels

        key = f"{video_id}:{local_idx}"
        counts: dict[str, int] = {}
        for lbl in kept:
            counts[lbl] = counts.get(lbl, 0) + 1
        cache[key] = {"labels": kept, "counts": counts}
    return cache


def build_faiss_index(features: np.ndarray) -> faiss.Index:
    feats = features.astype(np.float32).copy()
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    feats /= norms
    index = faiss.IndexFlatIP(feats.shape[1])
    index.add(feats)
    return index


def main() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print("Scanning data/features/*.npy and aligning with map-keyframes...")
    keyframe_index, features = build_keyframe_index_and_features()
    if not keyframe_index:
        raise FileNotFoundError(
            "No keyframes indexed. Check that data/features, data/keyframes and "
            "data/map-keyframes are populated."
        )
    print(f"Indexed {len(keyframe_index)} keyframes across "
          f"{len(set(r['video_id'] for r in keyframe_index))} videos.")

    KEYFRAME_INDEX_PATH.write_text(json.dumps(keyframe_index, ensure_ascii=False), encoding="utf-8")

    print("Building FAISS index (cosine similarity via normalized inner product)...")
    index = build_faiss_index(features)
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    print("Parsing Objects JSON (Faster R-CNN / OpenImages)...")
    objects_cache = build_objects_cache(keyframe_index)
    OBJECTS_CACHE_PATH.write_text(json.dumps(objects_cache, ensure_ascii=False), encoding="utf-8")
    print(f"Objects cache built for {len(objects_cache)} keyframes.")

    print("Index build complete.")


if __name__ == "__main__":
    main()
