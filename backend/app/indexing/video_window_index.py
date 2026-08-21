"""Checkpointable Qwen3-VL embedding index over overlapping video windows."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np

from app.config import (
    INDEX_DIR,
    KEYFRAME_INDEX_PATH,
    VIDEO_WINDOW_DEVICE,
    VIDEO_WINDOW_EMBEDDING_DIM,
    VIDEO_WINDOW_EMBEDDING_MODEL_NAME,
    VIDEO_WINDOW_FEATURES_PATH,
    VIDEO_WINDOW_INDEX_PATH,
    VIDEO_WINDOW_LOCAL_FILES_ONLY,
    VIDEO_WINDOW_METADATA_PATH,
    VIDEO_WINDOW_SIZE,
    VIDEO_WINDOW_STATE_PATH,
    VIDEO_WINDOW_STRIDE,
    VIDEO_WINDOW_TOTAL_PIXELS,
)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _windows(rows: list[dict], size: int, stride: int) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["video_id"]].append(row)
    windows: list[dict] = []
    for video_id in sorted(grouped):
        video_rows = sorted(grouped[video_id], key=lambda row: int(row["local_idx"]))
        if not video_rows:
            continue
        starts = list(range(0, max(1, len(video_rows) - size + 1), stride))
        last_start = max(0, len(video_rows) - size)
        if last_start not in starts:
            starts.append(last_start)
        for start in sorted(set(starts)):
            members = video_rows[start : start + size]
            if not members:
                continue
            center = members[len(members) // 2]
            windows.append(
                {
                    "video_id": video_id,
                    "local_idxs": [int(row["local_idx"]) for row in members],
                    "frame_ids": [int(row["frame_id"]) for row in members],
                    "pts_times": [
                        float(row["pts_time"])
                        if row.get("pts_time") is not None
                        else None
                        for row in members
                    ],
                    "keyframe_paths": [str(row["keyframe_path"]) for row in members],
                    "center_local_idx": int(center["local_idx"]),
                    "center_frame_id": int(center["frame_id"]),
                    "center_keyframe_path": str(center["keyframe_path"]),
                }
            )
    return windows


def _signature(
    keyframe_count: int,
    window_count: int,
    size: int,
    stride: int,
    total_pixels: int,
) -> str:
    return (
        f"qwen-window:{VIDEO_WINDOW_EMBEDDING_MODEL_NAME}:dim="
        f"{VIDEO_WINDOW_EMBEDDING_DIM}:keyframes={keyframe_count}:"
        f"windows={window_count}:size={size}:stride={stride}:"
        f"pixels={total_pixels}:timestamps=relative-ms:normalized=1"
    )


def _load_model(device: str):
    import torch
    from sentence_transformers import SentenceTransformer

    normalized = device.strip().lower().replace("gpu", "cuda", 1)
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Video-window embedding requested {normalized}, but CUDA is unavailable.")
    return SentenceTransformer(
        VIDEO_WINDOW_EMBEDDING_MODEL_NAME,
        device=normalized,
        truncate_dim=VIDEO_WINDOW_EMBEDDING_DIM,
        trust_remote_code=True,
        local_files_only=VIDEO_WINDOW_LOCAL_FILES_ONLY,
        model_kwargs={
            "dtype": torch.float16 if normalized.startswith("cuda") else torch.float32,
            "attn_implementation": "sdpa",
            "low_cpu_mem_usage": True,
        },
    )


def _encode(
    model,
    windows: list[dict],
    batch_size: int,
    total_pixels: int,
) -> np.ndarray:
    inputs = []
    for window in windows:
        raw_times = window.get("pts_times") or []
        if raw_times and all(value is not None for value in raw_times):
            origin = float(raw_times[0])
            frame_indices = [
                max(0, int(round((float(value) - origin) * 1000.0)))
                for value in raw_times
            ]
            metadata = {
                "fps": 1000.0,
                "total_num_frames": max(frame_indices[-1] + 1, len(frame_indices)),
                "duration": max(frame_indices[-1] / 1000.0, 0.001),
                "frames_indices": frame_indices,
            }
            video = {
                "array": window["keyframe_paths"],
                "video_metadata": metadata,
            }
        else:
            video = window["keyframe_paths"]
        inputs.append({"video": video})
    return np.asarray(
        model.encode(
            inputs,
            batch_size=max(1, batch_size),
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
            processing_kwargs={
                "video": {
                    "do_sample_frames": False,
                    "size": {
                        "shortest_edge": total_pixels,
                        "longest_edge": total_pixels,
                    },
                }
            },
        ),
        dtype=np.float16,
    )


def _publish(features_path: Path, index_path: Path) -> None:
    features = np.load(features_path, mmap_mode="r")
    index = faiss.IndexFlatIP(int(features.shape[1]))
    for start in range(0, len(features), 2048):
        chunk = np.asarray(features[start : start + 2048], dtype=np.float32)
        faiss.normalize_L2(chunk)
        index.add(chunk)
    temporary = index_path.with_name(f"{index_path.name}.tmp")
    faiss.write_index(index, str(temporary))
    temporary.replace(index_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default=VIDEO_WINDOW_DEVICE)
    parser.add_argument("--window-size", type=int, default=VIDEO_WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=VIDEO_WINDOW_STRIDE)
    parser.add_argument("--total-pixels", type=int, default=VIDEO_WINDOW_TOTAL_PIXELS)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be a positive integer")
    if args.window_size <= 0 or args.stride <= 0 or args.total_pixels <= 0:
        raise ValueError("window size, stride and total pixels must be positive")

    keyframes = json.loads(KEYFRAME_INDEX_PATH.read_text(encoding="utf-8"))
    windows = _windows(keyframes, args.window_size, args.stride)
    signature = _signature(
        len(keyframes),
        len(windows),
        args.window_size,
        args.stride,
        args.total_pixels,
    )
    state = {}
    if VIDEO_WINDOW_STATE_PATH.exists():
        state = json.loads(VIDEO_WINDOW_STATE_PATH.read_text(encoding="utf-8"))
    if state and state.get("signature") != signature:
        raise RuntimeError(
            "Video-window checkpoint belongs to another configuration. Move the "
            "old state/features aside before rebuilding."
        )
    start = int(state.get("next_index", 0))
    if start >= len(windows):
        if not VIDEO_WINDOW_METADATA_PATH.exists():
            _atomic_json(VIDEO_WINDOW_METADATA_PATH, windows)
        if not VIDEO_WINDOW_INDEX_PATH.exists() and VIDEO_WINDOW_FEATURES_PATH.exists():
            _publish(VIDEO_WINDOW_FEATURES_PATH, VIDEO_WINDOW_INDEX_PATH)
        return 0
    stop = (
        min(len(windows), start + args.limit)
        if args.limit is not None
        else len(windows)
    )

    model = _load_model(args.device)
    first_stop = min(stop, start + max(1, args.batch_size))
    first = _encode(
        model,
        windows[start:first_stop],
        args.batch_size,
        args.total_pixels,
    )
    dimension = int(first.shape[1])
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    if start == 0 or not VIDEO_WINDOW_FEATURES_PATH.exists():
        features = np.lib.format.open_memmap(
            VIDEO_WINDOW_FEATURES_PATH,
            mode="w+",
            dtype=np.float16,
            shape=(len(windows), dimension),
        )
    else:
        features = np.load(VIDEO_WINDOW_FEATURES_PATH, mmap_mode="r+")
        if features.shape != (len(windows), dimension):
            raise RuntimeError("Video-window feature checkpoint shape is invalid.")
    features[start:first_stop] = first
    next_index = first_stop
    checkpoint_every = max(1, args.checkpoint_every)
    while next_index < stop:
        batch_stop = min(stop, next_index + max(1, args.batch_size))
        features[next_index:batch_stop] = _encode(
            model,
            windows[next_index:batch_stop],
            args.batch_size,
            args.total_pixels,
        )
        next_index = batch_stop
        if next_index % checkpoint_every < max(1, args.batch_size):
            features.flush()
            _atomic_json(
                VIDEO_WINDOW_STATE_PATH,
                {
                    "signature": signature,
                    "next_index": next_index,
                    "total": len(windows),
                    "dimension": dimension,
                    "complete": False,
                },
            )
            print(f"Qwen video-window progress: {next_index}/{len(windows)}", flush=True)

    features.flush()
    complete = next_index >= len(windows)
    _atomic_json(
        VIDEO_WINDOW_STATE_PATH,
        {
            "signature": signature,
            "next_index": next_index,
            "total": len(windows),
            "dimension": dimension,
            "complete": complete,
        },
    )
    if complete:
        _atomic_json(VIDEO_WINDOW_METADATA_PATH, windows)
        _publish(VIDEO_WINDOW_FEATURES_PATH, VIDEO_WINDOW_INDEX_PATH)
        print(
            f"Published {VIDEO_WINDOW_INDEX_PATH} with {len(windows)} vectors.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
