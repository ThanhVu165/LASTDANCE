"""Extract a keyframe plan with atomic outputs and resumable checkpoints."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from offline.artifacts import sha256_file
from offline.checkpoints import CheckpointStore
from offline.config import DataLayout
from offline.preprocessing.keyframes import (
    extract_keyframes_exact_batch,
    load_keyframe_plan,
)


_STAGE = "keyframe-extraction"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--ffmpeg", default=os.environ.get("AIC_FFMPEG", "ffmpeg"))
    parser.add_argument("--jpeg-quality", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--state", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive")
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative")

    layout = DataLayout(args.data_root.resolve()) if args.data_root else DataLayout.from_environment()
    plan_path = args.plan.resolve()
    # The loader validates the canonical plan order. Do not sort here: checkpoints
    # address indexes in that exact order and reordering would corrupt resume semantics.
    video_id, relative_video_path, all_items = load_keyframe_plan(plan_path)
    total = len(all_items)
    run_stop = total if args.limit is None else min(args.limit, total)
    signature = f"{sha256_file(plan_path)}:jpeg-quality={args.jpeg_quality}"
    state_path = args.state or (layout.index / "keyframe_extraction_state.json")
    store = CheckpointStore(state_path)
    existing = store.get(video_id, _STAGE)
    if existing is not None:
        if existing.signature != signature:
            raise RuntimeError("checkpoint signature mismatch")
        if existing.total != total:
            raise RuntimeError("checkpoint total mismatch")
    next_index = existing.next_index if existing is not None else 0

    source = layout.root / relative_video_path
    if next_index > run_stop:
        raise RuntimeError(
            "checkpoint is already beyond this --limit; remove --limit or increase it"
        )

    def checkpoint_progress(batch_progress: int) -> None:
        progress = next_index + batch_progress
        if progress % args.checkpoint_every == 0 or progress == run_stop:
            store.update(
                video_id=video_id,
                stage=_STAGE,
                signature=signature,
                next_index=progress,
                total=total,
            )

    extract_keyframes_exact_batch(
        source,
        all_items[next_index:run_stop],
        data_root=layout.root,
        ffmpeg_binary=args.ffmpeg,
        jpeg_quality=args.jpeg_quality,
        on_progress=checkpoint_progress,
    )

    final_progress = store.get(video_id, _STAGE)
    completed = final_progress.next_index if final_progress is not None else next_index
    print(f"keyframes: {completed}/{total} extracted for {video_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
