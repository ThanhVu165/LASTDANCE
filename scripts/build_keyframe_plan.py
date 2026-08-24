"""Build a frame-accurate Begin-Middle-End keyframe plan for one shot manifest."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from offline.config import DataLayout
from offline.preprocessing.keyframes import (
    probe_frame_timestamps,
    select_keyframes,
    write_keyframe_plan_atomic,
)
from offline.preprocessing.shot_detection import load_shot_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("shot_manifest", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--ffprobe", default=os.environ.get("AIC_FFPROBE", "ffprobe"))
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    layout = DataLayout(args.data_root.resolve()) if args.data_root else DataLayout.from_environment()
    source = args.video.resolve()
    try:
        relative_source = source.relative_to(layout.root).as_posix()
    except ValueError as exc:
        raise ValueError("video must be inside AIC_DATA") from exc

    manifest_video_id, _, detection = load_shot_manifest(args.shot_manifest)
    if manifest_video_id != source.stem:
        raise RuntimeError("shot manifest video_id does not match the source video")
    timestamps = probe_frame_timestamps(
        source,
        ffprobe_binary=args.ffprobe,
    )
    items = select_keyframes(
        video_id=source.stem,
        shots=detection.shots,
        frame_timestamps=timestamps,
        excluded_transition_ranges=detection.excluded_transition_ranges,
    )
    output = args.output or (layout.index / "keyframe-plans" / f"{source.stem}.json")
    write_keyframe_plan_atomic(
        output,
        video_id=source.stem,
        relative_video_path=relative_source,
        items=items,
    )
    print(f"keyframe plan: {len(items)} frames -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
