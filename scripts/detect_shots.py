"""Detect shots for one video using the swappable ShotDetector contract."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from offline.config import DataLayout
from offline.preprocessing.shot_detection import (
    get_default_shot_detector,
    write_shot_manifest_atomic,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--weights-sha256")
    parser.add_argument("--model-dir", type=Path)
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

    configured_weights = args.weights or os.environ.get("AIC_TRANSNETV2_WEIGHTS")
    weights_path = Path(configured_weights) if configured_weights else None
    weights_sha256 = args.weights_sha256 or os.environ.get(
        "AIC_TRANSNETV2_WEIGHTS_SHA256"
    )
    detector = get_default_shot_detector(
        weights_path=weights_path,
        expected_weights_sha256=weights_sha256,
        model_dir=args.model_dir,
    )
    detection = detector.detect(source)
    output = args.output or (layout.shots / f"{source.stem}.json")
    write_shot_manifest_atomic(
        output,
        video_id=source.stem,
        relative_video_path=relative_source,
        detector=detector,
        detection=detection,
    )
    excluded_count = sum(
        item.frame_count for item in detection.excluded_transition_ranges
    )
    print(
        f"shots: {len(detection.shots)} boundaries, "
        f"{excluded_count}/{detection.total_frame_count} transition frames excluded "
        f"-> {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
