"""Measure keyframe quality and atomically publish a reversible selection manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from offline.artifacts import sha256_file
from offline.config import DataLayout
from offline.preprocessing.keyframes import load_keyframe_plan
from offline.preprocessing.quality import assess_keyframes, write_quality_manifest_atomic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--blur-threshold",
        type=float,
        help="minimum Laplacian variance; omit for report-only blur metrics",
    )
    parser.add_argument(
        "--phash-max-distance",
        type=int,
        help="maximum pHash Hamming distance treated as duplicate within a shot",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    layout = (
        DataLayout(args.data_root.resolve())
        if args.data_root
        else DataLayout.from_environment()
    )
    plan_path = args.plan.resolve()
    video_id, _, items = load_keyframe_plan(plan_path)
    decisions = assess_keyframes(
        items,
        data_root=layout.root,
        blur_threshold=args.blur_threshold,
        phash_max_distance=args.phash_max_distance,
    )
    output = args.output or (
        layout.index / "keyframe-quality" / f"{video_id}.json"
    )
    write_quality_manifest_atomic(
        output,
        video_id=video_id,
        source_plan_sha256=sha256_file(plan_path),
        blur_threshold=args.blur_threshold,
        phash_max_distance=args.phash_max_distance,
        decisions=decisions,
    )
    kept = sum(decision.kept for decision in decisions)
    mode = (
        "report-only"
        if args.blur_threshold is None and args.phash_max_distance is None
        else "filtered"
    )
    print(f"quality: {kept}/{len(decisions)} kept for {video_id} ({mode}) -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
