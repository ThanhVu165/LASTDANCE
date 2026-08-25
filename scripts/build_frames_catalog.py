"""Build frames.csv atomically from matching keyframe plans and quality manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

from offline.artifacts import sha256_file
from offline.catalog import (
    discover_catalog_inputs,
    load_inventory_video_ids,
    load_quality_manifest,
    select_catalog_records,
    validate_frames_catalog,
    write_frames_catalog_atomic,
)
from offline.config import DataLayout
from offline.preprocessing.keyframes import load_keyframe_plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, action="append")
    parser.add_argument("--quality", type=Path, action="append")
    parser.add_argument(
        "--collection",
        action="store_true",
        help="aggregate the exact inventory collection from plan/quality directories",
    )
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--plans-dir", type=Path)
    parser.add_argument("--quality-dir", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def resolve_catalog_inputs(
    args: argparse.Namespace,
    layout: DataLayout,
) -> list[tuple[Path, Path]]:
    plans = args.plan or []
    qualities = args.quality or []
    collection_options_used = any(
        value is not None
        for value in (args.inventory, args.plans_dir, args.quality_dir)
    )
    if args.collection:
        if plans or qualities:
            raise ValueError("--collection cannot be combined with --plan/--quality")
        inventory = (args.inventory or (layout.index / "inventory.json")).resolve()
        plans_dir = (args.plans_dir or (layout.index / "keyframe-plans")).resolve()
        quality_dir = (
            args.quality_dir or (layout.index / "keyframe-quality")
        ).resolve()
        return discover_catalog_inputs(
            plans_dir,
            quality_dir,
            expected_video_ids=load_inventory_video_ids(inventory),
        )
    if collection_options_used:
        raise ValueError("--inventory/--plans-dir/--quality-dir require --collection")
    if not plans or not qualities:
        raise ValueError("provide --collection or matching --plan/--quality arguments")
    if len(plans) != len(qualities):
        raise ValueError("--plan and --quality must be provided the same number of times")
    return [
        (plan.resolve(), quality.resolve())
        for plan, quality in zip(plans, qualities, strict=True)
    ]


def main() -> int:
    args = build_parser().parse_args()
    layout = (
        DataLayout(args.data_root.resolve())
        if args.data_root
        else DataLayout.from_environment()
    )
    all_records = []
    sources = []
    seen_videos: set[str] = set()
    catalog_inputs = resolve_catalog_inputs(args, layout)
    for plan_path, quality_path in catalog_inputs:
        video_id, _, items = load_keyframe_plan(plan_path)
        (
            quality_video_id,
            source_plan_sha256,
            quality_config_signature,
            decisions,
        ) = load_quality_manifest(quality_path)
        plan_sha256 = sha256_file(plan_path)
        if video_id != quality_video_id:
            raise RuntimeError("plan and quality manifest video_id values do not match")
        if video_id in seen_videos:
            raise RuntimeError(f"video supplied more than once: {video_id}")
        if source_plan_sha256 != plan_sha256:
            raise RuntimeError(f"quality manifest is stale for plan: {plan_path}")
        seen_videos.add(video_id)
        all_records.extend(select_catalog_records(items, decisions))
        sources.append(
            {
                "video_id": video_id,
                "plan_sha256": plan_sha256,
                "quality_sha256": sha256_file(quality_path),
                "quality_config_signature": quality_config_signature,
            }
        )

    output = args.output or (layout.index / "frames.csv")
    state = write_frames_catalog_atomic(output, records=all_records, sources=sources)
    if not validate_frames_catalog(output, state):
        raise RuntimeError("frames catalog failed post-publish validation")
    print(
        f"frames catalog: {len(all_records)} records / {len(seen_videos)} videos "
        f"-> {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
