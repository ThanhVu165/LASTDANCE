"""Run plan, exact extraction, and quality stages over a resumable video batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from offline.artifacts import sha256_file
from offline.catalog import (
    load_quality_manifest,
    select_catalog_records,
)
from offline.checkpoints import CheckpointStore
from offline.config import DataLayout
from offline.preprocessing.keyframes import KeyframePlanItem, load_keyframe_plan


_VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_CHECKPOINT_STAGE = "keyframe-pipeline"
_STAGE_TOTAL = 3
_Runner = Callable[..., subprocess.CompletedProcess[object]]
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_IMPLEMENTATION_FILES = (
    "offline/identifiers.py",
    "offline/preprocessing/keyframes.py",
    "offline/preprocessing/quality.py",
    "scripts/build_keyframe_plan.py",
    "scripts/extract_keyframes.py",
    "scripts/filter_keyframes.py",
    "scripts/run_keyframe_batch.py",
)


@dataclass(frozen=True, slots=True)
class InventoryVideo:
    video_id: str
    relative_path: str


@dataclass(frozen=True, slots=True)
class BatchConfig:
    layout: DataLayout
    shots_dir: Path
    plans_dir: Path
    quality_dir: Path
    extraction_state: Path
    ffprobe: str
    ffmpeg: str
    jpeg_quality: int
    checkpoint_every: int
    blur_threshold: float | None
    phash_max_distance: int | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument(
        "--collection",
        action="store_true",
        help="process the exact collection declared by inventory.json",
    )
    scope.add_argument(
        "--video-list",
        type=Path,
        help="UTF-8 subset file with one inventory video_id per line",
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--shots-dir", type=Path)
    parser.add_argument("--plans-dir", type=Path)
    parser.add_argument("--quality-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--extraction-state", type=Path)
    parser.add_argument("--ffprobe", default=os.environ.get("AIC_FFPROBE", "ffprobe"))
    parser.add_argument("--ffmpeg", default=os.environ.get("AIC_FFMPEG", "ffmpeg"))
    parser.add_argument("--jpeg-quality", type=int, default=3)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--blur-threshold", type=float)
    parser.add_argument("--phash-max-distance", type=int)
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def _resolve_inside(root: Path, path: Path, label: str) -> Path:
    destination = Path(path).resolve(strict=False)
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside AIC_DATA") from exc
    return destination


def load_inventory_videos(path: Path) -> list[InventoryVideo]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read inventory: {source}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RuntimeError("unsupported or invalid inventory schema")
    rows = payload.get("videos")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("inventory contains no videos")

    videos: list[InventoryVideo] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"inventory video[{index}] is not an object")
        raw_video_id = row.get("video_id")
        raw_relative_path = row.get("relative_path")
        if not isinstance(raw_video_id, str) or not _VIDEO_ID_PATTERN.fullmatch(
            raw_video_id
        ):
            raise RuntimeError(f"inventory video[{index}] has an invalid video_id")
        if not isinstance(raw_relative_path, str) or not raw_relative_path:
            raise RuntimeError(f"inventory video[{index}] has an invalid relative_path")
        relative_path = Path(raw_relative_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError(
                f"inventory video[{index}] relative_path escapes AIC_DATA"
            )
        if relative_path.stem != raw_video_id:
            raise RuntimeError(
                f"inventory video[{index}] path stem does not match video_id"
            )
        videos.append(InventoryVideo(raw_video_id, relative_path.as_posix()))

    video_ids = [video.video_id for video in videos]
    if len(set(video_ids)) != len(video_ids):
        raise RuntimeError("inventory contains duplicate video_id values")
    return sorted(videos, key=lambda video: video.video_id)


def read_video_ids(path: Path) -> list[str]:
    try:
        rows = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError(f"cannot read video list: {path}") from exc
    video_ids: list[str] = []
    for line_number, raw in enumerate(rows, start=1):
        if not raw:
            continue
        if raw != raw.strip() or not _VIDEO_ID_PATTERN.fullmatch(raw):
            raise ValueError(f"invalid video_id on line {line_number}: {raw!r}")
        video_ids.append(raw)
    if not video_ids:
        raise ValueError("video list contains no video_id values")
    if len(set(video_ids)) != len(video_ids):
        raise ValueError("video list contains duplicate video_id values")
    return video_ids


def select_inventory_videos(
    inventory: Sequence[InventoryVideo],
    requested_ids: Sequence[str] | None,
) -> list[InventoryVideo]:
    by_id = {video.video_id: video for video in inventory}
    if requested_ids is None:
        return list(inventory)
    unexpected = sorted(set(requested_ids) - set(by_id))
    if unexpected:
        raise RuntimeError(
            f"video list contains IDs outside inventory: {unexpected[:10]}"
        )
    return [by_id[video_id] for video_id in requested_ids]


def shard_inventory_videos(
    videos: Sequence[InventoryVideo],
    *,
    shard_count: int,
    shard_index: int,
) -> list[InventoryVideo]:
    if shard_count <= 0:
        raise ValueError("--shard-count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("--shard-index must satisfy 0 <= index < shard-count")
    selected = list(videos)[shard_index::shard_count]
    if not selected:
        raise RuntimeError("selected shard contains no videos")
    return selected


def validate_shot_membership(
    shots_dir: Path,
    videos: Sequence[InventoryVideo],
) -> None:
    root = Path(shots_dir)
    if not root.is_dir():
        raise RuntimeError(f"shot manifest directory not found: {root}")
    expected = {video.video_id for video in videos}
    paths = sorted(root.glob("*.json"))
    actual = {path.stem for path in paths if path.is_file()}
    if len(actual) != len(paths):
        raise RuntimeError("shot manifest directory contains duplicate stems")
    if actual != expected:
        raise RuntimeError(
            "shot manifest collection does not match inventory: "
            f"missing={sorted(expected - actual)[:10]}, "
            f"unexpected={sorted(actual - expected)[:10]}"
        )


def pipeline_signature(
    config: BatchConfig,
    video: InventoryVideo,
) -> str:
    video_path = (config.layout.root / video.relative_path).resolve(strict=False)
    shot_path = config.shots_dir / f"{video.video_id}.json"
    if not video_path.is_file():
        raise RuntimeError(f"inventory video not found: {video_path}")
    if not shot_path.is_file():
        raise RuntimeError(f"shot manifest not found: {shot_path}")
    stat = video_path.stat()
    payload = {
        "schema_version": 1,
        "video_id": video.video_id,
        "implementation_signature": implementation_signature(),
        "relative_video_path": video.relative_path,
        "video_size": stat.st_size,
        "video_mtime_ns": stat.st_mtime_ns,
        "shot_sha256": sha256_file(shot_path),
        "ffprobe": config.ffprobe,
        "ffmpeg": config.ffmpeg,
        "jpeg_quality": config.jpeg_quality,
        "blur_threshold": config.blur_threshold,
        "phash_max_distance": config.phash_max_distance,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@lru_cache(maxsize=1)
def implementation_signature() -> str:
    hashes = {
        relative_path: sha256_file(_REPOSITORY_ROOT / relative_path)
        for relative_path in _IMPLEMENTATION_FILES
    }
    encoded = json.dumps(
        hashes,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _expected_quality_signature(
    plan_sha256: str,
    *,
    blur_threshold: float | None,
    phash_max_distance: int | None,
) -> str:
    config = {
        "blur_threshold": blur_threshold,
        "phash_max_distance": phash_max_distance,
        "preserve_at_least_one_per_shot": True,
    }
    encoded = json.dumps(
        {"source_plan_sha256": plan_sha256, "config": config},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_plan(
    config: BatchConfig,
    video: InventoryVideo,
) -> tuple[Path, list[KeyframePlanItem]]:
    plan_path = config.plans_dir / f"{video.video_id}.json"
    plan_video_id, relative_video_path, items = load_keyframe_plan(plan_path)
    if plan_video_id != video.video_id:
        raise RuntimeError(f"keyframe plan video_id mismatch: {video.video_id}")
    if Path(relative_video_path).as_posix() != video.relative_path:
        raise RuntimeError(f"keyframe plan source path mismatch: {video.video_id}")
    return plan_path, items


def validate_extraction(
    config: BatchConfig,
    items: Sequence[KeyframePlanItem],
) -> None:
    root = config.layout.root
    for item in items:
        image_path = (root / item.relative_image_path).resolve(strict=False)
        try:
            image_path.relative_to(root)
        except ValueError as exc:
            raise RuntimeError("keyframe path escapes AIC_DATA") from exc
        if not image_path.is_file() or image_path.stat().st_size == 0:
            raise RuntimeError(f"keyframe image missing or empty: {image_path}")


def validate_quality(
    config: BatchConfig,
    video: InventoryVideo,
    plan_path: Path,
    items: Sequence[KeyframePlanItem],
) -> int:
    quality_path = config.quality_dir / f"{video.video_id}.json"
    (
        quality_video_id,
        source_plan_sha256,
        config_signature,
        decisions,
    ) = load_quality_manifest(quality_path)
    plan_sha256 = sha256_file(plan_path)
    if quality_video_id != video.video_id:
        raise RuntimeError(f"quality video_id mismatch: {video.video_id}")
    if source_plan_sha256 != plan_sha256:
        raise RuntimeError(f"quality manifest is stale: {video.video_id}")
    expected_signature = _expected_quality_signature(
        plan_sha256,
        blur_threshold=config.blur_threshold,
        phash_max_distance=config.phash_max_distance,
    )
    if config_signature != expected_signature:
        raise RuntimeError(f"quality config signature mismatch: {video.video_id}")
    return len(select_catalog_records(items, decisions))


def _run_stage(command: list[str], runner: _Runner) -> None:
    result = runner(command, check=False)
    if result.returncode != 0:
        rendered = subprocess.list2cmdline(command)
        raise RuntimeError(f"stage command failed ({result.returncode}): {rendered}")


def process_video(
    config: BatchConfig,
    video: InventoryVideo,
    checkpoint: CheckpointStore,
    *,
    runner: _Runner = subprocess.run,
) -> tuple[str, int]:
    signature = pipeline_signature(config, video)
    existing = checkpoint.get(video.video_id, _CHECKPOINT_STAGE)
    if existing is not None:
        if existing.signature != signature:
            raise RuntimeError(
                f"keyframe batch checkpoint signature mismatch: {video.video_id}"
            )
        if existing.total != _STAGE_TOTAL:
            raise RuntimeError(
                f"keyframe batch checkpoint total mismatch: {video.video_id}"
            )
    next_index = existing.next_index if existing is not None else 0
    initially_finished = existing.finished if existing is not None else False

    video_path = config.layout.root / video.relative_path
    shot_path = config.shots_dir / f"{video.video_id}.json"
    plan_path = config.plans_dir / f"{video.video_id}.json"
    quality_path = config.quality_dir / f"{video.video_id}.json"

    if next_index < 1:
        _run_stage(
            [
                sys.executable,
                "-m",
                "scripts.build_keyframe_plan",
                str(video_path),
                str(shot_path),
                "--data-root",
                str(config.layout.root),
                "--ffprobe",
                config.ffprobe,
                "--output",
                str(plan_path),
            ],
            runner,
        )
        plan_path, items = validate_plan(config, video)
        checkpoint.update(
            video_id=video.video_id,
            stage=_CHECKPOINT_STAGE,
            signature=signature,
            next_index=1,
            total=_STAGE_TOTAL,
        )
        next_index = 1
    else:
        plan_path, items = validate_plan(config, video)

    if next_index < 2:
        _run_stage(
            [
                sys.executable,
                "-m",
                "scripts.extract_keyframes",
                str(plan_path),
                "--data-root",
                str(config.layout.root),
                "--ffmpeg",
                config.ffmpeg,
                "--jpeg-quality",
                str(config.jpeg_quality),
                "--checkpoint-every",
                str(config.checkpoint_every),
                "--state",
                str(config.extraction_state),
            ],
            runner,
        )
        validate_extraction(config, items)
        checkpoint.update(
            video_id=video.video_id,
            stage=_CHECKPOINT_STAGE,
            signature=signature,
            next_index=2,
            total=_STAGE_TOTAL,
        )
        next_index = 2
    else:
        validate_extraction(config, items)

    if next_index < 3:
        command = [
            sys.executable,
            "-m",
            "scripts.filter_keyframes",
            str(plan_path),
            "--data-root",
            str(config.layout.root),
            "--output",
            str(quality_path),
        ]
        if config.blur_threshold is not None:
            command.extend(["--blur-threshold", str(config.blur_threshold)])
        if config.phash_max_distance is not None:
            command.extend(["--phash-max-distance", str(config.phash_max_distance)])
        _run_stage(command, runner)
        kept = validate_quality(config, video, plan_path, items)
        checkpoint.update(
            video_id=video.video_id,
            stage=_CHECKPOINT_STAGE,
            signature=signature,
            next_index=3,
            total=_STAGE_TOTAL,
        )
    else:
        kept = validate_quality(config, video, plan_path, items)

    return ("skipped" if initially_finished else "completed"), kept


def _write_report_atomic(path: Path, report: dict[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def main() -> int:
    args = build_parser().parse_args()
    if args.jpeg_quality <= 0:
        raise ValueError("--jpeg-quality must be positive")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive")
    if args.blur_threshold is not None and args.blur_threshold < 0:
        raise ValueError("--blur-threshold must be non-negative")
    if args.shard_count <= 0:
        raise ValueError("--shard-count must be positive")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("--shard-index must satisfy 0 <= index < shard-count")
    if args.phash_max_distance is not None and args.phash_max_distance < 0:
        raise ValueError("--phash-max-distance must be non-negative")

    layout = (
        DataLayout(args.data_root.resolve())
        if args.data_root
        else DataLayout.from_environment()
    )
    root = layout.root
    inventory_path = _resolve_inside(
        root,
        args.inventory or (layout.index / "inventory.json"),
        "--inventory",
    )
    shots_dir = _resolve_inside(
        root,
        args.shots_dir or layout.shots,
        "--shots-dir",
    )
    plans_dir = _resolve_inside(
        root,
        args.plans_dir or (layout.index / "keyframe-plans"),
        "--plans-dir",
    )
    quality_dir = _resolve_inside(
        root,
        args.quality_dir or (layout.index / "keyframe-quality"),
        "--quality-dir",
    )
    base_scope_name = "collection" if args.collection else args.video_list.stem
    scope_name = (
        base_scope_name
        if args.shard_count == 1
        else f"{base_scope_name}-shard-{args.shard_index + 1:02d}-of-{args.shard_count:02d}"
    )
    default_extraction_state = (
        layout.index / "keyframe_extraction_state.json"
        if args.shard_count == 1
        else layout.index / "keyframe-batches" / f"{scope_name}.extraction-state.json"
    )
    extraction_state = _resolve_inside(
        root,
        args.extraction_state or default_extraction_state,
        "--extraction-state",
    )
    checkpoint_path = _resolve_inside(
        root,
        args.checkpoint
        or (layout.index / "keyframe-batches" / f"{scope_name}.checkpoint.json"),
        "--checkpoint",
    )
    report_path = _resolve_inside(
        root,
        args.report or (layout.index / "keyframe-batches" / f"{scope_name}.json"),
        "--report",
    )

    inventory = load_inventory_videos(inventory_path)
    requested_ids = None if args.collection else read_video_ids(args.video_list)
    selected_videos = select_inventory_videos(inventory, requested_ids)
    if args.collection:
        validate_shot_membership(shots_dir, selected_videos)
    videos = shard_inventory_videos(
        selected_videos,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
    )
    config = BatchConfig(
        layout=layout,
        shots_dir=shots_dir,
        plans_dir=plans_dir,
        quality_dir=quality_dir,
        extraction_state=extraction_state,
        ffprobe=args.ffprobe,
        ffmpeg=args.ffmpeg,
        jpeg_quality=args.jpeg_quality,
        checkpoint_every=args.checkpoint_every,
        blur_threshold=args.blur_threshold,
        phash_max_distance=args.phash_max_distance,
    )
    checkpoint = CheckpointStore(checkpoint_path)
    report: dict[str, object] = {
        "schema_version": 1,
        "complete": False,
        "scope": scope_name,
        "requested": len(videos),
        "completed": [],
        "skipped": [],
        "failed": {},
        "kept_keyframes": {},
        "checkpoint": checkpoint_path.relative_to(root).as_posix(),
        "config": {
            "jpeg_quality": args.jpeg_quality,
            "checkpoint_every": args.checkpoint_every,
            "blur_threshold": args.blur_threshold,
            "phash_max_distance": args.phash_max_distance,
            "implementation_signature": implementation_signature(),
        },
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
    }
    _write_report_atomic(report_path, report)

    try:
        for position, video in enumerate(videos, start=1):
            print(f"[{position}/{len(videos)}] {video.video_id}", flush=True)
            try:
                status, kept = process_video(config, video, checkpoint)
            except Exception as exc:
                failed = report["failed"]
                if not isinstance(failed, dict):
                    raise RuntimeError("batch report failed mapping is invalid")
                failed[video.video_id] = f"{type(exc).__name__}: {exc}"
                print(f"FAILED {video.video_id}: {exc}", flush=True)
                _write_report_atomic(report_path, report)
                if args.fail_fast:
                    break
                continue
            rows = report[status]
            if not isinstance(rows, list):
                raise RuntimeError(f"batch report {status} list is invalid")
            rows.append(video.video_id)
            kept_rows = report["kept_keyframes"]
            if not isinstance(kept_rows, dict):
                raise RuntimeError("batch report kept_keyframes mapping is invalid")
            kept_rows[video.video_id] = kept
            _write_report_atomic(report_path, report)
    except KeyboardInterrupt:
        report["interrupted"] = True
        _write_report_atomic(report_path, report)
        raise

    completed = report["completed"]
    skipped = report["skipped"]
    failed = report["failed"]
    if not isinstance(completed, list) or not isinstance(skipped, list):
        raise RuntimeError("batch report completion lists are invalid")
    if not isinstance(failed, dict):
        raise RuntimeError("batch report failed mapping is invalid")
    report["complete"] = not failed and len(completed) + len(skipped) == len(videos)
    _write_report_atomic(report_path, report)
    print(
        "keyframe batch: "
        f"requested={len(videos)} completed={len(completed)} "
        f"skipped={len(skipped)} failed={len(failed)} -> {report_path}",
        flush=True,
    )
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
