"""Wait for keyframe shards, verify exact coverage, then publish frames.csv."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from offline.catalog import load_inventory_video_ids, validate_frames_catalog
from offline.config import DataLayout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--state", type=Path)
    return parser


def shard_scope(shard_index: int, shard_count: int) -> str:
    return f"collection-shard-{shard_index + 1:02d}-of-{shard_count:02d}"


def load_shard_report(path: Path, *, expected_scope: str) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read shard report: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RuntimeError(f"invalid shard report schema: {path}")
    if payload.get("scope") != expected_scope:
        raise RuntimeError(f"shard report scope mismatch: {path}")
    if not isinstance(payload.get("requested"), int) or payload["requested"] <= 0:
        raise RuntimeError(f"invalid shard report requested count: {path}")
    for field in ("completed", "skipped"):
        if not isinstance(payload.get(field), list) or any(
            not isinstance(video_id, str) for video_id in payload[field]
        ):
            raise RuntimeError(f"invalid shard report {field} list: {path}")
    if not isinstance(payload.get("failed"), dict):
        raise RuntimeError(f"invalid shard report failed mapping: {path}")
    return payload


def validate_completed_shard_reports(
    reports: list[dict[str, object]],
    *,
    expected_video_ids: list[str],
) -> None:
    all_processed: list[str] = []
    for report in reports:
        if report.get("complete") is not True:
            raise RuntimeError(f"shard is not complete: {report.get('scope')}")
        failed = report["failed"]
        if not isinstance(failed, dict) or failed:
            raise RuntimeError(f"shard contains failures: {report.get('scope')}")
        completed = report["completed"]
        skipped = report["skipped"]
        if not isinstance(completed, list) or not isinstance(skipped, list):
            raise RuntimeError("shard completion lists are invalid")
        processed = [*completed, *skipped]
        if len(processed) != report["requested"]:
            raise RuntimeError(f"shard processed count mismatch: {report.get('scope')}")
        if len(set(processed)) != len(processed):
            raise RuntimeError(
                f"shard contains duplicate video IDs: {report.get('scope')}"
            )
        all_processed.extend(processed)

    if len(set(all_processed)) != len(all_processed):
        raise RuntimeError("keyframe shard reports overlap")
    expected = set(expected_video_ids)
    actual = set(all_processed)
    if actual != expected:
        raise RuntimeError(
            "keyframe shard reports do not cover inventory: "
            f"missing={sorted(expected - actual)[:10]}, "
            f"unexpected={sorted(actual - expected)[:10]}"
        )


def _write_state_atomic(path: Path, payload: dict[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def _resolve_inside(root: Path, path: Path, label: str) -> Path:
    destination = Path(path).resolve(strict=False)
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside AIC_DATA") from exc
    return destination


def main() -> int:
    args = build_parser().parse_args()
    if args.shard_count <= 0:
        raise ValueError("--shard-count must be positive")
    if args.poll_seconds <= 0 or args.poll_seconds > 60:
        raise ValueError("--poll-seconds must satisfy 0 < value <= 60")

    layout = (
        DataLayout(args.data_root.resolve())
        if args.data_root
        else DataLayout.from_environment()
    )
    root = layout.root
    output = _resolve_inside(
        root,
        args.output or (layout.index / "frames.csv"),
        "--output",
    )
    state_path = _resolve_inside(
        root,
        args.state or (layout.index / "keyframe-batches" / "collection-finalizer.json"),
        "--state",
    )
    inventory_path = layout.index / "inventory.json"
    expected_video_ids = load_inventory_video_ids(inventory_path)
    report_paths = [
        layout.index
        / "keyframe-batches"
        / f"{shard_scope(index, args.shard_count)}.json"
        for index in range(args.shard_count)
    ]
    last_progress: tuple[int, ...] | None = None

    while True:
        reports: list[dict[str, object]] = []
        progress: list[int] = []
        all_present = True
        for index, report_path in enumerate(report_paths):
            if not report_path.is_file():
                all_present = False
                progress.append(0)
                continue
            report = load_shard_report(
                report_path,
                expected_scope=shard_scope(index, args.shard_count),
            )
            failed = report["failed"]
            if not isinstance(failed, dict):
                raise RuntimeError("shard failed mapping is invalid")
            if failed:
                raise RuntimeError(
                    f"keyframe shard failed: {report.get('scope')} {failed}"
                )
            completed = report["completed"]
            skipped = report["skipped"]
            if not isinstance(completed, list) or not isinstance(skipped, list):
                raise RuntimeError("shard completion lists are invalid")
            progress.append(len(completed) + len(skipped))
            reports.append(report)

        progress_tuple = tuple(progress)
        if progress_tuple != last_progress:
            print(
                "keyframe shard progress: "
                + ", ".join(
                    f"{index + 1}/{args.shard_count}={value}"
                    for index, value in enumerate(progress)
                ),
                flush=True,
            )
            _write_state_atomic(
                state_path,
                {
                    "schema_version": 1,
                    "complete": False,
                    "shard_count": args.shard_count,
                    "progress": progress,
                    "expected_videos": len(expected_video_ids),
                },
            )
            last_progress = progress_tuple

        if (
            all_present
            and len(reports) == args.shard_count
            and all(report.get("complete") is True for report in reports)
        ):
            validate_completed_shard_reports(
                reports,
                expected_video_ids=expected_video_ids,
            )
            break
        time.sleep(args.poll_seconds)

    command = [
        sys.executable,
        "-m",
        "scripts.build_frames_catalog",
        "--collection",
        "--data-root",
        str(root),
        "--output",
        str(output),
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"frames catalog command failed: {result.returncode}")
    if not validate_frames_catalog(output):
        raise RuntimeError("frames catalog failed finalizer validation")

    catalog_state = json.loads(
        output.with_name(f"{output.name}.state.json").read_text(encoding="utf-8")
    )
    _write_state_atomic(
        state_path,
        {
            "schema_version": 1,
            "complete": True,
            "shard_count": args.shard_count,
            "expected_videos": len(expected_video_ids),
            "catalog": output.relative_to(root).as_posix(),
            "catalog_record_count": catalog_state["record_count"],
            "catalog_video_count": catalog_state["video_count"],
            "catalog_sha256": catalog_state["csv_sha256"],
        },
    )
    print(
        f"collection finalized: {catalog_state['record_count']} frames / "
        f"{catalog_state['video_count']} videos -> {output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
