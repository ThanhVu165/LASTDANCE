"""Run one shared TransNetV2 instance over an assigned list of video IDs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

from offline.checkpoints import CheckpointStore
from offline.config import DataLayout
from offline.preprocessing.shot_detection import (
    DEFAULT_TRANSNETV2_DEVICE,
    ShotDetector,
    TRANSNETV2_PYTORCH_BUNDLED_WEIGHTS_SHA256,
    ensure_transnetv2_device_available,
    get_default_shot_detector,
    load_shot_manifest,
    normalize_transnetv2_device,
    resolve_and_verify_transnetv2_weights,
    write_shot_manifest_atomic,
)


_VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_CHECKPOINT_STAGE = "shot_detection"


def _binary_version(command: str) -> str:
    executable = os.environ.get(f"AIC_{command.upper()}") or shutil.which(command)
    if not executable:
        return "unavailable"
    result = subprocess.run(
        [executable, "-version"],
        capture_output=True,
        check=False,
        text=True,
    )
    output = result.stdout or result.stderr
    return output.splitlines()[0].strip() if output else "unknown"


def build_runtime_signature(device: str) -> dict[str, object]:
    import torch

    signature: dict[str, object] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "device": device,
        "ffmpeg": _binary_version("ffmpeg"),
        "ffprobe": _binary_version("ffprobe"),
    }
    if device == "cuda":
        signature["cuda_device_name"] = torch.cuda.get_device_name(0)
    return signature


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        check=False,
        text=True,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else "unknown"


def _update_cuda_peak(report: dict[str, object], device: str) -> None:
    if device != "cuda":
        return
    import torch

    runtime = report["runtime"]
    if not isinstance(runtime, dict):
        raise RuntimeError("batch report runtime payload is invalid")
    runtime["cuda_peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
    runtime["cuda_peak_memory_reserved_bytes"] = torch.cuda.max_memory_reserved()


def read_video_ids(path: Path) -> list[str]:
    rows = Path(path).read_text(encoding="utf-8").splitlines()
    video_ids: list[str] = []
    for line_number, raw in enumerate(rows, start=1):
        if raw == "":
            continue
        if raw != raw.strip():
            raise ValueError(
                f"video list line {line_number} has leading/trailing whitespace"
            )
        if not _VIDEO_ID_PATTERN.fullmatch(raw):
            raise ValueError(f"invalid video_id on line {line_number}: {raw!r}")
        video_ids.append(raw)
    if not video_ids:
        raise ValueError("video list contains no video_id values")
    if len(set(video_ids)) != len(video_ids):
        raise ValueError("video list contains duplicate video_id values")
    return video_ids


def resolve_shots_directory(layout: DataLayout, configured: Path | None) -> Path:
    destination = configured.resolve() if configured else layout.shots
    if not destination.is_relative_to(layout.root):
        raise ValueError("--shots-dir must stay inside AIC_DATA")
    return destination


def resolve_checkpoint_path(
    layout: DataLayout,
    configured: Path | None,
    *,
    video_list: Path,
    device: str,
    shots_directory: Path,
) -> Path:
    """Resolve a worker-local checkpoint without colliding across output namespaces."""

    if configured is not None:
        destination = configured.resolve()
        if not destination.is_relative_to(layout.root):
            raise ValueError("--checkpoint must stay inside AIC_DATA")
        return destination

    namespace = shots_directory.resolve().relative_to(layout.root).as_posix()
    namespace_hash = hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:12]
    return (
        layout.index
        / "shot-batches"
        / f"{video_list.stem}.{device}.{namespace_hash}.checkpoint.json"
    )


def build_video_checkpoint_signature(
    *,
    source: Path,
    relative_video_path: str,
    expected_signature: dict[str, object],
    shots_directory: Path,
    data_root: Path,
) -> str:
    """Bind resume state to the exact input, detector and output namespace."""

    stat = source.stat()
    namespace = shots_directory.resolve().relative_to(data_root.resolve()).as_posix()
    payload = {
        "schema_version": 1,
        "source": {
            "relative_video_path": relative_video_path,
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
        "detector_signature": expected_signature,
        "output_namespace": namespace,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def _validate_existing_manifest(
    path: Path,
    *,
    video_id: str,
    relative_video_path: str,
    expected_signature: dict[str, object],
) -> None:
    manifest_video_id, manifest_relative_path, _ = load_shot_manifest(path)
    if manifest_video_id != video_id:
        raise RuntimeError(f"existing manifest video_id mismatch for {video_id}")
    if manifest_relative_path != relative_video_path:
        raise RuntimeError(
            f"existing manifest relative_video_path mismatch for {video_id}"
        )
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    signature = payload.get("detector_signature")
    if not isinstance(signature, dict):
        raise RuntimeError(f"existing manifest signature is missing for {video_id}")
    for key, expected in expected_signature.items():
        if signature.get(key) != expected:
            raise RuntimeError(
                f"existing manifest detector_signature.{key} mismatch for {video_id}"
            )


def run_checkpointed_video(
    *,
    video_id: str,
    source: Path,
    relative_video_path: str,
    output: Path,
    detector: ShotDetector,
    expected_signature: dict[str, object],
    shots_directory: Path,
    data_root: Path,
    checkpoint_store: CheckpointStore,
    overwrite: bool = False,
) -> str:
    """Run or adopt one atomic manifest and durably advance its checkpoint."""

    if not source.is_file():
        raise RuntimeError(f"video file is missing for {video_id}")
    signature = build_video_checkpoint_signature(
        source=source,
        relative_video_path=relative_video_path,
        expected_signature=expected_signature,
        shots_directory=shots_directory,
        data_root=data_root,
    )
    progress = checkpoint_store.get(video_id, _CHECKPOINT_STAGE)
    if progress is not None and progress.signature != signature:
        raise RuntimeError(
            f"checkpoint signature mismatch for {video_id}; use a new checkpoint path"
        )

    manifest_is_valid = False
    if output.is_file():
        try:
            _validate_existing_manifest(
                output,
                video_id=video_id,
                relative_video_path=relative_video_path,
                expected_signature=expected_signature,
            )
            manifest_is_valid = True
        except Exception:
            if not overwrite:
                raise

    if manifest_is_valid and not overwrite:
        checkpoint_store.update(
            video_id=video_id,
            stage=_CHECKPOINT_STAGE,
            signature=signature,
            next_index=1,
            total=1,
        )
        return "skipped"

    if progress is not None and progress.finished and not overwrite:
        raise RuntimeError(
            f"checkpoint marks {video_id} complete but its manifest is missing"
        )

    if progress is None or not progress.finished:
        checkpoint_store.update(
            video_id=video_id,
            stage=_CHECKPOINT_STAGE,
            signature=signature,
            next_index=0,
            total=1,
        )

    detection = detector.detect(source)
    write_shot_manifest_atomic(
        output,
        video_id=video_id,
        relative_video_path=relative_video_path,
        detector=detector,
        detection=detection,
    )
    _validate_existing_manifest(
        output,
        video_id=video_id,
        relative_video_path=relative_video_path,
        expected_signature=expected_signature,
    )
    checkpoint_store.update(
        video_id=video_id,
        stage=_CHECKPOINT_STAGE,
        signature=signature,
        next_index=1,
        total=1,
    )
    return "completed"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_list", type=Path, help="UTF-8 file with one video_id/line")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--weights-sha256")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument(
        "--shots-dir",
        type=Path,
        help="manifest output directory; defaults to AIC_DATA/shots",
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "per-worker checkpoint file; defaults under "
            "AIC_DATA/index/shot-batches"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="rerun and atomically replace existing compatible/incompatible manifests",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop after the first failed video instead of continuing the assigned list",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    layout = (
        DataLayout(args.data_root.resolve())
        if args.data_root
        else DataLayout.from_environment()
    )
    video_ids = read_video_ids(args.video_list)
    configured_weights = args.weights or os.environ.get("AIC_TRANSNETV2_WEIGHTS")
    weights_path = Path(configured_weights) if configured_weights else None
    weights_sha256 = args.weights_sha256 or os.environ.get(
        "AIC_TRANSNETV2_WEIGHTS_SHA256"
    )
    device = normalize_transnetv2_device(
        args.device
        or os.environ.get("AIC_TRANSNETV2_DEVICE", DEFAULT_TRANSNETV2_DEVICE)
    )
    ensure_transnetv2_device_available(device)
    _, verified_sha256, weights_source = resolve_and_verify_transnetv2_weights(
        weights_path,
        weights_sha256,
    )
    expected_signature = {
        "name": "transnetv2",
        "implementation": "transnetv2-pytorch",
        "package_version": importlib.metadata.version("transnetv2-pytorch"),
        "device": device,
        "threshold": 0.5,
        "weights_source": weights_source,
        "weights_sha256": verified_sha256,
    }
    if (
        weights_path is None
        and verified_sha256 != TRANSNETV2_PYTORCH_BUNDLED_WEIGHTS_SHA256
    ):
        raise RuntimeError("bundled TransNetV2 weight does not match repository contract")

    detector = get_default_shot_detector(
        weights_path=weights_path,
        expected_weights_sha256=weights_sha256,
        model_dir=args.model_dir,
        device=device,
    )
    shots_directory = resolve_shots_directory(layout, args.shots_dir)
    checkpoint_path = resolve_checkpoint_path(
        layout,
        args.checkpoint,
        video_list=args.video_list,
        device=device,
        shots_directory=shots_directory,
    )
    checkpoint_store = CheckpointStore(checkpoint_path)
    report_path = args.report or (
        layout.index / "shot-batches" / f"{args.video_list.stem}.json"
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "source_video_list": args.video_list.name,
        "git_commit": _git_commit(),
        "runtime": build_runtime_signature(device),
        "detector_signature": expected_signature,
        "checkpoint_path": checkpoint_path.relative_to(layout.root).as_posix(),
        "requested_video_ids": video_ids,
        "completed_video_ids": [],
        "skipped_video_ids": [],
        "failures": [],
    }
    _write_json_atomic(report_path, report)

    for video_id in video_ids:
        source = layout.videos / f"{video_id}.mp4"
        relative_video_path = f"videos/{video_id}.mp4"
        output = shots_directory / f"{video_id}.json"
        try:
            outcome = run_checkpointed_video(
                video_id=video_id,
                source=source,
                relative_video_path=relative_video_path,
                output=output,
                detector=detector,
                expected_signature=expected_signature,
                shots_directory=shots_directory,
                data_root=layout.root,
                checkpoint_store=checkpoint_store,
                overwrite=args.overwrite,
            )
            if outcome == "skipped":
                report["skipped_video_ids"].append(video_id)
                print(f"SKIP: {video_id} already has a compatible schema-v2 manifest")
            else:
                report["detector_signature"] = detector.signature
                report["completed_video_ids"].append(video_id)
                print(f"PASS: {video_id} -> {output.name}")
        except Exception as exc:
            sanitized = str(exc).replace(str(layout.root), "${AIC_DATA}")
            report["failures"].append(
                {
                    "video_id": video_id,
                    "error_type": type(exc).__name__,
                    "message": sanitized,
                }
            )
            print(f"FAIL: {video_id}: {type(exc).__name__}: {sanitized}")
            _write_json_atomic(report_path, report)
            if args.fail_fast:
                break
        _update_cuda_peak(report, device)
        _write_json_atomic(report_path, report)

    failure_count = len(report["failures"])
    completed_count = len(report["completed_video_ids"])
    skipped_count = len(report["skipped_video_ids"])
    print(
        f"shot batch: {completed_count} completed, {skipped_count} skipped, "
        f"{failure_count} failed -> {report_path}"
    )
    return 1 if failure_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
