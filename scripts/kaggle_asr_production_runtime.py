"""Self-contained Kaggle ASR runtime (pasteable; intentionally no repo imports)."""

from __future__ import annotations

import hashlib
import json
import os
import csv
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {"completed": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _video_set_sha256(video_ids: list[str]) -> str:
    digest = hashlib.sha256()
    for video_id in sorted(set(video_ids)):
        digest.update(video_id.encode("utf-8") + b"\n")
    return digest.hexdigest()


def _load_frames(path: str | Path | None) -> dict[str, list[tuple[float, int]]]:
    if path is None:
        return {}
    result: dict[str, list[tuple[float, int]]] = {}
    with Path(path).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            result.setdefault(row["video_id"], []).append(
                (float(row["pts_time"]), int(row["keyframe_uid"]))
            )
    return result


def _nearest(video_id: str, start: float, end: float, frames: dict[str, list[tuple[float, int]]]) -> int:
    values = frames.get(video_id, [])
    if not values:
        raise RuntimeError(f"catalog has no keyframes for {video_id}")
    inside = [row for row in values if start <= row[0] <= end]
    target = (start + end) / 2 if inside else start
    return min(inside or values, key=lambda row: (abs(row[0] - target), row[0]))[1]


def _push_checkpoint(
    hf_repo_id: str,
    hf_token: str,
    batch_id: str,
    destination: Path,
    state_file: Path,
) -> None:
    """Best-effort push of in-progress JSONL + checkpoint state to HF.

    Unlike the final archive push, this is NOT gated on batch completion — it is
    called periodically so a Kaggle timeout or crash mid-batch never loses more
    than `checkpoint_every` videos of work. Failures are swallowed (network
    hiccups must not abort transcription) and logged to stdout.
    """
    try:
        from huggingface_hub import CommitOperationAdd, HfApi
        api = HfApi(token=hf_token)
        api.create_commit(
            repo_id=hf_repo_id,
            repo_type="dataset",
            operations=[
                CommitOperationAdd(
                    path_in_repo=f"asr/checkpoints/{batch_id}/{destination.name}",
                    path_or_fileobj=destination,
                ),
                CommitOperationAdd(
                    path_in_repo=f"asr/checkpoints/{batch_id}/{state_file.name}",
                    path_or_fileobj=state_file,
                ),
            ],
            commit_message=f"ASR checkpoint {batch_id} ({len(destination.read_text(encoding='utf-8').splitlines())} rows)",
        )
    except Exception as exc:  # noqa: BLE001 - checkpoint push must never crash the run
        print(f"[checkpoint] push failed for {batch_id}: {exc}")


def _restore_checkpoint(
    hf_repo_id: str,
    hf_token: str,
    batch_id: str,
    destination: Path,
    state_file: Path,
) -> None:
    """Best-effort download of a prior in-progress checkpoint from HF.

    Called once at the start of run_production, before transcription begins.
    If the local JSONL/state files are missing (fresh /kaggle/working after a
    timeout or restart) but a checkpoint exists on HF, restore it so already
    -transcribed videos are skipped instead of redone.
    """
    if destination.exists() or state_file.exists():
        return
    try:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi(token=hf_token)
        remote_jsonl = f"asr/checkpoints/{batch_id}/{destination.name}"
        remote_state = f"asr/checkpoints/{batch_id}/{state_file.name}"
        existing_paths = set(api.list_repo_files(repo_id=hf_repo_id, repo_type="dataset"))
        if remote_jsonl not in existing_paths or remote_state not in existing_paths:
            return
        for remote_path, local_path in ((remote_jsonl, destination), (remote_state, state_file)):
            local_path.parent.mkdir(parents=True, exist_ok=True)
            downloaded = hf_hub_download(
                repo_id=hf_repo_id, repo_type="dataset", filename=remote_path, token=hf_token,
            )
            local_path.write_bytes(Path(downloaded).read_bytes())
        print(f"[checkpoint] restored {batch_id} from HF (resume mode)")
    except Exception as exc:  # noqa: BLE001 - restore is opportunistic, never fatal
        print(f"[checkpoint] restore skipped for {batch_id}: {exc}")


def run_production(
    audio_dir: str | Path,
    output_jsonl: str | Path,
    *,
    batch_id: str,
    worker_id: str = "kaggle-01",
    state_path: str | Path | None = None,
    model_name: str = "large-v3",
    catalog_path: str | Path | None = None,
    hf_repo_id: str | None = None,
    hf_token: str | None = None,
    manifest_path: str | Path | None = None,
    checkpoint_every: int = 10,
) -> Path:
    """Transcribe each FLAC once and write one terminal envelope per video."""

    from faster_whisper import WhisperModel

    audio_root, destination = Path(audio_dir), Path(output_jsonl)
    destination.parent.mkdir(parents=True, exist_ok=True)
    state_file = Path(state_path or destination.with_suffix(".state.json"))
    if hf_repo_id and hf_token:
        _restore_checkpoint(hf_repo_id, hf_token, batch_id, destination, state_file)
    state = _load_state(state_file)
    completed = set(state.get("completed", []))
    model = WhisperModel(model_name, device="cuda", compute_type="float16")
    frames = _load_frames(catalog_path or os.environ.get("ASR_CATALOG_PATH"))
    existing = destination.read_text(encoding="utf-8").splitlines() if destination.exists() else []
    existing_rows = [json.loads(line) for line in existing if line.strip()]
    if any(not isinstance(row, dict) or not row.get("video_id") for row in existing_rows):
        raise RuntimeError("ASR output JSONL contains an invalid existing row")
    seen = {str(row["video_id"]) for row in existing_rows}
    all_rows = list(existing_rows)
    videos_since_checkpoint = 0
    with destination.open("a", encoding="utf-8") as output:
        for audio in sorted(audio_root.glob("*.flac")):
            video_id = audio.stem
            if video_id in seen:
                continue
            try:
                segments, info = model.transcribe(
                    str(audio),
                    language=None,
                    vad_filter=True,
                    condition_on_previous_text=False,
                )
                rows = []
                audio_duration = float(getattr(info, "duration", 0.0) or 0.0)
                for segment in segments:
                    text = str(segment.text).strip()
                    if not text:
                        continue
                    start_time = max(0.0, float(segment.start))
                    if audio_duration > 0 and start_time >= audio_duration:
                        continue
                    end_time = max(start_time, float(segment.end))
                    if audio_duration > 0:
                        end_time = min(end_time, audio_duration)
                    language = "vi" if str(info.language).lower().startswith("vi") else "en"
                    rows.append({
                        "video_id": video_id, "segment_id": f"s{len(rows):06d}",
                        "start_time": start_time,
                        "end_time": end_time,
                        "transcribed_text": text, "language": language,
                        "keyframe_uid_nearest": _nearest(video_id, start_time, end_time, frames)
                    })
                status = "success" if rows else "silent"
                envelope = {
                    "schema_version": 1, "batch_id": batch_id, "video_id": video_id,
                    "status": status, "engine": "whisper_large_v3",
                    "audio_path": f"asr/audio/{batch_id}/{audio.name}",
                    "audio_sha256": _sha256(audio),
                    "audio_duration_seconds": audio_duration,
                    "segments": rows
                }
            except (OSError, RuntimeError, ValueError) as exc:
                envelope = {
                    "schema_version": 1, "batch_id": batch_id, "video_id": video_id,
                    "status": "error", "engine": "whisper_large_v3",
                    "audio_path": f"asr/audio/{batch_id}/{audio.name}",
                    "audio_sha256": _sha256(audio), "audio_duration_seconds": 0.0,
                    "segments": [], "error_code": type(exc).__name__,
                    "error_message": str(exc)[:500]
                }
            output.write(json.dumps(envelope, ensure_ascii=False) + "\n")
            output.flush()
            all_rows.append(envelope)
            completed.add(video_id)
            seen.add(video_id)
            state["completed"] = sorted(completed)
            state["batch_id"], state["worker_id"] = batch_id, worker_id
            _save_state(state_file, state)
            videos_since_checkpoint += 1
            if hf_repo_id and hf_token and videos_since_checkpoint >= checkpoint_every:
                _push_checkpoint(hf_repo_id, hf_token, batch_id, destination, state_file)
                videos_since_checkpoint = 0
    audio_files = list(audio_root.glob("*.flac"))
    expected_ids = {audio.stem for audio in audio_files}
    row_ids = [str(row.get("video_id", "")) for row in all_rows]
    duplicate_videos = len(row_ids) - len(set(row_ids))
    foreign_videos = len(set(row_ids) - expected_ids)
    if duplicate_videos or foreign_videos:
        raise RuntimeError("ASR output JSONL contains duplicate or foreign video IDs")
    output_sha256 = _sha256(destination)
    assigned_video_sha256 = _video_set_sha256([audio.stem for audio in audio_files])
    manifest = {
        "schema_version": 1, "batch_id": batch_id, "worker_id": worker_id,
        "catalog_sha256": _sha256(Path(catalog_path)) if catalog_path else "0" * 64,
        "config_sha256": hashlib.sha256(
            f"{model_name}:float16:cuda:vad_filter:condition_on_previous_text=False".encode()
        ).hexdigest(),
        "assigned_video_sha256": assigned_video_sha256,
        "audio_root": f"asr/audio/{batch_id}",
        "engine": "whisper_large_v3",
        "output_jsonl_path": f"asr/archives/{batch_id}/{destination.name}",
        "output_jsonl_sha256": output_sha256,
        "shard_path": f"asr/archives/{batch_id}/{destination.name}",
        "shard_sha256": output_sha256,
        "expected_video_sha256": assigned_video_sha256,
        "record_count": len(all_rows),
        "success_records": sum(row["status"] == "success" for row in all_rows),
        "silent_records": sum(row["status"] == "silent" for row in all_rows),
        "error_records": sum(row["status"] == "error" for row in all_rows),
        "expected_videos": len(audio_files),
        "processed_videos": len(all_rows),
        "success_videos": sum(row["status"] == "success" for row in all_rows),
        "silent_videos": sum(row["status"] == "silent" for row in all_rows),
        "error_videos": sum(row["status"] == "error" for row in all_rows),
        "duplicate_videos": duplicate_videos,
        "missing_videos": len(expected_ids - set(row_ids)),
        "foreign_videos": foreign_videos,
        "completion_gate_passed": len(all_rows) == len(audio_files)
        and not any(row["status"] == "error" for row in all_rows),
    }
    manifest_file = Path(manifest_path or destination.with_suffix(".manifest.json"))
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if hf_repo_id and hf_token:
        from huggingface_hub import CommitOperationAdd, HfApi
        api = HfApi(token=hf_token)
        commit_message = f"ASR archive {batch_id}"
        remote_paths = {
            f"asr/archives/{batch_id}/{destination.name}",
            f"asr/archives/{batch_id}/{manifest_file.name}",
        }
        existing_paths = set(
            api.list_repo_files(repo_id=hf_repo_id, repo_type="dataset")
        )
        present = remote_paths & existing_paths
        if present and present != remote_paths:
            raise RuntimeError("remote ASR archive is partial; refusing overwrite")
        if not present:
            api.create_commit(
                repo_id=hf_repo_id,
                repo_type="dataset",
                operations=[
                    CommitOperationAdd(
                        path_in_repo=f"asr/archives/{batch_id}/{manifest_file.name}",
                        path_or_fileobj=manifest_file,
                    ),
                    CommitOperationAdd(
                        path_in_repo=f"asr/archives/{batch_id}/{destination.name}",
                        path_or_fileobj=destination,
                    ),
                ],
                commit_message=commit_message,
            )
        # Final archive is durable; the in-progress checkpoint is now redundant.
        # Best-effort cleanup only — never fail the run over this.
        try:
            from huggingface_hub import CommitOperationDelete
            checkpoint_paths = {
                f"asr/checkpoints/{batch_id}/{destination.name}",
                f"asr/checkpoints/{batch_id}/{state_file.name}",
            }
            remaining = checkpoint_paths & set(
                api.list_repo_files(repo_id=hf_repo_id, repo_type="dataset")
            )
            if remaining:
                api.create_commit(
                    repo_id=hf_repo_id,
                    repo_type="dataset",
                    operations=[CommitOperationDelete(path_in_repo=path) for path in remaining],
                    commit_message=f"Drop ASR checkpoint {batch_id} (archived)",
                )
        except Exception as exc:  # noqa: BLE001 - cleanup is best-effort
            print(f"[checkpoint] cleanup skipped for {batch_id}: {exc}")
    return destination


def run_batches(
    batch_ids: list[str],
    audio_root: str | Path,
    output_root: str | Path,
    *,
    worker_id: str = "kaggle-01",
    model_name: str = "large-v3",
    catalog_path: str | Path | None = None,
    hf_repo_id: str | None = None,
    hf_token: str | None = None,
    checkpoint_every: int = 10,
) -> None:
    """Process each batch sequentially on whichever GPU CUDA_VISIBLE_DEVICES exposes.

    Intended to run inside its own OS process (one per physical GPU) so that
    CUDA_VISIBLE_DEVICES set before this module is imported pins the faster-whisper
    model to a single device. Batches assigned to one process are still processed
    one-at-a-time to bound peak VRAM.
    """
    for batch_id in batch_ids:
        audio_dir = Path(audio_root) / batch_id
        output_dir = Path(output_root) / batch_id
        output_dir.mkdir(parents=True, exist_ok=True)
        run_production(
            audio_dir,
            output_dir / "asr-envelope.jsonl",
            batch_id=batch_id,
            worker_id=worker_id,
            state_path=output_dir / "batch-checkpoint.json",
            model_name=model_name,
            catalog_path=catalog_path,
            hf_repo_id=hf_repo_id,
            hf_token=hf_token,
            manifest_path=output_dir / "manifest.json",
            checkpoint_every=checkpoint_every,
        )


if __name__ == "__main__":
    _batch_ids_env = os.environ.get("ASR_BATCH_IDS")
    if _batch_ids_env:
        # Multi-batch mode: launched as a subprocess pinned to one GPU via
        # CUDA_VISIBLE_DEVICES, processing a disjoint slice of a worker's batches.
        run_batches(
            [batch.strip() for batch in _batch_ids_env.split(",") if batch.strip()],
            os.environ["ASR_AUDIO_ROOT"],
            os.environ["ASR_OUTPUT_ROOT"],
            worker_id=os.environ.get("ASR_WORKER_ID", "kaggle-01"),
            catalog_path=os.environ.get("ASR_CATALOG_PATH") or None,
            hf_repo_id=os.environ.get("ASR_HF_REPO_ID") or None,
            hf_token=os.environ.get("ASR_HF_TOKEN") or None,
            checkpoint_every=int(os.environ.get("ASR_CHECKPOINT_EVERY", "10")),
        )
    else:
        # Legacy single-batch mode (kept for backward compatibility).
        run_production(
            os.environ["ASR_AUDIO_DIR"], os.environ["ASR_OUTPUT_JSONL"],
            batch_id=os.environ.get("ASR_BATCH_ID", "batch-01")
        )
