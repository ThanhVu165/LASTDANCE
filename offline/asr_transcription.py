"""Hash-bound, interrupt/resume transcription runner for one ASR model."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from offline.asr_artifacts import AsrTranscriptRecord, TranscriptStatus
from offline.asr_audio import AudioArtifact, AudioStatus, validate_audio_artifact
from offline.asr_models import TranscriptionOutput


SCHEMA_VERSION = 1
_PROCESS_TOKEN = f"pid-{os.getpid()}-{uuid.uuid4().hex}"


class IntentionalAsrInterruption(RuntimeError):
    """Raised after a requested video boundary to demonstrate real resume."""


class AsrTranscriber(Protocol):
    model_key: str
    model_id: str
    model_revision: str

    def prepare(self) -> None: ...

    @property
    def weight_sha256(self) -> str: ...

    @property
    def runtime_metadata(self) -> dict[str, object]: ...

    def transcribe(self, audio_path: Path, *, video_id: str) -> TranscriptionOutput: ...


@dataclass(frozen=True, slots=True)
class SelectedAudio:
    artifact: AudioArtifact
    wav_path: Path


@dataclass(frozen=True, slots=True)
class AsrRunResult:
    output_dir: Path
    completed_videos: int
    total_videos: int
    complete: bool
    resumed: bool
    checkpoint_resume_verified: bool


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_json_atomic(path: Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def load_selected_audio(
    *,
    audio_root: Path,
    video_ids: Sequence[str],
) -> list[SelectedAudio]:
    if not video_ids or len(set(video_ids)) != len(video_ids):
        raise ValueError("video_ids must be non-empty and unique")
    root = Path(audio_root).resolve(strict=False)
    selected: list[SelectedAudio] = []
    for video_id in video_ids:
        manifest = root / "manifests" / f"{video_id}.json"
        artifact = validate_audio_artifact(manifest, audio_root=root)
        if artifact.video_id != video_id:
            raise RuntimeError("audio manifest video_id mismatch")
        if artifact.status != AudioStatus.READY or artifact.wav_path is None:
            raise RuntimeError(f"Dev Gate video has no ready audio: {video_id}")
        wav = (root / artifact.wav_path).resolve(strict=True)
        selected.append(SelectedAudio(artifact=artifact, wav_path=wav))
    return selected


def build_transcription_signature(
    *, transcriber: AsrTranscriber, selected: Sequence[SelectedAudio]
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "runner": "transformers_pytorch",
        "model_key": transcriber.model_key,
        "model_id": transcriber.model_id,
        "model_revision": transcriber.model_revision,
        "weight_sha256": transcriber.weight_sha256,
        "audio": [
            {
                "video_id": item.artifact.video_id,
                "wav_sha256": item.artifact.wav_sha256,
                "duration_seconds": item.artifact.wav_duration_seconds,
            }
            for item in selected
        ],
        "timestamps": "segment",
        "task": "transcribe",
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _load_record(path: Path) -> AsrTranscriptRecord:
    try:
        return AsrTranscriptRecord.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"invalid ASR transcript record: {path}") from error


def _scan_records(
    records_dir: Path,
    *,
    selected: Sequence[SelectedAudio],
    transcriber: AsrTranscriber,
    batch_id: str,
) -> int:
    if not records_dir.exists():
        return 0
    expected_ids = [item.artifact.video_id for item in selected]
    existing_paths = sorted(records_dir.glob("*.json"))
    existing_ids = [path.stem for path in existing_paths]
    if existing_ids != expected_ids[: len(existing_ids)]:
        raise RuntimeError("ASR transcript records are not a contiguous selected-video prefix")
    for path, item in zip(existing_paths, selected, strict=False):
        record = _load_record(path)
        artifact = item.artifact
        if (
            record.batch_id != batch_id
            or record.video_id != artifact.video_id
            or record.model_key != transcriber.model_key
            or record.model_id != transcriber.model_id
            or record.model_revision != transcriber.model_revision
            or record.source_wav_sha256 != artifact.wav_sha256
            or record.source_wav != artifact.wav_path
        ):
            raise RuntimeError(f"ASR transcript provenance mismatch: {path}")
    return len(existing_paths)


def run_asr_transcription(
    *,
    transcriber: AsrTranscriber,
    batch_id: str,
    audio_root: Path,
    output_root: Path,
    video_ids: Sequence[str],
    stop_after_videos: int | None = None,
) -> AsrRunResult:
    if not batch_id or batch_id.strip() != batch_id or any(char.isspace() for char in batch_id):
        raise ValueError("batch_id must be canonical and contain no whitespace")
    if stop_after_videos is not None and stop_after_videos <= 0:
        raise ValueError("stop_after_videos must be positive")
    selected = load_selected_audio(audio_root=audio_root, video_ids=video_ids)
    transcriber.prepare()
    signature = build_transcription_signature(transcriber=transcriber, selected=selected)
    output_dir = Path(output_root) / batch_id / transcriber.model_key
    records_dir = output_dir / "records"
    checkpoint_path = output_dir / "checkpoint.json"
    manifest_path = output_dir / "manifest.json"

    checkpoint: dict[str, object] | None = None
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if (
            checkpoint.get("schema_version") != SCHEMA_VERSION
            or checkpoint.get("signature") != signature
            or checkpoint.get("total_videos") != len(selected)
        ):
            raise RuntimeError("ASR checkpoint signature/scope mismatch")
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("ASR output exists without a valid checkpoint")

    completed = _scan_records(
        records_dir,
        selected=selected,
        transcriber=transcriber,
        batch_id=batch_id,
    )
    if checkpoint is not None and int(checkpoint.get("completed_videos", -1)) > completed:
        raise RuntimeError("ASR checkpoint is ahead of validated transcript records")

    resumed = completed > 0 and completed < len(selected)
    interrupted = bool(checkpoint and checkpoint.get("intentional_interruption_observed"))
    interruption_token = (
        str(checkpoint.get("interruption_process_token")) if interrupted else None
    )
    resume_verified = bool(checkpoint and checkpoint.get("checkpoint_resume_verified"))
    if resumed and interrupted and interruption_token != _PROCESS_TOKEN:
        resume_verified = True

    if manifest_path.exists():
        if completed != len(selected):
            raise RuntimeError("ASR final manifest exists while records are incomplete")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("signature") != signature
            or manifest.get("complete") is not True
            or manifest.get("record_count") != len(selected)
        ):
            raise RuntimeError("ASR final manifest failed validation")
        return AsrRunResult(
            output_dir=output_dir,
            completed_videos=completed,
            total_videos=len(selected),
            complete=True,
            resumed=False,
            checkpoint_resume_verified=bool(manifest.get("checkpoint_resume_verified")),
        )

    processed_this_run = 0
    for item in selected[completed:]:
        started = time.perf_counter()
        output = transcriber.transcribe(item.wav_path, video_id=item.artifact.video_id)
        elapsed = time.perf_counter() - started
        status = TranscriptStatus.SUCCESS if output.segments else TranscriptStatus.NO_SPEECH
        record = AsrTranscriptRecord(
            batch_id=batch_id,
            video_id=item.artifact.video_id,
            model_key=transcriber.model_key,
            model_id=transcriber.model_id,
            model_revision=transcriber.model_revision,
            source_wav=str(item.artifact.wav_path),
            source_wav_sha256=str(item.artifact.wav_sha256),
            source_duration_seconds=float(item.artifact.wav_duration_seconds),
            status=status,
            elapsed_seconds=elapsed,
            segments=list(output.segments),
        )
        _write_json_atomic(
            records_dir / f"{item.artifact.video_id}.json",
            record.model_dump(mode="json"),
        )
        completed += 1
        processed_this_run += 1
        checkpoint_payload = {
            "schema_version": SCHEMA_VERSION,
            "signature": signature,
            "model_key": transcriber.model_key,
            "completed_videos": completed,
            "total_videos": len(selected),
            "intentional_interruption_observed": interrupted,
            "interruption_process_token": interruption_token,
            "checkpoint_resume_verified": resume_verified,
            "complete": False,
        }
        _write_json_atomic(checkpoint_path, checkpoint_payload)
        if stop_after_videos is not None and processed_this_run >= stop_after_videos:
            checkpoint_payload.update(
                {
                    "intentional_interruption_observed": True,
                    "interruption_process_token": _PROCESS_TOKEN,
                }
            )
            _write_json_atomic(checkpoint_path, checkpoint_payload)
            raise IntentionalAsrInterruption(
                f"intentional ASR stop after {processed_this_run} video(s); rerun in a new process"
            )

    records = [
        _load_record(records_dir / f"{item.artifact.video_id}.json") for item in selected
    ]
    no_speech = sum(record.status == TranscriptStatus.NO_SPEECH for record in records)
    runtime = dict(transcriber.runtime_metadata)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "signature": signature,
        "batch_id": batch_id,
        "model": {
            "key": transcriber.model_key,
            "id": transcriber.model_id,
            "revision": transcriber.model_revision,
            "weight_sha256": transcriber.weight_sha256,
        },
        "video_ids": [item.artifact.video_id for item in selected],
        "record_count": len(records),
        "segment_count": sum(len(record.segments) for record in records),
        "no_speech_records": no_speech,
        "audio_duration_seconds": sum(record.source_duration_seconds for record in records),
        "inference_seconds": sum(record.elapsed_seconds for record in records),
        "runtime": runtime,
        "checkpoint_resume_verified": resume_verified,
        "complete": True,
    }
    _write_json_atomic(manifest_path, manifest)
    _write_json_atomic(
        checkpoint_path,
        {
            "schema_version": SCHEMA_VERSION,
            "signature": signature,
            "model_key": transcriber.model_key,
            "completed_videos": len(selected),
            "total_videos": len(selected),
            "intentional_interruption_observed": interrupted,
            "interruption_process_token": interruption_token,
            "checkpoint_resume_verified": resume_verified,
            "complete": True,
        },
    )
    return AsrRunResult(
        output_dir=output_dir,
        completed_videos=len(selected),
        total_videos=len(selected),
        complete=True,
        resumed=resumed,
        checkpoint_resume_verified=resume_verified,
    )
