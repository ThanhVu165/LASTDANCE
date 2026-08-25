"""Resumable, modality-independent visual embedding artifacts.

Each invocation owns exactly one modality namespace.  Shards are published as
atomic directories and are keyed by the canonical ``keyframe_uid`` values from
``frames.csv``; no insertion-order ID is ever persisted.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from offline.artifacts import sha256_file
from offline.catalog import FRAME_COLUMNS, validate_frames_catalog
from shared.schemas.frame import FrameRecord


SUPPORTED_MODALITIES = ("clip", "siglip", "beit3")
SCHEMA_VERSION = 1
_NORM_ATOL = 5e-3
_PROCESS_TOKEN = f"pid-{os.getpid()}-{uuid.uuid4().hex}"


class IntentionalEmbeddingInterruption(RuntimeError):
    """Raised after a requested shard boundary to exercise real resume logic."""


class VisualEncoder(Protocol):
    """Minimal model interface kept independent from Torch/Transformers imports."""

    modality: str
    model_id: str
    model_revision: str

    def encode(self, image_paths: Sequence[Path]) -> np.ndarray:
        """Return a finite two-dimensional float-compatible array."""


@dataclass(frozen=True, slots=True)
class CatalogKeyframe:
    frame: FrameRecord
    image_path: Path


@dataclass(frozen=True, slots=True)
class EmbeddingRunResult:
    output_dir: Path
    next_index: int
    total: int
    complete: bool
    resumed: bool
    checkpoint_resume_verified: bool


@dataclass(frozen=True, slots=True)
class CompletedEmbeddingData:
    """Validated arrays and provenance for one completed modality batch."""

    artifact_dir: Path
    modality: str
    batch_id: str
    signature: str
    model_id: str
    model_revision: str
    catalog_sha256: str
    video_ids: tuple[str, ...]
    keyframe_uids: np.ndarray
    vectors: np.ndarray
    artifact_content_sha256: str
    checkpoint_resume_verified: bool


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_component(value: str, *, name: str) -> str:
    if not value or value.strip() != value:
        raise ValueError(f"{name} must be non-empty and must not have edge whitespace")
    if value in {".", ".."} or any(character in value for character in "/\\"):
        raise ValueError(f"{name} must be a single safe path component")
    return value


def _image_path(keyframes_root: Path, frame: FrameRecord) -> Path:
    root = Path(keyframes_root).resolve(strict=False)
    candidate = (
        root / frame.video_id / f"{frame.shot_id}_{frame.local_idx}.jpg"
    ).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise RuntimeError(
            f"keyframe path escapes AIC_DATA/keyframes for UID {frame.keyframe_uid}"
        ) from error
    return candidate


def load_embedding_catalog(
    catalog_path: Path,
    *,
    keyframes_root: Path,
    video_ids: set[str] | None = None,
) -> list[CatalogKeyframe]:
    """Load a hash-bound catalog and resolve its conventional JPEG paths."""

    source = Path(catalog_path)
    if not validate_frames_catalog(source):
        raise RuntimeError("frames.csv or its hash-bound state is incomplete/invalid")
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FRAME_COLUMNS:
            raise RuntimeError("frames.csv columns do not match the canonical schema")
        frames = [
            FrameRecord(**{**row, "window_id": row["window_id"] or None})
            for row in reader
        ]

    if video_ids is not None:
        missing = sorted(video_ids - {frame.video_id for frame in frames})
        if missing:
            raise RuntimeError(f"requested video IDs are absent from catalog: {missing[:5]}")
        frames = [frame for frame in frames if frame.video_id in video_ids]
    if not frames:
        raise RuntimeError("embedding selection contains no keyframes")

    uids = [frame.keyframe_uid for frame in frames]
    if len(set(uids)) != len(uids):
        raise RuntimeError("embedding selection contains duplicate keyframe_uid values")
    selected = [
        CatalogKeyframe(frame=frame, image_path=_image_path(keyframes_root, frame))
        for frame in frames
    ]
    missing_images = [str(item.image_path) for item in selected if not item.image_path.is_file()]
    if missing_images:
        raise RuntimeError(
            f"missing {len(missing_images)} keyframe JPEG(s); first: {missing_images[0]}"
        )
    return selected


def _uid_sequence_sha256(items: Sequence[CatalogKeyframe]) -> str:
    return _sha256_text("\n".join(str(item.frame.keyframe_uid) for item in items) + "\n")


def build_embedding_signature(
    *,
    modality: str,
    model_id: str,
    model_revision: str,
    catalog_sha256: str,
    selected_uid_sha256: str,
    batch_size: int,
) -> str:
    return _sha256_text(
        _canonical_json(
            {
                "schema_version": SCHEMA_VERSION,
                "modality": modality,
                "model_id": model_id,
                "model_revision": model_revision,
                "catalog_sha256": catalog_sha256,
                "selected_uid_sha256": selected_uid_sha256,
                "batch_size": batch_size,
                "normalization": "l2_float32_then_cast_float16",
            }
        )
    )


def normalize_vectors(vectors: np.ndarray, *, expected_rows: int) -> np.ndarray:
    """Validate, L2-normalize in float32, then store as required float16."""

    values = np.asarray(vectors)
    if values.ndim != 2 or values.shape[0] != expected_rows or values.shape[1] <= 0:
        raise RuntimeError(
            f"encoder returned shape {values.shape}; expected ({expected_rows}, dim>0)"
        )
    values = values.astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise RuntimeError("encoder output contains NaN or Inf")
    norms = np.linalg.norm(values, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise RuntimeError("encoder output contains zero or invalid vector norms")
    stored = (values / norms[:, None]).astype(np.float16)
    stored_norms = np.linalg.norm(stored.astype(np.float32), axis=1)
    if not np.isfinite(stored).all() or not np.allclose(
        stored_norms, np.ones_like(stored_norms), atol=_NORM_ATOL, rtol=0
    ):
        raise RuntimeError("float16 vectors failed finite/L2-norm validation")
    return stored


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _checkpoint_payload(
    *,
    signature: str,
    modality: str,
    next_index: int,
    total: int,
    completed_shards: int,
    resume_count: int,
    intentional_interruption_observed: bool,
    interruption_process_token: str | None,
    checkpoint_resume_verified: bool,
    complete: bool,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "signature": signature,
        "modality": modality,
        "next_index": next_index,
        "total": total,
        "completed_shards": completed_shards,
        "resume_count": resume_count,
        "intentional_interruption_observed": intentional_interruption_observed,
        "interruption_process_token": interruption_process_token,
        "checkpoint_resume_verified": checkpoint_resume_verified,
        "complete": complete,
    }


def _read_checkpoint(path: Path, *, signature: str, modality: str, total: int) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("unsupported visual embedding checkpoint schema")
    if payload.get("signature") != signature:
        raise RuntimeError("visual embedding checkpoint signature mismatch")
    if payload.get("modality") != modality or payload.get("total") != total:
        raise RuntimeError("visual embedding checkpoint scope mismatch")
    return payload


def _validate_shard(
    shard_dir: Path,
    *,
    signature: str,
    expected_index: int,
    expected_start: int,
    expected_items: Sequence[CatalogKeyframe],
    expected_dim: int | None,
) -> tuple[int, int]:
    manifest_path = shard_dir / "manifest.json"
    uids_path = shard_dir / "keyframe_uids.npy"
    vectors_path = shard_dir / "vectors.npy"
    if not manifest_path.is_file() or not uids_path.is_file() or not vectors_path.is_file():
        raise RuntimeError(f"incomplete visual embedding shard: {shard_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_uids = np.asarray(
        [item.frame.keyframe_uid for item in expected_items], dtype=np.int64
    )
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("signature") != signature
        or manifest.get("shard_index") != expected_index
        or manifest.get("start_index") != expected_start
        or manifest.get("count") != len(expected_items)
        or manifest.get("uids_sha256") != sha256_file(uids_path)
        or manifest.get("vectors_sha256") != sha256_file(vectors_path)
    ):
        raise RuntimeError(f"visual embedding shard manifest mismatch: {shard_dir}")
    uids = np.load(uids_path, allow_pickle=False)
    vectors = np.load(vectors_path, allow_pickle=False)
    if uids.dtype != np.int64 or not np.array_equal(uids, expected_uids):
        raise RuntimeError(f"visual embedding shard UID mismatch: {shard_dir}")
    if vectors.dtype != np.float16 or vectors.ndim != 2 or vectors.shape[0] != len(uids):
        raise RuntimeError(f"visual embedding shard vector shape/dtype mismatch: {shard_dir}")
    if expected_dim is not None and vectors.shape[1] != expected_dim:
        raise RuntimeError(f"visual embedding dimension changed across shards: {shard_dir}")
    norms = np.linalg.norm(vectors.astype(np.float32), axis=1)
    if not np.isfinite(vectors).all() or not np.allclose(
        norms, np.ones_like(norms), atol=_NORM_ATOL, rtol=0
    ):
        raise RuntimeError(f"visual embedding shard failed vector health checks: {shard_dir}")
    return expected_start + len(expected_items), int(vectors.shape[1])


def _scan_shards(
    shards_dir: Path,
    *,
    signature: str,
    items: Sequence[CatalogKeyframe],
    batch_size: int,
) -> tuple[int, int, int | None]:
    if not shards_dir.exists():
        return 0, 0, None
    shard_dirs = sorted(
        path for path in shards_dir.iterdir() if path.is_dir() and path.name.isdigit()
    )
    next_index = 0
    vector_dim: int | None = None
    for expected_shard_index, shard_dir in enumerate(shard_dirs):
        if shard_dir.name != f"{expected_shard_index:06d}":
            raise RuntimeError("visual embedding shard numbering contains a gap")
        expected_items = items[next_index : next_index + batch_size]
        if not expected_items:
            raise RuntimeError("visual embedding output contains an unexpected extra shard")
        next_index, vector_dim = _validate_shard(
            shard_dir,
            signature=signature,
            expected_index=expected_shard_index,
            expected_start=next_index,
            expected_items=expected_items,
            expected_dim=vector_dim,
        )
    return next_index, len(shard_dirs), vector_dim


def _publish_shard(
    shards_dir: Path,
    *,
    signature: str,
    shard_index: int,
    start_index: int,
    items: Sequence[CatalogKeyframe],
    vectors: np.ndarray,
) -> None:
    destination = shards_dir / f"{shard_index:06d}"
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite existing shard: {destination}")
    shards_dir.mkdir(parents=True, exist_ok=True)
    temporary = shards_dir / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        uids_path = temporary / "keyframe_uids.npy"
        vectors_path = temporary / "vectors.npy"
        np.save(
            uids_path,
            np.asarray([item.frame.keyframe_uid for item in items], dtype=np.int64),
            allow_pickle=False,
        )
        np.save(vectors_path, vectors, allow_pickle=False)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "signature": signature,
            "shard_index": shard_index,
            "start_index": start_index,
            "end_index_exclusive": start_index + len(items),
            "count": len(items),
            "vector_dim": int(vectors.shape[1]),
            "vector_dtype": "float16",
            "uids_sha256": sha256_file(uids_path),
            "vectors_sha256": sha256_file(vectors_path),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _validate_encoder(encoder: VisualEncoder, modality: str) -> None:
    if modality not in SUPPORTED_MODALITIES:
        raise ValueError(f"unsupported modality: {modality}")
    if encoder.modality != modality:
        raise RuntimeError("encoder modality does not match requested artifact namespace")
    if not encoder.model_id.strip() or not encoder.model_revision.strip():
        raise RuntimeError("encoder model ID and immutable revision are required")
    revision = encoder.model_revision.lower()
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise RuntimeError("model_revision must be an immutable 40-character commit SHA")


def _runtime_metadata(encoder: VisualEncoder) -> dict[str, object]:
    metadata = getattr(encoder, "runtime_metadata", {})
    if callable(metadata):
        metadata = metadata()
    if not isinstance(metadata, dict):
        raise RuntimeError("encoder runtime_metadata must be a mapping")
    return dict(metadata)


def run_visual_embedding(
    *,
    encoder: VisualEncoder,
    modality: str,
    batch_id: str,
    catalog_path: Path,
    keyframes_root: Path,
    output_root: Path,
    batch_size: int,
    video_ids: set[str] | None = None,
    stop_after_shards: int | None = None,
) -> EmbeddingRunResult:
    """Build or resume one modality without consulting any other modality."""

    _validate_encoder(encoder, modality)
    _validate_component(batch_id, name="batch_id")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if stop_after_shards is not None and stop_after_shards <= 0:
        raise ValueError("stop_after_shards must be positive")

    catalog = Path(catalog_path)
    items = load_embedding_catalog(
        catalog, keyframes_root=keyframes_root, video_ids=video_ids
    )
    total = len(items)
    catalog_sha256 = sha256_file(catalog)
    selected_uid_sha256 = _uid_sequence_sha256(items)
    signature = build_embedding_signature(
        modality=modality,
        model_id=encoder.model_id,
        model_revision=encoder.model_revision,
        catalog_sha256=catalog_sha256,
        selected_uid_sha256=selected_uid_sha256,
        batch_size=batch_size,
    )
    output_dir = Path(output_root) / batch_id / modality
    shards_dir = output_dir / "shards"
    checkpoint_path = output_dir / "checkpoint.json"
    final_manifest_path = output_dir / "manifest.json"

    checkpoint: dict[str, object] | None = None
    if checkpoint_path.exists():
        checkpoint = _read_checkpoint(
            checkpoint_path, signature=signature, modality=modality, total=total
        )
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("embedding output exists without a valid checkpoint")

    next_index, completed_shards, vector_dim = _scan_shards(
        shards_dir,
        signature=signature,
        items=items,
        batch_size=batch_size,
    )
    if checkpoint is not None and int(checkpoint.get("next_index", -1)) > next_index:
        raise RuntimeError("checkpoint is ahead of validated embedding shards")
    if final_manifest_path.exists() and next_index != total:
        raise RuntimeError("final embedding manifest exists while shards are incomplete")
    if (
        final_manifest_path.exists()
        and checkpoint is not None
        and checkpoint.get("complete") is True
        and next_index == total
    ):
        manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))
        manifest_model = manifest.get("model")
        manifest_catalog = manifest.get("catalog")
        if (
            manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("signature") != signature
            or manifest.get("complete") is not True
            or manifest.get("modality") != modality
            or manifest.get("record_count") != total
            or manifest.get("batch_size") != batch_size
            or manifest.get("shard_count") != completed_shards
            or manifest.get("vector_dim") != vector_dim
            or not isinstance(manifest_model, dict)
            or manifest_model.get("id") != encoder.model_id
            or manifest_model.get("revision") != encoder.model_revision
            or not isinstance(manifest_catalog, dict)
            or manifest_catalog.get("sha256") != catalog_sha256
            or manifest_catalog.get("selected_uid_sha256") != selected_uid_sha256
            or bool(manifest.get("checkpoint_resume_verified", False))
            != bool(checkpoint.get("checkpoint_resume_verified", False))
        ):
            raise RuntimeError("final visual embedding manifest failed validation")
        return EmbeddingRunResult(
            output_dir=output_dir,
            next_index=total,
            total=total,
            complete=True,
            resumed=False,
            checkpoint_resume_verified=bool(
                checkpoint.get("checkpoint_resume_verified", False)
            ),
        )

    resumed = checkpoint is not None and next_index > 0 and next_index < total
    resume_count = int(checkpoint.get("resume_count", 0)) if checkpoint else 0
    interrupted = (
        bool(checkpoint.get("intentional_interruption_observed", False))
        if checkpoint
        else False
    )
    interruption_process_token = (
        str(checkpoint.get("interruption_process_token"))
        if checkpoint and checkpoint.get("interruption_process_token")
        else None
    )
    resume_verified = (
        bool(checkpoint.get("checkpoint_resume_verified", False))
        if checkpoint
        else False
    )
    if resumed:
        resume_count += 1
        if interrupted and interruption_process_token != _PROCESS_TOKEN:
            resume_verified = True

    _write_json_atomic(
        checkpoint_path,
        _checkpoint_payload(
            signature=signature,
            modality=modality,
            next_index=next_index,
            total=total,
            completed_shards=completed_shards,
            resume_count=resume_count,
            intentional_interruption_observed=interrupted,
            interruption_process_token=interruption_process_token,
            checkpoint_resume_verified=resume_verified,
            complete=False,
        ),
    )

    while next_index < total:
        batch_items = items[next_index : next_index + batch_size]
        encoded = encoder.encode([item.image_path for item in batch_items])
        vectors = normalize_vectors(encoded, expected_rows=len(batch_items))
        if vector_dim is not None and vectors.shape[1] != vector_dim:
            raise RuntimeError("encoder vector dimension changed between batches")
        vector_dim = int(vectors.shape[1])
        _publish_shard(
            shards_dir,
            signature=signature,
            shard_index=completed_shards,
            start_index=next_index,
            items=batch_items,
            vectors=vectors,
        )
        next_index += len(batch_items)
        completed_shards += 1
        should_interrupt = (
            stop_after_shards is not None
            and completed_shards >= stop_after_shards
            and next_index < total
        )
        if should_interrupt:
            interrupted = True
            interruption_process_token = _PROCESS_TOKEN
        _write_json_atomic(
            checkpoint_path,
            _checkpoint_payload(
                signature=signature,
                modality=modality,
                next_index=next_index,
                total=total,
                completed_shards=completed_shards,
                resume_count=resume_count,
                intentional_interruption_observed=interrupted,
                interruption_process_token=interruption_process_token,
                checkpoint_resume_verified=resume_verified,
                complete=False,
            ),
        )
        if should_interrupt:
            raise IntentionalEmbeddingInterruption(
                f"intentional interruption after {completed_shards} shard(s); "
                f"resume from {next_index}/{total} with the same command minus "
                "--stop-after-shards"
            )

    if vector_dim is None or not math.ceil(total / batch_size) == completed_shards:
        raise RuntimeError("embedding shard accounting is inconsistent")
    video_counts = Counter(item.frame.video_id for item in items)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "scope": "single_modality_batch",
        "complete": True,
        "signature": signature,
        "batch_id": batch_id,
        "modality": modality,
        "model": {
            "id": encoder.model_id,
            "revision": encoder.model_revision,
        },
        "runtime": _runtime_metadata(encoder),
        "catalog": {
            "sha256": catalog_sha256,
            "selected_uid_sha256": selected_uid_sha256,
        },
        "record_count": total,
        "batch_size": batch_size,
        "vector_dim": vector_dim,
        "vector_dtype": "float16",
        "normalization": "l2_float32_then_cast_float16",
        "shard_count": completed_shards,
        "videos": [
            {"video_id": video_id, "record_count": count, "complete": True}
            for video_id, count in sorted(video_counts.items())
        ],
        "checkpoint_resume_verified": resume_verified,
    }
    _write_json_atomic(final_manifest_path, manifest)
    _write_json_atomic(
        checkpoint_path,
        _checkpoint_payload(
            signature=signature,
            modality=modality,
            next_index=total,
            total=total,
            completed_shards=completed_shards,
            resume_count=resume_count,
            intentional_interruption_observed=interrupted,
            interruption_process_token=interruption_process_token,
            checkpoint_resume_verified=resume_verified,
            complete=True,
        ),
    )
    return EmbeddingRunResult(
        output_dir=output_dir,
        next_index=total,
        total=total,
        complete=True,
        resumed=resumed,
        checkpoint_resume_verified=resume_verified,
    )


def validate_completed_visual_embedding(
    artifact_dir: Path,
    *,
    catalog_path: Path,
    keyframes_root: Path,
    require_resume_verified: bool = False,
) -> EmbeddingRunResult:
    """Revalidate a completed modality artifact without loading its model."""

    output_dir = Path(artifact_dir)
    final_manifest_path = output_dir / "manifest.json"
    checkpoint_path = output_dir / "checkpoint.json"
    if not final_manifest_path.is_file() or not checkpoint_path.is_file():
        raise RuntimeError("completed visual embedding manifest/checkpoint is missing")
    manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("scope") != "single_modality_batch"
        or manifest.get("complete") is not True
    ):
        raise RuntimeError("visual embedding manifest is not complete")
    modality = str(manifest.get("modality", ""))
    model = manifest.get("model")
    manifest_catalog = manifest.get("catalog")
    videos = manifest.get("videos")
    if (
        modality not in SUPPORTED_MODALITIES
        or not isinstance(model, dict)
        or not isinstance(manifest_catalog, dict)
        or not isinstance(videos, list)
        or not videos
    ):
        raise RuntimeError("visual embedding manifest provenance is incomplete")
    model_id = str(model.get("id", ""))
    model_revision = str(model.get("revision", ""))
    batch_id = str(manifest.get("batch_id", ""))
    if (
        not model_id
        or model_id.strip() != model_id
        or len(model_revision) != 40
        or any(character not in "0123456789abcdef" for character in model_revision.lower())
        or not batch_id
        or batch_id.strip() != batch_id
    ):
        raise RuntimeError("visual embedding manifest model/batch provenance is invalid")
    batch_size = int(manifest.get("batch_size", 0))
    if batch_size <= 0:
        raise RuntimeError("visual embedding manifest batch_size is invalid")
    if any(
        not isinstance(row, dict)
        or row.get("complete") is not True
        or type(row.get("record_count")) is not int
        or int(row["record_count"]) <= 0
        for row in videos
    ):
        raise RuntimeError("visual embedding manifest video list is invalid")
    video_ids = {str(row["video_id"]) for row in videos}
    if len(video_ids) != len(videos) or any(not value for value in video_ids):
        raise RuntimeError("visual embedding manifest video list is invalid")

    catalog = Path(catalog_path)
    items = load_embedding_catalog(
        catalog, keyframes_root=keyframes_root, video_ids=video_ids
    )
    total = len(items)
    expected_video_counts = Counter(item.frame.video_id for item in items)
    manifest_video_counts = {
        str(row["video_id"]): int(row["record_count"]) for row in videos
    }
    if manifest_video_counts != dict(expected_video_counts):
        raise RuntimeError("visual embedding manifest video counts do not match frames.csv")
    catalog_sha256 = sha256_file(catalog)
    selected_uid_sha256 = _uid_sequence_sha256(items)
    signature = build_embedding_signature(
        modality=modality,
        model_id=model_id,
        model_revision=model_revision,
        catalog_sha256=catalog_sha256,
        selected_uid_sha256=selected_uid_sha256,
        batch_size=batch_size,
    )
    if (
        manifest.get("signature") != signature
        or manifest.get("record_count") != total
        or manifest_catalog.get("sha256") != catalog_sha256
        or manifest_catalog.get("selected_uid_sha256") != selected_uid_sha256
    ):
        raise RuntimeError("visual embedding manifest does not match frames.csv selection")
    next_index, shard_count, vector_dim = _scan_shards(
        output_dir / "shards",
        signature=signature,
        items=items,
        batch_size=batch_size,
    )
    checkpoint = _read_checkpoint(
        checkpoint_path,
        signature=signature,
        modality=modality,
        total=total,
    )
    resume_verified = bool(checkpoint.get("checkpoint_resume_verified", False))
    if (
        next_index != total
        or checkpoint.get("complete") is not True
        or checkpoint.get("next_index") != total
        or manifest.get("shard_count") != shard_count
        or manifest.get("vector_dim") != vector_dim
        or bool(manifest.get("checkpoint_resume_verified", False)) != resume_verified
    ):
        raise RuntimeError("visual embedding final artifact accounting is inconsistent")
    if require_resume_verified and not resume_verified:
        raise RuntimeError("checkpoint/resume has not been verified by a real interrupted run")
    return EmbeddingRunResult(
        output_dir=output_dir,
        next_index=total,
        total=total,
        complete=True,
        resumed=False,
        checkpoint_resume_verified=resume_verified,
    )


def load_completed_visual_embedding_data(
    artifact_dir: Path,
    *,
    catalog_path: Path,
    keyframes_root: Path,
    require_resume_verified: bool = True,
) -> CompletedEmbeddingData:
    """Load actual shard arrays only after a full catalog-bound validation.

    This is the handoff used by the local FAISS builder.  It deliberately reads
    ``keyframe_uid`` from the published shards/catalog and never recomputes the
    deterministic hash or substitutes a positional row ID.
    """

    result = validate_completed_visual_embedding(
        artifact_dir,
        catalog_path=catalog_path,
        keyframes_root=keyframes_root,
        require_resume_verified=require_resume_verified,
    )
    output_dir = Path(artifact_dir)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    shard_dirs = sorted(
        path
        for path in (output_dir / "shards").iterdir()
        if path.is_dir() and path.name.isdigit()
    )
    uids = np.concatenate(
        [np.load(path / "keyframe_uids.npy", allow_pickle=False) for path in shard_dirs]
    )
    vectors = np.concatenate(
        [np.load(path / "vectors.npy", allow_pickle=False) for path in shard_dirs]
    )
    if (
        uids.dtype != np.int64
        or uids.ndim != 1
        or len(uids) != result.total
        or len(set(uids.tolist())) != len(uids)
    ):
        raise RuntimeError("completed visual embedding UID arrays are invalid")
    if (
        vectors.dtype != np.float16
        or vectors.ndim != 2
        or vectors.shape[0] != len(uids)
        or int(manifest["vector_dim"]) != vectors.shape[1]
    ):
        raise RuntimeError("completed visual embedding vector arrays are invalid")
    norms = np.linalg.norm(vectors.astype(np.float32), axis=1)
    if not np.isfinite(vectors).all() or not np.allclose(
        norms, np.ones_like(norms), atol=_NORM_ATOL, rtol=0
    ):
        raise RuntimeError("completed visual embedding vectors failed health checks")

    model = manifest["model"]
    videos = tuple(sorted(str(row["video_id"]) for row in manifest["videos"]))
    content_digest = hashlib.sha256()
    for array in (uids, vectors):
        contiguous = np.ascontiguousarray(array)
        content_digest.update(str(contiguous.dtype).encode("ascii"))
        content_digest.update(_canonical_json(contiguous.shape).encode("ascii"))
        content_digest.update(contiguous.tobytes())
    return CompletedEmbeddingData(
        artifact_dir=output_dir,
        modality=str(manifest["modality"]),
        batch_id=str(manifest["batch_id"]),
        signature=str(manifest["signature"]),
        model_id=str(model["id"]),
        model_revision=str(model["revision"]),
        catalog_sha256=str(manifest["catalog"]["sha256"]),
        video_ids=videos,
        keyframe_uids=uids,
        vectors=vectors,
        artifact_content_sha256=content_digest.hexdigest(),
        checkpoint_resume_verified=result.checkpoint_resume_verified,
    )
