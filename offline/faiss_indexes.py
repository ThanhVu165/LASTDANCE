"""Build and validate independent frame-level FAISS indexes.

Each call owns exactly one visual modality.  The persistent FAISS IDs are the
canonical ``keyframe_uid`` values already present in ``frames.csv`` and the
embedding shards; insertion order is never exposed as an application key.
"""

from __future__ import annotations

import csv
import hashlib
import json
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from offline.artifacts import sha256_file
from offline.catalog import FRAME_COLUMNS, validate_frames_catalog
from offline.visual_embeddings import (
    SUPPORTED_MODALITIES,
    CompletedEmbeddingData,
    load_completed_visual_embedding_data,
)
from shared.schemas.frame import FrameRecord


SCHEMA_VERSION = 1
INDEX_FACTORY = "IndexIDMap(IndexFlatIP)"
_NORM_ATOL = 5e-4


@dataclass(frozen=True, slots=True)
class FaissIndexReport:
    index_path: Path
    state_path: Path
    modality: str
    record_count: int
    video_count: int
    vector_dim: int
    source_count: int


@dataclass(frozen=True, slots=True)
class FaissBuildResult:
    report: FaissIndexReport
    added_records: int
    added_sources: int


def _import_faiss() -> Any:
    try:
        import faiss
    except ImportError as error:
        raise RuntimeError(
            "faiss-cpu is required; install the requirements/index.txt profile"
        ) from error
    return faiss


def _state_path(index_path: Path) -> Path:
    return Path(index_path).with_name(f"{Path(index_path).name}.state.json")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_hex_digest(value: object, *, length: int) -> bool:
    text = str(value)
    return len(text) == length and all(
        character in "0123456789abcdef" for character in text.lower()
    )


def _uid_sequence_sha256(uids: np.ndarray) -> str:
    return _sha256_text("\n".join(str(int(uid)) for uid in uids) + "\n")


def _uid_set_sha256(uids: np.ndarray) -> str:
    values = np.sort(np.asarray(uids, dtype=np.int64))
    return _uid_sequence_sha256(values)


def _load_catalog_records(catalog_path: Path) -> list[FrameRecord]:
    source = Path(catalog_path)
    if not validate_frames_catalog(source):
        raise RuntimeError("frames.csv or its hash-bound state is incomplete/invalid")
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FRAME_COLUMNS:
            raise RuntimeError("frames.csv columns do not match the canonical schema")
        records = [
            FrameRecord(**{**row, "window_id": row["window_id"] or None})
            for row in reader
        ]
    return records


def _index_arrays(index: Any, faiss: Any) -> tuple[np.ndarray, np.ndarray]:
    if type(index).__name__ != "IndexIDMap":
        raise RuntimeError("FAISS index must be exactly IndexIDMap")
    base = faiss.downcast_index(index.index)
    if type(base).__name__ != "IndexFlatIP" or index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise RuntimeError("FAISS base index must be IndexFlatIP")
    ids = faiss.vector_to_array(index.id_map).astype(np.int64, copy=False)
    if len(ids) != index.ntotal:
        raise RuntimeError("FAISS ID map length does not match ntotal")
    vectors = np.asarray(base.reconstruct_n(0, index.ntotal), dtype=np.float32)
    if vectors.shape != (index.ntotal, index.d):
        raise RuntimeError("FAISS stored vector shape is invalid")
    return ids, vectors


def _prepare_vectors(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors)
    if values.dtype != np.float16 or values.ndim != 2 or values.shape[0] <= 0:
        raise RuntimeError("FAISS source vectors must be a non-empty float16 matrix")
    prepared = values.astype(np.float32)
    if not np.isfinite(prepared).all():
        raise RuntimeError("FAISS source vectors contain NaN or Inf")
    norms = np.linalg.norm(prepared, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise RuntimeError("FAISS source vectors contain zero or invalid norms")
    prepared /= norms[:, None]
    normalized_norms = np.linalg.norm(prepared, axis=1)
    if not np.allclose(
        normalized_norms,
        np.ones_like(normalized_norms),
        atol=_NORM_ATOL,
        rtol=0,
    ):
        raise RuntimeError("FAISS source vectors failed L2 normalization")
    return np.ascontiguousarray(prepared, dtype=np.float32)


def _source_entry(source: CompletedEmbeddingData) -> dict[str, object]:
    return {
        "signature": source.signature,
        "batch_id": source.batch_id,
        "record_count": int(len(source.keyframe_uids)),
        "uid_sequence_sha256": _uid_sequence_sha256(source.keyframe_uids),
        "artifact_content_sha256": source.artifact_content_sha256,
        "video_ids": list(source.video_ids),
        "checkpoint_resume_verified": source.checkpoint_resume_verified,
    }


def _validate_state_structure(state: object) -> dict[str, object]:
    if not isinstance(state, dict):
        raise RuntimeError("FAISS state must be a JSON object")
    if (
        state.get("schema_version") != SCHEMA_VERSION
        or state.get("scope") != "single_modality_index"
        or state.get("complete") is not True
        or state.get("index_factory") != INDEX_FACTORY
        or state.get("metric") != "inner_product"
        or state.get("normalization") != "l2_float32_before_add"
        or state.get("source_vector_dtype") != "float16"
        or state.get("faiss_vector_dtype") != "float32"
        or state.get("modality") not in SUPPORTED_MODALITIES
    ):
        raise RuntimeError("FAISS state contract is invalid")
    return state


def validate_faiss_index(
    index_path: Path,
    *,
    catalog_path: Path,
    state_path: Path | None = None,
    expected_modality: str | None = None,
) -> FaissIndexReport:
    """Diff actual FAISS IDs and vectors against the claimed catalog videos."""

    faiss = _import_faiss()
    index_file = Path(index_path)
    state_file = Path(state_path) if state_path is not None else _state_path(index_file)
    if not index_file.is_file() or not state_file.is_file():
        raise RuntimeError("FAISS index/state pair is incomplete")
    try:
        state = _validate_state_structure(
            json.loads(state_file.read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("FAISS state is unreadable") from error

    modality = str(state["modality"])
    if expected_modality is not None and modality != expected_modality:
        raise RuntimeError("FAISS state modality does not match the requested namespace")
    if state.get("index_sha256") != sha256_file(index_file):
        raise RuntimeError("FAISS index SHA-256 does not match its complete state")

    catalog = Path(catalog_path)
    if state.get("catalog_sha256") != sha256_file(catalog):
        raise RuntimeError("FAISS state was built against a different frames.csv")
    records = _load_catalog_records(catalog)

    videos = state.get("videos")
    sources = state.get("sources")
    model = state.get("model")
    if (
        not isinstance(videos, list)
        or not videos
        or not isinstance(sources, list)
        or not sources
        or not isinstance(model, dict)
        or not str(model.get("id", ""))
        or not _is_hex_digest(model.get("revision", ""), length=40)
    ):
        raise RuntimeError("FAISS state provenance is incomplete")
    if any(
        type(state.get(field)) is not int or int(state[field]) <= 0
        for field in ("record_count", "video_count", "vector_dim")
    ):
        raise RuntimeError("FAISS state counts/dimension are invalid")
    if any(
        not isinstance(row, dict)
        or row.get("complete") is not True
        or not str(row.get("video_id", ""))
        or type(row.get("record_count")) is not int
        or int(row["record_count"]) <= 0
        for row in videos
    ):
        raise RuntimeError("FAISS per-video state is invalid")
    video_ids = [str(row["video_id"]) for row in videos]
    if len(set(video_ids)) != len(video_ids) or video_ids != sorted(video_ids):
        raise RuntimeError("FAISS per-video state must be unique and sorted")

    selected = [record for record in records if record.video_id in set(video_ids)]
    expected_counts = Counter(record.video_id for record in selected)
    claimed_counts = {
        str(row["video_id"]): int(row["record_count"]) for row in videos
    }
    if claimed_counts != dict(expected_counts):
        raise RuntimeError("FAISS per-video counts do not match frames.csv")
    expected_uids = np.asarray([record.keyframe_uid for record in selected], dtype=np.int64)
    if len(expected_uids) == 0 or len(set(expected_uids.tolist())) != len(expected_uids):
        raise RuntimeError("FAISS catalog selection has no unique keyframe IDs")

    try:
        index = faiss.read_index(str(index_file))
    except RuntimeError as error:
        raise RuntimeError("FAISS index is unreadable") from error
    actual_uids, vectors = _index_arrays(index, faiss)
    if len(set(actual_uids.tolist())) != len(actual_uids):
        raise RuntimeError("FAISS index contains duplicate keyframe_uid values")
    missing = np.setdiff1d(expected_uids, actual_uids, assume_unique=True)
    unexpected = np.setdiff1d(actual_uids, expected_uids, assume_unique=True)
    if len(missing) or len(unexpected):
        raise RuntimeError(
            "FAISS keyframe_uid diff failed: "
            f"missing={missing[:5].tolist()}, unexpected={unexpected[:5].tolist()}"
        )

    if (
        state.get("record_count") != len(actual_uids)
        or state.get("video_count") != len(video_ids)
        or state.get("vector_dim") != index.d
        or state.get("keyframe_uid_set_sha256") != _uid_set_sha256(actual_uids)
    ):
        raise RuntimeError("FAISS state accounting does not match the actual index")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(vectors).all() or not np.allclose(
        norms, np.ones_like(norms), atol=_NORM_ATOL, rtol=0
    ):
        raise RuntimeError("FAISS vectors failed finite/L2-norm validation")

    if any(
        not isinstance(row, dict)
        or not _is_hex_digest(row.get("signature", ""), length=64)
        or not str(row.get("batch_id", ""))
        or type(row.get("record_count")) is not int
        or int(row["record_count"]) <= 0
        or not _is_hex_digest(row.get("uid_sequence_sha256", ""), length=64)
        or not _is_hex_digest(row.get("artifact_content_sha256", ""), length=64)
        or not isinstance(row.get("video_ids"), list)
        or not row["video_ids"]
        or row.get("checkpoint_resume_verified") is not True
        for row in sources
    ):
        raise RuntimeError("FAISS source provenance is invalid")
    signatures = [str(row["signature"]) for row in sources]
    source_video_ids = [
        str(video_id) for row in sources for video_id in row["video_ids"]
    ]
    for source in sources:
        source_ids = [str(video_id) for video_id in source["video_ids"]]
        if source_ids != sorted(source_ids) or len(set(source_ids)) != len(source_ids):
            raise RuntimeError("FAISS source video IDs must be unique and sorted")
        source_records = [record for record in records if record.video_id in set(source_ids)]
        source_uids = np.asarray(
            [record.keyframe_uid for record in source_records], dtype=np.int64
        )
        if (
            len(source_records) != int(source["record_count"])
            or _uid_sequence_sha256(source_uids) != source["uid_sequence_sha256"]
        ):
            raise RuntimeError("FAISS source UID provenance does not match frames.csv")
    if (
        len(set(signatures)) != len(signatures)
        or len(set(source_video_ids)) != len(source_video_ids)
        or set(source_video_ids) != set(video_ids)
        or sum(int(row["record_count"]) for row in sources) != len(actual_uids)
        or state.get("checkpoint_resume_verified") is not True
    ):
        raise RuntimeError("FAISS source accounting is inconsistent")

    return FaissIndexReport(
        index_path=index_file,
        state_path=state_file,
        modality=modality,
        record_count=len(actual_uids),
        video_count=len(video_ids),
        vector_dim=index.d,
        source_count=len(sources),
    )


def _build_state(
    *,
    index_file: Path,
    modality: str,
    catalog_path: Path,
    model_id: str,
    model_revision: str,
    index: Any,
    sources: list[dict[str, object]],
    records: list[FrameRecord],
    faiss: Any,
) -> dict[str, object]:
    actual_uids, _ = _index_arrays(index, faiss)
    video_ids = sorted(
        {str(video_id) for source in sources for video_id in source["video_ids"]}
    )
    counts = Counter(
        record.video_id for record in records if record.video_id in set(video_ids)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "single_modality_index",
        "complete": True,
        "modality": modality,
        "index_factory": INDEX_FACTORY,
        "metric": "inner_product",
        "normalization": "l2_float32_before_add",
        "source_vector_dtype": "float16",
        "faiss_vector_dtype": "float32",
        "catalog_sha256": sha256_file(catalog_path),
        "index_sha256": sha256_file(index_file),
        "keyframe_uid_set_sha256": _uid_set_sha256(actual_uids),
        "record_count": int(index.ntotal),
        "video_count": len(video_ids),
        "vector_dim": int(index.d),
        "model": {"id": model_id, "revision": model_revision},
        "checkpoint_resume_verified": all(
            source["checkpoint_resume_verified"] is True for source in sources
        ),
        "sources": sources,
        "videos": [
            {
                "video_id": video_id,
                "record_count": counts[video_id],
                "complete": True,
            }
            for video_id in video_ids
        ],
    }


def build_faiss_index(
    *,
    modality: str,
    embedding_dirs: list[Path],
    catalog_path: Path,
    keyframes_root: Path,
    output_path: Path,
) -> FaissBuildResult:
    """Create or increment one modality index from verified, disjoint batches."""

    if modality not in SUPPORTED_MODALITIES:
        raise ValueError(f"unsupported modality: {modality}")
    if not embedding_dirs:
        raise ValueError("at least one completed embedding directory is required")
    catalog = Path(catalog_path)
    records = _load_catalog_records(catalog)
    loaded = [
        load_completed_visual_embedding_data(
            path,
            catalog_path=catalog,
            keyframes_root=keyframes_root,
            require_resume_verified=True,
        )
        for path in embedding_dirs
    ]
    if any(source.modality != modality for source in loaded):
        raise RuntimeError("embedding modality does not match target FAISS namespace")
    if any(source.catalog_sha256 != sha256_file(catalog) for source in loaded):
        raise RuntimeError("embedding source was built against a different frames.csv")

    faiss = _import_faiss()
    destination = Path(output_path)
    state_file = _state_path(destination)
    if destination.exists() != state_file.exists():
        raise RuntimeError("refusing to update an incomplete FAISS index/state pair")

    if destination.exists():
        validate_faiss_index(
            destination,
            catalog_path=catalog,
            state_path=state_file,
            expected_modality=modality,
        )
        state = _validate_state_structure(
            json.loads(state_file.read_text(encoding="utf-8"))
        )
        index = faiss.read_index(str(destination))
        model = state["model"]
        model_id = str(model["id"])
        model_revision = str(model["revision"])
        sources = [dict(row) for row in state["sources"]]
        existing_uids, _ = _index_arrays(index, faiss)
    else:
        first = loaded[0]
        model_id = first.model_id
        model_revision = first.model_revision
        vector_dim = int(first.vectors.shape[1])
        index = faiss.IndexIDMap(faiss.IndexFlatIP(vector_dim))
        sources = []
        existing_uids = np.asarray([], dtype=np.int64)

    source_by_signature = {str(row["signature"]): row for row in sources}
    existing_uid_set = set(existing_uids.tolist())
    added_records = 0
    added_sources = 0
    for source in loaded:
        entry = _source_entry(source)
        prior = source_by_signature.get(source.signature)
        if prior is not None:
            if prior != entry:
                raise RuntimeError("existing FAISS source signature has different provenance")
            continue
        if source.model_id != model_id or source.model_revision != model_revision:
            raise RuntimeError("refusing to mix model IDs/revisions in one FAISS index")
        if source.vectors.shape[1] != index.d:
            raise RuntimeError("embedding dimension does not match the FAISS index")
        source_uid_set = set(source.keyframe_uids.tolist())
        overlap = sorted(existing_uid_set & source_uid_set)
        if overlap:
            raise RuntimeError(
                "unregistered embedding source overlaps existing FAISS IDs: "
                f"{overlap[:5]}"
            )
        vectors = _prepare_vectors(source.vectors)
        index.add_with_ids(
            vectors,
            np.ascontiguousarray(source.keyframe_uids, dtype=np.int64),
        )
        sources.append(entry)
        source_by_signature[source.signature] = entry
        existing_uid_set.update(source_uid_set)
        added_records += len(source.keyframe_uids)
        added_sources += 1

    if added_sources == 0:
        report = validate_faiss_index(
            destination,
            catalog_path=catalog,
            state_path=state_file,
            expected_modality=modality,
        )
        return FaissBuildResult(report=report, added_records=0, added_sources=0)

    destination.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    temporary_index = destination.with_name(f".{destination.name}.tmp-{token}")
    temporary_state = state_file.with_name(f".{state_file.name}.tmp-{token}")
    try:
        faiss.write_index(index, str(temporary_index))
        state = _build_state(
            index_file=temporary_index,
            modality=modality,
            catalog_path=catalog,
            model_id=model_id,
            model_revision=model_revision,
            index=index,
            sources=sources,
            records=records,
            faiss=faiss,
        )
        temporary_state.write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        validate_faiss_index(
            temporary_index,
            catalog_path=catalog,
            state_path=temporary_state,
            expected_modality=modality,
        )
        temporary_index.replace(destination)
        temporary_state.replace(state_file)
    finally:
        temporary_index.unlink(missing_ok=True)
        temporary_state.unlink(missing_ok=True)

    report = validate_faiss_index(
        destination,
        catalog_path=catalog,
        state_path=state_file,
        expected_modality=modality,
    )
    return FaissBuildResult(
        report=report,
        added_records=added_records,
        added_sources=added_sources,
    )
