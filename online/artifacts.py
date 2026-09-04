"""Fail-closed artifact registry for frame-level online retrieval."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from offline.artifacts import sha256_file
from offline.ocr_snapshot import OcrSnapshotManifest
from offline.ocr_snapshot_hf import SNAPSHOT_FILENAMES, validate_local_snapshot_for_publish
from offline.ocr_v2_snapshot import OcrV2SnapshotManifest, validate_ocr_v2_snapshot
from shared.schemas.frame import FrameRecord
from shared.schemas.online import ArtifactAvailability, ArtifactStatus

from .config import OnlineLayout


EXPECTED_VISUAL = {
    "clip": (512, "openai/clip-vit-base-patch32", "4c4a3e8bcc2b768a8b89fc83ed8c828345ca3bac"),
    "siglip": (768, "google/siglip-base-patch16-224", "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"),
    "eva_clip": (
        768,
        "timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k",
        "bf4190eb65dd5204ffb03e980108beb1200e0873",
    ),
}


@dataclass(frozen=True, slots=True)
class OcrSnapshotSummary:
    snapshot_id: str
    intended_use: str
    source_format: str
    catalog_sha256: str
    catalog_records: int
    catalog_videos: int
    observed_uid_sha256: str
    coverage_fraction: float
    error_keyframes: int
    missing_keyframes: int
    fts_rows: int
    sqlite_sha256: str
    sqlite_bytes: int
    engines: str
    residual_frames: int | None = None
    residual_regions: int | None = None


def load_ocr_snapshot_summary(path: Path) -> OcrSnapshotSummary:
    """Parse a supported OCR sidecar into fields shared by registry and provenance."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("OCR coverage must be a JSON object")
    schema_version = raw.get("schema_version")
    source_format = raw.get("source_format")
    if type(schema_version) is not int:
        raise ValueError("OCR coverage schema_version must be an integer")
    if schema_version in {1, 2}:
        manifest = OcrSnapshotManifest.model_validate(raw)
        tiers = ",".join(sorted({batch.tier for batch in manifest.batches.values()}))
        return OcrSnapshotSummary(
            snapshot_id=manifest.snapshot_id,
            intended_use=manifest.intended_use,
            source_format=manifest.source_format,
            catalog_sha256=manifest.catalog_sha256,
            catalog_records=manifest.catalog_records,
            catalog_videos=manifest.catalog_videos,
            observed_uid_sha256=manifest.observed_uid_sha256,
            coverage_fraction=manifest.coverage_fraction,
            error_keyframes=manifest.error_keyframes,
            missing_keyframes=manifest.missing_keyframes,
            fts_rows=manifest.fts_rows,
            sqlite_sha256=manifest.sqlite_sha256,
            sqlite_bytes=manifest.sqlite_bytes,
            engines=tiers or "unknown",
        )
    if schema_version == 3 and source_format == "ocr_v2_batch_union_v1":
        manifest_v2 = OcrV2SnapshotManifest.model_validate(raw)
        totals = manifest_v2.totals
        coverage = (
            totals.processed_keyframes / totals.expected_keyframes
            if totals.expected_keyframes
            else 0.0
        )
        engines = ",".join(
            f"{engine}={count}"
            for engine, count in sorted(totals.selected_region_engine_counts.items())
        )
        return OcrSnapshotSummary(
            snapshot_id=manifest_v2.snapshot_id,
            intended_use=manifest_v2.intended_use,
            source_format=manifest_v2.source_format,
            catalog_sha256=manifest_v2.catalog_sha256,
            catalog_records=manifest_v2.catalog_records,
            catalog_videos=manifest_v2.catalog_videos,
            observed_uid_sha256=totals.observed_uid_sha256,
            coverage_fraction=coverage,
            error_keyframes=totals.error_keyframes,
            missing_keyframes=totals.missing_keyframes,
            fts_rows=manifest_v2.fts_rows,
            sqlite_sha256=manifest_v2.sqlite_sha256,
            sqlite_bytes=manifest_v2.sqlite_bytes,
            engines=engines or "unknown",
            residual_frames=totals.residual_frames,
            residual_regions=totals.residual_regions,
        )
    raise ValueError(
        f"unsupported OCR snapshot schema/source_format: {schema_version}/{source_format}"
    )


@dataclass(frozen=True, slots=True)
class CatalogFrame:
    keyframe_uid: int
    video_id: str
    local_idx: int
    frame_id: int
    pts_time: float
    shot_id: str

    def image_path(self, keyframes_root: Path) -> Path:
        return keyframes_root / self.video_id / f"{self.shot_id}_{self.local_idx}.jpg"


class FrameCatalog:
    def __init__(self, frames: list[CatalogFrame], *, sha256: str) -> None:
        self.frames = frames
        self.sha256 = sha256
        ordered_uids = sorted(frame.keyframe_uid for frame in frames)
        payload = ("\n".join(str(uid) for uid in ordered_uids) + "\n").encode("utf-8")
        self.uid_set_sha256 = hashlib.sha256(payload).hexdigest()
        self.by_uid = {frame.keyframe_uid: frame for frame in frames}
        self.by_video: dict[str, list[CatalogFrame]] = {}
        for frame in frames:
            self.by_video.setdefault(frame.video_id, []).append(frame)
        for values in self.by_video.values():
            values.sort(key=lambda item: (item.pts_time, item.frame_id))
        self.position_in_video = {
            frame.keyframe_uid: index
            for values in self.by_video.values()
            for index, frame in enumerate(values)
        }

    def neighbors(self, uid: int, radius: int) -> list[CatalogFrame]:
        frame = self.by_uid[uid]
        values = self.by_video[frame.video_id]
        center = self.position_in_video[uid]
        start = max(0, center - radius)
        stop = min(len(values), center + radius + 1)
        candidates = [item for item in values[start:stop] if item.keyframe_uid != uid]
        same_shot = [item for item in candidates if item.shot_id == frame.shot_id]
        other = [item for item in candidates if item.shot_id != frame.shot_id]
        return (same_shot + other)[: radius * 2]


class FaissIndexHandle:
    def __init__(self, modality: str, path: Path, state: dict[str, Any], index: Any, faiss: Any) -> None:
        self.modality = modality
        self.path = path
        self.state = state
        self.index = index
        self.faiss = faiss
        self.ids = faiss.vector_to_array(index.id_map).astype(np.int64, copy=False)
        self.uid_to_position = {int(uid): position for position, uid in enumerate(self.ids)}
        self.base = faiss.downcast_index(index.index)

    @property
    def dimension(self) -> int:
        return int(self.index.d)

    def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        vector = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
        scores, uids = self.index.search(vector, min(top_k, int(self.index.ntotal)))
        valid = uids[0] >= 0
        return uids[0][valid].astype(np.int64, copy=False), scores[0][valid].astype(np.float32, copy=False)

    def scores_for(self, query: np.ndarray, uids: Iterable[int]) -> dict[int, float]:
        ordered = [int(uid) for uid in uids]
        positions = np.asarray([self.uid_to_position[uid] for uid in ordered], dtype=np.int64)
        try:
            vectors = np.asarray(self.base.reconstruct_batch(positions), dtype=np.float32)
        except (AttributeError, RuntimeError):
            vectors = np.vstack([self.base.reconstruct(int(position)) for position in positions]).astype(np.float32)
        scores = vectors @ np.asarray(query, dtype=np.float32)
        return {uid: float(score) for uid, score in zip(ordered, scores)}


class ArtifactRegistry:
    def __init__(
        self,
        *,
        layout: OnlineLayout,
        catalog: FrameCatalog,
        visual: dict[str, FaissIndexHandle],
        statuses: dict[str, ArtifactStatus],
    ) -> None:
        self.layout = layout
        self.catalog = catalog
        self.visual = visual
        self.statuses = statuses

    @classmethod
    def load(cls, layout: OnlineLayout | None = None, *, deep: bool = False) -> "ArtifactRegistry":
        layout = layout or OnlineLayout.from_environment()
        catalog = _load_catalog(layout)
        visual: dict[str, FaissIndexHandle] = {}
        statuses: dict[str, ArtifactStatus] = {
            "frames": ArtifactStatus(
                name="frames",
                availability=ArtifactAvailability.READY,
                path="index/frames.csv",
                record_count=len(catalog.frames),
                detail=(
                    f"videos={len(catalog.by_video)} sha256={catalog.sha256} "
                    f"uid_set_sha256={catalog.uid_set_sha256}"
                ),
            )
        }
        for modality in EXPECTED_VISUAL:
            handle = _load_visual(layout, catalog, modality, deep=deep)
            visual[modality] = handle
            statuses[modality] = ArtifactStatus(
                name=modality,
                availability=ArtifactAvailability.READY,
                path=f"index/{modality}.faiss",
                record_count=int(handle.index.ntotal),
                detail=f"dim={handle.dimension} revision={handle.state['model']['revision']}",
            )
        statuses["ocr"] = _inspect_ocr(layout, catalog)
        statuses["asr"] = _inspect_asr(layout.asr, catalog)
        return cls(layout=layout, catalog=catalog, visual=visual, statuses=statuses)


def _load_catalog(layout: OnlineLayout) -> FrameCatalog:
    if not layout.catalog.is_file() or not layout.catalog_state.is_file():
        raise RuntimeError("frames.csv/state pair is missing")
    state = json.loads(layout.catalog_state.read_text(encoding="utf-8"))
    if state.get("complete") is not True or state.get("schema_version") != 1:
        raise RuntimeError("frames.csv state is incomplete or unsupported")
    actual_sha = sha256_file(layout.catalog)
    if state.get("csv_sha256") != actual_sha:
        raise RuntimeError("frames.csv SHA-256 does not match state")
    required = ["video_id", "local_idx", "frame_id", "pts_time", "shot_id", "window_id", "keyframe_uid"]
    frames: list[CatalogFrame] = []
    with layout.catalog.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != required:
            raise RuntimeError("frames.csv columns do not match the baseline")
        for row in reader:
            FrameRecord.model_validate({**row, "window_id": row["window_id"] or None})
            frames.append(
                CatalogFrame(
                    keyframe_uid=int(row["keyframe_uid"]),
                    video_id=row["video_id"],
                    local_idx=int(row["local_idx"]),
                    frame_id=int(row["frame_id"]),
                    pts_time=float(row["pts_time"]),
                    shot_id=row["shot_id"],
                )
            )
    if len(frames) != state.get("record_count") or len({item.keyframe_uid for item in frames}) != len(frames):
        raise RuntimeError("frames.csv count or keyframe_uid uniqueness is invalid")
    if len({item.video_id for item in frames}) != state.get("video_count"):
        raise RuntimeError("frames.csv video count is invalid")
    return FrameCatalog(frames, sha256=actual_sha)


def _load_visual(layout: OnlineLayout, catalog: FrameCatalog, modality: str, *, deep: bool) -> FaissIndexHandle:
    try:
        import faiss
    except ImportError as error:
        raise RuntimeError("faiss-cpu is required for Online") from error
    path = layout.faiss_index(modality)
    state_path = layout.faiss_state(modality)
    if not path.is_file() or not state_path.is_file():
        raise RuntimeError(f"{modality} index/state pair is missing")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    dimension, model_id, revision = EXPECTED_VISUAL[modality]
    expected = {
        "schema_version": 1,
        "scope": "single_modality_index",
        "complete": True,
        "modality": modality,
        "index_factory": "IndexIDMap(IndexFlatIP)",
        "metric": "inner_product",
        "normalization": "l2_float32_before_add",
        "source_vector_dtype": "float16",
        "faiss_vector_dtype": "float32",
        "catalog_sha256": catalog.sha256,
        "keyframe_uid_set_sha256": catalog.uid_set_sha256,
        "record_count": len(catalog.frames),
        "video_count": len(catalog.by_video),
        "vector_dim": dimension,
    }
    if any(state.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"{modality} state does not match the production catalog contract")
    if state.get("model") != {"id": model_id, "revision": revision}:
        raise RuntimeError(f"{modality} model provenance mismatch")
    if state.get("checkpoint_resume_verified") is not True:
        raise RuntimeError(f"{modality} checkpoint/resume is not verified")
    if deep and state.get("index_sha256") != sha256_file(path):
        raise RuntimeError(f"{modality} FAISS SHA-256 mismatch")
    index = faiss.read_index(str(path))
    if type(index).__name__ != "IndexIDMap" or int(index.d) != dimension or int(index.ntotal) != len(catalog.frames):
        raise RuntimeError(f"{modality} FAISS structure/count/dimension mismatch")
    base = faiss.downcast_index(index.index)
    if type(base).__name__ != "IndexFlatIP" or index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise RuntimeError(f"{modality} must be IndexIDMap(IndexFlatIP)")
    handle = FaissIndexHandle(modality, path, state, index, faiss)
    catalog_uids = set(catalog.by_uid)
    actual_uids = set(int(uid) for uid in handle.ids)
    if deep:
        if actual_uids != catalog_uids:
            raise RuntimeError(f"{modality} keyframe_uid diff failed")
    elif len(handle.ids) == 0 or any(int(uid) not in catalog_uids for uid in handle.ids[[0, len(handle.ids) // 2, -1]]):
        raise RuntimeError(f"{modality} sampled keyframe_uid lookup failed")
    return handle


def _inspect_fts(path: Path, name: str, catalog: FrameCatalog) -> ArtifactStatus:
    relative = f"index/{path.name}"
    if not path.is_file():
        return ArtifactStatus(name=name, availability=ArtifactAvailability.UNAVAILABLE, path=relative, detail="artifact is absent")
    contracts = {
        "ocr": {
            "table": "ocr_fts",
            "columns": {"video_id", "keyframe_uid", "detected_text", "language", "confidence"},
            "uid": "keyframe_uid",
        },
        "asr": {
            "table": "asr_fts",
            "columns": {
                "video_id",
                "segment_id",
                "transcribed_text",
                "language",
                "keyframe_uid_nearest",
                "start_time",
                "end_time",
            },
            "uid": "keyframe_uid_nearest",
        },
    }
    contract = contracts[name]
    table = str(contract["table"])
    uid_column = str(contract["uid"])
    try:
        connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        try:
            definition_row = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            definition = str(definition_row[0] if definition_row else "").casefold()
            if "create virtual table" not in definition or "using fts5" not in definition:
                raise RuntimeError(f"{table} must be an FTS5 virtual table")
            columns = {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}
            missing = set(contract["columns"]) - columns
            if missing:
                raise RuntimeError(f"missing columns: {sorted(missing)}")
            integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
            if integrity.lower() != "ok":
                raise RuntimeError(f"SQLite integrity_check: {integrity}")
            count = int(connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
            cursor = connection.execute(f"SELECT video_id, {uid_column} FROM {table}")
            for video_id, raw_uid in cursor:
                uid = int(raw_uid)
                frame = catalog.by_uid.get(uid)
                if frame is None or frame.video_id != str(video_id):
                    raise RuntimeError(f"orphan or cross-video keyframe_uid: {uid}")
        finally:
            connection.close()
    except (sqlite3.Error, RuntimeError, TypeError, ValueError) as error:
        return ArtifactStatus(name=name, availability=ArtifactAvailability.INVALID, path=relative, detail=str(error))
    return ArtifactStatus(
        name=name,
        availability=ArtifactAvailability.READY,
        path=relative,
        detail=f"table={table}; integrity=ok; UID joins=valid",
        record_count=count,
    )


def _inspect_ocr(layout: OnlineLayout, catalog: FrameCatalog) -> ArtifactStatus:
    """Inspect OCR and fail closed on an explicitly selected development snapshot."""

    status = _inspect_fts(layout.ocr, "ocr", catalog)
    snapshot_dir = layout.ocr_snapshot_dir
    if snapshot_dir is None:
        return status
    path = str(layout.ocr)
    try:
        coverage_path = layout.ocr_coverage
        if coverage_path is None:
            raise RuntimeError("explicit OCR snapshot lacks coverage.json")
        if not snapshot_dir.is_dir():
            raise FileNotFoundError(f"snapshot directory does not exist: {snapshot_dir}")
        actual_names = {item.name for item in snapshot_dir.iterdir()}
        if actual_names != set(SNAPSHOT_FILENAMES):
            raise RuntimeError(
                "snapshot directory must contain exactly: "
                + ", ".join(SNAPSHOT_FILENAMES)
            )
        summary = load_ocr_snapshot_summary(coverage_path)
        if summary.source_format == "ocr_v2_batch_union_v1":
            manifest_v2 = validate_ocr_v2_snapshot(
                snapshot_dir=snapshot_dir,
                catalog_path=layout.catalog,
                catalog_state_path=layout.catalog_state,
            )
            if manifest_v2.totals.expected_keyframes != len(catalog.frames):
                raise RuntimeError("OCR v2 expected keyframes do not match frames.csv")
        else:
            validate_local_snapshot_for_publish(snapshot_dir)
        expected = {
            "catalog_sha256": catalog.sha256,
            "catalog_records": len(catalog.frames),
            "catalog_videos": len(catalog.by_video),
            "observed_uid_sha256": catalog.uid_set_sha256,
        }
        for field, value in expected.items():
            if getattr(summary, field) != value:
                raise RuntimeError(f"OCR snapshot {field} does not match frames.csv")
        if layout.ocr.stat().st_size != summary.sqlite_bytes:
            raise RuntimeError("OCR snapshot SQLite size does not match coverage.json")
        if status.availability != ArtifactAvailability.READY:
            raise RuntimeError(status.detail)
        if status.record_count != summary.fts_rows:
            raise RuntimeError("OCR snapshot FTS row count does not match coverage.json")
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return ArtifactStatus(
            name="ocr",
            availability=ArtifactAvailability.INVALID,
            path=path,
            detail=f"explicit snapshot invalid: {error}",
        )
    residual = ""
    if summary.residual_frames is not None and summary.residual_regions is not None:
        residual = (
            f"residual_frames={summary.residual_frames}; "
            f"residual_regions={summary.residual_regions}; "
        )
    return ArtifactStatus(
        name="ocr",
        availability=ArtifactAvailability.READY,
        path=path,
        record_count=summary.fts_rows,
        detail=(
            f"snapshot_id={summary.snapshot_id}; intended_use={summary.intended_use}; "
            f"source_format={summary.source_format}; coverage={summary.coverage_fraction:.2%}; "
            f"engines={summary.engines}; errors={summary.error_keyframes}; "
            f"missing={summary.missing_keyframes}; {residual}"
            "integrity=ok; UID joins=valid; production_ready=false"
        ),
    )


def _inspect_asr(path: Path, catalog: FrameCatalog) -> ArtifactStatus:
    from offline.asr_validation import validate_asr_bundle
    if not path.is_file():
        return _inspect_fts(path, "asr", catalog)
    try:
        manifest = validate_asr_bundle(path, path.with_name("asr.coverage.json"),
                                      catalog_sha256=catalog.sha256, frames=catalog.by_uid)
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        return ArtifactStatus(name="asr", availability=ArtifactAvailability.INVALID,
                              path=str(path), detail=f"ASR coverage invalid: {exc}")
    return ArtifactStatus(name="asr", availability=ArtifactAvailability.READY, path=str(path),
                          record_count=manifest.fts_rows,
                          detail=f"coverage={manifest.coverage_fraction:.2%}; errors={manifest.error_videos}; "
                          f"missing={manifest.missing_videos}; unverified_silence={manifest.unverified_silent_videos}; "
                          "production_ready=false; per-video coverage and checksum validated")
