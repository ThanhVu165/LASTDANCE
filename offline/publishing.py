"""Fail-closed evaluation of per-video publishing criteria."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping
from pathlib import Path
import json
from offline.artifacts import sha256_file
from offline.checkpoints import CheckpointStore
from offline.preprocessing.shot_detection import load_shot_manifest


@dataclass(frozen=True, slots=True)
class PublishingProofs:
    """Evidence paths, re-read on evaluation; absent or stale evidence fails closed."""
    data_root: Path
    catalog_path: Path
    shot_manifest: Path
    shot_checkpoint: Path
    mapping_report: Path
    resume_reports: Mapping[str, Path]

    def validate(self, video_id: str, frame_uids: frozenset[int]) -> bool:
        try:
            from scripts.run_shot_batch import build_video_checkpoint_signature
            manifest_video, relative_path, detection = load_shot_manifest(self.shot_manifest)
            if manifest_video != video_id:
                return False
            payload = json.loads(self.shot_manifest.read_text(encoding="utf-8"))
            detector = payload.get("detector_signature")
            if not isinstance(detector, dict) or not detector:
                return False
            source = self.data_root / relative_path
            signature = build_video_checkpoint_signature(source=source, relative_video_path=relative_path,
                expected_signature=detector, shots_directory=self.shot_manifest.parent, data_root=self.data_root)
            progress = CheckpointStore(self.shot_checkpoint).get(video_id, "shot_detection")
            if progress is None or progress.signature != signature or progress.total != 1 or not progress.finished:
                return False
            import csv
            with self.catalog_path.open(encoding="utf-8", newline="") as stream:
                frames = [row for row in csv.DictReader(stream) if row["video_id"] == video_id]
            if {int(row["keyframe_uid"]) for row in frames} != frame_uids:
                return False
            shots = {shot.shot_id: shot for shot in detection.shots}
            for row in frames:
                shot = shots.get(row["shot_id"])
                if shot is None or not shot.start_frame <= int(row["frame_id"]) <= shot.end_frame:
                    return False
            catalog_hash = sha256_file(self.catalog_path)
            mapping = json.loads(self.mapping_report.read_text(encoding="utf-8"))
            if (mapping.get("catalog_sha256") != catalog_hash or not str(mapping.get("reviewed_by", "")).strip()
                    or mapping.get("video_id") != video_id or not mapping.get("samples")
                    or mapping.get("source_sha256") != sha256_file(source)):
                return False
            frame_lookup = {int(row["frame_id"]): row for row in frames}
            for sample in mapping["samples"]:
                original = frame_lookup.get(sample["frame_id"])
                if original is None or sample.get("source_frame_id") != sample["frame_id"]:
                    return False
                if abs(float(original["pts_time"]) - float(sample["source_pts_time"])) > 1e-6:
                    return False
            if set(self.resume_reports) != set(REQUIRED_VISUAL_INDEXES):
                return False
            for modality, report_path in self.resume_reports.items():
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if (report.get("catalog_sha256") != catalog_hash or report.get("modality") != modality
                        or report.get("video_id") != video_id or report.get("interruption_exit_code") in (None, 0)
                        or report.get("resume_exit_code") != 0):
                    return False
                artifacts = report.get("artifacts", {})
                if not {"interrupted_checkpoint", "resumed_checkpoint", "run_log"} <= set(artifacts):
                    return False
                loaded = {}
                for name, artifact in artifacts.items():
                    path = (report_path.parent / artifact["path"]).resolve()
                    if report_path.parent.resolve() not in path.parents or sha256_file(path) != artifact["sha256"]:
                        return False
                    if name.endswith("checkpoint"):
                        loaded[name] = json.loads(path.read_text(encoding="utf-8"))
                before, after = loaded["interrupted_checkpoint"], loaded["resumed_checkpoint"]
                if (before.get("modality") != modality or after.get("modality") != modality
                        or not before.get("signature") or before["signature"] != after.get("signature")
                        or not 0 < before["next_index"] < before["total"] or before["total"] != after["total"]
                        or after["next_index"] != after["total"] or after.get("complete") is not True
                        or before.get("intentional_interruption_observed") is not True
                        or not before.get("interruption_process_token") or after.get("resume_count", 0) < 1
                        or after.get("checkpoint_resume_verified") is not True):
                    return False
                from offline.visual_embeddings import validate_completed_visual_embedding
                embedding_dir = Path(report["embedding_artifact_dir"])
                if not embedding_dir.is_absolute():
                    embedding_dir = report_path.parent / embedding_dir
                validate_completed_visual_embedding(embedding_dir, catalog_path=self.catalog_path,
                    keyframes_root=self.data_root / "keyframes", require_resume_verified=True)
                current = json.loads((embedding_dir / "checkpoint.json").read_text(encoding="utf-8"))
                manifest = json.loads((embedding_dir / "manifest.json").read_text(encoding="utf-8"))
                if current != after or video_id not in {row["video_id"] for row in manifest["videos"]}:
                    return False
            return True
        except (OSError, ValueError, TypeError, KeyError, RuntimeError):
            return False


REQUIRED_VISUAL_INDEXES = ("clip", "siglip", "eva_clip")


@dataclass(frozen=True, slots=True)
class VectorHealth:
    finite: bool
    normalized: bool


@dataclass(frozen=True, slots=True)
class PublishingReport:
    video_id: str
    has_frames: bool
    missing_ids: Mapping[str, frozenset[int]]
    unexpected_ids: Mapping[str, frozenset[int]]
    vector_health: Mapping[str, VectorHealth]
    mapping_verified: bool
    checkpoint_resume_verified: bool
    evidence_verified: bool = False

    @property
    def complete(self) -> bool:
        ids_match = all(
            not self.missing_ids[name] and not self.unexpected_ids[name]
            for name in REQUIRED_VISUAL_INDEXES
        )
        vectors_valid = all(
            self.vector_health.get(name) == VectorHealth(True, True)
            for name in REQUIRED_VISUAL_INDEXES
        )
        return (
            self.has_frames
            and ids_match
            and vectors_valid
            and self.mapping_verified
            and self.checkpoint_resume_verified
            and self.evidence_verified
        )

    def as_state(self) -> dict[str, object]:
        return {
            "video_id": self.video_id,
            "complete": self.complete,
            "criteria": {
                "has_frames": self.has_frames,
                "ids_match_all_indexes": all(
                    not self.missing_ids[name] and not self.unexpected_ids[name]
                    for name in REQUIRED_VISUAL_INDEXES
                ),
                "vectors_finite_and_normalized": all(
                    self.vector_health.get(name) == VectorHealth(True, True)
                    for name in REQUIRED_VISUAL_INDEXES
                ),
                "artifact_evidence_verified": self.evidence_verified,
                "mapping_verified": self.mapping_verified,
                "checkpoint_resume_verified": self.checkpoint_resume_verified,
            },
        }


def assess_publishing_readiness(
    *,
    video_id: str,
    frame_uids: Iterable[int],
    index_uids: Mapping[str, Iterable[int]],
    vector_health: Mapping[str, VectorHealth],
    mapping_verified: bool,
    checkpoint_resume_verified: bool,
    proofs: PublishingProofs | None = None,
) -> PublishingReport:
    expected = frozenset(frame_uids)
    actual = {
        name: frozenset(index_uids.get(name, ()))
        for name in REQUIRED_VISUAL_INDEXES
    }
    return PublishingReport(
        video_id=video_id,
        has_frames=bool(expected),
        missing_ids={name: expected - actual[name] for name in REQUIRED_VISUAL_INDEXES},
        unexpected_ids={name: actual[name] - expected for name in REQUIRED_VISUAL_INDEXES},
        vector_health=dict(vector_health),
        mapping_verified=mapping_verified,
        checkpoint_resume_verified=checkpoint_resume_verified,
        evidence_verified=proofs is not None and proofs.validate(video_id, expected),
    )
