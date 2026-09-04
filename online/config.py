"""Configuration and paths for the local online runtime."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Mapping

from offline.config import DataLayout


@dataclass(frozen=True, slots=True)
class OnlineConfig:
    visual_top_k: int = 1000
    srrf_eta: float = 60.0
    srrf_beta: float = 40.0
    query_consensus_bonus: float = 0.1
    neighbor_radius: int = 2
    neighbor_bonus: float = 0.15
    kis_anchor_bonus: float = 0.15
    kis_max_frames_per_shot: int = 3
    video_top_k: int = 12
    trake_video_top_k: int = 20
    video_part_top_k: int = 50
    video_frame_evidence_k: int = 3
    video_evidence_max_weight: float = 0.7
    video_locator_weight: float = 0.35
    video_target_weight: float = 0.45
    video_global_weight: float = 0.10
    video_consensus_weight: float = 0.10
    trake_locator_weight: float = 0.15
    trake_event_coverage_weight: float = 0.45
    trake_weakest_weight: float = 0.20
    trake_mean_weight: float = 0.10
    trake_consensus_weight: float = 0.10
    verified_base_weight: float = 0.85
    verified_must_weight: float = 0.10
    verified_should_weight: float = 0.05
    frame_base_weight: float = 0.85
    frame_vlm_weight: float = 0.15
    vlm_video_top_k: int = 1
    vlm_frame_top_k: int = 8
    vlm_sheet_size: int = 8
    gemini_max_calls_per_search: int = 14
    gemini_search_timeout_seconds: float = 3600.0
    trake_frame_top_k: int = 32
    trake_beam_width: int = 8
    trake_decay: float = 0.0
    qa_similarity_threshold: float = 0.85
    qa_vqa_agreement_similarity: float = 0.6
    qa_answer_video_top_k: int = 3
    portfolio_max_per_video: int = 40
    portfolio_primary_min: int = 30
    clip_tie_margin: float = 0.02

    def __post_init__(self) -> None:
        positive = {
            "visual_top_k": self.visual_top_k,
            "neighbor_radius": self.neighbor_radius,
            "kis_max_frames_per_shot": self.kis_max_frames_per_shot,
            "video_top_k": self.video_top_k,
            "trake_video_top_k": self.trake_video_top_k,
            "video_part_top_k": self.video_part_top_k,
            "video_frame_evidence_k": self.video_frame_evidence_k,
            "trake_frame_top_k": self.trake_frame_top_k,
            "trake_beam_width": self.trake_beam_width,
            "vlm_video_top_k": self.vlm_video_top_k,
            "vlm_frame_top_k": self.vlm_frame_top_k,
            "vlm_sheet_size": self.vlm_sheet_size,
            "gemini_max_calls_per_search": self.gemini_max_calls_per_search,
            "qa_answer_video_top_k": self.qa_answer_video_top_k,
            "portfolio_max_per_video": self.portfolio_max_per_video,
            "portfolio_primary_min": self.portfolio_primary_min,
        }
        if any(value <= 0 for value in positive.values()):
            raise ValueError(f"online integer config values must be positive: {positive}")
        weights = (
            self.video_locator_weight
            + self.video_target_weight
            + self.video_global_weight
            + self.video_consensus_weight
        )
        if abs(weights - 1.0) > 1e-9:
            raise ValueError("video ranking weights must sum to 1")
        trake_weights = (
            self.trake_locator_weight
            + self.trake_event_coverage_weight
            + self.trake_weakest_weight
            + self.trake_mean_weight
            + self.trake_consensus_weight
        )
        if abs(trake_weights - 1.0) > 1e-9:
            raise ValueError("TRAKE video ranking weights must sum to 1")
        verified_weights = (
            self.verified_base_weight + self.verified_must_weight + self.verified_should_weight
        )
        if abs(verified_weights - 1.0) > 1e-9:
            raise ValueError("verified video weights must sum to 1")
        if abs(self.frame_base_weight + self.frame_vlm_weight - 1.0) > 1e-9:
            raise ValueError("frame reranking weights must sum to 1")
        if not 0.0 <= self.qa_similarity_threshold <= 1.0:
            raise ValueError("qa_similarity_threshold must be between 0 and 1")
        if not 0.0 <= self.qa_vqa_agreement_similarity <= 1.0:
            raise ValueError("qa_vqa_agreement_similarity must be between 0 and 1")
        if self.portfolio_max_per_video > 100:
            raise ValueError("portfolio_max_per_video must not exceed submission limit")
        if self.portfolio_primary_min > self.portfolio_max_per_video:
            raise ValueError("portfolio_primary_min must not exceed portfolio_max_per_video")
        if self.trake_decay < 0:
            raise ValueError("trake_decay must be non-negative")
        if self.gemini_search_timeout_seconds <= 0:
            raise ValueError("gemini_search_timeout_seconds must be positive")
        if not 0.0 <= self.kis_anchor_bonus <= 1.0:
            raise ValueError("kis_anchor_bonus must be between 0 and 1")

    @classmethod
    def load(cls, path: Path | None = None) -> "OnlineConfig":
        source = path or Path(__file__).resolve().parents[1] / "configs" / "online_baseline.json"
        raw = json.loads(source.read_text(encoding="utf-8"))
        if raw.get("schema_version") != 3:
            raise RuntimeError("online config schema_version must be 3")
        allowed = {item.name for item in fields(cls)}
        values = {key: value for key, value in raw.items() if key in allowed}
        return cls(**values)


@dataclass(frozen=True, slots=True)
class OnlineLayout:
    data: DataLayout
    ocr_snapshot_dir: Path | None = None

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> "OnlineLayout":
        values = os.environ if environment is None else environment
        raw_snapshot = str(values.get("AIC_OCR_SNAPSHOT_DIR", "")).strip()
        snapshot_dir = Path(raw_snapshot).expanduser().resolve() if raw_snapshot else None
        return cls(data=DataLayout.from_environment(values), ocr_snapshot_dir=snapshot_dir)

    @property
    def catalog(self) -> Path:
        return self.data.index / "frames.csv"

    @property
    def catalog_state(self) -> Path:
        return self.data.index / "frames.csv.state.json"

    def faiss_index(self, modality: str) -> Path:
        return self.data.index / f"{modality}.faiss"

    def faiss_state(self, modality: str) -> Path:
        path = self.faiss_index(modality)
        return path.with_name(f"{path.name}.state.json")

    @property
    def ocr(self) -> Path:
        if self.ocr_snapshot_dir is not None:
            return self.ocr_snapshot_dir / "ocr.sqlite"
        return self.data.index / "ocr.sqlite"

    @property
    def ocr_coverage(self) -> Path | None:
        if self.ocr_snapshot_dir is None:
            return None
        return self.ocr_snapshot_dir / "coverage.json"

    @property
    def asr(self) -> Path:
        return self.data.index / "asr.sqlite"

    @property
    def submissions(self) -> Path:
        return self.data.root / "submissions"
