"""Canonical Pydantic schemas defined by the baseline specifications."""

from .asr import AsrSegment
from .frame import FrameRecord
from .ocr import OcrResult
from .online import (
    ArtifactAvailability,
    ArtifactStatus,
    FrameEvidence,
    KISCandidate,
    QACandidate,
    QuerySpec,
    SearchRequest,
    SearchRun,
    TaskCandidate,
    TaskType,
    TrakeCandidate,
    UnifiedQueryPlan,
    VideoHypothesis,
)

__all__ = [
    "ArtifactAvailability",
    "ArtifactStatus",
    "AsrSegment",
    "FrameEvidence",
    "FrameRecord",
    "KISCandidate",
    "OcrResult",
    "QACandidate",
    "QuerySpec",
    "SearchRequest",
    "SearchRun",
    "TaskCandidate",
    "TaskType",
    "TrakeCandidate",
    "UnifiedQueryPlan",
    "VideoHypothesis",
]
