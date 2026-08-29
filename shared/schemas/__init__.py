"""Canonical Pydantic schemas defined by the baseline specifications."""

from .asr import AsrSegment
from .frame import FrameRecord
from .ocr import OcrResult
from .online import (
    AnswerTarget,
    ArtifactAvailability,
    ArtifactStatus,
    FrameEvidence,
    KISCandidate,
    QACandidate,
    QueryRole,
    QuerySpec,
    QueryUnit,
    SearchRequest,
    SearchRun,
    TaskCandidate,
    TaskType,
    TrakeCandidate,
    UnifiedQueryPlan,
    VideoHypothesis,
)

__all__ = [
    "AnswerTarget",
    "ArtifactAvailability",
    "ArtifactStatus",
    "AsrSegment",
    "FrameEvidence",
    "FrameRecord",
    "KISCandidate",
    "OcrResult",
    "QACandidate",
    "QueryRole",
    "QuerySpec",
    "QueryUnit",
    "SearchRequest",
    "SearchRun",
    "TaskCandidate",
    "TaskType",
    "TrakeCandidate",
    "UnifiedQueryPlan",
    "VideoHypothesis",
]
