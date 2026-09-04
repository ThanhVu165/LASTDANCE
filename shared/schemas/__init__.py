"""Canonical Pydantic schemas defined by the baseline specifications."""

from .asr import AsrSegment
from .frame import FrameRecord, VerifiedFrameRef
from .ocr import OcrResult
from .online import (
    AnswerTarget,
    AnswerResult,
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
    "AnswerResult",
    "VerifiedFrameRef",
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
