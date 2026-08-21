"""Pydantic schemas shared by the KIS / QA / TRAKE routers.

Note on identifiers: every keyframe has two different indices —
  - `local_idx`  : the 1-based sequence number matching the keyframe/object JSON
                   filenames (e.g. 007.jpg / 007.json). Used only internally to
                   look up images, objects, and OCR text.
  - `frame_id`   : the REAL frame number in the source video, taken from the
                   `frame_idx` column of data/map-keyframes/<video_id>.csv. This is
                   the value that MUST be submitted in the contest CSV output.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class KisQuery(BaseModel):
    text: str = Field(min_length=1)


class KisResult(BaseModel):
    video_id: str
    frame_id: int
    local_idx: int
    score: float
    is_source_frame: bool = False


class KisResponse(BaseModel):
    results: List[KisResult]


class QaQuery(BaseModel):
    # Complete organizer query: event description and question in one string.
    text: str = Field(min_length=1)


class QaResult(BaseModel):
    video_id: str
    frame_id: int
    local_idx: int
    answer: str
    score: float


class QaResponse(BaseModel):
    results: List[QaResult]


class TrakeQuery(BaseModel):
    # Complete organizer query containing the whole ordered event sequence.
    text: str = Field(min_length=1)


class TrakeResult(BaseModel):
    video_id: str
    frame_ids: List[int]
    local_idxs: List[int]
    score: float
    is_source_frames: bool = False


class TrakeResponse(BaseModel):
    moments: List[str]
    results: List[TrakeResult]


class SubmissionRow(BaseModel):
    query_id: str
    query_type: str  # "kis" | "qa" | "trake"
    rank: int
    video_id: str
    frame_ids: List[int]
    answer: Optional[str] = None
