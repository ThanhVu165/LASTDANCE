"""Canonical Pydantic schemas defined by the baseline specifications."""

from .asr import AsrSegment
from .frame import FrameRecord
from .ocr import OcrResult

__all__ = ["AsrSegment", "FrameRecord", "OcrResult"]
