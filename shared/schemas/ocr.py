"""Canonical OCR result emitted for one keyframe."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class OcrResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_id: int = Field(ge=0)
    detected_text: list[str]
    bbox: list[list[float]]
    confidence: float = Field(ge=0.0, le=1.0)
    language: Literal["vi", "en", "mixed"]

    @model_validator(mode="after")
    def _text_and_geometry_must_align(self) -> "OcrResult":
        if len(self.detected_text) != len(self.bbox):
            raise ValueError("detected_text and bbox must have the same length")
        return self
