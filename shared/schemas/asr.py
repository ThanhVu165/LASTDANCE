"""Canonical ASR segment handed from Branch 3 to online retrieval."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_MAX_INT64 = (1 << 63) - 1


class AsrSegment(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    video_id: str
    segment_id: str
    start_time: float = Field(ge=0)
    end_time: float = Field(ge=0)
    transcribed_text: str
    language: Literal["vi", "en"]
    keyframe_uid_nearest: int = Field(gt=0, le=_MAX_INT64)

    @field_validator("video_id", "segment_id", "transcribed_text")
    @classmethod
    def _require_non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized

    @model_validator(mode="after")
    def _end_must_not_precede_start(self) -> "AsrSegment":
        if self.end_time < self.start_time:
            raise ValueError("end_time must be greater than or equal to start_time")
        return self
