"""Canonical frame catalog record for ``frames.csv``."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


_MAX_INT64 = (1 << 63) - 1


class FrameRecord(BaseModel):
    """One independently retrievable keyframe.

    ``local_idx`` addresses the extracted JPEG. ``frame_id`` is the source MP4
    frame used for preview and submission. ``keyframe_uid`` is the stable join
    key shared by FAISS, OCR, and ASR artifacts.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    video_id: str
    local_idx: int = Field(ge=0)
    frame_id: int = Field(ge=0)
    pts_time: float = Field(ge=0)
    shot_id: str
    window_id: str | None = None
    keyframe_uid: int = Field(gt=0, le=_MAX_INT64)

    @field_validator("video_id", "shot_id")
    @classmethod
    def _require_non_empty_identifier(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("identifier must not be empty")
        return normalized
    @field_validator("window_id")
    @classmethod
    def _normalize_optional_identifier(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("window_id must be null or non-empty")
        return normalized


class VerifiedFrameRef(BaseModel):
    """Source-video frame without a synthetic retrieval UID."""
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    video_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    frame_id: int = Field(ge=0, strict=True)
    pts_time: float = Field(ge=0)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
