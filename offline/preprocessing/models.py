"""Internal records for preprocessing artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class VideoInventoryRecord:
    video_id: str
    relative_path: str
    width: int
    height: int
    fps: float
    duration: float
    frame_count: int | None
    has_audio: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

@dataclass(frozen=True, slots=True)
class ShotBoundary:
    shot_id: str
    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        if not self.shot_id.strip():
            raise ValueError("shot_id must not be empty")
        if self.start_frame < 0:
            raise ValueError("start_frame must be non-negative")
        if self.end_frame < self.start_frame:
            raise ValueError("end_frame must be >= start_frame")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
