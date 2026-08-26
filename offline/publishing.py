"""Fail-closed evaluation of per-video publishing criteria."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


REQUIRED_VISUAL_INDEXES = ("clip", "siglip", "eva_clip")


@dataclass(frozen=True, slots=True)
class VectorHealth:
    finite: bool
    normalized: bool


@dataclass(frozen=True, slots=True)
class PublishingReport:
    video_id: str
    has_frames: bool
    missing_ids: Mapping[str, frozenset[int]]
    unexpected_ids: Mapping[str, frozenset[int]]
    vector_health: Mapping[str, VectorHealth]
    mapping_verified: bool
    checkpoint_resume_verified: bool

    @property
    def complete(self) -> bool:
        ids_match = all(
            not self.missing_ids[name] and not self.unexpected_ids[name]
            for name in REQUIRED_VISUAL_INDEXES
        )
        vectors_valid = all(
            self.vector_health.get(name) == VectorHealth(True, True)
            for name in REQUIRED_VISUAL_INDEXES
        )
        return (
            self.has_frames
            and ids_match
            and vectors_valid
            and self.mapping_verified
            and self.checkpoint_resume_verified
        )

    def as_state(self) -> dict[str, object]:
        return {
            "video_id": self.video_id,
            "complete": self.complete,
            "criteria": {
                "has_frames": self.has_frames,
                "ids_match_all_indexes": all(
                    not self.missing_ids[name] and not self.unexpected_ids[name]
                    for name in REQUIRED_VISUAL_INDEXES
                ),
                "vectors_finite_and_normalized": all(
                    self.vector_health.get(name) == VectorHealth(True, True)
                    for name in REQUIRED_VISUAL_INDEXES
                ),
                "mapping_verified": self.mapping_verified,
                "checkpoint_resume_verified": self.checkpoint_resume_verified,
            },
        }


def assess_publishing_readiness(
    *,
    video_id: str,
    frame_uids: Iterable[int],
    index_uids: Mapping[str, Iterable[int]],
    vector_health: Mapping[str, VectorHealth],
    mapping_verified: bool,
    checkpoint_resume_verified: bool,
) -> PublishingReport:
    expected = frozenset(frame_uids)
    actual = {
        name: frozenset(index_uids.get(name, ()))
        for name in REQUIRED_VISUAL_INDEXES
    }
    return PublishingReport(
        video_id=video_id,
        has_frames=bool(expected),
        missing_ids={name: expected - actual[name] for name in REQUIRED_VISUAL_INDEXES},
        unexpected_ids={name: actual[name] - expected for name in REQUIRED_VISUAL_INDEXES},
        vector_health=dict(vector_health),
        mapping_verified=mapping_verified,
        checkpoint_resume_verified=checkpoint_resume_verified,
    )
