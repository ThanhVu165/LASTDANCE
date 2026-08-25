"""Stable identifiers shared by independently built offline artifacts."""

import hashlib


def _validate_identity_component(value: str, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    if any(character.isspace() for character in value):
        raise ValueError(f"{field_name} must not contain whitespace")


def make_keyframe_uid(video_id: str, shot_id: str, local_idx: int) -> int:
    """Return the spec-defined positive signed-int64 keyframe identifier."""

    _validate_identity_component(video_id, "video_id")
    _validate_identity_component(shot_id, "shot_id")
    if local_idx < 0:
        raise ValueError("local_idx must be non-negative")

    raw = f"{video_id}:{shot_id}:{local_idx}"
    digest = hashlib.blake2b(raw.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False) >> 1
