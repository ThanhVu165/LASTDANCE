"""Integrity helpers for external model and dataset artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected_sha256: str) -> str:
    expected = expected_sha256.strip().lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError("expected SHA-256 must contain exactly 64 hexadecimal characters")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"SHA-256 mismatch for {Path(path).name}: expected {expected}, got {actual}"
        )
    return actual
