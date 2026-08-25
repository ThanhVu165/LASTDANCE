"""Atomic, signature-aware progress checkpoints for resumable batch stages."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class StageProgress:
    signature: str
    next_index: int
    total: int

    @property
    def finished(self) -> bool:
        return self.next_index == self.total


class CheckpointStore:
    """Store stage progress without exposing a manually editable ready flag."""

    schema_version = 1

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _read_payload(self) -> dict[str, object]:
        if not self.path.exists():
            return {"schema_version": self.schema_version, "videos": {}}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != self.schema_version:
            raise RuntimeError("unsupported checkpoint schema version")
        if not isinstance(payload.get("videos"), dict):
            raise RuntimeError("invalid checkpoint videos mapping")
        return payload

    def get(self, video_id: str, stage: str) -> StageProgress | None:
        payload = self._read_payload()
        videos = payload["videos"]
        row = videos.get(video_id, {}).get("stages", {}).get(stage)
        if row is None:
            return None
        return StageProgress(
            signature=str(row["signature"]),
            next_index=int(row["next_index"]),
            total=int(row["total"]),
        )

    def update(
        self,
        *,
        video_id: str,
        stage: str,
        signature: str,
        next_index: int,
        total: int,
    ) -> StageProgress:
        if not video_id.strip() or not stage.strip() or not signature.strip():
            raise ValueError("video_id, stage, and signature must be non-empty")
        if total < 0 or next_index < 0 or next_index > total:
            raise ValueError("checkpoint progress must satisfy 0 <= next_index <= total")

        existing = self.get(video_id, stage)
        if existing is not None:
            if existing.signature != signature:
                raise RuntimeError("checkpoint signature mismatch")
            if existing.total != total:
                raise RuntimeError("checkpoint total mismatch")
            if next_index < existing.next_index:
                raise RuntimeError("checkpoint progress cannot move backwards")

        payload = self._read_payload()
        videos = payload["videos"]
        video = videos.setdefault(video_id, {"stages": {}})
        stages = video.setdefault("stages", {})
        stages[stage] = {
            "signature": signature,
            "next_index": next_index,
            "total": total,
        }
        self._write_atomic(payload)
        return StageProgress(signature, next_index, total)

    def _write_atomic(self, payload: dict[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f"{self.path.name}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)
