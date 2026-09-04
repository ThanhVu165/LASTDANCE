"""Isolated Torch inference service. Communicates only through JSON lines."""

from __future__ import annotations

import contextlib
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from shared.schemas.online import FrameEvidence, TaskType


_ENCODERS: Any = None


@dataclass(frozen=True, slots=True)
class _ImageReference:
    path: Path

    def image_path(self, _root: Path) -> Path:
        return self.path


def _handle(message: dict[str, Any]) -> dict[str, Any]:
    global _ENCODERS
    operation = message.get("operation")
    if operation == "ping":
        return {"status": "READY"}
    if operation == "encode":
        from .encoders import TextEncoderRegistry

        if _ENCODERS is None:
            _ENCODERS = TextEncoderRegistry(device=str(message.get("device", "cpu")))
        vectors = _ENCODERS.encode(str(message["modality"]), [str(item) for item in message["texts"]])
        return {"vectors": vectors.tolist()}
    if operation == "qwen_plan":
        from .planners import QwenLocalQueryPlanner

        plan = QwenLocalQueryPlanner(str(message["model_id"])).plan(
            str(message["text"]), TaskType(str(message["task_type"]))
        )
        return {"plan": plan.model_dump(mode="json")}
    if operation == "qwen_vqa":
        from .vqa import QwenVQAAnswerer

        frames: list[FrameEvidence] = []
        images: dict[int, _ImageReference] = {}
        for raw in message["frames"]:
            value = dict(raw)
            path = Path(value.pop("image_path"))
            frame = FrameEvidence.model_validate(value)
            frames.append(frame)
            images[frame.keyframe_uid] = _ImageReference(path)
        registry = SimpleNamespace(
            catalog=SimpleNamespace(by_uid=images),
            layout=SimpleNamespace(data=SimpleNamespace(keyframes=Path("."))),
        )
        answer, confidence, warnings = QwenVQAAnswerer(
            registry,
            str(message["model_id"]),
            agreement_similarity=float(message.get("agreement_similarity", 0.6)),
        ).answer(
            video_id=str(message["video_id"]),
            frames=frames,
            question=str(message["question"]),
        )
        return {"answer": answer, "confidence": confidence, "warnings": warnings}
    if operation == "qwen_verify":
        from .verification import QwenVideoVerifier
        from shared.schemas.online import UnifiedQueryPlan, VideoHypothesis

        frames = []
        images: dict[int, _ImageReference] = {}
        for raw in message["frames"]:
            value = dict(raw)
            path = Path(value.pop("image_path"))
            frame = FrameEvidence.model_validate(value)
            frames.append(frame)
            images[frame.keyframe_uid] = _ImageReference(path)
        registry = SimpleNamespace(
            catalog=SimpleNamespace(by_uid=images),
            layout=SimpleNamespace(data=SimpleNamespace(keyframes=Path("."))),
        )
        result = QwenVideoVerifier(registry, str(message["model_id"])).verify(
            plan=UnifiedQueryPlan.model_validate(message["plan"]),
            hypothesis=VideoHypothesis.model_validate(message["hypothesis"]),
            frames=frames,
        )
        return {
            "must_have_score": result.must_have_score,
            "should_have_score": result.should_have_score,
            "scene_matches": result.scene_matches,
            "ranked_frame_ids": result.ranked_frame_ids,
            "provider": result.provider,
        }
    raise ValueError(f"unsupported Torch worker operation: {operation}")


def main() -> int:
    protocol_output = sys.stdout
    for line in sys.stdin:
        message: Any = None
        try:
            message = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                result = _handle(message)
            response = {"id": message.get("id"), "ok": True, "result": result}
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            response = {
                "id": message.get("id") if isinstance(message, dict) else None,
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
        protocol_output.write(json.dumps(response, ensure_ascii=False) + "\n")
        protocol_output.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
