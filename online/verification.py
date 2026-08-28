"""Optional Gemini/Qwen visual verification for the strongest video hypotheses."""

from __future__ import annotations

import base64
import io
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from PIL import Image, ImageDraw, ImageOps

from shared.schemas.online import FrameEvidence, UnifiedQueryPlan, VideoHypothesis

from .artifacts import ArtifactRegistry
from .config import OnlineConfig
from .gemini import GeminiJsonClient


@dataclass(frozen=True, slots=True)
class VideoVerification:
    must_have_score: float
    should_have_score: float
    scene_matches: list[str]
    ranked_frame_ids: list[int]
    provider: str


class VideoVerifier(Protocol):
    def verify(
        self,
        *,
        plan: UnifiedQueryPlan,
        hypothesis: VideoHypothesis,
        frames: Sequence[FrameEvidence],
    ) -> VideoVerification: ...


def _clamp(value: Any) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _contact_sheets(
    registry: ArtifactRegistry,
    frames: Sequence[FrameEvidence],
    *,
    sheet_size: int,
) -> list[Image.Image]:
    result: list[Image.Image] = []
    for offset in range(0, len(frames), sheet_size):
        chunk = frames[offset : offset + sheet_size]
        panel_width, image_height, label_height = 224, 224, 26
        columns = min(4, len(chunk))
        rows = math.ceil(len(chunk) / columns)
        sheet = Image.new("RGB", (columns * panel_width, rows * (image_height + label_height)), "white")
        draw = ImageDraw.Draw(sheet)
        included = 0
        for frame in chunk:
            internal = registry.catalog.by_uid[frame.keyframe_uid]
            path = internal.image_path(registry.layout.data.keyframes)
            if not path.is_file():
                continue
            with Image.open(path) as source:
                panel = ImageOps.fit(
                    source.convert("RGB"),
                    (panel_width, image_height),
                    method=Image.Resampling.LANCZOS,
                )
            x = (included % columns) * panel_width
            y = (included // columns) * (image_height + label_height)
            sheet.paste(panel, (x, y))
            draw.text((x + 4, y + image_height + 4), f"frame_id={frame.frame_id}", fill="black")
            included += 1
        if included:
            used_rows = math.ceil(included / columns)
            if used_rows < rows:
                cropped = sheet.crop((0, 0, columns * panel_width, used_rows * (image_height + label_height)))
                sheet.close()
                sheet = cropped
            result.append(sheet)
        else:
            sheet.close()
    if not result:
        raise RuntimeError("no keyframe image is available for VLM verification")
    return result


def _prompt(plan: UnifiedQueryPlan, video_id: str, frame_ids: list[int]) -> str:
    return (
        "Verify a candidate video for fine-grained moment retrieval. Use only the supplied "
        "images. Do not infer unseen actions. Return JSON with keys must_have_score and "
        "should_have_score (0..1), scene_matches (strings copied from the requested scenes), "
        "and ranked_frame_ids (only integers from the allowed list, best first).\n"
        f"video_id={video_id}\n"
        f"caption={plan.caption_en}\n"
        f"scenes={json.dumps(plan.scenes, ensure_ascii=False)}\n"
        f"must_have={json.dumps(plan.must_have, ensure_ascii=False)}\n"
        f"should_have={json.dumps(plan.should_have, ensure_ascii=False)}\n"
        f"allowed_frame_ids={frame_ids}"
    )


def _verification(payload: dict[str, Any], provider: str, allowed: set[int]) -> VideoVerification:
    ranked: list[int] = []
    for value in payload.get("ranked_frame_ids", []):
        try:
            frame_id = int(value)
        except (TypeError, ValueError):
            continue
        if frame_id in allowed and frame_id not in ranked:
            ranked.append(frame_id)
    scenes = [str(value).strip() for value in payload.get("scene_matches", []) if str(value).strip()]
    return VideoVerification(
        must_have_score=_clamp(payload.get("must_have_score")),
        should_have_score=_clamp(payload.get("should_have_score")),
        scene_matches=scenes,
        ranked_frame_ids=ranked,
        provider=provider,
    )


class GeminiVideoVerifier:
    def __init__(self, registry: ArtifactRegistry, client: GeminiJsonClient) -> None:
        self.registry = registry
        self.client = client

    def verify(
        self,
        *,
        plan: UnifiedQueryPlan,
        hypothesis: VideoHypothesis,
        frames: Sequence[FrameEvidence],
    ) -> VideoVerification:
        sheets = _contact_sheets(self.registry, frames, sheet_size=12)
        parts: list[dict[str, Any]] = []
        try:
            for sheet in sheets:
                buffer = io.BytesIO()
                sheet.save(buffer, format="JPEG", quality=85)
                parts.append(
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
                        }
                    }
                )
            parts.append(
                {
                    "text": _prompt(
                        plan,
                        hypothesis.video_id,
                        [frame.frame_id for frame in frames],
                    )
                }
            )
            payload = self.client.generate(parts, estimated_tokens=12000)
        finally:
            for sheet in sheets:
                sheet.close()
        return _verification(
            payload,
            "gemini",
            {frame.frame_id for frame in frames},
        )


class QwenVideoVerifier:
    def __init__(self, registry: ArtifactRegistry, model_id: str) -> None:
        self.registry = registry
        self.model_id = model_id

    def verify(
        self,
        *,
        plan: UnifiedQueryPlan,
        hypothesis: VideoHypothesis,
        frames: Sequence[FrameEvidence],
    ) -> VideoVerification:
        import torch
        from .qwen_runtime import get_qwen_components

        model, processor = get_qwen_components(self.model_id)
        sheets = _contact_sheets(self.registry, frames, sheet_size=12)
        content: list[dict[str, Any]] = [
            {"type": "image", "image": sheet} for sheet in sheets
        ]
        content.append(
            {
                "type": "text",
                "text": _prompt(
                    plan,
                    hypothesis.video_id,
                    [frame.frame_id for frame in frames],
                ),
            }
        )
        try:
            inputs = processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        finally:
            for sheet in sheets:
                sheet.close()
        inputs = inputs.to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=384, do_sample=False)
        generated = generated[:, inputs["input_ids"].shape[1] :]
        text = processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise RuntimeError("Qwen verifier did not return JSON")
        return _verification(
            json.loads(text[start : end + 1]),
            "qwen-local",
            {frame.frame_id for frame in frames},
        )


class WorkerQwenVideoVerifier:
    def __init__(self, registry: ArtifactRegistry, model_id: str, client: Any = None) -> None:
        if client is None:
            from .torch_worker_client import get_torch_worker_client

            client = get_torch_worker_client()
        self.registry = registry
        self.model_id = model_id
        self.client = client

    def verify(
        self,
        *,
        plan: UnifiedQueryPlan,
        hypothesis: VideoHypothesis,
        frames: Sequence[FrameEvidence],
    ) -> VideoVerification:
        payload = []
        for frame in frames:
            internal = self.registry.catalog.by_uid[frame.keyframe_uid]
            value = frame.model_dump(mode="json")
            value["image_path"] = str(
                internal.image_path(self.registry.layout.data.keyframes).resolve()
            )
            payload.append(value)
        response = self.client.request(
            "qwen_verify",
            model_id=self.model_id,
            plan=plan.model_dump(mode="json"),
            hypothesis=hypothesis.model_dump(mode="json"),
            frames=payload,
        )
        return VideoVerification(
            must_have_score=float(response["must_have_score"]),
            should_have_score=float(response["should_have_score"]),
            scene_matches=[str(value) for value in response["scene_matches"]],
            ranked_frame_ids=[int(value) for value in response["ranked_frame_ids"]],
            provider=str(response["provider"]),
        )


class VerifierChain:
    def __init__(self, providers: Sequence[VideoVerifier]) -> None:
        self.providers = list(providers)
        self.last_errors: list[str] = []

    def verify(self, **kwargs: Any) -> VideoVerification:
        self.last_errors = []
        for provider in self.providers:
            try:
                return provider.verify(**kwargs)
            except Exception as error:
                self.last_errors.append(f"{type(provider).__name__}: {type(error).__name__}: {error}")
        raise RuntimeError("; ".join(self.last_errors) or "no VLM verifier is available")


def get_video_verifier(registry: ArtifactRegistry) -> VerifierChain | None:
    providers: list[VideoVerifier] = []
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        providers.append(GeminiVideoVerifier(registry, GeminiJsonClient(api_key)))
    if os.environ.get("AIC_ENABLE_QWEN_VQA") == "1":
        model_id = os.environ.get("AIC_QWEN_MODEL", "Qwen/Qwen3-VL-2B-Instruct")
        default_worker = "1" if os.name == "nt" else "0"
        if os.environ.get("AIC_TORCH_WORKER", default_worker) != "0":
            providers.append(WorkerQwenVideoVerifier(registry, model_id))
        else:
            providers.append(QwenVideoVerifier(registry, model_id))
    return VerifierChain(providers) if providers else None


def rerank_with_verifier(
    hypotheses: list[VideoHypothesis],
    plan: UnifiedQueryPlan,
    verifier: VideoVerifier | None,
    config: OnlineConfig,
) -> tuple[list[VideoHypothesis], list[str]]:
    if verifier is None:
        return hypotheses, []
    updated = list(hypotheses)
    warnings: list[str] = []
    for index, hypothesis in enumerate(updated[: config.vlm_video_top_k]):
        frames = hypothesis.best_frames[: config.vlm_frame_top_k]
        if not frames:
            continue
        try:
            result = verifier.verify(plan=plan, hypothesis=hypothesis, frames=frames)
        except Exception as error:
            warnings.append(
                f"VLM verification kept retrieval scores for {hypothesis.video_id}: "
                f"{type(error).__name__}: {error}"
            )
            continue
        rank_score = {
            frame_id: 1.0 - position / max(1, len(result.ranked_frame_ids) - 1)
            for position, frame_id in enumerate(result.ranked_frame_ids)
        }
        reranked_frames = []
        for frame in hypothesis.best_frames:
            if frame.frame_id not in rank_score:
                reranked_frames.append(frame)
                continue
            vlm_score = rank_score[frame.frame_id]
            reranked_frames.append(
                frame.model_copy(
                    update={
                        "vlm_score": vlm_score,
                        "final_score": (
                            config.frame_base_weight * frame.final_score
                            + config.frame_vlm_weight * vlm_score
                        ),
                    }
                )
            )
        reranked_frames.sort(key=lambda frame: frame.final_score, reverse=True)
        base = hypothesis.base_video_score or hypothesis.video_score
        video_score = (
            config.verified_base_weight * base
            + config.verified_must_weight * result.must_have_score
            + config.verified_should_weight * result.should_have_score
        )
        matched = result.scene_matches or hypothesis.matched_scenes
        updated[index] = hypothesis.model_copy(
            update={
                "video_score": video_score,
                "must_have_score": result.must_have_score,
                "should_have_score": result.should_have_score,
                "vlm_verified": True,
                "best_frames": reranked_frames,
                "matched_scenes": matched,
                "missing_scenes": [
                    scene for scene in plan.scenes if scene not in set(matched)
                ],
            }
        )
    updated.sort(key=lambda item: item.video_score, reverse=True)
    return updated, warnings
