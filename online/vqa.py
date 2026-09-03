"""Optional local Qwen3-VL answerer with two-prompt agreement."""

from __future__ import annotations

import json
import base64
import io
import math
import os
import re
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageOps

from shared.schemas.online import FrameEvidence

from .artifacts import ArtifactRegistry
from .gemini import GeminiJsonClient
from .qwen_runtime import get_qwen_components
from .task_heads import UnavailableAnswerer, VideoAnswerer


def _json_answer(text: str) -> tuple[str, float]:
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(text[start : end + 1])
            answer = str(payload.get("answer", "")).strip()
            confidence = float(payload.get("confidence", 0.0))
            return answer[:100], min(1.0, max(0.0, confidence))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    answer = re.sub(r"\s+", " ", text).strip()[:100]
    return answer, 0.5 if answer else 0.0


def _normalize_answer(text: str) -> str:
    return re.sub(r"[^\w]+", " ", text.casefold()).strip()


class QwenVQAAnswerer(VideoAnswerer):
    def __init__(self, registry: ArtifactRegistry, model_id: str = "Qwen/Qwen3-VL-2B-Instruct") -> None:
        self.registry = registry
        self.model_id = model_id
        self._model: Any = None
        self._processor: Any = None

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None:
            return self._model, self._processor
        self._model, self._processor = get_qwen_components(self.model_id)
        return self._model, self._processor

    def _contact_sheet(self, frames: Sequence[FrameEvidence]) -> tuple[Image.Image, list[FrameEvidence]]:
        selected = sorted(frames, key=lambda item: (item.pts_time, item.frame_id))[:6]
        panel_width = 224
        image_height = 224
        label_height = 24
        columns = min(3, len(selected))
        rows = math.ceil(len(selected) / columns)
        sheet = Image.new("RGB", (columns * panel_width, rows * (image_height + label_height)), "white")
        draw = ImageDraw.Draw(sheet)
        included: list[FrameEvidence] = []
        for index, frame in enumerate(selected):
            internal = self.registry.catalog.by_uid[frame.keyframe_uid]
            path = internal.image_path(self.registry.layout.data.keyframes)
            if not path.is_file():
                continue
            with Image.open(path) as source:
                panel = ImageOps.fit(source.convert("RGB"), (panel_width, image_height), method=Image.Resampling.LANCZOS)
            slot = len(included)
            x = (slot % columns) * panel_width
            y = (slot // columns) * (image_height + label_height)
            sheet.paste(panel, (x, y))
            draw.rectangle((x, y + image_height, x + panel_width, y + image_height + label_height), fill="white")
            label = chr(ord("A") + slot)
            draw.text((x + 4, y + image_height + 4), f"panel {label}", fill="black")
            included.append(frame)
        if not included:
            sheet.close()
            raise RuntimeError("no keyframe image is available for VQA")
        used_rows = math.ceil(len(included) / columns)
        if used_rows < rows:
            cropped = sheet.crop((0, 0, columns * panel_width, used_rows * (image_height + label_height)))
            sheet.close()
            sheet = cropped
        return sheet, included

    def _ask(self, frames: Sequence[FrameEvidence], question: str, instruction: str) -> tuple[str, float]:
        import torch

        model, processor = self._load()
        sheet, included = self._contact_sheet(frames)
        panel_names = ", ".join(chr(ord("A") + index) for index in range(len(included)))
        content: list[dict[str, Any]] = [{"type": "image", "image": sheet}]
        content.append(
            {
                "type": "text",
                "text": (
                    f"{instruction}\nPanels {panel_names} are chronological.\nQuestion: {question}\n"
                    "Use only visible evidence. Return JSON: "
                    '{"answer":"short answer","evidence_panel":"A","confidence":0.0}.'
                ),
            }
        )
        messages = [{"role": "user", "content": content}]
        try:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        finally:
            sheet.close()
        inputs = inputs.to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=int(os.environ.get("AIC_QWEN_VQA_MAX_NEW_TOKENS", "64")),
                do_sample=False,
            )
        generated = generated[:, inputs["input_ids"].shape[1] :]
        response = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return _json_answer(response)

    def answer(
        self,
        *,
        video_id: str,
        frames: Sequence[FrameEvidence],
        question: str,
    ) -> tuple[str, float, list[str]]:
        try:
            first, first_confidence = self._ask(frames, question, "Answer the question from these chronological candidate frames.")
            second, second_confidence = self._ask(frames, question, "Independently verify the answer and reject unsupported guesses.")
        except Exception as error:
            return "Uncertain", 0.0, [f"Qwen VQA failed for {video_id}: {type(error).__name__}: {error}"]
        if not first or _normalize_answer(first) != _normalize_answer(second):
            return "Uncertain", 0.0, [f"Qwen VQA prompts disagreed for {video_id}: {first!r} vs {second!r}"]
        return first[:100], min(first_confidence, second_confidence), []


class WorkerQwenVQAAnswerer(VideoAnswerer):
    """VQA proxy sharing the same isolated Torch/Qwen process as the planner."""

    def __init__(self, registry: ArtifactRegistry, model_id: str, client: Any = None) -> None:
        if client is None:
            from .torch_worker_client import get_torch_worker_client

            client = get_torch_worker_client()
        self.registry = registry
        self.model_id = model_id
        self.client = client

    def answer(
        self,
        *,
        video_id: str,
        frames: Sequence[FrameEvidence],
        question: str,
    ) -> tuple[str, float, list[str]]:
        payload = []
        for frame in frames:
            internal = self.registry.catalog.by_uid[frame.keyframe_uid]
            value = frame.model_dump(mode="json")
            value["image_path"] = str(internal.image_path(self.registry.layout.data.keyframes).resolve())
            payload.append(value)
        response = self.client.request(
            "qwen_vqa",
            model_id=self.model_id,
            video_id=video_id,
            question=question,
            frames=payload,
        )
        return str(response["answer"]), float(response["confidence"]), [str(item) for item in response["warnings"]]


class GeminiVQAAnswerer(VideoAnswerer):
    def __init__(self, registry: ArtifactRegistry, client: GeminiJsonClient) -> None:
        self.registry = registry
        self.client = client

    def _ask(
        self,
        frames: Sequence[FrameEvidence],
        question: str,
        instruction: str,
    ) -> tuple[str, float]:
        sheet, included = QwenVQAAnswerer(self.registry)._contact_sheet(frames)
        try:
            buffer = io.BytesIO()
            sheet.save(buffer, format="JPEG", quality=88)
        finally:
            sheet.close()
        prompt = (
            f"{instruction}\nQuestion: {question}\n"
            f"Allowed evidence frame IDs: {[frame.frame_id for frame in included]}\n"
            "Use only visible evidence. Return JSON with answer (maximum 100 characters), "
            "evidence_frame_id and confidence (0..1)."
        )
        payload = self.client.generate(
            [
                {
                    "inlineData": {
                        "mimeType": "image/jpeg",
                        "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
                    }
                },
                {"text": prompt},
            ],
            estimated_tokens=6000,
        )
        answer = str(payload.get("answer", ""))[:100]
        return answer, min(1.0, max(0.0, float(payload.get("confidence", 0.0))))

    def answer(
        self,
        *,
        video_id: str,
        frames: Sequence[FrameEvidence],
        question: str,
    ) -> tuple[str, float, list[str]]:
        try:
            first, first_confidence = self._ask(
                frames,
                question,
                "Answer the question from these chronological candidate frames.",
            )
            second, second_confidence = self._ask(
                frames,
                question,
                "Independently verify the answer and reject unsupported guesses.",
            )
        except Exception as error:
            return "Uncertain", 0.0, [
                f"Gemini VQA failed for {video_id}: {type(error).__name__}: {error}"
            ]
        if not first.strip() or _normalize_answer(first) != _normalize_answer(second):
            return "Uncertain", 0.0, [
                f"Gemini VQA prompts disagreed for {video_id}: {first!r} vs {second!r}"
            ]
        return first, min(first_confidence, second_confidence), []


class FallbackVQAAnswerer(VideoAnswerer):
    def __init__(
        self,
        providers: Sequence[VideoAnswerer],
        *,
        accept_confidence: float = 0.85,
    ) -> None:
        self.providers = list(providers)
        self.accept_confidence = accept_confidence

    def answer(
        self,
        *,
        video_id: str,
        frames: Sequence[FrameEvidence],
        question: str,
    ) -> tuple[str, float, list[str]]:
        warnings: list[str] = []
        best_answer = "Uncertain"
        best_confidence = 0.0
        for provider in self.providers:
            answer, confidence, provider_warnings = provider.answer(
                video_id=video_id,
                frames=frames,
                question=question,
            )
            warnings.extend(provider_warnings)
            if answer.strip().casefold() != "uncertain" and confidence > 0:
                if best_answer != "Uncertain" and _normalize_answer(answer) == _normalize_answer(best_answer):
                    return answer, max(confidence, best_confidence), warnings
                if confidence > best_confidence:
                    best_answer, best_confidence = answer, confidence
                if confidence >= self.accept_confidence:
                    return answer, confidence, warnings
        if best_answer != "Uncertain":
            warnings.append("Low-confidence answer kept for operator review after fallback verification")
            return best_answer, best_confidence, warnings
        return "Uncertain", 0.0, warnings or ["VQA unavailable; answer requires operator review"]


def get_video_answerer(registry: ArtifactRegistry) -> VideoAnswerer:
    providers: list[VideoAnswerer] = []
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        providers.append(GeminiVQAAnswerer(registry, GeminiJsonClient(api_key)))
    if os.environ.get("AIC_ENABLE_QWEN_VQA") == "1":
        model_id = os.environ.get("AIC_QWEN_MODEL", "Qwen/Qwen3-VL-2B-Instruct")
        default_worker = "1" if os.name == "nt" else "0"
        if os.environ.get("AIC_TORCH_WORKER", default_worker) != "0":
            providers.append(WorkerQwenVQAAnswerer(registry, model_id))
        else:
            providers.append(QwenVQAAnswerer(registry, model_id))
    return FallbackVQAAnswerer(providers) if providers else UnavailableAnswerer()
