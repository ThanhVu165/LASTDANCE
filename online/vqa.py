"""Optional local Qwen3-VL answerer with two-prompt agreement."""

from __future__ import annotations

import json
import base64
import unicodedata
import io
import math
import os
import re
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageOps

from shared.schemas.online import FrameEvidence, AnswerResult

from .artifacts import ArtifactRegistry
from .config import OnlineConfig
from .gemini import GeminiJsonClient
from .qwen_runtime import get_qwen_components
from .task_heads import UnavailableAnswerer, VideoAnswerer


def _normalize_answer(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def _answers_agree(first: str, second: str, *, threshold: float = 0.6) -> bool:
    # Legacy threshold stays readable; lexical similarity never proves semantics.
    if not 0 <= threshold <= 1:
        raise ValueError("answer agreement threshold must be between 0 and 1")
    return bool(first.strip() and second.strip() and _normalize_answer(first) == _normalize_answer(second))


def _parse_result(payload: dict, frames: Sequence[FrameEvidence], provider: str) -> AnswerResult:
    answer = payload.get("answer", "")
    if not isinstance(answer, str):
        raise ValueError("answer must be a string")
    if not answer.strip() or _normalize_answer(answer) == "uncertain":
        return AnswerResult(provider=provider, warnings=["No supported answer"])
    panel = payload.get("evidence_panel")
    frame_id = payload.get("evidence_frame_id")
    selected = None
    if isinstance(panel, str) and len(panel) == 1 and "A" <= panel <= "Z":
        index = ord(panel) - ord("A")
        if index < len(frames):
            selected = frames[index]
    if frame_id is not None:
        if type(frame_id) is not int:
            raise ValueError("evidence frame ID must be an integer")
        match = next((f for f in frames if f.frame_id == frame_id), None)
        if match is None or (selected is not None and selected != match):
            raise ValueError("unknown or inconsistent answer evidence frame")
        selected = match
    if selected is None:
        raise ValueError("answer has no valid evidence panel/frame")
    return AnswerResult(answer=answer.strip(), confidence=payload.get("confidence", 0),
                        evidence=[selected], provider=provider,
                        value_type=payload.get("value_type", "free_text"), unit=payload.get("unit"))


def _verified_pair(first: AnswerResult, second: AnswerResult) -> AnswerResult:
    common = {f.keyframe_uid for f in second.evidence}
    evidence = [f for f in first.evidence if f.keyframe_uid in common]
    agreed = (_answers_agree(first.answer, second.answer) and first.unit == second.unit
              and first.value_type == second.value_type and bool(evidence))
    if not agreed:
        return first.model_copy(update={"requires_review": True,
                                       "warnings": [*first.warnings, *second.warnings,
                                                    "Independent answers or evidence disagree; operator review required"]})
    return first.model_copy(update={"evidence": evidence, "confidence": min(first.confidence, second.confidence),
                                   "requires_review": first.value_type == "free_text"})


class QwenVQAAnswerer(VideoAnswerer):
    def __init__(
        self,
        registry: ArtifactRegistry,
        model_id: str = "Qwen/Qwen3-VL-2B-Instruct",
        *,
        agreement_similarity: float = 0.6,
    ) -> None:
        self.registry = registry
        self.model_id = model_id
        self.agreement_similarity = agreement_similarity
        self._model: Any = None
        self._processor: Any = None

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None:
            return self._model, self._processor
        self._model, self._processor = get_qwen_components(self.model_id)
        return self._model, self._processor

    def _contact_sheet(self, frames: Sequence[FrameEvidence]) -> tuple[Image.Image, list[FrameEvidence]]:
        selected = sorted(frames[:6], key=lambda item: (item.pts_time, item.frame_id))
        if not selected:
            raise ValueError("VQA requires evidence frames")
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
                panel = ImageOps.pad(source.convert("RGB"), (panel_width, image_height), method=Image.Resampling.LANCZOS, color="white")
            slot = len(included)
            x = (slot % columns) * panel_width
            y = (slot // columns) * (image_height + label_height)
            sheet.paste(panel, (x, y))
            draw.rectangle((x, y + image_height, x + panel_width, y + image_height + label_height), fill="white")
            label = chr(ord("A") + slot)
            draw.text((x + 4, y + image_height + 4), f"panel {label} / frame {frame.frame_id}", fill="black")
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

    def _ask(self, frames: Sequence[FrameEvidence], question: str, instruction: str) -> AnswerResult:
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
                    '{"answer":"short answer","evidence_panel":"A","value_type":"number|color|person|place|free_text","unit":null,"confidence":0.0}.'
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
        start, end = response.find("{"), response.rfind("}")
        return _parse_result(json.loads(response[start:end + 1]), included, "qwen")

    def answer(self, *, video_id: str, frames: Sequence[FrameEvidence], question: str) -> AnswerResult:
        try:
            first = self._ask(frames, question, "Answer using only the evidence for this question.")
            second = self._ask(frames, question, "Independently verify the answer, its units, and the supporting frame.")
            return _verified_pair(first, second)
        except Exception as error:
            return AnswerResult(warnings=[f"VQA failed for {video_id}: {type(error).__name__}: {error}"])


class WorkerQwenVQAAnswerer(VideoAnswerer):
    """VQA proxy sharing the same isolated Torch/Qwen process as the planner."""

    def __init__(
        self,
        registry: ArtifactRegistry,
        model_id: str,
        client: Any = None,
        *,
        agreement_similarity: float = 0.6,
    ) -> None:
        if client is None:
            from .torch_worker_client import get_torch_worker_client

            client = get_torch_worker_client()
        self.registry = registry
        self.model_id = model_id
        self.client = client
        self.agreement_similarity = agreement_similarity

    def answer(
        self,
        *,
        video_id: str,
        frames: Sequence[FrameEvidence],
        question: str,
    ) -> AnswerResult:
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
            agreement_similarity=self.agreement_similarity,
            frames=payload,
        )
        return AnswerResult.model_validate(response)


class GeminiVQAAnswerer(QwenVQAAnswerer):
    def __init__(
        self,
        registry: ArtifactRegistry,
        client: GeminiJsonClient,
        *,
        agreement_similarity: float = 0.6,
    ) -> None:
        self.registry = registry
        self.client = client
        self.agreement_similarity = agreement_similarity

    def _ask(
        self,
        frames: Sequence[FrameEvidence],
        question: str,
        instruction: str,
    ) -> AnswerResult:
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
            "evidence_frame_id, value_type (number/color/person/place/free_text), unit (or null), and confidence (0..1)."
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
        return _parse_result(payload, included, "gemini")


class FallbackVQAAnswerer(VideoAnswerer):
    def __init__(
        self,
        providers: Sequence[VideoAnswerer],
        *,
        accept_confidence: float = 0.85,
    ) -> None:
        self.providers = list(providers)
        self.accept_confidence = accept_confidence

    def answer(self, *, video_id: str, frames: Sequence[FrameEvidence], question: str) -> AnswerResult:
        warnings = []
        best = AnswerResult()
        for provider in self.providers:
            try:
                result = AnswerResult.model_validate(provider.answer(video_id=video_id, frames=frames, question=question))
            except Exception as error:
                warnings.append(f"Answer provider failed: {error}")
                continue
            warnings.extend(result.warnings)
            if not result.answer.strip():
                continue
            if result.confidence > best.confidence:
                best = result
            if not result.requires_review and result.confidence >= self.accept_confidence:
                return result.model_copy(update={"warnings": warnings})
        return best.model_copy(update={"requires_review": True, "warnings": warnings})


def get_video_answerer(
    registry: ArtifactRegistry,
    config: OnlineConfig | None = None,
) -> VideoAnswerer:
    agreement_similarity = config.qa_vqa_agreement_similarity if config is not None else 0.6
    providers: list[VideoAnswerer] = []
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        providers.append(
            GeminiVQAAnswerer(
                registry,
                GeminiJsonClient(api_key),
                agreement_similarity=agreement_similarity,
            )
        )
    if os.environ.get("AIC_ENABLE_QWEN_VQA") == "1":
        model_id = os.environ.get("AIC_QWEN_MODEL", "Qwen/Qwen3-VL-2B-Instruct")
        default_worker = "1" if os.name == "nt" else "0"
        if os.environ.get("AIC_TORCH_WORKER", default_worker) != "0":
            providers.append(
                WorkerQwenVQAAnswerer(
                    registry,
                    model_id,
                    agreement_similarity=agreement_similarity,
                )
            )
        else:
            providers.append(
                QwenVQAAnswerer(
                    registry,
                    model_id,
                    agreement_similarity=agreement_similarity,
                )
            )
    return FallbackVQAAnswerer(providers) if providers else UnavailableAnswerer()
