"""Query planning providers with one centralized fallback chain."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable

from shared.interfaces import QueryPlanner
from shared.schemas.online import TaskType, UnifiedQueryPlan

from .qwen_runtime import get_qwen_components


_SENTENCE_SPLIT = re.compile(
    r"(?:\r?\n+|(?<=[.!?;])\s+|\b(?:sau đó|tiếp theo|cuối cùng|then|next|finally)\b)",
    flags=re.IGNORECASE,
)
_QUOTED_TEXT = re.compile(r"[\"“”']([^\"“”']{2,100})[\"“”']")
_TRAKE_EVENT = re.compile(
    r"(?im)^\s*E\d+\s*[:.\-]?\s*(.+?)(?=\r?\n\s*E\d+\s*[:.\-]?|\Z)",
    flags=re.DOTALL,
)
_SPOKEN_CUES = re.compile(
    r"\b(?:nói|hỏi|trả lời|phát biểu|lời thoại|nghe thấy|say|says|said|asks?|speaks?|spoken|dialogue|quote)\b",
    flags=re.IGNORECASE,
)
_VISIBLE_CUES = re.compile(
    r"\b(?:dòng chữ|con số|số được ghi|ghi trên|đọc được|hiển thị (?:số|chữ)|biển báo|"
    r"hiển thị|biển hiệu|bảng chữ|logo|phụ đề|tiêu đề|viết|text|sign|banner|subtitle|"
    r"title|written|displayed|on-screen)\b",
    flags=re.IGNORECASE,
)
_VISIBLE_VALUE_BEFORE_DISPLAY = re.compile(
    r"(?:thông tin\s+(?:về\s+)?)?"
    r"((?:giá|mã|tên|số|chữ|dòng chữ)\b.{0,80}?)"
    r"(?:\s+được)?\s+hiển thị\b",
    flags=re.IGNORECASE,
)

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"


def _scenes(text: str) -> list[str]:
    parts = [part.strip(" ,-:\t") for part in _SENTENCE_SPLIT.split(text)]
    result = [part for part in parts if len(part) >= 3]
    return result[:8] or [text.strip()]


def _modality_weights(visible: list[str], spoken: list[str]) -> dict[str, float]:
    if visible and spoken:
        return {"visual": 0.50, "ocr": 0.25, "asr": 0.25}
    if visible:
        return {"visual": 0.55, "ocr": 0.45}
    if spoken:
        return {"visual": 0.55, "asr": 0.45}
    return {"visual": 1.0}


def _literal_visible_terms(raw: str, proposed: Iterable[str] = ()) -> list[str]:
    """Keep OCR terms literal and prefer the discriminative phrase over the full query."""

    terms: list[str] = []
    for value in _QUOTED_TEXT.findall(raw):
        term = re.sub(r"\s+", " ", value).strip()
        if term and term not in terms:
            terms.append(term)
    for match in _VISIBLE_VALUE_BEFORE_DISPLAY.findall(raw):
        term = re.sub(r"\s+", " ", match).strip(" ,.:;?-")
        if term.casefold() not in {"con số", "số", "chữ", "dòng chữ"} and term not in terms:
            terms.append(term)
    folded_raw = raw.casefold()
    for value in proposed:
        term = re.sub(r"\s+", " ", value).strip()
        if term and term.casefold() in folded_raw and term not in terms:
            terms.append(term)
    return terms


class RuleBasedQueryPlanner(QueryPlanner):
    """Deterministic last-resort planner; never requires network or a model."""

    def plan(self, text: str, task_type: TaskType) -> UnifiedQueryPlan:
        raw = text.strip()
        if not raw:
            raise ValueError("query must not be empty")
        scenes = _scenes(raw)
        quoted = [match.strip() for match in _QUOTED_TEXT.findall(raw)]
        visible = _literal_visible_terms(raw) if _VISIBLE_CUES.search(raw) else []
        spoken = quoted if quoted and _SPOKEN_CUES.search(raw) else []
        if _VISIBLE_CUES.search(raw) and not visible:
            visible = [raw]
        if _SPOKEN_CUES.search(raw) and not spoken:
            spoken = [raw]

        question: str | None = raw if task_type == TaskType.QA else None
        context = raw
        if task_type == TaskType.QA and "?" in raw:
            before, _, tail = raw.rpartition("?")
            sentence_start = max(before.rfind("."), before.rfind("!"), before.rfind("\n"))
            question = (before[sentence_start + 1 :].strip() + "?") if before else raw
            prefix = before[: sentence_start + 1].strip()
            context = prefix or raw
            scenes = _scenes(context)

        discriminative = max(scenes, key=len) if len(scenes) > 1 else raw
        retrieval_queries = [context]
        if discriminative != context:
            retrieval_queries.append(discriminative)
        if task_type == TaskType.TRAKE:
            explicit_events = [re.sub(r"\s+", " ", item).strip(" .") for item in _TRAKE_EVENT.findall(raw)]
            ordered = explicit_events or scenes
        else:
            ordered = []
        answer_source: str | None = None
        if task_type == TaskType.QA:
            answer_source = "mixed" if visible and spoken else "visible_text" if visible else "spoken_text" if spoken else "visual"
        return UnifiedQueryPlan(
            raw_query=raw,
            caption_en=context,
            retrieval_queries=retrieval_queries[:2],
            scenes=scenes,
            anchor_moment_index=len(scenes) - 1 if scenes else None,
            must_have=scenes[:4],
            should_have=[],
            negative_constraints=[],
            visible_text=visible,
            spoken_text=spoken,
            modality_weights=_modality_weights(visible, spoken),
            question=question,
            answer_format="short answer, at most 100 characters" if task_type == TaskType.QA else None,
            answer_source=answer_source,
            ordered_moments=ordered,
            planner_provider="rule",
        )


_PLANNER_SYSTEM = """You plan video moment retrieval. Return JSON only. Stay faithful to the
raw query: never invent a person, object, action, place, visible text or speech. Translate all
visual descriptions to English. caption_en, retrieval_queries, scenes, must_have, should_have,
negative_constraints and ordered_moments must be English. Create one faithful global retrieval
query and at most one discriminative global query. Keep every atomic scene/event as a separate
English string instead of compressing the story. Return exactly one flat object with these types
(never nest scene objects):
{"caption_en":"string","retrieval_queries":["string"],"scenes":["string"],
"anchor_moment_index":0,"must_have":["string"],"should_have":["string"],
"negative_constraints":["string"],"visible_text":["literal text from the raw query"],
"spoken_text":["literal speech from the raw query"],"modality_weights":{"visual":1.0},
"question":null,"answer_format":null,"answer_source":null,"ordered_moments":[]}.
Only QA may set question/answer fields. Only TRAKE may populate ordered_moments, which must
preserve event order. visible_text/spoken_text must be empty unless the raw query explicitly
asks about visible/spoken words. modality_weights may only use visual, ocr and asr."""


def _extract_json(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.IGNORECASE)
    start, end = value.find("{"), value.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError("planner did not return a JSON object")
    parsed = json.loads(value[start : end + 1])
    if not isinstance(parsed, dict):
        raise RuntimeError("planner JSON must be an object")
    return parsed


def _as_text_list(value: Any, *preferred_keys: str) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    result: list[str] = []
    for item in values:
        if isinstance(item, dict):
            selected = next((item.get(key) for key in preferred_keys if item.get(key)), None)
            item = selected
        if not isinstance(item, str):
            continue
        text = re.sub(r"\s+", " ", item).strip()
        if text and text not in result:
            result.append(text)
    return result


def _validate_provider_plan(
    raw_query: str,
    payload: dict[str, Any],
    provider: str,
    task_type: TaskType,
) -> UnifiedQueryPlan:
    payload = dict(payload)
    payload["raw_query"] = raw_query
    payload["planner_provider"] = provider
    caption = payload.get("caption_en")
    payload["caption_en"] = caption.strip() if isinstance(caption, str) and caption.strip() else raw_query
    payload["retrieval_queries"] = _as_text_list(
        payload.get("retrieval_queries"), "query", "text", "caption"
    )[:2] or [payload["caption_en"]]
    payload["scenes"] = _as_text_list(
        payload.get("scenes"), "scene_caption", "caption", "description", "event", "text"
    ) or [payload["caption_en"]]
    for name in ("must_have", "should_have", "negative_constraints"):
        payload[name] = _as_text_list(payload.get(name), "text", "value", "description")

    visible = _as_text_list(payload.get("visible_text"), "text", "value")
    spoken = _as_text_list(payload.get("spoken_text"), "text", "value")
    # Intent must originate in the user's query; a provider cannot introduce a
    # text modality merely because it described the image using words.
    visible_intent = bool(_VISIBLE_CUES.search(raw_query))
    spoken_intent = bool(_SPOKEN_CUES.search(raw_query))
    payload["visible_text"] = (
        _literal_visible_terms(raw_query, visible) or [raw_query]
    ) if visible_intent else []
    payload["spoken_text"] = (spoken or [raw_query]) if spoken_intent else []
    payload["modality_weights"] = _modality_weights(payload["visible_text"], payload["spoken_text"])

    anchor = payload.get("anchor_moment_index")
    try:
        anchor = int(anchor) if anchor is not None else None
    except (TypeError, ValueError):
        anchor = None
    payload["anchor_moment_index"] = anchor if anchor is not None and 0 <= anchor < len(payload["scenes"]) else 0

    ordered = _as_text_list(
        payload.get("ordered_moments"), "event", "moment", "description", "text", "scene_caption"
    )
    if task_type == TaskType.TRAKE:
        if len(ordered) < 2:
            explicit_events = [
                re.sub(r"\s+", " ", item).strip(" .")
                for item in _TRAKE_EVENT.findall(raw_query)
            ]
            deterministic = explicit_events or _scenes(raw_query)
            if len(deterministic) >= 2:
                ordered = deterministic
        payload["ordered_moments"] = ordered or payload["scenes"]
        payload["question"] = None
        payload["answer_format"] = None
        payload["answer_source"] = None
    elif task_type == TaskType.QA:
        payload["ordered_moments"] = []
        question = payload.get("question")
        payload["question"] = question.strip() if isinstance(question, str) and question.strip() else raw_query
        answer_format = payload.get("answer_format")
        payload["answer_format"] = (
            answer_format.strip()
            if isinstance(answer_format, str) and answer_format.strip()
            else "short answer, at most 100 characters"
        )
        if payload["visible_text"] and payload["spoken_text"]:
            payload["answer_source"] = "mixed"
        elif payload["visible_text"]:
            payload["answer_source"] = "visible_text"
        elif payload["spoken_text"]:
            payload["answer_source"] = "spoken_text"
        else:
            payload["answer_source"] = "visual"
    else:
        payload["ordered_moments"] = []
        payload["question"] = None
        payload["answer_format"] = None
        payload["answer_source"] = None
    return UnifiedQueryPlan.model_validate(payload)


class GeminiQueryPlanner(QueryPlanner):
    def __init__(self, api_key: str, *, model: str = DEFAULT_GEMINI_MODEL, timeout: float = 12.0) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def plan(self, text: str, task_type: TaskType) -> UnifiedQueryPlan:
        from .gemini import get_gemini_quota_manager

        get_gemini_quota_manager().acquire(estimated_tokens=2048)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        prompt = f"{_PLANNER_SYSTEM}\nTask: {task_type.value}\nRaw query: {text}"
        body = json.dumps(
            {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            try:
                detail = error.read().decode("utf-8", errors="replace")[:1000]
            except Exception:
                detail = ""
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(f"Gemini HTTP {error.code}{suffix}") from error
        result = payload["candidates"][0]["content"]["parts"][0]["text"]
        return _validate_provider_plan(text.strip(), _extract_json(result), "gemini", task_type)


class QwenLocalQueryPlanner(QueryPlanner):
    """Lazy local Qwen planner; only loads when CUDA (or explicit CPU override) is available."""

    def __init__(self, model_id: str = "Qwen/Qwen3-VL-2B-Instruct") -> None:
        self.model_id = model_id
        self._model: Any = None
        self._processor: Any = None

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None:
            return self._model, self._processor
        self._model, self._processor = get_qwen_components(self.model_id)
        return self._model, self._processor

    def plan(self, text: str, task_type: TaskType) -> UnifiedQueryPlan:
        import torch

        model, processor = self._load()
        prompt = f"{_PLANNER_SYSTEM}\nTask: {task_type.value}\nRaw query: {text}"
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        inputs = inputs.to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=768, do_sample=False)
        generated = generated[:, inputs["input_ids"].shape[1] :]
        result = processor.batch_decode(generated, skip_special_tokens=True)[0]
        return _validate_provider_plan(text.strip(), _extract_json(result), "qwen-local", task_type)


class WorkerQwenQueryPlanner(QueryPlanner):
    """Qwen planner proxy that keeps Torch outside the FAISS process."""

    def __init__(self, model_id: str = "Qwen/Qwen3-VL-2B-Instruct", client: Any = None) -> None:
        if client is None:
            from .torch_worker_client import get_torch_worker_client

            client = get_torch_worker_client()
        self.model_id = model_id
        self.client = client

    def plan(self, text: str, task_type: TaskType) -> UnifiedQueryPlan:
        response = self.client.request(
            "qwen_plan",
            model_id=self.model_id,
            text=text,
            task_type=task_type.value,
        )
        return UnifiedQueryPlan.model_validate(response["plan"])


@dataclass(slots=True)
class _CircuitState:
    failures: int = 0
    opened_at: float = 0.0


class PlannerChain(QueryPlanner):
    def __init__(self, providers: Iterable[QueryPlanner], *, failure_limit: int = 2, cooldown: float = 60.0) -> None:
        self.providers = list(providers)
        self.failure_limit = failure_limit
        self.cooldown = cooldown
        self._state = {id(provider): _CircuitState() for provider in self.providers}
        self.last_errors: list[str] = []

    def plan(self, text: str, task_type: TaskType) -> UnifiedQueryPlan:
        self.last_errors = []
        now = time.monotonic()
        for provider in self.providers:
            state = self._state[id(provider)]
            if state.failures >= self.failure_limit and now - state.opened_at < self.cooldown:
                self.last_errors.append(f"{type(provider).__name__}: circuit open")
                continue
            try:
                plan = provider.plan(text, task_type)
            except Exception as error:  # Provider failures intentionally degrade to the next provider.
                state.failures += 1
                state.opened_at = now
                self.last_errors.append(f"{type(provider).__name__}: {type(error).__name__}: {error}")
                continue
            state.failures = 0
            return plan
        raise RuntimeError("all query planners failed: " + "; ".join(self.last_errors))


def get_query_planner(environment: dict[str, str] | None = None) -> PlannerChain:
    values = os.environ if environment is None else environment
    providers: list[QueryPlanner] = []
    api_key = values.get("GEMINI_API_KEY", "").strip()
    if api_key:
        requested_model = values.get("AIC_GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
        model = requested_model or DEFAULT_GEMINI_MODEL
        providers.append(GeminiQueryPlanner(api_key, model=model))
    model_id = values.get("AIC_QWEN_MODEL", "Qwen/Qwen3-VL-2B-Instruct")
    default_worker = "1" if os.name == "nt" else "0"
    if values.get("AIC_TORCH_WORKER", default_worker) != "0":
        providers.append(WorkerQwenQueryPlanner(model_id))
    else:
        providers.append(QwenLocalQueryPlanner(model_id))
    providers.append(RuleBasedQueryPlanner())
    return PlannerChain(providers)
