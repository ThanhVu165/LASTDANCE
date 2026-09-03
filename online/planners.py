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
from shared.schemas.online import QueryRole, TaskType, UnifiedQueryPlan

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
_NUMBER_ANSWER = re.compile(
    r"\b(?:bao nhiêu|số mấy|con số|mã số|number|how many|what number)\b",
    flags=re.IGNORECASE,
)
_COLOR_ANSWER = re.compile(r"\b(?:màu gì|màu nào|what colou?r)\b", flags=re.IGNORECASE)
_PERSON_ANSWER = re.compile(r"\b(?:ai|người nào|who)\b", flags=re.IGNORECASE)
_PLACE_ANSWER = re.compile(r"\b(?:ở đâu|nơi nào|where)\b", flags=re.IGNORECASE)
_DESCRIPTIVE_TEXT_COUNT = re.compile(
    r"\b(?:\d+|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|ten|one|two|three|four|five|six|seven|eight|nine)\s+"
    r"(?:ký tự|chữ|characters?|letters?|words?)\b",
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
        if _DESCRIPTIVE_TEXT_COUNT.search(term):
            continue
        if term and term.casefold() in folded_raw and term not in terms:
            terms.append(term)
    return terms


def _answer_value_type(raw: str) -> str:
    if _NUMBER_ANSWER.search(raw):
        return "number"
    if _COLOR_ANSWER.search(raw):
        return "color"
    if _PERSON_ANSWER.search(raw):
        return "person"
    if _PLACE_ANSWER.search(raw):
        return "place"
    return "free_text"


def _answer_source(raw: str) -> str:
    visible = bool(_VISIBLE_CUES.search(raw))
    spoken = bool(_SPOKEN_CUES.search(raw))
    if visible and spoken:
        return "mixed"
    if visible:
        return "ocr"
    if spoken:
        return "asr"
    return "visual"


def _question_text(raw: str) -> str:
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\r?\n+", raw) if part.strip()]
    return next((part for part in reversed(parts) if "?" in part), parts[-1] if parts else raw)


def _visual_text_attributes(text: str) -> list[str]:
    """Keep a visual/OCR clue separate from text that is safe to send to FTS."""

    if not _VISIBLE_CUES.search(text):
        return []
    cleaned = re.sub(r"\s+", " ", text).strip(" ,.:;?-")
    return [cleaned[:200]] if cleaned else []


def _normalize_role_values(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value]
    aliases = {
        "LOCATOR": QueryRole.VIDEO_LOCATOR.value,
        "VIDEO": QueryRole.VIDEO_LOCATOR.value,
        "TARGET": QueryRole.TARGET_MOMENT.value,
        "ANSWER": QueryRole.ANSWER_EVIDENCE.value,
        "EVIDENCE": QueryRole.ANSWER_EVIDENCE.value,
        "EVENT": QueryRole.ORDERED_EVENT.value,
        "TRAKE_EVENT": QueryRole.ORDERED_EVENT.value,
    }
    allowed = {role.value for role in QueryRole}
    result: list[str] = []
    for item in values:
        role = aliases.get(str(item).strip().upper(), str(item).strip().upper())
        if role in allowed and role not in result:
            result.append(role)
    return result


def _grounded_source_span(raw: str, proposed: Any, fallback: str) -> tuple[str, bool]:
    text = re.sub(r"\s+", " ", str(proposed or "")).strip()
    folded_raw = re.sub(r"\s+", " ", raw).casefold()
    if text and text.casefold() in folded_raw:
        return text, True
    return fallback.strip() or raw, False


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
        question: str | None = _question_text(raw) if task_type == TaskType.QA else None
        discriminative = scenes[-1] if len(scenes) > 1 else raw
        retrieval_queries = [raw]
        if discriminative != raw:
            retrieval_queries.append(discriminative)

        units: list[dict[str, Any]] = []
        if task_type == TaskType.TRAKE:
            explicit_events = [re.sub(r"\s+", " ", item).strip(" .") for item in _TRAKE_EVENT.findall(raw)]
            ordered = explicit_events or scenes
            if explicit_events:
                marker = re.search(r"(?im)^\s*E\d+\s*[:.\-]?", raw)
                locator = raw[: marker.start()].strip() if marker else ""
                if locator:
                    units.append(
                        {
                            "unit_id": "locator-1",
                            "description_original": locator,
                            "retrieval_query_en": locator,
                            "roles": [QueryRole.VIDEO_LOCATOR.value],
                            "temporal_group": 0,
                            "confidence": 0.75,
                        }
                    )
            for index, event in enumerate(ordered):
                units.append(
                    {
                        "unit_id": f"event-{index + 1}",
                        "description_original": event,
                        "retrieval_query_en": event,
                        "roles": [
                            QueryRole.VIDEO_LOCATOR.value,
                            QueryRole.ORDERED_EVENT.value,
                        ],
                        "temporal_group": index + 1,
                        "temporal_order": index,
                        "confidence": 0.8,
                    }
                )
            ordered_event_ids = [f"event-{index + 1}" for index in range(len(ordered))]
            submission_target_ids: list[str] = []
            answer_target = None
        else:
            ordered = []
            target_index = len(scenes) - 1
            for index, scene in enumerate(scenes):
                roles = [QueryRole.VIDEO_LOCATOR.value]
                if index == target_index:
                    roles.append(QueryRole.TARGET_MOMENT.value)
                    if task_type == TaskType.QA:
                        roles.append(QueryRole.ANSWER_EVIDENCE.value)
                modalities = ["visual"]
                if index == target_index and _VISIBLE_CUES.search(raw):
                    modalities.append("ocr")
                if index == target_index and _SPOKEN_CUES.search(raw):
                    modalities.append("asr")
                units.append(
                    {
                        "unit_id": f"unit-{index + 1}",
                        "description_original": scene,
                        "retrieval_query_en": scene,
                        "roles": roles,
                        "modalities": modalities,
                        "temporal_group": index,
                        "temporal_order": index,
                        "known_text_literals": visible + spoken if index == target_index else [],
                        "visual_text_attributes": _visual_text_attributes(scene),
                        "confidence": 0.75,
                    }
                )
            target_id = f"unit-{target_index + 1}"
            ordered_event_ids = []
            submission_target_ids = [target_id]
            answer_target = (
                {
                    "question": question or raw,
                    "value_type": _answer_value_type(raw),
                    "source": _answer_source(raw),
                    "evidence_unit_ids": [target_id],
                    "value_is_unknown": True,
                }
                if task_type == TaskType.QA
                else None
            )
        source = _answer_source(raw) if task_type == TaskType.QA else None
        legacy_source = {
            "ocr": "visible_text",
            "asr": "spoken_text",
        }.get(source, source)
        return UnifiedQueryPlan(
            raw_query=raw,
            global_context_en=raw,
            query_units=units,
            answer_target=answer_target,
            ordered_event_ids=ordered_event_ids,
            submission_target_ids=submission_target_ids,
            planner_warnings=[
                "Rule planner cannot translate visual queries to English; review every unit before retrieval"
            ],
            caption_en=raw,
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
            answer_source=legacy_source,
            ordered_moments=ordered,
            planner_provider="rule",
        ).validate_for_task(task_type)


_PLANNER_SYSTEM = """You plan video moment retrieval. Return exactly one JSON object and no
explanation. Never invent a person, object, action, place, negative constraint, visible text or
speech. First split the raw query into real temporal moments; attributes visible at the same time
must stay in one unit. Then assign one or more roles to every unit: VIDEO_LOCATOR identifies the
video, TARGET_MOMENT identifies a KIS/QA submission frame, ANSWER_EVIDENCE localizes where a QA
answer can be read or seen, and ORDERED_EVENT is one TRAKE event that must be submitted. Context
outside explicit TRAKE events is VIDEO_LOCATOR only. A unit may have multiple roles.

description_original must be an exact contiguous span copied from the raw query. Translate each
retrieval_query_en and global_context_en faithfully to English. known_text_literals may contain
only exact text whose value is already stated verbatim in the raw query; a description such as
'six Chinese characters' or an unknown number being asked about is not a literal. Put script,
colour, count and placement clues into visual_text_attributes. For QA, answer_target describes the
unknown value and value_is_unknown must be true. Use source visual, ocr, asr or mixed. Return:
{"global_context_en":"string","retrieval_queries":["faithful English","optional discriminative English"],
"query_units":[{"unit_id":"unit-1","description_original":"exact raw span",
"retrieval_query_en":"English visual description","roles":["VIDEO_LOCATOR","TARGET_MOMENT"],
"requiredness":"must","modalities":["visual"],"temporal_group":0,"temporal_order":0,
"known_text_literals":[],"visual_text_attributes":[],"confidence":0.0}],
"submission_target_ids":[],"ordered_event_ids":[],
"answer_target":null,"must_have":[],"should_have":[]}.

KIS requires at least one TARGET_MOMENT in submission_target_ids. QA requires one answer_target
whose evidence_unit_ids reference ANSWER_EVIDENCE units; do not put the unknown answer into
known_text_literals. TRAKE requires ordered_event_ids to contain only the requested events in
chronological order. Do not return negative_constraints."""

_GEMINI_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "required": ["global_context_en", "retrieval_queries", "query_units"],
    "properties": {
        "global_context_en": {"type": "STRING"},
        "retrieval_queries": {"type": "ARRAY", "items": {"type": "STRING"}},
        "query_units": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "required": [
                    "unit_id",
                    "description_original",
                    "retrieval_query_en",
                    "roles",
                ],
                "properties": {
                    "unit_id": {"type": "STRING"},
                    "description_original": {"type": "STRING"},
                    "retrieval_query_en": {"type": "STRING"},
                    "roles": {
                        "type": "ARRAY",
                        "items": {
                            "type": "STRING",
                            "enum": [role.value for role in QueryRole],
                        },
                    },
                    "requiredness": {"type": "STRING", "enum": ["must", "should"]},
                    "modalities": {
                        "type": "ARRAY",
                        "items": {"type": "STRING", "enum": ["visual", "ocr", "asr"]},
                    },
                    "temporal_group": {"type": "INTEGER"},
                    "temporal_order": {"type": "INTEGER"},
                    "known_text_literals": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "visual_text_attributes": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "confidence": {"type": "NUMBER"},
                },
            },
        },
        "submission_target_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
        "ordered_event_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
        "answer_target": {
            "type": "OBJECT",
            "nullable": True,
            "properties": {
                "question": {"type": "STRING"},
                "value_type": {
                    "type": "STRING",
                    "enum": ["number", "color", "person", "place", "free_text"],
                },
                "source": {"type": "STRING", "enum": ["visual", "ocr", "asr", "mixed"]},
                "evidence_unit_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
                "value_is_unknown": {"type": "BOOLEAN"},
            },
        },
        "must_have": {"type": "ARRAY", "items": {"type": "STRING"}},
        "should_have": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
}


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
    proposed = dict(payload)
    raw_scenes = _scenes(raw_query)
    warnings: list[str] = []
    global_context = str(
        proposed.get("global_context_en") or proposed.get("caption_en") or raw_query
    ).strip()
    retrieval_queries = _as_text_list(
        proposed.get("retrieval_queries"), "query", "text", "caption"
    )[:2] or [global_context]
    provider_scenes = _as_text_list(
        proposed.get("scenes"), "scene_caption", "caption", "description", "event", "text"
    )
    must_have = _as_text_list(proposed.get("must_have"), "text", "value", "description")
    should_have = _as_text_list(proposed.get("should_have"), "text", "value", "description")
    if _as_text_list(proposed.get("negative_constraints"), "text", "value", "description"):
        warnings.append("Planner-proposed negative constraints were removed")

    proposed_visible = _as_text_list(proposed.get("visible_text"), "text", "value")
    proposed_spoken = _as_text_list(proposed.get("spoken_text"), "text", "value")
    for item in proposed.get("query_units") or []:
        if isinstance(item, dict):
            proposed_visible.extend(_as_text_list(item.get("known_text_literals"), "text", "value"))
    visible_literals = (
        _literal_visible_terms(raw_query, proposed_visible)
        if _VISIBLE_CUES.search(raw_query)
        else []
    )
    spoken_literals = []
    if _SPOKEN_CUES.search(raw_query):
        candidates = [*_QUOTED_TEXT.findall(raw_query), *proposed_spoken]
        folded_raw = raw_query.casefold()
        spoken_literals = list(
            dict.fromkeys(
                item.strip()
                for item in candidates
                if item.strip() and item.strip().casefold() in folded_raw
            )
        )

    units: list[dict[str, Any]] = []
    raw_units = proposed.get("query_units")
    if isinstance(raw_units, list):
        for index, item in enumerate(raw_units):
            if not isinstance(item, dict):
                continue
            fallback_source = raw_scenes[min(index, len(raw_scenes) - 1)]
            source, grounded = _grounded_source_span(
                raw_query, item.get("description_original"), fallback_source
            )
            if not grounded:
                warnings.append(
                    f"Query unit {index + 1} source span was not grounded and was repaired"
                )
            provider_fallback = (
                provider_scenes[min(index, len(provider_scenes) - 1)]
                if provider_scenes
                else source
            )
            retrieval_en = str(
                item.get("retrieval_query_en")
                or item.get("scene_caption")
                or item.get("description")
                or provider_fallback
            ).strip()
            roles = _normalize_role_values(item.get("roles"))
            raw_modalities = item.get("modalities") or ["visual"]
            if not isinstance(raw_modalities, list):
                raw_modalities = [raw_modalities]
            modalities = [
                str(value).strip().lower()
                for value in raw_modalities
                if str(value).strip().lower() in {"visual", "ocr", "asr"}
            ]
            if "visual" not in modalities:
                modalities.insert(0, "visual")
            literals = _literal_visible_terms(
                raw_query, _as_text_list(item.get("known_text_literals"), "text", "value")
            )
            attributes = [
                value
                for value in _as_text_list(item.get("visual_text_attributes"), "text", "value")
                if value.casefold() in raw_query.casefold()
            ] or _visual_text_attributes(source)
            try:
                confidence = min(1.0, max(0.0, float(item.get("confidence", 0.7))))
            except (TypeError, ValueError):
                confidence = 0.7
            try:
                temporal_group = max(0, int(item.get("temporal_group", index)))
            except (TypeError, ValueError):
                temporal_group = index
            try:
                temporal_order = max(0, int(item.get("temporal_order", index)))
            except (TypeError, ValueError):
                temporal_order = index
            units.append(
                {
                    "unit_id": str(item.get("unit_id") or f"unit-{index + 1}"),
                    "description_original": source,
                    "retrieval_query_en": retrieval_en,
                    "roles": roles or [QueryRole.VIDEO_LOCATOR.value],
                    "requiredness": item.get("requiredness") if item.get("requiredness") in {"must", "should"} else "must",
                    "modalities": list(dict.fromkeys(modalities)),
                    "temporal_group": temporal_group,
                    "temporal_order": temporal_order,
                    "known_text_literals": literals,
                    "visual_text_attributes": attributes,
                    "confidence": confidence,
                }
            )

    ordered = _as_text_list(
        proposed.get("ordered_moments"), "event", "moment", "description", "text", "scene_caption"
    )
    explicit_events = [
        re.sub(r"\s+", " ", item).strip(" .") for item in _TRAKE_EVENT.findall(raw_query)
    ]
    if not units:
        if task_type == TaskType.TRAKE:
            ordered = ordered if len(ordered) >= 2 else explicit_events or raw_scenes
            marker = re.search(r"(?im)^\s*E\d+\s*[:.\-]?", raw_query)
            locator = raw_query[: marker.start()].strip() if marker else ""
            if locator and explicit_events:
                units.append(
                    {
                        "unit_id": "locator-1",
                        "description_original": locator,
                        "retrieval_query_en": global_context,
                        "roles": [QueryRole.VIDEO_LOCATOR.value],
                        "temporal_group": 0,
                        "confidence": 0.8,
                    }
                )
            source_events = explicit_events or raw_scenes
            for index, event in enumerate(ordered):
                source = source_events[min(index, len(source_events) - 1)]
                units.append(
                    {
                        "unit_id": f"event-{index + 1}",
                        "description_original": source,
                        "retrieval_query_en": event,
                        "roles": [QueryRole.VIDEO_LOCATOR.value, QueryRole.ORDERED_EVENT.value],
                        "temporal_group": index + 1,
                        "temporal_order": index,
                        "confidence": 0.8,
                    }
                )
        else:
            scene_queries = provider_scenes or [global_context]
            source_values = raw_scenes or [raw_query]
            anchor = proposed.get("anchor_moment_index")
            try:
                anchor_index = int(anchor) if anchor is not None else len(scene_queries) - 1
            except (TypeError, ValueError):
                anchor_index = len(scene_queries) - 1
            if task_type == TaskType.QA:
                anchor_index = len(scene_queries) - 1
            anchor_index = min(max(anchor_index, 0), len(scene_queries) - 1)
            for index, query in enumerate(scene_queries):
                source = source_values[min(index, len(source_values) - 1)]
                roles = [QueryRole.VIDEO_LOCATOR.value]
                if index == anchor_index:
                    roles.append(QueryRole.TARGET_MOMENT.value)
                    if task_type == TaskType.QA:
                        roles.append(QueryRole.ANSWER_EVIDENCE.value)
                units.append(
                    {
                        "unit_id": f"unit-{index + 1}",
                        "description_original": source,
                        "retrieval_query_en": query,
                        "roles": roles,
                        "modalities": [
                            "visual",
                            *(["ocr"] if index == anchor_index and _VISIBLE_CUES.search(raw_query) else []),
                            *(["asr"] if index == anchor_index and _SPOKEN_CUES.search(raw_query) else []),
                        ],
                        "temporal_group": index,
                        "temporal_order": index,
                        "confidence": 0.7,
                    }
                )

    unit_lookup = {str(unit["unit_id"]): unit for unit in units}
    if len(unit_lookup) != len(units):
        for index, unit in enumerate(units):
            unit["unit_id"] = f"unit-{index + 1}"
        unit_lookup = {str(unit["unit_id"]): unit for unit in units}

    answer_target: dict[str, Any] | None = None
    submission_target_ids: list[str] = []
    ordered_event_ids: list[str] = []
    if task_type == TaskType.TRAKE:
        for unit in units:
            unit["roles"] = [
                role
                for role in _normalize_role_values(unit.get("roles"))
                if role in {QueryRole.VIDEO_LOCATOR.value, QueryRole.ORDERED_EVENT.value}
            ] or [QueryRole.VIDEO_LOCATOR.value]
        requested_ids = _as_text_list(proposed.get("ordered_event_ids"))
        ordered_event_ids = [
            identifier
            for identifier in requested_ids
            if identifier in unit_lookup
            and QueryRole.ORDERED_EVENT.value in unit_lookup[identifier]["roles"]
        ]
        if len(ordered_event_ids) < 2:
            ordered_event_ids = [
                str(unit["unit_id"])
                for unit in units
                if QueryRole.ORDERED_EVENT.value in unit["roles"]
            ]
        if len(ordered_event_ids) < 2:
            raise ValueError("provider did not identify at least two TRAKE events")
    elif task_type == TaskType.KIS:
        for unit in units:
            unit["roles"] = [
                role
                for role in _normalize_role_values(unit.get("roles"))
                if role in {QueryRole.VIDEO_LOCATOR.value, QueryRole.TARGET_MOMENT.value}
            ] or [QueryRole.VIDEO_LOCATOR.value]
        requested_ids = _as_text_list(proposed.get("submission_target_ids"))
        submission_target_ids = [
            identifier
            for identifier in requested_ids
            if identifier in unit_lookup
            and QueryRole.TARGET_MOMENT.value in unit_lookup[identifier]["roles"]
        ]
        if not submission_target_ids:
            submission_target_ids = [
                str(unit["unit_id"])
                for unit in units
                if QueryRole.TARGET_MOMENT.value in unit["roles"]
            ]
        if not submission_target_ids:
            units[-1]["roles"].append(QueryRole.TARGET_MOMENT.value)
            submission_target_ids = [str(units[-1]["unit_id"])]
    else:
        for unit in units:
            unit["roles"] = [
                role
                for role in _normalize_role_values(unit.get("roles"))
                if role
                in {
                    QueryRole.VIDEO_LOCATOR.value,
                    QueryRole.TARGET_MOMENT.value,
                    QueryRole.ANSWER_EVIDENCE.value,
                }
            ] or [QueryRole.VIDEO_LOCATOR.value]
        target_payload = proposed.get("answer_target")
        target_payload = target_payload if isinstance(target_payload, dict) else {}
        evidence_ids = _as_text_list(target_payload.get("evidence_unit_ids"))
        evidence_ids = [
            identifier
            for identifier in evidence_ids
            if identifier in unit_lookup
            and QueryRole.ANSWER_EVIDENCE.value in unit_lookup[identifier]["roles"]
        ]
        if not evidence_ids:
            evidence_ids = [
                str(unit["unit_id"])
                for unit in units
                if QueryRole.ANSWER_EVIDENCE.value in unit["roles"]
            ]
        if not evidence_ids:
            units[-1]["roles"].extend(
                [QueryRole.TARGET_MOMENT.value, QueryRole.ANSWER_EVIDENCE.value]
            )
            units[-1]["roles"] = list(dict.fromkeys(units[-1]["roles"]))
            evidence_ids = [str(units[-1]["unit_id"])]
        for identifier in evidence_ids:
            unit_lookup[identifier]["roles"] = list(
                dict.fromkeys(
                    [
                        *unit_lookup[identifier]["roles"],
                        QueryRole.TARGET_MOMENT.value,
                        QueryRole.ANSWER_EVIDENCE.value,
                    ]
                )
            )
        submission_target_ids = evidence_ids
        inferred_type = _answer_value_type(raw_query)
        proposed_type = str(target_payload.get("value_type") or "free_text")
        value_type = inferred_type if inferred_type != "free_text" else proposed_type
        if value_type not in {"number", "color", "person", "place", "free_text"}:
            value_type = "free_text"
        source = _answer_source(raw_query)
        answer_target = {
            "question": _question_text(raw_query),
            "value_type": value_type,
            "source": source,
            "evidence_unit_ids": evidence_ids,
            "value_is_unknown": True,
        }

    target_like_ids = set(submission_target_ids) | set(ordered_event_ids)
    if visible_literals or spoken_literals:
        for literal in [*visible_literals, *spoken_literals]:
            matching = [
                unit
                for unit in units
                if literal.casefold() in str(unit["description_original"]).casefold()
            ]
            destinations = matching or [
                unit for unit in units if str(unit["unit_id"]) in target_like_ids
            ]
            for unit in destinations:
                unit["known_text_literals"] = list(
                    dict.fromkeys([*unit.get("known_text_literals", []), literal])
                )
    scenes = [str(unit["retrieval_query_en"]) for unit in units]
    ordered_moments = [str(unit_lookup[identifier]["retrieval_query_en"]) for identifier in ordered_event_ids]
    anchor_index = next(
        (index for index, unit in enumerate(units) if str(unit["unit_id"]) in submission_target_ids),
        0,
    )
    source = answer_target["source"] if answer_target else None
    legacy_source = {"ocr": "visible_text", "asr": "spoken_text"}.get(source, source)
    result = UnifiedQueryPlan.model_validate(
        {
            "raw_query": raw_query,
            "global_context_en": global_context,
            "query_units": units,
            "answer_target": answer_target,
            "ordered_event_ids": ordered_event_ids,
            "submission_target_ids": submission_target_ids,
            "planner_warnings": warnings,
            "caption_en": global_context,
            "retrieval_queries": retrieval_queries,
            "scenes": scenes,
            "anchor_moment_index": anchor_index if scenes else None,
            "must_have": must_have,
            "should_have": should_have,
            "negative_constraints": [],
            "visible_text": visible_literals,
            "spoken_text": spoken_literals,
            "modality_weights": _modality_weights(visible_literals, spoken_literals),
            "question": answer_target["question"] if answer_target else None,
            "answer_format": "short answer, at most 100 characters" if answer_target else None,
            "answer_source": legacy_source,
            "ordered_moments": ordered_moments,
            "planner_provider": provider,
        }
    )
    return result.validate_for_task(task_type)


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
                "generationConfig": {
                    "temperature": 0.0,
                    "responseMimeType": "application/json",
                    "responseSchema": _GEMINI_RESPONSE_SCHEMA,
                },
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
            generated = model.generate(
                **inputs,
                max_new_tokens=int(os.environ.get("AIC_QWEN_PLANNER_MAX_NEW_TOKENS", "1280")),
                do_sample=False,
            )
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
