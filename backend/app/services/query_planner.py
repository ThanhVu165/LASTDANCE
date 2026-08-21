"""Structured, model-first planning for complete visual retrieval queries.

The generative model is asked for a bounded JSON plan.  A deterministic parser
validates the schema; the previous rule parser remains only as a safe fallback,
not as the primary semantic understanding path.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache

from app.config import (
    MODEL_QUERY_PLANNER_ENABLED,
    MODEL_QUERY_PLANNER_MAX_NEW_TOKENS,
)
from app.services.query_processing import parse_semantic_query
from app.services.visual_qa import generate_text


MAX_PLAN_SCENES = 8
MAX_QUERIES_PER_SCENE = 4
MAX_CRITERIA_PER_SCENE = 8
MAX_REPAIR_QUERIES = 12


@dataclass(frozen=True)
class PlannedScene:
    summary: str
    retrieval_queries: tuple[str, ...]
    must_have: tuple[str, ...]
    visible_text: tuple[str, ...]


@dataclass(frozen=True)
class QueryPlan:
    original_text: str
    language: str
    scenes: tuple[PlannedScene, ...]
    global_must_have: tuple[str, ...]
    temporal_edges: tuple[tuple[int, int], ...]
    repair_queries: tuple[str, ...]
    source: str

    @property
    def retrieval_prompts(self) -> tuple[tuple[str, int], ...]:
        prompts: list[tuple[str, int]] = []
        for scene_index, scene in enumerate(self.scenes):
            for prompt in (scene.summary, *scene.retrieval_queries):
                cleaned = " ".join(prompt.split())
                if cleaned and (cleaned, scene_index) not in prompts:
                    prompts.append((cleaned, scene_index))
        return tuple(prompts)

    @property
    def verification_query(self) -> str:
        criteria: list[str] = list(self.global_must_have)
        for index, scene in enumerate(self.scenes, 1):
            criteria.extend(f"Scene {index}: {item}" for item in scene.must_have)
            criteria.extend(f"Scene {index} visible text: {item}" for item in scene.visible_text)
        if not criteria:
            return self.original_text
        return (
            f"{self.original_text}\nAll of the following visual constraints are required: "
            + "; ".join(dict.fromkeys(criteria))
        )


def _strings(value: object, limit: int) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    cleaned = [
        " ".join(str(item).strip().split())
        for item in value
        if isinstance(item, str) and str(item).strip()
    ]
    return tuple(dict.fromkeys(cleaned[:limit]))


def _json_object(text: str) -> dict:
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.I)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Planner did not return a JSON object.")
    value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Planner JSON root must be an object.")
    return value


def _parse_plan(original_text: str, payload: dict) -> QueryPlan:
    raw_scenes = payload.get("scenes")
    if not isinstance(raw_scenes, list) or not raw_scenes:
        raise ValueError("Planner JSON must contain at least one scene.")
    scenes: list[PlannedScene] = []
    for raw in raw_scenes[:MAX_PLAN_SCENES]:
        if not isinstance(raw, dict):
            continue
        summary = " ".join(str(raw.get("summary", "")).strip().split())
        if not summary:
            continue
        queries = _strings(raw.get("retrieval_queries"), MAX_QUERIES_PER_SCENE)
        scenes.append(
            PlannedScene(
                summary=summary,
                retrieval_queries=queries,
                must_have=_strings(raw.get("must_have"), MAX_CRITERIA_PER_SCENE),
                visible_text=_strings(raw.get("visible_text"), MAX_CRITERIA_PER_SCENE),
            )
        )
    if not scenes:
        raise ValueError("Planner returned no usable scene.")

    edges: list[tuple[int, int]] = []
    raw_edges = payload.get("temporal_edges")
    if isinstance(raw_edges, list):
        for edge in raw_edges:
            if not isinstance(edge, list) or len(edge) != 2:
                continue
            try:
                left, right = int(edge[0]) - 1, int(edge[1]) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= left < len(scenes) and 0 <= right < len(scenes) and left != right:
                edges.append((left, right))

    language = str(payload.get("language", "")).strip().lower()
    if language not in {"vi", "en"}:
        language = parse_semantic_query(original_text).language
    return QueryPlan(
        original_text=original_text,
        language=language,
        scenes=tuple(scenes),
        global_must_have=_strings(payload.get("global_must_have"), MAX_CRITERIA_PER_SCENE),
        temporal_edges=tuple(dict.fromkeys(edges)),
        repair_queries=_strings(payload.get("repair_queries"), MAX_REPAIR_QUERIES),
        source="model",
    )


def _fallback_plan(text: str) -> QueryPlan:
    semantic = parse_semantic_query(text)
    scenes = tuple(
        PlannedScene(
            summary=scene,
            retrieval_queries=(scene,),
            must_have=(scene,),
            visible_text=tuple(semantic.ocr_keywords),
        )
        for scene in semantic.scenes
    )
    return QueryPlan(
        original_text=semantic.original_text,
        language=semantic.language,
        scenes=scenes,
        global_must_have=(),
        temporal_edges=tuple(semantic.temporal_edges),
        repair_queries=tuple(semantic.expansions),
        source="fallback",
    )


@lru_cache(maxsize=256)
def plan_visual_query(text: str) -> QueryPlan:
    fallback = _fallback_plan(text)
    if not MODEL_QUERY_PLANNER_ENABLED:
        return fallback
    prompt = (
        "Act as a visual-video retrieval planner. Convert the complete user query "
        "into grounded visual evidence without answering it or inventing facts. "
        "Separate distinct scenes only when the query describes distinct moments. "
        "Preserve entities, actions, counts, colours, attributes, spatial relations, "
        "places, names and visible text. retrieval_queries must contain concise open-"
        "vocabulary visual captions in the source language and English. must_have "
        "contains atomic conditions that must jointly hold in the same video window. "
        "repair_queries are alternative captions targeting the most distinctive or "
        "easy-to-miss evidence. temporal_edges use one-based scene numbers and only "
        "explicit ordering. Output JSON only with this schema: "
        '{"language":"vi|en","scenes":[{"summary":"...",'
        '"retrieval_queries":["..."],"must_have":["..."],'
        '"visible_text":["..."]}],"global_must_have":["..."],'
        '"temporal_edges":[[1,2]],"repair_queries":["..."]}.\n'
        f"Query: {fallback.original_text}"
    )
    try:
        decoded = generate_text(prompt, max_new_tokens=MODEL_QUERY_PLANNER_MAX_NEW_TOKENS)
        return _parse_plan(fallback.original_text, _json_object(decoded))
    except (RuntimeError, OSError, ValueError, json.JSONDecodeError):
        return fallback
