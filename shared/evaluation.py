"""Official qualifier scoring; labels are human adjudications, never model guesses."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from shared.schemas.online import QuerySpec, TaskType

RANKS = (1, 5, 20, 50, 100)


def normalized_answer(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


class FrameInterval(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: int = Field(ge=0, strict=True)
    end: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def ordered(self) -> "FrameInterval":
        if self.end < self.start:
            raise ValueError("interval end precedes start")
        return self

    def contains(self, frame: int) -> bool:
        return self.start <= frame <= self.end


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: QuerySpec
    split: Literal["development", "held_out", "regression"]
    video_id: str = Field(min_length=1)
    intervals: list[FrameInterval] = Field(min_length=1)
    accepted_answers: list[str] = Field(default_factory=list)
    verified_by: str = Field(min_length=1)

    @model_validator(mode="after")
    def valid_label(self) -> "EvaluationCase":
        if not self.verified_by.strip():
            raise ValueError("labels require a human reviewer")
        expected = self.query.expected_event_count if self.query.task_type == TaskType.TRAKE else 1
        if len(self.intervals) != expected:
            raise ValueError("ground truth interval count does not match task")
        if self.query.task_type == TaskType.QA:
            if not self.accepted_answers or any(not normalized_answer(a) for a in self.accepted_answers):
                raise ValueError("QA requires human-approved semantic answer aliases")
        elif self.accepted_answers:
            raise ValueError("answer aliases are only valid for QA")
        return self


class Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    video_id: str = Field(min_length=1)
    frame_ids: list[int] = Field(min_length=1)
    answer: str | None = Field(default=None, max_length=100)

    @model_validator(mode="before")
    @classmethod
    def integer_frames(cls, value):
        if isinstance(value, dict):
            if any(type(x) is not int or x < 0 for x in value.get("frame_ids", [])):
                raise ValueError("prediction frame IDs must be nonnegative integers")
        return value


def score_case(case: EvaluationCase, predictions: list[Prediction]) -> dict[str, object]:
    if len(predictions) > 100:
        raise ValueError("a query may contain at most 100 predictions")
    scores: list[float] = []
    seen = set()
    for row in predictions:
        if len(row.frame_ids) != len(case.intervals):
            raise ValueError("prediction event count mismatch")
        key = (row.video_id, *row.frame_ids)
        if key in seen:
            raise ValueError("duplicate prediction")
        seen.add(key)
        if case.query.task_type == TaskType.TRAKE and any(b <= a for a, b in zip(row.frame_ids, row.frame_ids[1:])):
            raise ValueError("TRAKE frames must increase")
        if case.query.task_type == TaskType.QA and not (row.answer and row.answer.strip()):
            raise ValueError("QA prediction requires an answer")
        if case.query.task_type != TaskType.QA and row.answer is not None:
            raise ValueError("non-QA prediction cannot contain an answer")
        score = 0.0
        if row.video_id == case.video_id:
            hits = [span.contains(frame) for span, frame in zip(case.intervals, row.frame_ids)]
            if case.query.task_type == TaskType.TRAKE:
                score = sum(hits) / len(hits)
            elif hits[0]:
                score = 1.0
                if case.query.task_type == TaskType.QA:
                    score = float(normalized_answer(row.answer or "") in {normalized_answer(a) for a in case.accepted_answers})
        scores.append(score)
    ranks = {f"R@{k}": max(scores[:k], default=0.0) for k in RANKS}
    return {"query_name": case.query.query_name, "task_type": case.query.task_type.value,
            "row_scores": scores, **ranks, "final_score": sum(ranks.values()) / len(RANKS)}


def validate_suite(cases: list[EvaluationCase], *, acceptance: bool = False) -> None:
    names = [c.query.query_name for c in cases]
    if not cases or len(names) != len(set(names)):
        raise ValueError("evaluation requires nonempty unique queries")
    dev = {c.video_id for c in cases if c.split == "development"}
    held = {c.video_id for c in cases if c.split == "held_out"}
    if dev & held:
        raise ValueError("development and held-out videos overlap")
    if acceptance:
        counts = Counter((c.split, c.query.task_type) for c in cases)
        expected = {(split, task): 10 for split in ("development", "held_out") for task in TaskType}
        if dict(counts) != expected:
            raise ValueError("acceptance requires 60 reviewed queries: 10/task/split")


def evaluate(cases: list[EvaluationCase], predictions: dict[str, list[Prediction]], *, split: str) -> dict[str, object]:
    validate_suite(cases)
    unknown = set(predictions) - {c.query.query_name for c in cases}
    if unknown:
        raise ValueError(f"predictions contain unknown queries: {sorted(unknown)}")
    selected = [c for c in cases if c.split == split]
    if not selected:
        raise ValueError("selected split is empty")
    reports = [score_case(c, predictions.get(c.query.query_name, [])) for c in selected]
    per_task = {}
    for task in TaskType:
        rows = [r for r in reports if r["task_type"] == task.value]
        if rows:
            per_task[task.value] = {key: sum(float(r[key]) for r in rows) / len(rows)
                                    for key in [*(f"R@{k}" for k in RANKS), "final_score"]}
    label_hash = hashlib.sha256(json.dumps([c.model_dump(mode="json") for c in sorted(cases, key=lambda c: c.query.query_name)], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return {"schema_version": 1, "split": split, "labels_sha256": label_hash, "queries": reports,
            "per_task": per_task, "mean_final_score": sum(float(r["final_score"]) for r in reports) / len(reports)}


def diagnostic_metrics(cases, predictions, *, split, catalog_frames, runs=None):
    """Metrics derived from actual labels/predictions; missing timing is unavailable."""
    import math
    import statistics
    by_video = {}
    for frame in catalog_frames:
        by_video.setdefault(frame.video_id, []).append(frame.frame_id)
    selected = [case for case in cases if case.split == split]
    details = []
    latencies, review_flags = [], []
    for case in selected:
        rows = predictions.get(case.query.query_name, [])
        support = [any(span.contains(frame) for frame in by_video.get(case.video_id, [])) for span in case.intervals]
        aligned = [row for row in rows if row.video_id == case.video_id]
        distances = [min((max(span.start - row.frame_ids[i], 0, row.frame_ids[i] - span.end) for row in aligned), default=None)
                     for i, span in enumerate(case.intervals)]
        details.append({"query_name": case.query.query_name, "catalog_event_support": support,
                        "nearest_prediction_distance_frames": distances,
                        "video_recall": {f"R@{k}": int(any(r.video_id == case.video_id for r in rows[:k])) for k in RANKS}})
        run = (runs or {}).get(case.query.query_name)
        if run is not None:
            if run.get("config_sha256") is None:
                raise ValueError("benchmark trace lacks configuration identity")
            ms = run.get("elapsed_ms")
            if not isinstance(ms, (int, float)) or not math.isfinite(ms) or ms < 0:
                raise ValueError("benchmark elapsed_ms must be finite and nonnegative")
            latencies.append(ms)
            review_flags.append(bool(run["requires_operator_review"]))
    flattened = [value for row in details for value in row["catalog_event_support"]]
    return {"queries": details, "catalog_event_support_fraction": sum(flattened) / len(flattened),
            "video_recall": {f"R@{k}": sum(row["video_recall"][f"R@{k}"] for row in details) / len(details) for k in RANKS},
            "timed_queries": len(latencies), "latency_p50_ms": statistics.median(latencies) if latencies else None,
            "latency_p95_ms": sorted(latencies)[max(0, math.ceil(.95 * len(latencies)) - 1)] if latencies else None,
            "operator_review_query_fraction": sum(review_flags) / len(review_flags) if review_flags else None}
