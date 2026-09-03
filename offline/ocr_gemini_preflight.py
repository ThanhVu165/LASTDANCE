"""Build an exact, API-free Gemini residual/request manifest after Vintern."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GeminiProductionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    model_candidate: Literal["gemini-2.5-flash-lite"]
    model_requires_paid_canary_http_200: bool = True
    media_resolution: Literal["MEDIA_RESOLUTION_MEDIUM"]
    max_regions_per_contact_sheet: int = Field(ge=1, le=64)
    max_budget_vnd: int = Field(gt=0)
    retry_reserve_fraction: float = Field(ge=0, lt=1)
    usd_to_vnd: float = Field(gt=0)
    pricing_snapshot_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    standard_input_usd_per_million_tokens: float = Field(ge=0)
    standard_output_usd_per_million_tokens: float = Field(ge=0)
    batch_input_usd_per_million_tokens: float = Field(ge=0)
    batch_output_usd_per_million_tokens: float = Field(ge=0)
    gemini_2_5_medium_image_tokens: int = Field(gt=0)
    planning_prompt_tokens_per_request: int = Field(ge=0)
    planning_prompt_tokens_per_region: int = Field(ge=0)
    planning_output_tokens_per_request: int = Field(ge=0)
    planning_output_tokens_per_region: int = Field(ge=0)

    @model_validator(mode="after")
    def _batch_must_not_cost_more_than_standard(self) -> "GeminiProductionPolicy":
        if self.batch_input_usd_per_million_tokens > self.standard_input_usd_per_million_tokens:
            raise ValueError("batch input price cannot exceed standard price")
        if self.batch_output_usd_per_million_tokens > self.standard_output_usd_per_million_tokens:
            raise ValueError("batch output price cannot exceed standard price")
        return self


def stable_request_id(batch_id: str, video_id: str, shot_id: str, region_ids: Sequence[str]) -> str:
    payload = "|".join((batch_id, video_id, shot_id, *sorted(region_ids)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def residual_regions_from_materialized(
    frames: Sequence[dict[str, Any]], *, batch_id: str
) -> list[dict[str, Any]]:
    """Flatten calibrated residuals while preserving detector bbox/source evidence."""

    rows: list[dict[str, Any]] = []
    seen_regions: set[str] = set()
    for frame in frames:
        keyframe_uid = frame.get("keyframe_uid")
        if isinstance(keyframe_uid, bool) or not isinstance(keyframe_uid, int):
            raise ValueError("materialized frame requires integer keyframe_uid")
        for region in frame.get("regions") or []:
            region_id = str(region.get("region_id") or "")
            if not region_id:
                raise ValueError("materialized region missing region_id")
            if region_id in seen_regions:
                raise ValueError(f"duplicate materialized region_id: {region_id}")
            seen_regions.add(region_id)
            if not bool(region.get("gemini_residual")):
                continue
            bbox = region.get("bbox_px")
            if not isinstance(bbox, list) or len(bbox) != 8:
                raise ValueError(f"residual {region_id} requires bbox_px with exactly 8 values")
            rows.append(
                {
                    "schema_version": 1,
                    "batch_id": batch_id,
                    "region_id": region_id,
                    "candidate_id": region_id,
                    "video_id": str(frame["video_id"]),
                    "shot_id": str(frame["shot_id"]),
                    "local_idx": int(frame["local_idx"]),
                    "keyframe_uid": keyframe_uid,
                    "source_image": str(frame["source_image"]),
                    "bbox_px": [float(value) for value in bbox],
                    "easyocr_text": str(region.get("easyocr_text") or ""),
                    "easyocr_confidence": float(region.get("easyocr_confidence") or 0.0),
                    "final_text_before_gemini": str(region.get("final_text") or ""),
                    "final_confidence_before_gemini": float(
                        region.get("final_confidence") or 0.0
                    ),
                    "final_engine_before_gemini": str(region.get("final_engine") or ""),
                    "gemini_residual_reasons": list(
                        region.get("gemini_residual_reasons") or []
                    ),
                }
            )
    rows.sort(
        key=lambda row: (
            row["batch_id"],
            row["video_id"],
            row["shot_id"],
            row["local_idx"],
            row["region_id"],
        )
    )
    return rows


def build_shot_requests(
    residual_regions: Sequence[dict[str, Any]], *, policy: GeminiProductionPolicy
) -> list[dict[str, Any]]:
    """Group every residual in a shot into one request with one or more crop sheets."""

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for row in residual_regions:
        region_id = str(row.get("region_id") or "")
        if not region_id or region_id in seen:
            raise ValueError(f"missing/duplicate residual region_id: {region_id!r}")
        seen.add(region_id)
        grouped[(str(row["batch_id"]), str(row["video_id"]), str(row["shot_id"]))].append(row)

    requests: list[dict[str, Any]] = []
    for (batch_id, video_id, shot_id), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: (int(row["local_idx"]), str(row["region_id"])))
        region_ids = [str(row["region_id"]) for row in rows]
        pages = [
            region_ids[index : index + policy.max_regions_per_contact_sheet]
            for index in range(0, len(region_ids), policy.max_regions_per_contact_sheet)
        ]
        requests.append(
            {
                "schema_version": 1,
                "request_id": stable_request_id(batch_id, video_id, shot_id, region_ids),
                "batch_id": batch_id,
                "video_id": video_id,
                "shot_id": shot_id,
                "region_ids": region_ids,
                "region_pages": pages,
                "region_count": len(region_ids),
                "contact_sheet_count": len(pages),
                "keyframe_uids": sorted({int(row["keyframe_uid"]) for row in rows}),
            }
        )
    if len(requests) != len({row["request_id"] for row in requests}):
        raise ValueError("duplicate Gemini request_id")
    return requests


def planning_cost(
    *, region_count: int, request_count: int, contact_sheet_count: int, policy: GeminiProductionPolicy
) -> dict[str, Any]:
    """Return a conservative planning estimate; API usage remains authoritative."""

    input_tokens = (
        contact_sheet_count * policy.gemini_2_5_medium_image_tokens
        + request_count * policy.planning_prompt_tokens_per_request
        + region_count * policy.planning_prompt_tokens_per_region
    )
    output_tokens = (
        request_count * policy.planning_output_tokens_per_request
        + region_count * policy.planning_output_tokens_per_region
    )

    def estimate(input_price: float, output_price: float) -> dict[str, Any]:
        usd = input_tokens / 1_000_000 * input_price + output_tokens / 1_000_000 * output_price
        buffered_usd = usd * (1 + policy.retry_reserve_fraction)
        buffered_vnd = math.ceil(buffered_usd * policy.usd_to_vnd)
        return {
            "usd_before_retry_reserve": usd,
            "usd_with_retry_reserve": buffered_usd,
            "vnd_with_retry_reserve": buffered_vnd,
            "within_budget": buffered_vnd <= policy.max_budget_vnd,
        }

    return {
        "kind": "planning_estimate_not_billing_fact",
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": output_tokens,
        "pricing_snapshot_date": policy.pricing_snapshot_date,
        "standard": estimate(
            policy.standard_input_usd_per_million_tokens,
            policy.standard_output_usd_per_million_tokens,
        ),
        "batch": estimate(
            policy.batch_input_usd_per_million_tokens,
            policy.batch_output_usd_per_million_tokens,
        ),
        "max_budget_vnd": policy.max_budget_vnd,
        "retry_reserve_fraction": policy.retry_reserve_fraction,
        "limitations": [
            "Residual region/frame/shot/request counts are exact for the supplied artifacts.",
            "Token and cost values are planning estimates until a paid-key canary returns usageMetadata.",
            "One shot remains one request; dense shots may contain multiple image parts/contact sheets.",
        ],
    }


def build_preflight_report(
    residual_regions: Sequence[dict[str, Any]],
    requests: Sequence[dict[str, Any]],
    *,
    policy: GeminiProductionPolicy,
    batch_summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    region_ids = [str(row["region_id"]) for row in residual_regions]
    if len(region_ids) != len(set(region_ids)):
        raise ValueError("duplicate residual region_id")
    request_region_ids = [
        region_id for request in requests for region_id in request["region_ids"]
    ]
    if set(request_region_ids) != set(region_ids) or len(request_region_ids) != len(region_ids):
        raise ValueError("request manifest is not an exact residual partition")
    frames = {int(row["keyframe_uid"]) for row in residual_regions}
    shots = {
        (str(row["batch_id"]), str(row["video_id"]), str(row["shot_id"]))
        for row in residual_regions
    }
    reasons = Counter(
        reason
        for row in residual_regions
        for reason in row.get("gemini_residual_reasons") or []
    )
    sheets = sum(int(request["contact_sheet_count"]) for request in requests)
    decision = "NO_GEMINI_REQUIRED" if not residual_regions else "READY_FOR_USER_GEMINI_DECISION"
    return {
        "schema_version": 1,
        "decision": decision,
        "api_called": False,
        "model_candidate": policy.model_candidate,
        "runtime_model_pinned": False,
        "paid_canary_required": policy.model_requires_paid_canary_http_200,
        "media_resolution": policy.media_resolution,
        "exact_counts": {
            "regions": len(residual_regions),
            "frames": len(frames),
            "shots": len(shots),
            "requests": len(requests),
            "contact_sheets": sheets,
        },
        "residual_reasons": dict(sorted(reasons.items())),
        "max_regions_in_one_request": max(
            (int(request["region_count"]) for request in requests), default=0
        ),
        "max_contact_sheets_in_one_request": max(
            (int(request["contact_sheet_count"]) for request in requests), default=0
        ),
        "batches": {key: dict(value) for key, value in sorted(batch_summaries.items())},
        "cost": planning_cost(
            region_count=len(residual_regions),
            request_count=len(requests),
            contact_sheet_count=sheets,
            policy=policy,
        ),
        "gate": {
            "gemini_execution_authorized": False,
            "requires_user_decision_after_exact_report": True,
            "requires_report_sha256_pin": True,
            "requires_paid_canary_http_200_and_schema_valid": True,
        },
    }
