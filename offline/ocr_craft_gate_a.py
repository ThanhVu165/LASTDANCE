"""Evaluate human-labeled CRAFT Gate A threshold evidence without running OCR."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CraftThresholdConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    config_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    text_threshold: float = Field(gt=0, lt=1)
    low_text: float = Field(gt=0, lt=1)
    link_threshold: float = Field(gt=0, lt=1)


class CraftGateAPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1, 2] = 1
    sample_frames: int = Field(default=300, ge=1)
    video_ids: tuple[str, ...]
    frames_per_video: int | None = Field(default=60, ge=1)
    sample_video_counts: dict[str, int] | None = None
    sample_uid_set_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    allow_current_fallback_for_gate_b: bool = False
    evidence_limitations: tuple[str, ...] = ()
    min_region_recall: float = Field(default=0.98, gt=0, le=1)
    min_text_frame_recall: float = Field(default=0.99, gt=0, le=1)
    configs: tuple[CraftThresholdConfig, ...]

    @model_validator(mode="after")
    def _validate_plan(self) -> "CraftGateAPolicy":
        if not self.video_ids or len(self.video_ids) != len(set(self.video_ids)):
            raise ValueError("video_ids must be non-empty and unique")
        if self.sample_video_counts is None:
            if self.frames_per_video is None:
                raise ValueError("frames_per_video is required for a balanced sample")
            expected_counts = {
                video_id: self.frames_per_video for video_id in self.video_ids
            }
        else:
            if self.frames_per_video is not None:
                raise ValueError(
                    "frames_per_video must be null when sample_video_counts is provided"
                )
            if set(self.sample_video_counts) != set(self.video_ids):
                raise ValueError("sample_video_counts must cover every policy video_id")
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in self.sample_video_counts.values()
            ):
                raise ValueError("sample_video_counts must contain non-negative integers")
            expected_counts = self.sample_video_counts
        if self.sample_frames != sum(expected_counts.values()):
            raise ValueError("sample_frames must equal the configured per-video counts")
        if self.schema_version == 1 and self.sample_uid_set_sha256 is not None:
            raise ValueError("schema v1 cannot pin a sample UID-set checksum")
        if self.allow_current_fallback_for_gate_b and self.schema_version != 2:
            raise ValueError("deadline fallback requires policy schema v2")
        if self.allow_current_fallback_for_gate_b and not self.evidence_limitations:
            raise ValueError("deadline fallback must state its evidence limitations")
        ids = [value.config_id for value in self.configs]
        if len(ids) < 2 or len(ids) != len(set(ids)):
            raise ValueError("CRAFT config IDs must be unique and include at least two choices")
        if "recall_current" not in ids:
            raise ValueError("policy must retain recall_current as fail-closed fallback")
        return self

    @property
    def expected_video_counts(self) -> dict[str, int]:
        if self.sample_video_counts is not None:
            return dict(self.sample_video_counts)
        assert self.frames_per_video is not None
        return {video_id: self.frames_per_video for video_id in self.video_ids}


def policy_sha256(policy: CraftGateAPolicy) -> str:
    payload = json.dumps(
        policy.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_gate_a_selected_config(
    report_path: Path,
    *,
    policy: CraftGateAPolicy,
) -> CraftThresholdConfig:
    """Load a PASS report and reject any threshold/policy drift before Gate B."""

    try:
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Gate A report is not valid JSON") from exc
    if not isinstance(report, dict):
        raise ValueError("Gate A report must be an object")
    decision = report.get("decision")
    allowed_decisions = {"PASS_THRESHOLD_SELECTED"}
    if policy.allow_current_fallback_for_gate_b:
        allowed_decisions.add("DEADLINE_OVERRIDE_KEEP_CURRENT")
    if decision not in allowed_decisions or report.get("gate_b_allowed") is not True:
        raise ValueError("Gate A has not authorized Gate B")
    if report.get("policy_sha256") != policy_sha256(policy):
        raise ValueError("Gate A report/policy SHA mismatch")
    selected_id = str(report.get("selected_config_id") or "")
    config_by_id = {config.config_id: config for config in policy.configs}
    selected = config_by_id.get(selected_id)
    if selected is None:
        raise ValueError("Gate A selected an unknown CRAFT config")
    if report.get("selected_thresholds") != selected.model_dump(mode="json"):
        raise ValueError("Gate A selected threshold values drifted from policy")
    metrics = report.get("metrics")
    selected_metrics = metrics.get(selected_id) if isinstance(metrics, dict) else None
    if not isinstance(selected_metrics, dict):
        raise ValueError("Gate A selected config metrics are missing")
    if decision == "PASS_THRESHOLD_SELECTED":
        if selected_metrics.get("eligible") is not True:
            raise ValueError("Gate A selected config is not marked eligible")
        if float(selected_metrics.get("region_recall", -1)) < policy.min_region_recall:
            raise ValueError("Gate A selected config fails region recall")
        if float(selected_metrics.get("text_frame_recall", -1)) < policy.min_text_frame_recall:
            raise ValueError("Gate A selected config fails text-frame recall")
    else:
        if selected_id != "recall_current" or selected_metrics.get("eligible") is not False:
            raise ValueError("deadline override must retain the ineligible recall_current")
        if report.get("evidence_limitations") != list(policy.evidence_limitations):
            raise ValueError("Gate A deadline-override limitations drifted from policy")
    sample = report.get("sample")
    if not isinstance(sample, dict) or sample.get("frames") != policy.sample_frames:
        raise ValueError("Gate A report sample size mismatch")
    if sample.get("videos") != policy.expected_video_counts:
        raise ValueError("Gate A report per-video sample mismatch")
    return selected


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _parse_nonnegative_int(value: Any, *, field: str, row_number: int) -> int:
    rendered = str(value or "").strip()
    if not rendered or not rendered.isdigit():
        raise ValueError(f"{field} must be a non-negative integer at review row {row_number}")
    return int(rendered)


def evaluate_craft_gate_a(
    *,
    results_path: Path,
    review_csv_path: Path,
    policy: CraftGateAPolicy,
) -> dict[str, Any]:
    """Validate human labels and select the lightest threshold that meets recall gates."""

    config_by_id = {value.config_id: value for value in policy.configs}
    results = _load_jsonl(results_path)
    result_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    sample_uids: set[int] = set()
    sample_metadata: dict[int, tuple[str, str]] = {}
    for row in results:
        uid = row.get("keyframe_uid")
        config_id = str(row.get("config_id") or "")
        if isinstance(uid, bool) or not isinstance(uid, int):
            raise ValueError("CRAFT result requires integer keyframe_uid")
        if config_id not in config_by_id:
            raise ValueError(f"CRAFT result uses unknown config: {config_id}")
        video_id = str(row.get("video_id") or "")
        shot_id = str(row.get("shot_id") or "")
        if video_id not in policy.video_ids or not shot_id:
            raise ValueError(f"CRAFT result has invalid video/shot metadata: {(uid, config_id)}")
        metadata = (video_id, shot_id)
        if uid in sample_metadata and sample_metadata[uid] != metadata:
            raise ValueError(f"CRAFT result metadata changed across configs: {uid}")
        sample_metadata[uid] = metadata
        key = (uid, config_id)
        if key in result_by_key:
            raise ValueError(f"duplicate CRAFT result: {key}")
        expected = config_by_id[config_id]
        actual_thresholds = row.get("thresholds")
        if actual_thresholds != {
            "text_threshold": expected.text_threshold,
            "low_text": expected.low_text,
            "link_threshold": expected.link_threshold,
        }:
            raise ValueError(f"CRAFT result threshold mismatch: {key}")
        if row.get("status") != "success" or row.get("error") is not None:
            raise ValueError(f"CRAFT runtime result is not success: {key}")
        region_count = row.get("region_count")
        latency = row.get("latency_seconds")
        if (
            isinstance(region_count, bool)
            or not isinstance(region_count, int)
            or region_count < 0
        ):
            raise ValueError(f"invalid region_count: {key}")
        if (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(float(latency))
            or float(latency) < 0
        ):
            raise ValueError(f"invalid latency_seconds: {key}")
        result_by_key[key] = row
        sample_uids.add(uid)

    expected_result_count = policy.sample_frames * len(policy.configs)
    if len(results) != expected_result_count or len(sample_uids) != policy.sample_frames:
        raise ValueError("CRAFT result coverage is not exact for the configured policy")
    if policy.sample_uid_set_sha256 is not None:
        actual_uid_digest = hashlib.sha256(
            "".join(f"{value}\n" for value in sorted(sample_uids)).encode()
        ).hexdigest()
        if actual_uid_digest != policy.sample_uid_set_sha256:
            raise ValueError("CRAFT result UID-set checksum differs from policy")
    result_video_counts = Counter(video_id for video_id, _ in sample_metadata.values())
    expected_video_counts = policy.expected_video_counts
    actual_video_counts = {
        video_id: result_video_counts.get(video_id, 0) for video_id in policy.video_ids
    }
    if actual_video_counts != expected_video_counts:
        raise ValueError("CRAFT result sample does not match policy per-video counts")
    if len(set(sample_metadata.values())) != policy.sample_frames:
        raise ValueError("CRAFT result sample must use one frame per unique video/shot")
    for uid in sample_uids:
        if any((uid, config_id) not in result_by_key for config_id in config_by_id):
            raise ValueError(f"CRAFT result missing threshold for keyframe_uid {uid}")

    with review_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        review_rows = [dict(row) for row in csv.DictReader(handle)]
    if len(review_rows) != policy.sample_frames:
        raise ValueError("review CSV must contain exactly the pre-registered frame count")
    review_uids: set[int] = set()
    video_counts: Counter[str] = Counter()
    total_gt_regions = 0
    text_frame_count = 0
    missed_by_config: Counter[str] = Counter()
    text_frames_hit_by_config: Counter[str] = Counter()
    no_text_frames = 0
    no_text_fp_by_config: Counter[str] = Counter()

    for row_number, row in enumerate(review_rows, start=2):
        uid = _parse_nonnegative_int(
            row.get("keyframe_uid"), field="keyframe_uid", row_number=row_number
        )
        if uid in review_uids or uid not in sample_uids:
            raise ValueError(f"duplicate/foreign review keyframe_uid at row {row_number}")
        review_uids.add(uid)
        video_id = str(row.get("video_id") or "")
        if video_id not in policy.video_ids:
            raise ValueError(f"foreign video_id at review row {row_number}")
        if video_id != sample_metadata[uid][0]:
            raise ValueError(f"review video_id/result mismatch at row {row_number}")
        video_counts[video_id] += 1
        annotator = str(row.get("annotator") or "").strip()
        if not annotator:
            raise ValueError(f"annotator is required at review row {row_number}")
        has_text = str(row.get("gt_has_text") or "").strip().casefold()
        if has_text not in {"yes", "no"}:
            raise ValueError(f"gt_has_text must be yes/no at review row {row_number}")
        gt_regions = _parse_nonnegative_int(
            row.get("gt_region_count"), field="gt_region_count", row_number=row_number
        )
        if (has_text == "yes") != (gt_regions > 0):
            raise ValueError(f"gt_has_text/gt_region_count mismatch at row {row_number}")
        total_gt_regions += gt_regions
        text_frame_count += int(has_text == "yes")
        no_text_frames += int(has_text == "no")

        for config_id in config_by_id:
            missed = _parse_nonnegative_int(
                row.get(f"missed_gt_regions__{config_id}"),
                field=f"missed_gt_regions__{config_id}",
                row_number=row_number,
            )
            if missed > gt_regions:
                raise ValueError(f"missed GT regions exceed total at row {row_number}")
            missed_by_config[config_id] += missed
            if gt_regions > 0 and missed < gt_regions:
                text_frames_hit_by_config[config_id] += 1
            if gt_regions == 0 and result_by_key[(uid, config_id)]["region_count"] > 0:
                no_text_fp_by_config[config_id] += 1

    if review_uids != sample_uids:
        raise ValueError("review CSV UID set differs from CRAFT result UID set")
    actual_review_counts = {
        video_id: video_counts.get(video_id, 0) for video_id in policy.video_ids
    }
    if actual_review_counts != expected_video_counts:
        raise ValueError("review CSV does not match policy per-video counts")
    if total_gt_regions == 0 or text_frame_count == 0:
        raise ValueError("ground truth must contain real text regions and text-bearing frames")

    metrics: dict[str, dict[str, Any]] = {}
    for config_id in config_by_id:
        rows = [result_by_key[(uid, config_id)] for uid in sorted(sample_uids)]
        region_recall = 1 - missed_by_config[config_id] / total_gt_regions
        text_frame_recall = text_frames_hit_by_config[config_id] / text_frame_count
        detected_regions = sum(int(row["region_count"]) for row in rows)
        latency = sum(float(row["latency_seconds"]) for row in rows)
        eligible = (
            region_recall >= policy.min_region_recall
            and text_frame_recall >= policy.min_text_frame_recall
        )
        metrics[config_id] = {
            "region_recall": region_recall,
            "text_frame_recall": text_frame_recall,
            "missed_gt_regions": missed_by_config[config_id],
            "detected_regions": detected_regions,
            "regions_per_frame": detected_regions / policy.sample_frames,
            "no_text_false_positive_frames": no_text_fp_by_config[config_id],
            "no_text_false_positive_rate": (
                no_text_fp_by_config[config_id] / no_text_frames
                if no_text_frames
                else None
            ),
            "latency_seconds_total": latency,
            "frames_per_second": policy.sample_frames / latency if latency else None,
            "eligible": eligible,
        }

    eligible_ids = [config_id for config_id, value in metrics.items() if value["eligible"]]
    if eligible_ids:
        selected = min(
            eligible_ids,
            key=lambda config_id: (
                metrics[config_id]["regions_per_frame"],
                metrics[config_id]["no_text_false_positive_frames"],
                metrics[config_id]["latency_seconds_total"],
                config_id,
            ),
        )
        decision = "PASS_THRESHOLD_SELECTED"
        gate_b_allowed = True
    elif policy.allow_current_fallback_for_gate_b:
        selected = "recall_current"
        decision = "DEADLINE_OVERRIDE_KEEP_CURRENT"
        gate_b_allowed = True
    else:
        selected = "recall_current"
        decision = "FAIL_NO_THRESHOLD_MEETS_RECALL_KEEP_CURRENT"
        gate_b_allowed = False

    return {
        "schema_version": 1,
        "decision": decision,
        "gate_b_allowed": gate_b_allowed,
        "policy_sha256": policy_sha256(policy),
        "sample": {
            "frames": policy.sample_frames,
            "videos": policy.expected_video_counts,
            "text_frames": text_frame_count,
            "no_text_frames": no_text_frames,
            "ground_truth_regions": total_gt_regions,
            "annotators": sorted({str(row["annotator"]).strip() for row in review_rows}),
        },
        "requirements": {
            "min_region_recall": policy.min_region_recall,
            "min_text_frame_recall": policy.min_text_frame_recall,
        },
        "metrics": metrics,
        "selected_config_id": selected,
        "selected_thresholds": config_by_id[selected].model_dump(mode="json"),
        "evidence_limitations": list(policy.evidence_limitations),
    }
