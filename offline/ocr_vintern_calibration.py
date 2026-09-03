"""Ground-truth calibration and audited Vintern overrides for OCR Gate B.

The functions in this module never run Vintern.  They consume the results from
the normal Gate B inference pass, fit empirical confidence buckets on separate
human ground truth, and materialize an override only when that calibrated
confidence is strictly greater than the source EasyOCR region confidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from offline.ocr_vintern_gate2 import (
    VinternGate2Policy,
    route_vintern_region,
    vintern_output_rejection_reasons,
)


class VinternCalibrationPolicy(BaseModel):
    """Pre-registered bucket and support policy; do not tune after seeing scores."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1, 2, 3] = 1
    correctness_metric: Literal["unicode_nfc_casefold_whitespace_exact_match"] = (
        "unicode_nfc_casefold_whitespace_exact_match"
    )
    evidence_tier: Literal[
        "standard_300",
        "emergency_single_annotator_100",
        "emergency_single_annotator_98_of_100",
    ] = "standard_300"
    review_rows: int = Field(default=350, ge=1)
    review_rows_per_video: int = Field(default=70, ge=1)
    review_rows_per_stratum_per_video: int = Field(default=0, ge=0)
    review_selection_seed: str = Field(default="vintern-gate-b-standard-v1", min_length=1)
    min_ground_truth_frames: int = Field(default=300, ge=1)
    min_total_labeled_regions: int = Field(default=100, ge=1)
    min_bucket_samples: int = Field(default=20, ge=1)
    max_excluded_unreadable: int = Field(default=0, ge=0)
    allow_global_bucket_override: bool = True
    output_length_upper_bounds: tuple[int, ...] = (4, 12, 32, 96)
    guard_margin_upper_bounds: tuple[float, ...] = (0.25, 0.50, 0.75)
    mean_token_logprob_upper_bounds: tuple[float, ...] = (-2.0, -1.0, -0.5)

    @model_validator(mode="after")
    def _validate_bins(self) -> "VinternCalibrationPolicy":
        if self.schema_version == 2:
            if self.evidence_tier != "emergency_single_annotator_100":
                raise ValueError("schema v2 requires the emergency 100 evidence tier")
            if self.review_rows != 100 or self.review_rows_per_video != 20:
                raise ValueError("schema v2 requires exactly 100 review rows, 20/video")
            if self.review_rows_per_stratum_per_video != 4:
                raise ValueError("schema v2 requires a target of 4 rows/stratum/video")
            if self.min_ground_truth_frames != 100:
                raise ValueError("schema v2 requires exactly 100 ground-truth frames")
            if self.min_total_labeled_regions != 100:
                raise ValueError("schema v2 requires exactly 100 labeled regions")
            if self.allow_global_bucket_override:
                raise ValueError("schema v2 forbids global-bucket override")
            if self.max_excluded_unreadable != 0:
                raise ValueError("schema v2 does not allow unreadable exclusions")
        if self.schema_version == 3:
            if self.evidence_tier != "emergency_single_annotator_98_of_100":
                raise ValueError("schema v3 requires the emergency 98/100 evidence tier")
            if self.review_rows != 100 or self.review_rows_per_video != 20:
                raise ValueError("schema v3 requires the pinned 100-row review pool")
            if self.review_rows_per_stratum_per_video != 4:
                raise ValueError("schema v3 requires the pinned 4 rows/stratum/video")
            if self.min_ground_truth_frames != 98:
                raise ValueError("schema v3 requires exactly 98 usable ground-truth frames")
            if self.min_total_labeled_regions != 98:
                raise ValueError("schema v3 requires exactly 98 usable labeled regions")
            if self.max_excluded_unreadable != 2:
                raise ValueError("schema v3 permits exactly two unreadable exclusions")
            if self.allow_global_bucket_override:
                raise ValueError("schema v3 forbids global-bucket override")
        for name, values in (
            ("output_length_upper_bounds", self.output_length_upper_bounds),
            ("guard_margin_upper_bounds", self.guard_margin_upper_bounds),
            ("mean_token_logprob_upper_bounds", self.mean_token_logprob_upper_bounds),
        ):
            if not values or any(left >= right for left, right in zip(values, values[1:])):
                raise ValueError(f"{name} must be non-empty and strictly increasing")
        if self.output_length_upper_bounds[0] < 1:
            raise ValueError("output length bounds must be positive")
        return self


class VinternInternalSignals(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    output_length: int = Field(ge=0)
    guard_length_limit: int = Field(ge=1)
    guard_margin_ratio: float
    mean_token_logprob: float | None = None


class VinternCalibrationExample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(min_length=1)
    easyocr_text: str
    easyocr_confidence: float = Field(ge=0, le=1)
    vintern_text: str
    ground_truth_text: str
    signals: VinternInternalSignals
    guard_rejection_reasons: tuple[str, ...] = ()
    correct: bool


class VinternBucketStats(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    bucket_id: str
    support: int = Field(ge=1)
    correct: int = Field(ge=0)
    empirical_accuracy: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _validate_fraction(self) -> "VinternBucketStats":
        if self.correct > self.support:
            raise ValueError("correct cannot exceed support")
        if not math.isclose(
            self.empirical_accuracy,
            self.correct / self.support,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("empirical_accuracy must equal correct/support")
        return self


class VinternCalibrationTable(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    correctness_metric: Literal["unicode_nfc_casefold_whitespace_exact_match"]
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ground_truth_frames: int = Field(ge=1)
    labeled_regions: int = Field(ge=1)
    buckets: dict[str, VinternBucketStats]


class VinternCalibratedDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    bucket_id: str
    bucket_support: int = Field(ge=1)
    bucket_correct: int = Field(ge=0)
    calibrated_confidence: float = Field(ge=0, le=1)


def normalize_ocr_text(value: str) -> str:
    """Normalization used by the pre-registered exact-match correctness metric."""

    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def _visible_length(value: str) -> int:
    return len("".join(value.split()))


def _guard_length_limit(easyocr_text: str) -> int:
    return max(96, _visible_length(easyocr_text) * 8 + 48)


def derive_vintern_signals(
    *,
    easyocr_text: str,
    vintern_text: str,
    result: dict[str, Any],
) -> VinternInternalSignals:
    """Derive signals from one existing inference result without another model call."""

    output_length = _visible_length(vintern_text)
    limit = _guard_length_limit(easyocr_text)
    raw_logprob = result.get("mean_token_logprob")
    mean_logprob: float | None
    if raw_logprob is None:
        mean_logprob = None
    elif isinstance(raw_logprob, bool) or not isinstance(raw_logprob, (int, float)):
        raise ValueError("mean_token_logprob must be numeric or null")
    else:
        mean_logprob = float(raw_logprob)
        if not math.isfinite(mean_logprob):
            raise ValueError("mean_token_logprob must be finite")
    return VinternInternalSignals(
        output_length=output_length,
        guard_length_limit=limit,
        guard_margin_ratio=(limit - output_length) / limit,
        mean_token_logprob=mean_logprob,
    )


def _interval_label(value: float, upper_bounds: Sequence[float], prefix: str) -> str:
    lower: float | None = None
    for upper in upper_bounds:
        if value <= upper:
            left = "neg_inf" if lower is None else f"{lower:g}"
            return f"{prefix}_{left}_to_{upper:g}"
        lower = upper
    return f"{prefix}_gt_{upper_bounds[-1]:g}"


def _signal_bucket_parts(
    signals: VinternInternalSignals,
    policy: VinternCalibrationPolicy,
) -> tuple[str, str, str]:
    length = _interval_label(
        float(signals.output_length), policy.output_length_upper_bounds, "len"
    )
    margin = _interval_label(
        signals.guard_margin_ratio, policy.guard_margin_upper_bounds, "margin"
    )
    logprob = (
        "logprob_unavailable"
        if signals.mean_token_logprob is None
        else _interval_label(
            signals.mean_token_logprob,
            policy.mean_token_logprob_upper_bounds,
            "logprob",
        )
    )
    return length, margin, logprob


def vintern_bucket_candidates(
    signals: VinternInternalSignals,
    policy: VinternCalibrationPolicy,
) -> tuple[str, str, str]:
    """Return fine → coarse → global buckets for deterministic backoff."""

    length, margin, logprob = _signal_bucket_parts(signals, policy)
    return (
        f"fine:{length}|{margin}|{logprob}",
        f"structural:{length}|{margin}",
        "global",
    )


def make_calibration_example(
    *,
    candidate_id: str,
    easyocr_text: str,
    easyocr_confidence: float,
    result: dict[str, Any],
    ground_truth_text: str,
) -> VinternCalibrationExample:
    vintern_text = str(result.get("vintern_text") or "")
    signals = derive_vintern_signals(
        easyocr_text=easyocr_text,
        vintern_text=vintern_text,
        result=result,
    )
    rejection = ("runtime_error",)
    if result.get("status") == "success":
        rejection = vintern_output_rejection_reasons(
            easyocr_text=easyocr_text,
            vintern_text=vintern_text,
        )
    return VinternCalibrationExample(
        candidate_id=candidate_id,
        easyocr_text=easyocr_text,
        easyocr_confidence=easyocr_confidence,
        vintern_text=vintern_text,
        ground_truth_text=ground_truth_text,
        signals=signals,
        guard_rejection_reasons=rejection,
        correct=normalize_ocr_text(vintern_text) == normalize_ocr_text(ground_truth_text),
    )


def _policy_sha256(policy: VinternCalibrationPolicy) -> str:
    payload = json.dumps(
        policy.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def fit_vintern_calibration(
    examples: Sequence[VinternCalibrationExample],
    *,
    policy: VinternCalibrationPolicy,
    ground_truth_frame_count: int,
) -> VinternCalibrationTable:
    """Fit empirical bucket accuracies from human-ground-truth examples."""

    if ground_truth_frame_count < policy.min_ground_truth_frames:
        raise ValueError(
            "insufficient ground-truth frames: "
            f"{ground_truth_frame_count} < {policy.min_ground_truth_frames}"
        )
    if len(examples) < policy.min_total_labeled_regions:
        raise ValueError(
            "insufficient labeled Vintern regions: "
            f"{len(examples)} < {policy.min_total_labeled_regions}"
        )
    ids = [example.candidate_id for example in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate calibration candidate_id")

    counts: dict[str, Counter[str]] = {}
    for example in examples:
        for bucket_id in vintern_bucket_candidates(example.signals, policy):
            bucket = counts.setdefault(bucket_id, Counter())
            bucket["support"] += 1
            bucket["correct"] += int(example.correct)
    buckets = {
        bucket_id: VinternBucketStats(
            bucket_id=bucket_id,
            support=value["support"],
            correct=value["correct"],
            empirical_accuracy=value["correct"] / value["support"],
        )
        for bucket_id, value in sorted(counts.items())
    }
    return VinternCalibrationTable(
        correctness_metric=policy.correctness_metric,
        policy_sha256=_policy_sha256(policy),
        ground_truth_frames=ground_truth_frame_count,
        labeled_regions=len(examples),
        buckets=buckets,
    )


def calibrated_vintern_confidence(
    signals: VinternInternalSignals,
    *,
    table: VinternCalibrationTable,
    policy: VinternCalibrationPolicy,
) -> VinternCalibratedDecision | None:
    if table.policy_sha256 != _policy_sha256(policy):
        raise ValueError("calibration table/policy SHA mismatch")
    for bucket_id in vintern_bucket_candidates(signals, policy):
        stats = table.buckets.get(bucket_id)
        if stats is None:
            continue
        if bucket_id == "global" and not policy.allow_global_bucket_override:
            continue
        if bucket_id != "global" and stats.support < policy.min_bucket_samples:
            continue
        return VinternCalibratedDecision(
            bucket_id=bucket_id,
            bucket_support=stats.support,
            bucket_correct=stats.correct,
            calibrated_confidence=stats.empirical_accuracy,
        )
    return None


def materialize_calibrated_gate2_frames(
    easyocr_frames: Sequence[dict[str, Any]],
    vintern_results: Sequence[dict[str, Any]],
    *,
    table: VinternCalibrationTable,
    calibration_policy: VinternCalibrationPolicy,
    gate_policy: VinternGate2Policy,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply calibrated overrides and return frame rows plus region audit rows."""

    result_by_id: dict[str, dict[str, Any]] = {}
    for result in vintern_results:
        candidate_id = str(result.get("candidate_id") or "")
        if not candidate_id:
            raise ValueError("Vintern result missing candidate_id")
        if candidate_id in result_by_id:
            raise ValueError(f"duplicate Vintern candidate_id: {candidate_id}")
        result_by_id[candidate_id] = result

    all_region_ids: set[str] = set()
    seen_frame_uids: set[int] = set()
    materialized: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for frame in easyocr_frames:
        raw_frame_uid = frame.get("keyframe_uid")
        if isinstance(raw_frame_uid, bool) or not isinstance(raw_frame_uid, int):
            raise ValueError("EasyOCR frame requires integer keyframe_uid")
        if raw_frame_uid in seen_frame_uids:
            raise ValueError(f"duplicate EasyOCR keyframe_uid: {raw_frame_uid}")
        seen_frame_uids.add(raw_frame_uid)
        output_frame = dict(frame)
        output_regions: list[dict[str, Any]] = []
        for region in frame.get("regions") or []:
            output_region = dict(region)
            candidate_id = str(region.get("region_id") or "")
            if not candidate_id:
                raise ValueError("EasyOCR region missing region_id")
            if candidate_id in all_region_ids:
                raise ValueError(f"duplicate EasyOCR region_id: {candidate_id}")
            all_region_ids.add(candidate_id)

            easy_text = str(region.get("easyocr_text") or "").strip()
            raw_easy_confidence = region.get("easyocr_confidence")
            if raw_easy_confidence is None:
                easy_confidence = 0.0
            elif isinstance(raw_easy_confidence, bool) or not isinstance(
                raw_easy_confidence, (int, float)
            ):
                raise ValueError("easyocr_confidence must be numeric or null")
            else:
                easy_confidence = float(raw_easy_confidence)
            if not 0 <= easy_confidence <= 1:
                raise ValueError("easyocr_confidence must be in [0, 1]")

            decision = route_vintern_region(region, policy=gate_policy)
            result = result_by_id.get(candidate_id) if decision.candidate else None
            final_text = easy_text
            final_confidence = easy_confidence
            final_engine = "easyocr"
            overridden = False
            override_reason = "not_routed_to_vintern"
            gemini_residual = False
            gemini_residual_reasons: list[str] = []
            audit_row: dict[str, Any] | None = None
            if decision.candidate:
                gemini_residual = True
                gemini_residual_reasons = ["missing_vintern_result"]
                audit_row = {
                    "schema_version": 1,
                    "candidate_id": candidate_id,
                    "video_id": frame.get("video_id"),
                    "keyframe_uid": frame.get("keyframe_uid"),
                    "router_v2_reasons": list(decision.reasons),
                    "easyocr_text_old": easy_text,
                    "easyocr_confidence_old": easy_confidence,
                    "vintern_text": "",
                    "guard_rejection_reasons": [],
                    "signals": None,
                    "calibration_bucket_id": None,
                    "calibration_bucket_support": None,
                    "calibration_bucket_correct": None,
                    "vintern_confidence_calibrated": None,
                    "overwritten": False,
                    "decision_reason": "missing_vintern_result",
                    "gemini_residual": True,
                    "gemini_residual_reasons": ["missing_vintern_result"],
                    "final_text": final_text,
                    "final_confidence": final_confidence,
                    "final_engine": final_engine,
                }
                if result is not None:
                    vintern_text = str(result.get("vintern_text") or "").strip()
                    signals = derive_vintern_signals(
                        easyocr_text=easy_text,
                        vintern_text=vintern_text,
                        result=result,
                    )
                    rejection = ("runtime_error",)
                    if result.get("status") == "success":
                        rejection = vintern_output_rejection_reasons(
                            easyocr_text=easy_text,
                            vintern_text=vintern_text,
                        )
                    calibrated = calibrated_vintern_confidence(
                        signals, table=table, policy=calibration_policy
                    )
                    if rejection:
                        override_reason = "vintern_output_guard_rejected"
                        gemini_residual_reasons = ["vintern_output_guard_rejected"]
                    elif calibrated is None:
                        override_reason = "insufficient_calibration_bucket_support"
                        gemini_residual_reasons = [
                            "insufficient_calibration_bucket_support"
                        ]
                    elif calibrated.calibrated_confidence > easy_confidence:
                        final_text = vintern_text
                        final_confidence = calibrated.calibrated_confidence
                        final_engine = "vintern"
                        overridden = True
                        override_reason = "calibrated_confidence_strictly_greater"
                        gemini_residual = False
                        gemini_residual_reasons = []
                    else:
                        override_reason = "calibrated_confidence_not_greater"
                        gemini_residual_reasons = [
                            "calibrated_confidence_not_greater"
                        ]
                    audit_row.update(
                        {
                            "vintern_text": vintern_text,
                            "guard_rejection_reasons": list(rejection),
                            "signals": signals.model_dump(mode="json"),
                            "calibration_bucket_id": (
                                None if calibrated is None else calibrated.bucket_id
                            ),
                            "calibration_bucket_support": (
                                None if calibrated is None else calibrated.bucket_support
                            ),
                            "calibration_bucket_correct": (
                                None if calibrated is None else calibrated.bucket_correct
                            ),
                            "vintern_confidence_calibrated": (
                                None
                                if calibrated is None
                                else calibrated.calibrated_confidence
                            ),
                            "overwritten": overridden,
                            "decision_reason": override_reason,
                            "gemini_residual": gemini_residual,
                            "gemini_residual_reasons": gemini_residual_reasons,
                            "final_text": final_text,
                            "final_confidence": final_confidence,
                            "final_engine": final_engine,
                        }
                    )
                audit.append(audit_row)

            output_region.update(
                {
                    "final_text": final_text,
                    "final_confidence": final_confidence,
                    "final_engine": final_engine,
                    "vintern_override": overridden,
                    "vintern_candidate": decision.candidate,
                    "vintern_result_status": (
                        None if result is None else str(result.get("status") or "")
                    ),
                    "vintern_guard_rejection_reasons": (
                        []
                        if audit_row is None
                        else list(audit_row["guard_rejection_reasons"])
                    ),
                    "gemini_residual": gemini_residual,
                    "gemini_residual_reasons": gemini_residual_reasons,
                }
            )
            if audit_row is not None:
                output_region["calibration_bucket_id"] = audit_row[
                    "calibration_bucket_id"
                ]
            output_regions.append(output_region)
        output_frame["regions"] = output_regions
        output_frame["materialized_text_policy"] = "EasyOCR+Vintern calibrated"
        materialized.append(output_frame)

    foreign = set(result_by_id) - all_region_ids
    if foreign:
        raise ValueError(f"foreign Vintern result IDs: {len(foreign)}")
    return materialized, audit


def validate_materialized_calibration_audit(
    materialized_frames: Sequence[dict[str, Any]],
    audit_rows: Sequence[dict[str, Any]],
    *,
    table: VinternCalibrationTable,
) -> None:
    """Fail closed when the calibrated frame artifact and override audit diverge."""

    audit_by_id: dict[str, dict[str, Any]] = {}
    for row in audit_rows:
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id or candidate_id in audit_by_id:
            raise ValueError("missing or duplicate audit candidate_id")
        audit_by_id[candidate_id] = row

    candidate_regions: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for frame in materialized_frames:
        for region in frame.get("regions") or []:
            if not bool(region.get("vintern_candidate")):
                continue
            candidate_id = str(region.get("region_id") or "")
            if not candidate_id or candidate_id in candidate_regions:
                raise ValueError("missing or duplicate materialized candidate region_id")
            candidate_regions[candidate_id] = (frame, region)
    if set(candidate_regions) != set(audit_by_id):
        raise ValueError("materialized candidate IDs and calibration audit IDs differ")

    for candidate_id, (frame, region) in candidate_regions.items():
        audit = audit_by_id[candidate_id]
        if str(audit.get("video_id")) != str(frame.get("video_id")):
            raise ValueError(f"audit video_id mismatch: {candidate_id}")
        if int(audit.get("keyframe_uid")) != int(frame.get("keyframe_uid")):
            raise ValueError(f"audit keyframe_uid mismatch: {candidate_id}")
        comparisons = (
            ("final_text", str(region.get("final_text") or "")),
            ("final_engine", str(region.get("final_engine") or "")),
        )
        for field, expected in comparisons:
            if str(audit.get(field) or "") != expected:
                raise ValueError(f"audit {field} mismatch: {candidate_id}")
        if float(audit.get("final_confidence")) != float(region.get("final_confidence")):
            raise ValueError(f"audit final_confidence mismatch: {candidate_id}")
        overridden = bool(audit.get("overwritten"))
        if overridden != bool(region.get("vintern_override")):
            raise ValueError(f"audit override mismatch: {candidate_id}")
        if bool(audit.get("gemini_residual")) != bool(
            region.get("gemini_residual")
        ):
            raise ValueError(f"audit Gemini residual mismatch: {candidate_id}")
        if list(audit.get("gemini_residual_reasons") or []) != list(
            region.get("gemini_residual_reasons") or []
        ):
            raise ValueError(f"audit Gemini residual reasons mismatch: {candidate_id}")
        if overridden:
            if region.get("final_engine") != "vintern":
                raise ValueError(f"overridden region must use Vintern: {candidate_id}")
            if audit.get("guard_rejection_reasons"):
                raise ValueError(f"guard-rejected region was overridden: {candidate_id}")
            if not (
                float(audit["vintern_confidence_calibrated"])
                > float(audit["easyocr_confidence_old"])
            ):
                raise ValueError(f"override confidence rule violated: {candidate_id}")
            if bool(audit.get("gemini_residual")):
                raise ValueError(f"overridden region cannot be residual: {candidate_id}")
        elif region.get("final_engine") != "easyocr":
            raise ValueError(f"non-overridden region must retain EasyOCR: {candidate_id}")
        elif not bool(audit.get("gemini_residual")):
            raise ValueError(f"non-overridden candidate must be residual: {candidate_id}")

        bucket_id = audit.get("calibration_bucket_id")
        if bucket_id is None:
            if audit.get("vintern_confidence_calibrated") is not None:
                raise ValueError(f"confidence exists without calibration bucket: {candidate_id}")
            continue
        stats = table.buckets.get(str(bucket_id))
        if stats is None:
            raise ValueError(f"audit references unknown calibration bucket: {candidate_id}")
        if int(audit.get("calibration_bucket_support")) != stats.support:
            raise ValueError(f"audit bucket support mismatch: {candidate_id}")
        if int(audit.get("calibration_bucket_correct")) != stats.correct:
            raise ValueError(f"audit bucket correct mismatch: {candidate_id}")
        if not math.isclose(
            float(audit.get("vintern_confidence_calibrated")),
            stats.empirical_accuracy,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"audit calibrated confidence mismatch: {candidate_id}")


def validate_materialized_calibration_files(
    materialized_path: str,
    audit_path: str,
    calibration_path: str,
) -> None:
    """Path-level validation used by the snapshot builder before materialization."""

    def read_jsonl(path: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
                rows.append(value)
        return rows

    with open(calibration_path, "r", encoding="utf-8") as handle:
        table = VinternCalibrationTable.model_validate_json(handle.read())
    validate_materialized_calibration_audit(
        read_jsonl(materialized_path), read_jsonl(audit_path), table=table
    )


def calibration_examples_from_rows(
    easyocr_frames: Sequence[dict[str, Any]],
    vintern_results: Sequence[dict[str, Any]],
    ground_truth_rows: Iterable[dict[str, Any]],
    *,
    gate_policy: VinternGate2Policy,
) -> list[VinternCalibrationExample]:
    """Join human labels to the exact Gate B result rows used for calibration."""

    regions: dict[str, tuple[dict[str, Any], int]] = {}
    frame_uids: set[int] = set()
    for frame in easyocr_frames:
        raw_frame_uid = frame.get("keyframe_uid")
        if isinstance(raw_frame_uid, bool) or not isinstance(raw_frame_uid, int):
            raise ValueError("EasyOCR frame requires integer keyframe_uid")
        if raw_frame_uid in frame_uids:
            raise ValueError(f"duplicate EasyOCR keyframe_uid: {raw_frame_uid}")
        frame_uids.add(raw_frame_uid)
        for region in frame.get("regions") or []:
            candidate_id = str(region.get("region_id") or "")
            if not candidate_id or candidate_id in regions:
                raise ValueError("missing or duplicate EasyOCR region_id")
            if route_vintern_region(region, policy=gate_policy).candidate:
                regions[candidate_id] = (region, raw_frame_uid)
    results: dict[str, dict[str, Any]] = {}
    for row in vintern_results:
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id or candidate_id in results:
            raise ValueError("missing or duplicate Vintern candidate_id")
        results[candidate_id] = row

    examples: list[VinternCalibrationExample] = []
    seen: set[str] = set()
    for label in ground_truth_rows:
        candidate_id = str(label.get("candidate_id") or label.get("region_id") or "")
        if not candidate_id:
            # A Gate A frame-level/no-text control still proves sample coverage but
            # does not calibrate Vintern, which was never invoked for that row.
            continue
        label_status = str(label.get("label_status") or "labeled").strip().casefold()
        if label_status == "exclude_unreadable":
            continue
        if label_status != "labeled":
            raise ValueError(f"invalid label_status for {candidate_id}")
        if candidate_id in seen:
            raise ValueError(f"duplicate ground-truth candidate_id: {candidate_id}")
        seen.add(candidate_id)
        human_text = label.get("human_text")
        if not isinstance(human_text, str):
            raise ValueError(f"ground truth text is not a string for {candidate_id}")
        ground_truth_is_empty = str(
            label.get("ground_truth_is_empty") or "no"
        ).strip().casefold()
        if ground_truth_is_empty not in {"yes", "no"}:
            raise ValueError(f"ground_truth_is_empty must be yes/no for {candidate_id}")
        if ground_truth_is_empty == "yes" and human_text.strip():
            raise ValueError(f"empty ground truth cannot carry text: {candidate_id}")
        if ground_truth_is_empty == "no" and not human_text.strip():
            raise ValueError(f"ground truth text is blank for {candidate_id}")
        region_with_uid = regions.get(candidate_id)
        result = results.get(candidate_id)
        if region_with_uid is None or result is None:
            raise ValueError(f"ground truth candidate lacks routed Gate B result: {candidate_id}")
        region, expected_uid = region_with_uid
        raw_label_uid = label.get("keyframe_uid")
        if str(raw_label_uid or "").strip() == "":
            raise ValueError(f"ground truth candidate lacks keyframe_uid: {candidate_id}")
        if int(raw_label_uid) != expected_uid:
            raise ValueError(f"ground truth keyframe_uid mismatch: {candidate_id}")
        raw_confidence = region.get("easyocr_confidence")
        easy_confidence = 0.0 if raw_confidence is None else float(raw_confidence)
        examples.append(
            make_calibration_example(
                candidate_id=candidate_id,
                easyocr_text=str(region.get("easyocr_text") or ""),
                easyocr_confidence=easy_confidence,
                result=result,
                ground_truth_text=human_text.strip(),
            )
        )
    return examples
