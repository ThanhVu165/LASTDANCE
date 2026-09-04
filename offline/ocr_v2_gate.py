"""Fail-closed evaluation for the OCR-v2 detector and recognizer gates.

The module is deliberately model-free.  It consumes immutable sample mappings,
human labels, recognizer output, and measured runtime evidence; it never changes
production OCR artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


EASYOCR_MODEL_ID = "easyocr_latin_g2_cached"
PADDLE_MODEL_ID = "latin_PP-OCRv5_mobile_rec"
VIETOCR_MODEL_ID = "vietocr_vgg_seq2seq"
MODEL_IDS = (EASYOCR_MODEL_ID, PADDLE_MODEL_ID, VIETOCR_MODEL_ID)


class OcrV2GatePolicy(BaseModel):
    """Versioned thresholds agreed before looking at challenger results."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    detector_sample_frames: int = Field(ge=20)
    detector_video_ids: tuple[str, ...]
    recognizer_sample_regions: int = Field(ge=20)
    recognizer_min_usable_regions: int = Field(ge=1)
    numeric_name_min_regions: int = Field(ge=1)
    detector_failure_trigger: float = Field(ge=0.0, le=1.0)
    token_recall_min_absolute_gain: float = Field(ge=0.0, le=1.0)
    cer_min_relative_reduction: float = Field(ge=0.0, le=1.0)
    numeric_name_max_absolute_regression: float = Field(ge=0.0, le=1.0)
    token_recall_tie_margin: float = Field(ge=0.0, le=1.0)
    throughput_canary_regions: int = Field(ge=1)
    catalog_archive_count: int = Field(ge=1)
    production_workers: int = Field(ge=1)
    production_max_hours: float = Field(gt=0.0)
    model_ids: tuple[str, str, str]

    @field_validator("model_ids")
    @classmethod
    def model_ids_are_locked(cls, value: tuple[str, str, str]) -> tuple[str, str, str]:
        if value != MODEL_IDS:
            raise ValueError(f"model_ids must be exactly {MODEL_IDS!r}")
        return value

    @model_validator(mode="after")
    def usable_count_fits_sample(self) -> "OcrV2GatePolicy":
        if (
            not self.detector_video_ids
            or len(self.detector_video_ids) != len(set(self.detector_video_ids))
            or self.detector_sample_frames % len(self.detector_video_ids) != 0
            or self.recognizer_sample_regions % len(self.detector_video_ids) != 0
        ):
            raise ValueError("Gate A/B samples must divide evenly across unique detector_video_ids")
        if self.recognizer_min_usable_regions > self.recognizer_sample_regions:
            raise ValueError("recognizer_min_usable_regions exceeds sample size")
        if self.numeric_name_min_regions > self.recognizer_min_usable_regions:
            raise ValueError("numeric_name_min_regions exceeds minimum usable sample")
        return self


def load_policy(path: Path) -> OcrV2GatePolicy:
    return OcrV2GatePolicy.model_validate_json(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(value: object) -> str:
    return " ".join(unicodedata.normalize("NFC", str(value or "")).casefold().split())


def token_list(value: object) -> list[str]:
    return re.findall(r"\w+", normalize_text(value), flags=re.UNICODE)


def levenshtein(left: str, right: str) -> int:
    """Return Unicode-codepoint Levenshtein distance using one working row."""

    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _require_unique(rows: list[dict[str, Any]], key: str, *, identity: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(rows, start=2):
        value = str(row.get(key, "")).strip()
        if not value:
            raise ValueError(f"blank {key} in {identity} row {line_number}")
        if value in indexed:
            raise ValueError(f"duplicate {key}={value!r} in {identity}")
        indexed[value] = row
    return indexed


def evaluate_gate_a(review_csv: Path, policy: OcrV2GatePolicy) -> dict[str, Any]:
    rows = read_csv(review_csv)
    if len(rows) != policy.detector_sample_frames:
        raise ValueError(
            f"Gate A requires exactly {policy.detector_sample_frames} rows; got {len(rows)}"
        )
    _require_unique(rows, "keyframe_uid", identity="Gate A labels")
    expected_per_video = policy.detector_sample_frames // len(policy.detector_video_ids)
    video_counts = Counter(str(row.get("video_id") or "") for row in rows)
    expected_video_counts = {video_id: expected_per_video for video_id in policy.detector_video_ids}
    if dict(video_counts) != expected_video_counts:
        raise ValueError(
            f"Gate A must be balanced across the locked videos: expected={expected_video_counts}, "
            f"actual={dict(video_counts)}"
        )
    allowed_bbox = {"correct", "miss", "duplicate", "wrong", "not_applicable"}
    text_rows: list[dict[str, str]] = []
    no_text_rows = 0
    for row_number, row in enumerate(rows, start=2):
        has_text = row.get("gt_has_text", "").strip().casefold()
        bbox = row.get("bbox_quality", "").strip().casefold()
        if has_text not in {"yes", "no"}:
            raise ValueError(f"invalid gt_has_text in CSV row {row_number}")
        if bbox not in allowed_bbox:
            raise ValueError(f"invalid bbox_quality in CSV row {row_number}")
        if has_text == "yes":
            if bbox == "not_applicable":
                raise ValueError(f"text-bearing row {row_number} cannot be not_applicable")
            text_rows.append(row)
        else:
            no_text_rows += 1
    if not text_rows:
        raise ValueError("Gate A has no labelled text-bearing frame")
    issue_counts = Counter(row["bbox_quality"].strip().casefold() for row in text_rows)
    issue_total = sum(issue_counts[value] for value in ("miss", "duplicate", "wrong"))
    issue_rate = issue_total / len(text_rows)
    decision = (
        "RUN_DBNET_CHALLENGER"
        if issue_rate >= policy.detector_failure_trigger
        else "KEEP_CRAFT"
    )
    return {
        "schema_version": 1,
        "gate": "A_detector_triage",
        "decision": decision,
        "sample_frames": len(rows),
        "video_counts": expected_video_counts,
        "text_bearing_frames": len(text_rows),
        "no_text_frames": no_text_rows,
        "bbox_issue_frames": issue_total,
        "bbox_issue_rate": issue_rate,
        "trigger": policy.detector_failure_trigger,
        "bbox_counts_text_bearing": dict(sorted(issue_counts.items())),
        "important": (
            "RUN_DBNET_CHALLENGER only authorizes a small detector A/B; "
            "it does not select DBNet for production."
        ),
        "inputs": {"review_csv": str(review_csv), "review_csv_sha256": sha256_file(review_csv)},
    }


def _validate_sample_and_labels(
    sample_jsonl: Path,
    ground_truth_csv: Path,
    policy: OcrV2GatePolicy,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]], list[str]]:
    sample_rows = read_jsonl(sample_jsonl)
    if len(sample_rows) != policy.recognizer_sample_regions:
        raise ValueError(
            f"Gate B requires exactly {policy.recognizer_sample_regions} sample rows; "
            f"got {len(sample_rows)}"
        )
    samples = _require_unique(sample_rows, "region_id", identity="recognition sample")
    expected_per_video = policy.recognizer_sample_regions // len(policy.detector_video_ids)
    sample_video_counts = Counter(str(row.get("video_id") or "") for row in sample_rows)
    expected_video_counts = {video_id: expected_per_video for video_id in policy.detector_video_ids}
    if dict(sample_video_counts) != expected_video_counts:
        raise ValueError(
            f"Gate B sample must be balanced across locked videos: expected={expected_video_counts}, "
            f"actual={dict(sample_video_counts)}"
        )
    label_rows = read_csv(ground_truth_csv)
    labels = _require_unique(label_rows, "region_id", identity="recognition ground truth")
    if set(samples) != set(labels):
        missing = sorted(set(samples) - set(labels))
        extra = sorted(set(labels) - set(samples))
        raise ValueError(f"ground-truth region IDs differ: missing={missing[:3]}, extra={extra[:3]}")
    usable: list[str] = []
    for region_id, sample in samples.items():
        label = labels[region_id]
        if label.get("sample_row_sha256", "").strip() != str(sample.get("sample_row_sha256", "")):
            raise ValueError(f"immutable sample hash changed for region_id={region_id}")
        status = label.get("label_status", "").strip().casefold()
        if status not in {"labeled", "exclude_unreadable", "false_positive"}:
            raise ValueError(f"invalid label_status for region_id={region_id}")
        human_text = label.get("human_text", "")
        if status == "labeled":
            if not normalize_text(human_text):
                raise ValueError(f"blank human_text for labelled region_id={region_id}")
            usable.append(region_id)
        elif normalize_text(human_text):
            raise ValueError(f"excluded region_id={region_id} must have blank human_text")
        text_type = label.get("text_type", "").strip().casefold()
        if status == "labeled" and text_type not in {"ordinary", "ticker", "numeric_or_name", "other"}:
            raise ValueError(f"invalid text_type for region_id={region_id}")
    if len(usable) < policy.recognizer_min_usable_regions:
        raise ValueError(
            f"only {len(usable)} usable regions; need {policy.recognizer_min_usable_regions}"
        )
    numeric_name = [
        region_id
        for region_id in usable
        if labels[region_id].get("text_type", "").strip().casefold() == "numeric_or_name"
        or any(character.isdigit() for character in labels[region_id].get("human_text", ""))
    ]
    if len(numeric_name) < policy.numeric_name_min_regions:
        raise ValueError(
            f"only {len(numeric_name)} numeric/name regions; need {policy.numeric_name_min_regions}"
        )
    return samples, labels, usable


def _validate_results(
    results_jsonl: Path,
    samples: dict[str, dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    rows = read_jsonl(results_jsonl)
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=1):
        region_id = str(row.get("region_id", "")).strip()
        model_id = str(row.get("model_id", "")).strip()
        key = (region_id, model_id)
        if region_id not in samples:
            raise ValueError(f"foreign region_id in result row {row_number}: {region_id!r}")
        if model_id not in MODEL_IDS:
            raise ValueError(f"foreign model_id in result row {row_number}: {model_id!r}")
        if key in indexed:
            raise ValueError(f"duplicate recognizer result: {key!r}")
        if str(row.get("sample_row_sha256", "")) != str(samples[region_id].get("sample_row_sha256", "")):
            raise ValueError(f"sample hash mismatch in recognizer result: {key!r}")
        status = str(row.get("status", "")).strip().casefold()
        if status not in {"success", "error"}:
            raise ValueError(f"invalid recognizer status: {key!r}")
        indexed[key] = row
    expected = {(region_id, model_id) for region_id in samples for model_id in MODEL_IDS}
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        extra = sorted(set(indexed) - expected)
        raise ValueError(f"recognizer result coverage differs: missing={missing[:3]}, extra={extra[:3]}")
    return indexed


def _metrics(
    model_id: str,
    usable: list[str],
    labels: dict[str, dict[str, str]],
    results: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    exact = 0
    edit_total = 0
    character_total = 0
    matched_tokens = 0
    ground_truth_tokens = 0
    errors = 0
    numeric_exact = 0
    numeric_total = 0
    for region_id in usable:
        gt = normalize_text(labels[region_id]["human_text"])
        result = results[(region_id, model_id)]
        if str(result.get("status", "")).casefold() == "error":
            errors += 1
            prediction = ""
        else:
            prediction = normalize_text(result.get("text", ""))
        exact += prediction == gt
        edit_total += levenshtein(prediction, gt)
        character_total += len(gt)
        gt_counter = Counter(token_list(gt))
        prediction_counter = Counter(token_list(prediction))
        matched_tokens += sum((gt_counter & prediction_counter).values())
        ground_truth_tokens += sum(gt_counter.values())
        is_numeric_name = (
            labels[region_id].get("text_type", "").strip().casefold() == "numeric_or_name"
            or any(character.isdigit() for character in gt)
        )
        if is_numeric_name:
            numeric_total += 1
            numeric_exact += prediction == gt
    return {
        "model_id": model_id,
        "usable_regions": len(usable),
        "error_count": errors,
        "normalized_exact_line_accuracy": exact / len(usable),
        "exact_token_recall": matched_tokens / max(ground_truth_tokens, 1),
        "cer": edit_total / max(character_total, 1),
        "numeric_name_regions": numeric_total,
        "numeric_name_exact_accuracy": numeric_exact / max(numeric_total, 1),
        "raw": {
            "exact_lines": exact,
            "edit_distance": edit_total,
            "ground_truth_characters": character_total,
            "matched_tokens": matched_tokens,
            "ground_truth_tokens": ground_truth_tokens,
            "numeric_name_exact": numeric_exact,
        },
    }


def _runtime_evidence(
    runtime_report: dict[str, Any],
    model_id: str,
    policy: OcrV2GatePolicy,
    catalog_regions: int,
) -> dict[str, Any]:
    models = runtime_report.get("models")
    if not isinstance(models, dict) or not isinstance(models.get(model_id), dict):
        raise ValueError(f"runtime report is missing model {model_id}")
    evidence = models[model_id]
    benchmark_regions = int(evidence.get("benchmark_regions", -1))
    elapsed_seconds = float(evidence.get("elapsed_seconds", 0.0))
    regions_per_second = float(evidence.get("regions_per_second", 0.0))
    error_count = int(evidence.get("error_count", -1))
    if benchmark_regions < policy.throughput_canary_regions:
        raise ValueError(
            f"{model_id} benchmark has {benchmark_regions} regions; "
            f"need {policy.throughput_canary_regions}"
        )
    if elapsed_seconds <= 0.0 or regions_per_second <= 0.0 or error_count < 0:
        raise ValueError(f"invalid runtime evidence for {model_id}")
    measured_rps = benchmark_regions / elapsed_seconds
    if abs(measured_rps - regions_per_second) / measured_rps > 0.02:
        raise ValueError(f"inconsistent throughput evidence for {model_id}")
    eta_hours = catalog_regions / regions_per_second / policy.production_workers / 3600.0
    return {
        "benchmark_regions": benchmark_regions,
        "elapsed_seconds": elapsed_seconds,
        "regions_per_second": regions_per_second,
        "error_count": error_count,
        "peak_vram_mb": evidence.get("peak_vram_mb"),
        "four_t4_eta_hours": eta_hours,
        "eta_pass": eta_hours <= policy.production_max_hours,
    }


def evaluate_gate_b(
    sample_jsonl: Path,
    ground_truth_csv: Path,
    results_jsonl: Path,
    runtime_report_json: Path,
    policy: OcrV2GatePolicy,
) -> dict[str, Any]:
    samples, labels, usable = _validate_sample_and_labels(sample_jsonl, ground_truth_csv, policy)
    results = _validate_results(results_jsonl, samples)
    runtime_report = json.loads(Path(runtime_report_json).read_text(encoding="utf-8"))
    catalog_evidence = runtime_report.get("catalog_evidence")
    if not isinstance(catalog_evidence, dict):
        raise ValueError("runtime report is missing exact catalog evidence")
    try:
        archive_count = int(catalog_evidence.get("archive_count", -1))
        catalog_regions = int(catalog_evidence.get("catalog_regions", -1))
    except (TypeError, ValueError) as error:
        raise ValueError("catalog evidence is incomplete; attach all nine manifests") from error
    manifest_hashes = catalog_evidence.get("manifest_sha256s")
    if (
        archive_count != policy.catalog_archive_count
        or catalog_regions <= 0
        or not isinstance(manifest_hashes, list)
        or len(manifest_hashes) != archive_count
        or len(set(map(str, manifest_hashes))) != archive_count
        or any(not re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in manifest_hashes)
    ):
        raise ValueError("catalog evidence must contain every unique EasyOCR batch manifest")
    metrics = {model_id: _metrics(model_id, usable, labels, results) for model_id in MODEL_IDS}
    baseline = metrics[EASYOCR_MODEL_ID]
    decisions: dict[str, dict[str, Any]] = {}
    qualified: list[str] = []
    for model_id in (PADDLE_MODEL_ID, VIETOCR_MODEL_ID):
        candidate = metrics[model_id]
        runtime = _runtime_evidence(runtime_report, model_id, policy, catalog_regions)
        token_gain = candidate["exact_token_recall"] - baseline["exact_token_recall"]
        if baseline["cer"] == 0.0:
            cer_relative_reduction = 0.0
        else:
            cer_relative_reduction = (baseline["cer"] - candidate["cer"]) / baseline["cer"]
        numeric_regression = (
            baseline["numeric_name_exact_accuracy"] - candidate["numeric_name_exact_accuracy"]
        )
        quality_pass = (
            token_gain >= policy.token_recall_min_absolute_gain
            or cer_relative_reduction >= policy.cer_min_relative_reduction
        )
        numeric_pass = numeric_regression <= policy.numeric_name_max_absolute_regression
        integrity_pass = candidate["error_count"] == 0 and runtime["error_count"] == 0
        candidate_pass = quality_pass and numeric_pass and integrity_pass and runtime["eta_pass"]
        decisions[model_id] = {
            "qualified": candidate_pass,
            "quality_pass": quality_pass,
            "numeric_name_pass": numeric_pass,
            "integrity_pass": integrity_pass,
            "token_recall_absolute_gain": token_gain,
            "cer_relative_reduction": cer_relative_reduction,
            "numeric_name_absolute_regression": numeric_regression,
            "runtime": runtime,
        }
        if candidate_pass:
            qualified.append(model_id)
    if not qualified:
        selected = EASYOCR_MODEL_ID
        decision = "KEEP_EASYOCR_NO_CLEAR_WINNER"
    else:
        qualified.sort(
            key=lambda model_id: (
                metrics[model_id]["exact_token_recall"],
                -metrics[model_id]["cer"],
            ),
            reverse=True,
        )
        selected = qualified[0]
        if len(qualified) > 1:
            first, second = qualified[:2]
            recall_gap = abs(
                metrics[first]["exact_token_recall"] - metrics[second]["exact_token_recall"]
            )
            if recall_gap < policy.token_recall_tie_margin:
                selected = max(
                    (first, second),
                    key=lambda model_id: decisions[model_id]["runtime"]["regions_per_second"],
                )
        decision = "SELECT_RECOGNIZER_CHALLENGER"
    return {
        "schema_version": 1,
        "gate": "B_recognizer_ab",
        "decision": decision,
        "selected_model_id": selected,
        "baseline_model_id": EASYOCR_MODEL_ID,
        "sample_regions": len(samples),
        "usable_regions": len(usable),
        "excluded_regions": len(samples) - len(usable),
        "catalog_evidence": catalog_evidence,
        "policy": policy.model_dump(mode="json"),
        "metrics": metrics,
        "candidate_decisions": decisions,
        "inputs": {
            "sample_jsonl": str(sample_jsonl),
            "sample_jsonl_sha256": sha256_file(sample_jsonl),
            "ground_truth_csv": str(ground_truth_csv),
            "ground_truth_csv_sha256": sha256_file(ground_truth_csv),
            "results_jsonl": str(results_jsonl),
            "results_jsonl_sha256": sha256_file(results_jsonl),
            "runtime_report_json": str(runtime_report_json),
            "runtime_report_json_sha256": sha256_file(runtime_report_json),
        },
        "next_step": (
            "Run the selected recognizer on four UID-disjoint Kaggle T4 shards, then validate "
            "checkpoint/resume and exact UID/region coverage before building ocr.sqlite."
            if decision == "SELECT_RECOGNIZER_CHALLENGER"
            else "Keep the immutable EasyOCR artifacts; do not spend the one-day window on a marginal challenger."
        ),
    }
