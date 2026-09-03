"""Fit Vintern confidence and materialize Gate B without another model inference run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from offline.ocr_vintern_calibration import (
    VinternCalibrationPolicy,
    calibration_examples_from_rows,
    fit_vintern_calibration,
    materialize_calibrated_gate2_frames,
    validate_materialized_calibration_audit,
)
from offline.ocr_vintern_gate2 import VinternGate2Policy


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _read_ground_truth(path: Path) -> list[dict[str, Any]]:
    if path.suffix.casefold() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    return _read_jsonl(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--easyocr-jsonl", type=Path, required=True)
    parser.add_argument("--vintern-results-jsonl", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument(
        "--gate-policy",
        type=Path,
        default=root / "configs" / "ocr_vintern_gate2_policy.json",
    )
    parser.add_argument(
        "--calibration-policy",
        type=Path,
        default=root / "configs" / "ocr_vintern_calibration_policy.json",
    )
    parser.add_argument("--output-calibration-json", type=Path, required=True)
    parser.add_argument("--output-materialized-jsonl", type=Path, required=True)
    parser.add_argument("--output-audit-jsonl", type=Path, required=True)
    parser.add_argument("--output-summary-json", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    easy_rows = _read_jsonl(args.easyocr_jsonl)
    vintern_rows = _read_jsonl(args.vintern_results_jsonl)
    labels = _read_ground_truth(args.ground_truth)
    gate_policy = VinternGate2Policy.model_validate_json(
        args.gate_policy.read_text(encoding="utf-8")
    )
    calibration_policy = VinternCalibrationPolicy.model_validate_json(
        args.calibration_policy.read_text(encoding="utf-8")
    )
    examples = calibration_examples_from_rows(
        easy_rows, vintern_rows, labels, gate_policy=gate_policy
    )
    excluded_unreadable = sum(
        str(row.get("label_status") or "").strip().casefold()
        == "exclude_unreadable"
        for row in labels
    )
    if len(labels) != calibration_policy.review_rows:
        raise ValueError(
            f"ground-truth review row count mismatch: {len(labels)} != "
            f"{calibration_policy.review_rows}"
        )
    if excluded_unreadable > calibration_policy.max_excluded_unreadable:
        raise ValueError(
            "too many unreadable exclusions: "
            f"{excluded_unreadable} > {calibration_policy.max_excluded_unreadable}"
        )
    calibrated_candidate_ids = {example.candidate_id for example in examples}
    ground_truth_uids = {
        int(row["keyframe_uid"])
        for row in labels
        if str(row.get("candidate_id") or row.get("region_id") or "")
        in calibrated_candidate_ids
    }
    table = fit_vintern_calibration(
        examples,
        policy=calibration_policy,
        ground_truth_frame_count=len(ground_truth_uids),
    )
    materialized, audit = materialize_calibrated_gate2_frames(
        easy_rows,
        vintern_rows,
        table=table,
        calibration_policy=calibration_policy,
        gate_policy=gate_policy,
    )
    validate_materialized_calibration_audit(materialized, audit, table=table)

    args.output_calibration_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_calibration_json.write_text(
        json.dumps(table.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(args.output_materialized_jsonl, materialized)
    _write_jsonl(args.output_audit_jsonl, audit)

    decisions = Counter(row["decision_reason"] for row in audit)
    residual_reasons = Counter(
        reason
        for row in audit
        for reason in row.get("gemini_residual_reasons", [])
    )
    summary = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "decision": "PASS_CALIBRATION_MATERIALIZED",
        "model_inference_calls": 0,
        "same_gate_b_results_reused": True,
        "correctness_metric": calibration_policy.correctness_metric,
        "inputs": {
            "easyocr_jsonl": {"sha256": _sha256(args.easyocr_jsonl), "frames": len(easy_rows)},
            "vintern_results_jsonl": {
                "sha256": _sha256(args.vintern_results_jsonl),
                "results": len(vintern_rows),
            },
            "ground_truth": {"sha256": _sha256(args.ground_truth), "rows": len(labels)},
            "gate_policy": {"sha256": _sha256(args.gate_policy)},
            "calibration_policy": {"sha256": _sha256(args.calibration_policy)},
        },
        "calibration": {
            "evidence_tier": calibration_policy.evidence_tier,
            "labeled_regions": len(examples),
            "distinct_ground_truth_frames": len(ground_truth_uids),
            "excluded_or_control_rows": len(labels) - len(examples),
            "excluded_unreadable_rows": excluded_unreadable,
            "correct_regions": sum(example.correct for example in examples),
            "calibration_set_exact_match_not_holdout": (
                sum(example.correct for example in examples) / len(examples)
            ),
            "logprob_available_regions": sum(
                example.signals.mean_token_logprob is not None for example in examples
            ),
            "buckets": len(table.buckets),
        },
        "materialization": {
            "routed_regions_audited": len(audit),
            "overridden_regions": sum(bool(row["overwritten"]) for row in audit),
            "gemini_residual_regions": sum(
                bool(row.get("gemini_residual")) for row in audit
            ),
            "gemini_residual_reasons": dict(sorted(residual_reasons.items())),
            "decision_reasons": dict(sorted(decisions.items())),
            "policy": (
                "override iff guard passes, a supported non-global calibration bucket "
                "exists, and calibrated Vintern confidence > source EasyOCR confidence"
            ),
        },
        "outputs": {
            "calibration_json": str(args.output_calibration_json),
            "materialized_jsonl": str(args.output_materialized_jsonl),
            "audit_jsonl": str(args.output_audit_jsonl),
        },
    }
    args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
