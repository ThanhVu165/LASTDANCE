"""Simulate conservative logo/overlay suppression on calibrated Gemini residuals."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from offline.ocr_overlay_audit import OverlayAuditPolicy, audit_overlay_residuals


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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrated-frames-jsonl", type=Path, required=True)
    parser.add_argument("--ground-truth-jsonl", type=Path, required=True)
    parser.add_argument(
        "--policy",
        type=Path,
        default=root / "configs" / "ocr_low_information_overlay_audit_policy.json",
    )
    parser.add_argument("--catalog-keyframes", type=int, default=293_336)
    parser.add_argument("--output-report-json", type=Path, required=True)
    parser.add_argument("--output-residual-jsonl", type=Path, required=True)
    return parser


def _scale(value: int, ratio: float) -> int:
    return int(round(value * ratio))


def main() -> int:
    args = _parser().parse_args()
    if args.catalog_keyframes <= 0:
        raise ValueError("catalog-keyframes must be positive")
    policy = OverlayAuditPolicy.model_validate_json(args.policy.read_text(encoding="utf-8"))
    frames = _read_jsonl(args.calibrated_frames_jsonl)
    labels = _read_jsonl(args.ground_truth_jsonl)
    report, residual = audit_overlay_residuals(
        frames,
        policy=policy,
        ground_truth_rows=labels,
    )
    dev_frames = int(report["scope"]["frames"])
    ratio = args.catalog_keyframes / dev_frames
    projection: dict[str, Any] = {
        "warning": "Dev-subset-5 linear extrapolation only; not a production exact count.",
        "catalog_keyframes": args.catalog_keyframes,
        "dev_to_catalog_scale": ratio,
    }
    for stage in ("before_overlay_audit", "suppression_candidates", "after_overlay_audit"):
        source = report["gemini_residual"][stage]
        projection[stage] = {key: _scale(int(value), ratio) for key, value in source.items()}
    report["full_catalog_projection"] = projection
    report["created_utc"] = datetime.now(UTC).isoformat()
    report["inputs"] = {
        "calibrated_frames": {
            "path": str(args.calibrated_frames_jsonl),
            "sha256": _sha256(args.calibrated_frames_jsonl),
        },
        "ground_truth": {
            "path": str(args.ground_truth_jsonl),
            "sha256": _sha256(args.ground_truth_jsonl),
        },
        "policy": {"path": str(args.policy), "sha256": _sha256(args.policy)},
    }

    args.output_report_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(args.output_residual_jsonl, residual)
    report_sha = _sha256(args.output_report_json)
    residual_sha = _sha256(args.output_residual_jsonl)
    print(json.dumps(report["gemini_residual"], ensure_ascii=False, indent=2))
    print("DECISION", report["decision"])
    print("REPORT", args.output_report_json, report_sha)
    print("RESIDUAL", args.output_residual_jsonl, residual_sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
