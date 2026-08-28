"""Build an immutable development OCR SQLite snapshot; never publish it as final."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from offline.ocr_snapshot import (
    build_ocr_snapshot,
    gate2_calibrated_vintern_coverage,
    gate2_vintern_coverage,
    load_envelope_snapshot_records,
    load_gate2_calibrated_snapshot_records,
    load_gate2_easyocr_snapshot_records,
)
from offline.ocr_vintern_gate2 import VinternGate2Policy
from offline.ocr_vintern_calibration import validate_materialized_calibration_files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--catalog-state", type=Path)
    parser.add_argument("--source-jsonl", type=Path, action="append", required=True)
    parser.add_argument(
        "--source-format",
        choices=(
            "ocr_envelope_v1",
            "gate2_easyocr_dev_v1",
            "gate2_calibrated_dev_v1",
        ),
        required=True,
    )
    parser.add_argument("--vintern-results-jsonl", type=Path)
    parser.add_argument("--calibration-json", type=Path)
    parser.add_argument("--override-audit-jsonl", type=Path)
    parser.add_argument(
        "--vintern-policy",
        type=Path,
        default=Path("configs/ocr_vintern_gate2_policy.json"),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--parent-snapshot-id")
    args = parser.parse_args()

    output_root = args.output_root
    if output_root is None:
        aic_data = Path(os.environ.get("AIC_DATA", "data"))
        output_root = aic_data / "ocr" / "snapshots"

    vintern = None
    if args.source_format == "ocr_envelope_v1":
        if any(
            value is not None
            for value in (
                args.vintern_results_jsonl,
                args.calibration_json,
                args.override_audit_jsonl,
            )
        ):
            parser.error("envelope snapshots do not accept Gate 2 evidence arguments")
        records = load_envelope_snapshot_records(args.source_jsonl)
        materialized_policy = "canonical terminal envelope text"
        source_paths = list(args.source_jsonl)
    elif args.source_format == "gate2_easyocr_dev_v1":
        if args.calibration_json is not None or args.override_audit_jsonl is not None:
            parser.error("EasyOCR-only snapshot cannot claim calibration evidence")
        if len(args.source_jsonl) != 1:
            parser.error("gate2_easyocr_dev_v1 requires exactly one --source-jsonl")
        records = load_gate2_easyocr_snapshot_records(args.source_jsonl[0])
        policy = VinternGate2Policy.model_validate_json(
            args.vintern_policy.read_text(encoding="utf-8")
        )
        vintern = gate2_vintern_coverage(
            args.source_jsonl[0], args.vintern_results_jsonl, policy=policy
        )
        materialized_policy = (
            "EasyOCR text/confidence only; Vintern results are coverage evidence and are "
            "not materialized because Vintern confidence is not calibrated"
        )
        source_paths = list(args.source_jsonl)
        source_paths.append(args.vintern_policy)
        if args.vintern_results_jsonl is not None:
            source_paths.append(args.vintern_results_jsonl)
    else:
        if len(args.source_jsonl) != 1:
            parser.error("gate2_calibrated_dev_v1 requires exactly one --source-jsonl")
        if args.vintern_results_jsonl is not None:
            parser.error(
                "calibrated frame artifact already contains Vintern decisions; "
                "do not pass --vintern-results-jsonl"
            )
        if args.calibration_json is None or args.override_audit_jsonl is None:
            parser.error(
                "gate2_calibrated_dev_v1 requires --calibration-json and "
                "--override-audit-jsonl"
            )
        validate_materialized_calibration_files(
            str(args.source_jsonl[0]),
            str(args.override_audit_jsonl),
            str(args.calibration_json),
        )
        records = load_gate2_calibrated_snapshot_records(args.source_jsonl[0])
        vintern = gate2_calibrated_vintern_coverage(args.source_jsonl[0])
        materialized_policy = (
            "EasyOCR+Vintern calibrated; Vintern overrides only routed regions when "
            "empirical bucket confidence is strictly greater than source EasyOCR confidence"
        )
        source_paths = list(args.source_jsonl)
        source_paths.extend([args.calibration_json, args.override_audit_jsonl])

    destination, manifest = build_ocr_snapshot(
        catalog_path=args.catalog,
        catalog_state_path=args.catalog_state,
        records=records,
        source_paths=source_paths,
        source_format=args.source_format,
        materialized_text_policy=materialized_policy,
        output_root=output_root,
        vintern_by_video=vintern,
        parent_snapshot_id=args.parent_snapshot_id,
    )
    print(
        json.dumps(
            {
                "snapshot": str(destination),
                "snapshot_id": manifest.snapshot_id,
                "complete": manifest.complete,
                "production_ready": manifest.production_ready,
                "coverage_fraction": manifest.coverage_fraction,
                "fts_rows": manifest.fts_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
