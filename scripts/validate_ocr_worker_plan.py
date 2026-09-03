"""Validate a four-worker OCR plan without running any model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from offline.ocr_production import OcrWorkerPlan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args()
    plan = OcrWorkerPlan.model_validate_json(args.plan.read_text(encoding="utf-8"))
    summary = {
        "schema_version": plan.schema_version,
        "enabled_workers": sum(assignment.enabled for assignment in plan.assignments),
        "batch_count": len(plan.expected_batch_ids),
        "assignments": {
            assignment.worker_id: assignment.batch_ids
            for assignment in plan.assignments
            if assignment.enabled
        },
        "disjoint": True,
        "exhaustive": True,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
