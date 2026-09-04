"""Validate an ASR worker plan JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from offline.asr_production import AsrWorkerPlan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    plan = AsrWorkerPlan.model_validate_json(args.plan.read_text(encoding="utf-8"))
    print(json.dumps(plan.model_dump(mode="json"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
