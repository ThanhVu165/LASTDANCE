"""Evaluate completed human labels for the 300-frame CRAFT Gate A pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from offline.ocr_craft_gate_a import CraftGateAPolicy, evaluate_craft_gate_a


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-jsonl", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument(
        "--policy",
        type=Path,
        default=root / "configs" / "ocr_craft_gate_a_policy.json",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    policy = CraftGateAPolicy.model_validate_json(args.policy.read_text(encoding="utf-8"))
    report = evaluate_craft_gate_a(
        results_path=args.results_jsonl,
        review_csv_path=args.review_csv,
        policy=policy,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.output_json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["gate_b_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
