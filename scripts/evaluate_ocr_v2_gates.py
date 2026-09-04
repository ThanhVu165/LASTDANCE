"""Evaluate the OCR-v2 Gate A detector triage or Gate B recognizer A/B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from offline.ocr_v2_gate import evaluate_gate_a, evaluate_gate_b, load_policy


def write_report(report: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=Path("configs/ocr_v2_gate_policy.json"))
    subparsers = parser.add_subparsers(dest="gate", required=True)

    gate_a = subparsers.add_parser("gate-a", help="Evaluate CRAFT bounding-box triage labels")
    gate_a.add_argument("--review-csv", type=Path, required=True)
    gate_a.add_argument("--output", type=Path, required=True)

    gate_b = subparsers.add_parser("gate-b", help="Evaluate same-crop recognizer outputs")
    gate_b.add_argument("--sample-jsonl", type=Path, required=True)
    gate_b.add_argument("--ground-truth-csv", type=Path, required=True)
    gate_b.add_argument("--results-jsonl", type=Path, required=True)
    gate_b.add_argument("--runtime-report-json", type=Path, required=True)
    gate_b.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    policy = load_policy(args.policy)
    if args.gate == "gate-a":
        report = evaluate_gate_a(args.review_csv, policy)
    else:
        report = evaluate_gate_b(
            args.sample_jsonl,
            args.ground_truth_csv,
            args.results_jsonl,
            args.runtime_report_json,
            policy,
        )
    write_report(report, args.output)


if __name__ == "__main__":
    main()
