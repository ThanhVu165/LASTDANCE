"""Score predictions against reviewed labels without calling models or network services."""
import argparse
import csv
from types import SimpleNamespace
import hashlib
import json
from pathlib import Path

from shared.evaluation import EvaluationCase, Prediction, evaluate, validate_suite, diagnostic_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", choices=["development", "held_out", "regression"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, help="Development-selected config lock required for held-out scoring")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--acceptance", action="store_true")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--runs", type=Path)
    args = parser.parse_args()
    cases = [EvaluationCase.model_validate(r) for r in json.loads(args.labels.read_text(encoding="utf-8"))]
    validate_suite(cases, acceptance=args.acceptance)
    raw = json.loads(args.predictions.read_text(encoding="utf-8"))
    predictions = {k: [Prediction.model_validate(r) for r in rows] for k, rows in raw.items()}
    digest = hashlib.sha256(args.config.read_bytes()).hexdigest()
    if args.split == "held_out":
        if args.freeze is None:
            parser.error("held-out scoring requires --freeze from development selection")
        lock = json.loads(args.freeze.read_text(encoding="utf-8"))
        if lock.get("config_sha256") != digest or lock.get("selected_on") != "development":
            raise ValueError("configuration is not frozen from development")
    report = evaluate(cases, predictions, split=args.split)
    if args.split == "held_out" and lock.get("labels_sha256") != report["labels_sha256"]:
        raise ValueError("labels differ from the frozen development suite")
    if args.acceptance and (args.catalog is None or args.runs is None):
        parser.error("acceptance requires --catalog and --runs for diagnostic metrics")
    if args.catalog:
        with args.catalog.open(encoding="utf-8", newline="") as stream:
            frames = [SimpleNamespace(video_id=r["video_id"], frame_id=int(r["frame_id"])) for r in csv.DictReader(stream)]
        runs = json.loads(args.runs.read_text(encoding="utf-8")) if args.runs else None
        if runs and any(row.get("config_sha256") != digest for row in runs.values()):
            raise ValueError("trace configuration does not match evaluated configuration")
        if args.acceptance and set(runs or {}) != {c.query.query_name for c in cases if c.split == args.split}:
            raise ValueError("acceptance traces must cover exactly the selected split")
        report["diagnostics"] = diagnostic_metrics(cases, predictions, split=args.split, catalog_frames=frames, runs=runs)
        report["catalog_sha256"] = hashlib.sha256(args.catalog.read_bytes()).hexdigest()
    report["config_sha256"] = digest
    report["predictions_sha256"] = hashlib.sha256(args.predictions.read_bytes()).hexdigest()
    if args.baseline:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        if baseline.get("labels_sha256") != report["labels_sha256"] or baseline.get("split") != args.split:
            raise ValueError("baseline must use the same labels and split")
        report["non_regression"] = all(metrics["final_score"] >= baseline["per_task"][task]["final_score"] for task, metrics in report["per_task"].items())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"mean_final_score": report["mean_final_score"], "split": args.split}))
    return 1 if report.get("non_regression") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
