"""Freeze the highest development score among comparable, non-regressing runs."""
import argparse
import hashlib
import json
from pathlib import Path


def freeze(reports, output):
    choices = []
    for path in reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("split") != "development" or not report.get("config_sha256"):
            raise ValueError("only development reports with configuration hashes may select a configuration")
        if report.get("non_regression") is False:
            continue
        choices.append((report, path))
    if not choices or len({r["labels_sha256"] for r, _ in choices}) != 1:
        raise ValueError("selection requires comparable non-regressing development reports")
    selected, path = max(choices, key=lambda pair: (pair[0]["mean_final_score"], pair[0]["config_sha256"]))
    lock = {"schema_version": 1, "selected_on": "development", "labels_sha256": selected["labels_sha256"],
            "config_sha256": selected["config_sha256"], "development_score": selected["mean_final_score"],
            "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "compared_reports": [{"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                                  "config_sha256": r["config_sha256"], "score": r["mean_final_score"]} for r, p in choices]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(lock, stream, indent=2)
    return lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.reports, args.output)))


if __name__ == "__main__": main()
