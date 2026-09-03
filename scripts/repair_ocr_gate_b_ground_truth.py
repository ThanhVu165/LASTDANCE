"""Restore immutable Gate B review columns after a CSV was edited in Excel."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path


MUTABLE_COLUMNS = {
    "label_status",
    "ground_truth_is_empty",
    "human_text",
    "annotator",
    "notes",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-zip", type=Path, required=True)
    parser.add_argument("--edited-csv", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument(
        "--normalize-empty-exclusions",
        action="store_true",
        help="Treat exclude_unreadable + empty=yes as a completed no-text label.",
    )
    parser.add_argument("--expected-labeled", type=int, default=98)
    parser.add_argument("--expected-excluded", type=int, default=2)
    return parser


def main() -> int:
    args = _parser().parse_args()
    with zipfile.ZipFile(args.review_zip) as archive:
        if archive.testzip() is not None:
            raise ValueError("review ZIP CRC failure")
        original = list(
            csv.DictReader(
                io.StringIO(
                    archive.read("vintern-calibration-ground-truth.csv").decode(
                        "utf-8-sig"
                    )
                )
            )
        )
    with args.edited_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        edited = list(csv.DictReader(handle))
    if len(original) != 100 or len(edited) != 100:
        raise ValueError("Gate B review must contain exactly 100 rows")
    original_by_sample = {row["sample_index"]: row for row in original}
    edited_by_sample = {row["sample_index"]: row for row in edited}
    if len(original_by_sample) != 100 or set(original_by_sample) != set(edited_by_sample):
        raise ValueError("edited sample_index set differs from the immutable review")

    repaired: list[dict[str, str]] = []
    immutable_differences = 0
    normalized_samples: list[str] = []
    for sample_index in sorted(original_by_sample, key=int):
        source = original_by_sample[sample_index]
        human = edited_by_sample[sample_index]
        immutable_differences += sum(
            source[column] != human.get(column, "")
            for column in source
            if column not in MUTABLE_COLUMNS
        )
        row = dict(source)
        for column in MUTABLE_COLUMNS:
            row[column] = str(human.get(column) or "").strip()
        status = row["label_status"].casefold()
        is_empty = row["ground_truth_is_empty"].casefold()
        if (
            args.normalize_empty_exclusions
            and status == "exclude_unreadable"
            and is_empty == "yes"
        ):
            row["label_status"] = "labeled"
            normalized_samples.append(sample_index)
            status = "labeled"
        if status not in {"labeled", "exclude_unreadable"}:
            raise ValueError(f"invalid label_status at sample {sample_index}: {status}")
        if status == "labeled":
            if is_empty not in {"yes", "no"}:
                raise ValueError(f"invalid empty flag at sample {sample_index}")
            if is_empty == "yes" and row["human_text"]:
                raise ValueError(f"no-text sample carries text: {sample_index}")
            if is_empty == "no" and not row["human_text"]:
                raise ValueError(f"text sample is blank: {sample_index}")
        repaired.append(row)

    statuses = Counter(row["label_status"].casefold() for row in repaired)
    if statuses["labeled"] != args.expected_labeled:
        raise ValueError(f"labeled count mismatch: {statuses['labeled']}")
    if statuses["exclude_unreadable"] != args.expected_excluded:
        raise ValueError(f"excluded count mismatch: {statuses['exclude_unreadable']}")
    labeled_uids = {
        int(row["keyframe_uid"])
        for row in repaired
        if row["label_status"].casefold() == "labeled"
    }
    if len(labeled_uids) != args.expected_labeled:
        raise ValueError("labeled rows do not cover distinct immutable keyframe_uids")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for row in repaired:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "decision": "PASS_REPAIRED_98_OF_100",
        "inputs": {
            "review_zip_sha256": _sha256(args.review_zip),
            "edited_csv_sha256": _sha256(args.edited_csv),
        },
        "immutable_columns_restored_from_zip": True,
        "immutable_cell_differences_discarded": immutable_differences,
        "normalized_empty_exclusion_samples": normalized_samples,
        "status_counts": dict(statuses),
        "labeled_distinct_frames": len(labeled_uids),
        "excluded_samples": [
            row["sample_index"]
            for row in repaired
            if row["label_status"].casefold() == "exclude_unreadable"
        ],
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
