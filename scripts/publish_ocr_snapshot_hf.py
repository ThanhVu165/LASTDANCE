"""Publish one immutable OCR development snapshot to the shared private HF Dataset."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import get_token

from offline.ocr_snapshot_hf import publish_snapshot_and_verify


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument(
        "--repo-id",
        default="MinhThuw0103/lastdance-visual-embeddings",
    )
    parser.add_argument("--output-report", type=Path)
    args = parser.parse_args()
    try:
        report = publish_snapshot_and_verify(
            args.snapshot_dir,
            repo_id=args.repo_id,
            token=get_token() or "",
        )
    except Exception as exc:
        report = {
            "schema_version": 1,
            "created_utc": datetime.now(UTC).isoformat(),
            "action": "blocked_before_verified_publish",
            "repo_id": args.repo_id,
            "snapshot_dir": str(args.snapshot_dir.resolve()),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.output_report is not None:
            args.output_report.parent.mkdir(parents=True, exist_ok=True)
            args.output_report.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        raise
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output_report is not None:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
