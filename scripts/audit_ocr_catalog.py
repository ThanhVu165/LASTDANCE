"""Audit the exact frames.csv selected for production OCR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from offline.config import DataLayout
from offline.ocr_catalog import audit_ocr_catalog


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--expected-records", type=int)
    parser.add_argument("--expected-videos", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    layout = (
        DataLayout(args.data_root.resolve())
        if args.data_root
        else DataLayout.from_environment()
    )
    catalog = (args.catalog or (layout.index / "frames.csv")).resolve()
    report = audit_ocr_catalog(
        catalog,
        state_path=(args.state.resolve() if args.state else None),
    )
    if args.expected_records is not None and report["record_count"] != args.expected_records:
        raise RuntimeError(
            f"catalog record count mismatch: expected={args.expected_records}, "
            f"actual={report['record_count']}"
        )
    if args.expected_videos is not None and report["video_count"] != args.expected_videos:
        raise RuntimeError(
            f"catalog video count mismatch: expected={args.expected_videos}, "
            f"actual={report['video_count']}"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
