"""Verify pinned EasyOCR package/model bytes for a network-free fallback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from offline.ocr_models import (
    DEFAULT_EASYOCR_REGISTRY,
    verify_easyocr_offline_files,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-storage-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_EASYOCR_REGISTRY)
    parser.add_argument("--archive-dir", type=Path)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument(
        "--skip-package-check",
        action="store_true",
        help="Checksum audit only; production preflight must not use this option.",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = verify_easyocr_offline_files(
        args.model_storage_dir.resolve(),
        registry_path=args.registry.resolve(),
        archive_directory=(args.archive_dir.resolve() if args.archive_dir else None),
        wheel_path=(args.wheel.resolve() if args.wheel else None),
        verify_package_version=not args.skip_package_check,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for row in report["weights"]:
            print(
                f"PASS: {row['model']} {row['filename']} "
                f"size={row['size_bytes']} sha256={row['sha256']}"
            )
        if args.skip_package_check:
            print("AUDIT ONLY: EasyOCR package version check was skipped")
        else:
            print(f"PASS: easyocr=={report['installed_package_version']}")
        print("PASS: download_enabled=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
