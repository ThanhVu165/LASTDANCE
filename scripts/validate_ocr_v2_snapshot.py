"""Validate an immutable OCR v2 development snapshot without modifying it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from offline.ocr_v2_snapshot import validate_ocr_v2_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--catalog-state", type=Path)
    args = parser.parse_args()
    manifest = validate_ocr_v2_snapshot(
        snapshot_dir=args.snapshot,
        catalog_path=args.catalog,
        catalog_state_path=args.catalog_state,
    )
    print(
        json.dumps(
            {
                "snapshot_id": manifest.snapshot_id,
                "valid": True,
                "recognition_coverage_complete": (
                    manifest.totals.recognition_coverage_complete
                ),
                "complete": manifest.complete,
                "production_ready": manifest.production_ready,
                "fts_rows": manifest.fts_rows,
                "error": manifest.totals.error_keyframes,
                "residual_regions": manifest.totals.residual_regions,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
