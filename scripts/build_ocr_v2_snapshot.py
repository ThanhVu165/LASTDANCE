"""Validate nine OCR v2 production ZIPs and build an immutable dev SQLite."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from offline.ocr_v2_snapshot import build_ocr_v2_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--catalog-state", type=Path)
    parser.add_argument("--worker-plan", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Root for HF-relative source paths; defaults to source manifest directory",
    )
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    output_root = args.output_root
    if output_root is None:
        output_root = Path(os.environ.get("AIC_DATA", "data")) / "ocr" / "snapshots"
    destination, manifest = build_ocr_v2_snapshot(
        catalog_path=args.catalog,
        catalog_state_path=args.catalog_state,
        worker_plan_path=args.worker_plan,
        source_manifest_path=args.source_manifest,
        source_root=args.source_root,
        output_root=output_root,
        progress=lambda message: print(f"OCR_V2_UNION {message}", flush=True),
    )
    print(
        json.dumps(
            {
                "snapshot": str(destination),
                "snapshot_id": manifest.snapshot_id,
                "immutable": manifest.immutable,
                "complete": manifest.complete,
                "production_ready": manifest.production_ready,
                "recognition_coverage_complete": (
                    manifest.totals.recognition_coverage_complete
                ),
                "success": manifest.totals.success_keyframes,
                "no_text": manifest.totals.no_text_keyframes,
                "error": manifest.totals.error_keyframes,
                "residual_frames": manifest.totals.residual_frames,
                "residual_regions": manifest.totals.residual_regions,
                "fts_rows": manifest.fts_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
