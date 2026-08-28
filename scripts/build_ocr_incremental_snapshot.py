"""Union independent OCR batch JSONLs and build one immutable development snapshot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from offline.ocr_incremental_snapshot import prepare_incremental_snapshot_union
from offline.ocr_snapshot import build_ocr_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--catalog-state", type=Path)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--parent-snapshot-id")
    args = parser.parse_args()

    output_root = args.output_root
    if output_root is None:
        output_root = Path(os.environ.get("AIC_DATA", "data")) / "ocr" / "snapshots"
    records, batches, sources = prepare_incremental_snapshot_union(
        plan_path=args.plan,
        catalog_path=args.catalog,
        catalog_state_path=args.catalog_state,
    )
    destination, manifest = build_ocr_snapshot(
        catalog_path=args.catalog,
        catalog_state_path=args.catalog_state,
        records=records,
        source_paths=sources,
        source_format="incremental_batch_union_v1",
        materialized_text_policy=(
            "Per-batch highest completed tier; craft_only is coverage-only and never enters FTS"
        ),
        output_root=output_root,
        parent_snapshot_id=args.parent_snapshot_id,
        batch_coverage=batches,
    )
    print(
        json.dumps(
            {
                "snapshot": str(destination),
                "snapshot_id": manifest.snapshot_id,
                "complete": manifest.complete,
                "production_ready": manifest.production_ready,
                "fts_rows": manifest.fts_rows,
                "batches": {
                    batch_id: {
                        "tier": row.tier,
                        "complete": row.complete,
                        "processed_keyframes": row.processed_keyframes,
                        "expected_keyframes": row.expected_keyframes,
                    }
                    for batch_id, row in sorted(manifest.batches.items())
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
