"""Build an immutable development ASR SQLite snapshot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from offline.asr_snapshot import build_asr_snapshot, load_envelope_records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--catalog-state", type=Path)
    parser.add_argument("--source-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--source-format", default="asr_envelope_v1", choices=["asr_envelope_v1"])
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    output = args.output_root or Path(os.environ.get("AIC_DATA", "data")) / "asr" / "snapshots"
    records = load_envelope_records(args.source_jsonl)
    destination, manifest = build_asr_snapshot(
        catalog_path=args.catalog, catalog_state_path=args.catalog_state,
        records=records, source_paths=args.source_jsonl, output_root=output,
        source_format=args.source_format,
    )
    print(json.dumps({"snapshot": str(destination), "snapshot_id": manifest.snapshot_id, "fts_rows": manifest.fts_rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
