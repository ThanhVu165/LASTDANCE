"""Materialize validated ASR archives plus an optional atomic checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from offline.asr_handoff import materialize_asr_handoff


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-dir", type=Path, action="append", default=[])
    parser.add_argument("--checkpoint-jsonl", type=Path)
    parser.add_argument("--checkpoint-state", type=Path)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    args = parser.parse_args()
    if (args.checkpoint_jsonl is None) != (args.checkpoint_state is None):
        parser.error("--checkpoint-jsonl and --checkpoint-state must be supplied together")
    pairs = [
        (directory / "asr-envelope.jsonl", directory / "manifest.json")
        for directory in args.archive_dir
    ]
    _, audit = materialize_asr_handoff(
        archive_pairs=pairs,
        checkpoint_pair=(args.checkpoint_jsonl, args.checkpoint_state)
        if args.checkpoint_jsonl else None,
        catalog_path=args.catalog,
        inventory_path=args.inventory,
        output_jsonl=args.output_jsonl,
        audit_path=args.output_audit,
        source_revision=args.source_revision,
    )
    print(json.dumps({
        "output_jsonl": str(args.output_jsonl),
        "output_audit": str(args.output_audit),
        "accepted_records": audit["accepted_records"],
        "coverage_fraction": audit["coverage_fraction"],
        "conflict_videos": audit["conflict_videos"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
