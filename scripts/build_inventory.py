"""Build the real-metadata video inventory using ffprobe."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from offline.config import DataLayout
from offline.preprocessing.inventory import (
    build_inventory,
    discover_videos,
    write_inventory_atomic,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--ffprobe", default=os.environ.get("AIC_FFPROBE", "ffprobe"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    layout = DataLayout(args.data_root.resolve()) if args.data_root else DataLayout.from_environment()
    paths = discover_videos(layout.videos)
    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("--limit must be non-negative")
        paths = paths[: args.limit]
    records = build_inventory(
        paths,
        data_root=layout.root,
        ffprobe_binary=args.ffprobe,
    )
    output = args.output or (layout.index / "inventory.json")
    write_inventory_atomic(output, records)
    print(f"inventory: {len(records)} videos -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
