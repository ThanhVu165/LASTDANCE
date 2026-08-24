"""Compare CPU/GPU shot manifests exactly while allowing device provenance to differ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from offline.preprocessing.shot_parity import compare_shot_manifests


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="trusted CPU schema-v2 manifest")
    parser.add_argument("candidate", type=Path, help="candidate CUDA schema-v2 manifest")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    mismatches = compare_shot_manifests(args.reference, args.candidate)
    if mismatches:
        print("FAIL: shot manifests are not semantically identical")
        print(json.dumps(mismatches, ensure_ascii=False, indent=2))
        return 1
    print("PASS: every shot boundary and excluded transition range matches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
