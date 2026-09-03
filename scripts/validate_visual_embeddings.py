"""Validate one completed visual embedding modality without loading its model."""

from __future__ import annotations

import argparse
from pathlib import Path

from offline.config import DataLayout
from offline.visual_embeddings import validate_completed_visual_embedding


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--keyframes-root", type=Path)
    parser.add_argument("--require-resume-verified", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    layout = (
        DataLayout(args.data_root.resolve())
        if args.data_root
        else DataLayout.from_environment()
    )
    result = validate_completed_visual_embedding(
        args.artifact_dir.resolve(),
        catalog_path=args.catalog.resolve(),
        keyframes_root=(args.keyframes_root or layout.keyframes).resolve(),
        require_resume_verified=args.require_resume_verified,
    )
    print(
        f"PASS: {result.next_index}/{result.total} vectors; "
        f"resume_verified={result.checkpoint_resume_verified}; {result.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
