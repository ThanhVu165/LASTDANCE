"""Validate one FAISS file by diffing its actual IDs against frames.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

from offline.faiss_indexes import validate_faiss_index
from offline.visual_embeddings import SUPPORTED_MODALITIES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--modality", choices=SUPPORTED_MODALITIES)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = validate_faiss_index(
        args.index.resolve(),
        catalog_path=args.catalog.resolve(),
        state_path=args.state.resolve() if args.state else None,
        expected_modality=args.modality,
    )
    print(
        f"PASS: {report.modality} {report.record_count} UID / "
        f"{report.video_count} video / dim={report.vector_dim} / "
        f"{report.source_count} source(s); {report.index_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
