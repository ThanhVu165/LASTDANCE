"""Build or increment one local FAISS IndexIDMap from completed embedding batches."""

from __future__ import annotations

import argparse
from pathlib import Path

from offline.config import DataLayout
from offline.faiss_indexes import build_faiss_index
from offline.visual_embeddings import SUPPORTED_MODALITIES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modality", choices=SUPPORTED_MODALITIES, required=True)
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        action="append",
        required=True,
        help="completed single-modality artifact; repeat for disjoint batches",
    )
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--keyframes-root", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="defaults to AIC_DATA/index/<modality>.faiss",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    layout = (
        DataLayout(args.data_root.resolve())
        if args.data_root
        else DataLayout.from_environment()
    )
    output = (args.output or (layout.index / f"{args.modality}.faiss")).resolve()
    result = build_faiss_index(
        modality=args.modality,
        embedding_dirs=[path.resolve() for path in args.embedding_dir],
        catalog_path=args.catalog.resolve(),
        keyframes_root=(args.keyframes_root or layout.keyframes).resolve(),
        output_path=output,
    )
    report = result.report
    print(
        f"PASS: {report.modality} {report.record_count} UID / "
        f"{report.video_count} video / dim={report.vector_dim}; "
        f"added={result.added_records}; {report.index_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
