"""Build or resume one independent visual embedding modality on Kaggle CUDA."""

from __future__ import annotations

import argparse
from pathlib import Path

from offline.config import DataLayout
from offline.visual_embeddings import (
    IntentionalEmbeddingInterruption,
    SUPPORTED_MODALITIES,
    run_visual_embedding,
)
from offline.visual_models import DEFAULT_MODEL_CONFIG, create_visual_encoder


def _read_video_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    values: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        if not line or line.lstrip().startswith("#"):
            continue
        if line.strip() != line or any(character.isspace() for character in line):
            raise ValueError(f"non-canonical video_id at {path}:{line_number}")
        values.append(line)
    if not values:
        raise ValueError("video ID file contains no IDs")
    if len(set(values)) != len(values):
        raise ValueError("video ID file contains duplicate IDs")
    return set(values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modality", choices=SUPPORTED_MODALITIES, required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--keyframes-root",
        type=Path,
        help="read-only keyframes directory; defaults to AIC_DATA/keyframes",
    )
    parser.add_argument(
        "--embedding-root",
        type=Path,
        help="writable artifact root; defaults to AIC_DATA/index/visual-embeddings",
    )
    parser.add_argument("--video-id-file", type=Path)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--stop-after-shards", type=int)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    layout = (
        DataLayout(args.data_root.resolve())
        if args.data_root
        else DataLayout.from_environment()
    )
    encoder = create_visual_encoder(
        args.modality,
        config_path=args.model_config.resolve(),
        device=args.device,
    )
    try:
        result = run_visual_embedding(
            encoder=encoder,
            modality=args.modality,
            batch_id=args.batch_id,
            catalog_path=args.catalog.resolve(),
            keyframes_root=(args.keyframes_root or layout.keyframes).resolve(),
            output_root=(
                args.embedding_root or (layout.index / "visual-embeddings")
            ).resolve(),
            batch_size=args.batch_size,
            video_ids=_read_video_ids(args.video_id_file),
            stop_after_shards=args.stop_after_shards,
        )
    except IntentionalEmbeddingInterruption as error:
        print(error)
        return 75
    print(
        f"{args.modality}: {result.next_index}/{result.total} -> {result.output_dir}; "
        f"resume_verified={result.checkpoint_resume_verified}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
