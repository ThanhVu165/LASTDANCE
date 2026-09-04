"""Push ASR audio batches to HuggingFace Dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi


def push_audio_batches(
    audio_dir: Path,
    hf_repo_id: str,
    hf_token: str | None = None,
    batch_size: int = 100,
) -> None:
    """Push all FLAC audio files to HF in 9 batches."""

    audio_dir = Path(audio_dir).resolve()
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"audio directory not found: {audio_dir}")

    api = HfApi(token=hf_token or os.environ.get("HF_TOKEN"))

    # Get all FLAC files sorted
    audio_files = sorted(audio_dir.glob("*.flac"))
    print(f"Found {len(audio_files)} FLAC files")

    if not audio_files:
        raise ValueError("No FLAC files found")

    # Push to 9 batches
    for batch_num in range(1, 10):
        batch_id = f"batch-{batch_num:02d}"
        start_idx = (batch_num - 1) * batch_size
        end_idx = min(batch_num * batch_size, len(audio_files))

        batch_files = audio_files[start_idx:end_idx]
        if not batch_files:
            print(f"Skipping {batch_id} (no files)")
            continue

        print(f"\nPushing {batch_id}: {len(batch_files)} files")

        operations = [
            CommitOperationAdd(
                path_in_repo=f"asr/audio/{batch_id}/{file.name}",
                path_or_fileobj=file,
            )
            for file in batch_files
        ]

        try:
            api.create_commit(
                repo_id=hf_repo_id,
                repo_type="dataset",
                operations=operations,
                commit_message=f"Add audio {batch_id} ({len(batch_files)} files)",
            )
            print(f"✓ {batch_id} pushed successfully")
        except Exception as exc:
            print(f"✗ {batch_id} failed: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default="F:\\LASTDANCE-DATA\\data\\audio",
        help="Directory containing FLAC audio files",
    )
    parser.add_argument(
        "--hf-repo",
        default="MinhThuw0103/lastdance-visual-embeddings",
        help="HuggingFace Dataset repo ID",
    )
    parser.add_argument(
        "--hf-token",
        help="HuggingFace token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of files per batch",
    )
    args = parser.parse_args()

    push_audio_batches(
        args.audio_dir,
        args.hf_repo,
        hf_token=args.hf_token,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
