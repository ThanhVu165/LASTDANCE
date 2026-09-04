"""Extract 16 kHz mono FLAC audio for one video or an entire collection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from offline.artifacts import sha256_file
from offline.asr_audio import default_videos_root, extract_audio_flac
from offline.checkpoints import CheckpointStore
from offline.config import DataLayout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--video")
    parser.add_argument("--collection", action="store_true", help="process all MP4 files")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, dest="shards", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.shards < 1 or not 0 <= args.shard_index < args.shards:
        raise ValueError("invalid shard selection")
    layout = DataLayout(args.data_root.resolve()) if args.data_root else DataLayout.from_environment()
    if args.videos_root is not None:
        root = args.videos_root.resolve()
    elif args.video and Path(args.video).is_absolute():
        root = Path(args.video).resolve().parent
    else:
        root = default_videos_root().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"videos root does not exist: {root}")
    if args.video:
        source = Path(args.video)
        if not source.is_absolute():
            source = root / source
        sources = [source.resolve()]
    else:
        sources = sorted(path.relative_to(root) for path in root.rglob("*.mp4"))
    if args.video:
        selected = sources
    else:
        selected = [
            path for index, path in enumerate(sources)
            if index % args.shards == args.shard_index
        ]
    output = (args.output_dir or layout.root / "audio").resolve()
    state = CheckpointStore(args.state or output / "collection.checkpoint.json")
    for item in selected:
        source = item if args.video else (root / item).resolve()
        video_id = source.stem
        signature = hashlib.sha256(f"{source}:{sha256_file(source)}".encode()).hexdigest()
        existing = state.get(video_id, "asr-audio")
        destination = output / f"{video_id}.flac"
        if existing and existing.finished and destination.is_file():
            continue
        state.update(video_id=video_id, stage="asr-audio", signature=signature, next_index=0, total=1)
        extraction = extract_audio_flac(source, destination, videos_root=root, ffmpeg_binary=args.ffmpeg, ffprobe_binary=args.ffprobe)
        state.update(video_id=video_id, stage="asr-audio", signature=signature, next_index=1, total=1)
        print(json.dumps({"video_id": video_id, "audio": str(extraction.output_audio), "duration_seconds": extraction.duration_seconds}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
