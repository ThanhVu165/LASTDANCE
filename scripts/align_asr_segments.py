"""Align completed ASR transcript records to keyframe_uid values in frames.csv."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from offline.artifacts import sha256_file
from offline.asr_alignment import (
    align_transcript_records,
    load_frames_by_video,
    load_transcript_records,
    write_aligned_jsonl_atomic,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--transcript-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    records = load_transcript_records(args.transcript_records.resolve())
    frames = load_frames_by_video(args.catalog.resolve())
    segments = align_transcript_records(records, frames_by_video=frames)
    output = write_aligned_jsonl_atomic(args.output.resolve(), segments)
    state = {
        "schema_version": 1,
        "complete": True,
        "catalog_sha256": sha256_file(args.catalog.resolve()),
        "video_count": len(records),
        "segment_count": len(segments),
        "aligned_jsonl_sha256": sha256_file(output),
    }
    state_path = output.with_name(f"{output.name}.state.json")
    temporary = state_path.with_name(f"{state_path.name}.tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(state_path)
    print(f"ASR alignment: {len(segments)} segments -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
