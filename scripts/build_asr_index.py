"""Build asr.sqlite, coverage CSV, and hash-bound state on local CPU."""

from __future__ import annotations

import argparse
from pathlib import Path

from offline.asr_alignment import load_aligned_jsonl, load_frames_by_video, load_transcript_records
from offline.asr_audio import load_inventory, validate_audio_artifact
from offline.asr_catalog import (
    build_asr_sqlite_atomic,
    derive_asr_coverage,
    validate_asr_sqlite,
    write_asr_state_atomic,
    write_coverage_csv_atomic,
)
from offline.config import DataLayout
from scripts.extract_asr_audio import read_video_ids


def _optional_ids(path: Path | None) -> set[str]:
    return set() if path is None else set(read_video_ids(path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--transcript-records", type=Path, required=True)
    parser.add_argument("--aligned-jsonl", type=Path, required=True)
    parser.add_argument("--video-id-file", type=Path, required=True)
    parser.add_argument("--verified-no-speech-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    layout = DataLayout(args.data_root.resolve()) if args.data_root else DataLayout.from_environment()
    video_ids = read_video_ids(args.video_id_file)
    selected = set(video_ids)
    inventory = {
        record.video_id: record
        for record in load_inventory((args.inventory or (layout.index / "inventory.json")).resolve())
        if record.video_id in selected
    }
    if set(inventory) != selected:
        raise RuntimeError("ASR index selection does not match inventory")
    audio_root = (args.audio_root or (layout.root / "asr" / "audio")).resolve()
    audio = {
        video_id: validate_audio_artifact(
            audio_root / "manifests" / f"{video_id}.json", audio_root=audio_root
        )
        for video_id in video_ids
    }
    transcript_rows = load_transcript_records(args.transcript_records.resolve())
    transcripts = {row.video_id: row for row in transcript_rows if row.video_id in selected}
    if len(transcripts) != len([row for row in transcript_rows if row.video_id in selected]):
        raise RuntimeError("duplicate ASR transcript video_id")
    segments = load_aligned_jsonl(args.aligned_jsonl.resolve())
    frames = load_frames_by_video(args.catalog.resolve())
    for segment in segments:
        valid = {frame.keyframe_uid for frame in frames.get(segment.video_id, [])}
        if segment.video_id not in selected or segment.keyframe_uid_nearest not in valid:
            raise RuntimeError("aligned ASR segment references a foreign/missing keyframe_uid")

    coverage = derive_asr_coverage(
        expected_video_ids=video_ids,
        inventory_has_audio={video_id: inventory[video_id].has_audio for video_id in video_ids},
        audio_artifacts=audio,
        transcripts=transcripts,
        aligned_segments=segments,
        verified_no_speech=_optional_ids(args.verified_no_speech_file),
    )
    output_dir = (args.output_dir or (layout.index / "asr")).resolve()
    sqlite_path = build_asr_sqlite_atomic(output_dir / "asr.sqlite", segments)
    if not validate_asr_sqlite(sqlite_path, expected_segments=segments):
        raise RuntimeError("new asr.sqlite failed post-publish validation")
    coverage_path = write_coverage_csv_atomic(output_dir / "asr_coverage.csv", coverage)
    state_path = write_asr_state_atomic(
        output_dir / "asr.state.json",
        sqlite_path=sqlite_path,
        coverage_path=coverage_path,
        segments=segments,
        coverage=coverage,
    )
    print(
        f"ASR index: {len(segments)} segments, "
        f"complete_videos={sum(row.complete for row in coverage)}/{len(coverage)} -> {state_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
