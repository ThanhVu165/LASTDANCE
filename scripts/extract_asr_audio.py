"""Extract or resume PCM s16le 16 kHz mono WAV artifacts on local CPU."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from offline.asr_audio import (
    AudioStatus,
    extract_audio_artifact,
    ffmpeg_version,
    load_inventory,
    select_inventory_records,
)
from offline.config import DataLayout


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEV_VIDEO_IDS = REPOSITORY_ROOT / "configs" / "shot_parity_dev_subset_5.txt"


def read_video_ids(path: Path) -> list[str]:
    values: list[str] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line or line.lstrip().startswith("#"):
            continue
        if line.strip() != line or any(char.isspace() for char in line):
            raise ValueError(f"non-canonical video_id at {path}:{line_number}")
        values.append(line)
    if not values or len(set(values)) != len(values):
        raise ValueError("video ID file must contain unique IDs")
    return values


def _write_report(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--video-id-file", type=Path, default=DEFAULT_DEV_VIDEO_IDS)
    parser.add_argument("--ffmpeg", default=os.environ.get("AIC_FFMPEG", "ffmpeg"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    layout = DataLayout(args.data_root.resolve()) if args.data_root else DataLayout.from_environment()
    inventory_path = (args.inventory or (layout.index / "inventory.json")).resolve()
    audio_root = (args.audio_root or (layout.root / "asr" / "audio")).resolve()
    video_ids = read_video_ids(args.video_id_file)
    records = select_inventory_records(load_inventory(inventory_path), set(video_ids))
    ordered = {record.video_id: record for record in records}
    records = [ordered[video_id] for video_id in video_ids]
    version = ffmpeg_version(args.ffmpeg)
    artifacts = [
        extract_audio_artifact(
            record,
            data_root=layout.root,
            audio_root=audio_root,
            ffmpeg_binary=args.ffmpeg,
            ffmpeg_version_text=version,
        )
        for record in records
    ]
    ready = [artifact for artifact in artifacts if artifact.status == AudioStatus.READY]
    report = {
        "schema_version": 1,
        "video_ids": video_ids,
        "video_count": len(artifacts),
        "ready_count": len(ready),
        "no_audio_count": len(artifacts) - len(ready),
        "ffmpeg_version": version,
        "wav_size_bytes": sum(int(artifact.wav_size_bytes or 0) for artifact in ready),
        "audio_duration_seconds": sum(
            float(artifact.wav_duration_seconds or 0) for artifact in ready
        ),
        "mean_megabytes_per_minute": (
            sum(float(artifact.megabytes_per_minute or 0) for artifact in ready) / len(ready)
            if ready
            else 0.0
        ),
        "artifacts": [artifact.model_dump(mode="json") for artifact in artifacts],
    }
    report_path = audio_root / "dev-gate-audio-report.json"
    _write_report(report_path, report)
    print(
        f"ASR audio: ready={len(ready)}/{len(artifacts)}, "
        f"bytes={report['wav_size_bytes']} -> {audio_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
