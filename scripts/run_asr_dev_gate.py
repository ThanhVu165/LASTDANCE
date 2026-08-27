"""Run one pinned Whisper candidate on the five-video Kaggle ASR Dev Gate."""

from __future__ import annotations

import argparse
from pathlib import Path

from offline.asr_models import (
    DEFAULT_ASR_MODEL_CONFIG,
    SUPPORTED_ASR_MODELS,
    create_asr_transcriber,
)
from offline.asr_transcription import IntentionalAsrInterruption, run_asr_transcription
from offline.config import DataLayout
from scripts.extract_asr_audio import DEFAULT_DEV_VIDEO_IDS, read_video_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=SUPPORTED_ASR_MODELS, required=True)
    parser.add_argument("--batch-id", default="dev-subset-5")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--video-id-file", type=Path, default=DEFAULT_DEV_VIDEO_IDS)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_ASR_MODEL_CONFIG)
    parser.add_argument("--stop-after-videos", type=int)
    parser.add_argument("--require-resume-verified", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    layout = DataLayout(args.data_root.resolve()) if args.data_root else DataLayout.from_environment()
    transcriber = create_asr_transcriber(
        args.model,
        config_path=args.model_config.resolve(),
        purpose="dev_gate",
        device="cuda",
    )
    try:
        result = run_asr_transcription(
            transcriber=transcriber,
            batch_id=args.batch_id,
            audio_root=(args.audio_root or (layout.root / "asr" / "audio")).resolve(),
            output_root=(
                args.output_root or (layout.index / "asr" / "transcripts")
            ).resolve(),
            video_ids=read_video_ids(args.video_id_file),
            stop_after_videos=args.stop_after_videos,
        )
    except IntentionalAsrInterruption as error:
        print(error)
        return 75
    finally:
        transcriber.release()
    if args.require_resume_verified and not result.checkpoint_resume_verified:
        raise RuntimeError("ASR Dev Gate has not demonstrated new-process resume")
    print(
        f"{args.model}: {result.completed_videos}/{result.total_videos} -> "
        f"{result.output_dir}; resume_verified={result.checkpoint_resume_verified}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
