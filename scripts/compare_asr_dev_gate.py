"""Validate and compare completed Whisper/PhoWhisper Dev Gate artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from offline.asr_artifacts import AsrTranscriptRecord
from offline.asr_evaluation import word_error_rate
from scripts.extract_asr_audio import DEFAULT_DEV_VIDEO_IDS, read_video_ids


def _load_manifest(path: Path, expected_ids: list[str]) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    runtime = payload.get("runtime")
    if (
        payload.get("schema_version") != 1
        or payload.get("complete") is not True
        or payload.get("checkpoint_resume_verified") is not True
        or payload.get("video_ids") != expected_ids
        or payload.get("record_count") != len(expected_ids)
        or not isinstance(runtime, dict)
        or "T4" not in str(runtime.get("gpu_name", ""))
        or type(runtime.get("peak_cuda_memory_bytes")) is not int
        or int(runtime["peak_cuda_memory_bytes"]) <= 0
    ):
        raise RuntimeError(f"ASR Dev Gate manifest failed contract: {path}")
    return payload


def _load_records(manifest_path: Path, expected_ids: list[str]) -> dict[str, AsrTranscriptRecord]:
    records_dir = Path(manifest_path).parent / "records"
    records = {
        video_id: AsrTranscriptRecord.model_validate_json(
            (records_dir / f"{video_id}.json").read_text(encoding="utf-8")
        )
        for video_id in expected_ids
    }
    return records


def _load_ground_truth(path: Path) -> list[dict[str, object]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(samples, list) or not samples:
        raise RuntimeError("ground truth must contain a non-empty samples list")
    for sample in samples:
        if (
            not isinstance(sample, dict)
            or not str(sample.get("video_id", "")).strip()
            or float(sample.get("start_time", -1)) < 0
            or float(sample.get("end_time", -1)) < float(sample.get("start_time", -1))
            or not str(sample.get("reference_text", "")).strip()
        ):
            raise RuntimeError("invalid optional ASR ground-truth sample")
    return samples


def _wer_for_records(
    records: dict[str, AsrTranscriptRecord], samples: list[dict[str, object]]
) -> dict[str, int | float]:
    references: list[str] = []
    hypotheses: list[str] = []
    for sample in samples:
        video_id = str(sample["video_id"])
        if video_id not in records:
            raise RuntimeError(f"ground-truth video is outside Dev Gate: {video_id}")
        start = float(sample["start_time"])
        end = float(sample["end_time"])
        hypothesis = " ".join(
            segment.transcribed_text
            for segment in records[video_id].segments
            if segment.end_time >= start and segment.start_time <= end
        )
        references.append(str(sample["reference_text"]))
        hypotheses.append(hypothesis)
    return word_error_rate(references, hypotheses)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--whisper-manifest", type=Path, required=True)
    parser.add_argument("--phowhisper-manifest", type=Path, required=True)
    parser.add_argument("--video-id-file", type=Path, default=DEFAULT_DEV_VIDEO_IDS)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    expected_ids = read_video_ids(args.video_id_file)
    manifest_paths = {
        "whisper_large_v3": args.whisper_manifest.resolve(),
        "phowhisper_large": args.phowhisper_manifest.resolve(),
    }
    report: dict[str, object] = {
        "schema_version": 1,
        "video_ids": expected_ids,
        "models": {},
        "wer": None,
        "manual_review_required": True,
        "production_model_selected": False,
    }
    records_by_model: dict[str, dict[str, AsrTranscriptRecord]] = {}
    for model_key, path in manifest_paths.items():
        manifest = _load_manifest(path, expected_ids)
        model = manifest["model"]
        runtime = manifest["runtime"]
        audio_seconds = float(manifest["audio_duration_seconds"])
        inference_seconds = float(manifest["inference_seconds"])
        report["models"][model_key] = {
            "model": model,
            "segment_count": manifest["segment_count"],
            "no_speech_records": manifest["no_speech_records"],
            "audio_duration_seconds": audio_seconds,
            "inference_seconds": inference_seconds,
            "real_time_factor": inference_seconds / audio_seconds,
            "peak_cuda_memory_bytes": runtime["peak_cuda_memory_bytes"],
            "gpu_name": runtime["gpu_name"],
            "runtime": runtime,
        }
        records_by_model[model_key] = _load_records(path, expected_ids)
    if args.ground_truth:
        samples = _load_ground_truth(args.ground_truth)
        report["wer"] = {
            model_key: _wer_for_records(records, samples)
            for model_key, records in records_by_model.items()
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f"{args.output.name}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(f"ASR Dev Gate comparison -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
