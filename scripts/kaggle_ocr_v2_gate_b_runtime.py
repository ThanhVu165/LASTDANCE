"""Self-contained Kaggle T4 runtime for the OCR-v2 same-crop recognizer gate.

Inputs are immutable EasyOCR archives, a review bundle, and JPEG keyframes.  The
runtime never calls a paid API and never writes a production OCR artifact.
"""

from __future__ import annotations

import gc
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import tarfile
import time
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps


INPUT_ROOT = Path("/kaggle/input")
OUTPUT_ROOT = Path("/kaggle/working/ocr-v2-gate-b")
SAMPLE_SIZE = 120
BENCHMARK_REGIONS = 5000
PADDLE_BATCH_SIZE = int(globals().get("PADDLE_BATCH_SIZE", 128))
VIETOCR_BATCH_SIZE = int(globals().get("VIETOCR_BATCH_SIZE", 64))

EASYOCR_MODEL_ID = "easyocr_latin_g2_cached"
PADDLE_MODEL_ID = "latin_PP-OCRv5_mobile_rec"
VIETOCR_MODEL_ID = "vietocr_vgg_seq2seq"
CROP_SPEC_ID = "pil_quad_v1_pad08_edge"

PADDLE_MODEL = {
    "url": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/latin_PP-OCRv5_mobile_rec_infer.tar",
    "sha256": "b23105a6a1ea38e32a97c5a0ddc7e8a9bbf541d8e47421e2c99e9ccabe29509c",
    "bytes": 8202240,
}
VIETOCR_WEIGHT = {
    "url": "https://github.com/pbcquoc/vietocr/releases/download/v0.3.2/vgg-seq2seq.pth",
    "sha256": "0921503a41375a0584268e23ef3d414ea478a8fe8777865c7745d38f2d0bc5db",
    "bytes": 89575371,
}
VIETOCR_CONFIGS = {
    "base.yml": {
        "url": "https://raw.githubusercontent.com/pbcquoc/vietocr/fe8c3a7fc714aec57ab81cec844eb3adf0c1636c/config/base.yml",
        "sha256": "9c8283fadb950f06f5d3400475f80d5355700ff315c9c48b7875e6ea66647d1c",
    },
    "vgg-seq2seq.yml": {
        "url": "https://raw.githubusercontent.com/pbcquoc/vietocr/fe8c3a7fc714aec57ab81cec844eb3adf0c1636c/config/vgg-seq2seq.yml",
        "sha256": "0160ba8d442ae96f4c6095b92ac3521c59b83ce6eda9fd5459e8628a5586c3e8",
    },
}
ARCHIVE_PATTERN = re.compile(r"^ocr-production-batch-(0[1-9])-easyocr\.zip$")
EASYOCR_REQUIRED_MEMBERS = {
    "easyocr-frames.jsonl",
    "vintern-candidates.jsonl",
    "run-signature.json",
    "batch-manifest.json",
    "SHA256SUMS",
}
EASYOCR_ALLOWED_MEMBERS = EASYOCR_REQUIRED_MEMBERS | {"errors-history.jsonl"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_names(source: Path) -> set[str]:
    if source.is_dir():
        return {path.name for path in source.iterdir() if path.is_file()}
    with zipfile.ZipFile(source) as archive:
        if archive.testzip() is not None:
            raise RuntimeError(f"archive CRC failure: {source}")
        names = archive.namelist()
        if len(names) != len(set(names)) or any(name != Path(name).name for name in names):
            raise RuntimeError(f"unsafe/duplicate archive members: {source}")
        return set(names)


def source_read(source: Path, name: str) -> bytes:
    if source.is_dir():
        path = source / name
        if not path.is_file():
            raise RuntimeError(f"missing {name} in extracted archive {source}")
        return path.read_bytes()
    with zipfile.ZipFile(source) as archive:
        return archive.read(name)


def source_sha256(source: Path) -> str:
    if source.is_file():
        return sha256_file(source)
    digest = hashlib.sha256()
    files = sorted(path for path in source.iterdir() if path.is_file())
    if not files:
        raise RuntimeError(f"empty extracted archive: {source}")
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def parse_checksums(payload: bytes) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for raw_line in payload.decode("ascii").splitlines():
        digest, name = raw_line.split(maxsplit=1)
        name = name.strip()
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or name in checksums:
            raise RuntimeError("invalid SHA256SUMS")
        checksums[name] = digest
    return checksums


def discover_archive_evidence() -> tuple[dict[str, Path], dict[str, Any]]:
    archives: dict[str, Path] = {}
    manifest_hashes: list[str] = []
    total_regions = 0
    candidates = {
        path.resolve()
        for path in INPUT_ROOT.rglob("ocr-production-batch-*-easyocr.zip")
        if path.is_file() and ARCHIVE_PATTERN.fullmatch(path.name)
    }
    for manifest_path in INPUT_ROOT.rglob("batch-manifest.json"):
        parent = manifest_path.parent.resolve()
        if EASYOCR_REQUIRED_MEMBERS <= source_names(parent):
            candidates.add(parent)
    for path in sorted(candidates):
        print("[1/8] ARCHIVE_VALIDATE", path, flush=True)
        names = source_names(path)
        if not EASYOCR_REQUIRED_MEMBERS <= names or names - EASYOCR_ALLOWED_MEMBERS:
            raise RuntimeError(f"unexpected EasyOCR archive members in {path}: {sorted(names)}")
        checksums = parse_checksums(source_read(path, "SHA256SUMS"))
        manifest_payload = source_read(path, "batch-manifest.json")
        manifest_hash = sha256_bytes(manifest_payload)
        if checksums.get("batch-manifest.json") != manifest_hash:
            raise RuntimeError(f"manifest checksum mismatch in {path}")
        manifest = json.loads(manifest_payload)
        batch_id = str(manifest.get("batch_id") or "")
        if not re.fullmatch(r"batch-0[1-9]", batch_id):
            raise RuntimeError(f"invalid EasyOCR batch identity in {path}")
        if path.is_file():
            match = ARCHIVE_PATTERN.fullmatch(path.name)
            if match is None or batch_id != f"batch-{match.group(1)}":
                raise RuntimeError(f"EasyOCR filename/manifest identity mismatch in {path}")
        if batch_id in archives:
            raise RuntimeError(f"duplicate EasyOCR archive for {batch_id}")
        if manifest.get("batch_id") != batch_id or manifest.get("tier") != "easyocr":
            raise RuntimeError(f"unexpected EasyOCR manifest identity in {path}")
        if int(manifest.get("model_calls", {}).get("vintern", -1)) != 0:
            raise RuntimeError(f"archive unexpectedly contains Vintern calls: {path}")
        total_regions += int(manifest["regions"])
        manifest_hashes.append(manifest_hash)
        archives[batch_id] = path.resolve()
    expected = {f"batch-{index:02d}" for index in range(1, 10)}
    complete = set(archives) == expected
    evidence = {
        "complete": complete,
        "archive_count": len(archives),
        "catalog_regions": total_regions if complete else None,
        "manifest_sha256s": sorted(manifest_hashes),
        "batch_ids": sorted(archives),
        "limitation": None if complete else "Attach all nine EasyOCR archives for an exact full-catalog ETA.",
    }
    return archives, evidence


def load_review_sample() -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    zip_matches = sorted(
        {path.resolve() for path in INPUT_ROOT.rglob("ocr-v2-review-bundle.zip") if path.is_file()}
    )
    if zip_matches:
        if len(zip_matches) != 1:
            raise RuntimeError(f"attach exactly one review bundle: {zip_matches}")
        bundle_path = zip_matches[0]
    else:
        directory_matches = sorted(
            {
                path.parent.resolve()
                for path in INPUT_ROOT.rglob("recognition-sample.jsonl")
                if (path.parent / "review-report.json").is_file()
            }
        )
        if len(directory_matches) != 1:
            raise RuntimeError(f"expected one extracted review bundle: {directory_matches}")
        bundle_path = directory_matches[0]
    names = source_names(bundle_path)
    required = {"recognition-sample.jsonl", "review-report.json"}
    if not required <= names:
        raise RuntimeError("review bundle is missing the recognition sample/report")
    sample_payload = source_read(bundle_path, "recognition-sample.jsonl")
    report = json.loads(source_read(bundle_path, "review-report.json"))
    expected_sha = report.get("artifacts", {}).get("recognition_sample_jsonl_sha256")
    if expected_sha != sha256_bytes(sample_payload):
        raise RuntimeError("review sample/report checksum mismatch")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(sample_payload.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        region_id = str(row.get("region_id") or "")
        if not region_id or region_id in seen:
            raise RuntimeError(f"blank/duplicate region_id in sample line {line_number}")
        if row.get("crop_spec_id") != CROP_SPEC_ID:
            raise RuntimeError(f"unknown crop contract in sample line {line_number}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(row.get("sample_row_sha256") or "")):
            raise RuntimeError(f"invalid immutable sample hash in line {line_number}")
        seen.add(region_id)
        rows.append(row)
    if len(rows) != SAMPLE_SIZE:
        raise RuntimeError(f"recognition sample must contain exactly {SAMPLE_SIZE} regions")
    print(
        "[2/8] REVIEW_SAMPLE_READY",
        {"source": str(bundle_path), "regions": len(rows)},
        flush=True,
    )
    return bundle_path, rows, report


def _float_list(value: Any) -> list[float]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list) or len(value) != 8:
        raise RuntimeError("bbox_px must contain eight values")
    values = [float(item) for item in value]
    if not all(math.isfinite(item) for item in values):
        raise RuntimeError("bbox_px contains NaN/Inf")
    return values


def load_batch_one_population(
    archive_path: Path, sample_rows: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    sample_by_region = {str(row["region_id"]): row for row in sample_rows}
    selected_videos = {str(row["video_id"]) for row in sample_rows}
    observed: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    checksums = parse_checksums(source_read(archive_path, "SHA256SUMS"))
    frames_payload = source_read(archive_path, "easyocr-frames.jsonl")
    if checksums.get("easyocr-frames.jsonl") != sha256_bytes(frames_payload):
        raise RuntimeError("Batch 01 EasyOCR JSONL checksum mismatch")
    scanned_frames = 0
    for line in io.StringIO(frames_payload.decode("utf-8")):
        if not line.strip():
            continue
        scanned_frames += 1
        if scanned_frames % 10000 == 0:
            print(
                "[2/8] BATCH01_SCAN_PROGRESS",
                {"frames": scanned_frames, "candidate_regions": len(candidates)},
                flush=True,
            )
        frame = json.loads(line)
        if str(frame.get("video_id")) not in selected_videos:
            continue
        for region in frame.get("regions") or []:
            region_id = str(region.get("region_id") or "")
            row = {
                "region_id": region_id,
                "keyframe_uid": int(frame["keyframe_uid"]),
                "video_id": str(frame["video_id"]),
                "shot_id": str(frame["shot_id"]),
                "source_image": str(frame["source_image"]),
                "bbox_px": _float_list(region.get("bbox_px")),
                "easyocr_text": str(region.get("easyocr_text") or ""),
                "easyocr_confidence": float(region.get("easyocr_confidence") or 0.0),
            }
            candidates.append(row)
            if region_id in sample_by_region:
                if region_id in observed:
                    raise RuntimeError(f"duplicate sample region in EasyOCR archive: {region_id}")
                observed[region_id] = row
    if set(observed) != set(sample_by_region):
        raise RuntimeError("sample region coverage differs from the immutable EasyOCR archive")
    for region_id, sample in sample_by_region.items():
        source = observed[region_id]
        if (
            int(sample["keyframe_uid"]) != source["keyframe_uid"]
            or str(sample["video_id"]) != source["video_id"]
            or str(sample["shot_id"]) != source["shot_id"]
            or _float_list(sample["bbox_px"]) != source["bbox_px"]
            or str(sample.get("easyocr_text") or "") != source["easyocr_text"]
            or abs(float(sample.get("easyocr_confidence") or 0.0) - source["easyocr_confidence"]) > 1e-12
        ):
            raise RuntimeError(f"sample metadata drift for region_id={region_id}")
    if len(candidates) < BENCHMARK_REGIONS:
        raise RuntimeError(
            f"selected videos expose only {len(candidates)} regions; need {BENCHMARK_REGIONS}"
        )
    print(
        "[2/8] BATCH01_SCAN_DONE",
        {
            "frames": scanned_frames,
            "sample_regions_found": len(observed),
            "benchmark_population": len(candidates),
        },
        flush=True,
    )
    return observed, candidates


def find_video_directory(video_id: str) -> Path:
    matches = sorted(
        {
            path.resolve()
            for path in INPUT_ROOT.rglob(video_id)
            if path.is_dir() and any(path.glob("*.jpg"))
        }
    )
    if len(matches) != 1:
        raise RuntimeError(f"expected one JPEG directory for {video_id}: {matches}")
    return matches[0]


def _edge_pad(image: Image.Image, border: int) -> Image.Image:
    if border <= 0:
        return image
    width, height = image.size
    padded = Image.new(image.mode, (width + 2 * border, height + 2 * border))
    padded.paste(image, (border, border))
    padded.paste(image.crop((0, 0, width, 1)).resize((width, border)), (border, 0))
    padded.paste(image.crop((0, height - 1, width, height)).resize((width, border)), (border, border + height))
    padded.paste(image.crop((0, 0, 1, height)).resize((border, height)), (0, border))
    padded.paste(image.crop((width - 1, 0, width, height)).resize((border, height)), (border + width, border))
    padded.paste(Image.new(image.mode, (border, border), image.getpixel((0, 0))), (0, 0))
    padded.paste(Image.new(image.mode, (border, border), image.getpixel((width - 1, 0))), (border + width, 0))
    padded.paste(Image.new(image.mode, (border, border), image.getpixel((0, height - 1))), (0, border + height))
    padded.paste(Image.new(image.mode, (border, border), image.getpixel((width - 1, height - 1))), (border + width, border + height))
    return padded


def rectify_region_crop(image: Image.Image, bbox_px: list[float]) -> Image.Image:
    values = _float_list(bbox_px)
    points = [(values[index], values[index + 1]) for index in range(0, 8, 2)]

    def distance(left: tuple[float, float], right: tuple[float, float]) -> float:
        return math.hypot(left[0] - right[0], left[1] - right[1])

    width = max(2, round(max(distance(points[0], points[1]), distance(points[3], points[2]))))
    height = max(2, round(max(distance(points[0], points[3]), distance(points[1], points[2]))))
    quad = (
        points[0][0], points[0][1], points[3][0], points[3][1],
        points[2][0], points[2][1], points[1][0], points[1][1],
    )
    crop = image.convert("RGB").transform(
        (width, height), Image.Transform.QUAD, quad, resample=Image.Resampling.BICUBIC
    )
    return _edge_pad(crop, max(1, round(height * 0.08)))


def materialize_crops(rows: list[dict[str, Any]], *, label: str) -> list[Image.Image]:
    video_dirs = {video_id: find_video_directory(video_id) for video_id in {row["video_id"] for row in rows}}
    grouped: dict[Path, list[tuple[int, dict[str, Any]]]] = {}
    for index, row in enumerate(rows):
        image_path = video_dirs[row["video_id"]] / Path(row["source_image"]).name
        if not image_path.is_file():
            raise FileNotFoundError(f"missing sample keyframe: {image_path}")
        grouped.setdefault(image_path, []).append((index, row))
    crops: list[Image.Image | None] = [None] * len(rows)
    total_images = len(grouped)
    for image_index, (image_path, entries) in enumerate(grouped.items(), start=1):
        with Image.open(image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        for index, row in entries:
            crops[index] = rectify_region_crop(image, row["bbox_px"])
        if image_index % 250 == 0 or image_index == total_images:
            print(
                "[3/8] CROP_PROGRESS",
                {"set": label, "images": image_index, "total_images": total_images},
                flush=True,
            )
    if any(crop is None for crop in crops):
        raise RuntimeError("internal crop materialization coverage failure")
    return [crop for crop in crops if crop is not None]


def download_verified(spec: dict[str, Any], destination: Path) -> Path:
    if not destination.exists() or sha256_file(destination) != spec["sha256"]:
        print(
            "MODEL_DOWNLOAD_START",
            {"file": destination.name, "expected_bytes": int(spec["bytes"])},
            flush=True,
        )
        urllib.request.urlretrieve(spec["url"], destination)
    if destination.stat().st_size != int(spec["bytes"]) or sha256_file(destination) != spec["sha256"]:
        raise RuntimeError(f"download checksum/size mismatch: {destination.name}")
    print(
        "MODEL_ARTIFACT_READY",
        {"file": destination.name, "bytes": destination.stat().st_size},
        flush=True,
    )
    return destination


def prepare_paddle_model() -> Path:
    archive_path = download_verified(PADDLE_MODEL, OUTPUT_ROOT / "paddle-model.tar")
    extract_root = OUTPUT_ROOT / "paddle-model"
    extract_root.mkdir(exist_ok=True)
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        if not members or any(
            member.islnk()
            or member.issym()
            or Path(member.name).is_absolute()
            or ".." in Path(member.name).parts
            for member in members
        ):
            raise RuntimeError("unsafe Paddle model archive")
        archive.extractall(extract_root, members=members, filter="data")
    model_dirs = sorted({path.parent for path in extract_root.rglob("inference.json")})
    if len(model_dirs) != 1 or not (model_dirs[0] / "inference.pdiparams").is_file():
        raise RuntimeError("unexpected Paddle inference archive layout")
    return model_dirs[0]


def prepare_vietocr_model() -> tuple[Path, dict[str, Any]]:
    import yaml

    weight_path = download_verified(VIETOCR_WEIGHT, OUTPUT_ROOT / "vgg-seq2seq.pth")
    config_values: dict[str, dict[str, Any]] = {}
    for name, spec in VIETOCR_CONFIGS.items():
        path = OUTPUT_ROOT / name
        if not path.exists() or sha256_file(path) != spec["sha256"]:
            urllib.request.urlretrieve(spec["url"], path)
        if sha256_file(path) != spec["sha256"]:
            raise RuntimeError(f"VietOCR config checksum mismatch: {name}")
        config_values[name] = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = dict(config_values["base.yml"])
    config.update(config_values["vgg-seq2seq.yml"])
    config["device"] = "cuda:0"
    config["weights"] = str(weight_path)
    config["predictor"] = {"beamsearch": False}
    config["cnn"] = {**config["cnn"], "pretrained": False}
    return weight_path, config


def run_paddle(crops: list[Image.Image], benchmark_crops: list[Image.Image]) -> tuple[list[tuple[str, float | None]], dict[str, Any]]:
    import paddle
    from paddleocr import TextRecognition

    print("[5/8] PADDLE_PREPARE_START", flush=True)
    model_dir = prepare_paddle_model()
    paddle.device.set_device("gpu:0")
    init_started = time.perf_counter()
    print("[5/8] PADDLE_MODEL_INIT_START", {"model_dir": str(model_dir)}, flush=True)
    model = TextRecognition(
        model_name=PADDLE_MODEL_ID,
        model_dir=str(model_dir),
        device="gpu:0",
    )
    print(
        "[5/8] PADDLE_MODEL_INIT_DONE",
        {"seconds": round(time.perf_counter() - init_started, 3)},
        flush=True,
    )

    def parse_result(result: Any) -> tuple[str, float | None]:
        payload = result.json
        if callable(payload):
            payload = payload()
        if isinstance(payload, str):
            payload = json.loads(payload)
        if not isinstance(payload, dict):
            raise RuntimeError("Paddle result is not a JSON object")
        value = payload.get("res", payload)
        text = str(value.get("rec_text") or "")
        raw_score = value.get("rec_score")
        score = float(raw_score) if raw_score is not None else None
        return text, score

    def predict(values: list[Image.Image], *, log_prefix: str) -> list[tuple[str, float | None]]:
        output: list[tuple[str, float | None]] = []
        next_report = 1000
        for start in range(0, len(values), PADDLE_BATCH_SIZE):
            arrays = [np.asarray(image, dtype=np.uint8)[:, :, ::-1] for image in values[start : start + PADDLE_BATCH_SIZE]]
            rows = list(model.predict(input=arrays, batch_size=len(arrays)))
            if len(rows) != len(arrays):
                raise RuntimeError("Paddle recognizer returned incomplete output")
            output.extend(parse_result(row) for row in rows)
            if len(output) >= next_report or len(output) == len(values):
                print(log_prefix, len(output), "/", len(values), flush=True)
                while next_report <= len(output):
                    next_report += 1000
        return output

    predict(benchmark_crops[: min(16, len(benchmark_crops))], log_prefix="PADDLE_WARMUP")
    sample_output = predict(crops, log_prefix="PADDLE_SAMPLE")
    started = time.perf_counter()
    benchmark_output = predict(benchmark_crops, log_prefix="PADDLE_BENCHMARK")
    elapsed = time.perf_counter() - started
    if len(benchmark_output) != BENCHMARK_REGIONS:
        raise RuntimeError("Paddle throughput output count mismatch")
    try:
        peak_vram_mb = float(paddle.device.cuda.max_memory_allocated()) / (1024 * 1024)
    except Exception:
        peak_vram_mb = None
    del model, benchmark_output
    gc.collect()
    try:
        paddle.device.cuda.empty_cache()
    except Exception:
        pass
    return sample_output, {
        "benchmark_regions": BENCHMARK_REGIONS,
        "elapsed_seconds": elapsed,
        "regions_per_second": BENCHMARK_REGIONS / elapsed,
        "error_count": 0,
        "peak_vram_mb": peak_vram_mb,
    }


def run_vietocr(crops: list[Image.Image], benchmark_crops: list[Image.Image]) -> tuple[list[tuple[str, float | None]], dict[str, Any]]:
    import torch
    from vietocr.tool.predictor import Predictor

    print("[4/8] VIETOCR_PREPARE_START", flush=True)
    _, config = prepare_vietocr_model()
    init_started = time.perf_counter()
    print("[4/8] VIETOCR_MODEL_INIT_START", flush=True)
    predictor = Predictor(config)
    print(
        "[4/8] VIETOCR_MODEL_INIT_DONE",
        {"seconds": round(time.perf_counter() - init_started, 3)},
        flush=True,
    )

    def predict(values: list[Image.Image], *, log_prefix: str) -> list[tuple[str, float | None]]:
        output: list[tuple[str, float | None]] = []
        next_report = 1000
        for start in range(0, len(values), VIETOCR_BATCH_SIZE):
            texts, probabilities = predictor.predict_batch(
                values[start : start + VIETOCR_BATCH_SIZE], return_prob=True
            )
            if len(texts) != len(probabilities):
                raise RuntimeError("VietOCR recognizer returned incomplete output")
            for text, probability in zip(texts, probabilities):
                score = float(probability)
                output.append((str(text), score if math.isfinite(score) else None))
            if len(output) >= next_report or len(output) == len(values):
                print(log_prefix, len(output), "/", len(values), flush=True)
                while next_report <= len(output):
                    next_report += 1000
        return output

    predict(benchmark_crops[: min(16, len(benchmark_crops))], log_prefix="VIETOCR_WARMUP")
    torch.cuda.reset_peak_memory_stats(0)
    sample_output = predict(crops, log_prefix="VIETOCR_SAMPLE")
    started = time.perf_counter()
    benchmark_output = predict(benchmark_crops, log_prefix="VIETOCR_BENCHMARK")
    elapsed = time.perf_counter() - started
    if len(benchmark_output) != BENCHMARK_REGIONS:
        raise RuntimeError("VietOCR throughput output count mismatch")
    peak_vram_mb = float(torch.cuda.max_memory_allocated(0)) / (1024 * 1024)
    del predictor, benchmark_output
    gc.collect()
    torch.cuda.empty_cache()
    return sample_output, {
        "benchmark_regions": BENCHMARK_REGIONS,
        "elapsed_seconds": elapsed,
        "regions_per_second": BENCHMARK_REGIONS / elapsed,
        "error_count": 0,
        "peak_vram_mb": peak_vram_mb,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = Path(f"{path}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


if PADDLE_BATCH_SIZE <= 0 or VIETOCR_BATCH_SIZE <= 0:
    raise ValueError("recognizer batch sizes must be positive")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)

import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("enable exactly one Kaggle GPU")
if "T4" not in torch.cuda.get_device_name(0).upper():
    raise RuntimeError(f"this gate is calibrated for T4, got {torch.cuda.get_device_name(0)}")

print("[1/8] INPUT_DISCOVERY_START", {"root": str(INPUT_ROOT)}, flush=True)
archives, catalog_evidence = discover_archive_evidence()
if "batch-01" not in archives:
    raise RuntimeError("attach the immutable Batch 01 EasyOCR archive")
bundle_path, sample_rows, review_report = load_review_sample()
sample_observed, population = load_batch_one_population(archives["batch-01"], sample_rows)
ordered_sample_sources = [sample_observed[str(row["region_id"])] for row in sample_rows]
benchmark_sources = sorted(
    population,
    key=lambda row: hashlib.sha256(
        f"ocr-v2-throughput-v1|{row['region_id']}".encode("utf-8")
    ).hexdigest(),
)[:BENCHMARK_REGIONS]

print("[3/8] CROP_MATERIALIZATION_START", {"sample": len(sample_rows), "benchmark": BENCHMARK_REGIONS}, flush=True)
sample_crops = materialize_crops(ordered_sample_sources, label="sample")
benchmark_crops = materialize_crops(benchmark_sources, label="benchmark")
print("[3/8] CROP_MATERIALIZATION_DONE", flush=True)

vietocr_output, vietocr_runtime = run_vietocr(sample_crops, benchmark_crops)
paddle_output, paddle_runtime = run_paddle(sample_crops, benchmark_crops)

print("[6/8] RESULT_ASSEMBLY_START", flush=True)
rows: list[dict[str, Any]] = []
for index, sample in enumerate(sample_rows):
    common = {
        "region_id": sample["region_id"],
        "sample_row_sha256": sample["sample_row_sha256"],
        "keyframe_uid": int(sample["keyframe_uid"]),
        "video_id": sample["video_id"],
        "status": "success",
        "error": None,
    }
    rows.append(
        {
            **common,
            "model_id": EASYOCR_MODEL_ID,
            "text": sample_observed[str(sample["region_id"])]["easyocr_text"],
            "confidence": sample_observed[str(sample["region_id"])]["easyocr_confidence"],
            "provenance": "immutable_batch_01_cache",
        }
    )
    rows.append(
        {
            **common,
            "model_id": PADDLE_MODEL_ID,
            "text": paddle_output[index][0],
            "confidence": paddle_output[index][1],
            "provenance": "gate_b_inference",
        }
    )
    rows.append(
        {
            **common,
            "model_id": VIETOCR_MODEL_ID,
            "text": vietocr_output[index][0],
            "confidence": vietocr_output[index][1],
            "provenance": "gate_b_inference",
        }
    )
if len(rows) != SAMPLE_SIZE * 3 or len({(row["region_id"], row["model_id"]) for row in rows}) != len(rows):
    raise RuntimeError("final recognizer result coverage is not exact")

results_path = OUTPUT_ROOT / "recognizer-results.jsonl"
write_jsonl(results_path, rows)
print("[6/8] RESULT_ASSEMBLY_DONE", {"rows": len(rows)}, flush=True)
print("[7/8] REPORT_AND_ARCHIVE_START", flush=True)
runtime_report = {
    "schema_version": 1,
    "created_utc": utc_now(),
    "gate": "B_recognizer_same_crop",
    "decision": "PENDING_HUMAN_GROUND_TRUTH_EVALUATION",
    "gpu": torch.cuda.get_device_name(0),
    "sample_regions": SAMPLE_SIZE,
    "sample_jsonl_sha256": review_report["artifacts"]["recognition_sample_jsonl_sha256"],
    "review_bundle_sha256": source_sha256(bundle_path),
    "batch_01_archive_sha256": source_sha256(archives["batch-01"]),
    "catalog_evidence": catalog_evidence,
    "models": {
        PADDLE_MODEL_ID: paddle_runtime,
        VIETOCR_MODEL_ID: vietocr_runtime,
    },
    "model_artifacts": {
        "paddle": PADDLE_MODEL,
        "vietocr_weight": VIETOCR_WEIGHT,
        "vietocr_configs": VIETOCR_CONFIGS,
    },
    "model_calls": {"easyocr": 0, "paddleocr": SAMPLE_SIZE + BENCHMARK_REGIONS, "vietocr": SAMPLE_SIZE + BENCHMARK_REGIONS, "vintern": 0, "gemini": 0},
    "ready_for_final_gate_b": bool(catalog_evidence["complete"]),
    "limitations": [
        "Throughput excludes JPEG loading and crop rectification so it measures recognition only.",
        "No model agreement is treated as ground truth; final scoring requires the human CSV.",
    ],
}
runtime_path = OUTPUT_ROOT / "runtime-report.json"
runtime_path.write_text(json.dumps(runtime_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
checksums_path = OUTPUT_ROOT / "SHA256SUMS"
checksums_path.write_text(
    f"{sha256_file(results_path)}  {results_path.name}\n{sha256_file(runtime_path)}  {runtime_path.name}\n",
    encoding="ascii",
)
output_zip = Path("/kaggle/working/ocr-v2-gate-b-results.zip")
with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in (results_path, runtime_path, checksums_path):
        archive.write(path, path.name)
with zipfile.ZipFile(output_zip) as archive:
    if archive.testzip() is not None or set(archive.namelist()) != {"recognizer-results.jsonl", "runtime-report.json", "SHA256SUMS"}:
        raise RuntimeError("final Gate B archive validation failed")
print("[8/8] OCR_V2_GATE_B_COMPLETE", output_zip, sha256_file(output_zip), flush=True)
