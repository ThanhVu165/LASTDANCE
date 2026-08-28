"""Self-contained Kaggle runtime embedded into the production EasyOCR notebook."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
import unicodedata
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch


WORKER_SLOT = int(globals().get("WORKER_SLOT", 1))
PUBLISH_TO_HF = bool(globals().get("PUBLISH_TO_HF", True))
INTERRUPT_AFTER_NEW_FRAMES = int(globals().get("INTERRUPT_AFTER_NEW_FRAMES", 0))
CHECKPOINT_EVERY = int(globals().get("CHECKPOINT_EVERY", 250))
PROGRESS_EVERY = int(globals().get("PROGRESS_EVERY", 25))
HF_REPO_ID = str(
    globals().get("HF_REPO_ID", "MinhThuw0103/lastdance-visual-embeddings")
)

CATALOG_SHA256 = "ee9693e75580527a0a257e9ba003984e105b059b716922c03c7a0b72b1508a37"
BATCH_MAPPING_SHA256 = "e7e519e5fe3e47c3e487bfe0522c09c3f0bae6c7f67dff2d31168aead0b911d2"
NOTEBOOK_CONTRACT = "ocr-production-easyocr-v1"
SELECTED_CRAFT = {
    "config_id": "recall_current",
    "text_threshold": 0.6,
    "low_text": 0.3,
    "link_threshold": 0.3,
}
ROUTER_POLICY = {
    "schema_version": 1,
    "hard_confidence_threshold": 0.4,
    "selective_confidence_ceiling": 0.6,
    "escalate_region_mixed": True,
    "escalate_ambiguous_glyphs": True,
    "noisy_character_ratio_threshold": 0.34,
    "noisy_character_min_length": 3,
    "max_candidate_fraction": 0.4,
}
EASYOCR_WEIGHTS = {
    "craft_mlt_25k": {
        "url": "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip",
        "zip_sha256": "8dc6a1c703a89ed56308ef742d26ebd45c656248cbbbda6e7fe60e569f873e65",
        "weight_name": "craft_mlt_25k.pth",
        "weight_sha256": "4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17",
    },
    "latin_g2": {
        "url": "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/latin_g2.zip",
        "zip_sha256": "29f1920c493378da65a59793fb70e7e190504662b6bed57ab26f4067eb5f3769",
        "weight_name": "latin_g2.pth",
        "weight_sha256": "aaa95be1c4a9cb3496879bed7c520886ce1164f89e026f0c54488394e74e8c55",
    },
}
BATCHES = {
    "batch-01": {
        "keyframes_batch": "keyframes-batch-01",
        "video_count": 100,
        "keyframe_count": 59836,
        "first_video_id": "L21_V001",
        "last_video_id": "L24_V016",
        "uid_set_sha256": "ea3c0e39b65298e496472fa971b6ea421d93ab78df057980677a0a7fbc213049",
        "video_set_sha256": "f232ce735ca9b9346fe156baf550d76691a4f03cbae77af22749b925e4311da1",
    },
    "batch-02": {
        "keyframes_batch": "keyframes-batch-02",
        "video_count": 100,
        "keyframe_count": 29305,
        "first_video_id": "L24_V017",
        "last_video_id": "L25_V072",
        "uid_set_sha256": "1c8a0097fcc28f77bfb9278f89930fb181d80072b0a063f32c414ff8e5246fab",
        "video_set_sha256": "98e248a68bc9fa0b40d0a9ed4757acd0696fab1e0515b9626d9b343fad549a95",
    },
    "batch-03": {
        "keyframes_batch": "keyframes-batch-03",
        "video_count": 100,
        "keyframe_count": 26972,
        "first_video_id": "L25_V073",
        "last_video_id": "L26_V084",
        "uid_set_sha256": "270480578d4db21465bb73bf505655b963ac1d96c4df553724920527bc559900",
        "video_set_sha256": "437c75fcc31d943af71b80ab7ed50ef99ea01b0f482a3459569afe73e084cc34",
    },
    "batch-04": {
        "keyframes_batch": "keyframes-batch-04",
        "video_count": 100,
        "keyframe_count": 27521,
        "first_video_id": "L26_V085",
        "last_video_id": "L26_V184",
        "uid_set_sha256": "b1b47804e57983e4714be6f7c1b9a760ea560944ba73a7f5dab4040cc14171f6",
        "video_set_sha256": "45059387ca057ef98794e288290591b17e0797e65417e9454f8943e1c8391270",
    },
    "batch-05": {
        "keyframes_batch": "keyframes-batch-05",
        "video_count": 100,
        "keyframe_count": 31245,
        "first_video_id": "L26_V185",
        "last_video_id": "L26_V284",
        "uid_set_sha256": "673a5bbc2d76f4a3ebd876f2356d9a5c28dfef1151c09ebc6af4fb8ed1f7a358",
        "video_set_sha256": "8bfc42caf671b3dbf83b0246d2eea638d79b1fcc2cd5398c075711b4b4d26329",
    },
    "batch-06": {
        "keyframes_batch": "keyframes-batch-06",
        "video_count": 100,
        "keyframe_count": 31407,
        "first_video_id": "L26_V285",
        "last_video_id": "L26_V384",
        "uid_set_sha256": "d1082481a0cd031e5a1139ac4a6cd250e146237f692d46eed71fce2084a9235b",
        "video_set_sha256": "05276efa9112d39ca091f4214452f4f4547b0d023fab5a40ddb3bbd74e697ad3",
    },
    "batch-07": {
        "keyframes_batch": "keyframes-batch-07",
        "video_count": 100,
        "keyframe_count": 31642,
        "first_video_id": "L26_V385",
        "last_video_id": "L26_V485",
        "uid_set_sha256": "f3b26c7bd90f2396cf4d944fb403f4c201e6acb3bc6e406cfc97da23ce74f230",
        "video_set_sha256": "091d1aedbdb91fe5f83eee5e3f40547b5c76ed943a7d189920dadbecc80abd35",
    },
    "batch-08": {
        "keyframes_batch": "keyframes-batch-08",
        "video_count": 100,
        "keyframe_count": 45460,
        "first_video_id": "L26_V486",
        "last_video_id": "L30_V023",
        "uid_set_sha256": "0e72e26396dd2776e88d32f2a36a449243849e11a16270ac3a3756076c61d62a",
        "video_set_sha256": "3f778682a425c76224c651586e1a08b556eaa7f34c4ae1b272a46b309502facb",
    },
    "batch-09": {
        "keyframes_batch": "keyframes-batch-09",
        "video_count": 73,
        "keyframe_count": 9948,
        "first_video_id": "L30_V024",
        "last_video_id": "L30_V096",
        "uid_set_sha256": "31bba461f43fdc384b32012d877f3792a94fa51e65d66f3d68d6cca68656b06c",
        "video_set_sha256": "7e148550c4c2466e9ba1f4011d7dfe6c2183bc4a0cf1573d484e7fbf2bc3ea50",
    },
}
WORKER_BATCHES = {
    1: ("batch-01", "batch-09"),
    2: ("batch-02", "batch-03", "batch-04"),
    3: ("batch-05", "batch-08"),
    4: ("batch-06", "batch-07"),
}

INPUT_ROOT = Path("/kaggle/input")
OUTPUT_ROOT = Path("/kaggle/working/ocr-production-easyocr-v1")
MODEL_DIR = OUTPUT_ROOT / "easyocr-models"
IMAGE_PATTERN = re.compile(r"^(s\d+)_(\d+)\.(?:jpg|jpeg|png|webp)$", re.IGNORECASE)
ALLOWED_ARCHIVE_MEMBERS = {
    "easyocr-frames.jsonl",
    "vintern-candidates.jsonl",
    "run-signature.json",
    "batch-manifest.json",
    "SHA256SUMS",
    "errors-history.jsonl",
}


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def uid_set_sha256(values: set[int] | list[int]) -> str:
    return hashlib.sha256(
        "".join(f"{value}\n" for value in sorted(set(values))).encode("ascii")
    ).hexdigest()


def string_set_sha256(values: list[str]) -> str:
    return hashlib.sha256(
        "".join(f"{value}\n" for value in sorted(set(values))).encode("utf-8")
    ).hexdigest()


def make_uid(video_id: str, shot_id: str, local_idx: int) -> int:
    payload = f"{video_id}:{shot_id}:{local_idx}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") >> 1


def stable_id(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:24]


def atomic_json(path: Path, value: object) -> None:
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def rewrite_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = Path(f"{path}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_jsonl(path: Path, identity_field: str) -> list[dict]:
    rows: list[dict] = []
    seen: set[object] = set()
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            identity = row[identity_field]
            if identity in seen:
                raise ValueError(f"duplicate {identity_field}: {identity}")
            seen.add(identity)
            rows.append(row)
    return rows


def find_unique_input(name: str) -> Path | None:
    matches: list[Path] = []
    for pattern in (f"*/{name}", f"*/*/{name}", f"*/*/*/{name}"):
        matches.extend(path.resolve() for path in INPUT_ROOT.glob(pattern) if path.is_file())
    values = sorted(set(matches))
    if len(values) > 1:
        raise RuntimeError(f"attach at most one {name}: {values}")
    return values[0] if values else None


def find_batch_root(directory_name: str) -> Path:
    matches: list[Path] = []
    for pattern in (
        directory_name,
        f"*/{directory_name}",
        f"*/*/{directory_name}",
        f"*/*/*/{directory_name}",
    ):
        matches.extend(path.resolve() for path in INPUT_ROOT.glob(pattern) if path.is_dir())
    values = sorted(set(matches))
    if not values:
        values = sorted(
            path.resolve() for path in INPUT_ROOT.rglob(directory_name) if path.is_dir()
        )
    if len(values) != 1:
        raise RuntimeError(f"expected exactly one {directory_name}: {values}")
    return values[0]


def catalog_rows_for_batch(batch_id: str) -> tuple[list[dict], Path]:
    spec = BATCHES[batch_id]
    root = find_batch_root(spec["keyframes_batch"])
    video_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    video_ids = [path.name for path in video_dirs]
    if len(video_ids) != spec["video_count"]:
        raise RuntimeError(f"{batch_id} video count mismatch")
    if video_ids[0] != spec["first_video_id"] or video_ids[-1] != spec["last_video_id"]:
        raise RuntimeError(f"{batch_id} video range mismatch")
    if string_set_sha256(video_ids) != spec["video_set_sha256"]:
        raise RuntimeError(f"{batch_id} video-set checksum mismatch")
    rows: list[dict] = []
    for video_dir in video_dirs:
        entries: list[tuple[str, int, Path]] = []
        for image_path in video_dir.iterdir():
            match = IMAGE_PATTERN.match(image_path.name)
            if match:
                entries.append((match.group(1), int(match.group(2)), image_path))
        entries.sort(key=lambda value: (value[0], value[1]))
        for local_idx, (shot_id, filename_index, image_path) in enumerate(entries):
            rows.append(
                {
                    "video_id": video_dir.name,
                    "shot_id": shot_id,
                    "local_idx": local_idx,
                    "filename_index": filename_index,
                    "keyframe_uid": make_uid(video_dir.name, shot_id, local_idx),
                    "source_image": image_path.relative_to(INPUT_ROOT).as_posix(),
                    "source_path": str(image_path),
                }
            )
    rows.sort(key=lambda row: (row["video_id"], row["shot_id"], row["local_idx"]))
    uids = [int(row["keyframe_uid"]) for row in rows]
    if len(rows) != spec["keyframe_count"] or len(uids) != len(set(uids)):
        raise RuntimeError(f"{batch_id} keyframe count/duplicate mismatch")
    if uid_set_sha256(uids) != spec["uid_set_sha256"]:
        raise RuntimeError(f"{batch_id} UID-set checksum mismatch")
    return rows, root


def checkpoint_zip(batch_id: str, batch_dir: Path) -> Path:
    destination = Path(f"/kaggle/working/ocr-production-{batch_id}-easyocr-checkpoint.zip")
    temporary = Path(f"{destination}.tmp")
    names = (
        "run-signature.json",
        "easyocr-frames.jsonl",
        "vintern-candidates.jsonl",
        "batch-manifest.json",
        "SHA256SUMS",
        "errors-history.jsonl",
    )
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4
    ) as archive:
        for name in names:
            path = batch_dir / name
            if path.exists():
                archive.write(path, name)
    os.replace(temporary, destination)
    return destination


def restore_checkpoint(batch_id: str, batch_dir: Path) -> bool:
    name = f"ocr-production-{batch_id}-easyocr-checkpoint.zip"
    source = find_unique_input(name)
    if source is None:
        return False
    with zipfile.ZipFile(source) as archive:
        if archive.testzip() is not None:
            raise RuntimeError(f"{batch_id} checkpoint CRC failure")
        names = archive.namelist()
        if not names or len(names) != len(set(names)):
            raise RuntimeError(f"{batch_id} checkpoint member duplication")
        if any(
            name != Path(name).name or name not in ALLOWED_ARCHIVE_MEMBERS for name in names
        ):
            raise RuntimeError(f"{batch_id} checkpoint has unsafe/unknown members")
        for name in names:
            target = batch_dir / name
            payload = archive.read(name)
            if target.exists() and target.read_bytes() != payload:
                raise RuntimeError(f"{batch_id} checkpoint conflicts with working data")
            target.write_bytes(payload)
    print("CHECKPOINT_RESTORED", batch_id, source)
    return True


def download_and_verify_weights() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for key, spec in EASYOCR_WEIGHTS.items():
        archive_path = OUTPUT_ROOT / f"{key}.zip"
        weight_path = MODEL_DIR / spec["weight_name"]
        if not archive_path.exists() or sha256_file(archive_path) != spec["zip_sha256"]:
            urllib.request.urlretrieve(spec["url"], archive_path)
        if sha256_file(archive_path) != spec["zip_sha256"]:
            raise RuntimeError(f"{key} ZIP checksum mismatch")
        if not weight_path.exists() or sha256_file(weight_path) != spec["weight_sha256"]:
            with zipfile.ZipFile(archive_path) as archive:
                member = next(
                    name for name in archive.namelist() if name.endswith(spec["weight_name"])
                )
                with archive.open(member) as source, weight_path.open("wb") as target:
                    shutil.copyfileobj(source, target)
        if sha256_file(weight_path) != spec["weight_sha256"]:
            raise RuntimeError(f"{key} weight checksum mismatch")


def quad_flat(points: np.ndarray) -> list[float]:
    return [float(value) for point in points for value in point]


def normalize_quad(points: np.ndarray, width: int, height: int) -> list[float]:
    values = quad_flat(points)
    return [
        max(0.0, min(1.0, value / (width if index % 2 == 0 else height)))
        for index, value in enumerate(values)
    ]


def has_vi_marks(text: str) -> bool:
    return any(
        character in "ăâđêôơưĂÂĐÊÔƠƯ"
        or any(
            mark in unicodedata.normalize("NFD", character)
            for mark in "\u0300\u0301\u0303\u0309\u0323"
        )
        for character in text
    )


def has_ascii_word(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]{3,}", text))


def process_frame(reader, row: dict) -> dict:
    started = time.perf_counter()
    image = cv2.imread(row["source_path"], cv2.IMREAD_COLOR)
    if image is None:
        return {
            **row,
            "schema_version": 1,
            "status": "error",
            "error": "cv2_imread_failed",
            "regions": [],
        }
    height, width = image.shape[:2]
    try:
        detect_started = time.perf_counter()
        horizontal, free = reader.detect(
            image,
            min_size=10,
            text_threshold=SELECTED_CRAFT["text_threshold"],
            low_text=SELECTED_CRAFT["low_text"],
            link_threshold=SELECTED_CRAFT["link_threshold"],
            canvas_size=2560,
            mag_ratio=1.0,
            slope_ths=0.1,
            ycenter_ths=0.5,
            height_ths=0.5,
            width_ths=0.5,
            add_margin=0.1,
            reformat=True,
        )
        detect_seconds = time.perf_counter() - detect_started
        horizontal_rows = horizontal[0] if horizontal else []
        free_rows = free[0] if free else []
        detected_count = len(horizontal_rows) + len(free_rows)
        if detected_count == 0:
            return {
                **row,
                "schema_version": 1,
                "status": "no_text",
                "image_width": width,
                "image_height": height,
                "detected_region_count": 0,
                "detect_seconds": detect_seconds,
                "recognize_seconds": 0.0,
                "latency_seconds": time.perf_counter() - started,
                "error": None,
                "regions": [],
            }
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        recognize_started = time.perf_counter()
        results = reader.recognize(
            grey,
            horizontal_rows,
            free_rows,
            decoder="greedy",
            beamWidth=5,
            batch_size=1,
            workers=0,
            detail=1,
            paragraph=False,
            contrast_ths=0.1,
            adjust_contrast=0.5,
            filter_ths=0.003,
            reformat=False,
        )
        recognize_seconds = time.perf_counter() - recognize_started
        if len(results) != detected_count:
            raise RuntimeError(
                f"recognizer returned {len(results)} rows for {detected_count} regions"
            )
        regions: list[dict] = []
        for index, (raw_points, raw_text, raw_confidence) in enumerate(results):
            points = np.asarray(raw_points, dtype=np.float32).reshape(4, 2)
            text = str(raw_text).strip()
            confidence = float(max(0.0, min(1.0, raw_confidence)))
            x1 = max(0, int(math.floor(points[:, 0].min())) - 4)
            y1 = max(0, int(math.floor(points[:, 1].min())) - 4)
            x2 = min(width, int(math.ceil(points[:, 0].max())) + 4)
            y2 = min(height, int(math.ceil(points[:, 1].max())) + 4)
            region_id = stable_id(
                row["keyframe_uid"],
                index,
                ";".join(f"{value:.2f}" for value in quad_flat(points)),
            )
            regions.append(
                {
                    "region_id": region_id,
                    "bbox_px": quad_flat(points),
                    "bbox_normalized": normalize_quad(points, width, height),
                    "crop_width": max(0, x2 - x1),
                    "crop_height": max(0, y2 - y1),
                    "easyocr_text": text,
                    "easyocr_confidence": confidence,
                    "has_vi_marks": has_vi_marks(text),
                    "has_ascii_word": has_ascii_word(text),
                }
            )
        return {
            **row,
            "schema_version": 1,
            "status": "text_detected",
            "image_width": width,
            "image_height": height,
            "detected_region_count": detected_count,
            "detect_seconds": detect_seconds,
            "recognize_seconds": recognize_seconds,
            "latency_seconds": time.perf_counter() - started,
            "error": None,
            "regions": regions,
        }
    except Exception as exc:
        return {
            **row,
            "schema_version": 1,
            "status": "error",
            "image_width": width,
            "image_height": height,
            "detected_region_count": None,
            "detect_seconds": None,
            "recognize_seconds": None,
            "latency_seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            "regions": [],
        }


_ALLOWED_PUNCTUATION = frozenset(".,:;!?%+-/()[]{}'\"&@#_\\|~`=<>₫$€£¥…–—")


def noise_ratio(text: str) -> float:
    visible = [character for character in text if not character.isspace()]
    if not visible:
        return 0.0
    noisy = 0
    for character in visible:
        if character.isalnum() or character in _ALLOWED_PUNCTUATION:
            continue
        category = unicodedata.category(character)
        if not category.startswith(("L", "N", "P", "S", "M")):
            noisy += 1
    return noisy / len(visible)


def router_reasons(region: dict) -> list[str]:
    text = str(region.get("easyocr_text") or "").strip()
    confidence = float(region.get("easyocr_confidence") or 0.0)
    if not 0 <= confidence <= 1:
        raise ValueError("easyocr_confidence must be in [0,1]")
    reasons: list[str] = []
    if not text:
        reasons.append("empty_text")
    elif confidence < ROUTER_POLICY["hard_confidence_threshold"]:
        reasons.append("confidence_lt_0_40")
    elif confidence < ROUTER_POLICY["selective_confidence_ceiling"]:
        if region.get("has_vi_marks") and region.get("has_ascii_word"):
            reasons.append("region_mixed_0_40_to_0_60")
        if "?" in text or "�" in text:
            reasons.append("ambiguous_glyph_0_40_to_0_60")
        visible_length = len("".join(text.split()))
        if (
            visible_length >= ROUTER_POLICY["noisy_character_min_length"]
            and noise_ratio(text) >= ROUTER_POLICY["noisy_character_ratio_threshold"]
        ):
            reasons.append("noisy_text_0_40_to_0_60")
    return reasons


def candidate_rows(easy_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for frame in easy_rows:
        for region in frame.get("regions") or []:
            reasons = router_reasons(region)
            if not reasons:
                continue
            rows.append(
                {
                    "schema_version": 1,
                    "candidate_id": region["region_id"],
                    "video_id": frame["video_id"],
                    "shot_id": frame["shot_id"],
                    "local_idx": frame["local_idx"],
                    "keyframe_uid": frame["keyframe_uid"],
                    "source_image": frame["source_image"],
                    "source_path": frame["source_path"],
                    "bbox_px": region["bbox_px"],
                    "crop_width": region["crop_width"],
                    "crop_height": region["crop_height"],
                    "easyocr_text": region["easyocr_text"],
                    "easyocr_confidence": region["easyocr_confidence"],
                    "router_v2_reasons": reasons,
                }
            )
    rows.sort(key=lambda row: (row["video_id"], row["shot_id"], row["local_idx"], row["candidate_id"]))
    if len(rows) != len({row["candidate_id"] for row in rows}):
        raise RuntimeError("duplicate Vintern candidate_id")
    return rows


def build_complete_archive(
    *,
    batch_id: str,
    batch_dir: Path,
    catalog_rows: list[dict],
    easy_rows: list[dict],
    candidates: list[dict],
    restored_checkpoint: bool,
    resumed_with_new_work: bool,
    elapsed_seconds: float,
) -> tuple[Path, dict]:
    easy_path = batch_dir / "easyocr-frames.jsonl"
    candidate_path = batch_dir / "vintern-candidates.jsonl"
    state_path = batch_dir / "run-signature.json"
    manifest_path = batch_dir / "batch-manifest.json"
    checksums_path = batch_dir / "SHA256SUMS"
    errors_path = batch_dir / "errors-history.jsonl"
    statuses = Counter(row["status"] for row in easy_rows)
    region_count = sum(len(row.get("regions") or []) for row in easy_rows)
    error_count = statuses["error"]
    missing = len(catalog_rows) - len(easy_rows)
    gate = missing == 0 and error_count == 0
    if not gate:
        raise RuntimeError(f"{batch_id} completion gate failed")
    config_sha = hashlib.sha256(
        canonical_json(
            {
                "contract": NOTEBOOK_CONTRACT,
                "craft": SELECTED_CRAFT,
                "router": ROUTER_POLICY,
                "weights": EASYOCR_WEIGHTS,
            }
        )
    ).hexdigest()
    layer_manifest = {
        "schema_version": 1,
        "batch_id": batch_id,
        "worker_id": f"ocr-slot-{WORKER_SLOT}",
        "layer": "easyocr",
        "item_kind": "region",
        "catalog_sha256": CATALOG_SHA256,
        "config_sha256": config_sha,
        "assigned_uid_sha256": BATCHES[batch_id]["uid_set_sha256"],
        "input_artifact_path": f"catalog/{batch_id}.uid-set",
        "input_artifact_sha256": BATCHES[batch_id]["uid_set_sha256"],
        "output_jsonl_path": f"ocr/layers/easyocr/{batch_id}.jsonl",
        "output_jsonl_sha256": sha256_file(easy_path),
        "expected_keyframes": len(catalog_rows),
        "processed_keyframes": len(easy_rows),
        "expected_items": region_count,
        "processed_items": region_count,
        "error_items": error_count,
        "duplicate_items": 0,
        "missing_keyframes": missing,
        "foreign_keyframes": 0,
        "completion_gate_passed": gate,
    }
    manifest = {
        "schema_version": 2,
        "artifact_kind": "ocr_production_layer_archive",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "batch_id": batch_id,
        "tier": "easyocr",
        "complete": gate,
        "worker_slot": WORKER_SLOT,
        "catalog_sha256": CATALOG_SHA256,
        "batch_mapping_sha256": BATCH_MAPPING_SHA256,
        "assigned_uid_sha256": BATCHES[batch_id]["uid_set_sha256"],
        "observed_uid_sha256": uid_set_sha256(
            {int(row["keyframe_uid"]) for row in easy_rows}
        ),
        "frames": len(easy_rows),
        "expected_frames": len(catalog_rows),
        "status": dict(sorted(statuses.items())),
        "regions": region_count,
        "vintern_candidates": len(candidates),
        "vintern_candidate_fraction": len(candidates) / region_count if region_count else 0.0,
        "seconds": elapsed_seconds,
        "frames_per_second": len(easy_rows) / elapsed_seconds if elapsed_seconds else None,
        "checkpoint_resume": {
            "restored_from_input": restored_checkpoint,
            "new_work_after_restore": resumed_with_new_work,
            "verified": restored_checkpoint and resumed_with_new_work,
        },
        "model_calls": {"gemini": 0, "vintern": 0},
        "files": {
            "easyocr-frames.jsonl": sha256_file(easy_path),
            "vintern-candidates.jsonl": sha256_file(candidate_path),
            "run-signature.json": sha256_file(state_path),
        },
        "layer_manifest": layer_manifest,
    }
    if errors_path.exists():
        manifest["files"]["errors-history.jsonl"] = sha256_file(errors_path)
    atomic_json(manifest_path, manifest)
    checksum_files = [easy_path, candidate_path, state_path, manifest_path]
    if errors_path.exists():
        checksum_files.append(errors_path)
    checksums_path.write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in checksum_files),
        encoding="ascii",
    )
    archive_path = Path(f"/kaggle/working/ocr-production-{batch_id}-easyocr.zip")
    temporary = Path(f"{archive_path}.tmp")
    archive_files = checksum_files + [checksums_path]
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in archive_files:
            archive.write(path, path.name)
    os.replace(temporary, archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("final archive CRC failure")
        names = archive.namelist()
        if set(names) != {path.name for path in archive_files}:
            raise RuntimeError("final archive member mismatch")
        if any(Path(name).suffix.casefold() in {".jpg", ".jpeg", ".png", ".webp", ".mp4"} for name in names):
            raise RuntimeError("media file leaked into OCR archive")
    return archive_path, manifest


def get_hf_token() -> str:
    from kaggle_secrets import UserSecretsClient

    client = UserSecretsClient()
    for name in ("HF_TOKEN", "HK_TOKEN"):
        try:
            token = client.get_secret(name)
        except Exception:
            token = None
        if token:
            return token
    raise RuntimeError("PUBLISH_TO_HF=True but Kaggle secret HF_TOKEN/HK_TOKEN is missing")


def publish_archive(archive_path: Path, batch_id: str, token: str) -> dict:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)
    info = api.repo_info(repo_id=HF_REPO_ID, repo_type="dataset")
    if not bool(info.private):
        raise RuntimeError("HF Dataset must be private")
    remote_path = f"ocr/archives/{batch_id}/easyocr/{archive_path.name}"
    expected_prefix = f"ocr/archives/{batch_id}/"
    if not remote_path.startswith(expected_prefix):
        raise RuntimeError("HF path escaped OCR batch namespace")
    local_sha = sha256_file(archive_path)
    repo_files = set(api.list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset"))
    if remote_path in repo_files:
        revision = str(api.repo_info(repo_id=HF_REPO_ID, repo_type="dataset").sha)
        action = "already_present"
    else:
        commit = None
        for attempt in range(3):
            try:
                commit = api.upload_file(
                    path_or_fileobj=archive_path,
                    path_in_repo=remote_path,
                    repo_id=HF_REPO_ID,
                    repo_type="dataset",
                    commit_message=f"Add OCR EasyOCR production {batch_id}",
                )
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        revision = str(commit.oid)
        action = "uploaded"
    with tempfile.TemporaryDirectory(prefix=f"ocr-hf-{batch_id}-") as temporary:
        downloaded = Path(
            hf_hub_download(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                filename=remote_path,
                revision=revision,
                token=token,
                local_dir=temporary,
                force_download=True,
            )
        )
        if sha256_file(downloaded) != local_sha:
            raise RuntimeError("HF round-trip archive checksum mismatch")
    return {
        "schema_version": 1,
        "action": action,
        "repo_id": HF_REPO_ID,
        "private_repo_verified": True,
        "batch_id": batch_id,
        "remote_path": remote_path,
        "revision": revision,
        "archive_sha256": local_sha,
        "round_trip_verified": True,
    }


if WORKER_SLOT not in WORKER_BATCHES:
    raise ValueError("WORKER_SLOT must be 1, 2, 3, or 4")
if CHECKPOINT_EVERY <= 0 or PROGRESS_EVERY <= 0 or INTERRUPT_AFTER_NEW_FRAMES < 0:
    raise ValueError("checkpoint/progress/interrupt settings are invalid")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("select Kaggle GPU and expose exactly one CUDA device")
if "T4" not in torch.cuda.get_device_name(0).upper():
    raise RuntimeError(f"expected T4 GPU, got {torch.cuda.get_device_name(0)}")

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
hf_token = get_hf_token() if PUBLISH_TO_HF else ""
if PUBLISH_TO_HF:
    from huggingface_hub import HfApi

    if not bool(HfApi(token=hf_token).repo_info(repo_id=HF_REPO_ID, repo_type="dataset").private):
        raise RuntimeError("HF Dataset must be private")
download_and_verify_weights()

import easyocr

print(
    "OCR_PRODUCTION_START",
    {
        "worker_slot": WORKER_SLOT,
        "batches": WORKER_BATCHES[WORKER_SLOT],
        "frames": sum(BATCHES[batch]["keyframe_count"] for batch in WORKER_BATCHES[WORKER_SLOT]),
        "gpu": torch.cuda.get_device_name(0),
        "publish_to_hf": PUBLISH_TO_HF,
    },
    flush=True,
)

reader_started = time.perf_counter()
print("EASYOCR_MODEL_INIT_START", {"languages": ["vi", "en"]}, flush=True)
reader = easyocr.Reader(
    ["vi", "en"],
    gpu="cuda:0",
    model_storage_directory=str(MODEL_DIR),
    download_enabled=False,
    detector=True,
    recognizer=True,
    verbose=True,
)
print(
    "EASYOCR_MODEL_INIT_DONE",
    {"seconds": round(time.perf_counter() - reader_started, 3)},
    flush=True,
)

for batch_position, batch_id in enumerate(WORKER_BATCHES[WORKER_SLOT], start=1):
    batch_started = time.perf_counter()
    batch_dir = OUTPUT_ROOT / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    easy_path = batch_dir / "easyocr-frames.jsonl"
    candidate_path = batch_dir / "vintern-candidates.jsonl"
    state_path = batch_dir / "run-signature.json"
    errors_path = batch_dir / "errors-history.jsonl"
    print("BATCH_PREPARE_START", batch_id, flush=True)
    restored = restore_checkpoint(batch_id, batch_dir)
    catalog_started = time.perf_counter()
    print("CATALOG_SCAN_START", batch_id, flush=True)
    catalog_rows, batch_root = catalog_rows_for_batch(batch_id)
    print(
        "CATALOG_SCAN_DONE",
        batch_id,
        {"frames": len(catalog_rows), "seconds": round(time.perf_counter() - catalog_started, 3)},
        flush=True,
    )
    signature = {
        "schema_version": 1,
        "contract": NOTEBOOK_CONTRACT,
        "worker_slot": WORKER_SLOT,
        "batch_id": batch_id,
        "catalog_sha256": CATALOG_SHA256,
        "batch_mapping_sha256": BATCH_MAPPING_SHA256,
        "assigned_uid_sha256": BATCHES[batch_id]["uid_set_sha256"],
        "selected_craft": SELECTED_CRAFT,
        "router_policy": ROUTER_POLICY,
        "easyocr_weights": EASYOCR_WEIGHTS,
    }
    if state_path.exists():
        if json.loads(state_path.read_text(encoding="utf-8")) != signature:
            raise RuntimeError(f"{batch_id} stale checkpoint signature")
    else:
        atomic_json(state_path, signature)
    existing = load_jsonl(easy_path, "keyframe_uid")
    prior_errors = [row for row in existing if row.get("status") == "error"]
    if prior_errors:
        for row in prior_errors:
            append_jsonl(errors_path, row)
        existing = [row for row in existing if row.get("status") != "error"]
        rewrite_jsonl(easy_path, existing)
        print("RETRYING_PRIOR_ERRORS", batch_id, len(prior_errors))
    expected_uids = {int(row["keyframe_uid"]) for row in catalog_rows}
    existing_uids = {int(row["keyframe_uid"]) for row in existing}
    if not existing_uids <= expected_uids:
        raise RuntimeError(f"{batch_id} checkpoint contains foreign UID")
    new_frames = 0
    print("BATCH_RESUME", batch_id, len(existing_uids), "/", len(catalog_rows), batch_root, flush=True)
    for row in catalog_rows:
        uid = int(row["keyframe_uid"])
        if uid in existing_uids:
            continue
        result = process_frame(reader, row)
        append_jsonl(easy_path, result)
        existing_uids.add(uid)
        new_frames += 1
        completed = len(existing_uids)
        if completed % PROGRESS_EVERY == 0 or completed == len(catalog_rows):
            print("EASYOCR_PRODUCTION_PROGRESS", batch_id, completed, "/", len(catalog_rows), flush=True)
        if completed % CHECKPOINT_EVERY == 0:
            checkpoint_zip(batch_id, batch_dir)
        if (
            INTERRUPT_AFTER_NEW_FRAMES
            and batch_position == 1
            and new_frames >= INTERRUPT_AFTER_NEW_FRAMES
        ):
            checkpoint = checkpoint_zip(batch_id, batch_dir)
            print("INTENTIONAL_INTERRUPT_READY", checkpoint, sha256_file(checkpoint))
            raise RuntimeError(
                "Intentional checkpoint test completed. Download the checkpoint, attach it "
                "to a fresh session, set INTERRUPT_AFTER_NEW_FRAMES=0, and rerun."
            )
    easy_rows = load_jsonl(easy_path, "keyframe_uid")
    observed = {int(row["keyframe_uid"]) for row in easy_rows}
    if observed != expected_uids:
        raise RuntimeError(f"{batch_id} final UID set mismatch")
    errors = [row for row in easy_rows if row.get("status") == "error"]
    if errors:
        checkpoint = checkpoint_zip(batch_id, batch_dir)
        print("BATCH_HAS_ERRORS_RERUN_TO_RETRY", batch_id, len(errors), checkpoint)
        raise RuntimeError(f"{batch_id} contains {len(errors)} errors")
    candidates = candidate_rows(easy_rows)
    rewrite_jsonl(candidate_path, candidates)
    archive_path, manifest = build_complete_archive(
        batch_id=batch_id,
        batch_dir=batch_dir,
        catalog_rows=catalog_rows,
        easy_rows=easy_rows,
        candidates=candidates,
        restored_checkpoint=restored,
        resumed_with_new_work=restored and new_frames > 0,
        elapsed_seconds=time.perf_counter() - batch_started,
    )
    checkpoint = checkpoint_zip(batch_id, batch_dir)
    publish_report = None
    if PUBLISH_TO_HF:
        publish_report = publish_archive(archive_path, batch_id, hf_token)
        atomic_json(batch_dir / "hf-publish-report.json", publish_report)
    print(
        "BATCH_COMPLETE",
        json.dumps(
            {
                "batch_id": batch_id,
                "frames": len(easy_rows),
                "regions": manifest["regions"],
                "vintern_candidates": len(candidates),
                "archive": str(archive_path),
                "archive_sha256": sha256_file(archive_path),
                "checkpoint": str(checkpoint),
                "hf": publish_report,
            },
            ensure_ascii=False,
        ),
    )

del reader
gc.collect()
torch.cuda.empty_cache()
print("WORKER_SLOT_COMPLETE", WORKER_SLOT, WORKER_BATCHES[WORKER_SLOT])
