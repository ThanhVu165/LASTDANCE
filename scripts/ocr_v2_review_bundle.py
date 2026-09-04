"""Build a human-review bundle from one immutable EasyOCR production archive.

This is an OCR-v2 experiment helper, not a production OCR writer.  It never
calls a model, changes an archive, or produces a searchable SQLite artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import textwrap
import unicodedata
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont, ImageOps


ARCHIVE_MEMBERS = frozenset(
    {
        "easyocr-frames.jsonl",
        "vintern-candidates.jsonl",
        "run-signature.json",
        "batch-manifest.json",
        "SHA256SUMS",
    }
)
REVIEW_COLUMNS = (
    "review_id",
    "keyframe_uid",
    "video_id",
    "shot_id",
    "local_idx",
    "source_image",
    "stratum",
    "sheet_file",
    "sheet_slot",
    "region_count",
    "machine_duplicate_text_count",
    "machine_low_confidence_region_count",
    "gt_has_text",
    "bbox_quality",
    "easyocr_quality",
    "human_text",
    "notes",
)
RECOGNITION_LABEL_COLUMNS = (
    "sample_index",
    "region_id",
    "sample_row_sha256",
    "keyframe_uid",
    "video_id",
    "shot_id",
    "source_image",
    "frame_stratum",
    "region_stratum",
    "bbox_px",
    "crop_width",
    "crop_height",
    "easyocr_text",
    "easyocr_confidence",
    "crop_spec_id",
    "sheet_file",
    "sheet_slot",
    "label_status",
    "human_text",
    "text_type",
    "notes",
)
IMMUTABLE_REGION_FIELDS = tuple(
    column
    for column in RECOGNITION_LABEL_COLUMNS
    if column not in {"sample_row_sha256", "label_status", "human_text", "text_type", "notes"}
)
CROP_SPEC = {
    "id": "pil_quad_v1_pad08_edge",
    "method": "pil_quad_v1",
    "point_order": "top_left_top_right_bottom_right_bottom_left",
    "padding_ratio": 0.08,
    "padding_mode": "edge",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    """Hash a file or a Kaggle-auto-extracted archive directory deterministically."""

    path = Path(path)
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    files = sorted(value for value in path.iterdir() if value.is_file())
    if not files:
        raise RuntimeError(f"archive directory is empty: {path}")
    for value in files:
        digest.update(value.name.encode("utf-8"))
        digest.update(b"\0")
        with value.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: object) -> str:
    return " ".join(unicodedata.normalize("NFC", str(value or "")).casefold().split())


def allocate_quotas(weights: dict[str, float], total: int) -> dict[str, int]:
    """Allocate an exact non-negative total with deterministic largest remainders."""

    if total < 0 or not weights:
        raise ValueError("quota total/weights are invalid")
    weight_sum = sum(weights.values())
    if weight_sum <= 0 or any(value < 0 for value in weights.values()):
        raise ValueError("quota weights must be non-negative with a positive sum")
    exact = {key: total * weight / weight_sum for key, weight in weights.items()}
    allocation = {key: math.floor(value) for key, value in exact.items()}
    remainder = total - sum(allocation.values())
    order = sorted(weights, key=lambda key: (-(exact[key] - allocation[key]), key))
    for key in order[:remainder]:
        allocation[key] += 1
    if sum(allocation.values()) != total or any(value < 0 for value in allocation.values()):
        raise RuntimeError("internal quota allocation failure")
    return allocation


def read_jsonl_bytes(payload: bytes, *, identity: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[object] = set()
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        value = row.get(identity)
        if value in seen:
            raise ValueError(f"duplicate {identity} at JSONL line {line_number}")
        seen.add(value)
        rows.append(row)
    return rows


def _read_archive_members(archive_path: Path) -> dict[str, bytes]:
    if archive_path.is_dir():
        names = {path.name for path in archive_path.iterdir() if path.is_file()}
        if names != ARCHIVE_MEMBERS:
            raise RuntimeError(f"unexpected archive members: {sorted(names)}")
        return {name: (archive_path / name).read_bytes() for name in ARCHIVE_MEMBERS}
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("archive CRC validation failed")
        names = set(archive.namelist())
        if names != ARCHIVE_MEMBERS:
            raise RuntimeError(f"unexpected archive members: {sorted(names)}")
        return {name: archive.read(name) for name in ARCHIVE_MEMBERS}


def read_archive(archive_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    payload = _read_archive_members(archive_path)
    checksums = payload["SHA256SUMS"].decode("ascii").splitlines()
    expected = {}
    for line in checksums:
        digest, name = line.split(maxsplit=1)
        expected[name.strip()] = digest
    for name in ("easyocr-frames.jsonl", "vintern-candidates.jsonl", "run-signature.json", "batch-manifest.json"):
        actual = hashlib.sha256(payload[name]).hexdigest()
        if expected.get(name) != actual:
            raise RuntimeError(f"archive SHA-256 mismatch: {name}")
    manifest = json.loads(payload["batch-manifest.json"])
    frames = read_jsonl_bytes(payload["easyocr-frames.jsonl"], identity="keyframe_uid")
    candidates = read_jsonl_bytes(payload["vintern-candidates.jsonl"], identity="candidate_id")
    if manifest.get("tier") != "easyocr" or manifest.get("model_calls", {}).get("vintern") != 0:
        raise RuntimeError("this review tool accepts only an EasyOCR-only production archive")
    if int(manifest.get("frames", -1)) != len(frames):
        raise RuntimeError("frame count does not match manifest")
    if int(manifest.get("vintern_candidates", -1)) != len(candidates):
        raise RuntimeError("Vintern candidate count does not match manifest")
    return manifest, frames, candidates


def duplicate_text_count(frame: dict[str, Any]) -> int:
    values = [normalize_text(region.get("easyocr_text")) for region in frame.get("regions") or []]
    values = [value for value in values if value]
    return sum(count - 1 for count in Counter(values).values() if count > 1)


def low_confidence_count(frame: dict[str, Any]) -> int:
    return sum(float(region.get("easyocr_confidence") or 0.0) < 0.40 for region in frame.get("regions") or [])


def stable_order(rows: Iterable[dict[str, Any]], salt: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{salt}|{row['keyframe_uid']}".encode("utf-8")).hexdigest(),
    )


def stable_region_order(rows: Iterable[dict[str, Any]], salt: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{salt}|{row['keyframe_uid']}|{row['region']['region_id']}".encode("utf-8")
        ).hexdigest(),
    )


def choose_review_frames(frames: list[dict[str, Any]], *, sample_size: int, seed: str) -> list[dict[str, Any]]:
    """Choose deterministic, UID-unique difficult and control frames.

    The quotas deliberately overlap in their candidate pools; a selected UID is never
    selected again, and any shortage is filled from the remaining canonical population.
    """

    if sample_size < 20:
        raise ValueError("sample_size must be at least 20")
    quotas = allocate_quotas(
        {
            "no_text_control": 0.15,
            "low_confidence": 0.25,
            "repeated_text": 0.25,
            "dense_regions": 0.20,
            "random_control": 0.15,
        },
        sample_size,
    )
    ranked = {
        "no_text_control": stable_order(
            [row for row in frames if row.get("status") == "no_text"], f"{seed}|no_text"
        ),
        "low_confidence": sorted(
            [row for row in frames if row.get("status") == "text_detected"],
            key=lambda row: (-low_confidence_count(row), str(row["keyframe_uid"])),
        ),
        "repeated_text": sorted(
            [row for row in frames if row.get("status") == "text_detected"],
            key=lambda row: (-duplicate_text_count(row), str(row["keyframe_uid"])),
        ),
        "dense_regions": sorted(
            [row for row in frames if row.get("status") == "text_detected"],
            key=lambda row: (-len(row.get("regions") or []), str(row["keyframe_uid"])),
        ),
        "random_control": stable_order(frames, f"{seed}|random"),
    }
    selected: list[dict[str, Any]] = []
    selected_uids: set[int] = set()
    for stratum, quota in quotas.items():
        if quota == 0:
            continue
        accepted = 0
        for row in ranked[stratum]:
            uid = int(row["keyframe_uid"])
            if uid in selected_uids:
                continue
            selected.append({**row, "review_stratum": stratum})
            selected_uids.add(uid)
            accepted += 1
            if accepted == quota:
                break
    for row in stable_order(frames, f"{seed}|fill"):
        if len(selected) >= sample_size:
            break
        uid = int(row["keyframe_uid"])
        if uid not in selected_uids:
            selected.append({**row, "review_stratum": "random_fill"})
            selected_uids.add(uid)
    if len(selected) != sample_size:
        raise RuntimeError(f"only selected {len(selected)} of {sample_size} review frames")
    return selected


def choose_balanced_review_frames(
    frames: list[dict[str, Any]], *, video_ids: list[str], sample_size: int, seed: str
) -> list[dict[str, Any]]:
    """Choose equal numbers of UID-unique review frames from each named video."""

    if not video_ids or len(video_ids) != len(set(video_ids)):
        raise ValueError("--video-ids must be non-empty and contain no duplicate IDs")
    if sample_size % len(video_ids) != 0:
        raise ValueError("--sample-size must divide evenly across --video-ids")
    per_video = sample_size // len(video_ids)
    by_video: dict[str, list[dict[str, Any]]] = {video_id: [] for video_id in video_ids}
    for frame in frames:
        video_id = str(frame.get("video_id") or "")
        if video_id in by_video:
            by_video[video_id].append(frame)
    selected: list[dict[str, Any]] = []
    for video_id in video_ids:
        population = by_video[video_id]
        if len(population) < per_video:
            raise RuntimeError(
                f"video_id={video_id!r} has only {len(population)} frames; need {per_video}"
            )
        selected.extend(
            choose_review_frames(
                population,
                sample_size=per_video,
                seed=f"{seed}|video={video_id}",
            )
        )
    if len({int(frame["keyframe_uid"]) for frame in selected}) != sample_size:
        raise RuntimeError("balanced sample contains duplicate keyframe_uid")
    return selected


def _region_candidates(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen_region_ids: set[str] = set()
    for frame in frames:
        for region in frame.get("regions") or []:
            region_id = str(region.get("region_id") or "").strip()
            bbox = region.get("bbox_px")
            if not region_id or not isinstance(bbox, list) or len(bbox) != 8:
                raise ValueError(f"invalid region mapping in keyframe_uid={frame['keyframe_uid']}")
            if region_id in seen_region_ids:
                raise ValueError(f"duplicate region_id in review population: {region_id}")
            seen_region_ids.add(region_id)
            candidates.append({"keyframe_uid": int(frame["keyframe_uid"]), "frame": frame, "region": region})
    return candidates


def choose_recognition_regions(
    frames: list[dict[str, Any]], *, sample_size: int, seed: str
) -> list[dict[str, Any]]:
    """Choose a deterministic confidence/content-balanced same-crop sample.

    At most two crops are selected from one keyframe so a single dense overlay cannot
    dominate the human-labelled recognizer gate.
    """

    if sample_size < 20:
        raise ValueError("region sample_size must be at least 20")
    candidates = _region_candidates(frames)
    quotas = allocate_quotas(
        {
            "low_confidence": 0.25,
            "medium_confidence": 0.17,
            "high_confidence": 0.17,
            "vietnamese_mark_proxy": 0.17,
            "numeric_proxy": 0.12,
            "small_crop_proxy": 0.08,
            "random_control": 0.04,
        },
        sample_size,
    )

    def confidence(item: dict[str, Any]) -> float:
        return float(item["region"].get("easyocr_confidence") or 0.0)

    ranked = {
        "low_confidence": stable_region_order(
            [item for item in candidates if confidence(item) < 0.40], f"{seed}|low"
        ),
        "medium_confidence": stable_region_order(
            [item for item in candidates if 0.40 <= confidence(item) < 0.70], f"{seed}|medium"
        ),
        "high_confidence": stable_region_order(
            [item for item in candidates if confidence(item) >= 0.70], f"{seed}|high"
        ),
        "vietnamese_mark_proxy": stable_region_order(
            [item for item in candidates if bool(item["region"].get("has_vi_marks"))],
            f"{seed}|vi",
        ),
        "numeric_proxy": stable_region_order(
            [
                item
                for item in candidates
                if any(character.isdigit() for character in str(item["region"].get("easyocr_text") or ""))
            ],
            f"{seed}|numeric",
        ),
        "small_crop_proxy": sorted(
            candidates,
            key=lambda item: (
                int(item["region"].get("crop_height") or 0),
                str(item["region"]["region_id"]),
            ),
        ),
        "random_control": stable_region_order(candidates, f"{seed}|random"),
    }
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    per_frame: Counter[int] = Counter()
    for stratum, quota in quotas.items():
        if quota == 0:
            continue
        accepted = 0
        for item in ranked[stratum]:
            region_id = str(item["region"]["region_id"])
            uid = int(item["keyframe_uid"])
            if region_id in selected_ids or per_frame[uid] >= 2:
                continue
            selected.append({**item, "region_stratum": stratum})
            selected_ids.add(region_id)
            per_frame[uid] += 1
            accepted += 1
            if accepted == quota:
                break
    for item in stable_region_order(candidates, f"{seed}|fill"):
        if len(selected) >= sample_size:
            break
        region_id = str(item["region"]["region_id"])
        uid = int(item["keyframe_uid"])
        if region_id not in selected_ids and per_frame[uid] < 2:
            selected.append({**item, "region_stratum": "random_fill"})
            selected_ids.add(region_id)
            per_frame[uid] += 1
    if len(selected) != sample_size:
        raise RuntimeError(f"only selected {len(selected)} of {sample_size} recognition regions")
    return selected


def choose_balanced_recognition_regions(
    frames: list[dict[str, Any]], *, video_ids: list[str], sample_size: int, seed: str
) -> list[dict[str, Any]]:
    """Choose equal recognizer-region counts from every locked Gate A video."""

    if not video_ids or len(video_ids) != len(set(video_ids)):
        raise ValueError("balanced recognition video IDs must be unique and non-empty")
    if sample_size % len(video_ids) != 0:
        raise ValueError("region sample size must divide evenly across video IDs")
    per_video = sample_size // len(video_ids)
    selected: list[dict[str, Any]] = []
    for video_id in video_ids:
        population = [frame for frame in frames if str(frame.get("video_id")) == video_id]
        selected.extend(
            choose_recognition_regions(
                population,
                sample_size=per_video,
                seed=f"{seed}|video={video_id}",
            )
        )
    if len({str(item["region"]["region_id"]) for item in selected}) != sample_size:
        raise RuntimeError("balanced recognizer sample contains duplicate region_id")
    return selected


def image_index(keyframe_root: Path) -> dict[tuple[str, str], Path]:
    index: dict[tuple[str, str], Path] = {}
    for path in keyframe_root.rglob("*.jpg"):
        video_id = path.parent.name
        key = (video_id, path.name)
        if key in index:
            raise RuntimeError(f"ambiguous keyframe image: {key}")
        index[key] = path
    if not index:
        raise RuntimeError("no JPG keyframes found under --keyframe-root")
    return index


def source_path(frame: dict[str, Any], index: dict[tuple[str, str], Path]) -> Path:
    key = (str(frame["video_id"]), Path(str(frame["source_image"])).name)
    try:
        return index[key]
    except KeyError as error:
        raise FileNotFoundError(f"keyframe source is missing: {key}") from error


def font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ):
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


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
    padded.paste(
        Image.new(image.mode, (border, border), image.getpixel((width - 1, height - 1))),
        (border + width, border + height),
    )
    return padded


def rectify_region_crop(image: Image.Image, bbox_px: list[object]) -> Image.Image:
    """Rectify TL/TR/BR/BL CRAFT geometry using the versioned crop contract."""

    if len(bbox_px) != 8:
        raise ValueError("bbox_px must contain exactly four x/y points")
    values = [float(value) for value in bbox_px]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("bbox_px contains NaN/Inf")
    points = [(values[index], values[index + 1]) for index in range(0, 8, 2)]

    def distance(left: tuple[float, float], right: tuple[float, float]) -> float:
        return math.hypot(left[0] - right[0], left[1] - right[1])

    width = max(2, round(max(distance(points[0], points[1]), distance(points[3], points[2]))))
    height = max(2, round(max(distance(points[0], points[3]), distance(points[1], points[2]))))
    quad = (
        points[0][0],
        points[0][1],
        points[3][0],
        points[3][1],
        points[2][0],
        points[2][1],
        points[1][0],
        points[1][1],
    )
    crop = image.convert("RGB").transform(
        (width, height),
        Image.Transform.QUAD,
        quad,
        resample=Image.Resampling.BICUBIC,
    )
    return _edge_pad(crop, max(1, round(height * float(CROP_SPEC["padding_ratio"]))))


def draw_crop_panel(
    item: dict[str, Any], source: Path, *, width: int = 500, height: int = 180
) -> Image.Image:
    with Image.open(source) as opened:
        frame_image = ImageOps.exif_transpose(opened).convert("RGB")
    region = item["region"]
    crop = rectify_region_crop(frame_image, region["bbox_px"])
    crop.thumbnail((width - 20, 105))
    canvas = Image.new("RGB", (width, height), "white")
    if crop.width < 180 or crop.height < 45:
        scale = min((width - 20) / crop.width, 105 / crop.height)
        if scale > 1:
            crop = crop.resize(
                (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
                Image.Resampling.NEAREST,
            )
    canvas.paste(crop, ((width - crop.width) // 2, 4))
    draw = ImageDraw.Draw(canvas)
    confidence = float(region.get("easyocr_confidence") or 0.0)
    draw.text(
        (8, 116),
        f"{item['region_stratum']} | conf={confidence:.3f} | id={region['region_id']}",
        fill="black",
        font=font(13),
    )
    machine_text = str(region.get("easyocr_text") or "").replace("\n", " ")
    draw.text((8, 139), f"EasyOCR: {machine_text}"[:76], fill="black", font=font(13))
    return canvas


def immutable_region_hash(row: dict[str, Any]) -> str:
    payload = {field: row[field] for field in IMMUTABLE_REGION_FIELDS}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def draw_frame(frame: dict[str, Any], source: Path, *, width: int = 780, height: int = 530) -> Image.Image:
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    image.thumbnail((width - 20, 330))
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(image, ((width - image.width) // 2, 5))
    draw = ImageDraw.Draw(canvas)
    scale_x = image.width / max(1, int(frame.get("image_width") or image.width))
    scale_y = image.height / max(1, int(frame.get("image_height") or image.height))
    offset_x = (width - image.width) // 2
    for number, region in enumerate(frame.get("regions") or [], start=1):
        points = region.get("bbox_px") or []
        if len(points) != 8:
            continue
        polygon = [
            (offset_x + points[index] * scale_x, 5 + points[index + 1] * scale_y)
            for index in range(0, 8, 2)
        ]
        color = "red" if float(region.get("easyocr_confidence") or 0.0) < 0.40 else "lime"
        draw.line([*polygon, polygon[0]], fill=color, width=2)
        draw.text(polygon[0], str(number), fill=color, font=font(15), stroke_width=1, stroke_fill="black")
    header = f"{frame['video_id']} | uid={frame['keyframe_uid']} | {frame['review_stratum']}"
    draw.text((8, 340), header[:105], fill="black", font=font(14))
    lines: list[str] = []
    for number, region in enumerate(frame.get("regions") or [], start=1):
        text = str(region.get("easyocr_text") or "").replace("\n", " ")
        confidence = float(region.get("easyocr_confidence") or 0.0)
        lines.append(f"{number}. [{confidence:.2f}] {text}")
    if not lines:
        lines.append("CRAFT: no detected region")
    y = 362
    for line in lines[:8]:
        for wrapped in textwrap.wrap(line, width=92) or [line]:
            draw.text((8, y), wrapped, fill="black", font=font(13))
            y += 17
            if y > height - 16:
                return canvas
    return canvas


def write_bundle(
    frames: list[dict[str, Any]],
    recognition_regions: list[dict[str, Any]],
    *,
    archive: Path,
    manifest: dict[str, Any],
    keyframe_root: Path,
    output_dir: Path,
    seed: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    frame_sheets = output_dir / "frame-sheets"
    crop_sheets = output_dir / "crop-sheets"
    frame_sheets.mkdir(parents=True)
    crop_sheets.mkdir(parents=True)
    index = image_index(keyframe_root)
    review_rows: list[dict[str, Any]] = []
    for page_start in range(0, len(frames), 4):
        page = frames[page_start : page_start + 4]
        sheet_name = f"review-{page_start // 4 + 1:02d}.jpg"
        canvas = Image.new("RGB", (1600, 1060), "white")
        for offset, frame in enumerate(page):
            panel = draw_frame(frame, source_path(frame, index))
            x = (offset % 2) * 800 + 10
            y = (offset // 2) * 530
            canvas.paste(panel, (x, y))
            review_rows.append(
                {
                    "review_id": hashlib.sha256(f"ocr-v2-review|{frame['keyframe_uid']}".encode()).hexdigest()[:24],
                    "keyframe_uid": frame["keyframe_uid"],
                    "video_id": frame["video_id"],
                    "shot_id": frame["shot_id"],
                    "local_idx": frame["local_idx"],
                    "source_image": frame["source_image"],
                    "stratum": frame["review_stratum"],
                    "sheet_file": sheet_name,
                    "sheet_slot": offset + 1,
                    "region_count": len(frame.get("regions") or []),
                    "machine_duplicate_text_count": duplicate_text_count(frame),
                    "machine_low_confidence_region_count": low_confidence_count(frame),
                    "gt_has_text": "",
                    "bbox_quality": "",
                    "easyocr_quality": "",
                    "human_text": "",
                    "notes": "",
                }
            )
        canvas.save(frame_sheets / sheet_name, quality=92)
    review_csv = output_dir / "manual-review.csv"
    with review_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(review_rows)
    frame_sheet_zip = output_dir / "frame-review-sheets.zip"
    with zipfile.ZipFile(frame_sheet_zip, "w", zipfile.ZIP_DEFLATED) as archive_file:
        for sheet in sorted(frame_sheets.glob("*.jpg")):
            archive_file.write(sheet, sheet.name)

    recognition_rows: list[dict[str, Any]] = []
    for page_start in range(0, len(recognition_regions), 12):
        page = recognition_regions[page_start : page_start + 12]
        sheet_name = f"crops-{page_start // 12 + 1:02d}.jpg"
        canvas = Image.new("RGB", (1500, 720), "white")
        for offset, item in enumerate(page):
            frame = item["frame"]
            region = item["region"]
            panel = draw_crop_panel(item, source_path(frame, index))
            canvas.paste(panel, ((offset % 3) * 500, (offset // 3) * 180))
            row: dict[str, Any] = {
                "sample_index": page_start + offset + 1,
                "region_id": region["region_id"],
                "keyframe_uid": frame["keyframe_uid"],
                "video_id": frame["video_id"],
                "shot_id": frame["shot_id"],
                "source_image": frame["source_image"],
                "frame_stratum": frame["review_stratum"],
                "region_stratum": item["region_stratum"],
                "bbox_px": json.dumps(region["bbox_px"], separators=(",", ":")),
                "crop_width": region.get("crop_width"),
                "crop_height": region.get("crop_height"),
                "easyocr_text": region.get("easyocr_text") or "",
                "easyocr_confidence": float(region.get("easyocr_confidence") or 0.0),
                "crop_spec_id": CROP_SPEC["id"],
                "sheet_file": sheet_name,
                "sheet_slot": offset + 1,
            }
            row["sample_row_sha256"] = immutable_region_hash(row)
            recognition_rows.append(row)
        canvas.save(crop_sheets / sheet_name, quality=95)

    sample_jsonl = output_dir / "recognition-sample.jsonl"
    with sample_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for row in recognition_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    recognition_csv = output_dir / "recognition-ground-truth.csv"
    with recognition_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RECOGNITION_LABEL_COLUMNS)
        writer.writeheader()
        for row in recognition_rows:
            writer.writerow(
                {
                    **row,
                    "label_status": "",
                    "human_text": "",
                    "text_type": "",
                    "notes": "",
                }
            )
    crop_sheet_zip = output_dir / "recognition-crop-sheets.zip"
    with zipfile.ZipFile(crop_sheet_zip, "w", zipfile.ZIP_DEFLATED) as archive_file:
        for sheet in sorted(crop_sheets.glob("*.jpg")):
            archive_file.write(sheet, sheet.name)
    report = {
        "schema_version": 2,
        "created_utc": utc_now(),
        "decision": "PENDING_HUMAN_VISUAL_REVIEW",
        "scope": "ocr_v2_baseline_triage_no_new_inference",
        "input": {
            "archive": str(archive),
            "archive_sha256": sha256_path(archive),
            "batch_id": manifest["batch_id"],
            "catalog_sha256": manifest["catalog_sha256"],
            "frames": manifest["frames"],
            "regions": manifest["regions"],
            "vintern_candidates": manifest["vintern_candidates"],
            "vintern_results_available": False,
        },
        "sample": {
            "seed": seed,
            "frames": len(review_rows),
            "uid_unique": len({int(row["keyframe_uid"]) for row in review_rows}) == len(review_rows),
            "strata": dict(sorted(Counter(row["stratum"] for row in review_rows).items())),
            "videos": dict(sorted(Counter(row["video_id"] for row in review_rows).items())),
            "recognition_regions": len(recognition_rows),
            "recognition_region_unique": len({row["region_id"] for row in recognition_rows})
            == len(recognition_rows),
            "recognition_strata": dict(
                sorted(Counter(row["region_stratum"] for row in recognition_rows).items())
            ),
            "recognition_videos": dict(
                sorted(Counter(row["video_id"] for row in recognition_rows).items())
            ),
            "max_regions_per_frame": max(
                Counter(int(row["keyframe_uid"]) for row in recognition_rows).values()
            ),
            "crop_spec": CROP_SPEC,
        },
        "artifacts": {
            "manual_review_csv": review_csv.name,
            "manual_review_csv_sha256": sha256_file(review_csv),
            "frame_sheets_zip": frame_sheet_zip.name,
            "frame_sheets_zip_sha256": sha256_file(frame_sheet_zip),
            "recognition_sample_jsonl": sample_jsonl.name,
            "recognition_sample_jsonl_sha256": sha256_file(sample_jsonl),
            "recognition_ground_truth_csv": recognition_csv.name,
            "recognition_ground_truth_template_sha256": sha256_file(recognition_csv),
            "recognition_crop_sheets_zip": crop_sheet_zip.name,
            "recognition_crop_sheets_zip_sha256": sha256_file(crop_sheet_zip),
        },
        "review_instructions": {
            "gt_has_text": "yes|no",
            "bbox_quality": "correct|miss|duplicate|wrong",
            "easyocr_quality": "correct|near|wrong|empty|not_applicable",
            "human_text": "transcribe only when readable; leave blank for unreadable or no text",
            "notes": "record ticker, Vietnamese diacritic, overlap, reading-order, or other evidence",
            "recognition_label_status": "labeled|exclude_unreadable|false_positive",
            "recognition_human_text": "exact crop transcription for labeled rows only",
            "recognition_text_type": "ordinary|ticker|numeric_or_name|other",
        },
        "limitations": [
            "The attached production archive has no Vintern result rows; it cannot establish Vintern accuracy.",
            "Machine duplicate counts are triage hints, not ground truth.",
            "This bundle does not alter OCR production artifacts or authorize a model change.",
        ],
    }
    report_path = output_dir / "review-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    bundle = output_dir / "ocr-v2-review-bundle.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive_file:
        for path in (
            review_csv,
            frame_sheet_zip,
            sample_jsonl,
            recognition_csv,
            crop_sheet_zip,
            report_path,
        ):
            archive_file.write(path, path.name)
    return {**report, "artifacts": {**report["artifacts"], "bundle": bundle.name, "bundle_sha256": sha256_file(bundle)}}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--keyframe-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--region-sample-size", type=int, default=120)
    parser.add_argument("--seed", default="ocr-v2-batch-01-visual-review-v1")
    population = parser.add_mutually_exclusive_group()
    population.add_argument(
        "--video-id",
        default=None,
        help="restrict the review population to one video; useful for a fast preliminary gate",
    )
    population.add_argument(
        "--video-ids",
        nargs="+",
        default=None,
        help="choose an equal number of frames from each listed video (for example five videos x 20)",
    )
    args = parser.parse_args(argv)
    manifest, frames, _ = read_archive(args.archive)
    if args.video_id is not None:
        frames = [frame for frame in frames if frame.get("video_id") == args.video_id]
        if len(frames) < args.sample_size:
            raise RuntimeError(
                f"video_id={args.video_id!r} has only {len(frames)} frames; "
                f"need {args.sample_size}"
            )
    if args.video_ids is not None:
        selected = choose_balanced_review_frames(
            frames,
            video_ids=args.video_ids,
            sample_size=args.sample_size,
            seed=args.seed,
        )
    else:
        selected = choose_review_frames(frames, sample_size=args.sample_size, seed=args.seed)
    if args.video_ids is not None:
        recognition_regions = choose_balanced_recognition_regions(
            selected,
            video_ids=args.video_ids,
            sample_size=args.region_sample_size,
            seed=f"{args.seed}|recognition",
        )
    else:
        recognition_regions = choose_recognition_regions(
            selected,
            sample_size=args.region_sample_size,
            seed=f"{args.seed}|recognition",
        )
    report = write_bundle(
        selected,
        recognition_regions,
        archive=args.archive,
        manifest=manifest,
        keyframe_root=args.keyframe_root,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
