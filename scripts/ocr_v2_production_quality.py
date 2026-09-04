"""Build and score a blind human audit of OCR-v2 production selections.

The builder reads the nine immutable production ZIPs selected by the local source
manifest.  It downloads only the sampled Kaggle JPEGs, verifies their recorded
content hashes, and renders crops without showing any model text.  The scorer is
model-free and compares human transcription with cached EasyOCR, raw VietOCR,
conditional Paddle, and the final production ``selected_text`` policy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import os
import ssl
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import certifi
from PIL import Image, ImageOps

from offline.artifacts import sha256_file
from offline.ocr_v2_gate import levenshtein, normalize_text, token_list
from offline.ocr_v2_snapshot import (
    BATCH_IDS,
    _source_local_path,
    canonical_json,
    load_source_manifest,
    result_member_hashes,
)
from scripts.ocr_v2_review_bundle import allocate_quotas, font, rectify_region_crop


DEFAULT_DATASET = "thvu165/aic-2026-keyframes"
DEFAULT_SEED = "ocr-v2-production-blind-audit-v1"
STRATA = ("unresolved", "paddle", "vietocr_changed", "vietocr_agree")
STRATUM_WEIGHTS = {
    "unresolved": 0.30,
    "paddle": 0.20,
    "vietocr_changed": 0.30,
    "vietocr_agree": 0.20,
}
LABEL_COLUMNS = (
    "sample_index",
    "sample_row_sha256",
    "batch_id",
    "video_id",
    "frame_id",
    "keyframe_uid",
    "region_id",
    "audit_stratum",
    "sheet_file",
    "sheet_slot",
    "label_status",
    "human_text",
    "text_type",
    "notes",
)
MUTABLE_LABEL_COLUMNS = {"label_status", "human_text", "text_type", "notes"}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sample_hash(row: dict[str, Any]) -> str:
    return _sha256_bytes(canonical_json(row))


def _stratum(region: dict[str, Any]) -> str:
    engine = region.get("selected_engine")
    if engine is None or region.get("selection") == "unresolved":
        return "unresolved"
    if engine == "paddle":
        return "paddle"
    if engine != "vietocr":
        raise ValueError(f"unknown selected engine: {engine!r}")
    return (
        "vietocr_agree"
        if normalize_text(region.get("selected_text"))
        == normalize_text(region.get("easyocr_text"))
        else "vietocr_changed"
    )


class _TopK:
    """Keep the deterministically smallest hashes without retaining the catalog."""

    def __init__(self, limit: int, salt: str):
        self.limit = limit
        self.salt = salt
        self.heap: list[tuple[int, str, dict[str, Any]]] = []

    def offer(self, identity: str, row: dict[str, Any]) -> None:
        rank = int.from_bytes(
            hashlib.sha256(f"{self.salt}|{identity}".encode()).digest()[:8], "big"
        )
        entry = (-rank, identity, row)
        if len(self.heap) < self.limit:
            heapq.heappush(self.heap, entry)
        elif rank < -self.heap[0][0]:
            heapq.heapreplace(self.heap, entry)

    def rows(self) -> list[dict[str, Any]]:
        return [
            entry[2]
            for entry in sorted(
                self.heap,
                key=lambda value: (-value[0], value[1]),
            )
        ]


def _jsonl_rows(archive: zipfile.ZipFile, name: str):
    with archive.open(name) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid {name} row {line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"non-object {name} row {line_number}")
            yield row


def _select_batch_regions(
    archive_path: Path,
    *,
    batch_id: str,
    per_batch: int,
    seed: str,
) -> tuple[list[dict[str, Any]], Counter[str], dict[str, Any]]:
    quotas = allocate_quotas(STRATUM_WEIGHTS, per_batch)
    pools = {
        name: _TopK(max(quotas[name] * 8, 64), f"{seed}|{batch_id}|{name}")
        for name in STRATA
    }
    fallback = _TopK(max(per_batch * 20, 400), f"{seed}|{batch_id}|fallback")
    population: Counter[str] = Counter()
    with zipfile.ZipFile(archive_path) as archive:
        report = json.loads(archive.read("report.json"))
        for frame in _jsonl_rows(archive, "frame-selections.jsonl"):
            for raw_region in frame.get("regions") or []:
                region = {
                    **raw_region,
                    "batch_id": batch_id,
                    "frame_status": frame.get("status"),
                }
                name = _stratum(region)
                identity = str(region.get("region_id") or "")
                if not identity:
                    raise ValueError(f"blank region_id in {batch_id}")
                population[name] += 1
                pools[name].offer(identity, region)
                fallback.offer(identity, region)

        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()
        for name in STRATA:
            accepted = 0
            for region in pools[name].rows():
                identity = str(region["region_id"])
                if identity in selected_ids:
                    continue
                selected.append({**region, "audit_stratum": name})
                selected_ids.add(identity)
                accepted += 1
                if accepted == quotas[name]:
                    break
        for region in fallback.rows():
            if len(selected) == per_batch:
                break
            identity = str(region["region_id"])
            if identity not in selected_ids:
                selected.append({**region, "audit_stratum": _stratum(region)})
                selected_ids.add(identity)
        if len(selected) != per_batch:
            raise RuntimeError(
                f"{batch_id} has only {len(selected)} auditable regions; need {per_batch}"
            )

        wanted = set(selected_ids)
        predictions: dict[tuple[str, str], dict[str, Any]] = {}
        for prediction in _jsonl_rows(archive, "predictions.jsonl"):
            region_id = str(prediction.get("region_id") or "")
            if region_id not in wanted:
                continue
            key = (region_id, str(prediction.get("model") or ""))
            if key in predictions:
                raise ValueError(f"duplicate sampled prediction: {key}")
            predictions[key] = prediction

    for region in selected:
        region_id = str(region["region_id"])
        vietocr = predictions.get((region_id, "vietocr"))
        if vietocr is None:
            raise ValueError(f"missing VietOCR prediction for {region_id}")
        paddle = predictions.get((region_id, "paddle"))
        region["raw_vietocr_text"] = vietocr.get("text") or ""
        region["raw_vietocr_confidence"] = vietocr.get("confidence")
        region["raw_paddle_text"] = "" if paddle is None else paddle.get("text") or ""
        region["raw_paddle_confidence"] = None if paddle is None else paddle.get("confidence")
        region["paddle_ran"] = paddle is not None
    return selected, population, report


def _dataset_relative(source_image: str, dataset: str) -> str:
    marker = f"datasets/{dataset}/"
    normalized = source_image.replace("\\", "/")
    if marker not in normalized:
        raise ValueError(
            f"source image does not belong to Kaggle dataset {dataset!r}: {source_image!r}"
        )
    relative = normalized.split(marker, 1)[1]
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("unsafe Kaggle image path")
    return str(path)


def _download_image(
    *, dataset: str, relative: str, destination: Path, expected_sha256: str
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256_file(destination) == expected_sha256:
        return
    url = (
        "https://www.kaggle.com/api/v1/datasets/download/"
        + dataset
        + "/"
        # Kaggle treats the dataset-internal path as one route parameter.  A raw
        # slash starts another route segment and returns 404 for nested files.
        + urllib.parse.quote(relative, safe="")
    )
    request = urllib.request.Request(url, headers={"User-Agent": "LASTDANCE-OCR-audit/1"})
    context = ssl.create_default_context(cafile=certifi.where())
    temporary = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(request, context=context, timeout=120) as response, temporary.open(
            "wb"
        ) as handle:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                digest.update(block)
                handle.write(block)
        if digest.hexdigest() != expected_sha256:
            raise ValueError(f"Kaggle JPEG checksum mismatch: {relative}")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _render_blind_sheets(
    rows: list[dict[str, Any]], image_paths: dict[str, Path], output_dir: Path
) -> None:
    sheets = output_dir / "blind-crop-sheets"
    sheets.mkdir(parents=True)
    label_font = font(13)
    for page_start in range(0, len(rows), 12):
        page = rows[page_start : page_start + 12]
        sheet_name = f"crops-{page_start // 12 + 1:02d}.jpg"
        canvas = Image.new("RGB", (1500, 720), "white")
        for offset, row in enumerate(page):
            with Image.open(image_paths[row["source_image"]]) as source:
                crop = rectify_region_crop(ImageOps.exif_transpose(source).convert("RGB"), row["bbox_px"])
            crop.thumbnail((480, 135))
            if crop.width < 180 or crop.height < 45:
                scale = min(480 / crop.width, 135 / crop.height)
                if scale > 1:
                    crop = crop.resize(
                        (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
                        Image.Resampling.NEAREST,
                    )
            panel = Image.new("RGB", (500, 180), "white")
            panel.paste(crop, ((500 - crop.width) // 2, 2))
            text = (
                f"#{row['sample_index']:03d} | {row['batch_id']} | "
                f"{row['audit_stratum']} | id={row['region_id']}"
            )
            from PIL import ImageDraw

            ImageDraw.Draw(panel).text((7, 151), text, fill="black", font=label_font)
            canvas.paste(panel, ((offset % 3) * 500, (offset // 3) * 180))
            row["sheet_file"] = sheet_name
            row["sheet_slot"] = offset + 1
        canvas.save(sheets / sheet_name, quality=95)
    with zipfile.ZipFile(output_dir / "blind-crop-sheets.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for sheet in sorted(sheets.glob("*.jpg")):
            archive.write(sheet, sheet.name)


def build_bundle(
    *,
    source_manifest_path: Path,
    source_root: Path | None,
    output_dir: Path,
    dataset: str,
    sample_size: int,
    seed: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if sample_size < 90 or sample_size % len(BATCH_IDS) != 0:
        raise ValueError("sample size must be >=90 and divisible by nine batches")
    sources = load_source_manifest(source_manifest_path)
    root = source_root or source_manifest_path.parent
    per_batch = sample_size // len(BATCH_IDS)
    selected: list[dict[str, Any]] = []
    populations: dict[str, dict[str, int]] = {}
    reports: dict[str, dict[str, Any]] = {}
    for artifact in sorted(sources.artifacts, key=lambda value: value.batch_id):
        archive_path = _source_local_path(root, artifact.result_path)
        if archive_path.stat().st_size != artifact.result_bytes:
            raise ValueError(f"source size mismatch: {artifact.batch_id}")
        if sha256_file(archive_path) != artifact.result_sha256:
            raise ValueError(f"source SHA-256 mismatch: {artifact.batch_id}")
        result_member_hashes(archive_path, verify_members=True)
        batch_rows, population, report = _select_batch_regions(
            archive_path,
            batch_id=artifact.batch_id,
            per_batch=per_batch,
            seed=seed,
        )
        selected.extend(batch_rows)
        populations[artifact.batch_id] = {name: population[name] for name in STRATA}
        reports[artifact.batch_id] = report

    output_dir.mkdir(parents=True)
    images_dir = output_dir / "source-images"
    image_paths: dict[str, Path] = {}
    image_hashes: dict[str, str] = {}
    for row in selected:
        source_image = str(row["source_image"])
        expected = str(row["source_sha256"])
        if source_image in image_hashes and image_hashes[source_image] != expected:
            raise ValueError(f"conflicting source hash: {source_image}")
        image_hashes[source_image] = expected
    for number, (source_image, expected) in enumerate(sorted(image_hashes.items()), start=1):
        relative = _dataset_relative(source_image, dataset)
        destination = images_dir.joinpath(*PurePosixPath(relative).parts)
        print(f"DOWNLOAD {number}/{len(image_hashes)} {relative}")
        _download_image(
            dataset=dataset,
            relative=relative,
            destination=destination,
            expected_sha256=expected,
        )
        image_paths[source_image] = destination

    selected.sort(key=lambda row: (row["batch_id"], row["audit_stratum"], row["region_id"]))
    sample_rows: list[dict[str, Any]] = []
    machine_rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        sample = {
            "sample_index": index,
            "batch_id": row["batch_id"],
            "video_id": row["video_id"],
            "frame_id": row["frame_id"],
            "keyframe_uid": row["keyframe_uid"],
            "region_id": row["region_id"],
            "audit_stratum": row["audit_stratum"],
            "source_image": row["source_image"],
            "source_sha256": row["source_sha256"],
            "bbox_px": row["bbox_px"],
        }
        sample["sample_row_sha256"] = _sample_hash(sample)
        sample_rows.append(sample)
        machine_rows.append(
            {
                "sample_index": index,
                "sample_row_sha256": sample["sample_row_sha256"],
                "region_id": row["region_id"],
                "easyocr_text": row.get("easyocr_text") or "",
                "vietocr_text": row["raw_vietocr_text"],
                "vietocr_confidence": row["raw_vietocr_confidence"],
                "paddle_ran": row["paddle_ran"],
                "paddle_text": row["raw_paddle_text"],
                "paddle_confidence": row["raw_paddle_confidence"],
                "selected_engine": row.get("selected_engine"),
                "selected_text": row.get("selected_text") or "",
                "selected_confidence": row.get("selected_confidence"),
                "selection": row.get("selection"),
                "residual_reasons": row.get("residual_reasons") or [],
            }
        )
    _render_blind_sheets(sample_rows, image_paths, output_dir)

    sample_path = output_dir / "production-sample.jsonl"
    machine_path = output_dir / "machine-results.jsonl"
    sample_path.write_bytes(b"".join(canonical_json(row) + b"\n" for row in sample_rows))
    machine_path.write_bytes(b"".join(canonical_json(row) + b"\n" for row in machine_rows))
    labels_path = output_dir / "ground-truth.csv"
    with labels_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        for row in sample_rows:
            writer.writerow(
                {
                    **{column: row.get(column, "") for column in LABEL_COLUMNS},
                    "label_status": "",
                    "human_text": "",
                    "text_type": "",
                    "notes": "",
                }
            )
    report = {
        "schema_version": 1,
        "artifact_kind": "ocr_v2_production_blind_quality_audit",
        "created_utc": _utc_now(),
        "decision": "PENDING_HUMAN_GROUND_TRUTH",
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_revision": sources.revision,
        "dataset": dataset,
        "seed": seed,
        "sample_regions": len(sample_rows),
        "sample_per_batch": per_batch,
        "sample_counts": dict(sorted(Counter(row["audit_stratum"] for row in sample_rows).items())),
        "population_counts_by_batch": populations,
        "production_totals": {
            "frames": sum(int(report["frames"]) for report in reports.values()),
            "regions": sum(int(report["regions"]) for report in reports.values()),
            "error_frames": sum(int(report["status"].get("error", 0)) for report in reports.values()),
            "no_text_frames": sum(int(report["status"].get("no_text", 0)) for report in reports.values()),
            "residual_regions": sum(int(report["residual_regions"]) for report in reports.values()),
            "selected_engine_counts": {
                "vietocr": sum(
                    counts["vietocr_changed"] + counts["vietocr_agree"]
                    for counts in populations.values()
                ),
                "paddle": sum(counts["paddle"] for counts in populations.values()),
                "unresolved": sum(counts["unresolved"] for counts in populations.values()),
            },
        },
        "artifacts": {
            "sample": {"path": sample_path.name, "sha256": sha256_file(sample_path)},
            "machine_results": {"path": machine_path.name, "sha256": sha256_file(machine_path)},
            "ground_truth": {"path": labels_path.name, "sha256": sha256_file(labels_path)},
            "blind_sheets": {
                "path": "blind-crop-sheets.zip",
                "sha256": sha256_file(output_dir / "blind-crop-sheets.zip"),
            },
        },
        "limitations": [
            "The audit is stratified toward difficult production selections, not a simple random catalog sample.",
            "CRAFT detection quality is unchanged and is not re-estimated by this crop recognizer audit.",
            "No model output is shown on the blind crop sheets; machine text is sealed in machine-results.jsonl.",
        ],
    }
    report_path = output_dir / "audit-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _metrics(rows: list[dict[str, Any]], prediction_field: str) -> dict[str, Any]:
    readable = [row for row in rows if row["label_status"] == "labeled"]
    evaluated = [row for row in rows if row["label_status"] != "exclude_unreadable"]
    exact = sum(
        normalize_text(row[prediction_field]) == normalize_text(row["human_text"])
        for row in evaluated
    )
    edits = sum(
        levenshtein(normalize_text(row[prediction_field]), normalize_text(row["human_text"]))
        for row in readable
    )
    characters = sum(len(normalize_text(row["human_text"])) for row in readable)
    matched_tokens = 0
    truth_tokens = 0
    for row in readable:
        truth = Counter(token_list(row["human_text"]))
        prediction = Counter(token_list(row[prediction_field]))
        matched_tokens += sum((truth & prediction).values())
        truth_tokens += sum(truth.values())
    numeric = [
        row
        for row in readable
        if row["text_type"] == "numeric_or_name"
        or any(character.isdigit() for character in row["human_text"])
    ]
    return {
        "evaluated_regions": len(evaluated),
        "readable_regions": len(readable),
        "normalized_exact_line_accuracy": exact / max(len(evaluated), 1),
        "exact_token_recall": matched_tokens / max(truth_tokens, 1),
        "cer": edits / max(characters, 1),
        "empty_prediction_rate_on_readable": sum(
            not normalize_text(row[prediction_field]) for row in readable
        )
        / max(len(readable), 1),
        "numeric_name_regions": len(numeric),
        "numeric_name_exact_accuracy": sum(
            normalize_text(row[prediction_field]) == normalize_text(row["human_text"])
            for row in numeric
        )
        / max(len(numeric), 1),
    }


def score_bundle(*, review_dir: Path, output: Path) -> dict[str, Any]:
    audit = json.loads((review_dir / "audit-report.json").read_text(encoding="utf-8"))
    sample_path = review_dir / audit["artifacts"]["sample"]["path"]
    machine_path = review_dir / audit["artifacts"]["machine_results"]["path"]
    if sha256_file(sample_path) != audit["artifacts"]["sample"]["sha256"]:
        raise ValueError("immutable production sample changed")
    if sha256_file(machine_path) != audit["artifacts"]["machine_results"]["sha256"]:
        raise ValueError("sealed machine results changed")
    samples = {str(row["region_id"]): row for row in _read_jsonl(sample_path)}
    machines = {str(row["region_id"]): row for row in _read_jsonl(machine_path)}
    with (review_dir / "ground-truth.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        labels = list(csv.DictReader(handle))
    if len(labels) != len(samples) or {row["region_id"] for row in labels} != set(samples):
        raise ValueError("ground-truth rows differ from the immutable sample")
    joined: list[dict[str, Any]] = []
    for row in labels:
        region_id = row["region_id"]
        sample = samples[region_id]
        expected_label = {
            column: str(sample.get(column, ""))
            for column in LABEL_COLUMNS
            if column not in MUTABLE_LABEL_COLUMNS
        }
        for column, expected in expected_label.items():
            if str(row.get(column, "")) != expected:
                raise ValueError(f"immutable label column changed: {region_id} {column}")
        status = row["label_status"].strip().casefold()
        human_text = row["human_text"].strip()
        text_type = row["text_type"].strip().casefold()
        if status not in {"labeled", "exclude_unreadable", "false_positive"}:
            raise ValueError(f"invalid/blank label_status: {region_id}")
        if status == "labeled" and (not human_text or text_type not in {"ordinary", "ticker", "numeric_or_name", "other"}):
            raise ValueError(f"incomplete readable label: {region_id}")
        if status != "labeled" and human_text:
            raise ValueError(f"excluded/false-positive label must have blank human_text: {region_id}")
        machine = machines[region_id]
        joined.append(
            {
                **row,
                "label_status": status,
                "human_text": human_text,
                "text_type": text_type,
                "easyocr": machine["easyocr_text"],
                "vietocr": machine["vietocr_text"],
                "paddle": machine["paddle_text"],
                "production_selected": machine["selected_text"],
                "paddle_ran": bool(machine["paddle_ran"]),
            }
        )
    readable = sum(row["label_status"] == "labeled" for row in joined)
    numeric = sum(
        row["label_status"] == "labeled"
        and (row["text_type"] == "numeric_or_name" or any(c.isdigit() for c in row["human_text"]))
        for row in joined
    )
    if readable < 150:
        raise ValueError(f"need at least 150 readable regions; got {readable}")
    if numeric < 5:
        raise ValueError(f"need at least five numeric/name regions; got {numeric}")
    metrics = {
        "easyocr_cached": _metrics(joined, "easyocr"),
        "vietocr_raw": _metrics(joined, "vietocr"),
        "production_selected": _metrics(joined, "production_selected"),
        "paddle_when_called": _metrics([row for row in joined if row["paddle_ran"]], "paddle"),
    }
    baseline = metrics["easyocr_cached"]
    selected = metrics["production_selected"]
    token_gain = selected["exact_token_recall"] - baseline["exact_token_recall"]
    cer_reduction = (
        0.0 if baseline["cer"] == 0 else (baseline["cer"] - selected["cer"]) / baseline["cer"]
    )
    numeric_regression = (
        baseline["numeric_name_exact_accuracy"] - selected["numeric_name_exact_accuracy"]
    )
    quality_pass = token_gain >= 0.05 or cer_reduction >= 0.10
    numeric_pass = numeric_regression <= 0.02
    report = {
        "schema_version": 1,
        "artifact_kind": "ocr_v2_production_quality_score",
        "created_utc": _utc_now(),
        "decision": "PASS_BLIND_SAMPLE" if quality_pass and numeric_pass else "FAIL_BLIND_SAMPLE",
        "sample_regions": len(joined),
        "readable_regions": readable,
        "excluded_unreadable": sum(row["label_status"] == "exclude_unreadable" for row in joined),
        "false_positive_regions": sum(row["label_status"] == "false_positive" for row in joined),
        "metrics": metrics,
        "production_vs_easyocr": {
            "quality_pass": quality_pass,
            "numeric_name_pass": numeric_pass,
            "exact_token_recall_absolute_gain": token_gain,
            "cer_relative_reduction": cer_reduction,
            "numeric_name_absolute_regression": numeric_regression,
            "thresholds": {
                "token_recall_min_absolute_gain": 0.05,
                "cer_min_relative_reduction": 0.10,
                "numeric_name_max_absolute_regression": 0.02,
            },
        },
        "source_audit_report_sha256": sha256_file(review_dir / "audit-report.json"),
        "ground_truth_sha256": sha256_file(review_dir / "ground-truth.csv"),
        "limitations": audit["limitations"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    build = subparsers.add_parser("build", help="build blind sheets and label template")
    build.add_argument("--source-manifest", type=Path, required=True)
    build.add_argument("--source-root", type=Path)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--dataset", default=DEFAULT_DATASET)
    build.add_argument("--sample-size", type=int, default=180)
    build.add_argument("--seed", default=DEFAULT_SEED)
    score = subparsers.add_parser("score", help="score a completed ground-truth.csv")
    score.add_argument("--review-dir", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "build":
        build_bundle(
            source_manifest_path=args.source_manifest,
            source_root=args.source_root,
            output_dir=args.output_dir,
            dataset=args.dataset,
            sample_size=args.sample_size,
            seed=args.seed,
        )
    else:
        score_bundle(review_dir=args.review_dir, output=args.output)


if __name__ == "__main__":
    main()
