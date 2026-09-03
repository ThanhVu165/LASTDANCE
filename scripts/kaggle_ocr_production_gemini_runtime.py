"""Locked Kaggle runtime for Gemini 2.5 Flash-Lite OCR residuals."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import random
import shutil
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, ImageDraw, ImageFont


EXECUTION_MODE = str(globals().get("EXECUTION_MODE", "preflight"))
APPROVE_PAID_CANARY = bool(globals().get("APPROVE_PAID_CANARY", False))
APPROVE_GEMINI_PRODUCTION = bool(globals().get("APPROVE_GEMINI_PRODUCTION", False))
APPROVED_REPORT_SHA256 = str(globals().get("APPROVED_REPORT_SHA256", ""))
APPROVED_MODEL_VERSION = str(globals().get("APPROVED_MODEL_VERSION", ""))
APPROVED_MAX_REQUESTS = int(globals().get("APPROVED_MAX_REQUESTS", 0))
APPROVED_MAX_VND = int(globals().get("APPROVED_MAX_VND", 0))
CANARY_REQUESTS = int(globals().get("CANARY_REQUESTS", 100))
REQUESTS_PER_MINUTE = float(globals().get("REQUESTS_PER_MINUTE", 30.0))
PUBLISH_TO_HF = bool(globals().get("PUBLISH_TO_HF", True))
HF_REPO_ID = str(globals().get("HF_REPO_ID", "MinhThuw0103/lastdance-visual-embeddings"))

MODEL_ID = "gemini-2.5-flash-lite"
RUNNER_BILLING_MODE = "standard"
MEDIA_RESOLUTION = "MEDIA_RESOLUTION_MEDIUM"
MAX_BUDGET_VND = 400_000
STANDARD_INPUT_USD_PER_MILLION = 0.10
STANDARD_OUTPUT_USD_PER_MILLION = 0.40
USD_TO_VND = 26_300
PROGRESS_EVERY = 25
CHECKPOINT_EVERY = 100
MAX_ATTEMPTS = 3
INPUT_ROOT = Path("/kaggle/input")
OUTPUT_ROOT = Path("/kaggle/working/ocr-production-gemini-v1")
PREFLIGHT_MEMBERS = {
    "ocr-gemini-preflight-report.json",
    "gemini-residual-regions.jsonl",
    "gemini-request-manifest.jsonl",
    "SHA256SUMS",
}
OUTPUT_MEMBERS = {
    "gemini-results.jsonl",
    "attempts-history.jsonl",
    "run-signature.json",
    "batch-manifest.json",
    "SHA256SUMS",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = Path(f"{path}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def rewrite_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = Path(f"{path}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_jsonl(path: Path, identity: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[object] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            value = row[identity]
            if value in seen:
                raise ValueError(f"duplicate {identity}: {value}")
            seen.add(value)
            rows.append(row)
    return rows


def find_unique_input(name: str) -> Path:
    matches: list[Path] = []
    for pattern in (name, f"*/{name}", f"*/*/{name}", f"*/*/*/{name}"):
        matches.extend(path.resolve() for path in INPUT_ROOT.glob(pattern) if path.is_file())
    values = sorted(set(matches))
    if len(values) != 1:
        raise RuntimeError(f"attach exactly one {name}: {values}")
    return values[0]


def find_batch_root(batch_id: str) -> Path:
    directory_name = f"keyframes-{batch_id}"
    matches: list[Path] = []
    for pattern in (directory_name, f"*/{directory_name}", f"*/*/{directory_name}", f"*/*/*/{directory_name}"):
        matches.extend(path.resolve() for path in INPUT_ROOT.glob(pattern) if path.is_dir())
    values = sorted(set(matches))
    if not values:
        values = sorted(path.resolve() for path in INPUT_ROOT.rglob(directory_name) if path.is_dir())
    if len(values) != 1:
        raise RuntimeError(f"expected exactly one {directory_name}: {values}")
    return values[0]


def parse_sums(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        if len(digest) != 64 or Path(name).name != name or name in result:
            raise RuntimeError("invalid preflight SHA256SUMS")
        result[name] = digest
    return result


def load_preflight() -> tuple[Path, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    source = find_unique_input("ocr-gemini-preflight.zip")
    target = OUTPUT_ROOT / "preflight"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    with zipfile.ZipFile(source) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("preflight ZIP CRC failure")
        names = archive.namelist()
        if set(names) != PREFLIGHT_MEMBERS or len(names) != len(set(names)):
            raise RuntimeError(f"preflight ZIP member mismatch: {names}")
        if any(Path(name).name != name for name in names):
            raise RuntimeError("preflight ZIP contains unsafe paths")
        archive.extractall(target)
    sums = parse_sums(target / "SHA256SUMS")
    for name, expected in sums.items():
        if not (target / name).is_file() or sha256_file(target / name) != expected:
            raise RuntimeError(f"preflight checksum mismatch: {name}")
    report_path = target / "ocr-gemini-preflight-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    residuals = read_jsonl(target / "gemini-residual-regions.jsonl", "region_id")
    requests = read_jsonl(target / "gemini-request-manifest.jsonl", "request_id")
    expected_ids = {row["region_id"] for row in residuals}
    requested_ids = [region_id for row in requests for region_id in row["region_ids"]]
    if set(requested_ids) != expected_ids or len(requested_ids) != len(expected_ids):
        raise RuntimeError("preflight request/residual partition mismatch")
    counts = report["exact_counts"]
    if not (
        counts["regions"] == len(residuals)
        and counts["requests"] == len(requests)
        and counts["frames"] == len({row["keyframe_uid"] for row in residuals})
    ):
        raise RuntimeError("preflight exact-count report mismatch")
    print("GEMINI_PREFLIGHT_VERIFIED", counts, {"report_sha256": sha256_file(report_path)}, flush=True)
    return report_path, report, residuals, requests


def font(size: int):
    for candidate in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def build_contact_sheets(request: dict[str, Any], residual_by_id: dict[str, dict[str, Any]], batch_root: Path) -> list[bytes]:
    outputs: list[bytes] = []
    image_cache: dict[tuple[str, str], Image.Image] = {}
    label_font = font(19)
    try:
        for page_number, region_ids in enumerate(request["region_pages"], start=1):
            cell_width, cell_height, columns = 720, 180, 2
            rows = math.ceil(len(region_ids) / columns)
            sheet = Image.new("RGB", (cell_width * columns, cell_height * rows), "white")
            draw = ImageDraw.Draw(sheet)
            for slot, region_id in enumerate(region_ids):
                region = residual_by_id[region_id]
                filename = Path(region["source_image"]).name
                cache_key = (region["video_id"], filename)
                if cache_key not in image_cache:
                    source = batch_root / region["video_id"] / filename
                    if not source.is_file():
                        raise FileNotFoundError(f"missing keyframe for Gemini crop: {source}")
                    with Image.open(source) as handle:
                        image_cache[cache_key] = handle.convert("RGB")
                source_image = image_cache[cache_key]
                bbox = region["bbox_px"]
                xs = [float(value) for value in bbox[0::2]]
                ys = [float(value) for value in bbox[1::2]]
                x1 = max(0, int(math.floor(min(xs))) - 6)
                y1 = max(0, int(math.floor(min(ys))) - 6)
                x2 = min(source_image.width, int(math.ceil(max(xs))) + 6)
                y2 = min(source_image.height, int(math.ceil(max(ys))) + 6)
                if x2 <= x1 or y2 <= y1:
                    raise ValueError(f"invalid Gemini crop bbox: {region_id}")
                crop = source_image.crop((x1, y1, x2, y2))
                crop.thumbnail((cell_width - 20, cell_height - 40), Image.Resampling.LANCZOS)
                column, row = slot % columns, slot // columns
                left, top = column * cell_width, row * cell_height
                draw.rectangle((left, top, left + cell_width - 1, top + cell_height - 1), outline="#777777", width=1)
                draw.text((left + 8, top + 5), f"{page_number}.{slot + 1}  {region_id}", fill="black", font=label_font)
                sheet.paste(crop, (left + (cell_width - crop.width) // 2, top + 34 + (cell_height - 40 - crop.height) // 2))
            buffer = io.BytesIO()
            sheet.save(buffer, format="JPEG", quality=92, optimize=True)
            outputs.append(buffer.getvalue())
    finally:
        for image in image_cache.values():
            image.close()
    return outputs


def response_schema(region_ids: list[str]) -> dict[str, Any]:
    if not region_ids or len(region_ids) != len(set(region_ids)):
        raise ValueError("schema requires non-empty unique region IDs")
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "regions": {
                "type": "array",
                "minItems": len(region_ids),
                "maxItems": len(region_ids),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "region_id": {"type": "string", "enum": region_ids},
                        "text": {"type": "string"},
                        "language": {"type": "string", "enum": ["vi", "en", "mixed"]},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["region_id", "text", "language", "confidence"],
                    "propertyOrdering": ["region_id", "text", "language", "confidence"],
                },
            }
        },
        "required": ["regions"],
        "propertyOrdering": ["regions"],
    }


def request_payload(request: dict[str, Any], sheets: list[bytes]) -> bytes:
    mapping = [
        {"page": page_index, "region_ids_in_visual_order": region_ids}
        for page_index, region_ids in enumerate(request["region_pages"], start=1)
    ]
    prompt = (
        "OCR every labeled crop in every attached contact sheet. The labels and exact mapping are "
        f"{mapping}. Return exactly one row for every region_id and no others. Preserve Vietnamese "
        "diacritics, original casing and visible punctuation. Do not invent hidden text. If truly "
        "unreadable, return an empty text string with low confidence. Return only JSON matching the "
        "schema; never return bbox, frame ID, keyframe UID or explanation."
    )
    parts: list[dict[str, Any]] = [{"text": prompt}]
    for sheet in sheets:
        parts.append({"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(sheet).decode("ascii")}})
    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 4096,
            "mediaResolution": MEDIA_RESOLUTION,
            "responseMimeType": "application/json",
            "responseJsonSchema": response_schema(list(request["region_ids"])),
        },
    }
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def parse_response(response: dict[str, Any], expected_ids: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    text = response["candidates"][0]["content"]["parts"][0]["text"]
    payload = json.loads(text)
    rows = payload.get("regions") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("structured response missing regions")
    actual: list[str] = []
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"region_id", "text", "language", "confidence"}:
            raise ValueError("structured response row fields mismatch")
        region_id = str(row["region_id"])
        confidence = row["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
            raise ValueError("Gemini confidence must be numeric in [0,1]")
        if row["language"] not in {"vi", "en", "mixed"}:
            raise ValueError("Gemini language must be vi/en/mixed")
        actual.append(region_id)
        normalized.append({"region_id": region_id, "text": str(row["text"]).strip(), "language": row["language"], "confidence": float(confidence)})
    if len(actual) != len(set(actual)) or set(actual) != set(expected_ids):
        raise ValueError("Gemini region_id set is duplicate/missing/foreign")
    usage = response.get("usageMetadata") or {}
    prompt_tokens = int(usage.get("promptTokenCount", 0))
    total_tokens = int(usage.get("totalTokenCount", 0))
    candidate_tokens = int(usage.get("candidatesTokenCount", 0))
    thought_tokens = int(usage.get("thoughtsTokenCount", 0))
    billed_output = max(candidate_tokens + thought_tokens, total_tokens - prompt_tokens, 0)
    return normalized, {
        "model_version": str(response.get("modelVersion") or ""),
        "prompt_tokens": prompt_tokens,
        "candidate_tokens": candidate_tokens,
        "thought_tokens": thought_tokens,
        "billed_output_tokens": billed_output,
        "total_tokens": total_tokens,
    }


class PaceLimiter:
    def __init__(self, rpm: float):
        self.interval = 60.0 / rpm
        self.next_attempt = time.monotonic()

    def wait(self):
        delay = self.next_attempt - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self.next_attempt = time.monotonic() + self.interval


def get_secret(name: str) -> str:
    from kaggle_secrets import UserSecretsClient

    value = UserSecretsClient().get_secret(name)
    if not value:
        raise RuntimeError(f"Kaggle secret {name} is missing")
    return value


def sanitize(value: object, secret: str) -> str:
    return " ".join(str(value).replace(secret, "[REDACTED]").split())[:500]


def call_gemini(body: bytes, api_key: str, limiter: PaceLimiter) -> tuple[int | None, dict[str, Any], float]:
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:generateContent"
    limiter.wait()
    started = time.perf_counter()
    try:
        with httpx.Client(timeout=90.0) as client:
            response = client.post(endpoint, content=body, headers={"Content-Type": "application/json", "x-goog-api-key": api_key})
        try:
            payload = response.json()
        except json.JSONDecodeError:
            payload = {"raw": response.text[:500]}
        return response.status_code, payload, time.perf_counter() - started
    except (httpx.HTTPError, OSError) as exc:
        return None, {"transport_error": type(exc).__name__, "message": str(exc)}, time.perf_counter() - started


def request_cost_vnd(usage: dict[str, Any]) -> float:
    usd = usage["prompt_tokens"] / 1_000_000 * STANDARD_INPUT_USD_PER_MILLION + usage["billed_output_tokens"] / 1_000_000 * STANDARD_OUTPUT_USD_PER_MILLION
    return usd * USD_TO_VND


def find_checkpoint(batch_id: str) -> Path | None:
    name = f"ocr-production-{batch_id}-gemini-checkpoint.zip"
    matches: list[Path] = []
    for pattern in (f"*/{name}", f"*/*/{name}", f"*/*/*/{name}"):
        matches.extend(path.resolve() for path in INPUT_ROOT.glob(pattern) if path.is_file())
    values = sorted(set(matches))
    if len(values) > 1:
        raise RuntimeError(f"attach at most one Gemini checkpoint: {values}")
    return values[0] if values else None


def restore_checkpoint(batch_id: str, batch_dir: Path) -> bool:
    source = find_checkpoint(batch_id)
    if source is None:
        return False
    allowed = {"gemini-results.jsonl", "attempts-history.jsonl", "run-signature.json"}
    with zipfile.ZipFile(source) as archive:
        if archive.testzip() is not None or not set(archive.namelist()) <= allowed:
            raise RuntimeError(f"{batch_id} Gemini checkpoint invalid")
        for name in archive.namelist():
            if Path(name).name != name:
                raise RuntimeError("unsafe Gemini checkpoint path")
            destination = batch_dir / name
            payload = archive.read(name)
            if destination.exists() and destination.read_bytes() != payload:
                raise RuntimeError(f"{batch_id} Gemini checkpoint conflict")
            destination.write_bytes(payload)
    print("GEMINI_CHECKPOINT_RESTORED", batch_id, source, flush=True)
    return True


def restore_completed_from_hf(
    batch_id: str,
    batch_dir: Path,
    token: str,
    *,
    report_sha: str,
    expected_request_ids: set[str],
) -> bool:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)
    remote_path = f"ocr/archives/{batch_id}/gemini/ocr-production-{batch_id}-gemini.zip"
    if remote_path not in set(api.list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset")):
        return False
    revision = str(api.repo_info(repo_id=HF_REPO_ID, repo_type="dataset").sha)
    archive_path = Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            filename=remote_path,
            revision=revision,
            token=token,
            force_download=True,
        )
    )
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != OUTPUT_MEMBERS:
            raise RuntimeError(f"{batch_id} published Gemini archive is invalid")
        if any(Path(name).name != name for name in archive.namelist()):
            raise RuntimeError(f"{batch_id} published Gemini archive has unsafe paths")
        archive.extractall(batch_dir)
    sums = parse_sums(batch_dir / "SHA256SUMS")
    for name, expected in sums.items():
        if not (batch_dir / name).is_file() or sha256_file(batch_dir / name) != expected:
            raise RuntimeError(f"{batch_id} published Gemini checksum mismatch: {name}")
    manifest = json.loads((batch_dir / "batch-manifest.json").read_text(encoding="utf-8"))
    rows = read_jsonl(batch_dir / "gemini-results.jsonl", "request_id")
    if not (
        manifest.get("complete") is True
        and manifest.get("tier") == "gemini_final"
        and manifest.get("model_id") == MODEL_ID
        and manifest.get("model_version") == APPROVED_MODEL_VERSION
        and manifest.get("preflight_report_sha256") == report_sha
        and {row["request_id"] for row in rows} == expected_request_ids
        and all(row["status"] == "success" for row in rows)
    ):
        raise RuntimeError(f"{batch_id} published Gemini provenance/completion mismatch")
    print("GEMINI_HF_BATCH_RESTORED", batch_id, revision, len(rows), flush=True)
    return True


def checkpoint_zip(batch_id: str, batch_dir: Path) -> Path:
    destination = Path(f"/kaggle/working/ocr-production-{batch_id}-gemini-checkpoint.zip")
    temporary = Path(f"{destination}.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
        for name in ("gemini-results.jsonl", "attempts-history.jsonl", "run-signature.json"):
            path = batch_dir / name
            if path.exists():
                archive.write(path, name)
    os.replace(temporary, destination)
    return destination


def publish_batch(batch_id: str, batch_dir: Path, expected_requests: list[dict[str, Any]], token: str, report_sha: str) -> dict[str, Any]:
    from huggingface_hub import HfApi, hf_hub_download

    results_path = batch_dir / "gemini-results.jsonl"
    signature_path = batch_dir / "run-signature.json"
    history_path = batch_dir / "attempts-history.jsonl"
    rows = read_jsonl(results_path, "request_id")
    expected_ids = {row["request_id"] for row in expected_requests}
    if {row["request_id"] for row in rows} != expected_ids or any(row["status"] != "success" for row in rows):
        raise RuntimeError(f"{batch_id} Gemini completion gate failed")
    model_versions = sorted({row["usage"]["model_version"] for row in rows})
    if model_versions != [APPROVED_MODEL_VERSION]:
        raise RuntimeError(f"{batch_id} Gemini model version mismatch: {model_versions}")
    manifest = {
        "schema_version": 1,
        "artifact_kind": "ocr_production_layer_archive",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "batch_id": batch_id,
        "tier": "gemini_final",
        "complete": True,
        "model_id": MODEL_ID,
        "model_version": APPROVED_MODEL_VERSION,
        "media_resolution": MEDIA_RESOLUTION,
        "preflight_report_sha256": report_sha,
        "requests": len(rows),
        "regions": sum(len(row["regions"]) for row in rows),
        "prompt_tokens": sum(row["usage"]["prompt_tokens"] for row in rows),
        "billed_output_tokens": sum(row["usage"]["billed_output_tokens"] for row in rows),
        "actual_cost_vnd": sum(row["actual_cost_vnd"] for row in rows),
        "files": {"gemini-results.jsonl": sha256_file(results_path), "run-signature.json": sha256_file(signature_path)},
    }
    if history_path.exists():
        manifest["files"]["attempts-history.jsonl"] = sha256_file(history_path)
    manifest_path = batch_dir / "batch-manifest.json"
    atomic_json(manifest_path, manifest)
    files = [results_path, signature_path, manifest_path]
    if history_path.exists():
        files.append(history_path)
    sums_path = batch_dir / "SHA256SUMS"
    sums_path.write_text("".join(f"{sha256_file(path)}  {path.name}\n" for path in files), encoding="ascii")
    files.append(sums_path)
    archive_path = Path(f"/kaggle/working/ocr-production-{batch_id}-gemini.zip")
    temporary = Path(f"{archive_path}.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, path.name)
    os.replace(temporary, archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != {path.name for path in files}:
            raise RuntimeError("Gemini archive validation failed")
        if any(Path(name).suffix.casefold() in {".jpg", ".jpeg", ".png", ".webp", ".mp4"} for name in archive.namelist()):
            raise RuntimeError("media leaked into Gemini archive")
    api = HfApi(token=token)
    if not bool(api.repo_info(repo_id=HF_REPO_ID, repo_type="dataset").private):
        raise RuntimeError("HF Dataset must be private")
    remote_path = f"ocr/archives/{batch_id}/gemini/{archive_path.name}"
    if remote_path in set(api.list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset")):
        revision = str(api.repo_info(repo_id=HF_REPO_ID, repo_type="dataset").sha)
        action = "already_present"
    else:
        commit = api.upload_file(path_or_fileobj=archive_path, path_in_repo=remote_path, repo_id=HF_REPO_ID, repo_type="dataset", commit_message=f"OCR Gemini {batch_id}")
        revision = str(commit.oid)
        action = "uploaded"
    downloaded = Path(hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename=remote_path, revision=revision, token=token, force_download=True))
    if sha256_file(downloaded) != sha256_file(archive_path):
        raise RuntimeError("HF round-trip Gemini archive checksum mismatch")
    return {"action": action, "remote_path": remote_path, "revision": revision, "archive_sha256": sha256_file(archive_path), "round_trip_verified": True}


if EXECUTION_MODE not in {"preflight", "canary", "production"}:
    raise ValueError("EXECUTION_MODE must be preflight, canary, or production")
if REQUESTS_PER_MINUTE <= 0 or CANARY_REQUESTS <= 0:
    raise ValueError("request rate and canary size must be positive")

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
report_path, report, residuals, requests = load_preflight()
report_sha = sha256_file(report_path)
print("GEMINI_EXECUTION_MODE", EXECUTION_MODE, flush=True)

if EXECUTION_MODE == "preflight":
    print("GEMINI_API_LOCKED", {"reason": "waiting for user decision", "report_sha256": report_sha, "exact_counts": report["exact_counts"], "cost": report["cost"]}, flush=True)
else:
    if APPROVED_REPORT_SHA256 != report_sha:
        raise RuntimeError("APPROVED_REPORT_SHA256 must equal the verified preflight report SHA-256")
    if APPROVED_MAX_VND <= 0 or APPROVED_MAX_VND > min(MAX_BUDGET_VND, int(report["cost"]["max_budget_vnd"])):
        raise RuntimeError("APPROVED_MAX_VND is missing or exceeds the 400,000 VND cap")
    if EXECUTION_MODE == "canary":
        if not APPROVE_PAID_CANARY or APPROVE_GEMINI_PRODUCTION:
            raise RuntimeError("canary requires only APPROVE_PAID_CANARY=True")
        selected_requests = requests[: min(CANARY_REQUESTS, len(requests))]
    else:
        if not APPROVE_GEMINI_PRODUCTION or APPROVE_PAID_CANARY:
            raise RuntimeError("production requires only APPROVE_GEMINI_PRODUCTION=True")
        if not bool(report["cost"][RUNNER_BILLING_MODE]["within_budget"]):
            raise RuntimeError(
                "full production is blocked: this runner uses Standard synchronous API and "
                "the exact Standard estimate exceeds the approved budget; implement and "
                "validate a real Gemini Batch API runner before using the Batch estimate"
            )
        if not APPROVED_MODEL_VERSION:
            raise RuntimeError("production requires APPROVED_MODEL_VERSION from the paid canary")
        if APPROVED_MAX_REQUESTS <= 0 or APPROVED_MAX_REQUESTS > len(requests):
            raise RuntimeError("APPROVED_MAX_REQUESTS is missing or exceeds exact preflight requests")
        selected_requests = requests[:APPROVED_MAX_REQUESTS]

    selected_region_count = sum(int(row["region_count"]) for row in selected_requests)
    selected_sheet_count = sum(int(row["contact_sheet_count"]) for row in selected_requests)
    planned_input_tokens = selected_sheet_count * 256 + len(selected_requests) * 160 + selected_region_count * 8
    planned_output_tokens = len(selected_requests) * 20 + selected_region_count * 48
    planned_vnd_with_reserve = (
        (
            planned_input_tokens / 1_000_000 * STANDARD_INPUT_USD_PER_MILLION
            + planned_output_tokens / 1_000_000 * STANDARD_OUTPUT_USD_PER_MILLION
        )
        * USD_TO_VND
        * 1.15
    )
    if planned_vnd_with_reserve > APPROVED_MAX_VND:
        raise RuntimeError(
            f"selected requests exceed approved planning cap: "
            f"{planned_vnd_with_reserve:.2f} > {APPROVED_MAX_VND} VND"
        )

    api_key = get_secret("GEMINI_API_KEY")
    hf_token = get_secret("HF_TOKEN") if EXECUTION_MODE == "production" and PUBLISH_TO_HF else ""
    residual_by_id = {row["region_id"]: row for row in residuals}
    limiter = PaceLimiter(REQUESTS_PER_MINUTE)
    selected_by_batch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for request in selected_requests:
        selected_by_batch[request["batch_id"]].append(request)
    total_completed = 0
    cumulative_cost_vnd = 0.0
    canary_records: list[dict[str, Any]] = []
    for batch_id, batch_requests in sorted(selected_by_batch.items()):
        batch_dir = OUTPUT_ROOT / EXECUTION_MODE / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        restored_hf = False
        if EXECUTION_MODE == "production" and PUBLISH_TO_HF:
            restored_hf = restore_completed_from_hf(
                batch_id,
                batch_dir,
                hf_token,
                report_sha=report_sha,
                expected_request_ids={row["request_id"] for row in batch_requests},
            )
        restored = restored_hf or restore_checkpoint(batch_id, batch_dir)
        signature = {"schema_version": 1, "mode": EXECUTION_MODE, "billing_mode": RUNNER_BILLING_MODE, "model_id": MODEL_ID, "media_resolution": MEDIA_RESOLUTION, "preflight_report_sha256": report_sha, "selected_request_ids_sha256": hashlib.sha256("".join(f"{row['request_id']}\n" for row in batch_requests).encode("ascii")).hexdigest(), "approved_model_version": APPROVED_MODEL_VERSION if EXECUTION_MODE == "production" else None}
        signature_path = batch_dir / "run-signature.json"
        if signature_path.exists() and json.loads(signature_path.read_text(encoding="utf-8")) != signature:
            raise RuntimeError(f"{batch_id} stale Gemini checkpoint signature")
        atomic_json(signature_path, signature)
        results_path = batch_dir / "gemini-results.jsonl"
        history_path = batch_dir / "attempts-history.jsonl"
        prior = read_jsonl(results_path, "request_id") if results_path.exists() else []
        prior_errors = [row for row in prior if row["status"] != "success"]
        if prior_errors:
            for row in prior_errors:
                append_jsonl(history_path, {**row, "history_kind": "prior_final_error"})
            prior = [row for row in prior if row["status"] == "success"]
            rewrite_jsonl(results_path, prior)
        expected_ids = {row["request_id"] for row in batch_requests}
        done = {row["request_id"] for row in prior}
        if not done <= expected_ids:
            raise RuntimeError(f"{batch_id} checkpoint contains foreign request_id")
        for row in prior:
            cumulative_cost_vnd += float(row["actual_cost_vnd"])
        print("GEMINI_BATCH_RESUME", batch_id, len(done), "/", len(batch_requests), {"restored": restored}, flush=True)
        batch_root = find_batch_root(batch_id)
        for request in batch_requests:
            if request["request_id"] in done:
                continue
            sheets = build_contact_sheets(request, residual_by_id, batch_root)
            body = request_payload(request, sheets)
            del sheets
            final: dict[str, Any] | None = None
            for attempt in range(1, MAX_ATTEMPTS + 1):
                status, response, latency = call_gemini(body, api_key, limiter)
                attempt_row = {"created_utc": datetime.now(timezone.utc).isoformat(), "request_id": request["request_id"], "batch_id": batch_id, "attempt": attempt, "http_status": status, "latency_seconds": latency}
                if status == 200:
                    try:
                        region_rows, usage = parse_response(response, list(request["region_ids"]))
                        if EXECUTION_MODE == "production" and usage["model_version"] != APPROVED_MODEL_VERSION:
                            raise RuntimeError(f"runtime model version drift: {usage['model_version']}")
                        actual_cost = request_cost_vnd(usage)
                        if cumulative_cost_vnd + actual_cost > APPROVED_MAX_VND:
                            raise RuntimeError("actual Gemini usage would exceed approved VND cap")
                        cumulative_cost_vnd += actual_cost
                        final = {"schema_version": 1, "request_id": request["request_id"], "batch_id": batch_id, "video_id": request["video_id"], "shot_id": request["shot_id"], "status": "success", "regions": region_rows, "usage": usage, "actual_cost_vnd": actual_cost, "attempts": attempt, "error": None}
                        append_jsonl(history_path, {**attempt_row, "outcome": "success", "usage": usage, "error": None})
                        break
                    except Exception as exc:
                        append_jsonl(history_path, {**attempt_row, "outcome": "invalid_response", "error": sanitize(exc, api_key)})
                        if isinstance(exc, RuntimeError) and ("model version drift" in str(exc) or "VND cap" in str(exc)):
                            raise
                else:
                    message = sanitize(response, api_key)
                    retryable = status is None or status == 429 or (status is not None and status >= 500)
                    append_jsonl(history_path, {**attempt_row, "outcome": "retryable_error" if retryable else "terminal_error", "error": message})
                    if status in {401, 403}:
                        raise RuntimeError(f"Gemini auth failure HTTP {status}")
                    if not retryable:
                        break
                if attempt < MAX_ATTEMPTS:
                    time.sleep((2**attempt) + random.random())
            if final is None:
                final = {"schema_version": 1, "request_id": request["request_id"], "batch_id": batch_id, "video_id": request["video_id"], "shot_id": request["shot_id"], "status": "error", "regions": [], "usage": None, "actual_cost_vnd": 0.0, "attempts": MAX_ATTEMPTS, "error": "Gemini request failed or remained schema-invalid"}
            append_jsonl(results_path, final)
            done.add(request["request_id"])
            total_completed += 1
            if final["status"] == "success":
                canary_records.append(final)
            if total_completed % PROGRESS_EVERY == 0 or total_completed == len(selected_requests):
                print("GEMINI_PRODUCTION_PROGRESS", total_completed, "/", len(selected_requests), {"cost_vnd": round(cumulative_cost_vnd, 2)}, flush=True)
            if len(done) % CHECKPOINT_EVERY == 0:
                print("GEMINI_CHECKPOINT_READY", checkpoint_zip(batch_id, batch_dir), flush=True)
        checkpoint_zip(batch_id, batch_dir)
        final_rows = read_jsonl(results_path, "request_id")
        if any(row["status"] != "success" for row in final_rows):
            raise RuntimeError(f"{batch_id} has Gemini errors; rerun to retry from checkpoint")
        if EXECUTION_MODE == "production" and len(batch_requests) == len([row for row in requests if row["batch_id"] == batch_id]):
            publish_report = publish_batch(batch_id, batch_dir, batch_requests, hf_token, report_sha) if PUBLISH_TO_HF else None
            print("GEMINI_BATCH_COMPLETE", batch_id, publish_report, flush=True)
        elif EXECUTION_MODE == "production":
            print("GEMINI_BATCH_PARTIAL_NOT_PUBLISHED", batch_id, len(batch_requests), flush=True)

    if EXECUTION_MODE == "canary":
        model_versions = sorted({row["usage"]["model_version"] for row in canary_records})
        canary_report = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "decision": "PASS_PAID_CANARY" if len(canary_records) == len(selected_requests) and len(model_versions) == 1 else "FAIL_PAID_CANARY", "model_id": MODEL_ID, "model_versions": model_versions, "requests": len(selected_requests), "success": len(canary_records), "schema_valid_rate": len(canary_records) / len(selected_requests) if selected_requests else 0.0, "prompt_tokens": sum(row["usage"]["prompt_tokens"] for row in canary_records), "billed_output_tokens": sum(row["usage"]["billed_output_tokens"] for row in canary_records), "actual_cost_vnd": cumulative_cost_vnd, "preflight_report_sha256": report_sha}
        canary_path = Path("/kaggle/working/ocr-gemini-paid-canary-report.json")
        atomic_json(canary_path, canary_report)
        print("GEMINI_CANARY_COMPLETE", canary_report, canary_path, sha256_file(canary_path), flush=True)
    else:
        print("GEMINI_PRODUCTION_SELECTION_COMPLETE", {"selected_requests": len(selected_requests), "exact_requests": len(requests), "actual_cost_vnd": cumulative_cost_vnd}, flush=True)
