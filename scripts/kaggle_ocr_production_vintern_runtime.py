"""Self-contained Kaggle runtime for OCR production Vintern inference."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
import time
import unicodedata
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image


WORKER_SLOT = int(globals().get("WORKER_SLOT", 1))
PUBLISH_TO_HF = bool(globals().get("PUBLISH_TO_HF", True))
INTERRUPT_AFTER_NEW_CANDIDATES = int(
    globals().get("INTERRUPT_AFTER_NEW_CANDIDATES", 0)
)
CHECKPOINT_EVERY = int(globals().get("CHECKPOINT_EVERY", 10_000))
PROGRESS_EVERY = int(globals().get("PROGRESS_EVERY", 100))
HF_REPO_ID = str(
    globals().get("HF_REPO_ID", "MinhThuw0103/lastdance-visual-embeddings")
)

CATALOG_SHA256 = "ee9693e75580527a0a257e9ba003984e105b059b716922c03c7a0b72b1508a37"
BATCH_MAPPING_SHA256 = "e7e519e5fe3e47c3e487bfe0522c09c3f0bae6c7f67dff2d31168aead0b911d2"
NOTEBOOK_CONTRACT = "ocr-production-vintern-v1"
SOURCE_CONTRACT = "ocr-production-easyocr-v1"
MODEL_ID = "5CD-AI/Vintern-1B-v3_5"
MODEL_REVISION = "b98f263eab246eb5269ade64edbdca8a887dc44d"
MODEL_WEIGHT_BYTES = 3_752_849_256
MODEL_WEIGHT_SHA256 = "296a16a6bf28e6d3f0fb9298deba70b3cfa1d7519f4aa326e2f862bf2e63be05"
MODEL_REQUIRED_GIT_OIDS = {
    "config.json": "2668519f652eddcc2abbb56a52518d38c4f88887",
    "configuration_intern_vit.py": "7e630c456eb9cf350e55bf850c3ff72f445a7e17",
    "configuration_internvl_chat.py": "799209432caf749e77de4c889ec03fe6a32fcdf9",
    "conversation.py": "76fcea7f331de42b6aed7e39fdf80728f9784b7f",
    "modeling_intern_vit.py": "1c5c043a4b860720b3b6e55107e8e6ecf0c573de",
    "modeling_internvl_chat.py": "41f48cd5cb907e025725fc42ac819cf3f03c01b5",
}
BATCHES = {
    "batch-01": {"keyframes_batch": "keyframes-batch-01", "keyframe_count": 59836, "uid_set_sha256": "ea3c0e39b65298e496472fa971b6ea421d93ab78df057980677a0a7fbc213049"},
    "batch-02": {"keyframes_batch": "keyframes-batch-02", "keyframe_count": 29305, "uid_set_sha256": "1c8a0097fcc28f77bfb9278f89930fb181d80072b0a063f32c414ff8e5246fab"},
    "batch-03": {"keyframes_batch": "keyframes-batch-03", "keyframe_count": 26972, "uid_set_sha256": "270480578d4db21465bb73bf505655b963ac1d96c4df553724920527bc559900"},
    "batch-04": {"keyframes_batch": "keyframes-batch-04", "keyframe_count": 27521, "uid_set_sha256": "b1b47804e57983e4714be6f7c1b9a760ea560944ba73a7f5dab4040cc14171f6"},
    "batch-05": {"keyframes_batch": "keyframes-batch-05", "keyframe_count": 31245, "uid_set_sha256": "673a5bbc2d76f4a3ebd876f2356d9a5c28dfef1151c09ebc6af4fb8ed1f7a358"},
    "batch-06": {"keyframes_batch": "keyframes-batch-06", "keyframe_count": 31407, "uid_set_sha256": "d1082481a0cd031e5a1139ac4a6cd250e146237f692d46eed71fce2084a9235b"},
    "batch-07": {"keyframes_batch": "keyframes-batch-07", "keyframe_count": 31642, "uid_set_sha256": "f3b26c7bd90f2396cf4d944fb403f4c201e6acb3bc6e406cfc97da23ce74f230"},
    "batch-08": {"keyframes_batch": "keyframes-batch-08", "keyframe_count": 45460, "uid_set_sha256": "0e72e26396dd2776e88d32f2a36a449243849e11a16270ac3a3756076c61d62a"},
    "batch-09": {"keyframes_batch": "keyframes-batch-09", "keyframe_count": 9948, "uid_set_sha256": "31bba461f43fdc384b32012d877f3792a94fa51e65d66f3d68d6cca68656b06c"},
}
WORKER_BATCHES = {
    1: ("batch-01", "batch-09"),
    2: ("batch-02", "batch-03", "batch-04"),
    3: ("batch-05", "batch-08"),
    4: ("batch-06", "batch-07"),
}
INPUT_ROOT = Path("/kaggle/input")
OUTPUT_ROOT = Path("/kaggle/working/ocr-production-vintern-v1")
SOURCE_ROOT = OUTPUT_ROOT / "easyocr-source"
MODEL_RUNTIME = OUTPUT_ROOT / "vintern-local-runtime"
ALLOWED_SOURCE_MEMBERS = {
    "easyocr-frames.jsonl",
    "vintern-candidates.jsonl",
    "run-signature.json",
    "batch-manifest.json",
    "SHA256SUMS",
    "errors-history.jsonl",
}
ALLOWED_OUTPUT_MEMBERS = {
    "vintern-results.jsonl",
    "run-signature.json",
    "batch-manifest.json",
    "SHA256SUMS",
    "errors-history.jsonl",
}
QUESTION = (
    "<image>\nChép lại nguyên văn toàn bộ chữ nhìn thấy. Không sửa chính tả, "
    "không suy đoán phần bị che. Chỉ trả về văn bản."
)
PROMPT_MARKERS = (
    "i don't understand",
    "i do not understand",
    "this is a blurry image",
    "the image is blurry",
    "the text is blurry",
    "chữ bị che khuất",
    "chữ trong ảnh",
    "nội dung trong ảnh",
    "không thể đọc",
    "không đọc được",
    "từ đó tôi sẽ",
    "hãy chép lại",
    "chép lại nguyên văn",
)
ALLOWED_PUNCTUATION = frozenset(".,:;!?%+-/()[]{}'\"&@#_\\|~`=<>₫$€£¥…–—")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_oid(path: Path) -> str:
    payload = path.read_bytes()
    return hashlib.sha1(f"blob {len(payload)}\0".encode("ascii") + payload).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def rewrite_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = Path(f"{path}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_jsonl(path: Path, identity: str) -> list[dict]:
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
            value = row[identity]
            if value in seen:
                raise ValueError(f"duplicate {identity}: {value}")
            seen.add(value)
            rows.append(row)
    return rows


def parse_checksums(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        if len(digest) != 64 or Path(name).name != name or name in values:
            raise RuntimeError("invalid SHA256SUMS")
        values[name] = digest
    return values


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
    raise RuntimeError("Kaggle secret HF_TOKEN/HK_TOKEN is missing")


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
        values = sorted(path.resolve() for path in INPUT_ROOT.rglob(directory_name) if path.is_dir())
    if len(values) != 1:
        raise RuntimeError(f"expected exactly one {directory_name}: {values}")
    return values[0]


def download_easyocr_source(batch_id: str, token: str) -> tuple[Path, dict]:
    from huggingface_hub import HfApi, hf_hub_download

    remote_path = (
        f"ocr/archives/{batch_id}/easyocr/"
        f"ocr-production-{batch_id}-easyocr.zip"
    )
    api = HfApi(token=token)
    info = api.repo_info(repo_id=HF_REPO_ID, repo_type="dataset")
    if not bool(info.private):
        raise RuntimeError("HF Dataset must be private")
    revision = str(info.sha)
    if remote_path not in set(api.list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset", revision=revision)):
        raise RuntimeError(f"WAIT_EASYOCR_ARCHIVE: {remote_path} is not published yet")
    print("EASYOCR_ARCHIVE_DOWNLOAD_START", batch_id, revision, flush=True)
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
    target = SOURCE_ROOT / batch_id
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("EasyOCR archive CRC failure")
        names = archive.namelist()
        if len(names) != len(set(names)) or not set(names) <= ALLOWED_SOURCE_MEMBERS:
            raise RuntimeError("EasyOCR archive member mismatch")
        if any(Path(name).name != name for name in names):
            raise RuntimeError("EasyOCR archive contains unsafe paths")
        archive.extractall(target)
    checksums = parse_checksums(target / "SHA256SUMS")
    for name, expected in checksums.items():
        if name == "SHA256SUMS" or not (target / name).is_file():
            raise RuntimeError(f"EasyOCR checksum member missing: {name}")
        if sha256_file(target / name) != expected:
            raise RuntimeError(f"EasyOCR checksum mismatch: {name}")
    manifest = json.loads((target / "batch-manifest.json").read_text(encoding="utf-8"))
    signature = json.loads((target / "run-signature.json").read_text(encoding="utf-8"))
    expected = BATCHES[batch_id]
    if not (
        manifest.get("batch_id") == batch_id
        and manifest.get("tier") == "easyocr"
        and manifest.get("complete") is True
        and manifest.get("catalog_sha256") == CATALOG_SHA256
        and manifest.get("batch_mapping_sha256") == BATCH_MAPPING_SHA256
        and manifest.get("assigned_uid_sha256") == expected["uid_set_sha256"]
        and manifest.get("frames") == expected["keyframe_count"]
        and signature.get("contract") == SOURCE_CONTRACT
    ):
        raise RuntimeError(f"{batch_id} EasyOCR source provenance mismatch")
    evidence = {
        "hf_revision": revision,
        "remote_path": remote_path,
        "archive_sha256": sha256_file(archive_path),
        "candidate_jsonl_sha256": sha256_file(target / "vintern-candidates.jsonl"),
        "candidate_count": int(manifest["vintern_candidates"]),
    }
    print("EASYOCR_ARCHIVE_VERIFIED", batch_id, evidence, flush=True)
    return target, evidence


def find_checkpoint(batch_id: str) -> Path | None:
    name = f"ocr-production-{batch_id}-vintern-checkpoint.zip"
    matches: list[Path] = []
    for pattern in (f"*/{name}", f"*/*/{name}", f"*/*/*/{name}"):
        matches.extend(path.resolve() for path in INPUT_ROOT.glob(pattern) if path.is_file())
    values = sorted(set(matches))
    if len(values) > 1:
        raise RuntimeError(f"attach at most one checkpoint for {batch_id}: {values}")
    return values[0] if values else None


def restore_checkpoint(batch_id: str, batch_dir: Path) -> bool:
    source = find_checkpoint(batch_id)
    if source is None:
        return False
    with zipfile.ZipFile(source) as archive:
        if archive.testzip() is not None:
            raise RuntimeError(f"{batch_id} checkpoint CRC failure")
        names = archive.namelist()
        allowed = {"vintern-results.jsonl", "run-signature.json", "errors-history.jsonl"}
        if len(names) != len(set(names)) or not set(names) <= allowed:
            raise RuntimeError(f"{batch_id} checkpoint member mismatch")
        if any(Path(name).name != name for name in names):
            raise RuntimeError(f"{batch_id} checkpoint contains unsafe paths")
        for name in names:
            destination = batch_dir / name
            if destination.exists() and destination.read_bytes() != archive.read(name):
                raise RuntimeError(f"{batch_id} checkpoint conflicts with working data")
            destination.write_bytes(archive.read(name))
    print("CHECKPOINT_RESTORED", batch_id, source, flush=True)
    return True


def checkpoint_zip(batch_id: str, batch_dir: Path) -> Path:
    destination = Path(f"/kaggle/working/ocr-production-{batch_id}-vintern-checkpoint.zip")
    temporary = Path(f"{destination}.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
        for name in ("vintern-results.jsonl", "run-signature.json", "errors-history.jsonl"):
            path = batch_dir / name
            if path.exists():
                archive.write(path, name)
    os.replace(temporary, destination)
    return destination


def load_model():
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    print("VINTERN_MODEL_DOWNLOAD_START", MODEL_ID, MODEL_REVISION, flush=True)
    snapshot_dir = Path(snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION))
    weight_path = snapshot_dir / "model.safetensors"
    if weight_path.stat().st_size != MODEL_WEIGHT_BYTES or sha256_file(weight_path) != MODEL_WEIGHT_SHA256:
        raise RuntimeError("Vintern model weight checksum mismatch")
    for filename, expected in MODEL_REQUIRED_GIT_OIDS.items():
        if git_blob_oid(snapshot_dir / filename) != expected:
            raise RuntimeError(f"Vintern runtime provenance mismatch: {filename}")
    if MODEL_RUNTIME.exists():
        shutil.rmtree(MODEL_RUNTIME)
    MODEL_RUNTIME.mkdir(parents=True)
    for source in snapshot_dir.iterdir():
        target = MODEL_RUNTIME / source.name
        if source.name == "config.json":
            config = json.loads(source.read_text(encoding="utf-8"))
            config["auto_map"] = {key: value.split("--", 1)[-1] for key, value in config["auto_map"].items()}
            target.write_text(json.dumps(config, indent=2), encoding="utf-8")
        elif source.is_file():
            target.symlink_to(source.resolve())
    print("VINTERN_MODEL_LOAD_START", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_RUNTIME), trust_remote_code=True, use_fast=False, local_files_only=True
    )
    model = AutoModel.from_pretrained(
        str(MODEL_RUNTIME),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
        use_flash_attn=False,
    ).eval().to("cuda:0")
    if not all(parameter.dtype == torch.float16 for parameter in model.parameters() if parameter.is_floating_point()):
        raise RuntimeError("Vintern model is not fully FP16")
    print("VINTERN_MODEL_READY", {"allocated_mib": torch.cuda.memory_allocated(0) / 2**20}, flush=True)
    return model, tokenizer


def visible_length(text: str) -> int:
    return len("".join(text.split()))


def noise_ratio(text: str) -> float:
    visible = [character for character in text if not character.isspace()]
    if not visible:
        return 0.0
    def is_noisy(character: str) -> bool:
        if character.isalnum() or character in ALLOWED_PUNCTUATION:
            return False
        return not unicodedata.category(character).startswith(("L", "N", "P", "S", "M"))

    return sum(is_noisy(character) for character in visible) / len(visible)


def guard_reasons(easyocr_text: str, vintern_text: str) -> list[str]:
    candidate = vintern_text.strip()
    if not candidate:
        return ["empty_output"]
    reasons: list[str] = []
    normalized = " ".join(candidate.casefold().split())
    if any(marker in normalized for marker in PROMPT_MARKERS):
        reasons.append("prompt_or_explanation_leak")
    easy_length = visible_length(easyocr_text)
    candidate_length = visible_length(candidate)
    if candidate_length > max(96, easy_length * 8 + 48):
        reasons.append("gross_length_expansion")
    if noise_ratio(candidate) >= 0.5 and candidate_length >= 4:
        reasons.append("noisy_output")
    return reasons


def crop_candidate(candidate: dict, batch_root: Path) -> Image.Image:
    filename = Path(str(candidate["source_image"])).name
    source = batch_root / str(candidate["video_id"]) / filename
    if not source.is_file():
        direct = INPUT_ROOT / str(candidate["source_image"])
        source = direct if direct.is_file() else source
    if not source.is_file():
        raise FileNotFoundError(f"source keyframe missing: {source}")
    image = Image.open(source).convert("RGB")
    flat = candidate["bbox_px"]
    if len(flat) != 8:
        raise ValueError("bbox_px must contain exactly 8 values")
    x_values = [float(value) for value in flat[0::2]]
    y_values = [float(value) for value in flat[1::2]]
    x1 = max(0, int(math.floor(min(x_values))) - 4)
    y1 = max(0, int(math.floor(min(y_values))) - 4)
    x2 = min(image.width, int(math.ceil(max(x_values))) + 4)
    y2 = min(image.height, int(math.ceil(max(y_values))) + 4)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("invalid candidate crop bounds")
    return image.crop((x1, y1, x2, y2))


def build_archive(batch_id: str, batch_dir: Path, source: dict, rows: list[dict], elapsed: float) -> tuple[Path, dict]:
    results_path = batch_dir / "vintern-results.jsonl"
    signature_path = batch_dir / "run-signature.json"
    errors_path = batch_dir / "errors-history.jsonl"
    manifest_path = batch_dir / "batch-manifest.json"
    sums_path = batch_dir / "SHA256SUMS"
    statuses = Counter(row["status"] for row in rows)
    errors = statuses["error"]
    if len(rows) != source["candidate_count"] or errors:
        raise RuntimeError(f"{batch_id} Vintern completion gate failed")
    manifest = {
        "schema_version": 1,
        "artifact_kind": "ocr_production_layer_archive",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "batch_id": batch_id,
        "layer": "vintern",
        "complete": True,
        "calibrated": False,
        "searchable": False,
        "catalog_sha256": CATALOG_SHA256,
        "batch_mapping_sha256": BATCH_MAPPING_SHA256,
        "assigned_uid_sha256": BATCHES[batch_id]["uid_set_sha256"],
        "expected_candidates": source["candidate_count"],
        "processed_candidates": len(rows),
        "status": dict(sorted(statuses.items())),
        "guard_rejections": sum(bool(row["guard_rejection_reasons"]) for row in rows),
        "seconds": elapsed,
        "candidates_per_second": len(rows) / elapsed if elapsed else None,
        "source_easyocr": source,
        "model": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "weight_sha256": MODEL_WEIGHT_SHA256,
            "dtype": "float16",
        },
        "model_calls": {"vintern": len(rows), "gemini": 0},
        "files": {
            "vintern-results.jsonl": sha256_file(results_path),
            "run-signature.json": sha256_file(signature_path),
        },
        "next_step": "local calibrated materialization; this raw layer is not ocr.sqlite",
    }
    if errors_path.exists():
        manifest["files"]["errors-history.jsonl"] = sha256_file(errors_path)
    atomic_json(manifest_path, manifest)
    files = [results_path, signature_path, manifest_path]
    if errors_path.exists():
        files.append(errors_path)
    sums_path.write_text("".join(f"{sha256_file(path)}  {path.name}\n" for path in files), encoding="ascii")
    files.append(sums_path)
    destination = Path(f"/kaggle/working/ocr-production-{batch_id}-vintern.zip")
    temporary = Path(f"{destination}.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, path.name)
    os.replace(temporary, destination)
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != {path.name for path in files}:
            raise RuntimeError("Vintern archive validation failed")
        if any(Path(name).suffix.casefold() in {".jpg", ".jpeg", ".png", ".webp", ".mp4"} for name in archive.namelist()):
            raise RuntimeError("media file leaked into Vintern archive")
    return destination, manifest


def publish_archive(archive_path: Path, batch_id: str, token: str) -> dict:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)
    remote_path = f"ocr/archives/{batch_id}/vintern/{archive_path.name}"
    local_sha = sha256_file(archive_path)
    files = set(api.list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset"))
    if remote_path in files:
        revision = str(api.repo_info(repo_id=HF_REPO_ID, repo_type="dataset").sha)
        action = "already_present"
    else:
        commit = api.upload_file(
            path_or_fileobj=archive_path,
            path_in_repo=remote_path,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"OCR Vintern {batch_id}",
        )
        revision = str(commit.oid)
        action = "uploaded"
    downloaded = Path(hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename=remote_path, revision=revision, token=token, force_download=True))
    if sha256_file(downloaded) != local_sha:
        raise RuntimeError("HF round-trip Vintern archive checksum mismatch")
    return {"action": action, "remote_path": remote_path, "revision": revision, "archive_sha256": local_sha, "round_trip_verified": True}


if WORKER_SLOT not in WORKER_BATCHES:
    raise ValueError("WORKER_SLOT must be 1, 2, 3, or 4")
if CHECKPOINT_EVERY <= 0 or PROGRESS_EVERY <= 0 or INTERRUPT_AFTER_NEW_CANDIDATES < 0:
    raise ValueError("checkpoint/progress/interrupt settings are invalid")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("select Kaggle GPU and expose exactly one CUDA device")
if "T4" not in torch.cuda.get_device_name(0).upper():
    raise RuntimeError(f"expected T4 GPU, got {torch.cuda.get_device_name(0)}")

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
hf_token = get_hf_token()
print("VINTERN_PRODUCTION_START", {"worker_slot": WORKER_SLOT, "batches": WORKER_BATCHES[WORKER_SLOT], "gpu": torch.cuda.get_device_name(0)}, flush=True)

model = tokenizer = transform = generation = None
for batch_position, batch_id in enumerate(WORKER_BATCHES[WORKER_SLOT], start=1):
    started = time.perf_counter()
    batch_dir = OUTPUT_ROOT / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    source_dir, source_evidence = download_easyocr_source(batch_id, hf_token)
    candidates = load_jsonl(source_dir / "vintern-candidates.jsonl", "candidate_id")
    if len(candidates) != source_evidence["candidate_count"]:
        raise RuntimeError(f"{batch_id} candidate count mismatch")
    candidate_ids = {row["candidate_id"] for row in candidates}
    batch_root = find_batch_root(BATCHES[batch_id]["keyframes_batch"])
    stable_source_evidence = {
        key: value for key, value in source_evidence.items() if key != "hf_revision"
    }
    signature = {
        "schema_version": 1,
        "contract": NOTEBOOK_CONTRACT,
        "batch_id": batch_id,
        "worker_slot": WORKER_SLOT,
        "catalog_sha256": CATALOG_SHA256,
        "batch_mapping_sha256": BATCH_MAPPING_SHA256,
        "assigned_uid_sha256": BATCHES[batch_id]["uid_set_sha256"],
        "source_easyocr": stable_source_evidence,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_weight_sha256": MODEL_WEIGHT_SHA256,
    }
    restored = restore_checkpoint(batch_id, batch_dir)
    signature_path = batch_dir / "run-signature.json"
    if signature_path.exists() and json.loads(signature_path.read_text(encoding="utf-8")) != signature:
        raise RuntimeError(f"{batch_id} stale Vintern checkpoint signature")
    atomic_json(signature_path, signature)
    results_path = batch_dir / "vintern-results.jsonl"
    prior = load_jsonl(results_path, "candidate_id")
    prior_errors = [row for row in prior if row.get("status") == "error"]
    if prior_errors:
        errors_path = batch_dir / "errors-history.jsonl"
        for row in prior_errors:
            append_jsonl(errors_path, row)
        prior = [row for row in prior if row.get("status") != "error"]
        rewrite_jsonl(results_path, prior)
        print("RETRYING_PRIOR_ERRORS", batch_id, len(prior_errors), flush=True)
    done = {row["candidate_id"] for row in prior}
    if not done <= candidate_ids:
        raise RuntimeError(f"{batch_id} checkpoint contains foreign candidate_id")
    print("VINTERN_BATCH_RESUME", batch_id, len(done), "/", len(candidates), {"restored": restored}, flush=True)
    if len(done) < len(candidates) and model is None:
        model, tokenizer = load_model()
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        transform = T.Compose([
            T.Lambda(lambda image: image.convert("RGB")),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        generation = {"max_new_tokens": 96, "do_sample": False, "num_beams": 1, "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.eos_token_id}
    new_candidates = 0
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        if candidate_id in done:
            continue
        item_started = time.perf_counter()
        try:
            crop = crop_candidate(candidate, batch_root)
            pixels = transform(crop).unsqueeze(0).to("cuda:0", dtype=torch.float16)
            torch.cuda.synchronize()
            inference_started = time.perf_counter()
            with torch.inference_mode():
                text = str(model.chat(tokenizer, pixels, QUESTION, generation)).strip()
            torch.cuda.synchronize()
            inference_seconds = time.perf_counter() - inference_started
            del pixels, crop
            output_length = visible_length(text)
            limit = max(96, visible_length(str(candidate.get("easyocr_text") or "")) * 8 + 48)
            guards = guard_reasons(str(candidate.get("easyocr_text") or ""), text)
            result = {
                "schema_version": 1,
                "candidate_id": candidate_id,
                "video_id": candidate["video_id"],
                "keyframe_uid": candidate["keyframe_uid"],
                "status": "success" if text else "empty",
                "vintern_text": text,
                "inference_seconds": inference_seconds,
                "total_seconds": time.perf_counter() - item_started,
                "output_length": output_length,
                "guard_length_limit": limit,
                "guard_margin_ratio": (limit - output_length) / limit,
                "mean_token_logprob": None,
                "logprob_available": False,
                "guard_rejection_reasons": guards,
                "error": None,
            }
        except Exception as exc:
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
            result = {
                "schema_version": 1,
                "candidate_id": candidate_id,
                "video_id": candidate.get("video_id"),
                "keyframe_uid": candidate.get("keyframe_uid"),
                "status": "error",
                "vintern_text": "",
                "inference_seconds": None,
                "total_seconds": time.perf_counter() - item_started,
                "output_length": 0,
                "guard_length_limit": max(96, visible_length(str(candidate.get("easyocr_text") or "")) * 8 + 48),
                "guard_margin_ratio": 1.0,
                "mean_token_logprob": None,
                "logprob_available": False,
                "guard_rejection_reasons": ["runtime_error"],
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            }
        append_jsonl(results_path, result)
        done.add(candidate_id)
        new_candidates += 1
        completed = len(done)
        if completed % PROGRESS_EVERY == 0 or completed == len(candidates):
            print("VINTERN_PRODUCTION_PROGRESS", batch_id, completed, "/", len(candidates), flush=True)
        if completed % CHECKPOINT_EVERY == 0:
            checkpoint = checkpoint_zip(batch_id, batch_dir)
            print("VINTERN_CHECKPOINT_READY", batch_id, completed, checkpoint, flush=True)
        if INTERRUPT_AFTER_NEW_CANDIDATES and batch_position == 1 and new_candidates >= INTERRUPT_AFTER_NEW_CANDIDATES:
            checkpoint = checkpoint_zip(batch_id, batch_dir)
            print("INTENTIONAL_INTERRUPT_READY", checkpoint, sha256_file(checkpoint), flush=True)
            raise RuntimeError("Intentional Vintern checkpoint test completed; download and attach the ZIP, set INTERRUPT_AFTER_NEW_CANDIDATES=0, then rerun.")
    rows = load_jsonl(results_path, "candidate_id")
    if {row["candidate_id"] for row in rows} != candidate_ids:
        raise RuntimeError(f"{batch_id} final candidate set mismatch")
    errors = [row for row in rows if row["status"] == "error"]
    if errors:
        checkpoint = checkpoint_zip(batch_id, batch_dir)
        print("VINTERN_BATCH_HAS_ERRORS_RERUN", batch_id, len(errors), checkpoint, flush=True)
        raise RuntimeError(f"{batch_id} contains {len(errors)} Vintern errors")
    archive_path, manifest = build_archive(batch_id, batch_dir, source_evidence, rows, time.perf_counter() - started)
    checkpoint = checkpoint_zip(batch_id, batch_dir)
    publish_report = publish_archive(archive_path, batch_id, hf_token) if PUBLISH_TO_HF else None
    if publish_report:
        atomic_json(batch_dir / "hf-publish-report.json", publish_report)
    print("VINTERN_BATCH_COMPLETE", json.dumps({"batch_id": batch_id, "candidates": len(rows), "guard_rejections": manifest["guard_rejections"], "archive": str(archive_path), "checkpoint": str(checkpoint), "hf": publish_report}, ensure_ascii=False), flush=True)

if model is not None:
    del model, tokenizer
gc.collect()
torch.cuda.empty_cache()
print("VINTERN_WORKER_SLOT_COMPLETE", WORKER_SLOT, WORKER_BATCHES[WORKER_SLOT], flush=True)
