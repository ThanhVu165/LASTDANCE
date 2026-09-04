"""Small, resumable VietOCR preprocessing trial; no production writes or paid calls.

The generated notebook embeds only the pinned pure helpers from Gate B. Importing
this module does not import Torch, download models, or start inference.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import importlib.metadata as md
import io
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
import types
import zipfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps


CONTRACT = "ocr-v2-sharpen-trial-v1"
VIDEOS = ("L21_V001", "L21_V002", "L21_V003", "L21_V005", "L21_V006")
VARIANTS = ("original", "bicubic_2x", "bicubic_2x_unsharp")
REFERENCE_NAMES = {
    "VIETOCR_WEIGHT", "VIETOCR_CONFIGS", "VIETOCR_MODEL_ID", "CROP_SPEC_ID",
    "sha256_bytes", "sha256_file", "source_names", "source_read", "parse_checksums",
    "_float_list", "_edge_pad", "rectify_region_crop", "download_verified",
    "prepare_vietocr_model",
}
VIETOCR_WHEEL_SHA256 = "07b3777e5176b0d733cb056b68bd817371605f4b3514795fbf91ad4e181b8ccf"


def version_or_missing(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def ensure_vietocr_packages(cache):
    """Repair a fresh Kaggle session without resolving/replacing CUDA dependencies."""
    def protected_versions():
        result = {name: version_or_missing(name) for name in ("torch", "torchvision")}
        for dist in md.distributions():
            name = (dist.metadata.get("Name") or "").lower().replace("_", "-")
            if name.startswith("nvidia-"):
                result[name] = dist.version
        return result
    before = protected_versions()
    if not before["torch"] or not before["torchvision"]:
        raise RuntimeError("Select a Kaggle GPU image with Torch/torchvision installed")
    installed = version_or_missing("vietocr")
    if installed not in (None, "0.3.13"):
        raise RuntimeError(f"Found VietOCR {installed}; use a fresh T4 session for the pinned 0.3.13 trial")
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    pip = [sys.executable, "-m", "pip", "--disable-pip-version-check"]
    if installed is None:
        log("ENV_INSTALL", package="vietocr==0.3.13", dependency_resolution=False)
        with heartbeat("download pinned VietOCR wheel"):
            subprocess.run(pip + ["download", "--no-deps", "--only-binary=:all:", "--timeout", "30",
                                   "--retries", "2", "--dest", str(cache), "vietocr==0.3.13"],
                           check=True, timeout=180)
        wheels = sorted(cache.glob("vietocr-0.3.13-*.whl"))
        if len(wheels) != 1 or digest(wheels[0].read_bytes()) != VIETOCR_WHEEL_SHA256:
            raise RuntimeError("VietOCR wheel checksum mismatch; not installing")
        subprocess.run(pip + ["install", "--no-deps", str(wheels[0])], check=True, timeout=90)
    # Dependencies imported by VietOCR; preserve existing compatible packages.
    for name, requirement in (("einops", "einops==0.8.1"), ("gdown", "gdown==5.2.0"),
                              ("PyYAML", "PyYAML==6.0.2")):
        if version_or_missing(name) is None:
            log("ENV_INSTALL", package=requirement, dependency_resolution=False)
            with heartbeat(f"install {name}"):
                subprocess.run(pip + ["install", "--no-deps", "--timeout", "30", "--retries", "2", requirement],
                               check=True, timeout=180)
    if protected_versions() != before:
        raise RuntimeError("Torch/NVIDIA package versions changed during setup; start a fresh GPU session")
    probe = subprocess.run([sys.executable, "-c", "from vietocr.tool.predictor import Predictor; print('VIETOCR_IMPORT_OK')"],
                           text=True, capture_output=True, timeout=90)
    if probe.returncode:
        raise RuntimeError("VietOCR dependency probe failed before crop creation:\n" + probe.stderr)
    packages = {name: version_or_missing(name) for name in
                ("torch", "torchvision", "vietocr", "einops", "gdown", "PyYAML", "Pillow", "numpy")}
    if any(value is None for value in packages.values()):
        raise RuntimeError(f"Missing environment metadata: {packages}")
    log("ENV_READY", packages=packages)
    return packages


def reference_source(source):
    """Copy approved definitions, never Gate B's module-level execution loop."""
    nodes = []
    for node in ast.parse(source).body:
        name = node.name if isinstance(node, ast.FunctionDef) else None
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if name in REFERENCE_NAMES:
            nodes.append(node)
    if len(nodes) != len(REFERENCE_NAMES):
        raise ValueError("Gate B helper contract changed")
    return ast.unparse(ast.Module(body=nodes, type_ignores=[])) + "\n"


def helpers(model_dir):
    source = globals().get("GATE_B_REFERENCE_SOURCE")
    if source is None:
        source = reference_source(Path(__file__).with_name(
            "kaggle_ocr_v2_gate_b_runtime.py").read_text(encoding="utf-8"))
    namespace = {}
    exec("from __future__ import annotations\nimport hashlib, json, math, re, urllib.request, zipfile\n"
         "from pathlib import Path\nfrom PIL import Image\nfrom typing import Any\n" + source,
         namespace)
    namespace["OUTPUT_ROOT"] = Path(model_dir)
    return types.SimpleNamespace(**namespace), source


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def atomic_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def log(stage, **values):
    print(stage, json.dumps(values, ensure_ascii=False, allow_nan=False), flush=True)


@contextmanager
def heartbeat(stage):
    stop = threading.Event()
    started = time.monotonic()
    def tick():
        while not stop.wait(30):
            log("HEARTBEAT", stage=stage, elapsed_seconds=round(time.monotonic() - started),
                note="stage still active; not an additional completion")
    thread = threading.Thread(target=tick, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1)


def discover(root, explicit, zip_name, required):
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    archives, directories = [], []
    root = Path(root)
    for current, subdirs, filenames in os.walk(root):
        # Metadata can be nested by Kaggle; never walk thousands of frame folders.
        subdirs[:] = [name for name in subdirs if not re.fullmatch(r"L\d+_V\d+", name)]
        if len(Path(current).relative_to(root).parts) >= 8:
            subdirs[:] = []
        if zip_name in filenames:
            archives.append(Path(current) / zip_name)
        if set(required) <= set(filenames):
            directories.append(Path(current))
    candidates = archives or directories
    if len(candidates) != 1:
        raise ValueError(f"Set explicit path for {zip_name}; found {candidates}")
    return candidates[0]


def load_inputs(review, gate_b, ref):
    for source, required in (
        (review, {"recognition-sample.jsonl", "review-report.json"}),
        (gate_b, {"recognizer-results.jsonl", "runtime-report.json", "SHA256SUMS"}),
    ):
        if not required <= ref.source_names(source):
            raise ValueError(f"Incomplete input: {source}")
    sample_bytes = ref.source_read(review, "recognition-sample.jsonl")
    review_report = json.loads(ref.source_read(review, "review-report.json"))
    gate_bytes = ref.source_read(gate_b, "recognizer-results.jsonl")
    report_bytes = ref.source_read(gate_b, "runtime-report.json")
    sums = ref.parse_checksums(ref.source_read(gate_b, "SHA256SUMS"))
    for name, payload in (("recognizer-results.jsonl", gate_bytes), ("runtime-report.json", report_bytes)):
        if sums.get(name) != digest(payload):
            raise ValueError(f"Gate B checksum mismatch: {name}")
    report = json.loads(report_bytes)
    if (review_report.get("artifacts", {}).get("recognition_sample_jsonl_sha256") != digest(sample_bytes)
            or report.get("sample_jsonl_sha256") != digest(sample_bytes)):
        raise ValueError("Gate A/B sample hashes differ")
    samples = [json.loads(line) for line in sample_bytes.splitlines() if line.strip()]
    by_id = {row["region_id"]: row for row in samples}
    if len(samples) != 120 or len(by_id) != 120 or {r["video_id"] for r in samples} != set(VIDEOS):
        raise ValueError("Expected 120 unique Gate B regions across the five approved videos")
    models = {"easyocr_latin_g2_cached", "latin_PP-OCRv5_mobile_rec", ref.VIETOCR_MODEL_ID}
    seen, viet = set(), {}
    for line in gate_bytes.splitlines():
        row = json.loads(line)
        key = (row["region_id"], row["model_id"])
        if key in seen or key[0] not in by_id or key[1] not in models:
            raise ValueError("Duplicate/foreign Gate B result")
        seen.add(key)
        sample = by_id[key[0]]
        if any(row[field] != sample[field] for field in ("keyframe_uid", "video_id", "sample_row_sha256")):
            raise ValueError("Gate B immutable identity mismatch")
        if row.get("status") != "success" or row.get("error") is not None:
            raise ValueError("Gate B contains a failed result")
        if key[1] == ref.VIETOCR_MODEL_ID:
            viet[key[0]] = row
    if seen != {(rid, model) for rid in by_id for model in models}:
        raise ValueError("Incomplete Gate B coverage")
    for sample in samples:
        if sample.get("crop_spec_id") != ref.CROP_SPEC_ID:
            raise ValueError("Unknown crop contract")
        ref._float_list(sample["bbox_px"])
    return samples, viet, {"sample_sha256": digest(sample_bytes), "gate_b_results_sha256": digest(gate_bytes),
                           "gate_b_report_sha256": digest(report_bytes)}


def confidence(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) and 0 <= value <= 1 else None


def select_samples(samples, viet):
    chosen = []
    for video in VIDEOS:
        rows = [r for r in samples if r["video_id"] == video]
        if len(rows) < 6:
            raise ValueError(f"Not enough sample regions for {video}")
        def score(row):
            value = confidence(viet[row["region_id"]].get("confidence"))
            return -1 if value is None else value
        low = sorted(rows, key=lambda r: (score(r), r["region_id"]))[:4]
        low_ids = {r["region_id"] for r in low}
        high = sorted((r for r in rows if r["region_id"] not in low_ids),
                      key=lambda r: (-score(r), r["region_id"]))[:2]
        for row in low + high:
            chosen.append({**row, "selection_group": "low" if row["region_id"] in low_ids else "control",
                           "gate_b_vietocr_text": viet[row["region_id"]]["text"],
                           "gate_b_vietocr_confidence": confidence(viet[row["region_id"]].get("confidence"))})
    return chosen


def make_variant(crop, variant):
    if variant == "original":
        return crop.copy()
    enlarged = crop.resize((crop.width * 2, crop.height * 2), Image.Resampling.BICUBIC)
    if variant == "bicubic_2x":
        return enlarged
    if variant == "bicubic_2x_unsharp":
        return enlarged.filter(ImageFilter.UnsharpMask(radius=1, percent=100, threshold=3))
    raise ValueError(variant)


def build_tasks(selected, root, ref, output):
    video_dirs = {}
    for current, subdirs, _ in os.walk(root):
        path = Path(current)
        if path.name in VIDEOS:
            if path.name in video_dirs:
                raise ValueError(f"Ambiguous keyframe directory: {path.name}")
            video_dirs[path.name] = path
            subdirs[:] = []
        else:
            subdirs[:] = [d for d in subdirs if not re.fullmatch(r"L\d+_V\d+", d) or d in VIDEOS]
    if set(video_dirs) != set(VIDEOS):
        raise ValueError("Missing JPEG directories; set KEYFRAMES_ROOT to the Batch 01 dataset")
    tasks = []
    for index, sample in enumerate(selected, 1):
        path = video_dirs[sample["video_id"]] / Path(sample["source_image"]).name
        payload = path.read_bytes()
        with Image.open(io.BytesIO(payload)) as opened:
            crop = ref.rectify_region_crop(ImageOps.exif_transpose(opened).convert("RGB"), sample["bbox_px"])
        # Gate A copied these dimensions from the old EasyOCR rectangular crop.
        # Gate B actually used QUAD rectification + 8% edge padding. Those sizes
        # need not match; preserve the Gate B pixels and record both for audit.
        archive_size = [sample.get("crop_width"), sample.get("crop_height")]
        if list(crop.size) != archive_size:
            log("CROP_SIZE_METADATA_DIFFERENCE", region_id=sample["region_id"],
                archive_size=archive_size, gate_b_crop_size=list(crop.size))
        for variant in VARIANTS:
            image = make_variant(crop, variant)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            relative = f"crops/{index:02d}-{variant}.png"
            destination = output / relative
            if destination.exists() and destination.read_bytes() != buffer.getvalue():
                raise ValueError("Crop content changed; choose a separate SHARPEN_OUTPUT_ROOT")
            atomic_write(destination, buffer.getvalue())
            tasks.append({"region_id": sample["region_id"], "variant": variant,
                          "sample_row_sha256": sample["sample_row_sha256"],
                          "keyframe_uid": sample["keyframe_uid"], "video_id": sample["video_id"],
                          "archive_crop_size": archive_size, "gate_b_crop_size": list(crop.size),
                          "image_file": relative, "image_sha256": digest(buffer.getvalue()),
                          "source_image_sha256": digest(payload)})
        log("CROP_READY", crops=index, total=30)
    return tasks


def load_checkpoint(output, signature, expected, restore=None):
    path = output / "checkpoint.json"
    if not path.exists() and restore:
        restore = Path(restore)
        if restore.is_dir():
            payload = (restore / "checkpoint.json").read_bytes()
        else:
            with zipfile.ZipFile(restore) as archive:
                payload = archive.read("checkpoint.json")
        atomic_write(path, payload)
    if not path.exists():
        return []
    state = json.loads(path.read_bytes())
    if state.get("signature") != signature or state.get("rows_sha256") != digest(encoded(state.get("rows"))):
        raise ValueError("Checkpoint signature/checksum mismatch; use a separate output directory")
    rows = state["rows"]
    seen = set()
    for row in rows:
        key = (row["region_id"], row["variant"])
        if key not in expected or key in seen or row.get("status") != "success":
            raise ValueError("Checkpoint contains duplicate/foreign/incomplete rows")
        if any(row.get(k) != v for k, v in expected[key].items()):
            raise ValueError("Checkpoint task identity mismatch")
        if not isinstance(row.get("text"), str) or row.get("confidence") != confidence(row.get("confidence")):
            raise ValueError("Invalid checkpoint prediction")
        seen.add(key)
    return rows


def save_checkpoint(output, signature, rows):
    payload = encoded({"signature": signature, "rows": rows, "rows_sha256": digest(encoded(rows))})
    atomic_write(output / "checkpoint.json", payload)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        # Same rows must produce identical ZIP bytes across session restarts;
        # otherwise timestamps create false competing HF checkpoints at the same count.
        for name, value in (("checkpoint.json", payload),
                            ("SHA256SUMS", f"{digest(payload)}  checkpoint.json\n")):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, value)
    atomic_write(output / "ocr-v2-sharpen-checkpoint.zip", buffer.getvalue())


class HFCheckpointStore:
    """Immutable, checksum-verified trial checkpoints; never overwrite EasyOCR archives."""
    def __init__(self, repo_id, signature):
        from huggingface_hub import HfApi
        token = os.environ.get("HF_TOKEN")
        if not token:
            try:
                from kaggle_secrets import UserSecretsClient
                token = UserSecretsClient().get_secret("HF_TOKEN")
            except Exception:
                pass
        if not token:
            raise RuntimeError("Set Kaggle Secret HF_TOKEN with write access for durable checkpoints; no inference started")
        self.api, self.repo = HfApi(token=token), repo_id
        self.token = token
        self.prefix = f"ocr/archives/batch-01/ocr-v2-sharpen/{signature}"
        if not self.api.repo_info(repo_id=repo_id, repo_type="dataset").private:
            raise RuntimeError("Expected private OCR dataset; refusing to publish trial images to a public repo")

    def restore(self, output, signature, expected):
        from huggingface_hub import hf_hub_download
        revision = self.api.repo_info(repo_id=self.repo, repo_type="dataset").sha
        files = self.api.list_repo_files(repo_id=self.repo, repo_type="dataset", revision=revision)
        candidates = []
        pattern = re.compile(r"checkpoint-(\d{6})-([0-9a-f]{64})\.zip")
        for name in files:
            path = Path(name)
            match = pattern.fullmatch(path.name)
            if path.parent.as_posix() == self.prefix and match:
                candidates.append((int(match[1]), name, match[2]))
        if not candidates:
            return
        maximum = max(c[0] for c in candidates)
        latest = [c for c in candidates if c[0] == maximum]
        if len(latest) != 1:
            raise ValueError("Conflicting HF checkpoints; do not run the same trial concurrently")
        count, name, sha = latest[0]
        path = Path(hf_hub_download(repo_id=self.repo, repo_type="dataset", revision=revision,
                                   filename=name, token=self.token))
        if digest(path.read_bytes()) != sha:
            raise ValueError("HF checkpoint ZIP checksum mismatch")
        with zipfile.ZipFile(path) as archive:
            payload = archive.read("checkpoint.json")
        staging = output / "hf-restore"
        atomic_write(staging / "checkpoint.json", payload)
        remote = load_checkpoint(staging, signature, expected)
        local = load_checkpoint(output, signature, expected)
        if len(remote) != count:
            raise ValueError("HF checkpoint count mismatch")
        remote_map = {(r["region_id"], r["variant"]): r for r in remote}
        local_map = {(r["region_id"], r["variant"]): r for r in local}
        if any(remote_map[k] != local_map[k] for k in remote_map.keys() & local_map.keys()):
            raise ValueError("Local/HF checkpoint conflict")
        if not (local_map.keys() <= remote_map.keys() or remote_map.keys() <= local_map.keys()):
            raise ValueError("Divergent local/HF checkpoints")
        if len(remote) > len(local):
            atomic_write(output / "checkpoint.json", payload)
        log("HF_RESUME_VERIFIED", completed=count, revision=revision)

    def upload(self, path, stem):
        from huggingface_hub import hf_hub_download
        path = Path(path)
        sha = digest(path.read_bytes())
        name = f"{self.prefix}/{stem}-{sha}.zip"
        # A unique content-addressed path also makes retries after an uncertain response safe.
        for attempt in range(3):
            try:
                files = self.api.list_repo_files(repo_id=self.repo, repo_type="dataset")
                if name not in files:
                    commit = self.api.upload_file(path_or_fileobj=str(path), path_in_repo=name,
                                                  repo_id=self.repo, repo_type="dataset",
                                                  commit_message=f"batch-01 OCR sharpening {stem}")
                    revision = commit.oid
                else:
                    revision = self.api.repo_info(repo_id=self.repo, repo_type="dataset").sha
                verified = Path(hf_hub_download(repo_id=self.repo, repo_type="dataset", filename=name,
                                               revision=revision, token=self.token, force_download=True))
                if digest(verified.read_bytes()) != sha:
                    raise ValueError("HF upload round-trip checksum mismatch")
                log("HF_CHECKPOINT_VERIFIED", artifact=stem, revision=revision, sha256=sha)
                return
            except TrialTimeLimit:
                raise
            except Exception as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status in (401, 403) or attempt == 2:
                    raise RuntimeError(f"HF save failed ({type(exc).__name__}); local checkpoint kept, stopping inference") from None
                log("HF_SAVE_RETRY", attempt=attempt + 1, error_type=type(exc).__name__)
                time.sleep(2 ** (attempt + 1))


class TrialTimeLimit(RuntimeError):
    pass


@contextmanager
def time_limit(seconds):
    # Kaggle Linux: interrupt a stuck minibatch as well as checking between batches.
    supported = hasattr(signal, "setitimer") and threading.current_thread() is threading.main_thread()
    if supported:
        if signal.getitimer(signal.ITIMER_REAL)[0]:
            raise RuntimeError("Another SIGALRM timer is active; run in a clean cell")
        old = signal.getsignal(signal.SIGALRM)
        def expire(*_):
            raise TrialTimeLimit("600-second recognition budget reached; resume checkpoint in a new run")
        signal.signal(signal.SIGALRM, expire)
        signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        if supported:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)


def recognize(tasks, rows, predictor, output, signature, batch_size=6, budget=600, interrupt_after=0, stats=None, durable=None):
    stats = stats if stats is not None else {}
    done = {(r["region_id"], r["variant"]) for r in rows}
    new_count, started = 0, time.monotonic()
    with time_limit(budget):
        for variant in VARIANTS:
            pending = [t for t in tasks if t["variant"] == variant and (t["region_id"], variant) not in done]
            log("VARIANT_START", variant=variant, pending=len(pending), completed=len(rows), total=len(tasks))
            for start in range(0, len(pending), batch_size):
                if time.monotonic() - started >= budget:
                    raise TrialTimeLimit("Recognition time budget reached")
                batch = pending[start:start + batch_size]
                images = []
                for task in batch:
                    with Image.open(output / task["image_file"]) as opened:
                        images.append(opened.convert("RGB"))
                batch_started = time.monotonic()
                stats["attempted_predictions"] = stats.get("attempted_predictions", 0) + len(batch)
                with heartbeat(f"predict {variant}"):
                    texts, probabilities = predictor.predict_batch(images, return_prob=True)
                if len(texts) != len(batch) or len(probabilities) != len(batch):
                    raise ValueError("Incomplete VietOCR minibatch; not checkpointing it")
                elapsed = time.monotonic() - batch_started
                completed = [{**task, "text": str(text), "confidence": confidence(prob), "status": "success",
                              "minibatch_seconds": elapsed} for task, text, prob in zip(batch, texts, probabilities)]
                # Persist atomically before claiming completion in the log.
                save_checkpoint(output, signature, rows + completed)
                rows.extend(completed)
                if durable is not None:
                    with heartbeat("saving HF checkpoint"):
                        durable.upload(output / "ocr-v2-sharpen-checkpoint.zip", f"checkpoint-{len(rows):06d}")
                new_count += len(completed)
                log("MINIBATCH_SAVED", variant=variant, completed=len(rows), total=len(tasks),
                    elapsed_seconds=round(time.monotonic() - started, 2))
                if interrupt_after and new_count >= interrupt_after:
                    raise RuntimeError("INTENTIONAL_INTERRUPT: checkpoint saved; set INTERRUPT_AFTER_NEW=0 and rerun")
    return new_count


def review_font(size):
    candidates = [os.environ.get("OCR_REVIEW_FONT", ""), "DejaVuSans.ttf", "arial.ttf"]
    # Kaggle may have no system font cache; Matplotlib bundles DejaVu independently.
    try:
        candidates.append(str(md.distribution("matplotlib").locate_file(
            "matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf")))
    except md.PackageNotFoundError:
        pass
    for name in candidates:
        if name:
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                pass
    raise RuntimeError("Vietnamese-capable font missing; set OCR_REVIEW_FONT to a TTF file")


def wrapped(text, font, width):
    # Character-based wrapping also preserves long unbroken OCR garbage, without truncation.
    output = []
    for paragraph in (str(text).split("\n") or [""]):
        current = ""
        for char in paragraph:
            if current and font.getlength(current + char) > width:
                output.append(current)
                current = ""
            current += char
        output.append(current)
    return output


def render_sheets(output, selected, tasks, rows):
    font, small = review_font(20), review_font(16)
    by_key = {(r["region_id"], r["variant"]): r for r in rows}
    task_map = {(t["region_id"], t["variant"]): t for t in tasks}
    files = []
    for index, sample in enumerate(selected, 1):
        panels = []
        for variant in VARIANTS:
            key = (sample["region_id"], variant)
            result = by_key.get(key)
            text = result["text"] if result else "[PENDING — chưa chạy xong]"
            lines = wrapped(text, font, 570)
            panels.append((variant, result, lines, task_map[key]))
        height = 350 + 27 * max(len(p[2]) for p in panels)
        sheet = Image.new("RGB", (1800, height), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((12, 8), f"{index:02d} | {sample['video_id']} | {sample['selection_group']} | {sample['region_id']}",
                  font=small, fill="black")
        for column, (variant, result, lines, task) in enumerate(panels):
            x = column * 600 + 12
            draw.text((x, 42), variant, font=font, fill="black")
            with Image.open(output / task["image_file"]) as opened:
                preview = opened.convert("RGB")
                scale = min(570 / preview.width, 205 / preview.height)
                preview = preview.resize((max(1, round(preview.width * scale)), max(1, round(preview.height * scale))),
                                         Image.Resampling.BICUBIC)
                sheet.paste(preview, (x, 80))
            score = result.get("confidence") if result else None
            draw.text((x, 295), f"confidence={score if score is not None else 'unavailable'}", font=small, fill="black")
            for line_no, line in enumerate(lines):
                draw.text((x, 325 + 27 * line_no), line, font=font, fill="black")
        name = f"sheets/compare-{index:02d}.png"
        target = io.BytesIO()
        sheet.save(target, format="PNG")
        atomic_write(output / name, target.getvalue())
        files.append(name)
    return files


def package_results(output, selected, tasks, rows, signature, error, elapsed, stats=None):
    atomic_write(output / "selected-sample.jsonl", b"".join(encoded(r) + b"\n" for r in selected))
    atomic_write(output / "recognizer-results.jsonl", b"".join(encoded(r) + b"\n" for r in rows))
    sheets = render_sheets(output, selected, tasks, rows)
    report = {"contract": CONTRACT, "signature": signature, "trial_complete": len(rows) == 90 and error is None,
              "production_ready": False, "decision": "PENDING_VISUAL_REVIEW" if len(rows) == 90 and error is None else "INCOMPLETE",
              "expected_regions": 30, "expected_predictions": 90, "completed_predictions": len(rows),
              "by_variant": dict(Counter(r["variant"] for r in rows)), "error": error,
              "recognition_wall_seconds_this_run": elapsed, "inference_stats_this_run": stats or {},
              "original_exact_matches_gate_b": sum(r["text"] == s["gate_b_vietocr_text"]
                  for s in selected for r in rows if r["variant"] == "original" and r["region_id"] == s["region_id"]),
              "other_model_calls": {"paddle": 0, "easyocr": 0, "vintern": 0, "gemini": 0},
              "limitations": ["Confidence is not accuracy; labels must refer to original crop.",
                              "Ten-minute budget starts after model is ready; packaging is separate.",
                              "VM recovery requires an HF-verified checkpoint or a previously downloaded ZIP; unfinished minibatch may rerun.",
                              "Trial does not implement or modify production workers or their HF checkpoints."]}
    atomic_write(output / "runtime-report.json", encoded(report))
    # Never replace user's already-filled review file on resume.
    review = output / "visual-review.csv"
    if not review.exists():
        text = io.StringIO(newline="")
        fields = ["region_id", "video_id", "selection_group", "original_readable", "bicubic_2x", "bicubic_2x_unsharp", "notes"]
        writer = csv.DictWriter(text, fieldnames=fields)
        writer.writeheader()
        for sample in selected:
            writer.writerow({k: sample[k] for k in fields[:3]})
        atomic_write(review, text.getvalue().encode("utf-8-sig"))
    files = ["checkpoint.json", "run-signature.json", "selected-sample.jsonl", "recognizer-results.jsonl",
             "runtime-report.json", "visual-review.csv"] + sheets + [t["image_file"] for t in tasks]
    sums = "".join(f"{digest((output / f).read_bytes())}  {f}\n" for f in files)
    atomic_write(output / "SHA256SUMS", sums.encode("ascii"))
    destination = output.parent / "ocr-v2-sharpen-results.zip"
    temporary = destination.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in files + ["SHA256SUMS"]:
            archive.write(output / name, name)
    os.replace(temporary, destination)
    log("TRIAL_BUNDLE_READY", path=str(destination), completed=len(rows), expected=90, decision=report["decision"])
    return report


def main():
    input_root = Path(globals().get("INPUT_ROOT", os.environ.get("AIC_INPUT", "/kaggle/input")))
    data_root = Path(os.environ.get("AIC_DATA", "/kaggle/working" if Path("/kaggle").exists() else "data"))
    output = Path(globals().get("SHARPEN_OUTPUT_ROOT", data_root / "ocr-v2-sharpen"))
    output.mkdir(parents=True, exist_ok=True)
    log("[0/5] ENVIRONMENT_CHECK_START")
    packages = ensure_vietocr_packages(output / "environment-wheels")
    review_font(20)  # Fail during preflight, before spending any time on recognition.
    import torch
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0).upper():
        raise RuntimeError("This trial requires a Kaggle T4; no CPU fallback")
    model_dir = Path(globals().get("VIETOCR_CACHE_DIR", data_root / "ocr-v2-gate-b"))
    model_dir.mkdir(parents=True, exist_ok=True)
    ref, reference = helpers(model_dir)
    batch_size = int(globals().get("SHARPEN_BATCH_SIZE", 6))
    if not 1 <= batch_size <= 30:
        raise ValueError("SHARPEN_BATCH_SIZE must be between 1 and 30")
    log("[1/5] INPUT_VALIDATION_START")
    with heartbeat("input validation"):
        review = discover(input_root, globals().get("REVIEW_BUNDLE", ""), "ocr-v2-review-bundle.zip",
                          {"recognition-sample.jsonl", "review-report.json"})
        gate_path = globals().get("GATE_B_RESULTS", "")
        if not gate_path and (data_root / "ocr-v2-gate-b-results.zip").exists():
            gate_path = str(data_root / "ocr-v2-gate-b-results.zip")
        gate = discover(input_root, gate_path, "ocr-v2-gate-b-results.zip",
                        {"recognizer-results.jsonl", "runtime-report.json"})
        samples, viet, evidence = load_inputs(review, gate, ref)
        selected = select_samples(samples, viet)
    log("[2/5] SAMPLE_SELECTED", count=30, low=20, control=10)
    with heartbeat("crop creation"):
        tasks = build_tasks(selected, Path(globals().get("KEYFRAMES_ROOT") or input_root), ref, output)
    runtime_hash = globals().get("SHARPEN_RUNTIME_SHA256")
    if runtime_hash is None:
        runtime_hash = digest(Path(__file__).read_bytes())
    spec = {"contract": CONTRACT, "input": evidence, "runtime_sha256": runtime_hash,
            "reference_sha256": digest(reference.encode("utf-8")), "tasks": tasks, "packages": packages,
            "model": ref.VIETOCR_WEIGHT, "configs": ref.VIETOCR_CONFIGS, "batch_size": batch_size,
            "selection": "four lowest then two highest finite confidences per video; null lowest; region_id tie-break",
            "variants": {"original": "Gate B crop unchanged", "bicubic_2x": "2x bicubic",
                         "bicubic_2x_unsharp": {"scale": 2, "radius": 1, "percent": 100, "threshold": 3}}}
    signature = digest(encoded(spec))
    expected = {(t["region_id"], t["variant"]): t for t in tasks}
    rows = load_checkpoint(output, signature, expected, globals().get("RESTORE_CHECKPOINT", ""))
    durable = None
    if bool(globals().get("DURABLE_CHECKPOINT_TO_HF", True)):
        with heartbeat("HF checkpoint restore"):
            durable = HFCheckpointStore(str(globals().get("HF_REPO_ID", "MinhThuw0103/lastdance-visual-embeddings")), signature)
            durable.restore(output, signature, expected)
            rows = load_checkpoint(output, signature, expected)
    atomic_write(output / "run-signature.json", encoded(spec))
    save_checkpoint(output, signature, rows)
    if durable is not None:
        with heartbeat("HF checkpoint preflight"):
            durable.upload(output / "ocr-v2-sharpen-checkpoint.zip", f"checkpoint-{len(rows):06d}")
    log("RESUME", completed=len(rows), remaining=90 - len(rows), signature=signature)
    error, elapsed, stats = None, 0.0, {"attempted_predictions": 0, "resumed_predictions": len(rows)}
    if len(rows) < 90:
        try:
            log("[3/5] VIETOCR_MODEL_PREPARE")
            with heartbeat("VietOCR model prepare"):
                _, config = ref.prepare_vietocr_model()
                from vietocr.tool.predictor import Predictor
                predictor = Predictor(config)
            log("[4/5] RECOGNITION_START", budget_seconds=600, pending=90 - len(rows))
            started = time.monotonic()
            try:
                with torch.inference_mode():
                    recognize(tasks, rows, predictor, output, signature, batch_size, 600,
                              int(globals().get("INTERRUPT_AFTER_NEW", 0)), stats, durable)
            finally:
                elapsed = time.monotonic() - started
                del predictor
                torch.cuda.empty_cache()
        except (Exception, KeyboardInterrupt) as exc:
            error = f"{type(exc).__name__}: {exc}"
            rows = load_checkpoint(output, signature, expected)
            log("TRIAL_INTERRUPTED", reason=error, completed=len(rows), checkpoint=str(output / "ocr-v2-sharpen-checkpoint.zip"))
    log("[5/5] REVIEW_EXPORT_START")
    with heartbeat("review export"):
        report = package_results(output, selected, tasks, rows, signature, error, elapsed, stats)
    if durable is not None:
        with heartbeat("saving review ZIP to HF"):
            durable.upload(output.parent / "ocr-v2-sharpen-results.zip", f"results-{len(rows):06d}")
    if error:
        raise RuntimeError(f"Trial incomplete; ZIP/checkpoint exported. {error}")
    return report


if __name__ == "__main__":
    main()
