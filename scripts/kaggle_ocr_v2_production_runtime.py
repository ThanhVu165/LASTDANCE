"""OCR v2 recognition worker. Import-safe: no GPU, network or filesystem side effects.

Outputs are versioned recognition/selection evidence, NOT legacy terminal envelopes
or an Online SQLite snapshot. Models run in separate Kaggle subprocesses.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata as md
import io
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import threading
import time
import types
import unicodedata
import zipfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

CONTRACT = "ocr-v2-recognition-worker-v1"
BATCH_IDS = tuple(f"batch-{i:02d}" for i in range(1, 10))
MODEL_NAMES = ("vietocr", "paddle")
POLICY = {"vietocr_low": 0.60, "paddle_override": 0.90, "decode_limit": 128,
          "vietocr_batch": 64, "paddle_batch": 128, "crop": "pil_quad_v1_pad08_edge"}
REFERENCE_NAMES = {
    "VIETOCR_WEIGHT", "VIETOCR_CONFIGS", "PADDLE_MODEL", "PADDLE_MODEL_ID",
    "CROP_SPEC_ID", "_float_list", "_edge_pad", "rectify_region_crop",
    "sha256_file", "download_verified", "prepare_vietocr_model", "prepare_paddle_model",
}
REQUIRED = {"easyocr-frames.jsonl", "vintern-candidates.jsonl", "run-signature.json",
            "batch-manifest.json", "SHA256SUMS"}


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def log(event, **values):
    print(event, encoded(values).decode("utf-8"), flush=True)


@contextmanager
def heartbeat(stage, **identity):
    stop, started = threading.Event(), time.monotonic()
    def tick():
        while not stop.wait(30):
            log("HEARTBEAT", phase=stage, elapsed=round(time.monotonic() - started), **identity)
    thread = threading.Thread(target=tick, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1)


def reference_source(source):
    nodes = []
    for node in ast.parse(source).body:
        name = node.name if isinstance(node, ast.FunctionDef) else None
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if name in REFERENCE_NAMES:
            nodes.append(node)
    if len(nodes) != len(REFERENCE_NAMES):
        raise ValueError("Gate B crop/model helper contract changed")
    return ast.unparse(ast.Module(body=nodes, type_ignores=[])) + "\n"


def helpers(model_dir):
    source = reference_source(Path(__file__).with_name("kaggle_ocr_v2_gate_b_runtime.py").read_text(encoding="utf-8"))
    namespace = {}
    exec("from __future__ import annotations\nimport hashlib, math, re, json, tarfile, urllib.request\n"
         "from pathlib import Path\nfrom PIL import Image\nfrom typing import Any\n" + source, namespace)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    namespace["OUTPUT_ROOT"] = Path(model_dir)
    return types.SimpleNamespace(**namespace), sha(source.encode())


def uid_for(video, shot, local):
    return int.from_bytes(hashlib.blake2b(f"{video}:{shot}:{local}".encode(), digest_size=8).digest(), "big") >> 1


def uid_hash(values):
    return sha("".join(f"{v}\n" for v in sorted(values)).encode())


def catalog_hashes(path):
    path = Path(path)
    return {"catalog_sha256": file_sha(path),
            "catalog_state_sha256": file_sha(path.with_name(path.name + ".state.json"))}


def resolve_catalog(config):
    """Find the attached Kaggle catalog; never look for it in the HF output repo."""
    explicit = config.get("catalog_path")
    if explicit:
        paths = [Path(explicit)]
        if not paths[0].is_file():
            raise ValueError(f"CATALOG_PATH does not exist: {explicit!r}; use a Kaggle Input file path")
    else:
        root = Path(config.get("input_root") or os.environ.get("AIC_DATA", "data"))
        paths = sorted(p for p in root.rglob("frames.csv") if p.is_file())
        if not paths:
            raise ValueError(f"No frames.csv under {root}; attach the catalog Kaggle dataset "
                             "or set CATALOG_PATH to its local file path. HF stores OCR artifacts.")
    candidates = [{"path": str(p), "state": p.with_name(p.name + ".state.json").is_file()} for p in paths]
    log("CATALOG_CANDIDATES", candidates=candidates)
    if any(not entry["state"] for entry in candidates):
        raise ValueError("Missing frames.csv.state.json beside a catalog candidate; attach the original "
                         f"state or set CATALOG_PATH to a complete catalog pair: {candidates!r}")
    # Batch datasets may carry identical full-catalog copies. Accept only byte-identical pairs.
    identities = {tuple(catalog_hashes(p).values()) for p in paths}
    if len(identities) != 1:
        raise ValueError(f"Multiple different local catalogs; set CATALOG_PATH explicitly: {candidates!r}")
    log("CATALOG_SELECTED", path=str(paths[0]), identical_copies=len(paths))
    return paths[0]


def load_catalog(path):
    path = Path(path)
    state = json.loads(path.with_name(path.name + ".state.json").read_bytes())
    if state.get("schema_version") != 1 or state.get("complete") is not True or state.get("csv_sha256") != file_sha(path):
        raise ValueError("Catalog state/hash is not complete and valid")
    result = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["video_id", "local_idx", "frame_id", "pts_time", "shot_id", "window_id", "keyframe_uid"]:
            raise ValueError("Unexpected frames.csv schema")
        for row in reader:
            for key in ("local_idx", "frame_id", "keyframe_uid"):
                row[key] = int(row[key])
            row["pts_time"] = float(row["pts_time"])
            uid = row["keyframe_uid"]
            if (uid in result or uid <= 0 or row["local_idx"] < 0 or row["frame_id"] < 0
                    or not math.isfinite(row["pts_time"]) or row["pts_time"] < 0
                    or uid != uid_for(row["video_id"], row["shot_id"], row["local_idx"])):
                raise ValueError("Invalid catalog identity/mapping")
            result[uid] = row
    videos = {r["video_id"] for r in result.values()}
    source_videos = [r["video_id"] for r in state.get("sources", [])]
    if (len(result) != state.get("record_count") or len(videos) != state.get("video_count")
            or len(source_videos) != len(set(source_videos)) or set(source_videos) != videos):
        raise ValueError("Catalog counts/source videos mismatch")
    return result


def load_archive(path, catalog):
    """Validate archive byte checksums, canonical UID mapping and every region ID."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if (len(names) != len(set(names)) or not REQUIRED <= set(names)
                or set(names) - REQUIRED - {"errors-history.jsonl"}):
            raise ValueError("Unexpected/duplicate archive members")
        checks = {}
        for line in archive.read("SHA256SUMS").decode("ascii").splitlines():
            digest, name = line.split(maxsplit=1)
            if name in checks or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("Invalid archive SHA256SUMS")
            checks[name] = digest
        if set(checks) != set(names) - {"SHA256SUMS"}:
            raise ValueError("Archive checksum coverage mismatch")
        for name, digest in checks.items():
            h = hashlib.sha256()
            with archive.open(name) as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    h.update(block)
            if h.hexdigest() != digest:
                raise ValueError(f"Archive checksum mismatch: {name}")
        manifest = json.loads(archive.read("batch-manifest.json"))
        batch = manifest.get("batch_id")
        if batch not in BATCH_IDS or manifest.get("tier") != "easyocr" or manifest.get("complete") is not True:
            raise ValueError("Incomplete or wrong archive layer")
        rows, seen, regions = [], set(), set()
        with archive.open("easyocr-frames.jsonl") as handle:
            for line in io.TextIOWrapper(handle, encoding="utf-8"):
                if not line.strip():
                    continue
                row = json.loads(line)
                uid = int(row["keyframe_uid"])
                canonical = catalog.get(uid)
                if uid in seen or canonical is None or any(row[k] != canonical[k] for k in ("video_id", "shot_id", "local_idx")):
                    raise ValueError("Duplicate/foreign UID or catalog mapping drift")
                seen.add(uid)
                source = PurePosixPath(row["source_image"])
                if source.is_absolute() or ".." in source.parts or "\\" in row["source_image"] or ":" in row["source_image"] or source.parent.name != row["video_id"]:
                    raise ValueError("Invalid archive source_image")
                row["frame_id"], row["pts_time"] = canonical["frame_id"], canonical["pts_time"]
                if row.get("status") not in {"no_text", "text_detected", "error"}:
                    raise ValueError("Unknown CRAFT frame status")
                if row["status"] == "no_text" and row.get("regions"):
                    raise ValueError("No-text frame contains regions")
                for i, region in enumerate(row.get("regions") or []):
                    box = region["bbox_px"]
                    if len(box) != 8 or not all(math.isfinite(float(v)) for v in box):
                        raise ValueError("Invalid source bbox")
                    expected = sha(f"{uid}|{i}|{';'.join(f'{v:.2f}' for v in box)}".encode())[:24]
                    rid = region["region_id"]
                    if rid != expected or rid in regions:
                        raise ValueError("Duplicate or unstable region ID")
                    regions.add(rid)
                rows.append(row)
                if len(rows) % 10000 == 0:
                    log("ARCHIVE_SCAN", batch=batch, frames=len(rows), regions=len(regions))
    if (len(rows) != manifest.get("frames") or len(rows) != manifest.get("expected_frames")
            or len(regions) != manifest.get("regions")
            or uid_hash(seen) != manifest.get("assigned_uid_sha256")
            or uid_hash(seen) != manifest.get("observed_uid_sha256")
            or dict(Counter(r["status"] for r in rows)) != manifest.get("status")):
        raise ValueError("Archive manifest/actual coverage mismatch")
    return manifest, rows


def allocate(batches):
    if set(batches) != set(BATCH_IDS):
        raise ValueError("Need exactly nine batches")
    loads, assignments = {str(i): 0 for i in range(1, 5)}, {str(i): [] for i in range(1, 5)}
    for batch in sorted(batches, key=lambda b: (-batches[b]["regions"], b)):
        worker = min(loads, key=lambda w: (loads[w], int(w)))
        assignments[worker].append(batch)
        loads[worker] += batches[batch]["regions"]
    return assignments, loads


class HF:
    """Read pinned inputs; write content-addressed evidence only to a private dataset."""
    def __init__(self, repo):
        from huggingface_hub import HfApi
        self.token = os.environ.get("HF_TOKEN")
        if not self.token:
            try:
                from kaggle_secrets import UserSecretsClient
                self.token = UserSecretsClient().get_secret("HF_TOKEN")
            except Exception:
                pass
        if not self.token:
            raise RuntimeError("Add Kaggle Secret HF_TOKEN (write) before starting")
        self.repo, self.api = repo, HfApi(token=self.token)
        if not self.api.repo_info(repo_id=repo, repo_type="dataset").private:
            raise RuntimeError("Expected a private OCR dataset")

    def revision(self):
        return str(self.api.repo_info(repo_id=self.repo, repo_type="dataset").sha)

    def files(self, revision):
        return self.api.list_repo_files(repo_id=self.repo, repo_type="dataset", revision=revision)

    def download(self, name, revision):
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(repo_id=self.repo, repo_type="dataset", filename=name,
                                    revision=revision, token=self.token))

    def put(self, path, name):
        if not re.fullmatch(r"ocr/archives/batch-0[1-9]/ocr-v2/[0-9a-f]{64}/(?:canary|production)/.+", name):
            raise ValueError("Refusing write outside v2 evidence namespace")
        digest = file_sha(path)
        for attempt in range(3):
            try:
                revision = self.revision()
                if name not in self.files(revision):
                    commit = self.api.upload_file(repo_id=self.repo, repo_type="dataset",
                        path_or_fileobj=str(path), path_in_repo=name,
                        commit_message=f"{name.split('/')[2]} OCR v2 checkpoint")
                    revision = commit.oid
                downloaded = self.download(name, revision)
                if file_sha(downloaded) != digest:
                    raise ValueError("HF round-trip checksum mismatch")
                log("HF_VERIFIED", artifact=name, sha256=digest, revision=revision)
                return revision
            except Exception as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status in {401, 403} or attempt == 2:
                    raise RuntimeError("HF sync failed; local results kept; STOPPING inference") from None
                log("HF_RETRY", attempt=attempt + 1, error=type(exc).__name__)
                time.sleep(2 ** attempt)


def create_plan(config, hf):
    catalog_path = resolve_catalog(config)
    catalog = load_catalog(catalog_path)
    identity = catalog_hashes(catalog_path)
    revision = config.get("input_revision") or hf.revision()
    files = hf.files(revision)
    batches, all_uids, all_videos = {}, set(), set()
    for batch in BATCH_IDS:
        pattern = f"ocr-production-{batch}-easyocr.zip"
        candidates = [n for n in files if n.startswith("ocr/archives/") and PurePosixPath(n).name == pattern]
        if len(candidates) != 1:
            raise ValueError(f"Expected one HF archive for {batch}")
        with heartbeat("PLAN_ARCHIVE", batch=batch):
            path = hf.download(candidates[0], revision)
            manifest, rows = load_archive(path, catalog)
        if manifest["batch_id"] != batch or manifest["catalog_sha256"] != identity["catalog_sha256"]:
            raise ValueError("Archive batch/catalog hash drift")
        uids = {r["keyframe_uid"] for r in rows}
        videos = {r["video_id"] for r in rows}
        if uids & all_uids or videos & all_videos:
            raise ValueError("Batch partitions overlap")
        all_uids.update(uids)
        all_videos.update(videos)
        batches[batch] = {"archive": candidates[0], "sha256": file_sha(path),
                          "regions": manifest["regions"], "frames": len(rows),
                          "uid_sha256": uid_hash(uids), "video_ids": sorted(videos)}
        log("PLAN_BATCH_VALIDATED", batch=batch, **{k: batches[batch][k] for k in ("frames", "regions")})
        del rows
    if all_uids != set(catalog):
        raise ValueError("Nine archives do not cover exactly frames.csv")
    assignments, loads = allocate(batches)
    plan = {"contract": CONTRACT, "repo": hf.repo, "input_revision": revision,
            "catalog_source": "kaggle_input", "catalog": catalog_path.name, **identity,
            "batches": batches, "assignments": assignments, "worker_regions": loads}
    plan["plan_sha256"] = sha(encoded(plan))
    atomic(Path(config["output"]) / "ocr-v2-worker-plan.json", encoded(plan))
    log("PLAN_COMPLETE", assignments=assignments, regions=loads, plan_sha256=plan["plan_sha256"])
    return plan


def validate_plan(plan):
    body = {k: v for k, v in plan.items() if k != "plan_sha256"}
    if plan.get("contract") != CONTRACT or sha(encoded(body)) != plan.get("plan_sha256"):
        raise ValueError("Worker plan signature mismatch")
    if plan.get("catalog_source") != "kaggle_input":
        raise ValueError("Worker plan must use the attached Kaggle catalog; rerun ACTION='plan' with this notebook")
    assignments, loads = allocate(plan["batches"])
    if plan["assignments"] != assignments or plan["worker_regions"] != loads:
        raise ValueError("Worker assignment drift")
    if not re.fullmatch(r"[0-9a-f]{40}", plan["input_revision"]):
        raise ValueError("Input revision must be an immutable commit")


def normalized(text):
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def score(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and 0 <= value <= 1 else None


def guards(pred):
    if pred is None:
        return ["missing_prediction"]
    reasons = []
    text = pred["text"].strip()
    if not text:
        reasons.append("empty")
    confidence = score(pred.get("confidence"))
    if confidence is None:
        reasons.append("invalid_confidence")
    elif confidence < POLICY["vietocr_low"]:
        reasons.append("low_confidence")
    words = normalized(text).split()
    if any(words[i:i+n] * 4 == words[i:i+4*n]
           for n in (1, 2, 3) for i in range(max(0, len(words) - 4*n + 1))):
        reasons.append("repeated_phrase")
    if len(text) >= POLICY["decode_limit"]:
        reasons.append("decode_limit")
    return reasons


def numeric(text):
    text = text.strip()
    if not re.fullmatch(r"[+-]?[0-9]+(?:[.,:/-][0-9]+)*(?:\s*[%₫$€])?", text):
        return False
    if re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?", text):
        parts = list(map(int, text.split(":")))
        return parts[0] < 24 and all(p < 60 for p in parts[1:])
    return True


def paddle_candidate(viet, cache_text):
    texts = (viet["text"], cache_text)
    return bool(guards(viet) or any(numeric(t) or (t.isascii() and re.search(r"[A-Za-z]{2}", t)) for t in texts))


def select_result(viet, paddle, cache_text):
    reasons = guards(viet)
    accepted = viet if not reasons else None
    engine, decision = ("vietocr", "vietocr_default") if accepted else (None, "unresolved")
    if paddle is not None and not guards(paddle) and score(paddle["confidence"]) >= POLICY["paddle_override"]:
        text = paddle["text"]
        digits = lambda t: "".join(re.findall(r"[0-9]", t))
        if numeric(text) and any(numeric(t) and digits(t) == digits(text) for t in (viet["text"], cache_text)):
            accepted, engine, decision, reasons = paddle, "paddle", "numeric_cache_or_viet_guard", []
        elif reasons and text.isascii() and re.search(r"[A-Za-z]{2}", text) and normalized(text) == normalized(cache_text):
            accepted, engine, decision, reasons = paddle, "paddle", "ascii_cache_guard", []
    if paddle is not None and decision == "vietocr_default" and normalized(paddle["text"]) != normalized(viet["text"]):
        reasons = ["model_disagreement"]
    return {"selected_text": accepted["text"] if accepted else None,
            "selected_confidence": accepted["confidence"] if accepted else None,
            "selected_engine": engine, "selection": decision, "residual_reasons": reasons}


class Journal:
    """Local FULL-sync SQLite is a worker checkpoint, never the shared FTS database."""
    def __init__(self, path, signature):
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS predictions (n INTEGER PRIMARY KEY, model TEXT, region TEXT, payload BLOB, durable INTEGER NOT NULL DEFAULT 0, UNIQUE(model,region))")
        prior = self.get_meta("signature")
        if prior is not None and prior != signature:
            self.db.close()
            raise ValueError("Checkpoint signature mismatch")
        self.set_meta("signature", signature)

    def get_meta(self, key):
        row = self.db.execute("SELECT v FROM meta WHERE k=?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key, value):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, str(value)))

    def get(self, model, region):
        row = self.db.execute("SELECT payload FROM predictions WHERE model=? AND region=?", (model, region)).fetchone()
        return json.loads(row[0]) if row else None

    def save(self, rows, durable=False):
        with self.db:
            for row in rows:
                key = row["model"], row["region_id"]
                previous = self.get(*key)
                if previous is not None and previous != row:
                    raise ValueError("Conflicting checkpoint predictions")
                if previous is None:
                    self.db.execute("INSERT INTO predictions(model,region,payload,durable) VALUES (?,?,?,?)", (*key, encoded(row), int(durable)))
                elif durable:
                    self.db.execute("UPDATE predictions SET durable=1 WHERE model=? AND region=?", key)

    def count(self, model=None):
        if model:
            return self.db.execute("SELECT count(*) FROM predictions WHERE model=?", (model,)).fetchone()[0]
        return self.db.execute("SELECT count(*) FROM predictions").fetchone()[0]

    def close(self):
        self.db.close()


def zip_payload(path, members):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        checks = "".join(f"{sha(value)}  {name}\n" for name, value in sorted(members.items())).encode()
        for name, payload in sorted({**members, "SHA256SUMS": checks}.items()):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, payload)
    atomic(path, buffer.getvalue())


class DurableJournal:
    """Immutable delta chunks: avoids reuploading all prior predictions every five minutes."""
    def __init__(self, journal, hf, prefix, output, signature, expected):
        self.journal, self.hf, self.prefix, self.output = journal, hf, prefix, Path(output)
        self.signature, self.expected = signature, expected
        self.sequence, self.previous, self.last_sync = 0, None, time.monotonic()

    def validate_rows(self, rows):
        seen = set()
        for row in rows:
            key = row.get("model"), row.get("region_id")
            if key in seen or key[0] not in MODEL_NAMES or key[1] not in self.expected:
                raise ValueError("Duplicate/foreign checkpoint row")
            seen.add(key)
            if (row.get("task_sha256") != self.expected[key[1]] or row.get("signature") != self.signature
                    or not isinstance(row.get("text"), str) or score(row.get("confidence")) != row.get("confidence")):
                raise ValueError("Checkpoint task/signature/prediction mismatch")

    def restore(self):
        for model, region, payload in self.journal.db.execute("SELECT model,region,payload FROM predictions"):
            row = json.loads(payload)
            self.validate_rows([row])
            if (row["model"], row["region_id"]) != (model, region):
                raise ValueError("Local checkpoint index/payload mismatch")
        revision = self.hf.revision()
        chunks = []
        for name in self.hf.files(revision):
            if not name.startswith(self.prefix + "/checkpoints/"):
                continue
            match = re.fullmatch(r"part-(\d{6})-([0-9a-f]{64})\.zip", PurePosixPath(name).name)
            if not match:
                raise ValueError("Unexpected checkpoint filename")
            chunks.append((int(match[1]), name, match[2]))
        restored_keys = set()
        for sequence, name, digest in sorted(chunks):
            if sequence != self.sequence:
                raise ValueError("HF checkpoint chain gap/conflict; do not duplicate a worker")
            path = self.hf.download(name, revision)
            if file_sha(path) != digest:
                raise ValueError("HF checkpoint checksum mismatch")
            with zipfile.ZipFile(path) as archive:
                manifest = json.loads(archive.read("chunk.json"))
                payload = archive.read("predictions.jsonl")
            if (manifest["signature"] != self.signature or manifest["previous_sha256"] != self.previous
                    or manifest["sequence"] != sequence or manifest["rows_sha256"] != sha(payload)):
                raise ValueError("Invalid checkpoint chain/signature")
            rows = [json.loads(line) for line in payload.splitlines()]
            self.validate_rows(rows)
            keys = {(r["model"], r["region_id"]) for r in rows}
            if keys & restored_keys or len(rows) != manifest["count"]:
                raise ValueError("Duplicate remote prediction or count mismatch")
            restored_keys.update(keys)
            self.journal.save(rows, durable=True)
            self.sequence += 1
            self.previous = digest
        # Reject a local DB which claims durability absent from the remote chain.
        local_durable = set(self.journal.db.execute("SELECT model,region FROM predictions WHERE durable=1"))
        if local_durable != restored_keys:
            raise ValueError("Local/HF durable sets diverge")
        self.journal.set_meta("last_hf", self.previous or "none")
        log("RESUME_VERIFIED", predictions=len(restored_keys), chunks=self.sequence, signature=self.signature)
        return len(restored_keys)

    def sync(self, force=False):
        if not force and time.monotonic() - self.last_sync < 300:
            return
        if time.monotonic() - self.last_sync >= 300:
            log("CHECKPOINT_DUE", seconds_since_last_sync=round(time.monotonic() - self.last_sync))
        rows = [json.loads(r[0]) for r in self.journal.db.execute("SELECT payload FROM predictions WHERE durable=0 ORDER BY n")]
        if rows:
            self.validate_rows(rows)
            payload = b"".join(encoded(row) + b"\n" for row in rows)
            manifest = {"signature": self.signature, "previous_sha256": self.previous,
                        "sequence": self.sequence, "count": len(rows), "rows_sha256": sha(payload)}
            path = self.output / "pending-checkpoint.zip"
            zip_payload(path, {"chunk.json": encoded(manifest), "predictions.jsonl": payload})
            digest = file_sha(path)
            name = f"{self.prefix}/checkpoints/part-{self.sequence:06d}-{digest}.zip"
            with heartbeat("HF_SYNC", count=len(rows)):
                self.hf.put(path, name)
            self.journal.save(rows, durable=True)
            self.sequence += 1
            self.previous = digest
            self.journal.set_meta("last_hf", digest)
        self.last_sync = time.monotonic()


def video_directories(root, videos):
    found = {}
    for current, dirs, _ in os.walk(root):
        for name in list(dirs):
            if re.fullmatch(r"L\d+_V\d+", name):
                dirs.remove(name)
                if name in videos:
                    if name in found:
                        raise ValueError(f"Multiple JPEG directories for {name}")
                    found[name] = Path(current) / name
    if set(found) != set(videos):
        raise FileNotFoundError(f"Attach keyframe datasets for missing videos: {sorted(set(videos) - set(found))[:12]}")
    return found


def build_tasks(frames, root, mode):
    from PIL import Image, ImageOps
    regions = [(f, r) for f in frames for r in f.get("regions", [])]
    if mode == "canary":
        regions = sorted(regions, key=lambda pair: sha(pair[1]["region_id"].encode()))[:256]
    directories = video_directories(root, {f["video_id"] for f, _ in regions})
    images, tasks = {}, []
    for frame, region in regions:
        image_key = frame["video_id"] + "/" + PurePosixPath(frame["source_image"]).name
        if image_key not in images:
            path = directories[frame["video_id"]] / PurePosixPath(frame["source_image"]).name
            with Image.open(path) as opened:
                size = ImageOps.exif_transpose(opened).size
            if list(size) != [frame["image_width"], frame["image_height"]]:
                raise ValueError(f"Source image geometry drift: {image_key}")
            images[image_key] = {"path": str(path), "sha256": file_sha(path), "size": list(size)}
            if len(images) % 500 == 0:
                log("IMAGE_PREFLIGHT", images=len(images), regions=len(tasks))
        task = {"region_id": region["region_id"], "keyframe_uid": frame["keyframe_uid"],
                "video_id": frame["video_id"], "frame_id": frame["frame_id"],
                "shot_id": frame["shot_id"], "source_image": frame["source_image"],
                "image_key": image_key, "source_sha256": images[image_key]["sha256"],
                "bbox_px": region["bbox_px"], "easyocr_text": region.get("easyocr_text") or ""}
        task["task_sha256"] = sha(encoded(task))
        tasks.append(task)
    return tasks, images


def environment():
    names = ("torch", "torchvision", "vietocr", "einops", "paddlepaddle-gpu", "paddleocr", "paddlex", "Pillow", "numpy")
    versions = {}
    for name in names:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            raise RuntimeError(f"Missing {name}; run setup cell first") from None
    for name, wanted in {"vietocr": "0.3.13", "paddlepaddle-gpu": "3.2.2", "paddleocr": "3.7.0"}.items():
        if versions[name] != wanted:
            raise RuntimeError(f"Use pinned {name}=={wanted}; found {versions[name]}")
    for dist in md.distributions():
        name = (dist.metadata.get("Name") or "").lower().replace("_", "-")
        if name.startswith("nvidia-"):
            versions[name] = dist.version
    return versions


def model_predictor(model, model_dir):
    h, _ = helpers(model_dir)
    if model == "vietocr":
        import torch
        from vietocr.tool.predictor import Predictor
        if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0).upper():
            raise RuntimeError("Requires Kaggle T4, no CPU fallback")
        torch.cuda.reset_peak_memory_stats(0)
        _, config = h.prepare_vietocr_model()
        # Guard must agree with the real pinned VietOCR decoder limit.
        if int(config.get("predictor", {}).get("max_seq_length", 128)) != POLICY["decode_limit"]:
            raise ValueError("VietOCR decode limit drift")
        predictor = Predictor(config)
        def predict(images):
            with torch.inference_mode():
                texts, probabilities = predictor.predict_batch(images, return_prob=True)
            if len(texts) != len(images) or len(probabilities) != len(images):
                raise ValueError("Incomplete VietOCR minibatch")
            return [(str(text), score(prob)) for text, prob in zip(texts, probabilities)]
        predict.hardware = lambda: {"gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                                    "peak_vram_mb": torch.cuda.max_memory_allocated(0) / (1024 * 1024)}
        return predict, torch.cuda.empty_cache
    import paddle
    import numpy as np
    from paddleocr import TextRecognition
    if not paddle.is_compiled_with_cuda() or "T4" not in paddle.device.cuda.get_device_name(0).upper():
        raise RuntimeError("Requires Paddle CUDA on Kaggle T4, no CPU fallback")
    paddle.device.set_device("gpu:0")
    model_dir = h.prepare_paddle_model()
    predictor = TextRecognition(model_name=h.PADDLE_MODEL_ID, model_dir=str(model_dir), device="gpu:0")
    def predict(images):
        arrays = [np.asarray(im, dtype=np.uint8)[:, :, ::-1] for im in images]
        results = list(predictor.predict(input=arrays, batch_size=len(arrays)))
        if len(results) != len(images):
            raise ValueError("Incomplete Paddle minibatch")
        parsed = []
        for row in results:
            value = row.json
            value = value() if callable(value) else value
            value = json.loads(value) if isinstance(value, str) else value
            value = value.get("res", value)
            parsed.append((str(value.get("rec_text") or ""), score(value.get("rec_score"))))
        return parsed
    def hardware():
        try:
            peak = float(paddle.device.cuda.max_memory_allocated()) / (1024 * 1024)
        except (AttributeError, RuntimeError):
            peak = None
        return {"gpu": paddle.device.cuda.get_device_name(0), "peak_vram_mb": peak}
    predict.hardware = hardware
    return predict, paddle.device.cuda.empty_cache


def task_crops(tasks, images, rectify):
    from PIL import Image, ImageOps
    loaded, crops = {}, []
    try:
        for task in tasks:
            key = task["image_key"]
            if key not in loaded:
                payload = Path(images[key]["path"]).read_bytes()
                if sha(payload) != task["source_sha256"]:
                    raise ValueError(f"Source image changed after preflight: {key}")
                with Image.open(io.BytesIO(payload)) as opened:
                    loaded[key] = ImageOps.exif_transpose(opened).convert("RGB")
            crops.append(rectify(loaded[key], task["bbox_px"]))
        return crops
    finally:
        for im in loaded.values():
            im.close()


class IntentionalStop(RuntimeError):
    pass


def recognize(tasks, model, journal, durable, predict, get_crops, *, signature,
              batch_size, release=lambda: None, interrupt_after=0, identity=None):
    identity = identity or {}
    before = journal.count(model)
    durable_before = journal.db.execute("SELECT count(*) FROM predictions WHERE model=? AND durable=1", (model,)).fetchone()[0]
    eligible = []
    for task in tasks:
        if model == "paddle":
            viet = journal.get("vietocr", task["region_id"])
            if viet is None:
                raise ValueError("Paddle cannot run before VietOCR coverage is complete")
            if not paddle_candidate(viet, task["easyocr_text"]):
                continue
        eligible.append(task)
    pending = [t for t in eligible if journal.get(model, t["region_id"]) is None]
    started, cursor, minibatches = time.monotonic(), 0, 0
    log("PHASE_START", model=model, total=len(eligible), resumed=before, pending=len(pending), **identity)
    while cursor < len(pending):
        durable.sync()
        batch = pending[cursor:cursor + batch_size]
        crops = get_crops(batch)
        try:
            predictions = predict(crops)
        except Exception as exc:
            is_oom = "out of memory" in str(exc).lower() or "resourceexhausted" in type(exc).__name__.lower()
            if not is_oom or batch_size == 1:
                durable.sync(force=True)
                raise
            release()
            batch_size = max(1, batch_size // 2)
            log("OOM_REDUCE_BATCH", model=model, batch_size=batch_size, **identity)
            continue
        finally:
            for crop in crops:
                crop.close()
        if len(predictions) != len(batch):
            raise ValueError("Incomplete predictions; minibatch not saved")
        rows = [{"model": model, "region_id": t["region_id"], "task_sha256": t["task_sha256"],
                 "signature": signature, "text": text, "confidence": score(confidence)}
                for t, (text, confidence) in zip(batch, predictions)]
        durable.validate_rows(rows)
        journal.save(rows)
        cursor += len(batch)
        minibatches += 1
        elapsed = time.monotonic() - started
        rate = cursor / max(elapsed, 1e-9)
        log("MINIBATCH_SAVED", model=model, video=batch[-1]["video_id"], done=before + cursor,
            total=len(eligible), new=cursor, elapsed=round(elapsed, 2), regions_per_second=round(rate, 2),
            eta_seconds=round((len(pending) - cursor) / rate), last_hf=journal.get_meta("last_hf"), **identity)
        if durable_before and cursor:
            journal.set_meta("resume_newwork", "true")
        if interrupt_after and minibatches >= interrupt_after:
            durable.sync(force=True)
            raise IntentionalStop("Checkpoint verified on HF. Set INTERRUPT_AFTER_MINIBATCHES=0 and rerun in a new process/session.")
    durable.sync(force=True)
    return {"model": model, "expected": len(eligible), "completed": journal.count(model),
            "new_predictions": cursor, "resumed_predictions": before,
            "resume_with_new_work": journal.get_meta("resume_newwork") == "true",
            "recognition_and_sync_seconds": time.monotonic() - started}


def phase(context_path, model):
    ctx = json.loads(Path(context_path).read_bytes())
    output = Path(ctx["output"])
    hf = HF(ctx["repo"])
    tasks = [json.loads(line) for line in (output / "tasks.jsonl").read_bytes().splitlines()]
    if sha(b"".join(encoded(t) + b"\n" for t in tasks)) != ctx["tasks_sha256"]:
        raise ValueError("Task manifest drift")
    journal = Journal(output / "checkpoint.sqlite", ctx["signature"])
    durable = DurableJournal(journal, hf, ctx["prefix"], output, ctx["signature"],
                             {t["region_id"]: t["task_sha256"] for t in tasks})
    try:
        with heartbeat("HF_RESTORE", model=model, worker=ctx["worker"], batch=ctx["batch"]):
            durable.restore()
        # Verify write access before expensive model init, including empty/no-candidate phases.
        preflight = output / "run-signature.json"
        hf.put(preflight, f"{ctx['prefix']}/preflight-{file_sha(preflight)}.json")
        initialized = []
        def lazy_predict(crops):
            if not initialized:
                with heartbeat("MODEL_INIT", model=model):
                    initialized.extend(model_predictor(model, output.parent.parent / "models"))
            return initialized[0](crops)
        h, _ = helpers(output.parent.parent / "models")
        with heartbeat("RECOGNITION", model=model, worker=ctx["worker"], batch=ctx["batch"]):
            report = recognize(tasks, model, journal, durable, lazy_predict,
                lambda group: task_crops(group, ctx["images"], h.rectify_region_crop),
                signature=ctx["signature"], batch_size=POLICY[model + "_batch"],
                release=lambda: initialized[1]() if initialized else None,
                interrupt_after=ctx["interrupt_after"], identity={"worker": ctx["worker"], "batch": ctx["batch"]})
        report["hardware"] = initialized[0].hardware() if initialized and hasattr(initialized[0], "hardware") else None
        atomic(output / f"{model}-phase.json", encoded(report))
    finally:
        journal.close()


def export_results(ctx, frames, tasks, hf, started):
    output = Path(ctx["output"])
    journal = Journal(output / "checkpoint.sqlite", ctx["signature"])
    try:
        selections = {}
        predictions = [json.loads(row[0]) for row in journal.db.execute("SELECT payload FROM predictions ORDER BY model,region")]
        for task in tasks:
            rid = task["region_id"]
            viet, paddle = journal.get("vietocr", rid), journal.get("paddle", rid)
            if viet is None or (paddle_candidate(viet, task["easyocr_text"]) and paddle is None):
                raise ValueError("Cannot export incomplete recognizer coverage")
            if not paddle_candidate(viet, task["easyocr_text"]) and paddle is not None:
                raise ValueError("Unexpected Paddle prediction")
            selections[rid] = {**task, **select_result(viet, paddle, task["easyocr_text"])}
        results, residuals = [], []
        for frame in frames:
            source_regions = frame.get("regions") or []
            if ctx["mode"] == "canary" and not any(r["region_id"] in selections for r in source_regions):
                continue
            regions = [selections[r["region_id"]] for r in source_regions if r["region_id"] in selections]
            accepted = [r for r in regions if r["selected_text"] is not None]
            status = "success" if accepted else ("no_text" if frame["status"] == "no_text" else "error")
            result = None
            if accepted:
                boxes = []
                for r in accepted:
                    boxes.append([max(0.0, min(1.0, float(v) / (frame["image_width"] if i % 2 == 0 else frame["image_height"])))
                                  for i, v in enumerate(r["bbox_px"])])
                weights = [max(1, len("".join(r["selected_text"].split()))) for r in accepted]
                confidence = sum(w * r["selected_confidence"] for w, r in zip(weights, accepted)) / sum(weights)
                result = {"frame_id": frame["frame_id"], "detected_text": [r["selected_text"] for r in accepted],
                          "bbox": boxes, "confidence": confidence, "language": "mixed"}
            residuals.extend(r for r in regions if r["residual_reasons"])
            results.append({"artifact_kind": "ocr_v2_frame_selection_v1", "batch_id": ctx["batch"],
                "signature": ctx["signature"], "keyframe_uid": frame["keyframe_uid"], "video_id": frame["video_id"],
                "frame_id": frame["frame_id"], "source_image": frame["source_image"], "status": status,
                "result": result, "regions": regions, "source_status": frame["status"],
                "source_error": frame.get("error"), "complete": False, "production_ready": False})
        report = {"contract": CONTRACT, "run_id": ctx["run_id"], "signature": ctx["signature"],
                  "mode": ctx["mode"], "worker": ctx["worker"], "batch": ctx["batch"],
                  "tasks_sha256": ctx["tasks_sha256"], "sample_task_sha256": ctx["sample_task_sha256"],
                  "regions": len(tasks), "predictions": len(predictions), "frames": len(results),
                  "status": dict(Counter(r["status"] for r in results)), "residual_regions": len(residuals),
                  "residual_frames": len({r["keyframe_uid"] for r in residuals}),
                  "residual_shots": len({(r["video_id"], r["shot_id"]) for r in residuals}),
                  "recognition_complete": True, "complete": False, "production_ready": False,
                  "resume_with_new_work": journal.get_meta("resume_newwork") == "true",
                  "end_to_end_seconds_this_run": time.monotonic() - started,
                  "phases": {model: json.loads((output / f"{model}-phase.json").read_bytes())
                             if (output / f"{model}-phase.json").is_file() else None for model in MODEL_NAMES},
                  "model_calls_saved": dict(Counter(r["model"] for r in predictions)),
                  "other_model_calls": {"easyocr": 0, "vintern": 0, "gemini": 0},
                  "limitations": ["Not a quantitative accuracy gate; canary is not a representative sample.",
                                  "Not a legacy OcrRecordEnvelope or Online snapshot; migration/union still required.",
                                  "Resume timing excludes prior process runs; do not extrapolate a resumed run as full throughput.",
                                  "Language is mixed/undetermined; no ASCII-as-English classifier."]}
        members = {"report.json": encoded(report), "run-signature.json": (output / "run-signature.json").read_bytes()}
        for name, rows in (("predictions.jsonl", predictions), ("frame-selections.jsonl", results), ("residual.jsonl", residuals)):
            members[name] = b"".join(encoded(row) + b"\n" for row in rows)
        archive = output / f"ocr-v2-{ctx['batch']}-{ctx['mode']}-results.zip"
        zip_payload(archive, members)
        report_path = output / "report.json"
        atomic(report_path, members["report.json"])
        with heartbeat("RESULT_UPLOAD", worker=ctx["worker"], batch=ctx["batch"]):
            hf.put(archive, f"{ctx['prefix']}/results-{file_sha(archive)}.zip")
            hf.put(report_path, f"{ctx['prefix']}/reports/summary-{file_sha(report_path)}.json")
        log("WORKER_BATCH_COMPLETE", output=str(archive), report_sha256=file_sha(report_path), **report)
        return report
    finally:
        journal.close()


def run_worker(config, hf):
    plan = json.loads(Path(config["plan"]).read_bytes())
    validate_plan(plan)
    if hf.repo != plan["repo"]:
        raise ValueError("HF repo differs from worker plan")
    worker = str(config["worker"])
    if worker not in plan["assignments"] or config["mode"] not in {"canary", "production"}:
        raise ValueError("Choose worker 1–4 and canary/production mode")
    catalog_path = resolve_catalog(config)
    if any(plan[key] != value for key, value in catalog_hashes(catalog_path).items()):
        raise ValueError("Worker catalog hash drift")
    catalog = load_catalog(catalog_path)
    versions = environment()
    h, reference_hash = helpers(Path(config["output"]) / "models")
    resources = {"contract": CONTRACT, "runtime_sha256": file_sha(__file__),
                 "reference_sha256": reference_hash, "policy": POLICY, "packages": versions,
                 "plan_sha256": plan["plan_sha256"], "worker": worker,
                 "vietocr_weight": h.VIETOCR_WEIGHT, "vietocr_configs": h.VIETOCR_CONFIGS, "paddle": h.PADDLE_MODEL}
    run_id = sha(encoded(resources))
    batches = plan["assignments"][worker]
    if config["mode"] == "canary":
        batches = batches[:1]
    for batch in batches:
        started = time.monotonic()
        with heartbeat("INPUT_VALIDATE", worker=worker, batch=batch):
            evidence = plan["batches"][batch]
            archive = hf.download(evidence["archive"], plan["input_revision"])
            if file_sha(archive) != evidence["sha256"]:
                raise ValueError("Input archive changed")
            manifest, frames = load_archive(archive, catalog)
            if manifest["catalog_sha256"] != plan["catalog_sha256"] or manifest["batch_id"] != batch:
                raise ValueError("Worker input identity drift")
            if (manifest["regions"] != evidence["regions"] or manifest["frames"] != evidence["frames"]
                    or manifest["observed_uid_sha256"] != evidence["uid_sha256"]
                    or sorted({f["video_id"] for f in frames}) != evidence["video_ids"]):
                raise ValueError("Worker archive/plan coverage drift")
        with heartbeat("IMAGE_PREFLIGHT", worker=worker, batch=batch):
            tasks, images = build_tasks(frames, config["keyframes"], config["mode"])
        sample = sorted(tasks, key=lambda t: sha(t["region_id"].encode()))[:256]
        sample_sha = sha(encoded([t["task_sha256"] for t in sample]))
        if config["mode"] == "canary" and len(tasks) < 128:
            raise ValueError("Canary needs at least 128 regions for interrupt/new-work proof")
        if config["mode"] == "production":
            approved = config.get("approved_canary_sha256", "")
            if not re.fullmatch(r"[0-9a-f]{64}", approved):
                raise ValueError("Run canary first, then copy report_sha256 into APPROVED_CANARY_SHA256")
            # Canary on the first assigned batch authorizes only this worker/config/runtime.
            name = f"ocr/archives/{batches[0]}/ocr-v2/{run_id}/canary/reports/summary-{approved}.json"
            report_path = hf.download(name, hf.revision())
            report = json.loads(report_path.read_bytes())
            if (file_sha(report_path) != approved or report.get("run_id") != run_id
                    or report.get("mode") != "canary" or report.get("recognition_complete") is not True
                    or report.get("resume_with_new_work") is not True
                    or (batch == batches[0] and report.get("sample_task_sha256") != sample_sha)):
                raise ValueError("Canary is incomplete, has no resume/new-work proof, or source/config changed")
        task_bytes = b"".join(encoded(t) + b"\n" for t in tasks)
        signature = sha(encoded({"run_id": run_id, "batch": batch, "mode": config["mode"], "tasks": sha(task_bytes)}))
        output = Path(config["output"]) / run_id / batch / config["mode"]
        output.mkdir(parents=True, exist_ok=True)
        prefix = f"ocr/archives/{batch}/ocr-v2/{run_id}/{config['mode']}"
        ctx = {"run_id": run_id, "signature": signature, "batch": batch, "worker": worker,
               "mode": config["mode"], "repo": hf.repo, "output": str(output), "prefix": prefix,
               "images": images, "tasks_sha256": sha(task_bytes), "sample_task_sha256": sample_sha,
               "interrupt_after": int(config.get("interrupt_after", 0))}
        atomic(output / "tasks.jsonl", task_bytes)
        atomic(output / "run-signature.json", encoded({"resources": resources, "signature": signature,
                                                        "tasks_sha256": sha(task_bytes), "batch": batch, "mode": config["mode"]}))
        atomic(output / "context.json", encoded(ctx))
        env = {**os.environ, "HF_TOKEN": hf.token, "CUDA_VISIBLE_DEVICES": "0", "TOKENIZERS_PARALLELISM": "false"}
        for model in MODEL_NAMES:
            log("LAUNCH_PHASE", worker=worker, batch=batch, model=model)
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "phase",
                                      str(output / "context.json"), "--model", model], env=env)
            if process.returncode == 75:
                log("INTENTIONAL_STOP", worker=worker, batch=batch,
                    action="Set INTERRUPT_AFTER_MINIBATCHES=0, rerun; checkpoint already verified on HF.")
                return
            if process.returncode:
                raise RuntimeError(f"{model} phase failed; fix error then rerun same signature")
        export_results(ctx, frames, tasks, hf, started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "phase"))
    parser.add_argument("config", type=Path)
    parser.add_argument("--model", choices=MODEL_NAMES)
    args = parser.parse_args()
    if args.action == "phase":
        if not args.model:
            parser.error("phase requires --model")
        try:
            phase(args.config, args.model)
        except IntentionalStop as exc:
            log("INTENTIONAL_STOP", action=str(exc))
            raise SystemExit(75)
        return
    config = json.loads(args.config.read_bytes())
    hf = HF(config["repo"])
    with heartbeat(args.action, worker=config.get("worker")):
        if args.action == "plan":
            create_plan(config, hf)
        else:
            run_worker(config, hf)


if __name__ == "__main__":
    main()
