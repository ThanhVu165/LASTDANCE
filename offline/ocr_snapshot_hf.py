"""Fail-closed publication and round-trip verification for OCR snapshots on HF."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from offline.artifacts import sha256_file
from offline.ocr_production import ocr_hf_snapshot_root, validate_ocr_snapshot_hf_path
from offline.ocr_snapshot import OcrSnapshotManifest


SNAPSHOT_FILENAMES = ("ocr.sqlite", "coverage.json", "SHA256SUMS")


@dataclass(frozen=True)
class OcrSnapshotPublishPlan:
    snapshot_dir: Path
    snapshot_id: str
    remote_root: str
    remote_paths: tuple[str, ...]
    local_sha256: dict[str, str]


def _parse_sha256sums(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="ascii").splitlines(), start=1):
        parts = line.split("  ", maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"invalid SHA256SUMS line {line_number}")
        digest, filename = parts
        if any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"invalid SHA-256 at SHA256SUMS line {line_number}")
        if filename in values:
            raise ValueError(f"duplicate SHA256SUMS entry: {filename}")
        values[filename] = digest
    expected = {"ocr.sqlite", "coverage.json"}
    if set(values) != expected:
        raise ValueError("SHA256SUMS must contain exactly ocr.sqlite and coverage.json")
    return values


def validate_local_snapshot_for_publish(snapshot_dir: Path) -> OcrSnapshotPublishPlan:
    directory = Path(snapshot_dir).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"snapshot directory does not exist: {directory}")
    actual_names = {path.name for path in directory.iterdir()}
    if actual_names != set(SNAPSHOT_FILENAMES):
        raise ValueError(
            "snapshot directory must contain exactly: " + ", ".join(SNAPSHOT_FILENAMES)
        )
    manifest = OcrSnapshotManifest.model_validate_json(
        (directory / "coverage.json").read_text(encoding="utf-8")
    )
    if directory.name != manifest.snapshot_id:
        raise ValueError("snapshot directory name does not match coverage.json snapshot_id")
    recorded = _parse_sha256sums(directory / "SHA256SUMS")
    for filename, digest in recorded.items():
        if sha256_file(directory / filename) != digest:
            raise ValueError(f"local checksum mismatch: {filename}")
    if manifest.sqlite_sha256 != recorded["ocr.sqlite"]:
        raise ValueError("coverage.json sqlite_sha256 does not match SHA256SUMS")

    remote_root = ocr_hf_snapshot_root(manifest.snapshot_id)
    remote_paths = tuple(
        validate_ocr_snapshot_hf_path(
            f"{remote_root}/{filename}", snapshot_id=manifest.snapshot_id
        )
        for filename in SNAPSHOT_FILENAMES
    )
    return OcrSnapshotPublishPlan(
        snapshot_dir=directory,
        snapshot_id=manifest.snapshot_id,
        remote_root=remote_root,
        remote_paths=remote_paths,
        local_sha256={
            filename: sha256_file(directory / filename) for filename in SNAPSHOT_FILENAMES
        },
    )


def classify_remote_snapshot(
    *, plan: OcrSnapshotPublishPlan, repo_files: list[str]
) -> Literal["missing", "complete"]:
    under_root = {
        path
        for path in repo_files
        if PurePosixPath(plan.remote_root) in PurePosixPath(path).parents
    }
    expected = set(plan.remote_paths)
    if not under_root:
        return "missing"
    if under_root == expected:
        return "complete"
    raise RuntimeError(
        "remote snapshot namespace is partial or contains unexpected files; refusing overwrite"
    )


def publish_snapshot_and_verify(
    snapshot_dir: Path,
    *,
    repo_id: str,
    token: str,
) -> dict[str, object]:
    """Upload exactly three files in one commit, then snapshot-download and hash them."""

    if not token:
        raise RuntimeError("HF token is missing")
    from huggingface_hub import (
        CommitOperationAdd,
        HfApi,
        snapshot_download,
    )

    plan = validate_local_snapshot_for_publish(snapshot_dir)
    api = HfApi(token=token)
    info = api.repo_info(repo_id=repo_id, repo_type="dataset")
    if not bool(info.private):
        raise RuntimeError("HF Dataset must be private")
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    state = classify_remote_snapshot(plan=plan, repo_files=repo_files)
    if state == "missing":
        operations = [
            CommitOperationAdd(
                path_in_repo=remote_path,
                path_or_fileobj=plan.snapshot_dir / filename,
            )
            for filename, remote_path in zip(SNAPSHOT_FILENAMES, plan.remote_paths)
        ]
        commit = api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message=f"Add immutable OCR development snapshot {plan.snapshot_id}",
        )
        revision = str(commit.oid)
        action = "uploaded"
    else:
        revision = str(info.sha)
        action = "already_present"
    if not revision or revision == "None":
        raise RuntimeError("HF did not return a pinned commit revision")

    with tempfile.TemporaryDirectory(prefix="ocr-hf-roundtrip-") as temporary:
        downloaded_root = Path(
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                token=token,
                allow_patterns=[f"{plan.remote_root}/*"],
                local_dir=temporary,
                force_download=True,
            )
        )
        remote_dir = downloaded_root / Path(*PurePosixPath(plan.remote_root).parts)
        verified: dict[str, str] = {}
        for filename in SNAPSHOT_FILENAMES:
            remote_file = remote_dir / filename
            if not remote_file.is_file():
                raise RuntimeError(f"round-trip download missing {filename}")
            digest = sha256_file(remote_file)
            if digest != plan.local_sha256[filename]:
                raise RuntimeError(f"round-trip checksum mismatch: {filename}")
            verified[filename] = digest
        remote_recorded = _parse_sha256sums(remote_dir / "SHA256SUMS")
        if remote_recorded["ocr.sqlite"] != verified["ocr.sqlite"]:
            raise RuntimeError("remote SHA256SUMS does not verify ocr.sqlite")
        if remote_recorded["coverage.json"] != verified["coverage.json"]:
            raise RuntimeError("remote SHA256SUMS does not verify coverage.json")

    return {
        "schema_version": 1,
        "action": action,
        "repo_id": repo_id,
        "repo_type": "dataset",
        "private_repo_verified": True,
        "snapshot_id": plan.snapshot_id,
        "remote_root": plan.remote_root,
        "revision": revision,
        "round_trip_verified": True,
        "files": verified,
    }
