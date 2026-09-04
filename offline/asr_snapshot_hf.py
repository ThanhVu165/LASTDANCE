"""Publish and checksum-verify immutable ASR snapshots in a private HF Dataset."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from offline.artifacts import sha256_file
from offline.asr_production import asr_hf_snapshot_root, validate_asr_snapshot_hf_path
from offline.asr_snapshot import AsrSnapshotManifest

SNAPSHOT_FILENAMES = ("asr.sqlite", "coverage.json", "SHA256SUMS")


def _parse_sums(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, line in enumerate(
        Path(path).read_text(encoding="ascii").splitlines(), start=1
    ):
        parts = line.split("  ", 1)
        if (
            len(parts) != 2
            or len(parts[0]) != 64
            or any(character not in "0123456789abcdef" for character in parts[0])
            or not parts[1]
            or parts[1] in values
        ):
            raise ValueError(f"invalid SHA256SUMS line {line_number}")
        values[parts[1]] = parts[0]
    if set(values) != {"asr.sqlite", "coverage.json"}:
        raise ValueError("SHA256SUMS must contain exactly asr.sqlite and coverage.json")
    return values


@dataclass(frozen=True)
class AsrSnapshotPublishPlan:
    snapshot_dir: Path
    snapshot_id: str
    remote_root: str
    remote_paths: tuple[str, ...]
    local_sha256: dict[str, str]


def validate_local_snapshot_for_publish(snapshot_dir: Path) -> AsrSnapshotPublishPlan:
    directory = Path(snapshot_dir).resolve()
    if not directory.is_dir() or {p.name for p in directory.iterdir()} != set(SNAPSHOT_FILENAMES):
        raise ValueError("snapshot directory must contain exactly asr.sqlite, coverage.json, SHA256SUMS")
    manifest = AsrSnapshotManifest.model_validate_json((directory / "coverage.json").read_text())
    if directory.name != manifest.snapshot_id:
        raise ValueError("snapshot directory does not match snapshot_id")
    recorded = _parse_sums(directory / "SHA256SUMS")
    for name, digest in recorded.items():
        if sha256_file(directory / name) != digest:
            raise ValueError(f"local checksum mismatch: {name}")
    if manifest.sqlite_sha256 != recorded["asr.sqlite"]:
        raise ValueError("coverage sqlite_sha256 does not match SHA256SUMS")
    root = asr_hf_snapshot_root(manifest.snapshot_id)
    paths = tuple(validate_asr_snapshot_hf_path(f"{root}/{name}", snapshot_id=manifest.snapshot_id) for name in SNAPSHOT_FILENAMES)
    return AsrSnapshotPublishPlan(directory, manifest.snapshot_id, root, paths, {name: sha256_file(directory / name) for name in SNAPSHOT_FILENAMES})


def classify_remote_snapshot(*, plan: AsrSnapshotPublishPlan, repo_files: list[str]) -> str:
    under = {path for path in repo_files if PurePosixPath(plan.remote_root) in PurePosixPath(path).parents}
    expected = set(plan.remote_paths)
    if not under:
        return "missing"
    if under == expected:
        return "complete"
    raise RuntimeError("remote ASR snapshot namespace is partial or unexpected")


def publish_snapshot_and_verify(snapshot_dir: Path, *, repo_id: str, token: str) -> dict[str, object]:
    if not token:
        raise RuntimeError("HF token is missing")
    from huggingface_hub import CommitOperationAdd, HfApi, snapshot_download

    plan = validate_local_snapshot_for_publish(snapshot_dir)
    api = HfApi(token=token)
    info = api.repo_info(repo_id=repo_id, repo_type="dataset")
    if not bool(info.private):
        raise RuntimeError("HF Dataset must be private")
    state = classify_remote_snapshot(plan=plan, repo_files=api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
    if state == "missing":
        commit = api.create_commit(
            repo_id=repo_id, repo_type="dataset",
            operations=[CommitOperationAdd(path_in_repo=remote, path_or_fileobj=plan.snapshot_dir / name) for name, remote in zip(SNAPSHOT_FILENAMES, plan.remote_paths)],
            commit_message=f"Add immutable ASR development snapshot {plan.snapshot_id}",
        )
        revision = str(commit.oid)
        action = "uploaded"
    else:
        revision, action = str(info.sha), "already_present"
    if not revision or revision == "None":
        raise RuntimeError("HF did not return a revision")
    roundtrip = plan.snapshot_dir.parent / f".{plan.snapshot_id}.roundtrip"
    if roundtrip.exists():
        shutil.rmtree(roundtrip)
    try:
        root = Path(snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision, token=token, allow_patterns=[f"{plan.remote_root}/*"], local_dir=roundtrip, force_download=True)) / Path(*PurePosixPath(plan.remote_root).parts)
        verified = {}
        for name in SNAPSHOT_FILENAMES:
            path = root / name
            if not path.is_file() or sha256_file(path) != plan.local_sha256[name]:
                raise RuntimeError(f"round-trip checksum mismatch: {name}")
            verified[name] = sha256_file(path)
        sums = _parse_sums(root / "SHA256SUMS")
        if any(sums[name] != verified[name] for name in ("asr.sqlite", "coverage.json")):
            raise RuntimeError("remote SHA256SUMS mismatch")
    finally:
        shutil.rmtree(roundtrip, ignore_errors=True)
    return {"schema_version": 1, "action": action, "repo_id": repo_id, "repo_type": "dataset", "private_repo_verified": True, "snapshot_id": plan.snapshot_id, "remote_root": plan.remote_root, "revision": revision, "round_trip_verified": True, "files": verified}
