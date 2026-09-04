"""Download and pin the nine OCR v2 production result ZIPs from Hugging Face."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from offline.artifacts import sha256_file
from offline.ocr_v2_snapshot import (
    BATCH_IDS,
    CONTRACT,
    OcrV2SourceArtifact,
    OcrV2SourceManifest,
    canonical_json,
    load_worker_plan,
    read_result_identity,
    result_member_hashes,
    sha256_bytes,
    worker_for_batch,
)


_SEMANTIC_RESULT_MEMBERS = (
    "run-signature.json",
    "predictions.jsonl",
    "frame-selections.jsonl",
    "residual.jsonl",
)


def _load_run_ids(path: Path, repository: str) -> dict[str, str]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("artifact_kind") != "ocr_v2_production_run_ids"
        or value.get("repository") != repository
    ):
        raise ValueError("run-ID config identity/repository mismatch")
    workers = value.get("workers")
    if not isinstance(workers, dict) or set(workers) != {"1", "2", "3", "4"}:
        raise ValueError("run-ID config needs exactly workers 1 through 4")
    if any(re.fullmatch(r"[0-9a-f]{64}", str(run_id)) is None for run_id in workers.values()):
        raise ValueError("run-ID config contains an invalid run_id")
    return {str(worker): str(run_id) for worker, run_id in workers.items()}


def _download(
    *,
    repository: str,
    revision: str,
    filename: str,
    output_root: Path,
    token: str | bool,
) -> Path:
    return Path(
        hf_hub_download(
            repo_id=repository,
            repo_type="dataset",
            revision=revision,
            filename=filename,
            local_dir=output_root,
            token=token,
        )
    )


def _entry_commit(entry: Any) -> tuple[str | None, datetime | None]:
    last_commit = getattr(entry, "last_commit", None)
    oid = getattr(last_commit, "oid", None)
    committed = getattr(last_commit, "date", None)
    if isinstance(committed, str):
        committed = datetime.fromisoformat(committed.replace("Z", "+00:00"))
    if not isinstance(oid, str) or re.fullmatch(r"[0-9a-f]{40}", oid) is None:
        return None, None
    if not isinstance(committed, datetime) or committed.tzinfo is None:
        return None, None
    return oid, committed


def _validate_identity(
    *,
    report: dict[str, Any],
    run_signature: dict[str, Any],
    batch_id: str,
    worker: str,
    run_id: str,
) -> str:
    signature = report.get("signature")
    if (
        report.get("contract") != CONTRACT
        or report.get("batch") != batch_id
        or str(report.get("worker")) != worker
        or report.get("run_id") != run_id
        or report.get("mode") != "production"
        or report.get("recognition_complete") is not True
        or report.get("complete") is not False
        or report.get("production_ready") is not False
        or not isinstance(signature, str)
        or re.fullmatch(r"[0-9a-f]{64}", signature) is None
        or run_signature.get("signature") != signature
    ):
        raise ValueError(f"{batch_id} result identity/readiness validation failed")
    return signature


def sync(
    *,
    worker_plan_path: Path,
    run_ids_path: Path,
    output_root: Path,
    revision: str | None,
    token: str | None,
) -> tuple[Path, OcrV2SourceManifest]:
    plan = load_worker_plan(worker_plan_path)
    repository = str(plan["repo"])
    run_ids = _load_run_ids(run_ids_path, repository)
    auth: str | bool = token if token else True
    api = HfApi(token=auth)
    try:
        pinned_revision = revision or str(
            api.repo_info(repo_id=repository, repo_type="dataset").sha
        )
        if re.fullmatch(r"[0-9a-f]{40}", pinned_revision) is None:
            raise ValueError("HF source revision must resolve to a 40-character commit")
        repo_entries = {
            entry.path: entry
            for entry in api.list_repo_tree(
                repo_id=repository,
                repo_type="dataset",
                revision=pinned_revision,
                recursive=True,
                expand=True,
            )
            if getattr(entry, "path", None)
        }
        repo_files = set(repo_entries)
    except HfHubHTTPError as exc:
        raise RuntimeError(
            "Không đọc được HF Dataset riêng tư. Chạy `hf auth login --force` hoặc "
            "đặt HF_TOKEN trong environment rồi chạy lại; không dán token vào source/log."
        ) from exc

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    by_batch = worker_for_batch(plan)
    artifacts: list[OcrV2SourceArtifact] = []
    for batch_id in BATCH_IDS:
        worker = by_batch[batch_id]
        run_id = run_ids[worker]
        prefix = f"ocr/archives/{batch_id}/ocr-v2/{run_id}/production/"
        result_files = sorted(
            path
            for path in repo_files
            if path.startswith(prefix + "results-") and path.endswith(".zip")
        )
        report_files = sorted(
            path
            for path in repo_files
            if path.startswith(prefix + "reports/summary-") and path.endswith(".json")
        )
        if not result_files or not report_files:
            raise ValueError(
                f"{batch_id} thiếu production ZIP/summary trong run {run_id}; "
                f"tìm thấy ZIP={len(result_files)}, summary={len(report_files)}"
            )
        candidates: list[dict[str, Any]] = []
        try:
            for result_remote in result_files:
                result_local = _download(
                    repository=repository,
                    revision=pinned_revision,
                    filename=result_remote,
                    output_root=output_root,
                    token=auth,
                )
                result_sha = sha256_file(result_local)
                if result_remote != prefix + f"results-{result_sha}.zip":
                    raise ValueError(
                        f"{batch_id} downloaded ZIP content hash differs from HF path"
                    )
                member_hashes = result_member_hashes(
                    result_local, verify_members=True
                )
                report, run_signature, internal_report = read_result_identity(result_local)
                report_sha = sha256_bytes(internal_report)
                report_remote = prefix + f"reports/summary-{report_sha}.json"
                if report_remote not in report_files:
                    raise ValueError(
                        f"{batch_id} ZIP has no matching external summary: {report_sha}"
                    )
                report_local = _download(
                    repository=repository,
                    revision=pinned_revision,
                    filename=report_remote,
                    output_root=output_root,
                    token=auth,
                )
                if sha256_file(report_local) != report_sha:
                    raise ValueError(
                        f"{batch_id} downloaded report hash differs from HF path"
                    )
                if internal_report != report_local.read_bytes():
                    raise ValueError(
                        f"{batch_id} ZIP report differs from external summary"
                    )
                signature = _validate_identity(
                    report=report,
                    run_signature=run_signature,
                    batch_id=batch_id,
                    worker=worker,
                    run_id=run_id,
                )
                commit, committed = _entry_commit(repo_entries[result_remote])
                candidates.append(
                    {
                        "result_remote": result_remote,
                        "result_local": result_local,
                        "result_sha": result_sha,
                        "report_remote": report_remote,
                        "report_local": report_local,
                        "report_sha": report_sha,
                        "signature": signature,
                        "member_hashes": member_hashes,
                        "commit": commit,
                        "committed": committed,
                    }
                )
        except HfHubHTTPError as exc:
            raise RuntimeError(f"HF download failed for {batch_id}") from exc

        matched_reports = {candidate["report_remote"] for candidate in candidates}
        if matched_reports != set(report_files):
            raise ValueError(f"{batch_id} contains an orphan or duplicate summary")
        semantic_fingerprints = {
            tuple(candidate["member_hashes"][name] for name in _SEMANTIC_RESULT_MEMBERS)
            for candidate in candidates
        }
        if len(semantic_fingerprints) != 1:
            raise ValueError(
                f"{batch_id} has multiple production exports with different OCR content"
            )
        candidates.sort(
            key=lambda candidate: (
                candidate["committed"] or datetime.min.replace(tzinfo=UTC),
                candidate["result_remote"],
            )
        )
        selected = candidates[-1]
        equivalent = sorted(
            candidates[:-1], key=lambda candidate: candidate["result_remote"]
        )
        result_remote = selected["result_remote"]
        result_local = selected["result_local"]
        result_sha = selected["result_sha"]
        report_remote = selected["report_remote"]
        report_local = selected["report_local"]
        report_sha = selected["report_sha"]
        signature = selected["signature"]
        committed = selected["committed"]
        artifact = OcrV2SourceArtifact(
            batch_id=batch_id,
            worker=worker,
            run_id=run_id,
            signature=signature,
            result_path=result_remote,
            result_sha256=result_sha,
            result_bytes=result_local.stat().st_size,
            report_path=report_remote,
            report_sha256=report_sha,
            report_bytes=report_local.stat().st_size,
            source_commit=selected["commit"] if committed is not None else None,
            source_committed_utc=(
                committed.astimezone(UTC).isoformat() if committed is not None else None
            ),
            equivalent_result_sha256=tuple(
                candidate["result_sha"] for candidate in equivalent
            ),
            equivalent_report_sha256=tuple(
                candidate["report_sha"] for candidate in equivalent
            ),
        )
        artifacts.append(artifact)
        if equivalent:
            print(
                f"DEDUPLICATED {batch_id} equivalent_exports={len(candidates)} "
                f"selected={result_sha[:12]}",
                flush=True,
            )
        print(
            f"SYNCED {batch_id} worker={worker} result={result_sha[:12]} "
            f"report={report_sha[:12]}",
            flush=True,
        )

    manifest = OcrV2SourceManifest(
        created_utc=datetime.now(UTC).isoformat(),
        repository=repository,
        revision=pinned_revision,
        worker_plan_sha256=plan["plan_sha256"],
        artifacts=tuple(artifacts),
    )
    semantic = sha256_bytes(
        canonical_json(
            {
                "repository": repository,
                "revision": pinned_revision,
                "worker_plan_sha256": plan["plan_sha256"],
                "artifacts": [
                    artifact.model_dump(mode="json")
                    for artifact in sorted(artifacts, key=lambda item: item.batch_id)
                ],
            }
        )
    )
    destination = output_root / f"ocr-v2-production-sources-{semantic[:12]}.json"
    payload = json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
    if destination.exists():
        existing = OcrV2SourceManifest.model_validate_json(
            destination.read_text(encoding="utf-8")
        )
        if (
            existing.repository != manifest.repository
            or existing.revision != manifest.revision
            or existing.worker_plan_sha256 != manifest.worker_plan_sha256
            or existing.artifacts != manifest.artifacts
        ):
            raise FileExistsError(f"source manifest collision: {destination}")
        return destination, existing
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, destination)
    return destination, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-plan", type=Path, required=True)
    parser.add_argument(
        "--run-ids",
        type=Path,
        default=Path("configs/ocr_v2_production_run_ids.json"),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--revision", help="Optional immutable 40-character HF commit")
    parser.add_argument("--token-env", default="HF_TOKEN")
    args = parser.parse_args()
    output_root = args.output_root
    if output_root is None:
        output_root = Path(os.environ.get("AIC_DATA", "data")) / "ocr" / "v2-production"
    destination, manifest = sync(
        worker_plan_path=args.worker_plan,
        run_ids_path=args.run_ids,
        output_root=output_root,
        revision=args.revision,
        token=os.environ.get(args.token_env),
    )
    print(
        json.dumps(
            {
                "source_manifest": str(destination),
                "repository": manifest.repository,
                "revision": manifest.revision,
                "batches": len(manifest.artifacts),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
