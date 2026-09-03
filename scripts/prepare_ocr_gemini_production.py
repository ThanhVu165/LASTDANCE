"""Materialize Vintern calibration and build an exact API-free Gemini preflight bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from offline.ocr_gemini_preflight import (
    GeminiProductionPolicy,
    build_preflight_report,
    build_shot_requests,
    residual_regions_from_materialized,
)
from offline.ocr_vintern_calibration import (
    VinternCalibrationPolicy,
    VinternCalibrationTable,
    materialize_calibrated_gate2_frames,
    validate_materialized_calibration_audit,
)
from offline.ocr_vintern_gate2 import VinternGate2Policy


BATCH_IDS = tuple(f"batch-{index:02d}" for index in range(1, 10))
CATALOG_SHA256 = "ee9693e75580527a0a257e9ba003984e105b059b716922c03c7a0b72b1508a37"
BATCH_MAPPING_SHA256 = "e7e519e5fe3e47c3e487bfe0522c09c3f0bae6c7f67dff2d31168aead0b911d2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
                raise ValueError(f"duplicate {identity} in {path}: {value}")
            seen.add(value)
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_sums(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        if len(digest) != 64 or Path(name).name != name or name in result:
            raise RuntimeError(f"invalid checksum line in {path}")
        result[name] = digest
    return result


def find_archive(
    root: Path, batch_id: str, layer: str, *, required: bool = True
) -> Path | None:
    name = f"ocr-production-{batch_id}-{layer}.zip"
    matches = sorted(path.resolve() for path in root.rglob(name) if path.is_file())
    if not matches and not required:
        return None
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one {name}, found {matches}")
    return matches[0]


def extract_verified_archive(archive_path: Path, destination: Path) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError(f"CRC failure: {archive_path}")
        names = archive.namelist()
        if len(names) != len(set(names)) or any(Path(name).name != name for name in names):
            raise RuntimeError(f"unsafe/duplicate archive members: {archive_path}")
        if any(Path(name).suffix.casefold() in {".jpg", ".jpeg", ".png", ".webp", ".mp4"} for name in names):
            raise RuntimeError(f"media leaked into archive: {archive_path}")
        archive.extractall(destination)
    sums = parse_sums(destination / "SHA256SUMS")
    for name, expected in sums.items():
        path = destination / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"archive checksum mismatch: {archive_path}!{name}")
    manifest = json.loads((destination / "batch-manifest.json").read_text(encoding="utf-8"))
    return {"manifest": manifest, "archive_sha256": sha256_file(archive_path)}


def resolve_hf_artifacts(args: argparse.Namespace) -> tuple[Path, str | None]:
    if args.artifact_root is not None:
        return args.artifact_root.resolve(), None
    from huggingface_hub import HfApi, get_token, snapshot_download

    # Prefer an explicit environment token for unattended runs, but also honor
    # the credential saved by `hf auth login` on an interactive workstation.
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise RuntimeError(
            "Hugging Face authentication is required when --artifact-root is omitted; "
            "run `hf auth login` or set HF_TOKEN"
        )
    api = HfApi(token=token)
    info = api.repo_info(repo_id=args.hf_repo_id, repo_type="dataset", revision=args.hf_revision)
    if not bool(info.private):
        raise RuntimeError("HF Dataset must be private")
    revision = str(info.sha)
    destination = args.download_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    root = Path(
        snapshot_download(
            repo_id=args.hf_repo_id,
            repo_type="dataset",
            revision=revision,
            token=token,
            local_dir=destination,
            allow_patterns=[
                "ocr/archives/*/easyocr/ocr-production-*-easyocr.zip",
                "ocr/archives/*/vintern/ocr-production-*-vintern.zip",
            ],
        )
    )
    return root, revision


def validate_manifests(
    batch_id: str,
    easy: dict[str, Any],
    vintern: dict[str, Any] | None,
) -> None:
    easy_manifest = easy["manifest"]
    if not (
        easy_manifest.get("batch_id") == batch_id
        and easy_manifest.get("tier") == "easyocr"
        and easy_manifest.get("complete") is True
        and easy_manifest.get("catalog_sha256") == CATALOG_SHA256
        and easy_manifest.get("batch_mapping_sha256") == BATCH_MAPPING_SHA256
    ):
        raise RuntimeError(f"{batch_id} EasyOCR manifest provenance/completion mismatch")
    if vintern is None:
        return
    vintern_manifest = vintern["manifest"]
    if not (
        vintern_manifest.get("batch_id") == batch_id
        and vintern_manifest.get("layer") == "vintern"
        and vintern_manifest.get("complete") is True
        and vintern_manifest.get("calibrated") is False
        and vintern_manifest.get("searchable") is False
        and vintern_manifest.get("catalog_sha256") == CATALOG_SHA256
        and vintern_manifest.get("batch_mapping_sha256") == BATCH_MAPPING_SHA256
        and vintern_manifest.get("source_easyocr", {}).get("archive_sha256")
        == easy["archive_sha256"]
    ):
        raise RuntimeError(f"{batch_id} Vintern manifest provenance/completion mismatch")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--download-dir", type=Path, default=Path("tmp/ocr-production-hf"))
    parser.add_argument("--hf-repo-id", default="MinhThuw0103/lastdance-visual-embeddings")
    parser.add_argument("--hf-revision")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gate-policy", type=Path, default=root / "configs/ocr_vintern_gate2_policy.json")
    parser.add_argument("--calibration-policy", type=Path, default=root / "configs/ocr_vintern_calibration_policy_emergency_98.json")
    parser.add_argument("--calibration-table", type=Path, default=root / "configs/ocr_vintern_calibration_table_emergency_98.json")
    parser.add_argument("--gemini-policy", type=Path, default=root / "configs/ocr_gemini_production_policy.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    final_output = args.output_dir.resolve()
    if final_output.exists():
        raise FileExistsError(f"output directory already exists: {final_output}")
    output = Path(f"{final_output}.staging-{os.getpid()}")
    if output.exists():
        raise FileExistsError(f"staging directory already exists: {output}")
    output.mkdir(parents=True)
    artifact_root, hf_revision = resolve_hf_artifacts(args)
    gate_policy = VinternGate2Policy.model_validate_json(args.gate_policy.read_text(encoding="utf-8"))
    calibration_policy = VinternCalibrationPolicy.model_validate_json(args.calibration_policy.read_text(encoding="utf-8"))
    calibration_table = VinternCalibrationTable.model_validate_json(args.calibration_table.read_text(encoding="utf-8"))
    gemini_policy = GeminiProductionPolicy.model_validate_json(args.gemini_policy.read_text(encoding="utf-8"))

    all_residuals: list[dict[str, Any]] = []
    all_requests: list[dict[str, Any]] = []
    batch_summaries: dict[str, dict[str, Any]] = {}
    observed_frame_uids: set[int] = set()
    with tempfile.TemporaryDirectory(prefix="ocr-gemini-preflight-") as temporary:
        staging = Path(temporary)
        for index, batch_id in enumerate(BATCH_IDS, start=1):
            print("GEMINI_PREFLIGHT_BATCH_START", batch_id, index, "/", len(BATCH_IDS), flush=True)
            easy_archive = find_archive(artifact_root, batch_id, "easyocr")
            vintern_archive = find_archive(
                artifact_root, batch_id, "vintern", required=False
            )
            easy_dir = staging / batch_id / "easyocr"
            vintern_dir = staging / batch_id / "vintern"
            easy_evidence = extract_verified_archive(easy_archive, easy_dir)
            vintern_evidence = (
                None
                if vintern_archive is None
                else extract_verified_archive(vintern_archive, vintern_dir)
            )
            validate_manifests(batch_id, easy_evidence, vintern_evidence)
            easy_rows = read_jsonl(easy_dir / "easyocr-frames.jsonl", "keyframe_uid")
            vintern_rows = (
                []
                if vintern_evidence is None
                else read_jsonl(vintern_dir / "vintern-results.jsonl", "candidate_id")
            )
            frame_uids = {int(row["keyframe_uid"]) for row in easy_rows}
            if observed_frame_uids & frame_uids:
                raise RuntimeError(f"cross-batch keyframe UID overlap: {batch_id}")
            observed_frame_uids.update(frame_uids)
            materialized, audit = materialize_calibrated_gate2_frames(
                easy_rows,
                vintern_rows,
                table=calibration_table,
                calibration_policy=calibration_policy,
                gate_policy=gate_policy,
            )
            validate_materialized_calibration_audit(materialized, audit, table=calibration_table)
            if vintern_evidence is None:
                for frame in materialized:
                    frame["materialized_text_policy"] = (
                        "EasyOCR with Vintern unavailable; routed candidates remain Gemini residual"
                    )
                    for region in frame.get("regions") or []:
                        region["vintern_bypassed"] = bool(region.get("vintern_candidate"))
            residuals = residual_regions_from_materialized(materialized, batch_id=batch_id)
            requests = build_shot_requests(residuals, policy=gemini_policy)
            materialized_path = output / f"{batch_id}.ocr-calibrated-frames.jsonl"
            audit_path = output / f"{batch_id}.vintern-overrides-audit.jsonl"
            write_jsonl(materialized_path, materialized)
            write_jsonl(audit_path, audit)
            all_residuals.extend(residuals)
            all_requests.extend(requests)
            batch_summaries[batch_id] = {
                "frames": len(easy_rows),
                "regions": sum(len(row.get("regions") or []) for row in easy_rows),
                "vintern_candidates": int(
                    easy_evidence["manifest"].get("vintern_candidates", 0)
                ),
                "vintern_results": len(vintern_rows),
                "vintern_status": (
                    "complete_calibrated_materialization"
                    if vintern_evidence is not None
                    else "not_available_bypassed_to_gemini"
                ),
                "vintern_overrides": sum(bool(row.get("overwritten")) for row in audit),
                "gemini_residual_regions": len(residuals),
                "gemini_residual_frames": len({row["keyframe_uid"] for row in residuals}),
                "gemini_requests": len(requests),
                "easyocr_archive_sha256": easy_evidence["archive_sha256"],
                "vintern_archive_sha256": (
                    None if vintern_evidence is None else vintern_evidence["archive_sha256"]
                ),
                "materialized_sha256": sha256_file(materialized_path),
                "override_audit_sha256": sha256_file(audit_path),
            }
            print("GEMINI_PREFLIGHT_BATCH_DONE", batch_id, batch_summaries[batch_id], flush=True)

    all_residuals.sort(key=lambda row: (row["batch_id"], row["video_id"], row["shot_id"], row["local_idx"], row["region_id"]))
    all_requests.sort(key=lambda row: (row["batch_id"], row["video_id"], row["shot_id"], row["request_id"]))
    residual_path = output / "gemini-residual-regions.jsonl"
    requests_path = output / "gemini-request-manifest.jsonl"
    write_jsonl(residual_path, all_residuals)
    write_jsonl(requests_path, all_requests)
    report = build_preflight_report(
        all_residuals,
        all_requests,
        policy=gemini_policy,
        batch_summaries=batch_summaries,
    )
    report.update(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "catalog_sha256": CATALOG_SHA256,
            "batch_mapping_sha256": BATCH_MAPPING_SHA256,
            "hf_repo_id": args.hf_repo_id if hf_revision else None,
            "hf_revision": hf_revision,
            "calibration": {
                "policy_sha256": sha256_file(args.calibration_policy),
                "table_sha256": sha256_file(args.calibration_table),
                "evidence_tier": calibration_policy.evidence_tier,
                "ground_truth_frames": calibration_table.ground_truth_frames,
            },
            "vintern_requirement": {
                "required_for_gemini_preflight": False,
                "completed_batches_used": sorted(
                    batch_id
                    for batch_id, summary in batch_summaries.items()
                    if summary["vintern_status"] == "complete_calibrated_materialization"
                ),
                "bypassed_batches": sorted(
                    batch_id
                    for batch_id, summary in batch_summaries.items()
                    if summary["vintern_status"] == "not_available_bypassed_to_gemini"
                ),
                "rule": (
                    "Use calibrated Vintern when a complete verified archive exists; "
                    "otherwise route every EasyOCR router-v2 candidate directly to Gemini."
                ),
            },
            "files": {
                "gemini-residual-regions.jsonl": sha256_file(residual_path),
                "gemini-request-manifest.jsonl": sha256_file(requests_path),
            },
        }
    )
    report_path = output / "ocr-gemini-preflight-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sums_path = output / "SHA256SUMS"
    bundle_files = [report_path, residual_path, requests_path]
    sums_path.write_text("".join(f"{sha256_file(path)}  {path.name}\n" for path in bundle_files), encoding="ascii")
    bundle_files.append(sums_path)
    bundle_path = output / "ocr-gemini-preflight.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in bundle_files:
            archive.write(path, path.name)
    report_sha = sha256_file(report_path)
    bundle_sha = sha256_file(bundle_path)
    os.replace(output, final_output)
    report_path = final_output / report_path.name
    bundle_path = final_output / bundle_path.name
    print(json.dumps(report["exact_counts"], ensure_ascii=False), flush=True)
    print(json.dumps(report["cost"], ensure_ascii=False), flush=True)
    print("GEMINI_PREFLIGHT_READY", report_path, report_sha, flush=True)
    print("GEMINI_PREFLIGHT_BUNDLE", bundle_path, bundle_sha, flush=True)
    print("GEMINI_API_CALLED", False, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
