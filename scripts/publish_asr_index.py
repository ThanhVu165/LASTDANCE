"""Publish validated ASR files; readers reject a torn SQLite/coverage pair."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from offline.asr_snapshot_hf import validate_local_snapshot_for_publish
from offline.asr_snapshot import AsrSnapshotManifest, _load_catalog
from offline.asr_validation import validate_asr_bundle
from offline.artifacts import sha256_file


MIN_DEVELOPMENT_COVERAGE = 0.90
MAX_DEVELOPMENT_ERROR_FRACTION = 0.05


def publish_asr_index(
    snapshot_dir: Path,
    *,
    data_root: Path,
    allow_partial: bool = False,
    output_report: Path | None = None,
) -> Path:
    plan = validate_local_snapshot_for_publish(snapshot_dir)
    catalog_path = Path(data_root) / "index" / "frames.csv"
    catalog, _ = _load_catalog(catalog_path, None)
    manifest = validate_asr_bundle(plan.snapshot_dir / "asr.sqlite", plan.snapshot_dir / "coverage.json",
                                  catalog_sha256=sha256_file(catalog_path), frames=catalog)
    if manifest.covered_videos != manifest.catalog_videos and not allow_partial:
        raise RuntimeError("ASR incomplete/error/unverified silence coverage; use --allow-partial for development")
    is_partial = manifest.covered_videos != manifest.catalog_videos
    attempted = manifest.success_videos + manifest.silent_videos + manifest.error_videos
    error_fraction = manifest.error_videos / attempted if attempted else 1.0
    if allow_partial and is_partial:
        if manifest.coverage_fraction <= MIN_DEVELOPMENT_COVERAGE:
            raise RuntimeError("partial ASR coverage must be greater than 90%")
        if error_fraction >= MAX_DEVELOPMENT_ERROR_FRACTION:
            raise RuntimeError("partial ASR error fraction must be less than 5%")
    destination = Path(data_root) / "index" / "asr.sqlite"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(".asr.sqlite.staging")
    coverage_temporary = destination.with_name(".asr.coverage.json.staging")
    shutil.copy2(plan.snapshot_dir / "asr.sqlite", temporary)
    shutil.copy2(plan.snapshot_dir / "coverage.json", coverage_temporary)
    os.replace(temporary, destination)
    os.replace(coverage_temporary, destination.with_name("asr.coverage.json"))
    if output_report is not None:
        report = {
            "schema_version": 1, "action": "publish", "timestamp_utc": datetime.now(UTC).isoformat(),
            "snapshot_id": manifest.snapshot_id, "allow_partial": allow_partial,
            "coverage_fraction": manifest.coverage_fraction, "error_videos": manifest.error_videos,
            "error_fraction": error_fraction if is_partial else 0.0,
            "missing_videos": manifest.missing_videos,
            "unverified_silent_videos": manifest.unverified_silent_videos,
            "production_ready": False,
        }
        output_report.parent.mkdir(parents=True, exist_ok=True)
        output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output-report", type=Path)
    args = parser.parse_args()
    root = args.data_root or Path(os.environ.get("AIC_DATA", "data"))
    path = publish_asr_index(args.snapshot_dir, data_root=root, allow_partial=args.allow_partial, output_report=args.output_report)
    print(json.dumps({"asr_sqlite": str(path), "coverage": str(path.with_name("asr.coverage.json"))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
