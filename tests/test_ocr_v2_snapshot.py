import json
import shutil
import sqlite3
import tempfile
import unittest
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from offline.artifacts import sha256_file
from offline.catalog import write_frames_catalog_atomic
from offline.ocr_v2_snapshot import (
    BATCH_IDS,
    CONTRACT,
    OcrV2SourceArtifact,
    OcrV2SourceManifest,
    allocate_batches,
    build_ocr_v2_snapshot,
    canonical_json,
    sha256_bytes,
    uid_set_sha256,
    validate_ocr_v2_snapshot,
)
from shared.schemas.frame import FrameRecord
from scripts import sync_ocr_v2_results as syncer


def _zip(path: Path, members: dict[str, bytes]) -> None:
    checksums = "".join(
        f"{sha256_bytes(payload)}  {name}\n" for name, payload in sorted(members.items())
    ).encode("ascii")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in sorted({**members, "SHA256SUMS": checksums}.items()):
            archive.writestr(name, payload)


class OcrV2SnapshotTests(unittest.TestCase):
    def _fixture(self, root: Path):
        records = [
            FrameRecord(
                video_id=f"L01_V{index:03d}",
                local_idx=0,
                frame_id=index * 10,
                pts_time=float(index),
                shot_id="s000001",
                keyframe_uid=1000 + index,
            )
            for index in range(1, 10)
        ]
        catalog = root / "frames.csv"
        write_frames_catalog_atomic(
            catalog,
            records=records,
            sources=[
                {
                    "video_id": record.video_id,
                    "plan_sha256": "a" * 64,
                    "quality_sha256": "b" * 64,
                    "quality_config_signature": "c" * 64,
                }
                for record in records
            ],
        )
        state = catalog.with_name(catalog.name + ".state.json")
        batches = {
            batch_id: {
                "archive": f"ocr/archives/{batch_id}/easyocr/source.zip",
                "sha256": "d" * 64,
                "regions": 0 if index == 3 else 1,
                "frames": 1,
                "uid_sha256": uid_set_sha256({records[index - 1].keyframe_uid}),
                "video_ids": [records[index - 1].video_id],
            }
            for index, batch_id in enumerate(BATCH_IDS, start=1)
        }
        assignments, loads = allocate_batches(batches)
        plan = {
            "contract": CONTRACT,
            "repo": "private/test",
            "input_revision": "a" * 40,
            "catalog_source": "kaggle_input",
            "catalog": catalog.name,
            "catalog_sha256": sha256_file(catalog),
            "catalog_state_sha256": sha256_file(state),
            "batches": batches,
            "assignments": assignments,
            "worker_regions": loads,
        }
        plan["plan_sha256"] = sha256_bytes(canonical_json(plan))
        plan_path = root / "ocr-v2-worker-plan.json"
        plan_path.write_bytes(canonical_json(plan))

        by_batch_worker = {
            batch: worker for worker, assigned in assignments.items() for batch in assigned
        }
        resources_by_worker = {
            worker: {
                "contract": CONTRACT,
                "worker": worker,
                "plan_sha256": plan["plan_sha256"],
                "runtime_sha256": "e" * 64,
            }
            for worker in assignments
        }
        run_ids = {
            worker: sha256_bytes(canonical_json(resources))
            for worker, resources in resources_by_worker.items()
        }
        source_artifacts = []
        for index, (batch_id, frame) in enumerate(zip(BATCH_IDS, records), start=1):
            worker = by_batch_worker[batch_id]
            run_id = run_ids[worker]
            task = None
            regions = []
            predictions = []
            residuals = []
            if index != 3:
                task_without_hash = {
                    "region_id": f"region-{index}",
                    "keyframe_uid": frame.keyframe_uid,
                    "video_id": frame.video_id,
                    "frame_id": frame.frame_id,
                    "shot_id": frame.shot_id,
                    "source_image": f"keyframes/{frame.video_id}/s000001_0.jpg",
                    "image_key": f"{frame.video_id}/s000001_0.jpg",
                    "source_sha256": "f" * 64,
                    "bbox_px": [0, 0, 20, 0, 20, 10, 0, 10],
                    "easyocr_text": "CACHE",
                }
                task_sha = sha256_bytes(canonical_json(task_without_hash))
                task = {**task_without_hash, "task_sha256": task_sha}
                if index == 2:
                    selection = {
                        "selected_text": "2026",
                        "selected_confidence": 0.95,
                        "selected_engine": "paddle",
                        "selection": "numeric_cache_or_viet_guard",
                        "residual_reasons": [],
                    }
                elif index == 4:
                    selection = {
                        "selected_text": None,
                        "selected_confidence": None,
                        "selected_engine": None,
                        "selection": "unresolved",
                        "residual_reasons": ["low_confidence"],
                    }
                else:
                    selection = {
                        "selected_text": f"Văn bản {index}",
                        "selected_confidence": 0.8,
                        "selected_engine": "vietocr",
                        "selection": "vietocr_default",
                        "residual_reasons": [],
                    }
                region = {**task, **selection}
                regions = [region]
                predictions.append(
                    {
                        "model": "vietocr",
                        "region_id": task["region_id"],
                        "task_sha256": task_sha,
                        "signature": "pending",
                        "text": "202x" if index == 2 else ("" if index == 4 else selection["selected_text"]),
                        "confidence": 0.4 if index in {2, 4} else 0.8,
                    }
                )
                if index == 2:
                    predictions.append(
                        {
                            "model": "paddle",
                            "region_id": task["region_id"],
                            "task_sha256": task_sha,
                            "signature": "pending",
                            "text": "2026",
                            "confidence": 0.95,
                        }
                    )
                if selection["residual_reasons"]:
                    residuals = [region]

            tasks_payload = b"" if task is None else canonical_json(task) + b"\n"
            tasks_sha = sha256_bytes(tasks_payload)
            signature = sha256_bytes(
                canonical_json(
                    {
                        "run_id": run_id,
                        "batch": batch_id,
                        "mode": "production",
                        "tasks": tasks_sha,
                    }
                )
            )
            for prediction in predictions:
                prediction["signature"] = signature
            accepted = [region for region in regions if region["selected_text"] is not None]
            source_status = "no_text" if index == 3 else "text_detected"
            status = "success" if accepted else ("no_text" if index == 3 else "error")
            result = None
            if accepted:
                result = {
                    "frame_id": frame.frame_id,
                    "detected_text": [region["selected_text"] for region in accepted],
                    "bbox": [[0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5]],
                    "confidence": accepted[0]["selected_confidence"],
                    "language": "mixed",
                }
            frame_row = {
                "artifact_kind": "ocr_v2_frame_selection_v1",
                "batch_id": batch_id,
                "signature": signature,
                "keyframe_uid": frame.keyframe_uid,
                "video_id": frame.video_id,
                "frame_id": frame.frame_id,
                "source_image": f"keyframes/{frame.video_id}/s000001_0.jpg",
                "status": status,
                "result": result,
                "regions": regions,
                "source_status": source_status,
                "source_error": None,
                "complete": False,
                "production_ready": False,
            }
            model_counts = Counter(row["model"] for row in predictions)
            sample_ids = [] if task is None else [task["task_sha256"]]
            report = {
                "contract": CONTRACT,
                "run_id": run_id,
                "signature": signature,
                "mode": "production",
                "worker": worker,
                "batch": batch_id,
                "tasks_sha256": tasks_sha,
                "sample_task_sha256": sha256_bytes(canonical_json(sample_ids)),
                "regions": len(regions),
                "predictions": len(predictions),
                "frames": 1,
                "status": {status: 1},
                "residual_regions": len(residuals),
                "residual_frames": int(bool(residuals)),
                "residual_shots": int(bool(residuals)),
                "recognition_complete": True,
                "complete": False,
                "production_ready": False,
                "resume_with_new_work": batch_id == assignments[worker][0],
                "phases": {
                    model: {
                        "model": model,
                        "expected": model_counts[model],
                        "completed": model_counts[model],
                        "new_predictions": model_counts[model],
                        "resumed_predictions": 0,
                    }
                    for model in ("vietocr", "paddle")
                },
                "model_calls_saved": dict(model_counts),
                "other_model_calls": {"easyocr": 0, "vintern": 0, "gemini": 0},
            }
            run_signature = {
                "resources": resources_by_worker[worker],
                "signature": signature,
                "tasks_sha256": tasks_sha,
                "batch": batch_id,
                "mode": "production",
            }
            report_bytes = canonical_json(report)
            archive_path = root / f"temporary-{batch_id}.zip"
            _zip(
                archive_path,
                {
                    "report.json": report_bytes,
                    "run-signature.json": canonical_json(run_signature),
                    "predictions.jsonl": b"".join(canonical_json(row) + b"\n" for row in predictions),
                    "frame-selections.jsonl": canonical_json(frame_row) + b"\n",
                    "residual.jsonl": b"".join(canonical_json(row) + b"\n" for row in residuals),
                },
            )
            result_sha = sha256_file(archive_path)
            report_sha = sha256_bytes(report_bytes)
            prefix = f"ocr/archives/{batch_id}/ocr-v2/{run_id}/production"
            result_relative = f"{prefix}/results-{result_sha}.zip"
            report_relative = f"{prefix}/reports/summary-{report_sha}.json"
            result_local = root.joinpath(*result_relative.split("/"))
            report_local = root.joinpath(*report_relative.split("/"))
            result_local.parent.mkdir(parents=True, exist_ok=True)
            report_local.parent.mkdir(parents=True, exist_ok=True)
            archive_path.replace(result_local)
            report_local.write_bytes(report_bytes)
            source_artifacts.append(
                OcrV2SourceArtifact(
                    batch_id=batch_id,
                    worker=worker,
                    run_id=run_id,
                    signature=signature,
                    result_path=result_relative,
                    result_sha256=result_sha,
                    result_bytes=result_local.stat().st_size,
                    report_path=report_relative,
                    report_sha256=report_sha,
                    report_bytes=report_local.stat().st_size,
                )
            )
        source_manifest = OcrV2SourceManifest(
            created_utc=datetime(2026, 9, 4, 12, 0, tzinfo=UTC).isoformat(),
            repository=plan["repo"],
            revision="9" * 40,
            worker_plan_sha256=plan["plan_sha256"],
            artifacts=tuple(source_artifacts),
        )
        source_path = root / "ocr-v2-production-sources.json"
        source_path.write_text(
            json.dumps(source_manifest.model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
        )
        return catalog, state, plan_path, source_path

    def test_builds_immutable_v2_snapshot_with_exact_fts_and_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog, state, plan, sources = self._fixture(root)
            destination, manifest = build_ocr_v2_snapshot(
                catalog_path=catalog,
                catalog_state_path=state,
                worker_plan_path=plan,
                source_manifest_path=sources,
                source_root=root,
                output_root=root / "snapshots",
                created_utc=datetime(2026, 9, 4, 13, 0, tzinfo=UTC),
            )
            self.assertTrue(manifest.totals.recognition_coverage_complete)
            self.assertFalse(manifest.complete)
            self.assertFalse(manifest.production_ready)
            self.assertEqual(manifest.totals.success_keyframes, 7)
            self.assertEqual(manifest.totals.no_text_keyframes, 1)
            self.assertEqual(manifest.totals.error_keyframes, 1)
            self.assertEqual(manifest.totals.residual_regions, 1)
            self.assertEqual(manifest.fts_rows, 7)
            self.assertEqual(
                manifest.totals.selected_region_engine_counts,
                {"paddle": 1, "unresolved": 1, "vietocr": 6},
            )
            connection = sqlite3.connect(destination / "ocr.sqlite")
            try:
                columns = [row[1] for row in connection.execute("PRAGMA table_info(ocr_fts)")]
                match = connection.execute(
                    "SELECT keyframe_uid FROM ocr_fts WHERE ocr_fts MATCH ?", ('"2026"',)
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(
                columns,
                ["video_id", "keyframe_uid", "detected_text", "language", "confidence"],
            )
            self.assertEqual(match, (1002,))
            self.assertTrue((destination / "coverage.json").is_file())
            self.assertTrue((destination / "SHA256SUMS").is_file())
            validated = validate_ocr_v2_snapshot(
                snapshot_dir=destination,
                catalog_path=catalog,
                catalog_state_path=state,
            )
            self.assertEqual(validated.snapshot_id, manifest.snapshot_id)

    def test_rejects_source_manifest_with_wrong_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog, state, plan, sources = self._fixture(root)
            payload = json.loads(sources.read_text(encoding="utf-8"))
            payload["artifacts"][0]["worker"] = "4"
            sources.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source worker differs"):
                build_ocr_v2_snapshot(
                    catalog_path=catalog,
                    catalog_state_path=state,
                    worker_plan_path=plan,
                    source_manifest_path=sources,
                    source_root=root,
                    output_root=root / "snapshots",
                )

    def test_sync_deduplicates_equivalent_resume_exports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, plan_path, sources_path = self._fixture(root)
            original = OcrV2SourceManifest.model_validate_json(
                sources_path.read_text(encoding="utf-8")
            )
            run_ids = {
                artifact.worker: artifact.run_id for artifact in original.artifacts
            }
            run_ids_path = root / "run-ids.json"
            run_ids_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "artifact_kind": "ocr_v2_production_run_ids",
                        "repository": original.repository,
                        "workers": run_ids,
                    }
                ),
                encoding="utf-8",
            )
            files = {
                path
                for artifact in original.artifacts
                for path in (artifact.result_path, artifact.report_path)
            }
            first = original.artifacts[0]
            first_zip = root.joinpath(*first.result_path.split("/"))
            with zipfile.ZipFile(first_zip) as archive:
                duplicate_members = {
                    name: archive.read(name)
                    for name in archive.namelist()
                    if name != "SHA256SUMS"
                }
            duplicate_report = json.loads(duplicate_members["report.json"])
            duplicate_report["end_to_end_seconds_this_run"] = 0.25
            duplicate_report_bytes = canonical_json(duplicate_report)
            duplicate_members["report.json"] = duplicate_report_bytes
            duplicate_zip = root / "duplicate.zip"
            _zip(duplicate_zip, duplicate_members)
            duplicate_result_sha = sha256_file(duplicate_zip)
            duplicate_report_sha = sha256_bytes(duplicate_report_bytes)
            prefix = first.result_path.rsplit("/", 1)[0]
            duplicate_result = f"{prefix}/results-{duplicate_result_sha}.zip"
            duplicate_report_path = (
                f"{prefix}/reports/summary-{duplicate_report_sha}.json"
            )
            duplicate_result_local = root.joinpath(*duplicate_result.split("/"))
            duplicate_report_local = root.joinpath(*duplicate_report_path.split("/"))
            duplicate_zip.replace(duplicate_result_local)
            duplicate_report_local.parent.mkdir(parents=True, exist_ok=True)
            duplicate_report_local.write_bytes(duplicate_report_bytes)
            files.update({duplicate_result, duplicate_report_path})

            class FakeApi:
                def repo_info(self, **_kwargs):
                    return SimpleNamespace(sha=original.revision)

                def list_repo_tree(self, **_kwargs):
                    return [
                        SimpleNamespace(
                            path=path,
                            last_commit=SimpleNamespace(
                                oid=original.revision,
                                date=(
                                    datetime(2026, 9, 4, 13, tzinfo=UTC)
                                    if path == duplicate_result
                                    else datetime(2026, 9, 4, 12, tzinfo=UTC)
                                ),
                            ),
                        )
                        for path in sorted(files)
                    ]

            def fake_download(**kwargs):
                source = root.joinpath(*kwargs["filename"].split("/"))
                destination = Path(kwargs["local_dir"]).joinpath(
                    *kwargs["filename"].split("/")
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                return str(destination)

            with patch.object(syncer, "HfApi", return_value=FakeApi()), patch.object(
                syncer, "hf_hub_download", side_effect=fake_download
            ):
                destination, synced = syncer.sync(
                    worker_plan_path=plan_path,
                    run_ids_path=run_ids_path,
                    output_root=root / "download",
                    revision=None,
                    token="test-token",
                )
            self.assertTrue(destination.is_file())
            self.assertEqual(synced.revision, original.revision)
            self.assertEqual(len(synced.artifacts), 9)
            selected = next(
                artifact
                for artifact in synced.artifacts
                if artifact.batch_id == first.batch_id
            )
            self.assertEqual(selected.result_sha256, duplicate_result_sha)
            self.assertEqual(selected.equivalent_result_sha256, (first.result_sha256,))
            self.assertEqual(selected.equivalent_report_sha256, (first.report_sha256,))


if __name__ == "__main__":
    unittest.main()
