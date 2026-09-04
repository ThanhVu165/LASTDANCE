"""CPU-only tests. Fake predictors/HF; no GPU inference, API or real uploads."""
import ast
import copy
import csv
import importlib.util
import io
import json
import shutil
import os
import subprocess
import sys
import threading
from contextlib import redirect_stdout
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


r = module("ocr_v2_worker_test", "scripts/kaggle_ocr_v2_production_runtime.py")
builder = module("ocr_v2_builder_test", "scripts/build_kaggle_ocr_v2_production_notebook.py")


class FakeHF:
    repo = "private/test"
    token = "synthetic-test-token"
    def __init__(self, root):
        self.root, self.objects, self.writes, self.fail = Path(root), {}, [], False
    def revision(self):
        return "a" * 40
    def files(self, revision):
        return list(self.objects)
    def download(self, name, revision):
        return self.objects[name]
    def put(self, path, name):
        if self.fail:
            raise RuntimeError("simulated network outage")
        out = self.root / r.sha(name.encode())
        out.write_bytes(Path(path).read_bytes())
        self.objects[name] = out
        self.writes.append(name)
        return self.revision()


def fixture(directory, count=9, region_count=1):
    directory = Path(directory)
    catalog = directory / "frames.csv"
    fields = ["video_id", "local_idx", "frame_id", "pts_time", "shot_id", "window_id", "keyframe_uid"]
    records = [{"video_id": f"L21_V{i:03d}", "local_idx": 0, "frame_id": i * 100,
                "pts_time": i * 3.5, "shot_id": "s000", "window_id": "",
                "keyframe_uid": r.uid_for(f"L21_V{i:03d}", "s000", 0)} for i in range(1, count + 1)]
    with catalog.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    state = {"schema_version": 1, "complete": True, "csv_sha256": r.file_sha(catalog),
             "record_count": count, "video_count": count,
             "sources": [{"video_id": row["video_id"]} for row in records]}
    catalog.with_name(catalog.name + ".state.json").write_bytes(r.encoded(state))
    archives = {}
    for i, rec in enumerate(records, 1):
        box = [0, 0, 20, 0, 20, 10, 0, 10]
        regions = [{"region_id": r.sha(f"{rec['keyframe_uid']}|{n}|{';'.join(f'{v:.2f}' for v in box)}".encode())[:24],
                    "bbox_px": box, "easyocr_text": "Xin chào"} for n in range(region_count)]
        frame = {**rec, "source_image": f"keyframes/{rec['video_id']}/s000_0.jpg", "status": "text_detected",
                 "image_width": 24, "image_height": 16, "error": None,
                 "regions": regions}
        image_dir = directory / "images" / rec["video_id"]
        image_dir.mkdir(parents=True)
        Image.new("RGB", (24, 16), "white").save(image_dir / "s000_0.jpg")
        manifest = {"batch_id": f"batch-{i:02d}", "tier": "easyocr", "complete": True,
                    "catalog_sha256": r.file_sha(catalog), "frames": 1, "expected_frames": 1,
                    "regions": region_count, "status": {"text_detected": 1},
                    "assigned_uid_sha256": r.uid_hash({rec["keyframe_uid"]}),
                    "observed_uid_sha256": r.uid_hash({rec["keyframe_uid"]})}
        path = directory / f"ocr-production-batch-{i:02d}-easyocr.zip"
        r.zip_payload(path, {"easyocr-frames.jsonl": r.encoded(frame) + b"\n",
                             "vintern-candidates.jsonl": b"", "batch-manifest.json": r.encoded(manifest),
                             "run-signature.json": b"{}"})
        archives[f"batch-{i:02d}"] = path
    return catalog, archives


class ProductionTests(unittest.TestCase):
    def launch_helpers(self):
        source = (ROOT / "scripts/kaggle_ocr_v2_production_launch.py").read_text(encoding="utf-8")
        nodes = [n for n in ast.parse(source).body
                 if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))
                 and not (isinstance(n, ast.ImportFrom) and n.module == "kaggle_secrets")]
        namespace = {}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "launch_helpers", "exec"), namespace)
        return namespace

    def test_launcher_reports_heartbeat_while_stage_is_waiting(self):
        helpers = self.launch_helpers()
        seen = threading.Event()
        class Output(io.StringIO):
            def write(self, value):
                if '[WAIT]' in value:
                    seen.set()
                return super().write(value)
        output = Output()
        with redirect_stdout(output):
            with helpers['progress']('input scan', interval=.01):
                self.assertTrue(seen.wait(timeout=2))
        self.assertIn('[START] input scan', output.getvalue())
        self.assertIn('progress/ETA unavailable', output.getvalue())
        self.assertIn('[DONE] input scan', output.getvalue())

    def test_launcher_forwards_both_streams_redacts_secret_and_propagates_failure(self):
        helpers = self.launch_helpers()
        env = {**os.environ, 'HF_TOKEN': 'synthetic-secret', 'PYTHONUNBUFFERED': '1'}
        output = io.StringIO()
        with redirect_stdout(output):
            helpers['run_logged']([sys.executable, '-u', '-c',
                "import os,sys; print('OUT'); print('ERR', file=sys.stderr); print(os.environ['HF_TOKEN'])"],
                'test child', env)
            with self.assertRaises(subprocess.CalledProcessError) as failure:
                helpers['run_logged']([sys.executable, '-u', '-c', 'raise SystemExit(7)'], 'failure', env)
        self.assertEqual(failure.exception.returncode, 7)
        self.assertIn('OUT', output.getvalue())
        self.assertIn('ERR', output.getvalue())
        self.assertIn('[REDACTED]', output.getvalue())
        self.assertNotIn('synthetic-secret', output.getvalue())
        self.assertIn('[STOP] failure', output.getvalue())
        self.assertNotIn('[DONE] failure', output.getvalue())

    def test_plan_exact_nine_disjoint_and_greedy_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog, archives = fixture(tmp)
            hf = FakeHF(tmp)
            for batch, path in archives.items():
                hf.objects[f"ocr/archives/{batch}/easyocr/{path.name}"] = path
            plan = r.create_plan({"output": tmp, "input_root": tmp}, hf)
            r.validate_plan(plan)
            self.assertEqual(plan["catalog_source"], "kaggle_input")
            self.assertEqual(plan["catalog"], "frames.csv")  # No account-specific mount path in plan.
            self.assertEqual(plan["catalog_sha256"], r.file_sha(catalog))
            self.assertEqual(sorted(b for bs in plan["assignments"].values() for b in bs), list(r.BATCH_IDS))
            self.assertEqual(plan["worker_regions"], {"1": 3, "2": 2, "3": 2, "4": 2})
            self.assertFalse(hf.writes)  # Planner reads only; no model calls or uploads.
            plan["assignments"]["2"].append("batch-01")
            with self.assertRaises(ValueError):
                r.validate_plan(plan)

    def test_local_catalog_discovery_duplicates_and_explicit_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "No frames.csv"):
                r.resolve_catalog({"input_root": tmp})
            catalog, _ = fixture(tmp, 1)
            other = Path(tmp) / "other-dataset"
            other.mkdir()
            duplicate = other / "frames.csv"
            state = catalog.with_name(catalog.name + ".state.json")
            duplicate_state = duplicate.with_name(duplicate.name + ".state.json")
            shutil.copyfile(catalog, duplicate)
            shutil.copyfile(state, duplicate_state)
            self.assertEqual(r.resolve_catalog({"input_root": tmp}), catalog)
            duplicate.write_bytes(catalog.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "Multiple different local catalogs"):
                r.resolve_catalog({"input_root": tmp})
            self.assertEqual(r.resolve_catalog({"input_root": tmp, "catalog_path": str(catalog)}), catalog)

    def test_local_catalog_requires_original_state_and_existing_explicit_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog, _ = fixture(tmp, 1)
            with self.assertRaisesRegex(ValueError, "CATALOG_PATH does not exist"):
                r.resolve_catalog({"catalog_path": str(Path(tmp) / "missing.csv")})
            catalog.with_name(catalog.name + ".state.json").unlink()
            with self.assertRaisesRegex(ValueError, "Missing frames.csv.state.json"):
                r.resolve_catalog({"catalog_path": str(catalog)})

    def test_local_catalog_still_must_match_archive_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog, archives = fixture(tmp)
            hf = FakeHF(tmp)
            for batch, path in archives.items():
                hf.objects[f"ocr/archives/{batch}/easyocr/{path.name}"] = path
            catalog.write_bytes(catalog.read_bytes() + b"\n")
            state_path = catalog.with_name(catalog.name + ".state.json")
            state = json.loads(state_path.read_bytes())
            state["csv_sha256"] = r.file_sha(catalog)
            state_path.write_bytes(r.encoded(state))
            with self.assertRaisesRegex(ValueError, "Archive batch/catalog hash drift"):
                r.create_plan({"output": tmp, "catalog_path": str(catalog)}, hf)
            self.assertFalse((Path(tmp) / "ocr-v2-worker-plan.json").exists())
            self.assertFalse(hf.writes)

    def test_archive_binds_real_frame_id_not_local_idx(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog, archives = fixture(tmp, 1)
            manifest, frames = r.load_archive(archives["batch-01"], r.load_catalog(catalog))
            self.assertEqual(frames[0]["frame_id"], 100)
            self.assertNotEqual(frames[0]["frame_id"], frames[0]["local_idx"])
            tasks, images = r.build_tasks(frames, Path(tmp) / "images", "production")
            h, _ = r.helpers(Path(tmp) / "models")
            crops = r.task_crops(tasks, images, h.rectify_region_crop)
            self.assertEqual(crops[0].size, (22, 12))
            crops[0].close()
            Image.new("RGB", (24, 16), "black").save(next(iter(images.values()))["path"])
            with self.assertRaisesRegex(ValueError, "changed"):
                r.task_crops(tasks, images, h.rectify_region_crop)

    def test_archive_checksum_and_catalog_state_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog, archives = fixture(tmp, 1)
            mapping = r.load_catalog(catalog)
            state = catalog.with_name(catalog.name + ".state.json")
            state.write_bytes(b'{}')
            with self.assertRaises(ValueError):
                r.load_catalog(catalog)
            with zipfile.ZipFile(archives["batch-01"], "a") as z:
                z.writestr("unexpected.txt", "tamper")
            with self.assertRaises(ValueError):
                r.load_archive(archives["batch-01"], mapping)

    def test_guard_unicode_repetition_and_nonfinite(self):
        for text in ("a a a a", "xin chào xin chào xin chào xin chào", "a b c a b c a b c a b c"):
            self.assertIn("repeated_phrase", r.guards({"text": text, "confidence": .99}))
        self.assertIn("invalid_confidence", r.guards({"text": "xin", "confidence": float("nan")}))
        self.assertIn("decode_limit", r.guards({"text": "a" * 128, "confidence": .99}))

    def test_paddle_not_global_and_never_cross_model_confidence_ranking(self):
        viet = {"text": "Hà Nội", "confidence": .65}
        self.assertFalse(r.paddle_candidate(viet, "Hà Nội"))
        selected = r.select_result(viet, {"text": "Ha N0i", "confidence": .999}, "Hà Nội")
        self.assertEqual(selected["selected_engine"], "vietocr")
        self.assertEqual(selected["residual_reasons"], ["model_disagreement"])

    def test_numeric_override_requires_valid_structure_and_digit_agreement(self):
        viet = {"text": "06:33:04", "confidence": .7}
        good = {"text": "06:33:04", "confidence": .95}
        self.assertEqual(r.select_result(viet, good, "wrong")["selected_engine"], "paddle")
        bad = {"text": "06:33:49", "confidence": .999}
        self.assertEqual(r.select_result(viet, bad, "wrong")["selected_engine"], "vietocr")
        self.assertFalse(r.numeric("29:99:99"))
        self.assertFalse(r.numeric("06:33:04 rubbish"))

    def test_ascii_requires_failed_viet_and_cache_agreement(self):
        viet = {"text": "???", "confidence": .2}
        paddle = {"text": "KOREA EXCHANGE", "confidence": .96}
        self.assertEqual(r.select_result(viet, paddle, "Korea Exchange")["selected_engine"], "paddle")
        rejected = r.select_result(viet, paddle, "OTHER")
        self.assertIsNone(rejected["selected_text"])
        self.assertTrue(rejected["residual_reasons"])

    def journal_setup(self, tmp, name="local.sqlite", count=4):
        tasks = [{"region_id": f"r{i}", "task_sha256": r.sha(f"task{i}".encode()),
                  "video_id": "L21_V001", "easyocr_text": "tiếng Việt"} for i in range(count)]
        signature = r.sha(b"test signature")
        journal = r.Journal(Path(tmp) / name, signature)
        return tasks, signature, journal

    def run_fake(self, tasks, signature, journal, durable, interrupt=0, predict=None, batch_size=2):
        return r.recognize(tasks, "vietocr", journal, durable,
            predict or (lambda images: [("tiếng Việt", .8) for im in images]),
            lambda group: [Image.new("RGB", (10, 10)) for _ in group],
            signature=signature, batch_size=batch_size, interrupt_after=interrupt)

    def test_remote_resume_in_fresh_database_has_no_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks, signature, journal = self.journal_setup(tmp)
            hf = FakeHF(tmp)
            expected = {t["region_id"]: t["task_sha256"] for t in tasks}
            durable = r.DurableJournal(journal, hf, "prefix", tmp, signature, expected)
            self.assertEqual(durable.restore(), 0)
            with self.assertRaises(r.IntentionalStop):
                self.run_fake(tasks, signature, journal, durable, interrupt=1)
            self.assertEqual(journal.count(), 2)
            journal.close()
            _, _, resumed = self.journal_setup(tmp, "fresh.sqlite")
            remote = r.DurableJournal(resumed, hf, "prefix", tmp, signature, expected)
            self.assertEqual(remote.restore(), 2)
            report = self.run_fake(tasks, signature, resumed, remote)
            self.assertEqual(report["new_predictions"], 2)
            self.assertTrue(report["resume_with_new_work"])
            self.assertEqual(resumed.count(), 4)
            # Replaying remote chunks again does not duplicate, and rerun makes zero calls.
            replay = r.DurableJournal(resumed, hf, "prefix", tmp, signature, expected)
            self.assertEqual(replay.restore(), 4)
            final = self.run_fake(tasks, signature, resumed, replay, predict=lambda _: self.fail("must skip"))
            self.assertEqual(final["new_predictions"], 0)
            resumed.close()

    def test_hf_sync_failure_stops_and_keeps_local_minibatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks, signature, journal = self.journal_setup(tmp)
            hf = FakeHF(tmp)
            hf.fail = True
            durable = r.DurableJournal(journal, hf, "prefix", tmp, signature, {t["region_id"]: t["task_sha256"] for t in tasks})
            with self.assertRaisesRegex(RuntimeError, "outage"):
                self.run_fake(tasks, signature, journal, durable, interrupt=1)
            self.assertEqual(journal.count(), 2)
            self.assertEqual(journal.db.execute("SELECT sum(durable) FROM predictions").fetchone()[0], 0)
            journal.close()

    def test_local_or_remote_signature_foreign_rows_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks, signature, journal = self.journal_setup(tmp)
            durable = r.DurableJournal(journal, FakeHF(tmp), "prefix", tmp, signature, {t["region_id"]: t["task_sha256"] for t in tasks})
            journal.save([{"model": "vietocr", "region_id": "foreign", "signature": signature,
                           "task_sha256": "bad", "text": "wrong", "confidence": .8}])
            with self.assertRaises(ValueError):
                durable.restore()
            journal.close()
            with self.assertRaisesRegex(ValueError, "signature"):
                r.Journal(Path(tmp) / "local.sqlite", "different")

    def test_incomplete_prediction_minibatch_not_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks, signature, journal = self.journal_setup(tmp)
            durable = r.DurableJournal(journal, FakeHF(tmp), "prefix", tmp, signature, {t["region_id"]: t["task_sha256"] for t in tasks})
            with self.assertRaises(ValueError):
                self.run_fake(tasks, signature, journal, durable, predict=lambda ims: [])
            self.assertEqual(journal.count(), 0)
            journal.close()

    def test_oom_halves_batch_without_duplicate_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks, signature, journal = self.journal_setup(tmp)
            durable = r.DurableJournal(journal, FakeHF(tmp), "prefix", tmp, signature, {t["region_id"]: t["task_sha256"] for t in tasks})
            sizes = []
            def predict(images):
                sizes.append(len(images))
                if len(images) > 1:
                    raise RuntimeError("CUDA out of memory")
                return [("tiếng Việt", .8)]
            self.run_fake(tasks, signature, journal, durable, predict=predict, batch_size=4)
            self.assertEqual(sizes, [4, 2, 1, 1, 1, 1])
            self.assertEqual(journal.count(), 4)
            journal.close()

    def test_notebook_compiles_clean_and_embeds_no_gate_b_execution(self):
        notebook = builder.build_notebook()
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "cell", "exec")
                self.assertEqual(cell["outputs"], [])
        namespace = {}
        exec("".join(notebook["cells"][2]["source"]), namespace)
        reference = namespace["SOURCES"]["kaggle_ocr_v2_gate_b_runtime.py"]
        self.assertNotIn("BENCHMARK_REGIONS", reference)
        self.assertNotIn("run_vietocr", reference)
        self.assertIn("rectify_region_crop", reference)
        self.assertIn("RUN_MODE = 'canary'", "".join(notebook["cells"][1]["source"]))
        self.assertEqual(namespace["SOURCES"]["kaggle_ocr_v2_production_runtime.py"],
                         (ROOT / "scripts/kaggle_ocr_v2_production_runtime.py").read_text(encoding="utf-8"))
        self.assertNotIn("CATALOG_HF_PATH", json.dumps(notebook))
        launch = "".join(notebook["cells"][3]["source"])
        self.assertIn("'catalog_path': CATALOG_PATH", launch)
        self.assertIn("'input_root': INPUT_ROOT", launch)

    def test_export_keeps_canonical_content_and_real_provenance_not_ready(self):
        from shared.schemas.ocr import OcrResult
        with tempfile.TemporaryDirectory() as tmp:
            catalog, archives = fixture(tmp, 1)
            _, frames = r.load_archive(archives["batch-01"], r.load_catalog(catalog))
            tasks, images = r.build_tasks(frames, Path(tmp) / "images", "production")
            signature = r.sha(b"signature")
            journal = r.Journal(Path(tmp) / "checkpoint.sqlite", signature)
            journal.save([{"model": "vietocr", "region_id": tasks[0]["region_id"],
                           "task_sha256": tasks[0]["task_sha256"], "signature": signature,
                           "text": "Tiếng Việt rõ", "confidence": .85}])
            journal.close()
            (Path(tmp) / "run-signature.json").write_bytes(b"{}")
            ctx = {"output": tmp, "signature": signature, "batch": "batch-01", "worker": "1",
                   "mode": "production", "run_id": "a" * 64, "prefix": "prefix",
                   "tasks_sha256": "b" * 64, "sample_task_sha256": "c" * 64}
            report = r.export_results(ctx, frames, tasks, FakeHF(tmp), r.time.monotonic())
            self.assertFalse(report["production_ready"])
            self.assertEqual(report["status"], {"success": 1})
            with zipfile.ZipFile(Path(tmp) / "ocr-v2-batch-01-production-results.zip") as archive:
                row = json.loads(archive.read("frame-selections.jsonl"))
                OcrResult.model_validate(row["result"])
                self.assertEqual(row["regions"][0]["selected_engine"], "vietocr")
                self.assertEqual(row["frame_id"], 100)
                for line in archive.read("SHA256SUMS").decode().splitlines():
                    digest, name = line.split(maxsplit=1)
                    self.assertEqual(r.sha(archive.read(name)), digest)

    def test_remote_chain_gap_and_conflict_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks, signature, journal = self.journal_setup(tmp)
            hf = FakeHF(tmp)
            expected = {t["region_id"]: t["task_sha256"] for t in tasks}
            durable = r.DurableJournal(journal, hf, "prefix", tmp, signature, expected)
            self.run_fake(tasks, signature, journal, durable)
            original = hf.writes[0]
            hf.objects[original.replace("part-000000-", "part-000001-")] = hf.objects.pop(original)
            with self.assertRaisesRegex(ValueError, "gap/conflict"):
                r.DurableJournal(journal, hf, "prefix", tmp, signature, expected).restore()
            journal.close()

    def test_paddle_only_recognizes_candidates_and_releases_empty_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks, signature, journal = self.journal_setup(tmp, count=2)
            hf = FakeHF(tmp)
            expected = {t["region_id"]: t["task_sha256"] for t in tasks}
            journal.save([{"model": "vietocr", "region_id": t["region_id"], "task_sha256": t["task_sha256"],
                           "signature": signature, "text": "Tiếng Việt" if i == 0 else "",
                           "confidence": .8 if i == 0 else None} for i, t in enumerate(tasks)])
            durable = r.DurableJournal(journal, hf, "prefix", tmp, signature, expected)
            called = []
            def predict(images):
                called.append(len(images))
                return [("Tiếng Việt", .8) for _ in images]
            report = r.recognize(tasks, "paddle", journal, durable, predict,
                lambda group: [Image.new("RGB", (10, 10)) for _ in group],
                signature=signature, batch_size=128)
            self.assertEqual(called, [1])
            self.assertEqual(report["expected"], 1)
            self.assertIsNone(journal.get("paddle", "r0"))
            journal.close()

    def test_worker_orchestration_canary_stop_resume_and_production_hash_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog, archives = fixture(tmp, region_count=128)
            hf = FakeHF(tmp)
            for batch, path in archives.items():
                hf.objects[f"ocr/archives/{batch}/easyocr/{path.name}"] = path
            plan = r.create_plan({"output": tmp, "catalog_path": str(catalog)}, hf)
            worker_input = Path(tmp) / "worker-mount"
            worker_input.mkdir()
            worker_catalog = worker_input / "frames.csv"
            worker_state = worker_input / "frames.csv.state.json"
            shutil.copyfile(catalog, worker_catalog)
            shutil.copyfile(catalog.with_name(catalog.name + ".state.json"), worker_state)
            config = {"output": tmp, "plan": str(Path(tmp) / "ocr-v2-worker-plan.json"),
                      "input_root": str(worker_input),
                      "keyframes": str(Path(tmp) / "images"), "worker": 1, "mode": "canary", "interrupt_after": 1}
            for path in (worker_catalog, worker_state):
                original = path.read_bytes()
                path.write_bytes(original + b"\n")
                with patch.object(r, "environment") as env:
                    with self.assertRaisesRegex(ValueError, "Worker catalog hash drift"):
                        r.run_worker(config, hf)
                    env.assert_not_called()
                path.write_bytes(original)
            launches, calls = [], []
            def predictor(model, model_dir):
                def predict(images):
                    calls.append((model, len(images)))
                    return [("06:33:04", .95) for _ in images]
                return predict, lambda: None
            def subprocess_run(command, **kwargs):
                self.assertEqual(command[2], "phase")
                model = command[-1]
                launches.append(model)
                try:
                    r.phase(command[3], model)
                except r.IntentionalStop:
                    return types.SimpleNamespace(returncode=75)
                return types.SimpleNamespace(returncode=0)
            with patch.object(r, "environment", return_value={"synthetic": "test-only"}), \
                    patch.object(r, "HF", return_value=hf), \
                    patch.object(r, "model_predictor", side_effect=predictor), \
                    patch.object(r.subprocess, "run", side_effect=subprocess_run):
                r.run_worker(config, hf)
                self.assertEqual(calls, [("vietocr", 64)])
                config["interrupt_after"] = 0
                r.run_worker(config, hf)
                self.assertEqual(calls, [("vietocr", 64), ("vietocr", 64), ("paddle", 128)])
                summaries = [n for n in hf.objects if "/canary/reports/summary-" in n]
                self.assertEqual(len(summaries), 1)
                summary = hf.download(summaries[0], hf.revision())
                self.assertTrue(json.loads(summary.read_bytes())["resume_with_new_work"])
                config["mode"] = "production"
                with self.assertRaisesRegex(ValueError, "canary first"):
                    r.run_worker(config, hf)
                config["approved_canary_sha256"] = r.file_sha(summary)
                r.run_worker(config, hf)
                productions = [n for n in hf.objects if "/production/reports/summary-" in n]
                self.assertEqual(len(productions), len(plan["assignments"]["1"]))
                self.assertEqual(launches[:3], ["vietocr", "vietocr", "paddle"])


if __name__ == "__main__":
    unittest.main()
