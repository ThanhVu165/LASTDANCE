"""CPU-only contract tests; synthetic predictions, never download/run an OCR model."""
import importlib.util
import io
import json
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sharpen", ROOT / "scripts/kaggle_ocr_v2_sharpen_runtime.py")
trial = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trial)


class FakePredictor:
    def __init__(self, incomplete=False):
        self.calls = 0
        self.incomplete = incomplete

    def predict_batch(self, images, return_prob):
        self.calls += len(images)
        size = len(images) - int(self.incomplete)
        return ["Tiếng Việt rõ" for _ in range(size)], [0.8] * size


class SharpenTrialTests(unittest.TestCase):
    def test_fresh_environment_installs_only_missing_packages_without_cuda_resolution(self):
        versions = {name: "test" for name in ("torch", "torchvision", "Pillow", "numpy", "PyYAML")}
        calls = []
        wheel_bytes = b"mock pinned wheel"
        def version(name):
            if name not in versions:
                raise trial.md.PackageNotFoundError(name)
            return versions[name]
        def run(command, **kwargs):
            calls.append(command)
            if "download" in command:
                cache = Path(command[command.index("--dest") + 1])
                (cache / "vietocr-0.3.13-py3-none-any.whl").write_bytes(wheel_bytes)
            elif "install" in command:
                requirement = command[-1]
                if requirement.endswith(".whl"):
                    versions["vietocr"] = "0.3.13"
                else:
                    name, value = requirement.split("==")
                    versions[name] = value
            return types.SimpleNamespace(returncode=0, stderr="", stdout="VIETOCR_IMPORT_OK")
        with tempfile.TemporaryDirectory() as directory, patch.object(trial.md, "version", side_effect=version), \
                patch.object(trial.md, "distributions", return_value=[]), \
                patch.object(trial.subprocess, "run", side_effect=run), \
                patch.object(trial, "VIETOCR_WHEEL_SHA256", trial.digest(wheel_bytes)):
            result = trial.ensure_vietocr_packages(Path(directory))
        self.assertEqual(result["vietocr"], "0.3.13")
        installs = [c for c in calls if "install" in c]
        self.assertEqual(len(installs), 3)
        self.assertTrue(all("--no-deps" in c for c in installs))
        self.assertFalse(any("torch" in c[-1] or "paddle" in c[-1] for c in installs))

    def test_environment_rejects_unverified_wheel_before_install(self):
        def version(name):
            if name == "vietocr":
                raise trial.md.PackageNotFoundError(name)
            return "test"
        with tempfile.TemporaryDirectory() as directory, patch.object(trial.md, "version", side_effect=version), \
                patch.object(trial.md, "distributions", return_value=[]), patch.object(trial.subprocess, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                trial.ensure_vietocr_packages(Path(directory))
            self.assertEqual(run.call_count, 1)
            self.assertIn("download", run.call_args.args[0])

    def test_reference_extract_does_not_run_gate_b(self):
        ref, source = trial.helpers(Path("unused"))
        self.assertNotIn("run_vietocr", source)
        self.assertNotIn("run_paddle", source)
        self.assertNotIn("import torch", source)
        self.assertEqual(ref.CROP_SPEC_ID, "pil_quad_v1_pad08_edge")
        image = Image.fromarray(np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3))
        crop = ref.rectify_region_crop(image, "[5, 6, 45, 6, 45, 26, 5, 26]")
        self.assertEqual(crop.size, (44, 24))

    def test_selection_balanced_disjoint_deterministic(self):
        samples, viet = [], {}
        for video in trial.VIDEOS:
            for i in range(10):
                rid = f"{video}-{i:02d}"
                samples.append({"region_id": rid, "video_id": video})
                viet[rid] = {"text": str(i), "confidence": i / 10}
        selected = trial.select_samples(list(reversed(samples)), viet)
        self.assertEqual(len(selected), 30)
        self.assertEqual(len({r["region_id"] for r in selected}), 30)
        for video in trial.VIDEOS:
            rows = [r for r in selected if r["video_id"] == video]
            self.assertEqual([r["region_id"][-2:] for r in rows], ["00", "01", "02", "03", "09", "08"])
        self.assertEqual(sum(r["selection_group"] == "low" for r in selected), 20)

    def test_legacy_crop_dimensions_do_not_reject_gate_b_crop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for video in trial.VIDEOS:
                (root / video).mkdir()
            Image.new("RGB", (900, 600), "white").save(root / "L21_V001/s000161_485.jpg")
            sample = {"region_id": "a0974a8564c659d37bce561d", "video_id": "L21_V001",
                      "source_image": "L21_V001/s000161_485.jpg", "keyframe_uid": 3751546584916706228,
                      "sample_row_sha256": "test", "crop_width": 34, "crop_height": 36,
                      "bbox_px": "[724,248,750,248,750,276,724,276]"}
            ref, _ = trial.helpers(root)
            tasks = trial.build_tasks([sample], root, ref, root / "output")
            self.assertEqual(len(tasks), 3)
            self.assertEqual(tasks[0]["archive_crop_size"], [34, 36])
            self.assertEqual(tasks[0]["gate_b_crop_size"], [30, 32])
            with Image.open(root / "output" / tasks[0]["image_file"]) as crop:
                self.assertEqual(crop.size, (30, 32))

    def test_null_nonfinite_confidence(self):
        for value in (None, float("nan"), float("inf"), -0.1, 2):
            self.assertIsNone(trial.confidence(value))
        self.assertEqual(trial.confidence(0.9), 0.9)

    def test_three_variants_original_unchanged(self):
        crop = Image.fromarray(np.random.default_rng(42).integers(0, 255, (20, 40, 3), dtype=np.uint8))
        original = crop.tobytes()
        self.assertEqual(trial.make_variant(crop, "original").tobytes(), original)
        enlarged = trial.make_variant(crop, "bicubic_2x")
        sharp = trial.make_variant(crop, "bicubic_2x_unsharp")
        self.assertEqual(enlarged.size, (80, 40))
        self.assertEqual(sharp.size, enlarged.size)
        self.assertNotEqual(sharp.tobytes(), enlarged.tobytes())
        self.assertEqual(crop.tobytes(), original)

    def fixture(self, root):
        tasks = []
        for index in range(3):
            for variant in trial.VARIANTS:
                name = f"{index}-{variant}.png"
                Image.new("RGB", (20, 10), "white").save(root / name)
                tasks.append({"region_id": str(index), "variant": variant, "image_file": name})
        return tasks

    def test_interrupt_resume_exact_once_and_full_rerun_zero_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = self.fixture(root)
            expected = {(t["region_id"], t["variant"]): t for t in tasks}
            rows, predictor = [], FakePredictor()
            with self.assertRaisesRegex(RuntimeError, "INTENTIONAL_INTERRUPT"):
                trial.recognize(tasks, rows, predictor, root, "signature", batch_size=2, interrupt_after=2)
            self.assertEqual(predictor.calls, 2)
            self.assertTrue((root / "ocr-v2-sharpen-checkpoint.zip").exists())
            rows = trial.load_checkpoint(root, "signature", expected)
            resumed = FakePredictor()
            trial.recognize(tasks, rows, resumed, root, "signature", batch_size=2)
            self.assertEqual(resumed.calls, 7)
            self.assertEqual(len(rows), 9)
            trial.recognize(tasks, rows, resumed, root, "signature", batch_size=2)
            self.assertEqual(resumed.calls, 7)
            with self.assertRaisesRegex(ValueError, "signature"):
                trial.load_checkpoint(root, "changed", expected)
            with tempfile.TemporaryDirectory() as other:
                restored = trial.load_checkpoint(Path(other), "signature", expected, root / "ocr-v2-sharpen-checkpoint.zip")
                self.assertEqual(restored, rows)

    def test_incomplete_minibatch_not_committed_and_budget_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks, rows = self.fixture(root), []
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                trial.recognize(tasks, rows, FakePredictor(True), root, "sig", batch_size=2)
            self.assertEqual(rows, [])
            self.assertFalse((root / "checkpoint.json").exists())
            predictor = FakePredictor()
            with self.assertRaises(trial.TrialTimeLimit):
                trial.recognize(tasks, rows, predictor, root, "sig", budget=0)
            self.assertEqual(predictor.calls, 0)

    def test_remote_checkpoint_failure_stops_but_keeps_local_minibatch(self):
        class FailedRemote:
            def upload(self, *args):
                raise RuntimeError("HF unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks, rows = self.fixture(root), []
            predictor = FakePredictor()
            with self.assertRaisesRegex(RuntimeError, "HF unavailable"):
                trial.recognize(tasks, rows, predictor, root, "sig", batch_size=2, durable=FailedRemote())
            self.assertEqual(predictor.calls, 2)
            expected = {(t["region_id"], t["variant"]): t for t in tasks}
            self.assertEqual(len(trial.load_checkpoint(root, "sig", expected)), 2)

    def test_hf_upload_roundtrip_and_fresh_session_restore_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            saved = {}
            class API:
                def __init__(self, token=None):
                    pass
                def repo_info(self, **kwargs):
                    return types.SimpleNamespace(private=True, sha="fake-revision")
                def list_repo_files(self, **kwargs):
                    return list(saved)
                def upload_file(self, path_or_fileobj, path_in_repo, **kwargs):
                    saved[path_in_repo] = Path(path_or_fileobj).read_bytes()
                    return types.SimpleNamespace(oid="fake-revision")
            def download(filename, **kwargs):
                path = root / "fake-download.zip"
                path.write_bytes(saved[filename])
                return str(path)
            fake = types.SimpleNamespace(HfApi=API, hf_hub_download=download)
            with patch.dict("sys.modules", {"huggingface_hub": fake}), patch.dict("os.environ", {"HF_TOKEN": "test-only"}):
                store = trial.HFCheckpointStore("test/ocr", "sig")
                tasks = self.fixture(root)
                rows = [{**t, "text": "test", "confidence": 0.8, "status": "success"} for t in tasks[:2]]
                expected = {(t["region_id"], t["variant"]): t for t in tasks}
                trial.save_checkpoint(root, "sig", rows)
                store.upload(root / "ocr-v2-sharpen-checkpoint.zip", "checkpoint-000002")
                trial.save_checkpoint(root, "sig", rows)
                store.upload(root / "ocr-v2-sharpen-checkpoint.zip", "checkpoint-000002")
                self.assertEqual(len(saved), 1)
                with zipfile.ZipFile(root / "ocr-v2-sharpen-checkpoint.zip") as archive:
                    self.assertEqual(archive.getinfo("checkpoint.json").date_time, (1980, 1, 1, 0, 0, 0))
                fresh = root / "fresh-session"
                fresh.mkdir()
                store.restore(fresh, "sig", expected)
                self.assertEqual(trial.load_checkpoint(fresh, "sig", expected), rows)

    def test_render_full_length_sheet_and_partial_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "trial"
            root.mkdir()
            tasks = self.fixture(root)
            selected = [{"region_id": str(i), "video_id": "L21_V001", "selection_group": "low",
                         "gate_b_vietocr_text": "Tiếng Việt"} for i in range(3)]
            rows = [{**t, "text": "Dấu tiếng Việt " * 150, "confidence": 0.8, "status": "success"} for t in tasks]
            trial.save_checkpoint(root, "sig", rows)
            trial.atomic_write(root / "run-signature.json", b"{}")
            report = trial.package_results(root, selected, tasks, rows, "sig", "simulated partial", 0.1)
            self.assertEqual(report["decision"], "INCOMPLETE")
            self.assertFalse(report["production_ready"])
            with Image.open(root / "sheets/compare-01.png") as sheet:
                self.assertGreater(sheet.height, 1000)
            self.assertTrue((root.parent / "ocr-v2-sharpen-results.zip").exists())

    def test_checkpoint_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trial.save_checkpoint(root, "sig", [])
            state = json.loads((root / "checkpoint.json").read_bytes())
            state["rows"] = [{"bad": "changed"}]
            (root / "checkpoint.json").write_bytes(trial.encoded(state))
            with self.assertRaisesRegex(ValueError, "checksum"):
                trial.load_checkpoint(root, "sig", {})

    def test_zip_and_extracted_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "datasets" / "person" / "review"
            nested.mkdir(parents=True)
            (nested / "review-report.json").touch()
            (nested / "recognition-sample.jsonl").touch()
            self.assertEqual(trial.discover(root, "", "review.zip", {"review-report.json", "recognition-sample.jsonl"}), nested)
            with self.assertRaises(FileNotFoundError):
                trial.discover(root, str(root / "missing"), "review.zip", set())

    def test_long_text_wrapping_preserves_every_character(self):
        font = trial.review_font(20)
        text = "Tiếng Việt " + "rất dài " * 150 + "0123456789" * 20
        self.assertEqual("".join(trial.wrapped(text, font, 200)), text)

    def test_font_falls_back_to_matplotlib_without_system_fonts(self):
        expected = "bundled/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf"
        font = object()
        def truetype(name, size):
            if name == expected:
                return font
            raise OSError("No system fonts")
        dist = types.SimpleNamespace(locate_file=lambda name: "bundled/" + name)
        with patch.object(trial.md, "distribution", return_value=dist), \
                patch.object(trial.ImageFont, "truetype", side_effect=truetype):
            self.assertIs(trial.review_font(20), font)

    def test_generated_notebook_compiles_and_matches_runtime(self):
        path = ROOT / "notebooks/kaggle_ocr_v2_sharpen.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = []
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                compile(source, f"cell-{index}", "exec")
                code.append(source)
        self.assertIn((ROOT / "scripts/kaggle_ocr_v2_sharpen_runtime.py").read_text(encoding="utf-8"), code)
        self.assertIn("ensure_vietocr_packages", "".join(code))


if __name__ == "__main__":
    unittest.main()
