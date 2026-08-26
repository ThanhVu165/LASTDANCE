"""Run the mandatory EVA-CLIP dev-subset interrupt/resume/validate gate on Kaggle."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from offline.visual_models import DEFAULT_MODEL_CONFIG, load_model_config


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=_REPOSITORY_ROOT,
        check=False,
        text=True,
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return payload


def _assert_final_manifest(
    manifest: dict[str, Any],
    *,
    model_config: dict[str, Any],
    expected_record_count: int,
    expected_gpu_name: str,
) -> None:
    expected = {
        "modality": "eva_clip",
        "record_count": expected_record_count,
        "vector_dim": int(model_config["expected_vector_dim"]),
        "vector_dtype": "float16",
        "checkpoint_resume_verified": True,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise RuntimeError(
                f"EVA-CLIP dev manifest mismatch for {field}: "
                f"expected={value!r}, actual={manifest.get(field)!r}"
            )
    model = manifest.get("model")
    runtime = manifest.get("runtime")
    if not isinstance(model, dict) or (
        model.get("id") != model_config["model_id"]
        or model.get("revision") != model_config["revision"]
    ):
        raise RuntimeError("EVA-CLIP dev manifest model provenance mismatch")
    if not isinstance(runtime, dict) or (
        runtime.get("device") != "cuda"
        or runtime.get("gpu_name") != expected_gpu_name
        or runtime.get("open_clip_torch") != "3.3.0"
        or runtime.get("timm") != "1.0.28"
    ):
        raise RuntimeError("EVA-CLIP dev manifest CUDA/OpenCLIP provenance mismatch")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--keyframes-root", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--video-id-file", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--batch-id", default="dev-subset-5")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--expected-record-count", type=int, default=4164)
    parser.add_argument("--expected-gpu-name", default="Tesla T4")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size <= 0 or args.expected_record_count <= 0:
        raise ValueError("batch size and expected record count must be positive")
    paths = {
        "catalog": args.catalog.resolve(),
        "keyframes_root": args.keyframes_root.resolve(),
        "video_id_file": args.video_id_file.resolve(),
        "model_config": args.model_config.resolve(),
        "embedding_root": args.embedding_root.resolve(),
    }
    for name in ("catalog", "keyframes_root", "video_id_file", "model_config"):
        if not paths[name].exists():
            raise FileNotFoundError(f"missing {name}: {paths[name]}")

    artifact = paths["embedding_root"] / args.batch_id / "eva_clip"
    if artifact.exists():
        raise RuntimeError(
            f"refusing to overwrite an existing EVA-CLIP dev artifact: {artifact}; "
            "use a new --batch-id after inspecting the existing checkpoint"
        )

    model_config = load_model_config("eva_clip", path=paths["model_config"])
    verifier = _run(
        [
            sys.executable,
            "-m",
            "scripts.verify_visual_model_revisions",
            "--model-config",
            str(paths["model_config"]),
        ]
    )
    if verifier.returncode != 0:
        raise RuntimeError(f"visual model revision verifier failed: {verifier.returncode}")

    base = [
        sys.executable,
        "-m",
        "scripts.build_visual_embeddings",
        "--modality",
        "eva_clip",
        "--batch-id",
        args.batch_id,
        "--catalog",
        str(paths["catalog"]),
        "--keyframes-root",
        str(paths["keyframes_root"]),
        "--embedding-root",
        str(paths["embedding_root"]),
        "--video-id-file",
        str(paths["video_id_file"]),
        "--model-config",
        str(paths["model_config"]),
        "--batch-size",
        str(args.batch_size),
    ]
    stopped = _run(base + ["--stop-after-shards", "2"])
    if stopped.returncode != 75:
        raise RuntimeError(
            f"EVA-CLIP intentional interruption returned {stopped.returncode}, expected 75"
        )

    checkpoint_path = artifact / "checkpoint.json"
    checkpoint = _read_json(checkpoint_path)
    if (
        checkpoint.get("complete") is not False
        or checkpoint.get("next_index") != args.batch_size * 2
        or checkpoint.get("completed_shards") != 2
        or (artifact / "manifest.json").exists()
    ):
        raise RuntimeError("EVA-CLIP intentional interruption checkpoint is invalid")

    completed = _run(base)
    if completed.returncode != 0:
        raise RuntimeError(f"EVA-CLIP resume returned {completed.returncode}")
    validated = _run(
        [
            sys.executable,
            "-m",
            "scripts.validate_visual_embeddings",
            "--artifact-dir",
            str(artifact),
            "--catalog",
            str(paths["catalog"]),
            "--keyframes-root",
            str(paths["keyframes_root"]),
            "--require-resume-verified",
        ]
    )
    if validated.returncode != 0:
        raise RuntimeError(f"EVA-CLIP validator returned {validated.returncode}")

    manifest = _read_json(artifact / "manifest.json")
    _assert_final_manifest(
        manifest,
        model_config=model_config,
        expected_record_count=args.expected_record_count,
        expected_gpu_name=args.expected_gpu_name,
    )
    print("EVA_CLIP_DEV_GATE_PASS")
    print(
        json.dumps(
            {
                "artifact": str(artifact),
                "model": manifest["model"],
                "record_count": manifest["record_count"],
                "vector_dim": manifest["vector_dim"],
                "vector_dtype": manifest["vector_dtype"],
                "checkpoint_resume_verified": manifest[
                    "checkpoint_resume_verified"
                ],
                "runtime": manifest["runtime"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
