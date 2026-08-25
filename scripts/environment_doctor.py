"""Validate a LASTDANCE environment without modifying it."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from offline.config import get_data_root
from offline.preprocessing.shot_detection import resolve_and_verify_transnetv2_weights


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    ok: bool
    required: bool
    detail: str


_BASE_PACKAGES = {
    "pydantic": ("pydantic", "2.10.3"),
    "numpy": ("numpy", "1.26.4"),
    "pandas": ("pandas", "2.2.3"),
    "Pillow": ("PIL", "11.0.0"),
    "opencv-python-headless": ("cv2", "4.10.0.84"),
    "ImageHash": ("imagehash", "4.3.1"),
}

_PROFILE_PYTHON_MINORS = {
    "dev": (3, 11),
    "offline-local": (3, 11),
    "shot-colab-gpu": (3, 11),
    "shot-windows-gpu": (3, 11),
    "kaggle-gpu": (3, 12),
}

_PROFILE_PACKAGES = {
    "dev": _BASE_PACKAGES,
    "offline-local": {
        **_BASE_PACKAGES,
        "faiss-cpu": ("faiss", "1.9.0"),
        "torch": ("torch", "2.12.1"),
        "transnetv2-pytorch": ("transnetv2_pytorch", "1.0.5"),
        "ffmpeg-python": ("ffmpeg", "0.2.0"),
    },
    "shot-colab-gpu": {
        **_BASE_PACKAGES,
        "torch": ("torch", None),
        "transnetv2-pytorch": ("transnetv2_pytorch", "1.0.5"),
        "ffmpeg-python": ("ffmpeg", "0.2.0"),
    },
    "shot-windows-gpu": {
        **_BASE_PACKAGES,
        "torch": ("torch", "2.12.1+cu126"),
        "transnetv2-pytorch": ("transnetv2_pytorch", "1.0.5"),
        "ffmpeg-python": ("ffmpeg", "0.2.0"),
    },
    "kaggle-gpu": {
        **_BASE_PACKAGES,
        "torch": ("torch", None),
        "transformers": ("transformers", "5.15.1"),
        "accelerate": ("accelerate", "1.14.0"),
        "sentence-transformers": ("sentence_transformers", "6.0.0"),
        "huggingface-hub": ("huggingface_hub", "1.28.0"),
        "safetensors": ("safetensors", "0.8.0"),
        "sentencepiece": ("sentencepiece", "0.2.1"),
    },
}


def check_python(
    version_info: tuple[int, ...] | None = None,
    *,
    required_minor: tuple[int, int] = (3, 11),
) -> CheckResult:
    version = version_info or tuple(sys.version_info[:3])
    ok = version[:2] == required_minor
    return CheckResult(
        "python",
        ok,
        True,
        (
            f"{version[0]}.{version[1]}.{version[2]} "
            f"(required: {required_minor[0]}.{required_minor[1]}.x)"
        ),
    )


def check_package(distribution: str, module: str, expected: str | None) -> CheckResult:
    try:
        installed = importlib.metadata.version(distribution)
        importlib.import_module(module)
    except Exception as exc:
        return CheckResult(distribution, False, True, f"unavailable: {exc}")
    ok = expected is None or installed == expected
    expectation = expected if expected is not None else "CUDA-compatible environment version"
    return CheckResult(
        distribution,
        ok,
        True,
        f"{installed} (required: {expectation})",
    )


def _resolve_executable(configured: str) -> str | None:
    candidate = Path(configured).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    return shutil.which(configured)


def check_executable(name: str, configured: str) -> CheckResult:
    resolved = _resolve_executable(configured)
    if resolved is None:
        return CheckResult(name, False, True, f"not found: {configured}")
    try:
        result = subprocess.run(
            [resolved, "-version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CheckResult(name, False, True, f"failed to execute: {exc}")
    first_line = (result.stdout or result.stderr or "").splitlines()
    detail = first_line[0] if first_line else f"exit code {result.returncode}"
    return CheckResult(name, result.returncode == 0, True, detail)


def check_data_layout(environment: Mapping[str, str]) -> CheckResult:
    root = get_data_root(environment)
    missing = [name for name in ("videos",) if not (root / name).is_dir()]
    if missing:
        return CheckResult(
            "AIC_DATA",
            False,
            True,
            f"{root} missing directories: {', '.join(missing)}",
        )
    return CheckResult("AIC_DATA", True, True, str(root))


def check_transnet_weights(environment: Mapping[str, str]) -> list[CheckResult]:
    configured = environment.get("AIC_TRANSNETV2_WEIGHTS", "").strip()
    expected = environment.get("AIC_TRANSNETV2_WEIGHTS_SHA256", "").strip()
    if configured and (not expected or expected == "replace_with_verified_sha256"):
        return [
            CheckResult(
                "transnetv2-weights",
                False,
                True,
                "AIC_TRANSNETV2_WEIGHTS_SHA256 is not configured",
            )
        ]
    try:
        path, actual, source = resolve_and_verify_transnetv2_weights(
            Path(configured) if configured else None,
            expected or None,
        )
    except (ImportError, ValueError, RuntimeError, OSError) as exc:
        return [CheckResult("transnetv2-weights", False, True, str(exc))]
    return [
        CheckResult(
            "transnetv2-weights",
            True,
            True,
            f"{source}: {path} sha256={actual}",
        )
    ]


def collect_checks(
    profile: str,
    environment: Mapping[str, str] | None = None,
    *,
    check_data: bool = True,
) -> list[CheckResult]:
    if profile not in _PROFILE_PACKAGES:
        raise ValueError(f"unknown profile: {profile}")
    values = os.environ if environment is None else environment
    checks = [check_python(required_minor=_PROFILE_PYTHON_MINORS[profile])]
    checks.extend(
        check_package(distribution, module, expected)
        for distribution, (module, expected) in _PROFILE_PACKAGES[profile].items()
    )
    if profile in {"offline-local", "shot-colab-gpu", "shot-windows-gpu"}:
        if check_data:
            checks.append(check_data_layout(values))
        checks.append(check_executable("ffmpeg", values.get("AIC_FFMPEG", "ffmpeg")))
        checks.append(check_executable("ffprobe", values.get("AIC_FFPROBE", "ffprobe")))
        checks.extend(check_transnet_weights(values))
    if profile in {"kaggle-gpu", "shot-colab-gpu", "shot-windows-gpu"}:
        try:
            import torch

            cuda_ok = bool(torch.cuda.is_available())
            detail = torch.cuda.get_device_name(0) if cuda_ok else "CUDA unavailable"
        except Exception as exc:
            cuda_ok = False
            detail = str(exc)
        checks.append(CheckResult("cuda", cuda_ok, True, detail))
    return checks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=sorted(_PROFILE_PACKAGES),
        default="dev",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="validate the toolchain without requiring AIC_DATA/videos",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    checks = collect_checks(args.profile, check_data=not args.skip_data)
    if args.json:
        print(json.dumps([asdict(check) for check in checks], ensure_ascii=False, indent=2))
    else:
        for check in checks:
            label = "PASS" if check.ok else "FAIL"
            print(f"[{label}] {check.name}: {check.detail}")
    return 0 if all(check.ok or not check.required for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
