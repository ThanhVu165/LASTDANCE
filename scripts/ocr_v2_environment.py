"""Kaggle-only setup, preserving the preloaded Torch/NVIDIA/NumPy/Pillow stack.

Run by the production notebook, never implicitly on import or in local CPU tests.
"""
import importlib.metadata as md
import json
import os
import subprocess
import sys
from pathlib import Path

from kaggle_ocr_v2_production_runtime import atomic, file_sha, heartbeat, log

VIETOCR_WHEEL_SHA256 = "07b3777e5176b0d733cb056b68bd817371605f4b3514795fbf91ad4e181b8ccf"


def protected():
    versions = {}
    for dist in md.distributions():
        name = (dist.metadata.get("Name") or "").lower().replace("_", "-")
        if name in {"torch", "torchvision", "torchaudio", "numpy", "pillow"} or name.startswith("nvidia-"):
            versions[name] = dist.version
    return versions


def setup(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    before = protected()
    if not {"torch", "torchvision", "numpy", "pillow"} <= set(before):
        raise RuntimeError("Use a fresh Kaggle GPU image with Torch/torchvision/NumPy/Pillow")
    pip = [sys.executable, "-m", "pip", "--disable-pip-version-check"]
    def command(args):
        with heartbeat("ENV_INSTALL", package=args[-1]):
            subprocess.run(pip + args, check=True, timeout=1200)
        if protected() != before:
            raise RuntimeError("Protected GPU/image stack changed; stop and use a fresh session")
    log("ENV_1_GPU_RUNTIME", protected=before)
    command(["install", "--no-deps", "--timeout", "120", "--retries", "3",
             "--index-url", "https://www.paddlepaddle.org.cn/packages/stable/cu126/", "paddlepaddle-gpu==3.2.2"])
    log("ENV_2_VIETOCR_WHEEL")
    command(["download", "--no-deps", "--only-binary=:all:", "--dest", str(root), "vietocr==0.3.13"])
    wheels = list(root.glob("vietocr-0.3.13-*.whl"))
    if len(wheels) != 1 or file_sha(wheels[0]) != VIETOCR_WHEEL_SHA256:
        raise ValueError("VietOCR wheel checksum mismatch")
    command(["install", "--no-deps", str(wheels[0])])
    # Resolve the Python dependencies (PaddleX included) without changing GPU/image wheels.
    constraints = root / "protected-constraints.txt"
    pins = {**before, "paddlepaddle-gpu": "3.2.2", "vietocr": "0.3.13"}
    atomic(constraints, "".join(f"{name}=={version}\n" for name, version in sorted(pins.items())).encode())
    requests = ["paddleocr==3.7.0", "einops==0.8.1", "gdown==5.2.0", "PyYAML==6.0.2", "huggingface_hub"]
    report = root / "dependency-resolution.json"
    log("ENV_3_RESOLVE_PYTHON_DEPENDENCIES")
    command(["install", "--dry-run", "--report", str(report), "--constraint", str(constraints)] + requests)
    install = json.loads(report.read_bytes())["install"]
    resolved = []
    for row in install:
        metadata = row["metadata"]
        name = metadata["name"].lower().replace("_", "-")
        if name in pins or name.startswith("nvidia-") or name in {"torch", "torchvision", "torchaudio", "paddlepaddle"}:
            raise RuntimeError(f"Resolver wants to replace/add protected runtime {name}; no installation performed")
        resolved.append(f"{metadata['name']}=={metadata['version']}")
    if resolved:
        command(["install", "--constraint", str(constraints)] + resolved)
    log("ENV_4_SEPARATE_GPU_PROBES")
    probes = [
        "import torch; from vietocr.tool.predictor import Predictor; assert torch.cuda.is_available(); "
        "assert 'T4' in torch.cuda.get_device_name(0).upper(); print('VIETOCR_T4_IMPORT_OK')",
        "import paddle; from paddleocr import TextRecognition; assert paddle.is_compiled_with_cuda(); "
        "paddle.device.set_device('gpu:0'); assert 'T4' in paddle.device.cuda.get_device_name(0).upper(); "
        "print('PADDLE_T4_IMPORT_OK')",
    ]
    for probe in probes:
        with heartbeat("GPU_IMPORT_PROBE"):
            result = subprocess.run([sys.executable, "-c", probe], text=True, capture_output=True, timeout=180)
        if result.returncode:
            raise RuntimeError("GPU probe failed; inference not started:\n" + result.stderr)
        print(result.stdout, flush=True)
    log("ENV_READY", protected_unchanged=protected() == before)


if __name__ == "__main__":
    setup(sys.argv[1])
