"""Regenerate the self-contained Kaggle OCR-v2 Gate B notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "scripts" / "kaggle_ocr_v2_gate_b_runtime.py"
OUTPUT = ROOT / "notebooks" / "kaggle_ocr_v2_gate_b.ipynb"


def lines(value: str) -> list[str]:
    return value.splitlines(keepends=True)


markdown = """# OCR v2 — Gate B recognizer A/B on one Kaggle T4

Select a **T4 GPU**, enable Internet, then attach:

1. `ocr-v2-review-bundle.zip` from Gate A preparation;
2. exactly one `ocr-production-batch-01-easyocr.zip`;
3. JPEG folders for every video listed in the review bundle.

For a final full-catalog ETA, also attach the other eight immutable EasyOCR archives. Do not
attach a second copy of Batch 01. Archives are only opened for their signed manifests; model inference still uses the 120 labelled-sample crops
and a deterministic 5,000-crop Batch 01 speed canary.

The notebook compares cached EasyOCR with `latin_PP-OCRv5_mobile_rec` and VietOCR
`vgg_seq2seq`. It does not call Vintern, Gemini, or any paid API and does not write
`ocr.sqlite`. Download `/kaggle/working/ocr-v2-gate-b-results.zip` when complete.
"""

parameters = """# Reduce a batch size only if this exact T4 session reports OOM.
PADDLE_BATCH_SIZE = 128
VIETOCR_BATCH_SIZE = 64
"""

setup = r"""import hashlib, importlib.metadata as md, importlib.util, os, subprocess, sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def version_or_missing(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None

protected = ['torch', 'torchvision']
protected_before = {name: version_or_missing(name) for name in protected}

# Paddle 3.2.2 CUDA 12.6 wheel and the exact high-level OCR API used by this gate.
print('[ENV 1/4] Installing Paddle GPU runtime...', flush=True)
subprocess.check_call([
    # Kaggle's preloaded Torch and NVIDIA/NCCL wheels are a matched set.  Do
    # not let pip resolve Paddle's optional CUDA dependencies: doing so can
    # replace NCCL and makes libtorch_cuda fail to import (ncclCommShrink).
    sys.executable, '-m', 'pip', 'install', '--quiet', '--no-cache-dir', '--no-deps',
    'paddlepaddle-gpu==3.2.2',
    '-i', 'https://www.paddlepaddle.org.cn/packages/stable/cu126/'
])
print('[ENV 2/4] Installing PaddleOCR API...', flush=True)
subprocess.check_call([
    sys.executable, '-m', 'pip', 'install', '--quiet', '--no-cache-dir', '--no-deps',
    'paddleocr==3.7.0'
])

# Download first so the official VietOCR wheel itself is checksum-pinned.
wheel_dir = Path('/kaggle/working/ocr-v2-wheel')
wheel_dir.mkdir(parents=True, exist_ok=True)
print('[ENV 3/4] Downloading and verifying VietOCR wheel...', flush=True)
subprocess.check_call([
    sys.executable, '-m', 'pip', 'download', '--quiet', '--no-deps',
    '--dest', str(wheel_dir), 'vietocr==0.3.13'
])
wheels = list(wheel_dir.glob('vietocr-0.3.13-*.whl'))
assert len(wheels) == 1, wheels
wheel_sha = hashlib.sha256(wheels[0].read_bytes()).hexdigest()
assert wheel_sha == '07b3777e5176b0d733cb056b68bd817371605f4b3514795fbf91ad4e181b8ccf', wheel_sha
subprocess.check_call([
    sys.executable, '-m', 'pip', 'install', '--quiet', '--no-cache-dir', '--no-deps',
    str(wheels[0]), 'einops==0.8.1'
])

protected_after = {name: version_or_missing(name) for name in protected}
assert protected_before == protected_after, (
    'Dependency installation changed Kaggle Torch; start a fresh session.',
    protected_before,
    protected_after,
)
missing = [name for name in ['paddle', 'paddleocr', 'vietocr', 'yaml', 'PIL', 'numpy'] if importlib.util.find_spec(name) is None]
assert not missing, ('Missing Gate B imports', missing)

probe = subprocess.run(
    [sys.executable, '-c',
     'import paddle,torch; assert paddle.is_compiled_with_cuda(); '
     'print(paddle.__version__, torch.__version__, torch.cuda.get_device_name(0))'],
    text=True, capture_output=True,
)
assert probe.returncode == 0, 'CUDA import probe failed; start a fresh Kaggle session.\n' + probe.stderr
print('[ENV 4/4] CUDA dependency probe passed.', flush=True)
print('OCR_V2_ENV_READY', probe.stdout.strip(), {
    'paddleocr': version_or_missing('paddleocr'),
    'vietocr': version_or_missing('vietocr'),
})
"""

notebook = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": lines(markdown)},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines(parameters)},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines(setup)},
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines(RUNTIME.read_text(encoding="utf-8")),
        },
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
