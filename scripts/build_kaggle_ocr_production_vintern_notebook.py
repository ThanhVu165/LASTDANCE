"""Regenerate the self-contained Kaggle OCR production Vintern notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "scripts" / "kaggle_ocr_production_vintern_runtime.py"
OUTPUT = ROOT / "scripts" / "kaggle_ocr_production_vintern.ipynb"


def lines(value: str) -> list[str]:
    return value.splitlines(keepends=True)


parameters = """# Use the same slot number as phase 1 on each Kaggle account.
WORKER_SLOT = 1

# Read completed EasyOCR archives and publish completed Vintern archives to this private Dataset.
PUBLISH_TO_HF = True
HF_REPO_ID = 'MinhThuw0103/lastdance-visual-embeddings'

# Normal production: 0. Set 50 only once for the required interrupt/resume proof.
INTERRUPT_AFTER_NEW_CANDIDATES = 0
CHECKPOINT_EVERY = 10000
PROGRESS_EVERY = 100
"""

setup = """# Preserve Kaggle CUDA/Torch/NumPy ABI and install only the pinned pure-Python runtime layer.
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import importlib.metadata as md, importlib.util, subprocess, sys

def version_or_missing(name):
    try: return md.version(name)
    except md.PackageNotFoundError: return None

protected_names = ['torch','torchvision','numpy','opencv-python','opencv-python-headless','scikit-image','pandas']
protected_before = {name: version_or_missing(name) for name in protected_names}
packages = ['transformers==4.43.4','tokenizers==0.19.1','huggingface-hub==0.24.7','sentencepiece==0.2.0']
subprocess.check_call([sys.executable,'-m','pip','install','--quiet','--no-cache-dir','--no-deps',*packages])
missing = [name for name in ['accelerate','safetensors','timm','einops'] if importlib.util.find_spec(name) is None]
assert not missing, ('Kaggle image missing expected modules', missing)
protected_after = {name: version_or_missing(name) for name in protected_names}
assert protected_before == protected_after, ('pip changed Kaggle binary packages', protected_before, protected_after)
abi = subprocess.run([sys.executable,'-c','import numpy,cv2,pandas,skimage; print(numpy.__version__,cv2.__version__,pandas.__version__,skimage.__version__)'], text=True, capture_output=True)
assert abi.returncode == 0, 'Dirty binary ABI; start a fresh Kaggle session.\\n' + abi.stderr
print('VINTERN_ABI_PROBE', abi.stdout.strip(), {'protected': protected_after, 'transformers': version_or_missing('transformers')}, flush=True)
"""

markdown = """# OCR production phase 2 — official Vintern FP16

Run this only after the assigned EasyOCR batches from phase 1 have printed `BATCH_COMPLETE`
and passed their HF round-trip verification. Attach `thvu165/aic-2026-keyframes`, select one
T4, enable Internet, and provide the write-scope private `HF_TOKEN`/`HK_TOKEN` Kaggle secret.
Use the same `WORKER_SLOT` assignment as phase 1.

The notebook downloads and verifies each completed EasyOCR archive from the existing private
HF Dataset, reconstructs only router-v2 candidate crops from pinned bboxes, runs official
`5CD-AI/Vintern-1B-v3_5` FP16 at revision
`b98f263eab246eb5269ade64edbdca8a887dc44d`, resumes by `candidate_id`, and uploads a
separate raw Vintern archive after exact candidate coverage passes. It never calls Gemini,
never modifies bbox, never writes SQLite, and clearly marks the output as uncalibrated and
not searchable. Calibrated override/materialization happens locally after both archives are
downloaded.
"""

notebook = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": lines(markdown)},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines(parameters)},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines(setup)},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines(RUNTIME.read_text(encoding="utf-8"))},
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
