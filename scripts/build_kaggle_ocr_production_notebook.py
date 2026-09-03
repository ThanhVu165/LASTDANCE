"""Regenerate the self-contained Kaggle OCR production EasyOCR notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "scripts" / "kaggle_ocr_production_easyocr_runtime.py"
OUTPUT = ROOT / "scripts" / "kaggle_ocr_production_easyocr.ipynb"


def lines(value: str) -> list[str]:
    return value.splitlines(keepends=True)


setup = """# Preserve Kaggle's CUDA/Torch/NumPy binary stack.
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import importlib.metadata as md, importlib.util, subprocess, sys

def version_or_missing(name):
    try: return md.version(name)
    except md.PackageNotFoundError: return None

protected_names = ['torch','torchvision','numpy','opencv-python','opencv-python-headless','scikit-image','pandas']
protected_before = {name: version_or_missing(name) for name in protected_names}
packages = ['huggingface-hub==0.24.7','easyocr==1.7.2','python-bidi==0.6.6','pyclipper==1.3.0.post6','ninja==1.11.1.4']
subprocess.check_call([sys.executable,'-m','pip','install','--quiet','--no-cache-dir','--no-deps',*packages])
missing = [name for name in ['yaml','scipy','skimage','shapely'] if importlib.util.find_spec(name) is None]
assert not missing, ('Kaggle image missing expected modules', missing)
protected_after = {name: version_or_missing(name) for name in protected_names}
assert protected_before == protected_after, ('pip changed Kaggle binary packages', protected_before, protected_after)
abi = subprocess.run([sys.executable,'-c','import numpy,cv2,pandas,skimage; print(numpy.__version__,cv2.__version__,pandas.__version__,skimage.__version__)'], text=True, capture_output=True)
assert abi.returncode == 0, 'Dirty binary ABI; start a fresh Kaggle session.\\n' + abi.stderr
print('ABI_PROBE', abi.stdout.strip(), {'protected': protected_after, 'easyocr': version_or_missing('easyocr')})
"""

parameters = """# CHANGE ONLY THIS NUMBER on each Kaggle account: 1, 2, 3, or 4.
WORKER_SLOT = 1

# True uploads each completed batch to the private HF Dataset and verifies round-trip SHA-256.
PUBLISH_TO_HF = True
HF_REPO_ID = 'MinhThuw0103/lastdance-visual-embeddings'

# Normal production: 0. For the one required interrupt/resume test, set 50 once,
# download the checkpoint ZIP, attach it to a fresh session, restore 0, and rerun.
INTERRUPT_AFTER_NEW_FRAMES = 0
CHECKPOINT_EVERY = 250
PROGRESS_EVERY = 25
"""

markdown = """# OCR production phase 1 — CRAFT + EasyOCR (4 accounts / 9 UID-disjoint batches)

Attach **only** `thvu165/aic-2026-keyframes`, select a T4 GPU, enable Internet, and add a
Kaggle secret named `HF_TOKEN` (or legacy `HK_TOKEN`) with write access to the private HF
Dataset. Use the same notebook on four accounts and change only `WORKER_SLOT` in the first
code cell.

- slot 1: batch-01 + batch-09 (69,784 frames)
- slot 2: batch-02 + batch-03 + batch-04 (83,798 frames)
- slot 3: batch-05 + batch-08 (76,705 frames)
- slot 4: batch-06 + batch-07 (63,049 frames)

This notebook pins catalog/batch UID hashes and EasyOCR weights, resumes by keyframe UID,
creates Vintern candidates without archiving images, uploads only completed per-batch
archives, and verifies HF round-trip checksums. It never calls Vintern/Gemini and never
builds or writes a shared SQLite file.
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
