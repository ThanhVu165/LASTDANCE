"""Regenerate the locked Gemini OCR production Kaggle notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "scripts" / "kaggle_ocr_production_gemini_runtime.py"
OUTPUT = ROOT / "scripts" / "kaggle_ocr_production_gemini.ipynb"


def lines(value: str) -> list[str]:
    return value.splitlines(keepends=True)


parameters = """# SAFE DEFAULT: validates and prints exact counts/cost without reading GEMINI_API_KEY.
EXECUTION_MODE = 'preflight'  # preflight | canary | production

# Canary approval: change only after the user has reviewed the exact preflight report.
APPROVE_PAID_CANARY = False
CANARY_REQUESTS = 100

# Production approval: fill only from an accepted canary + explicit user budget decision.
APPROVE_GEMINI_PRODUCTION = False
APPROVED_REPORT_SHA256 = ''
APPROVED_MODEL_VERSION = ''
APPROVED_MAX_REQUESTS = 0
APPROVED_MAX_VND = 0

# Set this from the paid project's real console quota; do not guess the ceiling.
REQUESTS_PER_MINUTE = 30.0

# Only complete production batches are uploaded to this existing private Dataset.
PUBLISH_TO_HF = True
HF_REPO_ID = 'MinhThuw0103/lastdance-visual-embeddings'
"""

setup = """# CPU/API notebook. Preserve Kaggle binary packages; install no model runtime.
import importlib.metadata as md, subprocess, sys

def version_or_missing(name):
    try: return md.version(name)
    except md.PackageNotFoundError: return None

protected_names = ['torch','torchvision','numpy','opencv-python','opencv-python-headless','scikit-image','pandas']
protected_before = {name: version_or_missing(name) for name in protected_names}
subprocess.check_call([sys.executable,'-m','pip','install','--quiet','--no-cache-dir','--no-deps','huggingface-hub==0.24.7','httpx==0.28.1'])
protected_after = {name: version_or_missing(name) for name in protected_names}
assert protected_before == protected_after, ('pip changed Kaggle binary packages', protected_before, protected_after)
print('GEMINI_RUNTIME_READY', {'protected': protected_after, 'httpx': version_or_missing('httpx')}, flush=True)
"""

markdown = """# OCR production phase 3 — Gemini 2.5 Flash-Lite residual

Attach two inputs: `thvu165/aic-2026-keyframes` and the exact
`ocr-gemini-preflight.zip` produced after all EasyOCR+Vintern archives pass. This notebook
uses CPU only; no Kaggle GPU is required.

The safe default `EXECUTION_MODE='preflight'` validates checksums and prints the exact
region/frame/shot/request count plus planning cost. It does not read `GEMINI_API_KEY` and
cannot call the API. After explicit user approval, run `canary` for up to 100 requests,
download `ocr-gemini-paid-canary-report.json`, and pin its single `model_version`. Only then
may `production` be enabled with the exact preflight report SHA, approved model version,
request cap and VND cap.

The request uses the Standard synchronous API, one shot per API request, and one or more labeled contact sheets for dense
shots, global `MEDIA_RESOLUTION_MEDIUM`, and strict structured output over the exact string
`region_id` set. Gemini returns only text/language/confidence; detector bbox and UID remain
local. `401/403` fail immediately, `429/5xx` retry with backoff, every final result resumes
by `request_id`, and completed batches are uploaded to the existing private HF Dataset at
`ocr/archives/{batch_id}/gemini/` with round-trip checksum verification.

This notebook is not a Gemini Batch API implementation. Full production fails closed when
the exact Standard estimate exceeds the approved budget; a Batch price estimate cannot be
used to unlock this Standard runner.
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
