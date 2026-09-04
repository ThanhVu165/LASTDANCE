"""Generate a Kaggle notebook embedding the self-contained ASR runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_notebook(output: Path, *, worker_count: int = 4) -> Path:
    if not 1 <= worker_count <= 4:
        raise ValueError("worker_count must be between 1 and 4")
    runtime = Path(__file__).with_name("kaggle_asr_production_runtime.py").read_text(
        encoding="utf-8"
    )
    setup = """import os
import subprocess
import sys
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir",
    "faster-whisper==1.1.1", "ctranslate2==4.5.0", "huggingface-hub==0.27.1",
])

# Download audio and catalog from HF Dataset.
# /kaggle/input is read-only, so keep the snapshot and HF cache in /kaggle/working.
print("Downloading audio batches from HuggingFace...")
from huggingface_hub import snapshot_download
DOWNLOAD_ROOT = "/kaggle/working/hf-asr"
snapshot_download(
    repo_id="Vu165/lastdance-asr",
    repo_type="dataset",
    local_dir=DOWNLOAD_ROOT,
    cache_dir="/kaggle/working/.cache/huggingface",
    allow_patterns=["asr/audio/**", "frames.csv"],
)
print(f"Audio ready at {DOWNLOAD_ROOT}")
"""
    parameters = f"""# Change only WORKER_SLOT on each Kaggle account.
WORKER_SLOT = 1
WORKER_BATCHES = {{
    1: ("batch-01", "batch-09"),
    2: ("batch-02", "batch-03", "batch-04"),
    3: ("batch-05", "batch-08"),
    4: ("batch-06", "batch-07"),
}}
WORKER_COUNT = {worker_count}
# Kaggle T4 x2 accelerator exposes 2 physical GPUs; set to 1 if only a single
# GPU is available on the notebook's accelerator setting.
GPU_COUNT = 2
AUDIO_ROOT = "/kaggle/working/hf-asr/asr/audio"
CATALOG_PATH = "/kaggle/working/hf-asr/frames.csv"
OUTPUT_ROOT = "/kaggle/working/asr/archives"
HF_REPO_ID = "Vu165/lastdance-asr"
# Push in-progress JSONL + checkpoint to HF every N transcribed videos, so a
# Kaggle timeout/crash mid-batch never loses more than this many videos. Set
# lower for shorter Kaggle quota windows, higher to reduce HF commit traffic.
CHECKPOINT_EVERY = 10
"""
    # Cell 3 writes the self-contained runtime to disk instead of executing it
    # inline, so Cell 4 can launch one OS subprocess per physical GPU. Each
    # subprocess pins CUDA_VISIBLE_DEVICES before importing faster-whisper,
    # which avoids the CUDA/fork-safety issues multiprocessing has inside
    # Jupyter kernels (functions defined in notebook cells are not reliably
    # picklable/importable by spawned child processes).
    write_runtime = "%%writefile kaggle_asr_runtime.py\n" + runtime
    run = """import os
import subprocess
import sys
from kaggle_secrets import UserSecretsClient

HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")

batch_ids = list(WORKER_BATCHES[WORKER_SLOT])
gpu_batches = [[] for _ in range(GPU_COUNT)]
for index, batch_id in enumerate(batch_ids):
    gpu_batches[index % GPU_COUNT].append(batch_id)

processes = []
for gpu_id, batches in enumerate(gpu_batches):
    if not batches:
        continue
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["ASR_BATCH_IDS"] = ",".join(batches)
    env["ASR_AUDIO_ROOT"] = AUDIO_ROOT
    env["ASR_OUTPUT_ROOT"] = OUTPUT_ROOT
    env["ASR_WORKER_ID"] = f"kaggle-{WORKER_SLOT:02d}-gpu{gpu_id}"
    env["ASR_CATALOG_PATH"] = CATALOG_PATH
    env["ASR_HF_REPO_ID"] = HF_REPO_ID
    env["ASR_HF_TOKEN"] = HF_TOKEN or ""
    env["ASR_CHECKPOINT_EVERY"] = str(CHECKPOINT_EVERY)
    print(f"Launching GPU {gpu_id} for batches: {batches}")
    process = subprocess.Popen([sys.executable, "kaggle_asr_runtime.py"], env=env)
    processes.append((gpu_id, batches, process))

failures = []
for gpu_id, batches, process in processes:
    return_code = process.wait()
    print(f"GPU {gpu_id} (batches={batches}) exited with code {return_code}")
    if return_code != 0:
        failures.append((gpu_id, batches, return_code))

if failures:
    raise RuntimeError(f"ASR worker subprocess(es) failed: {failures}")

print("All GPU workers completed. Batches processed:", batch_ids)
"""
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# ASR production — faster-whisper large-v3\n",
                    "Select the GPU T4 x2 accelerator, enable Internet, "
                    "and set the Kaggle secret `HF_TOKEN`. Change only `WORKER_SLOT` "
                    "per account. Batches assigned to a worker are split across both "
                    "GPUs (`GPU_COUNT`) and processed in parallel subprocesses; set "
                    "`GPU_COUNT = 1` if only a single GPU is selected. Progress is "
                    "checkpointed to HF every `CHECKPOINT_EVERY` videos, so a Kaggle "
                    "timeout or session kill mid-batch resumes automatically instead "
                    "of restarting the batch.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": setup.splitlines(True),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": parameters.splitlines(True),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": write_runtime.splitlines(True),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": run.splitlines(True),
            },
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "--output-notebook", dest="output", type=Path, required=True)
    parser.add_argument("--worker-count", type=int, default=4)
    args = parser.parse_args()
    print(build_notebook(args.output, worker_count=args.worker_count))
