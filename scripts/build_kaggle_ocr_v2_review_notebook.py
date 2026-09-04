"""Regenerate the self-contained Kaggle OCR-v2 review-bundle notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "scripts" / "ocr_v2_review_bundle.py"
OUTPUT = ROOT / "notebooks" / "kaggle_ocr_v2_review.ipynb"


def lines(value: str) -> list[str]:
    return value.splitlines(keepends=True)


markdown = """# OCR v2 — build the Gate A/B human-review bundle

CPU is sufficient. Attach exactly one `ocr-production-batch-01-easyocr.zip` and JPEG
directories for the five configured videos. The notebook validates the immutable archive,
selects 20 frames per video plus 120 region crops, and writes:

`/kaggle/working/ocr-v2-batch-01-review/ocr-v2-review-bundle.zip`

It does not run any OCR model, call an API, or create a production artifact.
"""

parameters = """# Keep five unique Batch 01 videos. With SAMPLE_FRAMES=100 this gives 20/video.
VIDEO_IDS = ['L21_V001', 'L21_V002', 'L21_V003', 'L21_V005', 'L21_V006']
SAMPLE_FRAMES = 100
SAMPLE_REGIONS = 120

# /kaggle/input is safe when the attached Dataset contains only these five video folders.
# Point this to a narrower mounted directory if the same video IDs exist in multiple inputs.
KEYFRAME_ROOT = '/kaggle/input'
OUTPUT_DIR = '/kaggle/working/ocr-v2-batch-01-review'
"""

launcher = """from pathlib import Path
import json
import sys

zip_matches = sorted(Path('/kaggle/input').rglob('ocr-production-batch-01-easyocr.zip'))
if zip_matches:
    assert len(zip_matches) == 1, ('Attach exactly one Batch 01 EasyOCR archive', zip_matches)
    archive_source = zip_matches[0]
else:
    directory_matches = []
    for manifest_path in Path('/kaggle/input').rglob('batch-manifest.json'):
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        except Exception:
            continue
        required = {'easyocr-frames.jsonl', 'vintern-candidates.jsonl', 'run-signature.json', 'SHA256SUMS'}
        if (
            manifest.get('batch_id') == 'batch-01'
            and manifest.get('tier') == 'easyocr'
            and all((manifest_path.parent / name).is_file() for name in required)
        ):
            directory_matches.append(manifest_path.parent)
    assert len(directory_matches) == 1, ('Attach exactly one extracted Batch 01 archive', directory_matches)
    archive_source = directory_matches[0]
assert len(VIDEO_IDS) == 5 and len(set(VIDEO_IDS)) == 5, VIDEO_IDS
assert SAMPLE_FRAMES % len(VIDEO_IDS) == 0, (SAMPLE_FRAMES, VIDEO_IDS)

sys.argv = [
    'ocr_v2_review_bundle.py',
    '--archive', str(archive_source),
    '--keyframe-root', KEYFRAME_ROOT,
    '--video-ids', *VIDEO_IDS,
    '--sample-size', str(SAMPLE_FRAMES),
    '--region-sample-size', str(SAMPLE_REGIONS),
    '--output-dir', OUTPUT_DIR,
]
"""

notebook = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": lines(markdown)},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines(parameters)},
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines(launcher),
        },
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
