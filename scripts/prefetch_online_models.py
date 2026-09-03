"""Prefetch immutable query/VQA model snapshots before an offline competition run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from online.artifacts import EXPECTED_VISUAL
from online.qwen_runtime import DEFAULT_QWEN_MODEL_ID, DEFAULT_QWEN_REVISION


ALLOW_PATTERNS = {
    "clip": ["*.json", "*.txt", "*.model", "*.safetensors", "pytorch_model.bin"],
    "siglip": ["*.json", "*.txt", "*.model", "*.safetensors", "pytorch_model.bin"],
    "eva_clip": ["open_clip_config.json", "open_clip_model.safetensors"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--include-qwen", action="store_true")
    parser.add_argument("--local-files-only", action="store_true", help="validate cache without network")
    args = parser.parse_args()

    resolved: dict[str, str] = {}
    for modality, (_, model_id, revision) in EXPECTED_VISUAL.items():
        path = snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
            allow_patterns=ALLOW_PATTERNS[modality],
            local_files_only=args.local_files_only,
        )
        resolved[modality] = str(Path(path).resolve())
    if args.include_qwen:
        path = snapshot_download(
            repo_id=DEFAULT_QWEN_MODEL_ID,
            revision=DEFAULT_QWEN_REVISION,
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
            local_files_only=args.local_files_only,
        )
        resolved["qwen"] = str(Path(path).resolve())
    print(json.dumps(resolved, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
