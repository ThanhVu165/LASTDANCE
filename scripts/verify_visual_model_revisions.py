"""Resolve every enabled visual model revision through the Hugging Face API."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from offline.visual_models import DEFAULT_MODEL_CONFIG


def verify_visual_model_revisions(
    config_path: Path,
    *,
    model_info_loader: Callable[..., Any] | None = None,
) -> list[dict[str, str]]:
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise RuntimeError("unsupported visual embedding model config schema")
    if model_info_loader is None:
        try:
            from huggingface_hub import model_info
        except ImportError as error:
            raise RuntimeError(
                "huggingface-hub is missing; install requirements/kaggle-gpu.txt"
            ) from error
        model_info_loader = model_info

    results: list[dict[str, str]] = []
    for modality in ("clip", "siglip", "beit3"):
        row = payload.get(modality)
        if not isinstance(row, dict):
            raise RuntimeError(f"missing visual model config row: {modality}")
        if row.get("status") == "blocked_model_selection":
            continue
        model_id = str(row.get("model_id", ""))
        revision = str(row.get("revision", "")).lower()
        if (
            not model_id
            or len(revision) != 40
            or any(character not in "0123456789abcdef" for character in revision)
        ):
            raise RuntimeError(f"{modality} model ID/revision is not immutable")
        info = model_info_loader(model_id, revision=revision)
        resolved_id = str(getattr(info, "id", ""))
        resolved_sha = str(getattr(info, "sha", "")).lower()
        if resolved_id != model_id or resolved_sha != revision:
            raise RuntimeError(
                f"{modality} Hugging Face revision mismatch: "
                f"requested={model_id}@{revision}, resolved={resolved_id}@{resolved_sha}"
            )
        results.append(
            {
                "modality": modality,
                "model_id": model_id,
                "requested_revision": revision,
                "resolved_revision": resolved_sha,
            }
        )
    if not results:
        raise RuntimeError("no enabled Hugging Face visual models were configured")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    results = verify_visual_model_revisions(args.model_config.resolve())
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for result in results:
            print(
                f"PASS: {result['modality']} {result['model_id']} "
                f"revision={result['resolved_revision']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
