"""Resolve every enabled visual model revision through the Hugging Face API."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from offline.visual_models import DEFAULT_MODEL_CONFIG


ENABLED_VISUAL_MODALITIES = ("clip", "siglip", "eva_clip")


def _metadata_value(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def verify_visual_model_revisions(
    config_path: Path,
    *,
    model_info_loader: Callable[..., Any] | None = None,
) -> list[dict[str, object]]:
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

    results: list[dict[str, object]] = []
    for modality in ENABLED_VISUAL_MODALITIES:
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
        info = model_info_loader(model_id, revision=revision, files_metadata=True)
        resolved_id = str(getattr(info, "id", ""))
        resolved_sha = str(getattr(info, "sha", "")).lower()
        if resolved_id != model_id or resolved_sha != revision:
            raise RuntimeError(
                f"{modality} Hugging Face revision mismatch: "
                f"requested={model_id}@{revision}, resolved={resolved_id}@{resolved_sha}"
            )
        result = {
            "modality": modality,
            "model_id": model_id,
            "requested_revision": revision,
            "resolved_revision": resolved_sha,
        }
        if modality == "eva_clip":
            expected_filename = str(row.get("weights_filename", ""))
            expected_sha256 = str(row.get("weights_sha256", "")).lower()
            expected_size = row.get("weights_size_bytes")
            if (
                expected_filename != "open_clip_model.safetensors"
                or len(expected_sha256) != 64
                or type(expected_size) is not int
                or int(expected_size) <= 0
            ):
                raise RuntimeError("eva_clip safetensors metadata is not immutable")
            sibling = next(
                (
                    item
                    for item in (getattr(info, "siblings", None) or ())
                    if str(_metadata_value(item, "rfilename")) == expected_filename
                ),
                None,
            )
            lfs = _metadata_value(sibling, "lfs") if sibling is not None else None
            resolved_weights_sha256 = str(_metadata_value(lfs, "sha256") or "").lower()
            resolved_weights_size = _metadata_value(lfs, "size")
            if resolved_weights_size is None and sibling is not None:
                resolved_weights_size = _metadata_value(sibling, "size")
            if (
                sibling is None
                or resolved_weights_sha256 != expected_sha256
                or resolved_weights_size != expected_size
            ):
                raise RuntimeError(
                    "eva_clip official safetensors metadata does not match the registry"
                )
            result.update(
                {
                    "weights_filename": expected_filename,
                    "weights_sha256": resolved_weights_sha256,
                    "weights_size_bytes": resolved_weights_size,
                }
            )
        results.append(result)
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
            suffix = ""
            if result["modality"] == "eva_clip":
                suffix = (
                    f" safetensors={result['weights_filename']}"
                    f" sha256={result['weights_sha256']}"
                )
            print(
                f"PASS: {result['modality']} {result['model_id']} "
                f"revision={result['resolved_revision']}{suffix}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
