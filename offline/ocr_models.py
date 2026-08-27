"""Fail-closed registry and offline preflight for EasyOCR model files."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EASYOCR_REGISTRY = (
    _REPOSITORY_ROOT / "configs" / "ocr_easyocr_models.json"
)
_MODEL_KEYS = ("craft", "latin_g2")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_hex(value: object, *, digits: int, field: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != digits or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise RuntimeError(f"invalid {field}")
    return normalized


def load_easyocr_registry(
    path: Path = DEFAULT_EASYOCR_REGISTRY,
) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read EasyOCR registry: {source}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RuntimeError("unsupported EasyOCR registry schema")

    package = payload.get("package")
    runtime = payload.get("runtime")
    models = payload.get("models")
    if not isinstance(package, dict) or not isinstance(runtime, dict):
        raise RuntimeError("EasyOCR package/runtime registry is incomplete")
    if not isinstance(models, dict) or set(models) != set(_MODEL_KEYS):
        raise RuntimeError("EasyOCR registry must pin exactly craft and latin_g2")
    if package.get("name") != "easyocr" or package.get("version") != "1.7.2":
        raise RuntimeError("EasyOCR package version is not pinned to 1.7.2")
    if package.get("wheel_filename") != "easyocr-1.7.2-py3-none-any.whl":
        raise RuntimeError("unexpected EasyOCR wheel filename")
    if type(package.get("wheel_size_bytes")) is not int:
        raise RuntimeError("EasyOCR wheel size is not pinned")
    _require_hex(package.get("wheel_sha256"), digits=64, field="wheel SHA-256")
    _require_hex(package.get("upstream_revision"), digits=40, field="upstream revision")

    if runtime != {
        "languages": ["vi", "en"],
        "detect_network": "craft",
        "recognition_network": "latin_g2",
        "download_enabled": False,
    }:
        raise RuntimeError("EasyOCR offline runtime contract changed unexpectedly")

    expected_filenames = {
        "craft": ("craft_mlt_25k.zip", "craft_mlt_25k.pth"),
        "latin_g2": ("latin_g2.zip", "latin_g2.pth"),
    }
    for key in _MODEL_KEYS:
        row = models[key]
        if not isinstance(row, dict):
            raise RuntimeError(f"EasyOCR model row is invalid: {key}")
        archive_name, weights_name = expected_filenames[key]
        if row.get("archive_filename") != archive_name:
            raise RuntimeError(f"unexpected EasyOCR archive filename: {key}")
        if row.get("weights_filename") != weights_name:
            raise RuntimeError(f"unexpected EasyOCR weights filename: {key}")
        if not str(row.get("archive_url", "")).startswith(
            "https://github.com/JaidedAI/EasyOCR/releases/download/"
        ):
            raise RuntimeError(f"EasyOCR model URL is not an official release asset: {key}")
        for size_field in ("archive_size_bytes", "weights_size_bytes"):
            if type(row.get(size_field)) is not int or int(row[size_field]) <= 0:
                raise RuntimeError(f"invalid {key} {size_field}")
        _require_hex(row.get("archive_sha256"), digits=64, field=f"{key} archive SHA-256")
        _require_hex(row.get("weights_sha256"), digits=64, field=f"{key} weights SHA-256")
        _require_hex(row.get("weights_md5_upstream"), digits=32, field=f"{key} MD5")
    return payload


def _verify_file(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    expected_md5: str | None = None,
) -> dict[str, object]:
    source = Path(path)
    if not source.is_file():
        raise RuntimeError(f"required offline file is missing: {source}")
    actual_size = source.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"size mismatch for {source.name}: expected={expected_size}, actual={actual_size}"
        )
    actual_sha256 = sha256_file(source)
    if actual_sha256 != expected_sha256.lower():
        raise RuntimeError(
            f"SHA-256 mismatch for {source.name}: "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )
    result: dict[str, object] = {
        "filename": source.name,
        "size_bytes": actual_size,
        "sha256": actual_sha256,
    }
    if expected_md5 is not None:
        actual_md5 = md5_file(source)
        if actual_md5 != expected_md5.lower():
            raise RuntimeError(
                f"upstream MD5 mismatch for {source.name}: "
                f"expected={expected_md5}, actual={actual_md5}"
            )
        result["md5_upstream"] = actual_md5
    return result


def verify_easyocr_offline_files(
    model_storage_directory: Path,
    *,
    registry_path: Path = DEFAULT_EASYOCR_REGISTRY,
    archive_directory: Path | None = None,
    wheel_path: Path | None = None,
    verify_package_version: bool = True,
) -> dict[str, object]:
    """Verify every byte needed before constructing a download-disabled Reader."""

    registry = load_easyocr_registry(registry_path)
    package = registry["package"]
    models = registry["models"]
    if verify_package_version:
        try:
            installed_version = importlib.metadata.version("easyocr")
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError("easyocr==1.7.2 is not installed") from error
        if installed_version != package["version"]:
            raise RuntimeError(
                f"EasyOCR version mismatch: expected={package['version']}, "
                f"actual={installed_version}"
            )
    else:
        installed_version = None

    storage = Path(model_storage_directory)
    weights = []
    archives = []
    for key in _MODEL_KEYS:
        row = models[key]
        result = _verify_file(
            storage / row["weights_filename"],
            expected_size=row["weights_size_bytes"],
            expected_sha256=row["weights_sha256"],
            expected_md5=row["weights_md5_upstream"],
        )
        result["model"] = key
        weights.append(result)
        if archive_directory is not None:
            archive_result = _verify_file(
                Path(archive_directory) / row["archive_filename"],
                expected_size=row["archive_size_bytes"],
                expected_sha256=row["archive_sha256"],
            )
            archive_result["model"] = key
            archives.append(archive_result)

    wheel = None
    if wheel_path is not None:
        wheel = _verify_file(
            Path(wheel_path),
            expected_size=package["wheel_size_bytes"],
            expected_sha256=package["wheel_sha256"],
        )
    return {
        "schema_version": 1,
        "package": package["name"],
        "expected_package_version": package["version"],
        "installed_package_version": installed_version,
        "download_enabled": False,
        "weights": weights,
        "archives": archives,
        "wheel": wheel,
    }
