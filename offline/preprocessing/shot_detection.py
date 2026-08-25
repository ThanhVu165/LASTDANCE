"""Swappable shot-detection interface with a lazy TransNetV2 default."""

from __future__ import annotations

import json
import importlib.metadata
import os
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from offline.artifacts import verify_sha256

from .models import ShotBoundary


TRANSNETV2_PYTORCH_BUNDLED_WEIGHTS_SHA256 = (
    "a313d0b3bebfa9a71914b375bfdf918a30b5c3b1e6be51972d35dd8078b442de"
)
TRANSITION_EXCLUSION_REASON = "transition_score_above_threshold"
DEFAULT_TRANSITION_EXCLUSION_WARNING_THRESHOLD = 0.01
DEFAULT_TRANSNETV2_DEVICE = "cpu"
SUPPORTED_TRANSNETV2_DEVICES = frozenset({"cpu", "cuda"})


def normalize_transnetv2_device(device: str) -> str:
    normalized = str(device).strip().lower()
    if normalized not in SUPPORTED_TRANSNETV2_DEVICES:
        choices = ", ".join(sorted(SUPPORTED_TRANSNETV2_DEVICES))
        raise ValueError(f"TransNetV2 device must be one of: {choices}")
    return normalized


def ensure_transnetv2_device_available(device: str) -> str:
    """Reject unavailable CUDA explicitly; never fall back to CPU silently."""

    normalized = normalize_transnetv2_device(device)
    if normalized == "cuda":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("CUDA Shot Detection requires PyTorch") from exc
        if not torch.cuda.is_available():
            raise RuntimeError(
                "TransNetV2 device=cuda was requested but CUDA is unavailable"
            )
    return normalized


@dataclass(frozen=True, slots=True)
class ExcludedTransitionRange:
    start_frame: int
    end_frame: int
    reason: str = TRANSITION_EXCLUSION_REASON

    def __post_init__(self) -> None:
        if self.start_frame < 0:
            raise ValueError("excluded transition start_frame must be non-negative")
        if self.end_frame < self.start_frame:
            raise ValueError(
                "excluded transition end_frame must be >= start_frame"
            )
        if self.reason != TRANSITION_EXCLUSION_REASON:
            raise ValueError("unsupported excluded transition reason")

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame + 1

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ShotDetectionResult:
    shots: tuple[ShotBoundary, ...]
    total_frame_count: int
    excluded_transition_ranges: tuple[ExcludedTransitionRange, ...] = ()

    def __post_init__(self) -> None:
        if self.total_frame_count <= 0:
            raise ValueError("total_frame_count must be positive")
        if not self.shots:
            raise ValueError("shot detector returned no boundaries")


@contextmanager
def _configured_ffmpeg_on_path() -> Iterable[None]:
    """Expose an explicitly configured FFmpeg binary to ffmpeg-python temporarily."""

    configured = os.environ.get("AIC_FFMPEG", "").strip()
    candidate = Path(configured).expanduser() if configured else None
    if candidate is None or not candidate.is_file():
        yield
        return
    previous = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{candidate.resolve().parent}{os.pathsep}{previous}"
    try:
        yield
    finally:
        os.environ["PATH"] = previous


class ShotDetector(ABC):
    """Model-independent shot boundary contract."""

    name: str

    @abstractmethod
    def detect(self, video_path: Path) -> ShotDetectionResult:
        raise NotImplementedError

    @property
    def signature(self) -> dict[str, object]:
        return {"name": self.name}


def resolve_and_verify_transnetv2_weights(
    weights_path: Path | None = None,
    expected_weights_sha256: str | None = None,
) -> tuple[Path, str, str]:
    """Resolve either a verified external override or the pinned bundled weight."""

    if weights_path is not None:
        resolved = Path(weights_path).expanduser().resolve()
        if not expected_weights_sha256:
            raise RuntimeError("external TransNetV2 weights require a SHA-256 checksum")
        if not resolved.is_file():
            raise RuntimeError(f"TransNetV2 weight file not found: {resolved}")
        actual = verify_sha256(resolved, expected_weights_sha256)
        return resolved, actual, "external"

    try:
        import transnetv2_pytorch
    except ImportError as exc:
        raise RuntimeError(
            "transnetv2-pytorch is not installed and no external weight was supplied"
        ) from exc
    package_file = Path(transnetv2_pytorch.__file__).resolve()
    resolved = package_file.with_name("transnetv2-pytorch-weights.pth")
    if not resolved.is_file():
        raise RuntimeError(f"bundled TransNetV2 weight file not found: {resolved}")
    actual = verify_sha256(resolved, TRANSNETV2_PYTORCH_BUNDLED_WEIGHTS_SHA256)
    return resolved, actual, "bundled"


class TransNetV2ShotDetector(ShotDetector):
    """TransNetV2 adapter with a Python 3.11-compatible PyTorch path.

    The model import and weight load are lazy. Tests and orchestration may inject
    ``model_factory`` so this module never downloads weights by itself.
    """

    name = "transnetv2"

    def __init__(
        self,
        *,
        weights_path: Path | None = None,
        expected_weights_sha256: str | None = None,
        model_dir: Path | None = None,
        model_factory: Callable[[], Any] | None = None,
        threshold: float = 0.5,
        device: str = DEFAULT_TRANSNETV2_DEVICE,
    ) -> None:
        self._weights_path = Path(weights_path) if weights_path is not None else None
        self._expected_weights_sha256 = expected_weights_sha256
        self._model_dir = Path(model_dir) if model_dir is not None else None
        self._model_factory = model_factory
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self._threshold = float(threshold)
        self._device = normalize_transnetv2_device(device)
        self._model: Any | None = None
        self._weights_sha256: str | None = None
        self._weights_source: str | None = None
        self._package_version: str | None = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        ensure_transnetv2_device_available(self._device)
        if self._model_factory is not None:
            self._model = self._model_factory()
            self._weights_source = "injected-model-factory"
            return self._model

        try:
            from transnetv2_pytorch import TransNetV2
        except ImportError:
            if self._device != "cpu":
                raise RuntimeError(
                    "device=cuda requires the transnetv2-pytorch implementation"
                )
            return self._load_legacy_tensorflow_model()

        resolved_weights, actual_sha256, weights_source = (
            resolve_and_verify_transnetv2_weights(
                self._weights_path,
                self._expected_weights_sha256,
            )
        )
        self._weights_sha256 = actual_sha256
        self._weights_source = weights_source
        self._package_version = importlib.metadata.version("transnetv2-pytorch")
        if weights_source == "external":
            # The package constructor loads its bundled weight before we apply the
            # override, so that bundled file must still pass integrity validation.
            resolve_and_verify_transnetv2_weights()

        try:
            self._model = TransNetV2(device=self._device)
        except TypeError as exc:
            if self._device != "cpu":
                raise RuntimeError(
                    "installed TransNetV2 implementation does not support device=cuda"
                ) from exc
            self._model = TransNetV2()

        if weights_source == "external":
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError(
                    "PyTorch is required by transnetv2-pytorch but is not installed."
                ) from exc
            try:
                state = torch.load(resolved_weights, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(resolved_weights, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self._model.load_state_dict(state)
        self._model.eval()
        return self._model

    def _load_legacy_tensorflow_model(self) -> Any:
        try:
            from transnetv2 import TransNetV2
        except ImportError as exc:
            raise RuntimeError(
                "TransNetV2 is not installed. Install requirements/shot-transnetv2.txt "
                "or inject another ShotDetector implementation."
            ) from exc
        options = {}
        if self._model_dir is not None:
            options["model_dir"] = str(self._model_dir)
        self._model = TransNetV2(**options)
        return self._model

    def detect(self, video_path: Path) -> ShotDetectionResult:
        model = self._load_model()
        with _configured_ffmpeg_on_path():
            prediction = model.predict_video(str(Path(video_path)), quiet=True)
        if not isinstance(prediction, tuple) or len(prediction) != 3:
            raise RuntimeError("unexpected TransNetV2 predict_video result")
        _, single_frame_predictions, _ = prediction
        if hasattr(single_frame_predictions, "detach"):
            single_frame_predictions = (
                single_frame_predictions.detach().cpu().numpy()
            )
        raw_scenes = model.predictions_to_scenes(
            single_frame_predictions,
            threshold=self._threshold,
        )
        shots = tuple(_normalize_scenes(raw_scenes))
        total_frame_count = len(single_frame_predictions)
        excluded_transition_ranges = _find_excluded_transition_ranges(
            shots,
            total_frame_count=total_frame_count,
        )
        return ShotDetectionResult(
            shots=shots,
            total_frame_count=total_frame_count,
            excluded_transition_ranges=excluded_transition_ranges,
        )

    @property
    def signature(self) -> dict[str, object]:
        return {
            "name": self.name,
            "implementation": (
                "injected-model-factory"
                if self._model_factory is not None
                else "transnetv2-pytorch"
            ),
            "package_version": self._package_version,
            "device": self._device,
            "threshold": self._threshold,
            "weights_source": self._weights_source,
            "weights_sha256": self._weights_sha256,
        }


def _normalize_scenes(raw_scenes: Iterable[Iterable[object]]) -> list[ShotBoundary]:
    boundaries: list[ShotBoundary] = []
    previous_end = -1
    for index, scene in enumerate(raw_scenes):
        values = list(scene)
        if len(values) != 2:
            raise RuntimeError("shot boundary must contain start and end frame")
        boundary = ShotBoundary(
            shot_id=f"s{index:06d}",
            start_frame=int(values[0]),
            end_frame=int(values[1]),
        )
        if boundary.start_frame <= previous_end:
            raise RuntimeError("shot boundaries must be sorted and non-overlapping")
        boundaries.append(boundary)
        previous_end = boundary.end_frame
    if not boundaries:
        raise RuntimeError("shot detector returned no boundaries")
    return boundaries


def _find_excluded_transition_ranges(
    shots: Iterable[ShotBoundary],
    *,
    total_frame_count: int,
) -> tuple[ExcludedTransitionRange, ...]:
    if total_frame_count <= 0:
        raise ValueError("total_frame_count must be positive")

    ranges: list[ExcludedTransitionRange] = []
    expected_start = 0
    seen_shot = False
    for shot in shots:
        seen_shot = True
        if shot.start_frame < expected_start:
            raise RuntimeError("shot boundaries must be sorted and non-overlapping")
        if shot.end_frame >= total_frame_count:
            raise RuntimeError("shot boundary exceeds total_frame_count")
        if shot.start_frame > expected_start:
            ranges.append(
                ExcludedTransitionRange(
                    start_frame=expected_start,
                    end_frame=shot.start_frame - 1,
                )
            )
        expected_start = shot.end_frame + 1

    if not seen_shot:
        raise RuntimeError("shot detector returned no boundaries")
    if expected_start < total_frame_count:
        ranges.append(
            ExcludedTransitionRange(
                start_frame=expected_start,
                end_frame=total_frame_count - 1,
            )
        )
    return tuple(ranges)


def validate_shot_detection_result(
    result: ShotDetectionResult,
    *,
    exclusion_warning_threshold: float = (
        DEFAULT_TRANSITION_EXCLUSION_WARNING_THRESHOLD
    ),
) -> dict[str, object]:
    """Validate exact coverage accounting and warn on abnormal exclusions."""

    if not 0.0 <= exclusion_warning_threshold <= 1.0:
        raise ValueError("exclusion_warning_threshold must be between 0 and 1")
    expected_ranges = _find_excluded_transition_ranges(
        result.shots,
        total_frame_count=result.total_frame_count,
    )
    if result.excluded_transition_ranges != expected_ranges:
        raise RuntimeError(
            "excluded_transition_ranges do not match gaps between shot boundaries"
        )

    excluded_frame_count = sum(
        item.frame_count for item in result.excluded_transition_ranges
    )
    excluded_fraction = excluded_frame_count / result.total_frame_count
    exceeds_warning_threshold = excluded_fraction > exclusion_warning_threshold
    if exceeds_warning_threshold:
        warnings.warn(
            "excluded transition frames exceed warning threshold: "
            f"{excluded_fraction:.3%} > {exclusion_warning_threshold:.3%}",
            RuntimeWarning,
            stacklevel=2,
        )
    return {
        "total_frame_count": result.total_frame_count,
        "excluded_frame_count": excluded_frame_count,
        "excluded_frame_fraction": excluded_fraction,
        "warning_threshold": exclusion_warning_threshold,
        "exceeds_warning_threshold": exceeds_warning_threshold,
    }


def load_shot_manifest(
    path: Path,
    *,
    exclusion_warning_threshold: float = (
        DEFAULT_TRANSITION_EXCLUSION_WARNING_THRESHOLD
    ),
) -> tuple[str, str, ShotDetectionResult]:
    """Load and fail-closed validate a schema-v2 shot manifest."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise RuntimeError("shot manifest schema_version must be 2")
    video_id = str(payload["video_id"])
    relative_video_path = str(payload["relative_video_path"])
    stored_validation = payload.get("transition_exclusion_validation")
    if not isinstance(stored_validation, dict):
        raise RuntimeError("shot manifest is missing transition exclusion validation")

    detection = ShotDetectionResult(
        shots=tuple(ShotBoundary(**row) for row in payload.get("shots", [])),
        total_frame_count=int(stored_validation["total_frame_count"]),
        excluded_transition_ranges=tuple(
            ExcludedTransitionRange(**row)
            for row in payload.get("excluded_transition_ranges", [])
        ),
    )
    actual_validation = validate_shot_detection_result(
        detection,
        exclusion_warning_threshold=exclusion_warning_threshold,
    )
    if stored_validation != actual_validation:
        raise RuntimeError("shot manifest transition exclusion validation is stale")
    return video_id, relative_video_path, detection


def get_default_shot_detector(
    *,
    weights_path: Path | None = None,
    expected_weights_sha256: str | None = None,
    model_dir: Path | None = None,
    device: str = DEFAULT_TRANSNETV2_DEVICE,
) -> ShotDetector:
    return TransNetV2ShotDetector(
        weights_path=weights_path,
        expected_weights_sha256=expected_weights_sha256,
        model_dir=model_dir,
        device=device,
    )


def write_shot_manifest_atomic(
    output_path: Path,
    *,
    video_id: str,
    relative_video_path: str,
    detector: ShotDetector,
    detection: ShotDetectionResult,
    exclusion_warning_threshold: float = (
        DEFAULT_TRANSITION_EXCLUSION_WARNING_THRESHOLD
    ),
) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    transition_exclusion_validation = validate_shot_detection_result(
        detection,
        exclusion_warning_threshold=exclusion_warning_threshold,
    )
    payload = {
        "schema_version": 2,
        "video_id": video_id,
        "relative_video_path": relative_video_path,
        "detector": detector.name,
        "detector_signature": detector.signature,
        "excluded_transition_ranges": [
            item.as_dict() for item in detection.excluded_transition_ranges
        ],
        "transition_exclusion_validation": transition_exclusion_validation,
        "shots": [shot.as_dict() for shot in detection.shots],
    }
    temporary = destination.with_name(f"{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
