"""Build exact frame/timestamp plans for Begin-Middle-End keyframes."""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from offline.identifiers import make_keyframe_uid
from shared.schemas.frame import FrameRecord

from .models import ShotBoundary
from .shot_detection import ExcludedTransitionRange


_Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True, slots=True)
class KeyframePlanItem:
    frame: FrameRecord
    relative_image_path: str

    def as_dict(self) -> dict[str, object]:
        return {
            **self.frame.model_dump(),
            "relative_image_path": self.relative_image_path,
        }


def probe_frame_timestamps(
    video_path: Path,
    *,
    ffprobe_binary: str = "ffprobe",
    runner: _Runner = subprocess.run,
) -> list[float]:
    """Read each decoded frame's best-effort timestamp from ffprobe."""

    command = [
        ffprobe_binary,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=best_effort_timestamp_time",
        "-of",
        "json",
        str(Path(video_path).resolve(strict=False)),
    ]
    result = runner(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or "ffprobe failed").strip()
        raise RuntimeError(f"ffprobe frame timestamp scan failed: {detail}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("ffprobe returned invalid frame timestamp JSON") from exc

    frames = payload.get("frames") or []
    timestamps: list[float] = []
    for frame_index, row in enumerate(frames):
        value = row.get("best_effort_timestamp_time")
        if value in (None, "", "N/A"):
            raise RuntimeError(f"frame {frame_index} has no usable timestamp")
        timestamp = float(value)
        if timestamp < 0:
            raise RuntimeError(f"frame {frame_index} has a negative timestamp")
        if timestamps and timestamp < timestamps[-1]:
            raise RuntimeError("frame timestamps must be monotonic")
        timestamps.append(timestamp)
    if not timestamps:
        raise RuntimeError("ffprobe returned no frame timestamps")
    return timestamps


def _representative_frame_ids(shot: ShotBoundary) -> list[int]:
    candidates = [
        shot.start_frame,
        (shot.start_frame + shot.end_frame) // 2,
        shot.end_frame,
    ]
    return list(dict.fromkeys(candidates))


def select_keyframes(
    *,
    video_id: str,
    shots: Iterable[ShotBoundary],
    frame_timestamps: Sequence[float],
    starting_local_idx: int = 0,
    excluded_transition_ranges: Iterable[ExcludedTransitionRange] = (),
) -> list[KeyframePlanItem]:
    """Select up to three unique representative frames for every shot."""

    normalized_video_id = video_id.strip()
    if not normalized_video_id:
        raise ValueError("video_id must not be empty")
    if starting_local_idx < 0:
        raise ValueError("starting_local_idx must be non-negative")
    if not frame_timestamps:
        raise ValueError("frame_timestamps must not be empty")
    excluded_ranges = tuple(excluded_transition_ranges)

    plan: list[KeyframePlanItem] = []
    local_idx = starting_local_idx
    for shot in shots:
        for frame_id in _representative_frame_ids(shot):
            if frame_id >= len(frame_timestamps):
                raise RuntimeError(
                    f"shot {shot.shot_id} references frame {frame_id}, but only "
                    f"{len(frame_timestamps)} timestamps were probed"
                )
            if any(
                excluded.start_frame <= frame_id <= excluded.end_frame
                for excluded in excluded_ranges
            ):
                raise RuntimeError(
                    f"shot {shot.shot_id} selects excluded transition frame {frame_id}"
                )
            uid = make_keyframe_uid(normalized_video_id, shot.shot_id, local_idx)
            frame = FrameRecord(
                video_id=normalized_video_id,
                local_idx=local_idx,
                frame_id=frame_id,
                pts_time=float(frame_timestamps[frame_id]),
                shot_id=shot.shot_id,
                window_id=None,
                keyframe_uid=uid,
            )
            plan.append(
                KeyframePlanItem(
                    frame=frame,
                    relative_image_path=(
                        f"keyframes/{normalized_video_id}/"
                        f"{shot.shot_id}_{local_idx}.jpg"
                    ),
                )
            )
            local_idx += 1
    if not plan:
        raise RuntimeError("cannot build a keyframe plan without shots")
    return plan


def write_keyframe_plan_atomic(
    output_path: Path,
    *,
    video_id: str,
    relative_video_path: str,
    items: Iterable[KeyframePlanItem],
) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [item.as_dict() for item in items]
    if not rows:
        raise ValueError("keyframe plan must contain at least one item")
    payload = {
        "schema_version": 1,
        "video_id": video_id,
        "relative_video_path": relative_video_path,
        "items": rows,
    }
    temporary = destination.with_name(f"{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def load_keyframe_plan(path: Path) -> tuple[str, str, list[KeyframePlanItem]]:
    """Load and validate a keyframe plan shared by downstream local stages."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    video_id = str(payload["video_id"])
    relative_video_path = str(payload["relative_video_path"])
    items: list[KeyframePlanItem] = []
    for row in payload.get("items", []):
        frame_payload = {
            key: row[key]
            for key in (
                "video_id",
                "local_idx",
                "frame_id",
                "pts_time",
                "shot_id",
                "window_id",
                "keyframe_uid",
            )
        }
        items.append(
            KeyframePlanItem(
                frame=FrameRecord(**frame_payload),
                relative_image_path=str(row["relative_image_path"]),
            )
        )
    if not items:
        raise RuntimeError("keyframe plan contains no items")
    if any(item.frame.video_id != video_id for item in items):
        raise RuntimeError("keyframe plan contains a mismatched video_id")
    if len({item.frame.keyframe_uid for item in items}) != len(items):
        raise RuntimeError("keyframe plan contains duplicate keyframe_uid values")
    frame_ids = [item.frame.frame_id for item in items]
    if frame_ids != sorted(set(frame_ids)):
        raise RuntimeError(
            "keyframe plan frame_id values must be unique and strictly increasing"
        )
    return video_id, relative_video_path, items


def extract_keyframe_exact(
    video_path: Path,
    item: KeyframePlanItem,
    *,
    data_root: Path,
    ffmpeg_binary: str = "ffmpeg",
    jpeg_quality: int = 3,
    runner: _Runner = subprocess.run,
) -> Path:
    """Extract one source frame atomically using FFmpeg's decoded-frame index."""

    if jpeg_quality < 2 or jpeg_quality > 5:
        raise ValueError("jpeg_quality must be between 2 and 5")
    root = Path(data_root).resolve(strict=False)
    source = Path(video_path).resolve(strict=False)
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise ValueError("video must be inside AIC_DATA") from exc

    relative_output = Path(item.relative_image_path)
    if relative_output.is_absolute() or ".." in relative_output.parts:
        raise ValueError("relative_image_path must stay inside AIC_DATA")
    output = (root / relative_output).resolve(strict=False)
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise ValueError("keyframe output must stay inside AIC_DATA") from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.stem}.tmp{output.suffix}")
    command = [
        ffmpeg_binary,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vf",
        f"select=eq(n\\,{item.frame.frame_id})",
        "-fps_mode",
        "vfr",
        "-frames:v",
        "1",
        "-q:v",
        str(jpeg_quality),
        "-y",
        str(temporary),
    ]
    result = runner(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        temporary.unlink(missing_ok=True)
        detail = (result.stderr or "ffmpeg failed").strip()
        raise RuntimeError(
            f"ffmpeg failed for frame {item.frame.frame_id}: {detail}"
        )
    if not temporary.is_file() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"ffmpeg did not produce frame {item.frame.frame_id} for {source.name}"
        )
    temporary.replace(output)
    return output


def extract_keyframes_exact_batch(
    video_path: Path,
    items: Sequence[KeyframePlanItem],
    *,
    data_root: Path,
    ffmpeg_binary: str = "ffmpeg",
    jpeg_quality: int = 3,
    runner: _Runner = subprocess.run,
    on_progress: Callable[[int], None] | None = None,
) -> list[Path]:
    """Decode a video once and atomically publish multiple exact frame indexes."""

    if jpeg_quality < 2 or jpeg_quality > 5:
        raise ValueError("jpeg_quality must be between 2 and 5")
    if not items:
        return []

    root = Path(data_root).resolve(strict=False)
    source = Path(video_path).resolve(strict=False)
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise ValueError("video must be inside AIC_DATA") from exc

    frame_ids = [item.frame.frame_id for item in items]
    if frame_ids != sorted(set(frame_ids)):
        raise ValueError("batch frame_id values must be unique and strictly increasing")
    video_ids = {item.frame.video_id for item in items}
    if len(video_ids) != 1:
        raise ValueError("batch items must belong to exactly one video_id")

    outputs: list[Path] = []
    for item in items:
        relative_output = Path(item.relative_image_path)
        if relative_output.is_absolute() or ".." in relative_output.parts:
            raise ValueError("relative_image_path must stay inside AIC_DATA")
        output = (root / relative_output).resolve(strict=False)
        try:
            output.relative_to(root)
        except ValueError as exc:
            raise ValueError("keyframe output must stay inside AIC_DATA") from exc
        outputs.append(output)

    staging_parent = root / "keyframes"
    staging_parent.mkdir(parents=True, exist_ok=True)
    video_id = next(iter(video_ids))
    with tempfile.TemporaryDirectory(
        prefix=f".{video_id}-extract-",
        dir=staging_parent,
    ) as staging_folder:
        staging_root = Path(staging_folder)
        output_pattern = staging_root / "%08d.jpg"
        selection = "+".join(f"eq(n\\,{frame_id})" for frame_id in frame_ids)
        command = [
            ffmpeg_binary,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-vf",
            f"select={selection}",
            "-fps_mode",
            "vfr",
            "-frames:v",
            str(len(items)),
            "-start_number",
            "0",
            "-q:v",
            str(jpeg_quality),
            "-y",
            str(output_pattern),
        ]
        result = runner(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or "ffmpeg failed").strip()
            raise RuntimeError(f"ffmpeg batch extraction failed: {detail}")

        staged_outputs = [
            staging_root / f"{index:08d}.jpg" for index in range(len(items))
        ]
        actual_outputs = sorted(staging_root.glob("*.jpg"))
        if actual_outputs != staged_outputs or any(
            not path.is_file() or path.stat().st_size == 0
            for path in staged_outputs
        ):
            raise RuntimeError(
                "ffmpeg batch extraction did not produce every planned keyframe"
            )

        published: list[Path] = []
        for batch_index, (staged, output) in enumerate(
            zip(staged_outputs, outputs, strict=True),
            start=1,
        ):
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(f"{output.stem}.tmp{output.suffix}")
            staged.replace(temporary)
            temporary.replace(output)
            published.append(output)
            if on_progress is not None:
                on_progress(batch_index)
        return published
