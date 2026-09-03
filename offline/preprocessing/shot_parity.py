"""Exact semantic parity checks for CPU/GPU shot manifests."""

from __future__ import annotations

import json
from pathlib import Path

from .shot_detection import load_shot_manifest


_PARITY_SIGNATURE_KEYS = (
    "name",
    "implementation",
    "package_version",
    "threshold",
    "weights_sha256",
)


def compare_shot_manifests(
    reference_path: Path,
    candidate_path: Path,
) -> list[str]:
    """Return exact contract mismatches; device may differ intentionally."""

    reference_payload = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    candidate_payload = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
    reference_video_id, reference_relative_path, reference = load_shot_manifest(
        reference_path
    )
    candidate_video_id, candidate_relative_path, candidate = load_shot_manifest(
        candidate_path
    )

    mismatches: list[str] = []
    if reference_video_id != candidate_video_id:
        mismatches.append(
            "video_id differs: "
            f"reference={reference_video_id!r}, candidate={candidate_video_id!r}"
        )
    if reference_relative_path != candidate_relative_path:
        mismatches.append(
            "relative_video_path differs: "
            f"reference={reference_relative_path!r}, "
            f"candidate={candidate_relative_path!r}"
        )
    if reference.total_frame_count != candidate.total_frame_count:
        mismatches.append(
            "total_frame_count differs: "
            f"reference={reference.total_frame_count}, "
            f"candidate={candidate.total_frame_count}"
        )

    reference_signature = reference_payload.get("detector_signature")
    candidate_signature = candidate_payload.get("detector_signature")
    if not isinstance(reference_signature, dict):
        mismatches.append("reference detector_signature is missing")
        reference_signature = {}
    if not isinstance(candidate_signature, dict):
        mismatches.append("candidate detector_signature is missing")
        candidate_signature = {}
    for key in _PARITY_SIGNATURE_KEYS:
        if reference_signature.get(key) != candidate_signature.get(key):
            mismatches.append(
                f"detector_signature.{key} differs: "
                f"reference={reference_signature.get(key)!r}, "
                f"candidate={candidate_signature.get(key)!r}"
            )

    if reference.shots != candidate.shots:
        common_count = min(len(reference.shots), len(candidate.shots))
        first_mismatch = next(
            (
                index
                for index in range(common_count)
                if reference.shots[index] != candidate.shots[index]
            ),
            None,
        )
        if first_mismatch is None:
            mismatches.append(
                "shot count differs: "
                f"reference={len(reference.shots)}, candidate={len(candidate.shots)}"
            )
        else:
            mismatches.append(
                f"shot[{first_mismatch}] differs: "
                f"reference={reference.shots[first_mismatch].as_dict()}, "
                f"candidate={candidate.shots[first_mismatch].as_dict()}"
            )

    if reference.excluded_transition_ranges != candidate.excluded_transition_ranges:
        mismatches.append(
            "excluded_transition_ranges differ: "
            f"reference={[item.as_dict() for item in reference.excluded_transition_ranges]}, "
            f"candidate={[item.as_dict() for item in candidate.excluded_transition_ranges]}"
        )
    return mismatches
