"""Re-route recovered Gate 2 regions without running an OCR model.

Generated reports/candidate JSONL files are audit artifacts and must stay out of
Git.  The decision authorizes only a dev-only end-to-end pilot, never production.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from offline.ocr_vintern_gate2 import (
    VinternGate2Policy,
    route_vintern_region,
    vintern_output_rejection_reasons,
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--easyocr-jsonl", type=Path, required=True)
    parser.add_argument("--vintern-results-jsonl", type=Path, required=True)
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs"
        / "ocr_vintern_gate2_policy.json",
    )
    parser.add_argument("--full-catalog-frames", type=int, default=293_336)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-candidates-jsonl", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.full_catalog_frames <= 0:
        raise ValueError("--full-catalog-frames must be positive")
    policy = VinternGate2Policy.model_validate_json(
        args.policy.read_text(encoding="utf-8")
    )
    easy_rows = _load_jsonl(args.easyocr_jsonl)
    vintern_rows = _load_jsonl(args.vintern_results_jsonl)

    frame_uids: set[int] = set()
    region_ids: set[str] = set()
    old_candidates: set[str] = set()
    candidate_rows: list[dict[str, Any]] = []
    reasons = Counter()
    by_video: dict[str, Counter[str]] = defaultdict(Counter)
    total_regions = 0
    for frame in easy_rows:
        frame_uid = frame.get("keyframe_uid")
        if not isinstance(frame_uid, int) or isinstance(frame_uid, bool):
            raise ValueError("every EasyOCR frame requires an integer keyframe_uid")
        if frame_uid in frame_uids:
            raise ValueError(f"duplicate keyframe_uid: {frame_uid}")
        frame_uids.add(frame_uid)
        video_id = str(frame.get("video_id") or "")
        if not video_id:
            raise ValueError(f"missing video_id for keyframe_uid {frame_uid}")
        for region in frame.get("regions") or []:
            total_regions += 1
            region_id = str(region.get("region_id") or "")
            if not region_id:
                raise ValueError(f"missing region_id for keyframe_uid {frame_uid}")
            if region_id in region_ids:
                raise ValueError(f"duplicate region_id: {region_id}")
            region_ids.add(region_id)
            if region.get("escalation_reasons"):
                old_candidates.add(region_id)
            decision = route_vintern_region(region, policy=policy)
            by_video[video_id]["regions"] += 1
            if not decision.candidate:
                continue
            by_video[video_id]["candidates"] += 1
            reasons.update(decision.reasons)
            candidate_rows.append(
                {
                    "schema_version": 1,
                    "candidate_id": region_id,
                    "video_id": video_id,
                    "keyframe_uid": frame_uid,
                    "shot_id": frame.get("shot_id"),
                    "local_idx": frame.get("local_idx"),
                    "source_image": frame.get("source_image"),
                    "bbox_px": region.get("bbox_px"),
                    "easyocr_text": region.get("easyocr_text") or "",
                    "easyocr_confidence": region.get("easyocr_confidence"),
                    "router_v2_reasons": list(decision.reasons),
                }
            )

    vintern_by_id: dict[str, dict[str, Any]] = {}
    latencies: list[float] = []
    for row in vintern_rows:
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id:
            raise ValueError("Vintern result missing candidate_id")
        if candidate_id in vintern_by_id:
            raise ValueError(f"duplicate Vintern candidate_id: {candidate_id}")
        if candidate_id not in region_ids:
            raise ValueError(f"foreign Vintern candidate_id: {candidate_id}")
        vintern_by_id[candidate_id] = row
        if row.get("status") == "success" and isinstance(row.get("inference_seconds"), (int, float)):
            latencies.append(float(row["inference_seconds"]))

    new_candidate_ids = {row["candidate_id"] for row in candidate_rows}
    completed_selected = new_candidate_ids & vintern_by_id.keys()
    guard_reasons = Counter()
    accepted_vintern = 0
    guard_rejected_candidates = 0
    for candidate_id in completed_selected:
        row = vintern_by_id[candidate_id]
        rejected = vintern_output_rejection_reasons(
            easyocr_text=str(row.get("easyocr_text") or ""),
            vintern_text=str(row.get("vintern_text") or ""),
        )
        if rejected:
            guard_rejected_candidates += 1
            guard_reasons.update(rejected)
        elif row.get("status") == "success":
            accepted_vintern += 1

    candidate_fraction = len(candidate_rows) / max(1, total_regions)
    mean_latency = statistics.fmean(latencies) if latencies else None
    estimated_regions = round(len(candidate_rows) / max(1, len(easy_rows)) * args.full_catalog_frames)
    estimated_hours = (
        estimated_regions * mean_latency / 3600 if mean_latency is not None else None
    )
    decision = (
        "PASS_ROUTER_V2_DEV_ONLY"
        if candidate_fraction <= policy.max_candidate_fraction
        else "FAIL_ROUTER_V2_FANOUT"
    )
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "production_authorized": False,
        "accuracy_status": "AI_ASSISTED_VISUAL_REVIEW_NOT_HUMAN_GROUND_TRUTH",
        "inputs": {
            "easyocr_jsonl": {
                "path": str(args.easyocr_jsonl.resolve()),
                "sha256": _sha256(args.easyocr_jsonl),
                "frames": len(easy_rows),
            },
            "vintern_results_jsonl": {
                "path": str(args.vintern_results_jsonl.resolve()),
                "sha256": _sha256(args.vintern_results_jsonl),
                "records": len(vintern_rows),
            },
            "policy": {
                "path": str(args.policy.resolve()),
                "sha256": _sha256(args.policy),
                "value": policy.model_dump(mode="json"),
            },
        },
        "router": {
            "total_regions": total_regions,
            "old_candidates": len(old_candidates),
            "old_candidate_fraction": len(old_candidates) / max(1, total_regions),
            "v2_candidates": len(candidate_rows),
            "v2_candidate_fraction": candidate_fraction,
            "candidate_reduction": len(old_candidates) - len(candidate_rows),
            "candidate_reduction_fraction": 1 - len(candidate_rows) / max(1, len(old_candidates)),
            "reasons": dict(sorted(reasons.items())),
            "by_video": {
                video_id: {
                    "regions": counts["regions"],
                    "candidates": counts["candidates"],
                    "candidate_fraction": counts["candidates"] / max(1, counts["regions"]),
                }
                for video_id, counts in sorted(by_video.items())
            },
        },
        "existing_vintern_evidence": {
            "selected_candidates_with_result": len(completed_selected),
            "selected_candidates_without_result": len(new_candidate_ids - vintern_by_id.keys()),
            "guard_accepted_success": accepted_vintern,
            "guard_rejected_candidates": guard_rejected_candidates,
            "guard_reasons": dict(sorted(guard_reasons.items())),
            "observed_success_latency_seconds_mean": mean_latency,
        },
        "production_extrapolation_single_t4": {
            "full_catalog_frames": args.full_catalog_frames,
            "estimated_vintern_regions": estimated_regions,
            "estimated_vintern_hours": estimated_hours,
            "warning": "Dev-subset extrapolation only; excludes EasyOCR, I/O, retries and Gemini.",
        },
        "next_gate": {
            "allowed": decision == "PASS_ROUTER_V2_DEV_ONLY",
            "scope": "small end-to-end dev pilot with output guards",
            "requires_before_production": "human ground-truth accuracy plus terminal artifact validation",
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.output_candidates_jsonl is not None:
        args.output_candidates_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_candidates_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
            for row in candidate_rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({
        "decision": decision,
        "old_candidates": len(old_candidates),
        "v2_candidates": len(candidate_rows),
        "v2_candidate_fraction": candidate_fraction,
        "estimated_vintern_hours": estimated_hours,
        "output_json": str(args.output_json),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
