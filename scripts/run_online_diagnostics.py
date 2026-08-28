"""Run a compact KIS/QA/TRAKE diagnostic suite from the qualifier query archive."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import zipfile
from pathlib import Path

from online.config import OnlineLayout
from online.engine import OnlineEngine
from online.planners import RuleBasedQueryPlanner
from shared.schemas.online import SearchRequest, TaskType, TrakeCandidate


DEFAULT_CASES = [
    "query-p1-1-kis.txt",
    "query-p1-18-kis.txt",
    "query-p1-3-qa.txt",
    "query-p1-15-qa.txt",
    "query-p1-16-trake.txt",
]


def _task(name: str) -> TaskType:
    suffix = Path(name).stem.rsplit("-", 1)[-1].upper()
    return TaskType(suffix)


def _read_cases(path: Path) -> list[tuple[str, TaskType, str]]:
    cases: list[tuple[str, TaskType, str]] = []
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        for name in DEFAULT_CASES:
            if name not in names:
                raise RuntimeError(f"diagnostic query is absent: {name}")
            cases.append((name, _task(name), archive.read(name).decode("utf-8-sig").strip()))
    # The supplied round-1 set contains one TRAKE query. Add a clearly labelled two-event
    # derivative of that same query for beam/order regression coverage, not as an eval item.
    trake = next(case for case in cases if case[1] == TaskType.TRAKE)
    moments = RuleBasedQueryPlanner().plan(trake[2], TaskType.TRAKE).ordered_moments
    if len(moments) >= 2:
        derived = "\n".join(f"E{index + 1} {moment}" for index, moment in enumerate(moments[:2]))
        cases.append(("query-p1-16-trake-derived-e1-e2", TaskType.TRAKE, derived))
    return cases


def _candidate_row(candidate: object) -> dict[str, object]:
    if isinstance(candidate, TrakeCandidate):
        return {
            "video_id": candidate.video_id,
            "frame_ids": candidate.frame_ids,
            "score": candidate.score,
            "channel_scores": [
                {
                    "clip": item.score_clip,
                    "siglip": item.score_siglip,
                    "eva": item.score_eva,
                    "visual": item.score_visual,
                    "final": item.final_score,
                }
                for item in candidate.evidence
            ],
        }
    evidence = candidate.evidence
    row: dict[str, object] = {
        "video_id": candidate.video_id,
        "frame_id": candidate.frame_id,
        "score": candidate.score,
        "channel_scores": {
            "clip": evidence.score_clip,
            "siglip": evidence.score_siglip,
            "eva": evidence.score_eva,
            "visual": evidence.score_visual,
            "ocr": evidence.score_ocr,
            "asr": evidence.score_asr,
            "neighbor": evidence.neighbor_support,
            "final": evidence.final_score,
        },
    }
    if hasattr(candidate, "answer"):
        row["answer"] = candidate.answer
        row["confidence"] = candidate.confidence
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-results", type=int, default=10)
    args = parser.parse_args()
    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)
    layout = OnlineLayout.from_environment()
    output = args.output or (layout.data.root / "diagnostics" / "online-qualifier-round1.json")
    engine = OnlineEngine.from_environment()
    reports = []
    for name, task_type, query in _read_cases(args.archive):
        try:
            run = engine.search(
                SearchRequest(task_type=task_type, raw_query=query, max_results=args.max_results)
            )
            reports.append(
                {
                    "case": name,
                    "task_type": task_type.value,
                    "status": "PASS",
                    "planner": run.query_plan.model_dump(mode="json"),
                    "top_videos": [
                        {"video_id": item.video_id, "score": item.video_score, "coverage": item.coverage}
                        for item in run.video_hypotheses[:5]
                    ],
                    "candidates": [_candidate_row(item) for item in run.task_candidates],
                    "warnings": run.warnings,
                    "timings_ms": run.timings_ms,
                    "provenance": run.provenance,
                }
            )
        except Exception as error:
            reports.append(
                {
                    "case": name,
                    "task_type": task_type.value,
                    "status": "FAIL",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "source_archive": args.archive.name, "reports": reports}
    fd, temporary = tempfile.mkstemp(prefix="online-diagnostic-", suffix=".json", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(output.resolve())
    return 0 if all(item["status"] == "PASS" for item in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
