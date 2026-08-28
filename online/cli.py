"""Diagnostics for preflight and non-interactive search."""

from __future__ import annotations

import argparse
import json

from shared.schemas.online import SearchRequest, TaskType

from .artifacts import ArtifactRegistry
from .engine import OnlineEngine


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LASTDANCE Online diagnostics")
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight", help="validate production artifacts")
    preflight.add_argument("--deep", action="store_true", help="hash all indexes and compare all UIDs")
    search = subparsers.add_parser("search", help="run one query and print SearchRun JSON")
    search.add_argument("--task", choices=[item.value for item in TaskType], required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--max-results", type=int, default=100)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "preflight":
        registry = ArtifactRegistry.load(deep=args.deep)
        print(json.dumps({key: value.model_dump(mode="json") for key, value in registry.statuses.items()}, indent=2))
        return 0
    engine = OnlineEngine.from_environment()
    run = engine.search(
        SearchRequest(task_type=TaskType(args.task), raw_query=args.query, max_results=args.max_results)
    )
    print(run.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
