"""Compare Gemini OCR models and shot-packaging strategies on a small canary."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from offline.ocr_cost_canary import (
    CANARY_STRATEGIES,
    CanaryStrategy,
    SyntheticRegion,
    build_crop_sheet_request_payload,
    build_request_payload,
    build_separate_payloads,
    expected_line_recall,
    make_synthetic_shot,
    parse_crop_sheet_results,
    parse_strict_results,
    validate_model_id,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sanitize(value: Any, api_key: str) -> str:
    return " ".join(str(value).replace(api_key, "[REDACTED]").split())[:500]


class PaceLimiter:
    def __init__(self, rpm: float) -> None:
        self.interval = 60.0 / rpm
        self.next_attempt = time.monotonic()

    def wait(self) -> None:
        delay = self.next_attempt - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self.next_attempt = time.monotonic() + self.interval


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def call_model(
    *,
    model_id: str,
    request_body: bytes,
    api_key: str,
    timeout: float,
    limiter: PaceLimiter,
) -> tuple[int | None, dict[str, Any], int]:
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{validate_model_id(model_id)}:generateContent"
    )
    limiter.wait()
    started = time.perf_counter()
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                endpoint,
                content=request_body,
                headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            )
        try:
            body: Any = response.json()
        except json.JSONDecodeError:
            body = {"raw": response.text}
        return (
            int(response.status_code),
            body,
            round((time.perf_counter() - started) * 1000),
        )
    except (httpx.HTTPError, OSError) as error:
        return None, {"transport_error": type(error).__name__, "message": str(error)}, round(
            (time.perf_counter() - started) * 1000
        )


def error_fields(body: Any, api_key: str) -> tuple[str, str]:
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        error = body["error"]
        return (
            sanitize(error.get("status", error.get("code", "HTTP_ERROR")), api_key),
            sanitize(error.get("message", "HTTP request failed"), api_key),
        )
    if isinstance(body, dict) and "transport_error" in body:
        return sanitize(body["transport_error"], api_key), sanitize(body.get("message"), api_key)
    return "HTTP_ERROR", sanitize(body, api_key)


def execute_payload(
    *,
    model_id: str,
    strategy: CanaryStrategy,
    shot_id: str,
    payload_index: int,
    body: bytes,
    request_ids: tuple[int, ...],
    regions: tuple[SyntheticRegion, ...] | None,
    api_key: str,
    timeout: float,
    limiter: PaceLimiter,
    log_path: Path,
) -> tuple[list[Any], dict[str, Any], bool]:
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        status, response, latency_ms = call_model(
            model_id=model_id,
            request_body=body,
            api_key=api_key,
            timeout=timeout,
            limiter=limiter,
        )
        base = {
            "timestamp_utc": utc_now(),
            "model_id": model_id,
            "strategy": strategy,
            "shot_id": shot_id,
            "payload_index": payload_index,
            "attempt": attempt,
            (
                "requested_keyframe_uids"
                if strategy == "crop_sheet"
                else "requested_frame_ids"
            ): list(request_ids),
            "http_status": status,
            "latency_ms": latency_ms,
        }
        if status == 200:
            try:
                if strategy == "crop_sheet":
                    if regions is None:
                        raise ValueError("crop_sheet request is missing detector regions")
                    results, usage = parse_crop_sheet_results(response, regions)
                else:
                    results, usage = parse_strict_results(response, request_ids)
                record = {
                    **base,
                    "outcome": "success",
                    "schema_valid": True,
                    "detected_text": {
                        str(result.frame_id): result.detected_text for result in results
                    },
                    **usage,
                    "error_code": None,
                    "error_message": None,
                }
                append_jsonl(log_path, record)
                return results, record, False
            except (KeyError, IndexError, TypeError, ValueError) as error:
                record = {
                    **base,
                    "outcome": "invalid_response",
                    "schema_valid": False,
                    "error_code": type(error).__name__,
                    "error_message": sanitize(error, api_key),
                }
                append_jsonl(log_path, record)
                return [], record, False

        code, message = error_fields(response, api_key)
        terminal_auth = status in {401, 403}
        retryable = status is None or status == 429 or (status is not None and status >= 500)
        record = {
            **base,
            "outcome": "retryable_error" if retryable else "terminal_error",
            "schema_valid": False,
            "error_code": code,
            "error_message": message,
        }
        append_jsonl(log_path, record)
        if terminal_auth:
            return [], record, True
        if retryable and attempt < max_attempts:
            time.sleep((2**attempt) + random.random())
            continue
        return [], record, False
    raise AssertionError("unreachable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "gemini-2.5-flash-lite",
            "gemini-3.1-flash-lite",
            "gemini-3.5-flash-lite",
        ],
    )
    parser.add_argument(
        "--strategies", nargs="+", choices=CANARY_STRATEGIES, default=list(CANARY_STRATEGIES)
    )
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--rpm", type=float, default=6.0)
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    args = parser.parse_args()
    if args.shots <= 0 or args.rpm <= 0 or args.timeout_seconds <= 0:
        parser.error("shots, rpm and timeout must be positive")
    args.models = [validate_model_id(model_id) for model_id in args.models]
    return args


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key.strip():
        print("ERROR: GEMINI_API_KEY is missing", flush=True)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / f"{args.run_id}.jsonl"
    summary_path = args.output_dir / f"{args.run_id}.summary.json"
    if log_path.exists() or summary_path.exists():
        print(f"ERROR: artifacts already exist for run_id={args.run_id}", flush=True)
        return 2

    limiter = PaceLimiter(args.rpm)
    aggregates: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "shots": 0,
            "requests": 0,
            "successful_requests": 0,
            "prompt_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "expected_lines": 0,
            "matched_lines": 0,
            "latencies_ms": [],
            "model_versions": set(),
        }
    )
    started_at = utc_now()
    halt = False
    for model_id in args.models:
        for strategy in args.strategies:
            for shot_index in range(1, args.shots + 1):
                shot = make_synthetic_shot(shot_index)
                if strategy == "separate":
                    payloads = [(*payload, None) for payload in build_separate_payloads(shot)]
                elif strategy == "crop_sheet":
                    payloads = [build_crop_sheet_request_payload(shot)]
                else:
                    body, request_ids = build_request_payload(shot, strategy)
                    payloads = [(body, request_ids, None)]
                all_results = []
                key = (model_id, strategy)
                aggregate = aggregates[key]
                aggregate["shots"] += 1
                for payload_index, (body, request_ids, regions) in enumerate(payloads, start=1):
                    results, record, halt = execute_payload(
                        model_id=model_id,
                        strategy=strategy,
                        shot_id=shot.shot_id,
                        payload_index=payload_index,
                        body=body,
                        request_ids=request_ids,
                        regions=regions,
                        api_key=api_key,
                        timeout=args.timeout_seconds,
                        limiter=limiter,
                        log_path=log_path,
                    )
                    aggregate["requests"] += 1
                    aggregate["latencies_ms"].append(record["latency_ms"])
                    if record["outcome"] == "success":
                        aggregate["successful_requests"] += 1
                        aggregate["prompt_tokens"] += int(record.get("prompt_tokens", 0))
                        aggregate["output_tokens"] += int(record.get("output_tokens", 0))
                        aggregate["total_tokens"] += int(record.get("total_tokens", 0))
                        if record.get("model_version"):
                            aggregate["model_versions"].add(record["model_version"])
                    all_results.extend(results)
                    if halt:
                        break
                recall = expected_line_recall(
                    shot,
                    all_results,
                    propagate_single_result=strategy == "middle_only" and len(all_results) == 1,
                )
                aggregate["expected_lines"] += recall["expected_lines"]
                aggregate["matched_lines"] += recall["matched_lines"]
                if halt:
                    break
            if halt:
                break
        if halt:
            break

    rows = []
    for (model_id, strategy), aggregate in sorted(aggregates.items()):
        latencies = sorted(aggregate.pop("latencies_ms"))
        versions = sorted(aggregate.pop("model_versions"))
        expected = aggregate["expected_lines"]
        rows.append(
            {
                "model_id": model_id,
                "strategy": strategy,
                **aggregate,
                "schema_valid_rate": (
                    aggregate["successful_requests"] / aggregate["requests"]
                    if aggregate["requests"]
                    else 0.0
                ),
                "line_recall": aggregate["matched_lines"] / expected if expected else 0.0,
                "latency_ms": {
                    "min": min(latencies) if latencies else None,
                    "max": max(latencies) if latencies else None,
                },
                "model_versions": versions,
            }
        )
    summary = {
        "schema_version": 1,
        "run_id": args.run_id,
        "input_kind": "synthetic_cost_schema_preflight",
        "production_recall_claimed": False,
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "rpm": args.rpm,
        "models": args.models,
        "strategies": args.strategies,
        "shots_per_configuration": args.shots,
        "halted_for_auth": halt,
        "results": rows,
        "secrets_logged": False,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 3 if halt else 0


if __name__ == "__main__":
    raise SystemExit(main())
