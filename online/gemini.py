"""Shared Gemini JSON client with conservative free-tier quota guards."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from datetime import date
from typing import Any


class GeminiQuotaManager:
    def __init__(self) -> None:
        self.rpm = int(os.environ.get("AIC_GEMINI_SAFE_RPM", "14"))
        self.tpm = int(os.environ.get("AIC_GEMINI_SAFE_TPM", "225000"))
        self.rpd = int(os.environ.get("AIC_GEMINI_SAFE_RPD", "450"))
        self._requests: deque[float] = deque()
        self._tokens: deque[tuple[float, int]] = deque()
        self._day = date.today()
        self._daily_requests = 0
        self._lock = threading.Lock()
        self._local = threading.local()

    def begin_search(self, *, max_calls: int = 8, timeout_seconds: float = 300.0) -> None:
        self._local.calls = 0
        self._local.max_calls = max_calls
        self._local.deadline = time.monotonic() + timeout_seconds

    def acquire(self, estimated_tokens: int) -> None:
        with self._lock:
            now = time.monotonic()
            if date.today() != self._day:
                self._day = date.today()
                self._daily_requests = 0
            calls = int(getattr(self._local, "calls", 0))
            max_calls = int(getattr(self._local, "max_calls", 8))
            if calls >= max_calls:
                raise RuntimeError("Gemini per-search call budget exhausted")
            deadline = float(getattr(self._local, "deadline", now + 300.0))
            if now >= deadline:
                raise RuntimeError("Gemini per-search deadline exceeded")
            while self._requests and now - self._requests[0] >= 60.0:
                self._requests.popleft()
            while self._tokens and now - self._tokens[0][0] >= 60.0:
                self._tokens.popleft()
            if self._daily_requests >= self.rpd:
                raise RuntimeError("Gemini safe daily request budget exhausted")
            token_total = sum(value for _, value in self._tokens)
            wait_for = 0.0
            if len(self._requests) >= self.rpm:
                wait_for = max(wait_for, 60.0 - (now - self._requests[0]))
            if token_total + estimated_tokens > self.tpm and self._tokens:
                wait_for = max(wait_for, 60.0 - (now - self._tokens[0][0]))
            if wait_for > 0:
                if wait_for > 30.0 or now + wait_for >= deadline:
                    raise RuntimeError("Gemini quota wait would exceed the online search budget")
                time.sleep(wait_for)
                now = time.monotonic()
            self._requests.append(now)
            self._tokens.append((now, max(1, estimated_tokens)))
            self._daily_requests += 1
            self._local.calls = calls + 1


_QUOTA = GeminiQuotaManager()


def get_gemini_quota_manager() -> GeminiQuotaManager:
    return _QUOTA


class GeminiJsonClient:
    def __init__(
        self,
        api_key: str,
        *,
        model: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.model = model or os.environ.get("AIC_GEMINI_MODEL", "gemini-3.5-flash-lite")
        self.timeout = timeout

    def generate(self, parts: list[dict[str, Any]], *, estimated_tokens: int = 4096) -> dict[str, Any]:
        _QUOTA.acquire(estimated_tokens)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        body = json.dumps(
            {
                "contents": [{"parts": parts}],
                "generationConfig": {
                    "temperature": 0.0,
                    "responseMimeType": "application/json",
                },
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"Gemini HTTP {error.code}: {detail}") from error
        text = payload["candidates"][0]["content"]["parts"][0]["text"]
        value = text.strip()
        if value.startswith("```"):
            value = value.strip("`").removeprefix("json").strip()
        start, end = value.find("{"), value.rfind("}")
        if start < 0 or end <= start:
            raise RuntimeError("Gemini did not return a JSON object")
        result = json.loads(value[start : end + 1])
        if not isinstance(result, dict):
            raise RuntimeError("Gemini JSON response must be an object")
        return result
