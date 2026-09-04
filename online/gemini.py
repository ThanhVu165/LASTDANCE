"""Shared Gemini JSON client with conservative free-tier quota guards."""

from __future__ import annotations

import json
import os
import random
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


_TRANSIENT_HTTP_CODES = {429, 500, 502, 503, 504}


def request_gemini_json(
    request: urllib.request.Request,
    *,
    timeout: float | None,
    estimated_tokens: int,
    opener: Any | None = None,
) -> dict[str, Any]:
    """Send one logical Gemini request with bounded transient-error retries."""
    max_attempts = max(1, int(os.environ.get("AIC_GEMINI_TRANSIENT_MAX_ATTEMPTS", "6")))
    base_delay = max(0.0, float(os.environ.get("AIC_GEMINI_RETRY_BASE_SECONDS", "1.5")))
    open_options = {} if timeout is None else {"timeout": timeout}
    open_call = opener or urllib.request.urlopen
    last_detail = ""
    for attempt in range(1, max_attempts + 1):
        _QUOTA.acquire(estimated_tokens)
        try:
            with open_call(request, **open_options) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise RuntimeError("Gemini HTTP response must be a JSON object")
            return payload
        except urllib.error.HTTPError as error:
            last_detail = error.read().decode("utf-8", errors="replace")[:1000]
            if error.code not in _TRANSIENT_HTTP_CODES or attempt >= max_attempts:
                raise RuntimeError(
                    f"Gemini HTTP {error.code} after {attempt} attempt(s): {last_detail}"
                ) from error
            retry_after = error.headers.get("Retry-After") if error.headers else None
            try:
                delay = float(retry_after) if retry_after else base_delay * (2 ** (attempt - 1))
            except ValueError:
                delay = base_delay * (2 ** (attempt - 1))
            delay = min(30.0, delay) + random.uniform(0.0, min(1.0, base_delay / 3.0))
            time.sleep(delay)
        except (TimeoutError, urllib.error.URLError, OSError) as error:
            last_detail = str(error)
            if attempt >= max_attempts:
                raise RuntimeError(
                    f"Gemini transport failed after {attempt} attempt(s): {last_detail}"
                ) from error
            delay = min(30.0, base_delay * (2 ** (attempt - 1)))
            time.sleep(delay + random.uniform(0.0, min(1.0, base_delay / 3.0)))
    raise RuntimeError(f"Gemini request failed: {last_detail}")


class GeminiJsonClient:
    def __init__(
        self,
        api_key: str,
        *,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model or os.environ.get("AIC_GEMINI_MODEL", "gemini-3.5-flash-lite")
        raw_timeout = timeout if timeout is not None else os.environ.get(
            "AIC_GEMINI_REQUEST_TIMEOUT_SECONDS", "20"
        )
        normalized_timeout = str(raw_timeout).strip().casefold()
        self.timeout = None if normalized_timeout in {"0", "none", "off"} else float(raw_timeout)
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("Gemini request timeout must be positive or 0 to disable")

    def generate(
        self,
        parts: list[dict[str, Any]],
        *,
        estimated_tokens: int = 4096,
        max_output_tokens: int = 512,
    ) -> dict[str, Any]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        body = json.dumps(
            {
                "contents": [{"parts": parts}],
                "generationConfig": {
                    "temperature": 0.0,
                    "responseMimeType": "application/json",
                    "maxOutputTokens": max_output_tokens,
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
        payload = request_gemini_json(
            request,
            timeout=self.timeout,
            estimated_tokens=estimated_tokens,
        )
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
