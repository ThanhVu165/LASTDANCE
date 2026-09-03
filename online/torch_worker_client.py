"""JSON-lines client for the isolated Torch model process.

Windows pip wheels for PyTorch and FAISS load incompatible OpenMP runtimes.
Running all Torch inference in one child keeps FAISS in the Online process and
avoids the unsafe KMP_DUPLICATE_LIB_OK workaround.
"""

from __future__ import annotations

import atexit
import json
import os
import queue
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Any


class TorchWorkerClient:
    def __init__(self, *, timeout: float | None = None) -> None:
        self.timeout = timeout or float(os.environ.get("AIC_TORCH_WORKER_TIMEOUT", "600"))
        self._process: subprocess.Popen[str] | None = None
        self._responses: queue.Queue[str] = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=80)
        self._lock = threading.RLock()
        self._next_id = 1

    def _start(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        root = Path(__file__).resolve().parents[1]
        self._process = subprocess.Popen(
            [sys.executable, "-u", "-m", "online.torch_worker"],
            cwd=root,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        threading.Thread(target=self._read_stdout, args=(self._process,), daemon=True).start()
        threading.Thread(target=self._read_stderr, args=(self._process,), daemon=True).start()
        return self._process

    def _read_stdout(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            self._responses.put(line)

    def _read_stderr(self, process: subprocess.Popen[str]) -> None:
        assert process.stderr is not None
        for line in process.stderr:
            self._stderr.append(line.rstrip())

    def request(self, operation: str, **payload: Any) -> dict[str, Any]:
        with self._lock:
            process = self._start()
            request_id = self._next_id
            self._next_id += 1
            assert process.stdin is not None
            message = {"id": request_id, "operation": operation, **payload}
            try:
                process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
                process.stdin.flush()
                line = self._responses.get(timeout=self.timeout)
            except (BrokenPipeError, OSError, queue.Empty) as error:
                detail = "\n".join(self._stderr)
                self.close()
                raise RuntimeError(f"Torch worker failed or timed out: {error}; stderr={detail}") from error
            try:
                response = json.loads(line)
            except json.JSONDecodeError as error:
                detail = "\n".join(self._stderr)
                self.close()
                raise RuntimeError(f"Torch worker protocol returned invalid JSON: {line!r}; stderr={detail}") from error
            if response.get("id") != request_id:
                self.close()
                raise RuntimeError("Torch worker response id is out of sequence")
            if not response.get("ok"):
                raise RuntimeError(str(response.get("error", "Torch worker request failed")))
            result = response.get("result")
            if not isinstance(result, dict):
                raise RuntimeError("Torch worker result must be an object")
            return result

    def close(self) -> None:
        with self._lock:
            process, self._process = self._process, None
            if process is None or process.poll() is not None:
                return
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


_CLIENT: TorchWorkerClient | None = None
_CLIENT_LOCK = threading.Lock()


def get_torch_worker_client() -> TorchWorkerClient:
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            _CLIENT = TorchWorkerClient()
            atexit.register(_CLIENT.close)
        return _CLIENT
