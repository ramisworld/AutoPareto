"""Trusted-parent / untrusted-model-host inference IPC.

The evaluator process owns this client, the benchmark inputs, all clocks, the
correctness comparison, stability gates, and Pareto arithmetic.  The model host
is a disposable child process that owns only candidate model execution and
returns typed observations over an anonymous Unix socketpair.

This is a process-integrity boundary, not confidential-compute hardware.  The
model host must receive the benchmark tensors to execute them, so a malicious
candidate can observe those tensors in its own address space.  It cannot access
the evaluator's Python frames, globals, acceptance state, or campaign history.
CUDA allocator values are host-process observations because CUDA memory is
owned by the model-host process; they are retained as a documented limitation
until a separate trusted GPU accounting mechanism exists.
"""

from __future__ import annotations

import json
import socket
import subprocess
import time
from typing import Any, Mapping, Sequence


IPC_PROTOCOL = "autopareto_inference_ipc_v1"
MAX_MESSAGE_BYTES = 64 * 1024 * 1024


class InferenceIPCError(RuntimeError):
    """The model host violated the typed IPC contract or terminated."""


def _json_line(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InferenceIPCError(f"{name} must be a non-negative integer")
    return value


def validate_logits_payload(value: Any, *, expected_vocab: int | None = None) -> list[float]:
    if not isinstance(value, list) or not value:
        raise InferenceIPCError("model host returned no logits")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise InferenceIPCError("model host logits are not numeric")
        number = float(item)
        if number != number or number in (float("inf"), float("-inf")):
            raise InferenceIPCError("model host returned non-finite logits")
        result.append(number)
    if expected_vocab is not None and len(result) != expected_vocab:
        raise InferenceIPCError(
            f"model host returned vocabulary size {len(result)}, expected {expected_vocab}"
        )
    return result


def greedy_token(logits: Sequence[float]) -> int:
    values = validate_logits_payload(list(logits))
    return max(range(len(values)), key=values.__getitem__)


class InferenceHostClient:
    """Synchronous parent-side client for one disposable model host."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: str,
        env: Mapping[str, str],
    ) -> None:
        parent_socket, child_socket = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        child_socket.set_inheritable(True)
        child_fd = child_socket.fileno()
        self._socket = parent_socket
        self._socket.settimeout(300.0)
        self._buffer = bytearray()
        self._request_id = 0
        self._closed = False
        self._process = subprocess.Popen(
            list(command) + ["--ipc-fd", str(child_fd)],
            cwd=cwd,
            env=dict(env),
            pass_fds=(child_fd,),
            close_fds=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=False,
        )
        # The wrapper/host receives the inherited descriptor number.  The
        # socket object itself is intentionally not shared with candidate code.
        child_socket.close()

    @property
    def process(self) -> subprocess.Popen:
        return self._process

    def _read_line(self) -> dict[str, Any]:
        while b"\n" not in self._buffer:
            chunk = self._socket.recv(1024 * 1024)
            if not chunk:
                stderr = b""
                if self._process.stderr is not None:
                    try:
                        stderr = self._process.stderr.read(4096)
                    except OSError:
                        pass
                raise InferenceIPCError(
                    f"inference model host exited without a response; "
                    f"returncode={self._process.poll()} stderr={stderr.decode(errors='replace')}"
                )
            self._buffer.extend(chunk)
            if len(self._buffer) > MAX_MESSAGE_BYTES:
                raise InferenceIPCError("model host IPC message exceeds the protected limit")
        line, _, remainder = self._buffer.partition(b"\n")
        self._buffer = bytearray(remainder)
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InferenceIPCError("model host returned malformed JSON") from exc
        if not isinstance(payload, dict) or payload.get("protocol") != IPC_PROTOCOL:
            raise InferenceIPCError("model host IPC protocol mismatch")
        return payload

    def request(self, operation: str, **payload: Any) -> dict[str, Any]:
        if self._closed:
            raise InferenceIPCError("inference model host client is closed")
        self._request_id += 1
        request = {
            "protocol": IPC_PROTOCOL,
            "request_id": self._request_id,
            "operation": operation,
            **payload,
        }
        encoded = _json_line(request)
        if len(encoded) > MAX_MESSAGE_BYTES:
            raise InferenceIPCError("parent IPC request exceeds the protected limit")
        try:
            self._socket.sendall(encoded)
            response = self._read_line()
        except (OSError, TimeoutError) as exc:
            stderr = b""
            returncode = self._process.poll()
            if returncode is None:
                try:
                    returncode = self._process.wait(timeout=0.25)
                except subprocess.TimeoutExpired:
                    returncode = self._process.poll()
            if returncode is not None and self._process.stderr is not None:
                try:
                    stderr = self._process.stderr.read(4096)
                except OSError:
                    pass
            detail = stderr.decode(errors="replace").strip()
            suffix = f" stderr={detail}" if detail else ""
            raise InferenceIPCError(
                f"inference model host IPC failed: {exc}; "
                f"returncode={returncode}{suffix}"
            ) from exc
        if response.get("request_id") != self._request_id:
            raise InferenceIPCError("model host response request id is out of order")
        if response.get("ok") is not True:
            raise InferenceIPCError(str(response.get("error", "model host rejected request")))
        return response

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._socket.sendall(_json_line({
                "protocol": IPC_PROTOCOL,
                "request_id": self._request_id + 1,
                "operation": "shutdown",
            }))
        except OSError:
            pass
        try:
            self._socket.close()
        except OSError:
            pass
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=10)
        if self._process.stderr is not None:
            self._process.stderr.close()

    def __enter__(self) -> "InferenceHostClient":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()


def monotonic_request(client: InferenceHostClient, operation: str, **payload: Any) -> tuple[dict[str, Any], float]:
    """Time a complete host operation using the trusted parent's monotonic clock."""
    started = time.perf_counter_ns()
    response = client.request(operation, **payload)
    ended = time.perf_counter_ns()
    return response, (ended - started) / 1_000_000_000.0
