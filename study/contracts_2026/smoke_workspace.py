"""Workspace lifecycle primitives for the non-study T/N smoke wrappers."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import pathlib
import shutil
import tempfile
import time
import uuid
from collections.abc import Iterator


@contextlib.contextmanager
def exclusive_smoke_lock(attempt_root: pathlib.Path) -> Iterator[None]:
    """Serialize smoke runs without changing the campaign execution path."""
    attempt_root.mkdir(mode=0o711, parents=True, exist_ok=True)
    lock_path = attempt_root / ".smoke.lock"
    with lock_path.open("a+") as lock:
        os.chmod(lock_path, 0o600)
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another T/N smoke run is already active") from exc
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def allocate_smoke_attempt(attempt_root: pathlib.Path, prefix: str) -> pathlib.Path:
    """Allocate a unique attempt directory atomically under the trusted root."""
    attempt_root.mkdir(mode=0o711, parents=True, exist_ok=True)
    attempt = pathlib.Path(tempfile.mkdtemp(prefix=f"{prefix}-", dir=attempt_root))
    attempt.chmod(0o700)
    return attempt


def preserve_failure_evidence(
    attempt: pathlib.Path | None,
    state: pathlib.Path,
    prefix: str,
    error: BaseException,
) -> pathlib.Path | None:
    """Copy an unsuccessful disposable attempt before its finally cleanup."""
    if attempt is None or not attempt.exists():
        return None
    failure_root = state / "nonstudy_tests" / "failures"
    failure_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    destination = failure_root / f"{prefix}-{stamp}-{uuid.uuid4().hex[:12]}"
    shutil.copytree(attempt, destination)
    metadata = {
        "prefix": prefix,
        "original_attempt": str(attempt),
        "preserved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "error_type": type(error).__name__,
        "error": str(error),
    }
    metadata_path = destination / "failure.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    metadata_path.chmod(0o400)
    return destination
