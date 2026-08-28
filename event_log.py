#!/usr/bin/env python3
"""Small append-only JSONL event transport shared by live runs and replay."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import uuid
from typing import Any


_lock = threading.Lock()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def make_event(
    event_type: str,
    *,
    campaign_id: str | None = None,
    experiment_id: int | None = None,
    training_run_id: int | None = None,
    status: str | None = None,
    elapsed_seconds: float | None = None,
    payload: dict[str, Any] | None = None,
    source: str = "live",
) -> dict[str, Any]:
    return {
        "event_uuid": str(uuid.uuid4()),
        "event_type": event_type,
        "campaign_id": campaign_id,
        "experiment_id": experiment_id,
        "training_run_id": training_run_id,
        "status": status,
        "event_time": timestamp(),
        "elapsed_seconds": elapsed_seconds,
        "payload": payload or {},
        "source": source,
    }


def append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, separators=(",", ":"), sort_keys=True) + "\n"
    with _lock, path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events
