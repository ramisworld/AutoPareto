#!/usr/bin/env python3
"""Requeue a campaign position whose payload transfer failed before it reached the A40.

Covers one case only. The controller paused the position as ``reconciliation_required`` in phase
``payload_transfer``; its execution is still ``prepared`` (no remote payload directory was ever recorded);
and an operator check on the A40 found no payload, invocation receipt or candidate process for it.

The Scientist's prepared candidate and metadata are kept byte for byte: no new agent call, no new
proposal. A read-only copy of the controller state is kept. The transport failure stays recorded inside
the position (``requeued_transport_failures``), and the position goes back to ``running``/``prepared``.
Then ``study.contracts_2026.continuation.arm_continuation_payload`` transfers, verifies and arms the
payload, and the controller resumes it like any armed position.

    python3 -B scripts/requeue_undelivered_payload.py --state-root DIR --global-position N \
        --absence-evidence FILE.json

The absence evidence is a JSON object with schema ``autopareto_undelivered_payload_absence_v1``:
``remote_run_id`` and ``invocation_id`` (both must match the execution), ``payload_dir_present``,
``invocation_receipt_present`` (both false), ``matching_processes`` (empty) and ``checked_utc``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import sys
import tempfile
from typing import Any

EVIDENCE_SCHEMA = "autopareto_undelivered_payload_absence_v1"
EXECUTION_SCHEMA = "autopareto_pending_execution_v1"


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write_bytes(path: pathlib.Path, data: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        pathlib.Path(temporary).unlink(missing_ok=True)
        raise


def requeue_undelivered_payload(
    *, state_root: pathlib.Path, global_position: int, absence_evidence_path: pathlib.Path
) -> dict[str, Any]:
    state_root = state_root.resolve(strict=True)
    state_path = state_root / "controller_state.json"
    before = state_path.read_bytes()
    state = json.loads(before)
    position = state.get("positions", {}).get(str(global_position))
    if not isinstance(position, dict) or position.get("status") != "reconciliation_required":
        raise RuntimeError(f"position {global_position} is not reconciliation_required")
    failure = position.get("reconciliation_failure")
    if not isinstance(failure, dict) or failure.get("phase") != "payload_transfer":
        raise RuntimeError("only a failure in phase payload_transfer can be requeued by this tool")
    if position.get("result_file") is not None:
        raise RuntimeError("position has a local result and cannot be requeued")
    execution = position.get("execution")
    if not isinstance(execution, dict) or execution.get("schema") != EXECUTION_SCHEMA:
        raise RuntimeError("position has no durable pending execution record")
    if execution.get("dispatch_status") != "prepared" or execution.get("remote_payload_dir"):
        raise RuntimeError("execution is not an undelivered prepared payload")

    # Every earlier studied position is completed and every later one is pending, so the
    # controller will resume exactly this position next.
    inactive = {"not_in_study"} if state.get("active_contracts") else set()
    for row in state.get("schedule", []):
        other = int(row["global_position"])
        status = state["positions"][str(other)].get("status")
        if status in inactive or other == global_position:
            continue
        if other < global_position and status != "completed":
            raise RuntimeError(f"earlier position {other} is {status!r}, not completed")
        if other > global_position and status != "pending":
            raise RuntimeError(f"later position {other} is {status!r}, not pending")

    candidate = pathlib.Path(str(execution.get("candidate_path", "")))
    metadata = pathlib.Path(str(execution.get("metadata_path", "")))
    if sha256_file(candidate) != execution.get("candidate_source_after_sha256"):
        raise RuntimeError("prepared candidate hash does not match the execution record")
    if sha256_file(metadata) != execution.get("metadata_sha256"):
        raise RuntimeError("prepared metadata hash does not match the execution record")

    evidence = json.loads(absence_evidence_path.read_text())
    if evidence.get("schema") != EVIDENCE_SCHEMA:
        raise RuntimeError("absence evidence has an unexpected schema")
    if evidence.get("remote_run_id") != execution.get("remote_run_id"):
        raise RuntimeError("absence evidence was taken for a different remote_run_id")
    if evidence.get("invocation_id") != execution.get("invocation_id"):
        raise RuntimeError("absence evidence was taken for a different invocation_id")
    if evidence.get("payload_dir_present") is not False or evidence.get("invocation_receipt_present") is not False:
        raise RuntimeError("absence evidence does not prove that nothing reached the A40")
    if evidence.get("matching_processes") != []:
        raise RuntimeError("absence evidence lists a matching remote process")

    now = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = state_root / f"controller_state.json.pre_requeue_position_{global_position:03d}_{now}.json"
    with open(backup, "xb") as handle:  # never overwrite an earlier copy
        handle.write(before)
    backup.chmod(0o444)

    record = {
        **failure,
        "requeued_utc": now,
        "requeued_by": "scripts/requeue_undelivered_payload.py",
        "pre_requeue_state_copy": backup.name,
        "pre_requeue_state_sha256": hashlib.sha256(before).hexdigest(),
        "absence_evidence_path": str(absence_evidence_path.resolve()),
        "absence_evidence_sha256": sha256_file(absence_evidence_path),
        "absence_checked_utc": evidence.get("checked_utc"),
        "candidate_source_after_sha256": execution["candidate_source_after_sha256"],
        "scientific_position_consumed": False,
        "new_agent_call": False,
    }
    position.setdefault("requeued_transport_failures", []).append(record)
    del position["reconciliation_failure"]
    position["status"] = "running"
    _atomic_write_bytes(state_path, (json.dumps(state, indent=2, sort_keys=True) + "\n").encode())
    return {"status": "requeued", "global_position": global_position, "record": record,
            "controller_state_sha256": sha256_file(state_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state-root", type=pathlib.Path, required=True)
    parser.add_argument("--global-position", type=int, required=True)
    parser.add_argument("--absence-evidence", type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = requeue_undelivered_payload(
        state_root=args.state_root, global_position=args.global_position,
        absence_evidence_path=args.absence_evidence,
    )
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
