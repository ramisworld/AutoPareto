#!/usr/bin/env python3
"""Requeue a campaign position whose Scientist call timed out before any candidate or A40 work existed.

Covers one case only. The controller paused the position as ``reconciliation_required`` with a
``ScientistExecutionTimeout`` in phase ``pre_remote_execution``. No pending execution record, pending
candidate or result exists for it, and the failed invocation's own diagnostics record the bounded timeout.

This matches the Codex comparator's handling of the same event: position 12, ``operational_amendments[1]`` of
``final_pair_a_continuation_manifest_20260916_r2_requeue031.json``. That record says to "requeue the same
unconsumed scheduled position without changing scientific components or timeout".

The position goes back to ``pending`` and the controller runs it again with fresh Scientist calls. The timed-out
call's diagnostics stay where they are. A read-only copy of the controller state is kept, and the failure is
recorded inside the position (``requeued_scientist_timeouts``). Nothing else changes: not the timeout, the
manifest or the code.

    python3 -B scripts/requeue_scientist_timeout.py --state-root DIR --global-position N
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import sys
import tempfile
from typing import Any


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write_bytes(path: pathlib.Path, data: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    except BaseException:
        pathlib.Path(temporary).unlink(missing_ok=True)
        raise


def requeue_scientist_timeout(*, state_root: pathlib.Path, global_position: int) -> dict[str, Any]:
    state_root = state_root.resolve(strict=True)
    state_path = state_root / "controller_state.json"
    before = state_path.read_bytes()
    state = json.loads(before)
    position = state.get("positions", {}).get(str(global_position))
    if not isinstance(position, dict) or position.get("status") != "reconciliation_required":
        raise RuntimeError(f"position {global_position} is not reconciliation_required")
    failure = position.get("reconciliation_failure")
    if not isinstance(failure, dict) or failure.get("type") != "ScientistExecutionTimeout":
        raise RuntimeError("only a ScientistExecutionTimeout can be requeued by this tool")
    if failure.get("phase") != "pre_remote_execution":
        raise RuntimeError("the timeout did not happen before remote execution")
    if "execution" in position or position.get("result_file") is not None:
        raise RuntimeError("position already has a pending execution or a result; this is not a pre-candidate timeout")
    campaign_id = str(position.get("campaign_id", ""))
    if (state_root / campaign_id / "pending" / f"position-{global_position:03d}").exists():
        raise RuntimeError("a pending candidate directory exists for this position")

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

    match = re.search(r"diagnostics=(\S+invocation\.json)", str(failure.get("message", "")))
    if match is None:
        raise RuntimeError("the timeout record does not name its Scientist diagnostics")
    diagnostics_path = pathlib.Path(match.group(1)).resolve(strict=True)
    if not diagnostics_path.is_relative_to(state_root / campaign_id / "scientist_diagnostics"):
        raise RuntimeError("the Scientist diagnostics are outside this campaign")
    diagnostics = json.loads(diagnostics_path.read_text())
    if diagnostics.get("status") != "FAILED" or "bounded timeout" not in str(diagnostics.get("failure", "")):
        raise RuntimeError("the Scientist diagnostics do not record a bounded timeout")
    if int((diagnostics.get("metadata") or {}).get("attempt_number", -1)) != int(position.get("campaign_attempt", -2)):
        raise RuntimeError("the Scientist diagnostics belong to another attempt")

    now = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = state_root / f"controller_state.json.pre_requeue_scientist_timeout_position_{global_position:03d}_{now}.json"
    with open(backup, "xb") as handle:  # never overwrite an earlier copy
        handle.write(before)
    backup.chmod(0o444)
    record = {
        **failure,
        "requeued_utc": now,
        "requeued_by": "scripts/requeue_scientist_timeout.py",
        "pre_requeue_state_copy": backup.name,
        "pre_requeue_state_sha256": hashlib.sha256(before).hexdigest(),
        "scientist_diagnostics": str(diagnostics_path),
        "scientist_diagnostics_sha256": sha256_file(diagnostics_path),
        "phase_timed_out": (diagnostics.get("metadata") or {}).get("phase"),
        "timeout_seconds": diagnostics.get("timeout_seconds"),
        "candidate_existed": False,
        "a40_work_existed": False,
        "scientific_position_consumed": False,
        "timeout_changed": False,
        "precedent": "Codex comparator position 12, operational_amendments[1] of the requeue031 manifest",
    }
    position.setdefault("requeued_scientist_timeouts", []).append(record)
    del position["reconciliation_failure"]
    position["status"] = "pending"
    _atomic_write_bytes(state_path, (json.dumps(state, indent=2, sort_keys=True) + "\n").encode())
    return {"status": "requeued", "global_position": global_position, "record": record,
            "controller_state_sha256": sha256_file(state_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state-root", type=pathlib.Path, required=True)
    parser.add_argument("--global-position", type=int, required=True)
    args = parser.parse_args()
    result = requeue_scientist_timeout(state_root=args.state_root, global_position=args.global_position)
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
