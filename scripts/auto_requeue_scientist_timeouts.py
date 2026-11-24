#!/usr/bin/env python3
"""Run the supervised campaign and requeue clean pre-candidate Scientist timeouts without waiting for the PI.

PI decision, 2026-11-06 (NZDT): "Auto-requeue at 900 s". The rule this script carries out:

* After the supervisor exits, it looks at the one position the controller paused. It acts only when the supervisor
  classified the exit ``paused_reconciliation_required`` and that position failed with a
  ``ScientistExecutionTimeout`` in phase ``pre_remote_execution``. In that case it calls
  ``requeue_scientist_timeout.requeue_scientist_timeout``, the procedure already used by hand (doc 20 sections 13
  and 18). That procedure refuses any position with a pending candidate, an execution record, a result, or
  diagnostics that do not record the bounded timeout.
* The third timeout of the same position is not requeued. Earlier requeues of that position count, including
  manual ones. The script stops there and leaves the position paused for the PI.
* Any other exit also stops the script and is left for the PI. That includes a completed campaign, the extension
  cap, a usage pause, a controller failure or any other reconciliation.
* It never changes the timeout, the manifest, the code or the command. It relaunches the identical supervisor
  command.

Each requeue is appended to ``<supervisor-dir>/auto_requeue.jsonl``. The requeue record itself stays inside the
position (``requeued_scientist_timeouts``, ``requeued_by`` names this script), together with a read-only copy of
the state taken before it.

    python3 -B scripts/auto_requeue_scientist_timeouts.py --state-root DIR --supervisor-dir DIR \\
        [--max-timeouts-per-position 3] [--max-total-requeues 12] -- <supervisor command ...>

Exit status: 0 when the campaign completed, 4 when it stopped for the PI. The last supervisor classification is
printed and logged in both cases.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import subprocess
import sys
from typing import Any, Callable, Sequence

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import requeue_scientist_timeout  # noqa: E402

LOG = "auto_requeue.jsonl"
EVENTS = "events.jsonl"
STOPPED_FOR_PI_EXIT_STATUS = 4
DEFAULT_MAX_TIMEOUTS_PER_POSITION = 3
DEFAULT_MAX_TOTAL_REQUEUES = 12
APPROVAL = "PI decision 2026-11-06 (NZDT): \"Auto-requeue at 900 s\"; stop at the third timeout of one attempt"


def _utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _append(path: pathlib.Path, record: dict[str, Any]) -> None:
    with open(path, "a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def _line_count(path: pathlib.Path) -> int:
    if not path.exists():
        return 0
    with open(path, "rb") as handle:
        return sum(1 for _ in handle)


def last_supervisor_exit(events_path: pathlib.Path, after_line: int) -> dict[str, Any] | None:
    """The last ``supervisor_exit`` event written after line ``after_line`` (1-based count already seen)."""
    found = None
    if not events_path.exists():
        return None
    with open(events_path) as handle:
        for number, line in enumerate(handle, 1):
            if number <= after_line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "supervisor_exit":
                found = event
    return found


def paused_timeout_position(state_root: pathlib.Path) -> tuple[int, dict[str, Any]] | None:
    """The single paused position, if it is a pre-candidate Scientist timeout; otherwise None."""
    state = json.loads((state_root / "controller_state.json").read_text())
    paused = [(int(key), value) for key, value in state.get("positions", {}).items()
              if isinstance(value, dict) and value.get("status") == "reconciliation_required"]
    if len(paused) != 1:
        return None
    number, position = paused[0]
    failure = position.get("reconciliation_failure") or {}
    if failure.get("type") != "ScientistExecutionTimeout" or failure.get("phase") != "pre_remote_execution":
        return None
    return number, position


def run(*, state_root: pathlib.Path, supervisor_dir: pathlib.Path, command: Sequence[str],
        max_timeouts_per_position: int = DEFAULT_MAX_TIMEOUTS_PER_POSITION,
        max_total_requeues: int = DEFAULT_MAX_TOTAL_REQUEUES,
        requeue: Callable[..., dict[str, Any]] = requeue_scientist_timeout.requeue_scientist_timeout,
        ) -> tuple[int, dict[str, Any]]:
    if not command:
        raise ValueError("no supervisor command given")
    if max_timeouts_per_position < 2:
        raise ValueError("max_timeouts_per_position must allow at least one requeue")
    state_root = state_root.resolve(strict=True)
    supervisor_dir.mkdir(parents=True, exist_ok=True)
    log_path = supervisor_dir / LOG
    events_path = supervisor_dir / EVENTS
    requeues = 0
    while True:
        seen = _line_count(events_path)
        _append(log_path, {"event": "supervisor_start", "utc": _utc(), "command": list(command),
                           "requeues_so_far": requeues})
        returncode = subprocess.call(list(command), stdin=subprocess.DEVNULL)
        exit_event = last_supervisor_exit(events_path, seen) or {}
        classification = exit_event.get("classification")
        summary = {"event": "supervisor_finished", "utc": _utc(), "returncode": returncode,
                   "classification": classification, "requeues_so_far": requeues}
        if classification == "campaign_completed":
            _append(log_path, {**summary, "decision": "stop_completed"})
            return 0, summary
        stop_reason = None
        target = None
        if classification != "paused_reconciliation_required":
            stop_reason = "the supervisor exit is not a reconciliation pause"
        else:
            target = paused_timeout_position(state_root)
            if target is None:
                stop_reason = "the paused position is not a pre-candidate Scientist timeout"
            else:
                earlier = len(target[1].get("requeued_scientist_timeouts") or [])
                if earlier + 1 >= max_timeouts_per_position:
                    stop_reason = (f"position {target[0]} timed out {earlier + 1} times; the PI decides from "
                                   f"timeout {max_timeouts_per_position}")
                elif requeues >= max_total_requeues:
                    stop_reason = f"the cap of {max_total_requeues} automatic requeues in one run is reached"
        if stop_reason is not None:
            summary = {**summary, "decision": "stop_for_pi", "reason": stop_reason,
                       "global_position": target[0] if target else None}
            _append(log_path, summary)
            return STOPPED_FOR_PI_EXIT_STATUS, summary
        number, position = target
        result = requeue(state_root=state_root, global_position=number)
        record = result["record"]
        record_note = {"requeued_by_wrapper": "scripts/auto_requeue_scientist_timeouts.py", "approval": APPROVAL}
        _annotate_last_requeue(state_root, number, record_note)
        requeues += 1
        _append(log_path, {**summary, "decision": "requeued", "global_position": number,
                           "campaign_attempt": position.get("campaign_attempt"),
                           "timeout_number_for_position": len(position.get("requeued_scientist_timeouts") or []) + 1,
                           "phase_timed_out": record.get("phase_timed_out"),
                           "timeout_seconds": record.get("timeout_seconds"),
                           "scientist_diagnostics": record.get("scientist_diagnostics"),
                           "scientist_diagnostics_sha256": record.get("scientist_diagnostics_sha256"),
                           "pre_requeue_state_copy": record.get("pre_requeue_state_copy"),
                           "approval": APPROVAL})


def _annotate_last_requeue(state_root: pathlib.Path, number: int, note: dict[str, Any]) -> None:
    """Mark the requeue record just written as automatic. Touches only that record."""
    state_path = state_root / "controller_state.json"
    state = json.loads(state_path.read_text())
    records = state["positions"][str(number)]["requeued_scientist_timeouts"]
    records[-1].update(note)
    requeue_scientist_timeout._atomic_write_bytes(
        state_path, (json.dumps(state, indent=2, sort_keys=True) + "\n").encode())


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        print("usage: ... -- <supervisor command ...>", file=sys.stderr)
        return 2
    split = argv.index("--")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state-root", type=pathlib.Path, required=True)
    parser.add_argument("--supervisor-dir", type=pathlib.Path, required=True)
    parser.add_argument("--max-timeouts-per-position", type=int, default=DEFAULT_MAX_TIMEOUTS_PER_POSITION)
    parser.add_argument("--max-total-requeues", type=int, default=DEFAULT_MAX_TOTAL_REQUEUES)
    args = parser.parse_args(argv[:split])
    status, summary = run(state_root=args.state_root, supervisor_dir=args.supervisor_dir, command=argv[split + 1:],
                          max_timeouts_per_position=args.max_timeouts_per_position,
                          max_total_requeues=args.max_total_requeues)
    print(json.dumps(summary, sort_keys=True))
    return status


if __name__ == "__main__":
    sys.exit(main())
