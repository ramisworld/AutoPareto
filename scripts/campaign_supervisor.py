#!/usr/bin/env python3
"""Supervise one AutoPareto campaign controller: exit detection, persistent logs, progress, alerts.

The controller is the only thing that changes campaign state. This wrapper never edits
``controller_state.json``, never retries a position and never restarts the controller.
Resuming is always a deliberate re-run of the identical command, which the controller's
state machine makes safe (completed positions are skipped, a remotely launched position is
reconciled against its durable receipt, and an interrupted pre-launch position halts as
``reconciliation_required``).

Why it exists: on 2026-11-04 E0 finished unnoticed because its watcher looked for the job
with ``pgrep -f <pattern>``, and the pattern also matched the watcher's own command line.
Here liveness never comes from searching process names:

* the supervisor learns of the controller's exit from ``waitpid`` (``Popen.wait``);
* an exclusive ``flock`` on ``<supervisor-dir>/lock`` is held by the supervisor and inherited
  by the controller, so the lock is free only once both are gone. Observers test it with a
  non-blocking lock attempt;
* leftover processes are found by recorded process-group IDs and by the PIDs in the
  Scientist invocation records, never by command-line patterns.

Seat usage: the controller rejects a Claude call whose stream shows any usage window at 95% or
more, but only after that call has finished, and the CLI reports usage once, mid-call. So once
any window reaches ``--usage-pause-utilization`` (default 80%), the supervisor stops the
controller at the next safe point: while the current attempt is training on the A40, after its
agent calls. That stop is resumable; the identical command later reconciles the attempt from its
durable receipt without retraining it.

Usage::

    campaign_supervisor.py run --supervisor-dir D --state-root S [--notify] -- <controller argv>
    campaign_supervisor.py status --supervisor-dir D --state-root S
    campaign_supervisor.py watch --supervisor-dir D --state-root S   # for a monitor; exits on a terminal event
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import fcntl
import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Iterable

EVENTS = "events.jsonl"
STATUS = "status.json"
LOCK = "lock"
LOG = "controller.log"

EXIT_REFUSED_LOCKED = 75
EXIT_REFUSED_LEFTOVER = 76
EXIT_WATCH_LOST = 3
RATE_LIMIT_WARN_UTILIZATION = 0.80  # the controller itself halts at 0.95
DEFAULT_STALL_SECONDS = 45 * 60  # a normal attempt (two agent calls + ~12 min on the A40) shows activity well within this
STOP_GRACE_SECONDS = 120.0
NOTIFY_ENV = "AUTOPARETO_SUPERVISOR_NOTIFY_CMD"


# ---------------------------------------------------------------- small utilities

def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def append_event(supervisor_dir: pathlib.Path, event: str, **fields: Any) -> dict[str, Any]:
    """Append one JSON line and fsync it: events survive a crash of this process."""
    record = {"utc": utc_now(), "event": event, **fields}
    line = (json.dumps(record, sort_keys=True, default=str) + "\n").encode()
    fd = os.open(supervisor_dir / EVENTS, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, line)
        os.fsync(fd)
    finally:
        os.close(fd)
    return record


def read_events(supervisor_dir: pathlib.Path) -> list[dict[str, Any]]:
    path = supervisor_dir / EVENTS
    if not path.exists():
        return []
    events = []
    for line in path.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a torn last line from a crash is ignored, never rewritten
    return events


def atomic_write_json(path: pathlib.Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_json(path: pathlib.Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def lock_is_held(supervisor_dir: pathlib.Path) -> bool:
    """True while the supervisor or the controller it started is alive."""
    path = supervisor_dir / LOCK
    if not path.exists():
        return False
    fd = os.open(path, os.O_RDONLY)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def process_command(pid: int) -> str:
    try:
        return subprocess.run(
            ["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True, timeout=10, check=False
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def log_tail(path: pathlib.Path, lines: int = 25) -> list[str]:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 16384))
            return handle.read().decode(errors="replace").splitlines()[-lines:]
    except OSError:
        return []


# ---------------------------------------------------------------- campaign state (read-only)

def load_state(state_root: pathlib.Path) -> dict[str, Any] | None:
    state = load_json(state_root / "controller_state.json")
    return state if isinstance(state, dict) else None


def active_positions(state: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
    if not state:
        return {}
    positions = state.get("positions") or {}
    return {
        int(key): value
        for key, value in positions.items()
        if isinstance(value, dict) and value.get("status") != "not_in_study"
    }


def summarize_state(state: dict[str, Any] | None) -> dict[str, Any]:
    active = active_positions(state)
    if not active:
        return {"state": "absent" if state is None else "no_active_positions", "active_positions": 0, "counts": {}}
    counts = collections.Counter(str(value.get("status")) for value in active.values())
    outcomes = collections.Counter(
        str(value.get("outcome", "completed")) for value in active.values() if value.get("status") == "completed"
    )
    current = None
    for key in sorted(active):
        value = active[key]
        if value.get("status") != "completed":
            execution = value.get("execution") if isinstance(value.get("execution"), dict) else {}
            current = {
                "global_position": key,
                "status": value.get("status"),
                "campaign_attempt": value.get("campaign_attempt"),
                "dispatch_status": execution.get("dispatch_status"),
                "reconciliation_failure": value.get("reconciliation_failure"),
                "last_transport_error": value.get("last_transport_error"),
            }
            break
    return {
        "state": "present",
        "active_positions": len(active),
        "counts": dict(counts),
        "completed_outcomes": dict(outcomes),
        "attempts_done": counts.get("completed", 0),
        "current": current,
    }


def result_summary(position: dict[str, Any]) -> dict[str, Any]:
    result = load_json(pathlib.Path(str(position.get("result_file", "")))) if position.get("result_file") else None
    if not isinstance(result, dict):
        return {}
    inference = result.get("inference_result") if isinstance(result.get("inference_result"), dict) else {}
    decision = result.get("pareto_decision") if isinstance(result.get("pareto_decision"), dict) else {}
    return {
        "result_status": result.get("status"),
        "protected_validation_bpb": result.get("protected_validation_bpb"),
        "pareto_decision": decision.get("decision"),
        "pareto_eligible": result.get("pareto_eligible"),
        "inference_status": inference.get("inference_status"),
        "error": (str(result.get("error"))[:300] if result.get("error") else None),
    }


def scientist_records(state_root: pathlib.Path) -> Iterable[tuple[pathlib.Path, dict[str, Any]]]:
    for path in sorted(state_root.glob("*/scientist_diagnostics/*/invocation.json")):
        record = load_json(path)
        if isinstance(record, dict):
            yield path, record


def rate_limit_from_stream(stdout_path: pathlib.Path) -> dict[str, Any] | None:
    """Highest seat-window utilization reported by a Claude Code stream, if any."""
    try:
        lines = stdout_path.read_text(errors="replace").split("\n")
    except OSError:
        return None
    found: dict[str, Any] | None = None
    for line in lines:
        if '"rate_limit_event"' not in line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        info = event.get("rate_limit_info") or {}
        raw = {name: state for name, state in (info.get("unifiedWindows") or {}).items() if isinstance(state, dict)}
        windows = {name: float(state.get("utilization", 0.0)) for name, state in raw.items()}
        resets = {}
        for name, state in raw.items():
            try:
                resets[name] = dt.datetime.fromtimestamp(float(state["resetsAt"]), dt.timezone.utc).isoformat(timespec="minutes")
            except (KeyError, TypeError, ValueError, OverflowError):
                pass
        found = {
            "status": info.get("status"),
            "overage_in_use": bool(info.get("isUsingOverage") or info.get("overageInUse")),
            "windows": windows,
            "resets_at_utc": resets,
            "max_utilization": max(windows.values(), default=0.0),
        }
    return found


def live_scientist(path: pathlib.Path, record: dict[str, Any]) -> int | None:
    """PID of a still-running Scientist process recorded as RUNNING, after checking its identity."""
    pid = record.get("pid")
    if record.get("status") != "RUNNING" or not isinstance(pid, int) or not pid_alive(pid):
        return None
    summary = record.get("command_summary") or []
    executable = pathlib.Path(str(summary[0])).name if summary else ""
    command = process_command(pid)
    # A recycled PID belonging to something else is never treated as a Scientist.
    return pid if executable and executable in command else None


# ---------------------------------------------------------------- notifications

def notify(enabled: bool, title: str, message: str) -> None:
    if not enabled:
        return
    custom = os.environ.get(NOTIFY_ENV)
    try:
        if custom:
            subprocess.run([custom, title, message], timeout=15, check=False)
        else:
            script = f"display notification {json.dumps(message)} with title {json.dumps(title)} sound name \"Basso\""
            subprocess.run(["osascript", "-e", script], timeout=15, check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        pass


# ---------------------------------------------------------------- supervisor

class Supervisor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.dir: pathlib.Path = args.supervisor_dir.resolve()
        self.state_root: pathlib.Path = args.state_root.resolve()
        self.command: list[str] = list(args.command)
        self.notify_enabled: bool = args.notify
        self.poll_seconds: float = args.poll_seconds
        self.stall_seconds: float = args.stall_seconds
        self.label: str = args.label or self.state_root.name
        self.lock_fd: int | None = None
        self.child: subprocess.Popen[bytes] | None = None
        self.launch_epoch = time.time()
        self.stop_signal: int | None = None
        self.stop_deadline: float | None = None
        self.escalated: list[int] = []
        self.positions_seen: dict[int, tuple[Any, Any]] = {}
        self.scientists_seen: dict[str, Any] = {}
        self.last_rate_limit: dict[str, Any] | None = None
        self.usage_pause_utilization: float = args.usage_pause_utilization
        self.usage_pause_pending: dict[str, Any] | None = None
        self.stop_reason: str | None = None
        self.last_activity = time.time()
        self.stall_alerted = False

    def event(self, name: str, **fields: Any) -> dict[str, Any]:
        record = append_event(self.dir, name, **fields)
        if fields.get("alert"):
            headline = fields.get("classification") or name
            notify(self.notify_enabled, f"AutoPareto {self.label}: ALERT", f"{headline}: {fields.get('message', '')}"[:220])
        return record

    # -- refusal checks
    def acquire_lock(self) -> bool:
        fd = os.open(self.dir / LOCK, os.O_RDWR | os.O_CREAT, 0o644)
        for _ in range(3):
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                os.set_inheritable(fd, True)
                self.lock_fd = fd
                return True
            except BlockingIOError:
                time.sleep(0.5)
        os.close(fd)
        return False

    def leftovers(self) -> list[dict[str, Any]]:
        found = []
        for record in read_events(self.dir):
            pgid = record.get("controller_pgid")
            if record.get("event") == "launch" and isinstance(pgid, int) and group_alive(pgid):
                found.append({"kind": "controller_process_group", "pgid": pgid})
        for path, record in scientist_records(self.state_root):
            pid = live_scientist(path, record)
            if pid is not None:
                found.append({"kind": "scientist", "pid": pid, "record": str(path)})
        return found

    # -- progress
    def poll_progress(self) -> dict[str, Any]:
        state = load_state(self.state_root)
        active = active_positions(state)
        for key in sorted(active):
            value = active[key]
            execution = value.get("execution") if isinstance(value.get("execution"), dict) else {}
            marker = (value.get("status"), execution.get("dispatch_status"))
            previous = self.positions_seen.get(key)
            if previous is None:
                self.positions_seen[key] = marker
                continue
            if previous == marker:
                continue
            self.positions_seen[key] = marker
            self.last_activity = time.time()
            status, dispatch = marker
            fields: dict[str, Any] = {
                "global_position": key,
                "campaign_attempt": value.get("campaign_attempt"),
                "status": status,
                "dispatch_status": dispatch,
                "previous": {"status": previous[0], "dispatch_status": previous[1]},
            }
            if status == "completed":
                fields.update(outcome=value.get("outcome", "completed"), **result_summary(value))
                self.event("position_completed", **fields)
            elif status == "reconciliation_required":
                failure = value.get("reconciliation_failure") or {}
                self.event("position_reconciliation_required", alert=True,
                           message=f"position {key}: {failure.get('type')}: {str(failure.get('message', ''))[:150]}",
                           reconciliation_failure=failure, **fields)
            else:
                self.event("position_progress", **fields)
        for path, record in scientist_records(self.state_root):
            if record.get("started_at_epoch", 0) < self.launch_epoch:
                continue
            key = path.parent.name
            status = record.get("status")
            if self.scientists_seen.get(key) == status:
                continue
            self.scientists_seen[key] = status
            self.last_activity = time.time()
            metadata = record.get("metadata") or {}
            if status == "RUNNING":
                continue  # the finished event carries the information
            rate = rate_limit_from_stream(path.parent / "stdout.jsonl")
            if rate:
                self.last_rate_limit = rate
            fields = {
                "invocation": key,
                "phase": metadata.get("phase"),
                "campaign_attempt": metadata.get("attempt_number", metadata.get("campaign_attempt")),
                "status": status,
                "returncode": record.get("returncode"),
                "seconds": round(float(record.get("completed_at_epoch", time.time())) - float(record.get("started_at_epoch", time.time())), 1),
                "rate_limit": rate,
            }
            self.event("scientist_call_finished", **fields)
            high = bool(rate and (rate["overage_in_use"] or rate["max_utilization"] >= self.usage_pause_utilization))
            if high and self.usage_pause_pending is None:
                # The controller's own check (>= 95%) runs only after a call has finished, so one long
                # call could still cross into overage. Pausing here, at the next safe point, keeps the
                # campaign well below that edge.
                self.usage_pause_pending = rate
                self.event("usage_pause_pending", alert=True, rate_limit=rate,
                           message=(f"seat usage {rate['max_utilization']:.0%} >= {self.usage_pause_utilization:.0%}; "
                                    f"pausing once the current attempt is training on the A40 "
                                    f"(resets {rate.get('resets_at_utc')})"))
        # Activity also includes Scientist output still being streamed.
        for candidate in (self.state_root / "controller_state.json", self.dir / LOG):
            try:
                self.last_activity = max(self.last_activity, candidate.stat().st_mtime)
            except OSError:
                pass
        for stdout in self.state_root.glob("*/scientist_diagnostics/*/stdout.jsonl"):
            try:
                self.last_activity = max(self.last_activity, stdout.stat().st_mtime)
            except OSError:
                pass
        idle = time.time() - self.last_activity
        if idle > self.stall_seconds and not self.stall_alerted:
            self.stall_alerted = True
            self.event("stall_warning", alert=True, idle_seconds=round(idle),
                       message=f"no controller, Scientist or log activity for {idle / 60:.0f} min")
        elif idle <= self.stall_seconds:
            self.stall_alerted = False
        summary = summarize_state(state)
        atomic_write_json(self.dir / STATUS, {
            "schema": "autopareto_campaign_supervisor_status_v1",
            "updated_utc": utc_now(),
            "supervisor_pid": os.getpid(),
            "controller_pid": self.child.pid if self.child else None,
            "controller_running": bool(self.child and self.child.poll() is None),
            "stop_requested": self.stop_signal,
            "summary": summary,
            "last_rate_limit": self.last_rate_limit,
            "idle_seconds": round(idle),
        })
        return summary

    # -- signals
    def request_stop(self, reason: str, signum: int) -> None:
        if self.stop_signal is not None or self.child is None:
            return
        self.stop_signal = signum
        self.stop_reason = reason
        self.stop_deadline = time.time() + STOP_GRACE_SECONDS
        self.event("stop_requested", signal=signal.Signals(signum).name, reason=reason,
                   note="SIGINT sent to the controller process group; it records the interrupted position itself")
        try:
            os.killpg(self.child.pid, signal.SIGINT)
        except ProcessLookupError:
            pass

    def on_stop_signal(self, signum: int, _frame: Any) -> None:
        self.request_stop("signal", signum)

    def pause_for_usage_if_safe(self, summary: dict[str, Any]) -> None:
        """Stop only while the current attempt is launched remotely: resumable, nothing lost or repeated."""
        if self.usage_pause_pending is None or self.stop_signal is not None:
            return
        current = summary.get("current") or {}
        if current.get("status") == "running" and current.get("dispatch_status") == "launch_requested":
            self.request_stop("seat_usage", signal.SIGINT)

    def escalate_if_needed(self) -> None:
        if self.stop_deadline is None or self.child is None or time.time() < self.stop_deadline:
            return
        next_signal = signal.SIGTERM if signal.SIGTERM not in self.escalated else signal.SIGKILL
        if next_signal in self.escalated:
            return
        self.escalated.append(next_signal)
        self.event("stop_escalated", signal=next_signal.name)
        try:
            os.killpg(self.child.pid, next_signal)
        except ProcessLookupError:
            pass
        self.stop_deadline = time.time() + 15

    # -- cleanup after the controller exits
    def cleanup(self, pgid: int) -> list[dict[str, Any]]:
        cleaned = []
        if group_alive(pgid):
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(pgid, sig)
                except ProcessLookupError:
                    break
                deadline = time.time() + 5
                while time.time() < deadline and group_alive(pgid):
                    time.sleep(0.2)
                if not group_alive(pgid):
                    break
            cleaned.append({"kind": "controller_process_group", "pgid": pgid, "alive_after": group_alive(pgid)})
        for path, record in scientist_records(self.state_root):
            if record.get("started_at_epoch", 0) < self.launch_epoch:
                continue
            pid = live_scientist(path, record)
            if pid is None:
                continue
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(pid, sig)  # the Scientist runs as its own session leader
                except ProcessLookupError:
                    break
                deadline = time.time() + 5
                while time.time() < deadline and pid_alive(pid):
                    time.sleep(0.2)
                if not pid_alive(pid):
                    break
            cleaned.append({"kind": "scientist", "pid": pid, "record": str(path), "alive_after": pid_alive(pid)})
        return cleaned

    def classify(self, returncode: int, summary: dict[str, Any]) -> tuple[str, bool, str]:
        counts = summary.get("counts", {})
        active = summary.get("active_positions", 0)
        current = summary.get("current") or {}
        if counts.get("reconciliation_required"):
            failure = current.get("reconciliation_failure") or {}
            return ("paused_reconciliation_required", True,
                    f"position {current.get('global_position')} needs reconciliation: {failure.get('type')}; "
                    "the controller will not resume until it is reconciled")
        if returncode == 0 and active and counts.get("completed", 0) == active:
            return "campaign_completed", False, f"all {active} positions completed"
        if current.get("status") == "running":
            if current.get("dispatch_status") == "launch_requested" and self.stop_reason == "seat_usage":
                rate = self.usage_pause_pending or {}
                return ("paused_for_seat_usage", True,
                        f"seat usage {rate.get('max_utilization', 0):.0%}; position {current.get('global_position')} "
                        f"keeps training on the A40. After the window resets ({rate.get('resets_at_utc')}), re-run "
                        "the identical command: it reconciles that attempt from its receipt (no retraining)")
            if current.get("dispatch_status") == "launch_requested":
                return ("interrupted_remote_attempt_resumable", True,
                        f"position {current.get('global_position')} was launched remotely; re-run the identical "
                        "command to reconcile it against its durable receipt (no retraining)")
            return ("interrupted_before_remote_launch", True,
                    f"position {current.get('global_position')} stopped before remote launch; manual reconciliation required")
        if self.stop_signal is not None:
            return "stopped_by_request", True, "stopped on request before a position was reserved"
        if returncode == 0:
            return "controller_exited_incomplete", True, "controller exited 0 with positions still pending"
        return "controller_failed", True, f"controller exited with {returncode}; see log tail"

    def run(self) -> int:
        self.dir.mkdir(parents=True, exist_ok=True)
        if not self.acquire_lock():
            print(f"refusing: {self.dir / LOCK} is held; a supervisor or its controller is still running", file=sys.stderr)
            return EXIT_REFUSED_LOCKED
        leftovers = self.leftovers()
        if leftovers:
            self.event("refused", alert=True, leftovers=leftovers,
                       message="processes from an earlier controller run are still alive; not starting")
            print(f"refusing: leftover processes {leftovers}", file=sys.stderr)
            return EXIT_REFUSED_LEFTOVER
        before = summarize_state(load_state(self.state_root))
        log = open(self.dir / LOG, "ab", buffering=0)
        log.write(f"\n===== supervisor launch {utc_now()} pid {os.getpid()} =====\n".encode())
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, self.on_stop_signal)
        signal.signal(signal.SIGINT, self.on_stop_signal)
        self.launch_epoch = time.time()
        self.child = subprocess.Popen(
            self.command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            pass_fds=(self.lock_fd,),
        )
        pgid = self.child.pid
        self.event("launch", supervisor_pid=os.getpid(), controller_pid=self.child.pid, controller_pgid=pgid,
                   argv=self.command, cwd=os.getcwd(), host=socket.gethostname(), state_before=before)
        notify(self.notify_enabled, f"AutoPareto {self.label}", f"controller started at {before.get('attempts_done', 0)} completed positions")
        self.poll_progress()
        while True:
            try:
                returncode = self.child.wait(timeout=self.poll_seconds)
                break
            except subprocess.TimeoutExpired:
                pass
            try:
                self.pause_for_usage_if_safe(self.poll_progress())
            except Exception as exc:  # monitoring must never take the controller down
                append_event(self.dir, "supervisor_poll_error", error=f"{type(exc).__name__}: {exc}")
            self.escalate_if_needed()
        summary = self.poll_progress()
        cleaned = self.cleanup(pgid)
        classification, alert, message = self.classify(returncode, summary)
        exit_fields = {
            "returncode": returncode,
            "signal": signal.Signals(-returncode).name if returncode < 0 else None,
            "classification": classification,
            "message": message,
            "duration_seconds": round(time.time() - self.launch_epoch, 1),
            "summary": summary,
            "cleanup": cleaned,
            "stop_requested": signal.Signals(self.stop_signal).name if self.stop_signal else None,
            "stop_reason": self.stop_reason,
            "last_rate_limit": self.last_rate_limit,
            "log_tail": log_tail(self.dir / LOG),
        }
        self.event("supervisor_exit", alert=alert, **exit_fields)
        atomic_write_json(self.dir / STATUS, {
            "schema": "autopareto_campaign_supervisor_status_v1",
            "updated_utc": utc_now(),
            "supervisor_pid": os.getpid(),
            "controller_pid": self.child.pid,
            "controller_running": False,
            "exit": {key: exit_fields[key] for key in ("returncode", "signal", "classification", "message")},
            "summary": summary,
            "last_rate_limit": self.last_rate_limit,
        })
        if not alert:
            notify(self.notify_enabled, f"AutoPareto {self.label}", message)
        log.close()
        return returncode if returncode >= 0 else 128 - returncode


# ---------------------------------------------------------------- observers

def liveness(supervisor_dir: pathlib.Path) -> dict[str, Any]:
    events = read_events(supervisor_dir)
    launches = [index for index, event in enumerate(events) if event.get("event") == "launch"]
    held = lock_is_held(supervisor_dir)
    if not launches:
        return {"liveness": "running" if held else "never_started", "lock_held": held}
    last = launches[-1]
    launch = events[last]
    exits = [event for event in events[last + 1:] if event.get("event") == "supervisor_exit"]
    supervisor_alive = pid_alive(int(launch.get("supervisor_pid", -1)))
    if exits:
        state = "exited"
    elif held and supervisor_alive:
        state = "running"
    elif held:
        state = "controller_running_unsupervised"
    else:
        state = "lost"  # both gone without an exit record (for example, the supervisor was SIGKILLed)
    return {"liveness": state, "lock_held": held, "launch": launch, "exit": exits[-1] if exits else None,
            "supervisor_alive": supervisor_alive}


def cmd_status(args: argparse.Namespace) -> int:
    supervisor_dir = args.supervisor_dir.resolve()
    report = liveness(supervisor_dir)
    report["summary"] = summarize_state(load_state(args.state_root.resolve()))
    report["status_file"] = load_json(supervisor_dir / STATUS)
    if report.get("exit"):
        report["exit"] = {key: report["exit"].get(key) for key in ("utc", "returncode", "signal", "classification", "message")}
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


def watch_line(event: dict[str, Any]) -> str | None:
    name = event.get("event")
    prefix = "ALERT " if event.get("alert") else ""
    if name == "launch":
        before = event.get("state_before") or {}
        return f"{event['utc']} launch: controller pid {event.get('controller_pid')}, {before.get('attempts_done', 0)} positions already completed"
    if name == "position_completed":
        bpb = event.get("protected_validation_bpb")
        bpb_text = f"{bpb:.4f}" if isinstance(bpb, (int, float)) else "n/a"
        return (f"{event['utc']} attempt {event.get('campaign_attempt')} done: {event.get('outcome')}, "
                f"BPB {bpb_text}, {event.get('pareto_decision')}, inference {event.get('inference_status')}")
    if name == "supervisor_exit":
        return (f"{event['utc']} {prefix}EXIT {event.get('classification')} (rc {event.get('returncode')}): "
                f"{event.get('message')}")
    if event.get("alert"):
        return f"{event['utc']} ALERT {name}: {event.get('message', '')}"
    if name in ("stop_requested", "stop_escalated", "supervisor_poll_error"):
        return f"{event['utc']} {name}: {event.get('signal') or event.get('error')}"
    return None


def cmd_watch(args: argparse.Namespace) -> int:
    """One line per significant event; exits on the supervisor's exit record or on loss of both processes."""
    supervisor_dir = args.supervisor_dir.resolve()
    seen = 0 if args.from_start else len(read_events(supervisor_dir))
    started = time.time()
    lost_polls = 0
    stale_reported = False
    while True:
        events = read_events(supervisor_dir)
        for event in events[seen:]:
            line = watch_line(event)
            if line:
                print(line, flush=True)
            if event.get("event") == "supervisor_exit":
                return 0
        seen = len(events)
        report = liveness(supervisor_dir)
        state = report["liveness"]
        if state in ("lost", "never_started"):
            if state == "never_started" and time.time() - started < args.startup_grace:
                lost_polls = 0
            else:
                lost_polls += 1
                if lost_polls >= 3:  # the supervisor may be writing its exit record
                    print(f"{utc_now()} ALERT LOST: no supervisor or controller is running and no exit was recorded "
                          f"({state}); check {supervisor_dir / LOG}", flush=True)
                    return EXIT_WATCH_LOST
        elif state == "exited":
            print(f"{utc_now()} EXIT already recorded: {(report.get('exit') or {}).get('classification')}", flush=True)
            return 0
        else:
            lost_polls = 0
        if state == "controller_running_unsupervised" and not stale_reported:
            stale_reported = True
            print(f"{utc_now()} ALERT supervisor died; its controller is still running and holds the lock "
                  "(progress is no longer recorded)", flush=True)
        time.sleep(args.interval)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("run", "status", "watch"):
        p = sub.add_parser(name)
        p.add_argument("--supervisor-dir", type=pathlib.Path, required=True)
        p.add_argument("--state-root", type=pathlib.Path, required=True)
        if name == "run":
            p.add_argument("--notify", action="store_true", help="macOS notifications on start, exit and alerts")
            p.add_argument("--poll-seconds", type=float, default=15.0)
            p.add_argument("--stall-seconds", type=float, default=DEFAULT_STALL_SECONDS)
            p.add_argument("--usage-pause-utilization", type=float, default=RATE_LIMIT_WARN_UTILIZATION,
                           help="pause at the next safe point once any seat window reaches this fraction")
            p.add_argument("--label", default=None)
            p.add_argument("command", nargs=argparse.REMAINDER)
        if name == "watch":
            p.add_argument("--interval", type=float, default=20.0)
            p.add_argument("--startup-grace", type=float, default=120.0)
            p.add_argument("--from-start", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "run":
        if args.command and args.command[0] == "--":
            args.command = args.command[1:]
        if not args.command:
            raise SystemExit("run needs the controller command after --")
        return Supervisor(args).run()
    if args.mode == "status":
        return cmd_status(args)
    return cmd_watch(args)


if __name__ == "__main__":
    raise SystemExit(main())
