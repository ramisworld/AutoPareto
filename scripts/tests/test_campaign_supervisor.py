"""Tests for scripts/campaign_supervisor.py against the real controller loop (offline).

Each test drives ``dry_campaign_process.py`` (the real ``run_campaign`` with the fake Claude CLI
and a file-backed mock A40) under the supervisor, interrupts it in a different way, and checks
that the exit is detected and classified, alerts are raised, a second controller cannot start,
and a resume neither loses nor repeats an attempt.

Run from the repository root:  python3 -B -m unittest scripts.tests.test_campaign_supervisor -v
"""

from __future__ import annotations

import json
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SUPERVISOR = ROOT / "scripts" / "campaign_supervisor.py"
PROCESS = ROOT / "scripts" / "tests" / "dry_campaign_process.py"
sys.path.insert(0, str(ROOT / "scripts"))

import campaign_supervisor as sup  # noqa: E402


def wait_for(predicate, timeout: float, what: str):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.1)
    raise AssertionError(f"timed out after {timeout}s waiting for {what}")


def read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class SupervisorCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="supervisor-test-")
        self.tmp = pathlib.Path(self._tmp.name).resolve()
        self.sdir = self.tmp / "supervisor"
        self.state = self.tmp / "state"
        self.a40 = self.tmp / "a40"
        self.a40.mkdir()
        self.notifications = self.tmp / "notifications.log"
        notifier = self.tmp / "notify.sh"
        notifier.write_text(f"#!/bin/sh\nprintf '%s|%s\\n' \"$1\" \"$2\" >> {self.notifications}\n")
        notifier.chmod(0o755)
        self.env = {**os.environ, sup.NOTIFY_ENV: str(notifier)}
        self.processes: list[subprocess.Popen] = []

    def tearDown(self) -> None:
        for process in self.processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
        for event in sup.read_events(self.sdir):
            pgid = event.get("controller_pgid")
            if isinstance(pgid, int):
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self._tmp.cleanup()

    # -- helpers
    def supervisor_argv(self) -> list[str]:
        return [sys.executable, "-B", str(SUPERVISOR), "run", "--supervisor-dir", str(self.sdir),
                "--state-root", str(self.state), "--poll-seconds", "0.3", "--notify", "--label", "test",
                "--", sys.executable, "-B", str(PROCESS), "--state-root", str(self.state), "--a40-dir", str(self.a40)]

    def start(self) -> subprocess.Popen:
        process = subprocess.Popen(self.supervisor_argv(), cwd=ROOT, env=self.env, start_new_session=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.processes.append(process)
        return process

    def watch(self) -> subprocess.Popen:
        process = subprocess.Popen(
            [sys.executable, "-B", str(SUPERVISOR), "watch", "--supervisor-dir", str(self.sdir),
             "--state-root", str(self.state), "--interval", "0.3", "--startup-grace", "60", "--from-start"],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
        self.processes.append(process)
        return process

    def events(self, name: str | None = None) -> list[dict]:
        events = sup.read_events(self.sdir)
        return [event for event in events if name is None or event.get("event") == name]

    def last_exit(self) -> dict:
        exits = self.events("supervisor_exit")
        self.assertTrue(exits, "no supervisor_exit event")
        return exits[-1]

    def status(self) -> dict:
        out = subprocess.run([sys.executable, "-B", str(SUPERVISOR), "status", "--supervisor-dir", str(self.sdir),
                              "--state-root", str(self.state)], capture_output=True, text=True, check=True).stdout
        return json.loads(out)

    def assert_campaign_complete_without_repeats(self) -> None:
        trainings = read_jsonl(self.a40 / "trainings.jsonl")
        self.assertEqual(len(trainings), 25)
        self.assertEqual(len({t["invocation_id"] for t in trainings}), 25, "an invocation trained twice")
        self.assertEqual(sorted(t["attempt"] for t in trainings), list(range(1, 26)))
        history = read_jsonl(self.state / "pair-a-fixed-time-seed3" / "history.jsonl")
        attempts = [record.get("campaign_attempt") for record in history]
        self.assertEqual(sorted(attempts), list(range(1, 26)), "history lost or repeated an attempt")
        state = json.loads((self.state / "controller_state.json").read_text())
        active = [v for v in state["positions"].values() if v["status"] != "not_in_study"]
        self.assertEqual([v["status"] for v in active], ["completed"] * 25)

    def set_slow(self, **attempts: float) -> None:
        (self.a40 / "slow_attempts.json").write_text(json.dumps({k.removeprefix("a"): v for k, v in attempts.items()}))

    def clear_slow(self) -> None:
        (self.a40 / "slow_attempts.json").unlink(missing_ok=True)


class TestCompleteRun(SupervisorCase):
    def test_exit_detected_progress_recorded_and_watch_terminates(self) -> None:
        watcher = self.watch()  # its own argv names the state root: the old pgrep self-match
        supervisor = self.start()
        self.assertEqual(supervisor.wait(timeout=180), 0, supervisor.stderr.read())
        exit_event = self.last_exit()
        self.assertEqual(exit_event["classification"], "campaign_completed")
        self.assertFalse(exit_event["alert"])
        completed = self.events("position_completed")
        self.assertEqual(sorted(e["campaign_attempt"] for e in completed), list(range(1, 26)))
        failed = [e for e in completed if e["outcome"] == "scientific_failure"]
        self.assertEqual([e["campaign_attempt"] for e in failed], [3])
        self.assertTrue(all(isinstance(e["protected_validation_bpb"], float) for e in completed if e["outcome"] == "completed"))
        self.assertEqual(len(self.events("scientist_call_finished")), 50)
        self.assert_campaign_complete_without_repeats()
        self.assertFalse(sup.lock_is_held(self.sdir))
        self.assertEqual(self.status()["liveness"], "exited")
        self.assertIn("all 25 positions completed", self.notifications.read_text())
        # The watcher exits on the exit record instead of waiting on itself.
        self.assertEqual(watcher.wait(timeout=15), 0)
        output = watcher.stdout.read()
        self.assertIn("EXIT campaign_completed", output)
        self.assertEqual(output.count(" done: "), 25)
        # The controller log is persistent and appended, never truncated.
        self.assertIn("supervisor launch", (self.sdir / sup.LOG).read_text())


class TestSecondSupervisorAndGracefulStop(SupervisorCase):
    def test_second_launch_refused_then_stop_and_resume(self) -> None:
        self.set_slow(a2=60)
        first = self.start()
        wait_for(lambda: (self.a40 / "waiting-2").exists(), 60, "attempt 2 in its remote wait")
        second = self.start()
        self.assertEqual(second.wait(timeout=30), sup.EXIT_REFUSED_LOCKED)
        self.assertIn("refusing", second.stderr.read())
        self.assertEqual(len(self.events("launch")), 1)
        # Graceful stop while attempt 2 is running remotely.
        os.kill(first.pid, signal.SIGTERM)
        self.assertNotEqual(first.wait(timeout=60), 0)
        exit_event = self.last_exit()
        self.assertEqual(exit_event["classification"], "interrupted_remote_attempt_resumable")
        self.assertTrue(exit_event["alert"])
        self.assertEqual(exit_event["stop_requested"], "SIGTERM")
        self.assertIn("ALERT", self.notifications.read_text())
        # Resume with the identical command.
        self.clear_slow()
        resumed = self.start()
        self.assertEqual(resumed.wait(timeout=180), 0, resumed.stderr.read())
        self.assertEqual(self.last_exit()["classification"], "campaign_completed")
        self.assert_campaign_complete_without_repeats()
        requests = [r for r in read_jsonl(self.a40 / "launch_requests.jsonl") if r["attempt"] == 2]
        self.assertEqual([r["new"] for r in requests], [True, False], "resume must re-request, not retrain")


class TestControllerKilled(SupervisorCase):
    def test_sigkill_during_remote_wait_is_detected_and_resumes_without_retraining(self) -> None:
        self.set_slow(a5=120)
        first = self.start()
        wait_for(lambda: (self.a40 / "waiting-5").exists(), 90, "attempt 5 in its remote wait")
        pgid = self.events("launch")[-1]["controller_pgid"]
        os.killpg(pgid, signal.SIGKILL)
        self.assertNotEqual(first.wait(timeout=30), 0)
        exit_event = self.last_exit()
        self.assertEqual(exit_event["signal"], "SIGKILL")
        self.assertEqual(exit_event["classification"], "interrupted_remote_attempt_resumable")
        self.assertEqual(exit_event["summary"]["attempts_done"], 4)
        self.assertIn("interrupted_remote_attempt_resumable", self.notifications.read_text())
        self.clear_slow()
        resumed = self.start()
        self.assertEqual(resumed.wait(timeout=180), 0, resumed.stderr.read())
        self.assert_campaign_complete_without_repeats()
        self.assertEqual(sum(1 for t in read_jsonl(self.a40 / "trainings.jsonl") if t["attempt"] == 5), 1)


class TestStopDuringScientistCall(SupervisorCase):
    def test_stop_pauses_the_position_kills_the_agent_and_never_reruns(self) -> None:
        (self.a40 / "scientist_sleep").write_text("60")
        first = self.start()

        def running_scientist():
            for path, record in sup.scientist_records(self.state):
                if record.get("status") == "RUNNING" and isinstance(record.get("pid"), int):
                    return record["pid"]
            return None

        agent_pid = wait_for(running_scientist, 60, "a running Scientist call")
        os.kill(first.pid, signal.SIGTERM)
        self.assertNotEqual(first.wait(timeout=60), 0)
        exit_event = self.last_exit()
        self.assertEqual(exit_event["classification"], "paused_reconciliation_required")
        self.assertTrue(exit_event["alert"])
        self.assertFalse(sup.pid_alive(agent_pid), "the agent process outlived the controller")
        self.assertEqual(read_jsonl(self.a40 / "trainings.jsonl"), [])
        state = json.loads((self.state / "controller_state.json").read_text())
        first_active = min(sup.active_positions(state))
        self.assertEqual(state["positions"][str(first_active)]["status"], "reconciliation_required")
        self.assertEqual(state["positions"][str(first_active)]["reconciliation_failure"]["phase"], "pre_remote_execution")
        # Re-running the identical command does not re-run the paused position.
        (self.a40 / "scientist_sleep").unlink()
        again = self.start()
        self.assertNotEqual(again.wait(timeout=60), 0)
        self.assertEqual(self.last_exit()["classification"], "paused_reconciliation_required")
        self.assertIn("campaign is paused", "\n".join(self.last_exit()["log_tail"]))
        self.assertEqual(read_jsonl(self.a40 / "trainings.jsonl"), [])


class TestSupervisorKilled(SupervisorCase):
    def test_controller_keeps_the_lock_and_loss_is_reported(self) -> None:
        self.set_slow(a2=6)
        first = self.start()
        wait_for(lambda: (self.a40 / "waiting-2").exists(), 60, "attempt 2 in its remote wait")
        os.kill(first.pid, signal.SIGKILL)
        first.wait(timeout=10)
        self.assertEqual(self.status()["liveness"], "controller_running_unsupervised")
        refused = self.start()
        self.assertEqual(refused.wait(timeout=30), sup.EXIT_REFUSED_LOCKED)
        watcher = self.watch()
        # The orphaned controller finishes the campaign; then nothing holds the lock and no exit was recorded.
        wait_for(lambda: not sup.lock_is_held(self.sdir), 180, "the orphaned controller to finish")
        self.assertEqual(watcher.wait(timeout=30), sup.EXIT_WATCH_LOST)
        output = watcher.stdout.read()
        self.assertIn("supervisor died", output)
        self.assertIn("ALERT LOST", output)
        self.assertEqual(self.status()["liveness"], "lost")
        self.assert_campaign_complete_without_repeats()
        # A new supervisor may start once nothing is left; the controller finds nothing to do.
        again = self.start()
        self.assertEqual(again.wait(timeout=60), 0, again.stderr.read())
        self.assertEqual(self.last_exit()["classification"], "campaign_completed")
        self.assert_campaign_complete_without_repeats()


class TestSeatUsagePause(SupervisorCase):
    def test_high_usage_pauses_only_at_a_remote_launch_and_resumes_exactly(self) -> None:
        # Every attempt spends 2 s in its remote wait, so the safe point is observable at 0.3 s polling.
        self.set_slow(**{f"a{n}": 2 for n in range(1, 26)})
        (self.a40 / "rate_limit_utilization").write_text("0.12")
        first = self.start()
        wait_for(lambda: (self.a40 / "waiting-2").exists(), 120, "attempt 2 in its remote wait")
        (self.a40 / "rate_limit_utilization").write_text("0.83")  # the seat fills up from attempt 3 on
        self.assertNotEqual(first.wait(timeout=120), 0)
        pending = self.events("usage_pause_pending")
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["rate_limit"]["max_utilization"], 0.83)
        self.assertEqual(pending[0]["rate_limit"]["resets_at_utc"]["five_hour"], "2026-11-04T14:30+00:00")
        exit_event = self.last_exit()
        self.assertEqual(exit_event["classification"], "paused_for_seat_usage")
        self.assertEqual(exit_event["stop_reason"], "seat_usage")
        self.assertTrue(exit_event["alert"])
        # Paused while attempt 3 was launched remotely: its two agent calls were made, nothing was lost.
        self.assertEqual(exit_event["summary"]["current"]["dispatch_status"], "launch_requested")
        self.assertEqual(exit_event["summary"]["current"]["campaign_attempt"], 3)
        state = json.loads((self.state / "controller_state.json").read_text())
        self.assertFalse([v for v in state["positions"].values() if v["status"] == "reconciliation_required"])
        self.assertIn("paused_for_seat_usage", self.notifications.read_text())
        # After the window "resets", the identical command finishes the campaign without repeats.
        (self.a40 / "rate_limit_utilization").write_text("0.05")
        self.clear_slow()
        resumed = self.start()
        self.assertEqual(resumed.wait(timeout=180), 0, resumed.stderr.read())
        self.assertEqual(self.last_exit()["classification"], "campaign_completed")
        self.assert_campaign_complete_without_repeats()


class TestUnits(SupervisorCase):
    def test_rate_limit_parsing_uses_the_highest_window(self) -> None:
        stream = self.tmp / "stdout.jsonl"
        event = {"type": "rate_limit_event", "rate_limit_info": {
            "status": "allowed", "isUsingOverage": False,
            "unifiedWindows": {"five_hour": {"utilization": 0.85}, "seven_day": {"utilization": 0.1}}}}
        stream.write_text('{"type":"system","subtype":"init"}\n' + json.dumps(event) + "\n")
        parsed = sup.rate_limit_from_stream(stream)
        self.assertEqual(parsed["max_utilization"], 0.85)
        self.assertFalse(parsed["overage_in_use"])

    def test_leftover_scientist_refuses_start_and_recycled_pid_is_ignored(self) -> None:
        sleeper = subprocess.Popen(["sleep", "60"], start_new_session=True)
        self.processes.append(sleeper)
        record_dir = self.state / "pair-a-fixed-time-seed3" / "scientist_diagnostics" / "scientist-1-x"
        record_dir.mkdir(parents=True)
        record = {"status": "RUNNING", "pid": sleeper.pid, "command_summary": ["/bin/sleep"], "started_at_epoch": 0}
        (record_dir / "invocation.json").write_text(json.dumps(record))
        self.assertEqual(sup.live_scientist(record_dir / "invocation.json", record), sleeper.pid)
        self.assertIsNone(sup.live_scientist(record_dir / "invocation.json", {**record, "command_summary": ["/opt/homebrew/bin/claude"]}))
        refused = self.start()
        self.assertEqual(refused.wait(timeout=30), sup.EXIT_REFUSED_LEFTOVER)
        self.assertEqual(self.events("refused")[-1]["leftovers"][0]["pid"], sleeper.pid)
        self.assertEqual(self.events("launch"), [])


if __name__ == "__main__":
    unittest.main()
