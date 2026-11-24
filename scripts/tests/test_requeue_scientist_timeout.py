"""End-to-end test of scripts/requeue_scientist_timeout.py on the real controller loop (offline).

Attempt 4's child-edit Scientist call is held past a short timeout, as attempt 9's was on 2026-11-05. The
controller pauses the position. The tool refuses the wrong cases, then requeues it; the controller reruns the
position with fresh Scientist calls and completes the campaign: 25 attempts trained once each, the timed-out
call's diagnostics kept, nothing else changed.

Run from the repository root:  python3 -B -m unittest scripts.tests.test_requeue_scientist_timeout -v
"""

from __future__ import annotations

import json
import pathlib
import stat
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROCESS = ROOT / "scripts" / "tests" / "dry_campaign_process.py"
sys.path.insert(0, str(ROOT / "scripts"))

import requeue_scientist_timeout as requeue  # noqa: E402

CAMPAIGN = "pair-a-fixed-time-seed3"


def read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class RequeueScientistTimeoutTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="scientist-timeout-test-")
        self.tmp = pathlib.Path(self._tmp.name).resolve()
        self.state = self.tmp / "state"
        self.a40 = self.tmp / "a40"
        self.a40.mkdir()
        (self.a40 / "scientist_timeout").write_text("3")
        (self.a40 / "scientist_hang").write_text("attempt 4, phase edit_child\n20\n")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def run_controller(self) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-B", str(PROCESS), "--state-root", str(self.state), "--a40-dir", str(self.a40)],
            cwd=ROOT, capture_output=True, text=True, timeout=600,
        )

    def calls(self) -> list[dict]:
        return [json.loads(path.read_text())
                for path in sorted((self.state / CAMPAIGN / "scientist_diagnostics").glob("*/invocation.json"))]

    def test_timeout_is_requeued_with_fresh_scientist_calls(self) -> None:
        first = self.run_controller()
        self.assertNotEqual(first.returncode, 0)
        self.assertIn("ScientistExecutionTimeout", first.stderr)
        state_path = self.state / "controller_state.json"
        state = json.loads(state_path.read_text())
        [global_position] = [int(row["global_position"]) for row in state["schedule"]
                             if row["campaign_id"] == CAMPAIGN and int(row["campaign_attempt"]) == 4]
        position = state["positions"][str(global_position)]
        self.assertEqual((position["status"], position["reconciliation_failure"]["type"],
                          position["reconciliation_failure"]["phase"]),
                         ("reconciliation_required", "ScientistExecutionTimeout", "pre_remote_execution"))
        self.assertNotIn("execution", position)
        self.assertEqual(sorted(t["attempt"] for t in read_jsonl(self.a40 / "trainings.jsonl")), [1, 2, 3])
        timed_out = [c for c in self.calls() if c["status"] == "FAILED"]
        self.assertEqual(len(timed_out), 1)
        paused = state_path.read_bytes()

        with self.assertRaisesRegex(RuntimeError, "not reconciliation_required"):
            requeue.requeue_scientist_timeout(state_root=self.state, global_position=global_position - 1)
        self.assertEqual(state_path.read_bytes(), paused)

        result = requeue.requeue_scientist_timeout(state_root=self.state, global_position=global_position)
        self.assertEqual(result["status"], "requeued")
        position = json.loads(state_path.read_text())["positions"][str(global_position)]
        self.assertEqual(position["status"], "pending")
        self.assertNotIn("reconciliation_failure", position)
        [record] = position["requeued_scientist_timeouts"]
        self.assertEqual((record["phase_timed_out"], record["timeout_seconds"], record["timeout_changed"]),
                         ("edit_child", 3.0, False))
        [backup] = self.state.glob("controller_state.json.pre_requeue_scientist_timeout_*")
        self.assertEqual(backup.read_bytes(), paused)
        self.assertFalse(backup.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        with self.assertRaisesRegex(RuntimeError, "not reconciliation_required"):
            requeue.requeue_scientist_timeout(state_root=self.state, global_position=global_position)

        (self.a40 / "scientist_hang").unlink()
        final = self.run_controller()
        self.assertEqual(final.returncode, 0, final.stderr[-2000:])
        trainings = read_jsonl(self.a40 / "trainings.jsonl")
        self.assertEqual(sorted(t["attempt"] for t in trainings), list(range(1, 26)))
        self.assertEqual(len({t["invocation_id"] for t in trainings}), 25)
        calls = self.calls()
        attempt_4 = [c for c in calls if int(c["metadata"]["attempt_number"]) == 4]
        self.assertEqual(sorted((c["metadata"]["phase"], c["status"]) for c in attempt_4),
                         [("edit_child", "COMPLETE"), ("edit_child", "FAILED"),
                          ("select_parent", "COMPLETE"), ("select_parent", "COMPLETE")])
        self.assertEqual(len(calls), 2 * 25 + 2)
        self.assertTrue(pathlib.Path(record["scientist_diagnostics"]).is_file())
        state = json.loads(state_path.read_text())
        self.assertEqual(state["positions"][str(global_position)]["status"], "completed")
        self.assertEqual(len(state["positions"][str(global_position)]["requeued_scientist_timeouts"]), 1)


if __name__ == "__main__":
    unittest.main()
