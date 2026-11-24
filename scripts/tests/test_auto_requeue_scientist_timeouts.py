"""End-to-end tests of scripts/auto_requeue_scientist_timeouts.py around the real supervisor and controller (offline).

1. Attempt 4's child edit times out once. The wrapper requeues it, relaunches the identical supervisor command and
   the campaign completes: 25 trainings, one automatic requeue, recorded in the log and in the position.
2. Attempt 4's child edit keeps timing out. The wrapper requeues the first two timeouts and stops at the third,
   leaving the position paused for the PI (exit status 4).
3. A post-training failure of unknown origin pauses the campaign. The wrapper does not requeue it and stops.

Run from the repository root:  python3 -B -m unittest scripts.tests.test_auto_requeue_scientist_timeouts -v
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SUPERVISOR = ROOT / "scripts" / "campaign_supervisor.py"
PROCESS = ROOT / "scripts" / "tests" / "dry_campaign_process.py"
sys.path.insert(0, str(ROOT / "scripts"))

import auto_requeue_scientist_timeouts as auto  # noqa: E402
import requeue_scientist_timeout  # noqa: E402

CAMPAIGN = "pair-a-fixed-time-seed3"


def read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class AutoRequeueTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="auto-requeue-test-")
        self.tmp = pathlib.Path(self._tmp.name).resolve()
        self.state = self.tmp / "state"
        self.state.mkdir()
        self.sdir = self.tmp / "supervisor"
        self.a40 = self.tmp / "a40"
        self.a40.mkdir()
        (self.a40 / "scientist_timeout").write_text("3")
        self._cwd = os.getcwd()
        os.chdir(ROOT)

    def tearDown(self) -> None:
        os.chdir(self._cwd)
        self._tmp.cleanup()

    def command(self) -> list[str]:
        return [sys.executable, "-B", str(SUPERVISOR), "run", "--supervisor-dir", str(self.sdir),
                "--state-root", str(self.state), "--poll-seconds", "0.3", "--label", "test",
                "--", sys.executable, "-B", str(PROCESS), "--state-root", str(self.state), "--a40-dir", str(self.a40)]

    def hang_attempt_4_child_edit(self) -> None:
        (self.a40 / "scientist_hang").write_text("attempt 4, phase edit_child\n20\n")

    def position_of_attempt(self, attempt: int) -> tuple[int, dict]:
        state = json.loads((self.state / "controller_state.json").read_text())
        [number] = [int(row["global_position"]) for row in state["schedule"]
                    if row["campaign_id"] == CAMPAIGN and int(row["campaign_attempt"]) == attempt]
        return number, state["positions"][str(number)]

    def test_one_timeout_is_requeued_and_the_campaign_completes(self) -> None:
        self.hang_attempt_4_child_edit()

        def requeue_then_let_the_call_finish(**kwargs):
            result = requeue_scientist_timeout.requeue_scientist_timeout(**kwargs)
            (self.a40 / "scientist_hang").unlink()
            return result

        status, summary = auto.run(state_root=self.state, supervisor_dir=self.sdir, command=self.command(),
                                   requeue=requeue_then_let_the_call_finish)
        self.assertEqual((status, summary["classification"]), (0, "campaign_completed"))
        trainings = read_jsonl(self.a40 / "trainings.jsonl")
        self.assertEqual(sorted(t["attempt"] for t in trainings), list(range(1, 26)))
        self.assertEqual(len({t["invocation_id"] for t in trainings}), 25, "an invocation trained twice")
        log = read_jsonl(self.sdir / auto.LOG)
        decisions = [entry["decision"] for entry in log if entry["event"] == "supervisor_finished"]
        self.assertEqual(decisions, ["requeued", "stop_completed"])
        number, position = self.position_of_attempt(4)
        self.assertEqual(position["status"], "completed")
        [record] = position["requeued_scientist_timeouts"]
        self.assertEqual(record["requeued_by_wrapper"], "scripts/auto_requeue_scientist_timeouts.py")
        self.assertEqual((record["phase_timed_out"], record["scientific_position_consumed"],
                          record["timeout_changed"]), ("edit_child", False, False))
        requeued = [entry for entry in log if entry.get("decision") == "requeued"]
        self.assertEqual((requeued[0]["global_position"], requeued[0]["campaign_attempt"],
                          requeued[0]["timeout_number_for_position"]), (number, 4, 1))
        copy = self.state / record["pre_requeue_state_copy"]
        self.assertTrue(copy.exists())
        self.assertFalse(copy.stat().st_mode & 0o222, "the pre-requeue copy must be read-only")

    def test_third_timeout_of_one_attempt_stops_for_the_pi(self) -> None:
        self.hang_attempt_4_child_edit()
        status, summary = auto.run(state_root=self.state, supervisor_dir=self.sdir, command=self.command())
        self.assertEqual(status, auto.STOPPED_FOR_PI_EXIT_STATUS)
        self.assertEqual(summary["decision"], "stop_for_pi")
        self.assertIn("timed out 3 times", summary["reason"])
        number, position = self.position_of_attempt(4)
        self.assertEqual(summary["global_position"], number)
        self.assertEqual((position["status"], position["reconciliation_failure"]["type"]),
                         ("reconciliation_required", "ScientistExecutionTimeout"))
        self.assertEqual(len(position["requeued_scientist_timeouts"]), 2)
        self.assertEqual(sorted(t["attempt"] for t in read_jsonl(self.a40 / "trainings.jsonl")), [1, 2, 3])
        launches = [e for e in read_jsonl(self.sdir / "events.jsonl") if e.get("event") == "launch"]
        self.assertEqual(len(launches), 3)
        self.assertEqual({tuple(e["argv"]) for e in launches}, {tuple(self.command()[self.command().index("--") + 1:])})

    def test_other_pauses_are_not_requeued(self) -> None:
        (self.a40 / "failed_receipts.json").write_text(json.dumps([4]))
        status, summary = auto.run(state_root=self.state, supervisor_dir=self.sdir, command=self.command())
        self.assertEqual(status, auto.STOPPED_FOR_PI_EXIT_STATUS)
        self.assertEqual(summary["classification"], "paused_reconciliation_required")
        self.assertIn("not a pre-candidate Scientist timeout", summary["reason"])
        _, position = self.position_of_attempt(4)
        self.assertEqual(position["status"], "reconciliation_required")
        self.assertNotIn("requeued_scientist_timeouts", position)
        self.assertEqual([e["decision"] for e in read_jsonl(self.sdir / auto.LOG)
                          if e["event"] == "supervisor_finished"], ["stop_for_pi"])


if __name__ == "__main__":
    unittest.main()
