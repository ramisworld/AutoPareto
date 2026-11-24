"""End-to-end test of scripts/requeue_undelivered_payload.py on the real controller loop (offline).

A payload transfer for attempt 4 fails before anything reaches the mock A40, as the RunPod proxy did on
2026-11-05. The controller pauses the position. The tool refuses bad absence evidence, then requeues the
position, ``arm_continuation_payload`` arms it, and the campaign completes: attempt 4 trains once, with the
Scientist's original candidate, and no new Scientist call is made for it.

Run from the repository root:  python3 -B -m unittest scripts.tests.test_requeue_undelivered_payload -v
"""

from __future__ import annotations

import json
import os
import pathlib
import stat
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROCESS = ROOT / "scripts" / "tests" / "dry_campaign_process.py"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "tests"))

import requeue_undelivered_payload as requeue  # noqa: E402
from dry_campaign_process import FileBackedA40  # noqa: E402
from study.contracts_2026.continuation import arm_continuation_payload  # noqa: E402

CAMPAIGN = "pair-a-fixed-time-seed3"


def read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class RequeueUndeliveredPayloadTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="requeue-test-")
        self.tmp = pathlib.Path(self._tmp.name).resolve()
        self.state = self.tmp / "state"
        self.a40 = self.tmp / "a40"
        self.a40.mkdir()
        (self.a40 / "fail_transfers.json").write_text(json.dumps({"4": 1}))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def run_controller(self) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-B", str(PROCESS), "--state-root", str(self.state), "--a40-dir", str(self.a40)],
            cwd=ROOT, capture_output=True, text=True, timeout=600,
        )

    def controller_state(self) -> dict:
        return json.loads((self.state / "controller_state.json").read_text())

    def position_of(self, state: dict, attempt: int) -> int:
        rows = [row for row in state["schedule"]
                if row["campaign_id"] == CAMPAIGN and int(row["campaign_attempt"]) == attempt]
        self.assertEqual(len(rows), 1)
        return int(rows[0]["global_position"])

    def scientist_calls(self) -> int:
        return len(list((self.state / CAMPAIGN / "scientist_diagnostics").iterdir()))

    def evidence(self, execution: dict, **overrides) -> pathlib.Path:
        payload = {
            "schema": requeue.EVIDENCE_SCHEMA,
            "checked_utc": "2026-11-05T06:23:34Z",
            "remote_run_id": execution["remote_run_id"],
            "invocation_id": execution["invocation_id"],
            "payload_dir_present": (self.a40 / "payloads" / f"{execution['remote_run_id']}.json").exists(),
            "invocation_receipt_present": (self.a40 / "invocations" / f"{execution['invocation_id']}.json").exists(),
            "matching_processes": [],
            **overrides,
        }
        path = self.tmp / f"evidence-{len(list(self.tmp.glob('evidence-*')))}.json"
        path.write_text(json.dumps(payload))
        return path

    def test_requeue_resumes_the_same_candidate_without_a_new_scientist_call(self) -> None:
        first = self.run_controller()
        self.assertNotEqual(first.returncode, 0, first.stdout)
        self.assertIn("A40TransportCommandError", first.stderr)
        state = self.controller_state()
        global_position = self.position_of(state, 4)
        position = state["positions"][str(global_position)]
        self.assertEqual(position["status"], "reconciliation_required")
        self.assertEqual(position["reconciliation_failure"]["phase"], "payload_transfer")
        execution = position["execution"]
        self.assertEqual(execution["dispatch_status"], "prepared")
        candidate_sha = execution["candidate_source_after_sha256"]
        self.assertEqual(len(read_jsonl(self.a40 / "transfer_failures.jsonl")), 1)
        self.assertEqual(sorted(t["attempt"] for t in read_jsonl(self.a40 / "trainings.jsonl")), [1, 2, 3])
        calls_before = self.scientist_calls()
        state_path = self.state / "controller_state.json"
        paused_bytes = state_path.read_bytes()

        # Fail closed: evidence that something reached the A40, or evidence for another run, is refused.
        for bad in (self.evidence(execution, payload_dir_present=True),
                    self.evidence(execution, invocation_receipt_present=True),
                    self.evidence(execution, matching_processes=[{"pid": 1}]),
                    self.evidence(execution, remote_run_id="another-run")):
            with self.assertRaises(RuntimeError):
                requeue.requeue_undelivered_payload(
                    state_root=self.state, global_position=global_position, absence_evidence_path=bad)
            self.assertEqual(state_path.read_bytes(), paused_bytes)
        self.assertEqual(list(self.state.glob("controller_state.json.pre_requeue_*")), [])

        good = self.evidence(execution)
        result = requeue.requeue_undelivered_payload(
            state_root=self.state, global_position=global_position, absence_evidence_path=good)
        self.assertEqual(result["status"], "requeued")
        position = self.controller_state()["positions"][str(global_position)]
        self.assertEqual(position["status"], "running")
        self.assertNotIn("reconciliation_failure", position)
        self.assertEqual(position["execution"], execution)
        [record] = position["requeued_transport_failures"]
        self.assertEqual(record["type"], "A40TransportCommandError")
        self.assertEqual(record["phase"], "payload_transfer")
        self.assertFalse(record["new_agent_call"])
        [backup] = self.state.glob("controller_state.json.pre_requeue_*")
        self.assertEqual(backup.read_bytes(), paused_bytes)
        self.assertFalse(backup.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        with self.assertRaises(RuntimeError):  # a second requeue of the same position is refused
            requeue.requeue_undelivered_payload(
                state_root=self.state, global_position=global_position, absence_evidence_path=good)

        # Without arming, the controller refuses to resume and leaves the state alone.
        requeued_bytes = state_path.read_bytes()
        unarmed = self.run_controller()
        self.assertNotEqual(unarmed.returncode, 0)
        self.assertIn("interrupted before durable remote launch", unarmed.stderr)
        self.assertEqual(state_path.read_bytes(), requeued_bytes)

        armed = arm_continuation_payload(state_root=self.state, transport=FileBackedA40(self.a40))
        self.assertEqual(armed["status"], "armed")
        self.assertEqual(armed["execution"]["dispatch_status"], "launch_requested")

        final = self.run_controller()
        self.assertEqual(final.returncode, 0, final.stderr[-2000:])
        trainings = read_jsonl(self.a40 / "trainings.jsonl")
        self.assertEqual(sorted(t["attempt"] for t in trainings), list(range(1, 26)))
        self.assertEqual(len({t["invocation_id"] for t in trainings}), 25, "an invocation trained twice")
        trained_4 = [t for t in trainings if t["attempt"] == 4]
        self.assertEqual(trained_4[0]["invocation_id"], execution["invocation_id"])
        payload = json.loads((self.a40 / "payloads" / f"{execution['remote_run_id']}.json").read_text())
        self.assertEqual(payload["_source_sha256"], candidate_sha)
        # Two Scientist calls (parent selection, child edit) per later attempt; none again for attempt 4.
        self.assertEqual(self.scientist_calls() - calls_before, 2 * (25 - 4))
        state = self.controller_state()
        position = state["positions"][str(global_position)]
        self.assertEqual(position["status"], "completed")
        self.assertEqual(len(position["requeued_transport_failures"]), 1)
        history = read_jsonl(self.state / CAMPAIGN / "history.jsonl")
        self.assertEqual(sorted(int(h["campaign_attempt"]) for h in history), list(range(1, 26)))


if __name__ == "__main__":
    unittest.main()
