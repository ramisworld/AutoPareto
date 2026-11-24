"""End-to-end tests of scripts/run_extension_attempt.py on the real controller loop (offline).

1. A dry 25-attempt campaign (attempt 3 is a candidate failure) is interrupted by a transfer drop at attempt 25
   and finished with 24 valid results. prepare refuses until the 25 attempts are complete, keeps the frozen
   25-attempt snapshot, changes only the view's attempt number, and refuses to run twice. Attempt 26 survives a
   transfer drop through requeue and arming, then fails as a candidate failure; attempt 27 opens by itself and
   brings the arm to 25 valid results. Every attempt trains once; each extra attempt's two Scientist prompts carry
   its own note and no other prompt carries one.
2. The cap stops the extension (exit status 3) instead of opening another attempt.
3. The `campaign` step runs the remaining scheduled attempts and the extension in one process.

Run from the repository root:  python3 -B -m unittest scripts.tests.test_run_extension_attempt -v
"""

from __future__ import annotations

import json
import pathlib
import stat
import subprocess
import sys
import tarfile
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROCESS = ROOT / "scripts" / "tests" / "dry_campaign_process.py"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "tests"))

import requeue_undelivered_payload as requeue  # noqa: E402
import run_extension_attempt as extension  # noqa: E402
from dry_campaign_process import FileBackedA40  # noqa: E402
from study.contracts_2026.continuation import arm_continuation_payload  # noqa: E402

CAMPAIGN = "pair-a-fixed-time-seed3"


def read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_only(path: pathlib.Path) -> bool:
    return not path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)


class RunExtensionAttemptTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="extension-test-")
        self.tmp = pathlib.Path(self._tmp.name).resolve()
        self.state = self.tmp / "state"
        self.a40 = self.tmp / "a40"
        self.a40.mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def run_controller(self, *extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-B", str(PROCESS), "--state-root", str(self.state), "--a40-dir", str(self.a40), *extra],
            cwd=ROOT, capture_output=True, text=True, timeout=600,
        )

    def controller_state(self) -> dict:
        return json.loads((self.state / "controller_state.json").read_text())

    def fail_attempts(self, *attempts: int) -> None:
        (self.a40 / "fail_attempts.json").write_text(json.dumps(list(attempts)))

    def requeue_after_drop(self, global_position: int) -> None:
        execution = self.controller_state()["positions"][str(global_position)]["execution"]
        evidence = self.tmp / f"evidence-{global_position}.json"
        evidence.write_text(json.dumps({
            "schema": requeue.EVIDENCE_SCHEMA, "checked_utc": "2026-11-05T00:00:00Z",
            "remote_run_id": execution["remote_run_id"], "invocation_id": execution["invocation_id"],
            "payload_dir_present": (self.a40 / "payloads" / f"{execution['remote_run_id']}.json").exists(),
            "invocation_receipt_present": (self.a40 / "invocations" / f"{execution['invocation_id']}.json").exists(),
            "matching_processes": [],
        }))
        requeue.requeue_undelivered_payload(
            state_root=self.state, global_position=global_position, absence_evidence_path=evidence)

    def scientist_prompts(self) -> list[tuple[int, str]]:
        prompts = []
        for record in sorted((self.state / CAMPAIGN / "scientist_diagnostics").glob("*/invocation.json")):
            invocation = json.loads(record.read_text())
            argv = invocation["command_summary"]
            prompts.append((int(invocation["metadata"]["attempt_number"]), argv[argv.index("-p") + 1]))
        return prompts

    def last_output(self, process: subprocess.CompletedProcess) -> dict:
        return json.loads(process.stdout.strip().splitlines()[-1])

    def test_extension_runs_until_25_valid_results(self) -> None:
        # The 25-attempt campaign, with a transfer drop at its last attempt.
        (self.a40 / "fail_transfers.json").write_text(json.dumps({"25": 1}))
        self.assertNotEqual(self.run_controller().returncode, 0)
        last = max(int(row["global_position"]) for row in self.controller_state()["schedule"]
                   if row["contract_id"] == "T")
        with self.assertRaisesRegex(RuntimeError, "not all complete"):
            extension.prepare_extension(state_root=self.state)
        self.requeue_after_drop(last)
        self.assertEqual(arm_continuation_payload(state_root=self.state, transport=FileBackedA40(self.a40))["status"], "armed")
        self.assertEqual(self.run_controller().returncode, 0)

        unprepared = self.run_controller("--extension")
        self.assertNotEqual(unprepared.returncode, 0)
        self.assertIn("has not been prepared", unprepared.stderr)

        state_path = self.state / "controller_state.json"
        view_path = self.state / CAMPAIGN / "campaign_view.json"
        state_before, view_before = state_path.read_bytes(), view_path.read_bytes()
        prepared = extension.prepare_extension(state_root=self.state)
        self.assertEqual((prepared["valid_results"], prepared["failed_attempts"]), (24, [3]))
        amendment = json.loads(pathlib.Path(prepared["amendment"]).read_text())
        self.assertEqual((amendment["target_valid_results"], amendment["max_total_attempts"]), (25, 30))
        frozen = amendment["frozen_25_attempt_result"]
        self.assertEqual((frozen["completed_attempts"], frozen["valid_results"], frozen["failed_attempts"]),
                         (25, 24, [3]))
        self.assertEqual([a["answer"][:20] for a in amendment["approvals"]],
                         ["Yes, add one attempt", "Charge attempt 9 as "])
        for path in (pathlib.Path(prepared["amendment"]), pathlib.Path(prepared["snapshot"]),
                     self.state / CAMPAIGN / extension.view_copy_name(26)):
            self.assertTrue(read_only(path), path)
        with tarfile.open(prepared["snapshot"]) as archive:
            member = archive.extractfile(f"{self.state.name}/controller_state.json")
            self.assertEqual(member.read(), state_before)
            member = archive.extractfile(f"{self.state.name}/{CAMPAIGN}/campaign_view.json")
            self.assertEqual(member.read(), view_before)
        self.assertEqual((self.state / CAMPAIGN / extension.view_copy_name(26)).read_bytes(), view_before)
        old_view, new_view = json.loads(view_before), json.loads(view_path.read_text())
        self.assertEqual(new_view["attempt_number"], 26)
        self.assertEqual({key for key in old_view if old_view[key] != new_view[key]}, {"attempt_number"})
        state = self.controller_state()
        note_26 = state["positions"]["51"]["extension"]["scientist_prompt_note"]
        self.assertIn("Attempt 3 ended as a scientific failure and produced no measurement, so 24 of the 25 "
                      "attempts so far are valid.", note_26)
        self.assertEqual(state["positions"]["51"]["status"], "pending")
        self.assertNotIn("52", state["positions"])
        self.assertEqual(state["schedule"], json.loads(state_before)["schedule"])
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            extension.prepare_extension(state_root=self.state)

        # Attempt 26, with a transfer drop: requeue, refuse unarmed, arm; it then fails as a candidate failure.
        self.fail_attempts(3, 26)
        (self.a40 / "fail_transfers.json").write_text(json.dumps({"26": 1}))
        dropped = self.run_controller("--extension")
        self.assertNotEqual(dropped.returncode, 0)
        self.assertIn("A40TransportCommandError", dropped.stderr)
        position = self.controller_state()["positions"]["51"]
        self.assertEqual((position["status"], position["reconciliation_failure"]["phase"]),
                         ("reconciliation_required", "payload_transfer"))
        calls_before = len(self.scientist_prompts())
        self.requeue_after_drop(51)
        unarmed = self.run_controller("--extension")
        self.assertNotEqual(unarmed.returncode, 0)
        self.assertIn("interrupted before durable remote launch", unarmed.stderr)
        with self.assertRaisesRegex(RuntimeError, "already complete"):
            arm_continuation_payload(state_root=self.state, transport=FileBackedA40(self.a40))
        self.assertEqual(extension.arm_extension_payload(state_root=self.state, transport=FileBackedA40(self.a40))["status"], "armed")
        final = self.run_controller("--extension")
        self.assertEqual(final.returncode, 0, final.stderr[-2000:])
        result = self.last_output(final)
        self.assertEqual((result["status"], result["valid_results"], result["completed_attempts"],
                          result["failed_attempts"]), ("completed", 25, 27, [3, 26]))
        self.assertEqual([(a["campaign_attempt"], a["outcome"]) for a in result["attempts_run"]],
                         [(26, "scientific_failure"), (27, "completed")])

        trainings = read_jsonl(self.a40 / "trainings.jsonl")
        self.assertEqual(sorted(t["attempt"] for t in trainings), list(range(1, 28)))
        self.assertEqual(len({t["invocation_id"] for t in trainings}), 27, "an invocation trained twice")
        prompts = self.scientist_prompts()
        self.assertEqual(len(prompts), calls_before + 2, "a new Scientist call was made for attempt 26 after the requeue")
        self.assertEqual(len(prompts), 54)
        state = self.controller_state()
        note_27 = state["positions"]["52"]["extension"]["scientist_prompt_note"]
        self.assertIn("Attempts 3 and 26 ended as scientific failures and produced no measurement, so 24 of the 26 "
                      "attempts so far are valid.", note_27)
        self.assertEqual(sorted(a for a, p in prompts if note_26 in p), [26, 26])
        self.assertEqual(sorted(a for a, p in prompts if note_27 in p), [27, 27])
        self.assertEqual(sorted(a for a, p in prompts if "Orchestration note" in p), [26, 26, 27, 27])
        self.assertTrue(all(f"attempt {attempt}," in prompt for attempt, prompt in prompts))
        history = read_jsonl(self.state / CAMPAIGN / "history.jsonl")
        self.assertEqual(sorted(int(h["campaign_attempt"]) for h in history), list(range(1, 28)))
        self.assertEqual([state["positions"][p]["status"] for p in ("51", "52")], ["completed", "completed"])
        self.assertEqual(state["positions"]["51"]["outcome"], "scientific_failure")
        self.assertEqual(len(state["positions"]["51"]["requeued_transport_failures"]), 1)
        copy_27 = self.state / CAMPAIGN / extension.view_copy_name(27)
        self.assertTrue(read_only(copy_27))
        self.assertEqual(json.loads(copy_27.read_text())["attempt_number"], 25)
        view = json.loads(view_path.read_text())
        self.assertEqual(sorted(f["attempt"] for f in view["failures"]), [3, 26])
        self.assertIn(f"{CAMPAIGN}-attempt-27", [record["experiment_id"] for record in view["same_arm_evidence"]])

        again = self.run_controller("--extension")
        self.assertEqual(again.returncode, 0)
        self.assertEqual(self.last_output(again)["status"], "already_completed")
        self.assertNotIn("53", self.controller_state()["positions"])

    def test_cap_stops_the_extension(self) -> None:
        self.fail_attempts(3, 26, 27)
        self.assertEqual(self.run_controller().returncode, 0)
        extension.prepare_extension(state_root=self.state, max_total_attempts=27)
        for _ in range(2):
            stopped = self.run_controller("--extension")
            self.assertEqual(stopped.returncode, 3, stopped.stderr[-2000:])
            result = self.last_output(stopped)
            self.assertEqual((result["status"], result["completed_attempts"], result["valid_results"]),
                             ("cap_reached", 27, 24))
        self.assertNotIn("53", self.controller_state()["positions"])
        self.assertEqual(sorted(t["attempt"] for t in read_jsonl(self.a40 / "trainings.jsonl")), list(range(1, 28)))

    def test_campaign_and_extension_in_one_process(self) -> None:
        through = self.run_controller("--campaign-and-extension")
        self.assertEqual(through.returncode, 0, through.stderr[-2000:])
        result = self.last_output(through)
        self.assertEqual((result["status"], result["valid_results"], result["completed_attempts"]),
                         ("completed", 25, 26))
        self.assertEqual(len(list(self.tmp.glob("state.snapshot_25_attempts_*.tar"))), 1)
        self.assertEqual(sorted(t["attempt"] for t in read_jsonl(self.a40 / "trainings.jsonl")), list(range(1, 27)))
        again = self.run_controller("--campaign-and-extension")
        self.assertEqual(again.returncode, 0, again.stderr[-2000:])
        self.assertEqual(self.last_output(again)["status"], "already_completed")
        self.assertEqual(len(list(self.tmp.glob("state.snapshot_25_attempts_*.tar"))), 1)


if __name__ == "__main__":
    unittest.main()
