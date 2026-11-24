"""End-to-end test of scripts/reconcile_post_training_candidate_failure.py on the real controller loop (offline).

Attempt 4's invocation ends with a FAILED durable receipt of unknown origin, as attempt 9's did on 2026-11-05
after the protected cache-size check rejected it. The controller pauses. The tool refuses mismatched evidence and an
error the launcher does not classify as candidate. It then charges the attempt to the candidate under a new
manifest that pins only the new launcher. The controller resumes under that manifest, never retrains attempt 4,
makes no extra Scientist call and completes the campaign. The exact failure reason stays in the view, history and
result.

Run from the repository root:  python3 -B -m unittest scripts.tests.test_reconcile_post_training_candidate_failure -v
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROCESS = ROOT / "scripts" / "tests" / "dry_campaign_process.py"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import reconcile_post_training_candidate_failure as reconcile  # noqa: E402
from study.contracts_2026 import campaign_controller as controller  # noqa: E402
from study.contracts_2026.tools import dry_run_claude_campaign as dry  # noqa: E402

CAMPAIGN = "pair-a-fixed-time-seed3"
CACHE_ERROR = (
    "RuntimeError: bracketed inference did not produce /opt/x/position-008.json.inference.json: returncode=1; "
    "stderr=Traceback (most recent call last):\n  File \"/opt/autopareto_trusted/protected/inference_adapter.py\", "
    "line 340, in validated_cache_nbytes\n    raise RuntimeError(\nRuntimeError: candidate cache_nbytes reported "
    "7864320 bytes but protected state traversal found 7907344 bytes\n"
)


def read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReconcilePostTrainingCandidateFailureTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="reconcile-test-")
        self.tmp = pathlib.Path(self._tmp.name).resolve()
        self.state = self.tmp / "state"
        self.a40 = self.tmp / "a40"
        self.a40.mkdir()
        (self.a40 / "failed_receipts.json").write_text("[4]")
        (self.tmp / "state.baseline.json").write_text(json.dumps(dry.dry_run_baseline()))
        self.manifest = self.tmp / "state.manifest.json"
        self.manifest.write_text(json.dumps({**dry.DRY_RUN_MANIFEST,
                                             "immutable_deployment": {reconcile.LAUNCHER: "1" * 64}}))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def run_controller(self, manifest: pathlib.Path | None = None) -> subprocess.CompletedProcess:
        extra = ["--manifest", str(manifest)] if manifest else []
        return subprocess.run(
            [sys.executable, "-B", str(PROCESS), "--state-root", str(self.state), "--a40-dir", str(self.a40), *extra],
            cwd=ROOT, capture_output=True, text=True, timeout=600,
        )

    def write_evidence(self, receipt: dict, error: str = CACHE_ERROR) -> pathlib.Path:
        folder = self.tmp / f"evidence-{hashlib.sha256(error.encode()).hexdigest()[:8]}"
        folder.mkdir()
        (folder / "invocation_receipt.json").write_text(json.dumps(receipt, indent=2))
        (folder / "worker_failure_receipt.json").write_text(
            json.dumps({"status": "error", "failure_origin": "unknown", "error": error}))
        (folder / "launcher_output.json").write_text(json.dumps({"status": "error", "error": error}))
        (folder / "invocation.log").write_text("training ...\n" + error + "\nRuntimeError: "
                                               + reconcile.UNKNOWN_ORIGIN_MESSAGE + "\n")
        (folder / "absence.json").write_text(json.dumps({
            "schema": reconcile.EVIDENCE_SCHEMA, "checked_utc": "2026-11-05T13:29:43Z",
            "invocation_id": receipt["invocation_id"], "matching_processes": [], "bridge_result_present": False,
            "pod_files": {name: {"sha256": sha(folder / name)} for name in reconcile.EVIDENCE_FILES},
        }))
        return folder

    def reconcile(self, evidence: pathlib.Path, position: int, name: str = "reconciled.json") -> dict:
        with mock.patch.object(controller, "_manifest_hashes", lambda *_a, **_k: {}):
            return reconcile.reconcile(
                state_root=self.state, manifest_path=self.manifest, new_manifest_name=name,
                global_position=position, evidence_dir=evidence,
                reason="protected cache-size check rejected the candidate", approval="PI: charge as candidate failure")

    def test_post_training_failure_is_charged_once_and_the_campaign_completes(self) -> None:
        paused = self.run_controller()
        self.assertNotEqual(paused.returncode, 0)
        state_path = self.state / "controller_state.json"
        state = json.loads(state_path.read_text())
        [position] = [int(r["global_position"]) for r in state["schedule"]
                      if r["campaign_id"] == CAMPAIGN and int(r["campaign_attempt"]) == 4]
        failure = state["positions"][str(position)]["reconciliation_failure"]
        self.assertEqual((state["positions"][str(position)]["status"], failure["type"]),
                         ("reconciliation_required", "RemoteInfrastructureFailure"))
        paused_bytes = state_path.read_bytes()
        calls_before = len(list((self.state / CAMPAIGN / "scientist_diagnostics").glob("*/invocation.json")))
        evidence = self.write_evidence(failure["receipt"])

        # Refusals leave everything as it was.
        with self.assertRaisesRegex(RuntimeError, "not reconciliation_required"):
            self.reconcile(evidence, position - 2)
        tampered = self.write_evidence(failure["receipt"], CACHE_ERROR + " ")
        (tampered / "invocation.log").write_text("edited\n")
        with self.assertRaisesRegex(RuntimeError, "differs from the pod's SHA-256"):
            self.reconcile(tampered, position)
        opaque = self.write_evidence(failure["receipt"], "RuntimeError: CUDA error: an illegal memory access")
        with self.assertRaisesRegex(RuntimeError, "classifies this error as 'unknown'"):
            self.reconcile(opaque, position)
        self.assertEqual(state_path.read_bytes(), paused_bytes)
        self.assertFalse((self.tmp / "reconciled.json").exists())

        result = self.reconcile(evidence, position)
        self.assertEqual((result["status"], result["campaign_attempt"], result["view_attempt_number"]),
                         ("reconciled", 4, 5))
        new_manifest = json.loads(pathlib.Path(result["manifest"]).read_text())
        self.assertFalse(pathlib.Path(result["manifest"]).stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        [amendment] = new_manifest["operational_amendments"]
        launcher_now = sha(ROOT / reconcile.LAUNCHER)
        self.assertEqual(amendment["changed_immutable_paths"],
                         {reconcile.LAUNCHER: {"before": "1" * 64, "after": launcher_now}})
        self.assertEqual((amendment["reconciled_prior_position"], amendment["scientific_contract_changed"]),
                         (position, False))
        stripped = json.loads(json.dumps(new_manifest))
        stripped["immutable_deployment"][reconcile.LAUNCHER] = "1" * 64
        del stripped["operational_amendments"]
        self.assertEqual(stripped, json.loads(self.manifest.read_text()))
        [backup] = self.state.glob(f"controller_state.json.pre_reconcile_position_{position:03d}_*")
        self.assertEqual(backup.read_bytes(), paused_bytes)

        state = json.loads(state_path.read_text())
        record = state["positions"][str(position)]
        self.assertEqual((record["status"], record["outcome"]), ("completed", "scientific_failure"))
        self.assertEqual(record["reconciliation_failure"], failure, "the original pause record must be kept")
        self.assertEqual(state["manifest_sha256"], result["manifest_sha256"])
        stored = json.loads(pathlib.Path(record["result_file"]).read_text())
        self.assertEqual((stored["status"], stored["failure_class"], stored["error"]),
                         ("scientific_failure", "candidate", CACHE_ERROR))
        self.assertEqual(stored["worker_payload"]["failure_origin"], "candidate")
        self.assertEqual(stored["worker_payload"]["reconciliation_evidence"]["recorded_failure_origin"], "unknown")
        view = json.loads((self.state / CAMPAIGN / "campaign_view.json").read_text())
        self.assertIn({"attempt": 4, "failure_class": "candidate", "error": CACHE_ERROR}, view["failures"])
        evidence_ids = [r["experiment_id"] for r in view["same_arm_evidence"]]
        self.assertIn(f"{CAMPAIGN}-attempt-04", evidence_ids)
        history = read_jsonl(self.state / CAMPAIGN / "history.jsonl")
        self.assertEqual([h["status"] for h in history if int(h["campaign_attempt"]) == 4], ["scientific_failure"])
        with self.assertRaisesRegex(RuntimeError, "does not run under the given manifest"):
            self.reconcile(evidence, position, name="again.json")

        # The campaign resumes under the new manifest only.
        self.assertNotEqual(self.run_controller().returncode, 0)
        (self.a40 / "failed_receipts.json").unlink()
        final = self.run_controller(pathlib.Path(result["manifest"]))
        self.assertEqual(final.returncode, 0, final.stderr[-2000:])
        trainings = read_jsonl(self.a40 / "trainings.jsonl")
        self.assertEqual(sorted(t["attempt"] for t in trainings), list(range(1, 26)))
        calls = list((self.state / CAMPAIGN / "scientist_diagnostics").glob("*/invocation.json"))
        self.assertEqual(len(calls), 50)
        self.assertEqual(calls_before, 8)
        view = json.loads((self.state / CAMPAIGN / "campaign_view.json").read_text())
        self.assertEqual(sorted(f["attempt"] for f in view["failures"]), [3, 4])
        history = read_jsonl(self.state / CAMPAIGN / "history.jsonl")
        self.assertEqual(sorted(int(h["campaign_attempt"]) for h in history), list(range(1, 26)))


if __name__ == "__main__":
    unittest.main()
