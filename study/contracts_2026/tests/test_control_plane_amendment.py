import hashlib
import json
import pathlib
import tempfile
import unittest
from unittest import mock

from study.contracts_2026.tools.amend_control_plane_manifest import amend


class ControlPlaneAmendmentTests(unittest.TestCase):
    def test_requeues_only_a_pre_remote_reconciliation_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            repo = root / "repo"
            repo.mkdir()
            source = repo / "fixture.txt"
            source.write_text("fixture\n")
            prior = root / "prior.json"
            prior.write_text(json.dumps({"immutable_deployment": {"fixture.txt": hashlib.sha256(source.read_bytes()).hexdigest()}}))
            state_root = root / "state"
            state_root.mkdir()
            positions = {str(index): {"status": "pending"} for index in range(1, 51)}
            for index in range(1, 8):
                positions[str(index)] = {"status": "completed"}
            positions["8"] = {"status": "reconciliation_required", "reconciliation_failure": {"phase": "pre_remote_execution"}}
            (state_root / "controller_state.json").write_text(json.dumps({
                "manifest_sha256": hashlib.sha256(prior.read_bytes()).hexdigest(), "positions": positions,
            }))
            output = root / "amended.json"
            with mock.patch("study.contracts_2026.tools.amend_control_plane_manifest.ROOT", repo), \
                 mock.patch("study.contracts_2026.tools.amend_control_plane_manifest.NEW_RUNTIME_PATHS", ()):
                result = amend(
                    prior_manifest=prior,
                    output_manifest=output,
                    state_root=state_root,
                    effective_position=8,
                    remote_absence_verified_at="2026-09-09T00:00:00Z",
                    reason="fixture",
                )
            state = json.loads((state_root / "controller_state.json").read_text())
            self.assertEqual(result["requeued_status"], "pending")
            self.assertEqual(state["positions"]["8"]["status"], "pending")
            self.assertEqual(state["positions"]["1"]["status"], "completed")
            self.assertEqual(state["positions"]["9"]["status"], "pending")
            self.assertEqual(len(state["positions"]["8"]["non_consuming_infrastructure_events"]), 1)
            manifest = json.loads(output.read_text())
            self.assertEqual(len(manifest["operational_amendments"]), 1)

    def _single_arm_post_remote_state(self, root, prior, *, active_contracts):
        """T-only study: odd positions are N (not_in_study), position 2 completed, position 4 failed remotely."""
        state_root = root / "state"
        state_root.mkdir()
        old_invocation = "pair-a-004-old-invocation"
        positions = {
            str(index): {"status": "not_in_study" if index % 2 else "pending"} for index in range(1, 51)
        }
        positions["2"] = {"status": "completed"}
        positions["4"] = {
            "status": "reconciliation_required",
            "execution": {
                "schema": "autopareto_pending_execution_v1",
                "invocation_id": old_invocation,
                "remote_run_id": "claude-opus55-position-004",
                "remote_payload_dir": "/opt/payloads/position-004",
                "dispatch_status": "launch_requested",
                "reservation": {},
            },
            "reconciliation_failure": {
                "type": "RemoteInfrastructureFailure",
                "receipt": {"status": "FAILED", "invocation_id": old_invocation, "exit_status": 1},
            },
        }
        state = {"manifest_sha256": hashlib.sha256(prior.read_bytes()).hexdigest(), "positions": positions}
        if active_contracts:
            state["active_contracts"] = active_contracts
        (state_root / "controller_state.json").write_text(json.dumps(state))
        return state_root, old_invocation

    def test_single_arm_study_skips_inactive_positions_and_retries_the_same_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            repo = root / "repo"
            (repo / "protected").mkdir(parents=True)
            adapter = repo / "protected" / "inference_adapter.py"
            adapter.write_text("adapter\n")
            prior = root / "prior.json"
            prior.write_text(json.dumps({"immutable_deployment": {
                "protected/inference_adapter.py": hashlib.sha256(adapter.read_bytes()).hexdigest(),
            }}))
            state_root, old_invocation = self._single_arm_post_remote_state(root, prior, active_contracts=["T"])
            output = root / "amended.json"
            with mock.patch("study.contracts_2026.tools.amend_control_plane_manifest.ROOT", repo), \
                 mock.patch("study.contracts_2026.tools.amend_control_plane_manifest.NEW_RUNTIME_PATHS", ()):
                result = amend(
                    prior_manifest=prior, output_manifest=output, state_root=state_root,
                    effective_position=4, remote_absence_verified_at="2026-11-04T00:00:00Z",
                    reason="fixture", retry_existing_execution=True,
                )
            state = json.loads((state_root / "controller_state.json").read_text())
            self.assertEqual(result["requeued_status"], "running")
            self.assertEqual(state["positions"]["4"]["execution"]["retry_of_invocation_id"], old_invocation)
            self.assertEqual(state["positions"]["4"]["execution"]["remote_payload_dir"], "/opt/payloads/position-004")
            self.assertEqual(state["positions"]["1"]["status"], "not_in_study")
            self.assertEqual(state["positions"]["2"]["status"], "completed")
            self.assertEqual(state["positions"]["5"]["status"], "not_in_study")
            self.assertEqual(state["positions"]["6"]["status"], "pending")

    def test_inactive_positions_are_rejected_in_a_two_arm_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            prior = root / "prior.json"
            prior.write_text(json.dumps({"immutable_deployment": {}}))
            state_root, _ = self._single_arm_post_remote_state(root, prior, active_contracts=None)
            with self.assertRaisesRegex(RuntimeError, "completed prefix is not intact at position 1"):
                amend(
                    prior_manifest=prior, output_manifest=root / "out.json", state_root=state_root,
                    effective_position=4, remote_absence_verified_at="fixture", reason="fixture",
                    retry_existing_execution=True,
                )

    def test_rejects_any_position_with_execution_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            prior = root / "prior.json"
            prior.write_text(json.dumps({"immutable_deployment": {}}))
            state_root = root / "state"
            state_root.mkdir()
            positions = {str(index): {"status": "pending"} for index in range(1, 51)}
            positions["1"] = {"status": "reconciliation_required", "execution": {}, "reconciliation_failure": {"phase": "pre_remote_execution"}}
            (state_root / "controller_state.json").write_text(json.dumps({
                "manifest_sha256": hashlib.sha256(prior.read_bytes()).hexdigest(), "positions": positions,
            }))
            with self.assertRaisesRegex(RuntimeError, "execution state"):
                amend(
                    prior_manifest=prior,
                    output_manifest=root / "out.json",
                    state_root=state_root,
                    effective_position=1,
                    remote_absence_verified_at="fixture",
                    reason="fixture",
                )

    def test_timeout_amendment_updates_manifest_and_state_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            repo = root / "repo"
            repo.mkdir()
            source = repo / "fixture.txt"
            source.write_text("fixture\n")
            prior = root / "prior.json"
            prior.write_text(json.dumps({
                "immutable_deployment": {"fixture.txt": hashlib.sha256(source.read_bytes()).hexdigest()},
                "scientist": {"configuration": {"timeout_seconds": 600.0}},
            }))
            state_root = root / "state"
            state_root.mkdir()
            positions = {str(index): {"status": "pending"} for index in range(1, 51)}
            for index in range(1, 8):
                positions[str(index)] = {"status": "completed"}
            positions["8"] = {
                "status": "reconciliation_required",
                "reconciliation_failure": {"phase": "pre_remote_execution"},
            }
            prior_sha = hashlib.sha256(prior.read_bytes()).hexdigest()
            (state_root / "controller_state.json").write_text(json.dumps({
                "manifest_sha256": prior_sha,
                "scientist_config": {"timeout_seconds": 600.0},
                "positions": positions,
            }))
            output = root / "amended.json"
            with mock.patch("study.contracts_2026.tools.amend_control_plane_manifest.ROOT", repo), \
                 mock.patch("study.contracts_2026.tools.amend_control_plane_manifest.NEW_RUNTIME_PATHS", ()):
                result = amend(
                    prior_manifest=prior,
                    output_manifest=output,
                    state_root=state_root,
                    effective_position=8,
                    remote_absence_verified_at="2026-09-10T00:00:00Z",
                    reason="bounded Scientist timeout was reached before remote execution",
                    scientist_timeout_seconds=900.0,
                )
            state = json.loads((state_root / "controller_state.json").read_text())
            manifest = json.loads(output.read_text())
            self.assertEqual(result["requeued_status"], "pending")
            self.assertEqual(state["scientist_config"]["timeout_seconds"], 900.0)
            self.assertEqual(manifest["scientist"]["configuration"]["timeout_seconds"], 900.0)
            amendment = manifest["operational_amendments"][0]
            self.assertEqual(amendment["scientist_timeout_seconds"], {"before": 600.0, "after": 900.0})

    def test_requeues_same_candidate_only_for_verified_post_remote_harness_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            repo = root / "repo"
            (repo / "protected").mkdir(parents=True)
            adapter = repo / "protected" / "inference_adapter.py"
            adapter.write_text("adapter fix\n")
            prior = root / "prior.json"
            prior.write_text(json.dumps({
                "immutable_deployment": {
                    "protected/inference_adapter.py": hashlib.sha256(adapter.read_bytes()).hexdigest(),
                },
            }))
            state_root = root / "state"
            state_root.mkdir()
            old_invocation = "pair-a-001-old-invocation"
            execution = {
                "schema": "autopareto_pending_execution_v1",
                "invocation_id": old_invocation,
                "remote_run_id": "pair-a-position-001",
                "remote_payload_dir": "/opt/payloads/position-001",
                "dispatch_status": "launch_requested",
                "reservation": {"manifest_sha256": hashlib.sha256(prior.read_bytes()).hexdigest()},
            }
            positions = {str(index): {"status": "pending"} for index in range(1, 51)}
            positions["1"] = {
                "status": "reconciliation_required",
                "execution": execution,
                "reconciliation_failure": {
                    "type": "RemoteInfrastructureFailure",
                    "receipt": {"status": "FAILED", "invocation_id": old_invocation, "exit_status": 1},
                },
            }
            prior_sha = hashlib.sha256(prior.read_bytes()).hexdigest()
            (state_root / "controller_state.json").write_text(json.dumps({
                "manifest_sha256": prior_sha,
                "positions": positions,
            }))
            output = root / "amended.json"
            with mock.patch("study.contracts_2026.tools.amend_control_plane_manifest.ROOT", repo), \
                 mock.patch("study.contracts_2026.tools.amend_control_plane_manifest.NEW_RUNTIME_PATHS", ()):
                result = amend(
                    prior_manifest=prior,
                    output_manifest=output,
                    state_root=state_root,
                    effective_position=1,
                    remote_absence_verified_at="2026-09-15T00:00:00Z",
                    reason="protected adapter compatibility defect; durable FAILED receipt and no result",
                    retry_existing_execution=True,
                )
            state = json.loads((state_root / "controller_state.json").read_text())
            position = state["positions"]["1"]
            self.assertEqual(result["requeued_status"], "running")
            self.assertEqual(position["status"], "running")
            self.assertEqual(position["execution"]["dispatch_status"], "launch_requested")
            self.assertNotEqual(position["execution"]["invocation_id"], old_invocation)
            self.assertEqual(position["execution"]["remote_payload_dir"], "/opt/payloads/position-001")
            self.assertEqual(position["execution"]["retry_of_invocation_id"], old_invocation)
            self.assertEqual(len(position["non_consuming_infrastructure_events"]), 1)
            self.assertEqual(
                position["non_consuming_infrastructure_events"][0]["prior_execution"]["invocation_id"],
                old_invocation,
            )
            self.assertEqual(state["manifest_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
