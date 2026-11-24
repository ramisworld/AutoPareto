import json
import pathlib
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

from study.contracts_2026 import campaign_controller as controller
from study.contracts_2026 import remote_runner
from study.contracts_2026.bridge import REMOTE_RESULT_SCHEMA, sha256_file
from study.contracts_2026.context import build_campaign_view


class DurableInvocationTests(unittest.TestCase):
    def test_candidate_disk_capacity_fails_closed_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            attempt = root / "attempts"
            state = root / "state"
            attempt.mkdir()
            state.mkdir()
            with mock.patch.object(
                remote_runner.shutil,
                "disk_usage",
                return_value=shutil._ntuple_diskusage(total=10_000, used=9_500, free=500),
            ):
                with self.assertRaisesRegex(RuntimeError, "insufficient candidate attempt filesystem capacity"):
                    remote_runner._require_candidate_disk_capacity(
                        attempt,
                        state,
                        minimum_free_bytes=1_000,
                    )

    def test_candidate_disk_capacity_checks_shared_filesystem_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            attempt = root / "attempts"
            state = root / "state"
            attempt.mkdir()
            state.mkdir()
            usage = shutil._ntuple_diskusage(total=10_000, used=1_000, free=9_000)
            with mock.patch.object(remote_runner.shutil, "disk_usage", return_value=usage) as disk_usage:
                remote_runner._require_candidate_disk_capacity(
                    attempt,
                    state,
                    minimum_free_bytes=1_000,
                )
            self.assertEqual(disk_usage.call_count, 1)

    def test_completed_attempt_cleanup_is_confined_to_one_direct_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            attempt = root / "attempt-1"
            attempt.mkdir()
            (attempt / "checkpoint.pt").write_bytes(b"checkpoint")
            remote_runner._remove_disposable_attempt_tree(attempt, root)
            self.assertFalse(attempt.exists())
            nested = root / "outer" / "attempt-2"
            nested.mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, "unsafe"):
                remote_runner._remove_disposable_attempt_tree(nested, root)

    def _args(self, root: pathlib.Path) -> types.SimpleNamespace:
        payload = root / "payload"
        payload.mkdir()
        candidate = payload / "train.py"
        metadata = payload / "registered_metadata.json"
        candidate.write_text("# candidate\n")
        metadata.write_text("{}\n")
        return types.SimpleNamespace(
            payload_dir=payload,
            output=root / "result.json",
            repository=pathlib.Path(__file__).resolve().parents[3],
            durable_record_root=root / "invocations",
            durable_invocation_id="nonstudy-12345678",
            durable_metadata_json=json.dumps({"study_scope": "nonstudy"}),
            durable_command_json=None,
        )

    def _running_record(self, args: types.SimpleNamespace) -> pathlib.Path:
        identity = remote_runner._invocation_identity(args)
        path = remote_runner._invocation_path(args.durable_record_root, args.durable_invocation_id)
        remote_runner._atomic_json_write(path, {
            "schema": remote_runner.DURABLE_INVOCATION_SCHEMA,
            "invocation_id": args.durable_invocation_id,
            "status": "RUNNING",
            "command_identity": identity,
            "command_sha256": remote_runner._canonical_sha256(identity),
            "metadata": {"study_scope": "nonstudy"},
            "worker_pid": None,
            "log_path": str(args.durable_record_root / "nonstudy-12345678.log"),
        })
        return path

    def test_normal_completion_writes_hash_verified_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(pathlib.Path(tmp))
            path = self._running_record(args)

            def fake_execute(**_kwargs):
                args.output.write_text(json.dumps({"schema": REMOTE_RESULT_SCHEMA, "status": "completed"}))
                return {"status": "completed"}

            with mock.patch.object(remote_runner, "execute", side_effect=fake_execute):
                self.assertEqual(remote_runner._run_durable_worker(args), 0)
            record = remote_runner._read_invocation(path)
            self.assertEqual(record["status"], "COMPLETE")
            self.assertEqual(record["result_bytes"], args.output.stat().st_size)
            self.assertEqual(record["result_sha256"], sha256_file(args.output))

    def test_fresh_durable_state_root_is_private_before_candidate_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            state = root / "fresh-state"
            args = self._args(root)
            args.output = state / "bridge_results" / "result.json"
            args.durable_record_root = state / "invocations"
            trusted_root, record_root = remote_runner._prepare_durable_state_root(args)
            self.assertEqual(trusted_root, state.resolve())
            self.assertEqual(record_root, (state / "invocations").resolve())
            self.assertEqual(trusted_root.stat().st_mode & 0o777, 0o700)
            self.assertEqual(record_root.stat().st_mode & 0o777, 0o700)
            args.output = root / "outside.json"
            with self.assertRaisesRegex(ValueError, "one trusted state root"):
                remote_runner._prepare_durable_state_root(args)

    def test_nested_bridge_result_parent_is_created_before_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            state_root = root / "state"
            output = state_root / "attempts" / "position-001" / "bridge_results" / "result.json"
            prepared = remote_runner._prepare_result_output(output, state_root)
            prepared.write_text('{"status":"completed"}\n')
            self.assertTrue(prepared.is_file())
            self.assertTrue((state_root / "attempts" / "position-001" / "bridge_results").is_dir())
            with self.assertRaisesRegex(ValueError, "below"):
                remote_runner._prepare_result_output(root / "outside.json", state_root)

    def test_reconnect_does_not_spawn_duplicate_existing_invocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(pathlib.Path(tmp))
            self._running_record(args)
            with mock.patch.object(remote_runner.subprocess, "Popen") as popen:
                self.assertEqual(remote_runner._launch_durable_invocation(args), 0)
            popen.assert_not_called()

    def test_trusted_command_completion_writes_the_same_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(pathlib.Path(tmp))
            args.durable_command_json = json.dumps([
                sys.executable,
                "-c",
                "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('{\\\"ok\\\":true}\\n')",
                str(args.output),
            ])
            path = self._running_record(args)
            self.assertEqual(remote_runner._run_durable_worker(args), 0)
            record = remote_runner._read_invocation(path)
            self.assertEqual(record["status"], "COMPLETE")
            self.assertEqual(record["result_sha256"], sha256_file(args.output))

    def test_dead_running_process_fails_closed(self):
        record = {"status": "RUNNING", "worker_pid": 2_147_483_647, "invocation_id": "nonstudy-12345678"}
        self.assertFalse(remote_runner._process_matches(record))

    def test_scientific_failure_is_committed_once_but_infrastructure_is_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            campaign = controller.CAMPAIGNS["T"]
            row = controller._schedule_from_frozen_file()[1]
            self.assertEqual(row.contract_id, "T")
            paths = controller._campaign_paths(root, campaign)
            paths["root"].mkdir(parents=True)
            paths["source_dir"].mkdir()
            baseline_source = paths["source_dir"] / "baseline.train.py"
            baseline_source.write_text("# baseline\n")
            paths["train"].write_text("# baseline\n")
            baseline = {"baseline_experiment_id": "baseline", "record_scope": "same_seed_baseline"}
            controller._atomic_json_write(
                paths["view"],
                build_campaign_view(campaign_id=row.campaign_id, contract_id=row.contract_id, baseline=baseline, frontier=[], attempt_number=row.campaign_attempt),
            )
            controller._append_history(paths["source_registry"], {
                "experiment_id": "baseline",
                "campaign_id": row.campaign_id,
                "source_sha256": sha256_file(baseline_source),
                "source_path": str(baseline_source),
                "parent_eligible": True,
            })
            state = {"positions": {str(row.global_position): {"status": "running"}}}
            state_path = root / "controller_state.json"
            candidate = root / "candidate.train.py"
            candidate.write_text("# failed child\n")
            metadata = {
                "candidate_source_before_sha256": sha256_file(baseline_source),
                "candidate_source_after_sha256": sha256_file(candidate),
                "hypothesis": "fixture",
                "intended_change": "fixture",
                "change_category": "fixture",
            }
            controller._persist_pending_execution(
                paths=paths, row=row, state=state, state_path=state_path,
                parent_experiment_id="baseline", parent_source=baseline_source,
                candidate=candidate, metadata=metadata,
                reservation={"status": "running", "provenance": {}}, scientist={},
            )
            controller._commit_pending_result(
                state=state, state_path=state_path, row=row, paths=paths,
                remote_result={
                    "schema": REMOTE_RESULT_SCHEMA,
                    "status": "scientific_failure",
                    "failure_class": "candidate",
                    "campaign_id": row.campaign_id,
                    "attempt": row.campaign_attempt,
                    "error": "fixture OOM",
                },
            )
            self.assertEqual(state["positions"][str(row.global_position)]["status"], "completed")
            self.assertEqual(state["positions"][str(row.global_position)]["outcome"], "scientific_failure")
            interrupted = {"positions": {str(row.global_position): {"status": "running"}}}
            controller._mark_reconciliation_required(
                state=interrupted, state_path=root / "interrupted_state.json", row=row,
                reason={"type": "TransportLoss"},
            )
            self.assertEqual(interrupted["positions"][str(row.global_position)]["status"], "reconciliation_required")

    def test_harness_failure_is_not_committed_or_added_to_campaign_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            campaign = controller.CAMPAIGNS["N"]
            row = controller._schedule_from_frozen_file()[0]
            paths = controller._campaign_paths(root, campaign)
            paths["root"].mkdir(parents=True)
            paths["source_dir"].mkdir()
            parent = paths["source_dir"] / "baseline.train.py"
            parent.write_text("# baseline\n")
            paths["train"].write_text("# baseline\n")
            controller._atomic_json_write(paths["view"], build_campaign_view(
                campaign_id=row.campaign_id, contract_id=row.contract_id,
                baseline={"baseline_experiment_id": "baseline", "record_scope": "same_seed_baseline"},
                frontier=[], attempt_number=row.campaign_attempt,
            ))
            controller._append_history(paths["source_registry"], {
                "experiment_id": "baseline", "campaign_id": row.campaign_id,
                "source_sha256": sha256_file(parent), "source_path": str(parent), "parent_eligible": True,
            })
            state = {"positions": {str(row.global_position): {"status": "running"}}}
            state_path = root / "controller_state.json"
            child = root / "child.train.py"
            child.write_text("# child\n")
            metadata = {
                "candidate_source_before_sha256": sha256_file(parent),
                "candidate_source_after_sha256": sha256_file(child),
                "hypothesis": "fixture", "intended_change": "fixture", "change_category": "fixture",
            }
            controller._persist_pending_execution(
                paths=paths, row=row, state=state, state_path=state_path,
                parent_experiment_id="baseline", parent_source=parent, candidate=child,
                metadata=metadata, reservation={"status": "running", "provenance": {}}, scientist={},
            )
            with self.assertRaisesRegex(RuntimeError, "was not consumed"):
                controller._commit_pending_result(
                    state=state, state_path=state_path, row=row, paths=paths,
                    remote_result={
                        "schema": REMOTE_RESULT_SCHEMA,
                        "status": "harness_failure",
                        "failure_class": "harness",
                        "campaign_id": row.campaign_id,
                        "attempt": row.campaign_attempt,
                        "error": "CandidateObserver.note_progress() takes 2 positional arguments but 5 were given",
                    },
                )
            self.assertEqual(state["positions"][str(row.global_position)]["status"], "reconciliation_required")
            self.assertFalse(paths["history"].exists())

    def test_quality_valid_speed_uncertain_parent_is_staged_by_exact_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            row = controller._schedule_from_frozen_file()[1]
            campaign = controller.CAMPAIGNS[row.contract_id]
            paths = controller._campaign_paths(root, campaign)
            paths["root"].mkdir(parents=True)
            paths["source_dir"].mkdir()
            parent = paths["source_dir"] / "quality-valid.train.py"
            parent.write_text("# exact eligible source\n")
            controller._append_history(paths["source_registry"], {
                "experiment_id": "quality-valid-speed-uncertain",
                "campaign_id": row.campaign_id,
                "source_sha256": sha256_file(parent),
                "source_path": str(parent),
                "parent_eligible": True,
                "pareto_eligible": False,
                "quality_status": "quality_valid_speed_uncertain",
            })
            selected, staged, digest = controller._validate_parent_selection(
                paths=paths,
                row=row,
                proposal={"parent_experiment_id": "quality-valid-speed-uncertain"},
            )
            self.assertEqual(selected, "quality-valid-speed-uncertain")
            self.assertEqual(staged, parent.resolve())
            self.assertEqual(digest, sha256_file(parent))

    def test_transport_loss_leaves_running_position_for_idempotent_reconciliation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            campaign = controller.CAMPAIGNS["T"]
            row = controller._schedule_from_frozen_file()[1]
            paths = controller._campaign_paths(root, campaign)
            paths["root"].mkdir(parents=True)
            paths["source_dir"].mkdir()
            parent = paths["source_dir"] / "baseline.train.py"
            parent.write_text("# baseline\n")
            paths["train"].write_text("# baseline\n")
            controller._atomic_json_write(paths["view"], build_campaign_view(
                campaign_id=row.campaign_id, contract_id=row.contract_id,
                baseline={"baseline_experiment_id": "baseline", "record_scope": "same_seed_baseline"},
                frontier=[], attempt_number=row.campaign_attempt,
            ))
            controller._append_history(paths["source_registry"], {
                "experiment_id": "baseline", "campaign_id": row.campaign_id,
                "source_sha256": sha256_file(parent), "source_path": str(parent), "parent_eligible": True,
            })
            state = {"positions": {str(row.global_position): {"status": "running"}}}
            state_path = root / "controller_state.json"
            child = root / "child.train.py"
            child.write_text("# child\n")
            metadata = {
                "candidate_source_before_sha256": sha256_file(parent),
                "candidate_source_after_sha256": sha256_file(child),
                "hypothesis": "fixture", "intended_change": "fixture", "change_category": "fixture",
            }
            execution = controller._persist_pending_execution(
                paths=paths, row=row, state=state, state_path=state_path,
                parent_experiment_id="baseline", parent_source=parent, candidate=child,
                metadata=metadata, reservation={"status": "running", "provenance": {}}, scientist={},
            )
            execution["remote_payload_dir"] = "/remote/payload"
            execution["dispatch_status"] = "launch_requested"
            state["positions"][str(row.global_position)]["execution"] = execution

            class LostThenComplete:
                def __init__(self):
                    self.waits = 0
                    self.launches = 0
                def launch(self, **_kwargs):
                    self.launches += 1
                    return {"record": {"status": "RUNNING"}, "process_alive": True}
                def wait_for_invocation(self, _invocation_id):
                    self.waits += 1
                    if self.waits == 1:
                        raise RuntimeError("PTY lost")
                    return {"record": {"status": "COMPLETE"}, "process_alive": False}
                def fetch_durable_result(self, _record):
                    return {
                        "schema": REMOTE_RESULT_SCHEMA, "status": "scientific_failure",
                        "failure_class": "candidate", "campaign_id": row.campaign_id,
                        "attempt": row.campaign_attempt, "error": "fixture OOM",
                    }

            transport = LostThenComplete()
            with self.assertRaisesRegex(RuntimeError, "PTY lost"):
                controller._dispatch_and_wait(
                    state=state, state_path=state_path, row=row, paths=paths, transport=transport
                )
            self.assertEqual(state["positions"][str(row.global_position)]["status"], "running")
            record = controller._resume_running_position(
                state_root=root, state=state, state_path=state_path, row=row, transport=transport
            )
            self.assertEqual(record["status"], "scientific_failure")
            self.assertEqual(state["positions"][str(row.global_position)]["status"], "completed")
            self.assertEqual(transport.launches, 2)  # remote launch is idempotent by invocation ID
