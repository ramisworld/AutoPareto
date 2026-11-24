from __future__ import annotations

import hashlib
import json
import pathlib
import tempfile
import unittest

from study.contracts_2026.campaign_controller import (
    _registered_cache_file_hashes,
    _schedule_from_frozen_file,
    _verify_local_state_config,
    _verify_transport_config,
)
from study.contracts_2026.bridge import A40RemoteConfig
from study.contracts_2026.continuation import (
    arm_continuation_payload,
    audit_completed_prefix,
    build_continuation_manifest,
    prepare_continuation_state,
    reconcile_uncommitted_position,
    sha256_file,
)


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


class FakeTransport:
    def __init__(self):
        self.transfers = 0

    def transfer(self, *, payload: pathlib.Path, remote_run_id: str) -> str:
        self.transfers += 1
        self.payload = payload
        return f"/remote/{remote_run_id}"

    def verify_payload(self, *, remote_dir: str, candidate: pathlib.Path, metadata: pathlib.Path):
        return {
            "train.py": {"bytes": candidate.stat().st_size, "sha256": sha256_file(candidate)},
            "registered_metadata.json": {"bytes": metadata.stat().st_size, "sha256": sha256_file(metadata)},
        }


class ContinuationTests(unittest.TestCase):
    def test_reconcile_uncommitted_position_requires_absence_proofs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source = self.make_source(root)
            state_path = source / "controller_state.json"
            state = json.loads(state_path.read_text())
            state["positions"]["9"]["status"] = "running"
            write_json(state_path, state)
            with self.assertRaisesRegex(RuntimeError, "absence proofs"):
                reconcile_uncommitted_position(
                    source,
                    global_position=9,
                    reason="fixture",
                    remote_process_verified_absent=False,
                    durable_receipt_verified_absent=True,
                )
            result = reconcile_uncommitted_position(
                source,
                global_position=9,
                reason="fixture",
                remote_process_verified_absent=True,
                durable_receipt_verified_absent=True,
            )
            self.assertEqual(result["status"], "reconciliation_required")
            observed = json.loads(state_path.read_text())
            self.assertEqual(observed["positions"]["9"]["status"], "reconciliation_required")
            self.assertFalse(observed["positions"]["9"]["reconciliation_failure"]["scientific_position_consumed"])

    def test_continuation_launch_paths_fail_closed_on_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            local_state = pathlib.Path(tmp) / "continued"
            manifest = {
                "a40": {
                    "ssh_host": "new@example",
                    "trusted_repository": "/opt/autopareto_trusted",
                    "rootfs_runtime": "/opt/runtime/bin/python",
                    "study_venv": "/opt/runtime",
                    "paired_baseline_checkpoints": {
                        "T": {"path": "/t"},
                        "N": {"path": "/n"},
                    },
                },
                "persistent_paths": {
                    "local_official_state": str(local_state),
                    "registered_cache": "/var/cache/autopareto/cache",
                    "trusted_state_root": "/var/lib/autopareto/continued",
                    "candidate_workspace_root": "/opt/autopareto_candidate_contexts",
                    "candidate_attempt_root": "/opt/autopareto-candidate-attempts",
                    "candidate_output_root": "/var/lib/autopareto/candidate_attempts",
                    "candidate_runtime_root": "/opt/autopareto_candidate_runtime",
                    "candidate_input_root": "/opt/autopareto_candidate_inputs/per_attempts",
                },
            }
            config = A40RemoteConfig(
                host="new@example",
                repository="/opt/autopareto_trusted",
                python_path="/opt/runtime/bin/python",
                study_venv="/opt/runtime",
                registered_cache="/var/cache/autopareto/cache",
                trusted_state_root="/var/lib/autopareto/continued",
                paired_baseline_t_checkpoint="/t",
                paired_baseline_n_checkpoint="/n",
            )
            _verify_transport_config(manifest, config)
            _verify_local_state_config(manifest, local_state)
            manifest["persistent_paths"]["candidate_runtime_root"] = "/wrong"
            with self.assertRaisesRegex(RuntimeError, "candidate-runtime paths"):
                _verify_transport_config(manifest, config)
            with self.assertRaisesRegex(RuntimeError, "state root"):
                _verify_local_state_config(manifest, pathlib.Path(tmp) / "wrong")

    def test_registered_cache_preflight_covers_every_manifest_file(self):
        expected = _registered_cache_file_hashes("/var/cache/autopareto/pair_a_registered_cache")
        manifest = json.loads(
            pathlib.Path("study/contracts_2026/provenance/data_manifest.json").read_text()
        )
        self.assertEqual(len(expected), len(manifest["files"]))
        self.assertEqual(
            expected["/var/cache/autopareto/pair_a_registered_cache/tokenizer/tokenizer.pkl"],
            "888c9fa865c72c25f9c5e9467136a9cc8655253db459fb97481deb326681c629",
        )

    def make_source(self, root: pathlib.Path, failed_positions: tuple[int, ...] = ()) -> pathlib.Path:
        source = root / "old-state"
        source.mkdir()
        schedule = [row.as_dict() for row in _schedule_from_frozen_file()]
        positions = {str(row["global_position"]): {"status": "pending"} for row in schedule}
        registries: dict[str, list[dict]] = {row["campaign_id"]: [] for row in schedule}
        latest: dict[str, tuple[str, str]] = {}
        for campaign_id in registries:
            source_dir = source / campaign_id / "sources"
            source_dir.mkdir(parents=True)
            baseline = source_dir / "baseline.train.py"
            baseline.write_text("baseline\n")
            digest = sha256_file(baseline)
            registries[campaign_id].append({
                "experiment_id": "baseline",
                "campaign_id": campaign_id,
                "source_sha256": digest,
                "source_path": str(baseline),
                "parent_eligible": True,
                "pareto_eligible": True,
            })
            latest[campaign_id] = ("baseline", digest)
            (source / campaign_id / "train.py").write_text("baseline\n")
            write_json(source / campaign_id / "campaign_view.json", {
                "campaign_id": campaign_id,
                "contract_id": "N" if "tokens" in campaign_id else "T",
                "attempt_number": 1,
                "baseline": {"record_scope": "same_seed_baseline"},
                "frontier": [],
            })
            (source / campaign_id / "history.jsonl").write_text("")

        for row in schedule[:8]:
            campaign_id = row["campaign_id"]
            attempt = row["campaign_attempt"]
            parent_id, parent_sha = latest[campaign_id]
            experiment_id = f"{campaign_id}-attempt-{attempt:02d}"
            if row["global_position"] in failed_positions:
                # As the controller records a candidate failure: a result and history, no source registry row.
                result = source / campaign_id / f"position-{row['global_position']:03d}-{campaign_id}.json"
                write_json(result, {
                    "status": "scientific_failure",
                    "failure_class": "candidate",
                    "campaign_id": campaign_id,
                    "contract_id": row["contract_id"],
                    "attempt": attempt,
                    "experiment_id": experiment_id,
                    "parent_experiment_id": parent_id,
                    "candidate_source_before_sha256": parent_sha,
                })
                positions[str(row["global_position"])] = {
                    "status": "completed", "outcome": "scientific_failure", "result_file": str(result),
                }
                continue
            candidate = source / campaign_id / "sources" / f"{experiment_id}.train.py"
            candidate.write_text(f"candidate {row['global_position']}\n")
            candidate_sha = sha256_file(candidate)
            registries[campaign_id].append({
                "experiment_id": experiment_id,
                "campaign_id": campaign_id,
                "parent_experiment_id": parent_id,
                "source_sha256": candidate_sha,
                "source_path": str(candidate),
                "parent_eligible": True,
                "pareto_eligible": False,
            })
            latest[campaign_id] = (experiment_id, candidate_sha)
            result = source / campaign_id / f"position-{row['global_position']:03d}-{campaign_id}.json"
            write_json(result, {
                "status": "completed",
                "campaign_id": campaign_id,
                "contract_id": row["contract_id"],
                "attempt": attempt,
                "experiment_id": experiment_id,
                "parent_experiment_id": parent_id,
                "candidate_source_before_sha256": parent_sha,
                "candidate_source_after_sha256": candidate_sha,
            })
            positions[str(row["global_position"])] = {"status": "completed", "result_file": str(result)}

        for campaign_id, rows in registries.items():
            (source / campaign_id / "source_registry.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
            )
            next_attempt = 5
            view = json.loads((source / campaign_id / "campaign_view.json").read_text())
            view["attempt_number"] = next_attempt
            write_json(source / campaign_id / "campaign_view.json", view)

        row = schedule[8]
        campaign_id = row["campaign_id"]
        parent_id, parent_sha = latest[campaign_id]
        pending = source / campaign_id / "pending" / "position-009"
        pending.mkdir(parents=True)
        candidate = pending / "train.py"
        candidate.write_text("prepared position nine\n")
        candidate_sha = sha256_file(candidate)
        metadata = {
            "candidate_source_after_sha256": candidate_sha,
            "candidate_source_before_sha256": parent_sha,
            "contract_parameters": {"n_star": 88_604_672, "paired_baseline_checkpoint": "/old/n.pt"},
            "evaluator_hashes": {"evaluator": "a" * 64},
            "hypothesis": "fixture",
            "intended_change": "fixture",
            "change_category": "fixture",
        }
        write_json(pending / "registered_metadata.json", metadata)
        metadata_file_sha = sha256_file(pending / "registered_metadata.json")
        old_execution = {
            "schema": "autopareto_pending_execution_v1",
            "remote_run_id": "old-position-009",
            "output_name": f"position-009-{campaign_id}.json",
            "invocation_id": "pair-a-009-oldinvocation000000000000",
            "parent_experiment_id": parent_id,
            "parent_source_path": registries[campaign_id][-1]["source_path"],
            "candidate_path": str(candidate),
            "metadata_path": str(pending / "registered_metadata.json"),
            "candidate_source_before_sha256": parent_sha,
            "candidate_source_after_sha256": candidate_sha,
            "metadata_sha256": metadata_file_sha,
            "reservation": {
                "campaign_id": campaign_id,
                "campaign_attempt": row["campaign_attempt"],
                "manifest_path": "/old/manifest.json",
                "manifest_sha256": "1" * 64,
                "a40": {"gpu_uuid": "GPU-old", "manifest_sha256": "1" * 64},
            },
            "scientist": {"handoff": {"proposal": {"hypothesis": "fixture"}}},
            "dispatch_status": "launch_requested",
        }
        positions["9"] = {
            "status": "reconciliation_required",
            "campaign_id": campaign_id,
            "campaign_attempt": row["campaign_attempt"],
            "execution": old_execution,
        }
        write_json(source / "controller_state.json", {
            "schema": "autopareto_pair_a_controller_v1",
            "study_id": "autopareto-contracts-2026-pair-a",
            "n_star": 88_604_672,
            "a40_uuid": "GPU-old",
            "manifest_sha256": "1" * 64,
            "scientist_config": {"model": "gpt-test"},
            "schedule_sha256": "schedule",
            "schedule": schedule,
            "positions": positions,
        })
        return source

    def test_audit_and_prepare_preserve_prefix_and_exact_next_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source = self.make_source(root)
            source_state_sha = sha256_file(source / "controller_state.json")
            audit = audit_completed_prefix(source, completed_positions=8)
            self.assertEqual(audit["completed_positions"], 8)
            self.assertEqual(audit["next_position"], 9)
            manifest = root / "manifest.json"
            manifest.write_text('{"manifest":"fixture"}\n')
            destination = root / "continued"
            prepared = prepare_continuation_state(
                source_root=source,
                destination_root=destination,
                manifest_source=manifest,
                manifest_relative_path=pathlib.PurePosixPath("provenance/final.json"),
                completed_positions=8,
                new_gpu_uuid="GPU-new",
                new_remote_state_root="/var/lib/autopareto/continued",
                paired_baseline_n_checkpoint="/var/lib/autopareto/n.pt",
                evaluator_hashes={"evaluator": "b" * 64},
                remote_run_namespace="continuation-test",
            )
            self.assertEqual(sha256_file(source / "controller_state.json"), source_state_sha)
            state = json.loads((destination / "controller_state.json").read_text())
            self.assertEqual(sum(v["status"] == "completed" for v in state["positions"].values()), 8)
            self.assertEqual(state["positions"]["9"]["status"], "running")
            execution = state["positions"]["9"]["execution"]
            self.assertEqual(execution["dispatch_status"], "prepared")
            self.assertEqual(execution["candidate_source_after_sha256"], audit["next_candidate_sha256"])
            self.assertEqual(execution["candidate_source_before_sha256"], audit["next_parent_source_sha256"])
            metadata = json.loads(pathlib.Path(execution["metadata_path"]).read_text())
            self.assertEqual(metadata["contract_parameters"]["paired_baseline_checkpoint"], "/var/lib/autopareto/n.pt")
            self.assertTrue((destination / "migration" / "abandoned_position_009.json").is_file())
            self.assertEqual(prepared["next_global_position"], 9)

            transport = FakeTransport()
            armed = arm_continuation_payload(state_root=destination, transport=transport)
            self.assertEqual(armed["status"], "armed")
            self.assertEqual(transport.transfers, 1)
            state = json.loads((destination / "controller_state.json").read_text())
            self.assertEqual(state["positions"]["9"]["execution"]["dispatch_status"], "launch_requested")
            armed_again = arm_continuation_payload(state_root=destination, transport=transport)
            self.assertEqual(armed_again["status"], "already_armed")
            self.assertEqual(transport.transfers, 1)

    def test_prefix_with_a_recorded_candidate_failure_is_accepted_and_carried_over(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            failed = 3
            source = self.make_source(root, failed_positions=(failed,))
            audit = audit_completed_prefix(source, completed_positions=8)
            rows = {row["global_position"]: row for row in audit["completed_results"]}
            self.assertEqual(rows[failed]["status"], "scientific_failure")
            self.assertIsNone(rows[failed]["candidate_source_sha256"])
            manifest = root / "manifest.json"
            manifest.write_text('{"manifest":"fixture"}\n')
            destination = root / "continued"
            prepare_continuation_state(
                source_root=source, destination_root=destination,
                manifest_source=manifest, manifest_relative_path=pathlib.PurePosixPath("provenance/final.json"),
                completed_positions=8, new_gpu_uuid="GPU-new", new_remote_state_root="/var/lib/autopareto/continued",
                paired_baseline_n_checkpoint="/var/lib/autopareto/n.pt", evaluator_hashes={"evaluator": "b" * 64},
                remote_run_namespace="continuation-test",
            )
            state = json.loads((destination / "controller_state.json").read_text())
            self.assertEqual(state["positions"][str(failed)]["outcome"], "scientific_failure")
            self.assertTrue(
                pathlib.Path(state["positions"][str(failed)]["result_file"]).is_relative_to(destination.resolve()))

    def test_chained_continuation_keeps_predecessor_migration_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source = self.make_source(root)
            write_json(source / "migration" / "abandoned_position_004.json", {"predecessor": True})
            manifest = root / "manifest.json"
            manifest.write_text('{"manifest":"fixture"}\n')
            arguments = dict(
                source_root=source, manifest_source=manifest,
                manifest_relative_path=pathlib.PurePosixPath("provenance/final.json"), completed_positions=8,
                new_gpu_uuid="GPU-new", new_remote_state_root="/var/lib/autopareto/continued",
                paired_baseline_n_checkpoint="/var/lib/autopareto/n.pt", evaluator_hashes={"evaluator": "b" * 64},
                remote_run_namespace="continuation-test",
            )
            destination = root / "continued"
            prepare_continuation_state(destination_root=destination, **arguments)
            migration = destination / "migration"
            self.assertEqual(sorted(path.name for path in migration.iterdir()),
                             ["abandoned_position_004.json", "abandoned_position_009.json"])
            self.assertEqual(json.loads((migration / "abandoned_position_004.json").read_text()), {"predecessor": True})
            write_json(source / "migration" / "abandoned_position_009.json", {"predecessor": True})
            with self.assertRaisesRegex(FileExistsError, "position 9 already exists"):
                prepare_continuation_state(destination_root=root / "again", **arguments)
            self.assertFalse((root / "again").exists())

    def test_prefix_audit_rejects_a_failure_registered_as_a_source_or_an_outcome_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            failed = 3
            source = self.make_source(root, failed_positions=(failed,))
            state_path = source / "controller_state.json"
            state = json.loads(state_path.read_text())
            del state["positions"][str(failed)]["outcome"]
            write_json(state_path, state)
            with self.assertRaisesRegex(RuntimeError, "outcome disagrees"):
                audit_completed_prefix(source, completed_positions=8)
            state["positions"][str(failed)]["outcome"] = "scientific_failure"
            write_json(state_path, state)
            result = json.loads(pathlib.Path(state["positions"][str(failed)]["result_file"]).read_text())
            campaign_id = result["campaign_id"]
            fake = source / campaign_id / "sources" / f"{result['experiment_id']}.train.py"
            fake.write_text("failed candidate\n")
            registry = source / campaign_id / "source_registry.jsonl"
            registry.write_text(registry.read_text() + json.dumps({
                "experiment_id": result["experiment_id"], "campaign_id": campaign_id,
                "source_sha256": sha256_file(fake), "source_path": str(fake), "parent_eligible": False,
            }) + "\n")
            with self.assertRaisesRegex(RuntimeError, "registered as a source"):
                audit_completed_prefix(source, completed_positions=8)

    def test_amendment_record_describes_a_same_gpu_harness_fix_accurately(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source = self.make_source(root)
            repo = root / "repo"
            (repo / "x").mkdir(parents=True)
            (repo / "x/file.py").write_text("fixed\n")
            prior = root / "prior.json"
            write_json(prior, {"immutable_deployment": {"x/file.py": "0" * 64}})
            runs = [{"run_index": i, "total_tokens": 88_604_672, "optimizer_updates": 169,
                     "measured_training_seconds": 300.1, "error": None} for i in (1, 2, 3)]
            common = dict(
                prior_manifest_path=prior, source_state_root=source, completed_positions=8, repository_root=repo,
                new_gpu_uuid="GPU-same", new_ssh_host="same@example", new_local_state_root=str(root / "continued"),
                new_remote_state_root="/var/lib/autopareto/continued", registered_cache="/var/cache/autopareto/cache",
                remote_python="/opt/runtime/bin/python", study_venv="/opt/runtime",
                paired_baseline_checkpoints={"T": {"path": "/t", "sha256": "a" * 64}, "N": {"path": "/n", "sha256": "b" * 64}},
                calibration_result={"choice": {"n_star": 88_604_672, "token_quantum": 524_288, "source_runs": runs}},
                calibration_path="/remote/calibration.json", calibration_sha256="c" * 64,
                registered_cache_runtime_provenance={"status": "verified", "files": []},
            )
            plain = build_continuation_manifest(**common)
            self.assertNotIn("transition_kind", plain["continuation"])
            record = {
                "transition_kind": "protected_harness_fix_same_a40",
                "amendments": [{"file": "x/file.py", "reason": "fp32 adjudication defect"}],
                "scientific_closure_changed": False,
                "abandoned_invocation": {
                    "reason": "protected harness defect after training",
                    "durable_receipt_verified_absent": False,
                    "durable_receipt": {"status": "FAILED"},
                },
            }
            amended = build_continuation_manifest(**common, amendment_record=record)
            continuation = amended["continuation"]
            self.assertEqual(continuation["transition_kind"], "protected_harness_fix_same_a40")
            self.assertEqual(continuation["amendments"], record["amendments"])
            abandoned = continuation["abandoned_unconsumed_invocation"]
            self.assertEqual(abandoned["reason"], "protected harness defect after training")
            self.assertFalse(abandoned["durable_receipt_verified_absent"])
            self.assertTrue(abandoned["remote_process_verified_absent"])
            self.assertFalse(abandoned["scientific_position_consumed"])
            self.assertEqual(amended["immutable_deployment"], plain["immutable_deployment"])
            # A file the manifest also pins elsewhere (the evaluator) moves only with a listed amendment.
            write_json(prior, {"immutable_deployment": {"x/file.py": "0" * 64},
                               "inference": {"evaluator": "x/file.py", "evaluator_sha256": "0" * 64}})
            with self.assertRaisesRegex(RuntimeError, "pinned as evaluator_sha256"):
                build_continuation_manifest(**common)
            pinned = build_continuation_manifest(**common, amendment_record=record)
            self.assertEqual(pinned["inference"]["evaluator_sha256"], pinned["immutable_deployment"]["x/file.py"])
            unlisted = {**record, "amendments": [{"file": "other.py"}]}
            with self.assertRaisesRegex(RuntimeError, "pinned as evaluator_sha256"):
                build_continuation_manifest(**common, amendment_record=unlisted)
            write_json(prior, {"immutable_deployment": {"x/file.py": "0" * 64}})
            for bad in ({**record, "amendments": []}, {**record, "transition_kind": ""},
                        {**record, "abandoned_invocation": {"scientific_position_consumed": True}}):
                with self.assertRaises(ValueError):
                    build_continuation_manifest(**common, amendment_record=bad)

            manifest = root / "manifest.json"
            manifest.write_text('{"manifest":"fixture"}\n')
            with self.assertRaises(ValueError):
                prepare_continuation_state(
                    source_root=source, destination_root=root / "rejected",
                    manifest_source=manifest, manifest_relative_path=pathlib.PurePosixPath("provenance/final.json"),
                    completed_positions=8, new_gpu_uuid="GPU-same", new_remote_state_root="/var/lib/autopareto/continued",
                    paired_baseline_n_checkpoint="/var/lib/autopareto/n.pt", evaluator_hashes={"evaluator": "b" * 64},
                    remote_run_namespace="continuation-test", abandoned_invocation={"invocation_id": "other"},
                )
            self.assertFalse((root / "rejected").exists())
            destination = root / "continued"
            prepare_continuation_state(
                source_root=source, destination_root=destination,
                manifest_source=manifest, manifest_relative_path=pathlib.PurePosixPath("provenance/final.json"),
                completed_positions=8, new_gpu_uuid="GPU-same", new_remote_state_root="/var/lib/autopareto/continued",
                paired_baseline_n_checkpoint="/var/lib/autopareto/n.pt", evaluator_hashes={"evaluator": "b" * 64},
                remote_run_namespace="continuation-test", abandoned_invocation=record["abandoned_invocation"],
            )
            migration = json.loads((destination / "migration" / "abandoned_position_009.json").read_text())
            self.assertFalse(migration["durable_receipt_verified_absent"])
            self.assertEqual(migration["durable_receipt"], {"status": "FAILED"})
            self.assertFalse(migration["scientific_position_consumed"])

    def test_manifest_accepts_at_most_one_quantum_hardware_epoch_delta_without_changing_n_star(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source = self.make_source(root)
            repo = root / "repo"
            (repo / "x").mkdir(parents=True)
            (repo / "x/file.py").write_text("x\n")
            prior = root / "prior.json"
            write_json(prior, {"immutable_deployment": {"x/file.py": "0" * 64}})
            calibration = {
                "choice": {
                    "n_star": 88_604_672,
                    "token_quantum": 524_288,
                    "source_runs": [
                        {
                            "run_index": i,
                            "total_tokens": 88_604_672,
                            "optimizer_updates": 169,
                            "measured_training_seconds": 300.1,
                            "error": None,
                        }
                        for i in (1, 2, 3)
                    ],
                }
            }
            manifest = build_continuation_manifest(
                prior_manifest_path=prior,
                source_state_root=source,
                completed_positions=8,
                repository_root=repo,
                new_gpu_uuid="GPU-new",
                new_ssh_host="new@example",
                new_local_state_root=str(root / "continued"),
                new_remote_state_root="/var/lib/autopareto/continued",
                registered_cache="/var/cache/autopareto/cache",
                remote_python="/opt/runtime/bin/python",
                study_venv="/opt/runtime",
                paired_baseline_checkpoints={"T": {"path": "/t", "sha256": "a" * 64}, "N": {"path": "/n", "sha256": "b" * 64}},
                calibration_result=calibration,
                calibration_path="/remote/calibration.json",
                calibration_sha256="c" * 64,
                registered_cache_runtime_provenance={"status": "verified", "files": []},
            )
            self.assertEqual(manifest["continuation"]["next_global_position"], 9)
            self.assertEqual(manifest["a40"]["gpu_uuid"], "GPU-new")
            self.assertEqual(
                manifest["continuation"]["replacement_a40_calibration"]["frozen_continuation_n_star"],
                88_604_672,
            )
            bad = json.loads(json.dumps(calibration))
            bad["choice"]["n_star"] = 88_080_384
            for run in bad["choice"]["source_runs"]:
                run["total_tokens"] = 88_080_384
                run["optimizer_updates"] = 168
            one_quantum = build_continuation_manifest(
                prior_manifest_path=prior,
                source_state_root=source,
                completed_positions=8,
                repository_root=repo,
                new_gpu_uuid="GPU-new",
                new_ssh_host="new@example",
                new_local_state_root=str(root / "continued"),
                new_remote_state_root="/var/lib/autopareto/continued",
                registered_cache="/var/cache/autopareto/cache",
                remote_python="/opt/runtime/bin/python",
                study_venv="/opt/runtime",
                paired_baseline_checkpoints={"T": {"path": "/t", "sha256": "a" * 64}, "N": {"path": "/n", "sha256": "b" * 64}},
                calibration_result=bad,
                calibration_path="/remote/calibration.json",
                calibration_sha256="c" * 64,
                registered_cache_runtime_provenance={"status": "verified", "files": []},
            )
            epoch = one_quantum["continuation"]["replacement_a40_calibration"]
            self.assertEqual(epoch["calibrated_n_star"], 88_080_384)
            self.assertEqual(epoch["frozen_continuation_n_star"], 88_604_672)
            self.assertEqual(epoch["delta_legal_updates"], -1)

            too_far = json.loads(json.dumps(bad))
            too_far["choice"]["n_star"] = 87_556_096
            for run in too_far["choice"]["source_runs"]:
                run["total_tokens"] = 87_556_096
                run["optimizer_updates"] = 167
            with self.assertRaisesRegex(RuntimeError, "more than one legal update"):
                build_continuation_manifest(
                    prior_manifest_path=prior,
                    source_state_root=source,
                    completed_positions=8,
                    repository_root=repo,
                    new_gpu_uuid="GPU-new",
                    new_ssh_host="new@example",
                    new_local_state_root=str(root / "continued"),
                    new_remote_state_root="/var/lib/autopareto/continued",
                    registered_cache="/var/cache/autopareto/cache",
                    remote_python="/opt/runtime/bin/python",
                    study_venv="/opt/runtime",
                    paired_baseline_checkpoints={"T": {"path": "/t", "sha256": "a" * 64}, "N": {"path": "/n", "sha256": "b" * 64}},
                    calibration_result=too_far,
                    calibration_path="/remote/calibration.json",
                    calibration_sha256="c" * 64,
                    registered_cache_runtime_provenance={"status": "verified", "files": []},
                )

    def test_audit_rejects_symlinked_campaign_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source = self.make_source(root)
            result = pathlib.Path(json.loads((source / "controller_state.json").read_text())["positions"]["1"]["result_file"])
            real_result = result.with_suffix(".real.json")
            result.rename(real_result)
            result.symlink_to(real_result.name)
            with self.assertRaisesRegex(RuntimeError, "symlinked"):
                audit_completed_prefix(source, completed_positions=8)

    def test_continuation_destination_cannot_be_nested_in_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source = self.make_source(root)
            manifest = root / "manifest.json"
            manifest.write_text('{"manifest":"fixture"}\n')
            with self.assertRaisesRegex(ValueError, "separate from source"):
                prepare_continuation_state(
                    source_root=source,
                    destination_root=source / "nested" / "continued",
                    manifest_source=manifest,
                    manifest_relative_path=pathlib.PurePosixPath("provenance/final.json"),
                    completed_positions=8,
                    new_gpu_uuid="GPU-new",
                    new_remote_state_root="/var/lib/autopareto/continued",
                    paired_baseline_n_checkpoint="/var/lib/autopareto/n.pt",
                    evaluator_hashes={"evaluator": "b" * 64},
                    remote_run_namespace="continuation-test",
                )

    def make_single_arm_source(self, root: pathlib.Path, *, keep_active_contracts: bool = True) -> tuple[pathlib.Path, int]:
        """A T-only state shaped like the Claude study: one completed T attempt, the next T
        attempt reconciliation-required with a durable execution record, every N position
        not_in_study, and no N campaign directory."""
        source = root / "single-arm-state"
        source.mkdir()
        schedule = [row.as_dict() for row in _schedule_from_frozen_file()]
        active = [row for row in schedule if row["contract_id"] == "T"]
        first, second = active[0], active[1]
        campaign_id = first["campaign_id"]
        positions = {
            str(row["global_position"]): {"status": "pending" if row["contract_id"] == "T" else "not_in_study"}
            for row in schedule
        }
        sources = source / campaign_id / "sources"
        sources.mkdir(parents=True)
        baseline = sources / "baseline.train.py"
        baseline.write_text("baseline\n")
        baseline_sha = sha256_file(baseline)
        candidate = sources / f"{campaign_id}-attempt-01.train.py"
        candidate.write_text("candidate one\n")
        candidate_sha = sha256_file(candidate)
        registry = [
            {"experiment_id": "baseline", "campaign_id": campaign_id, "source_sha256": baseline_sha,
             "source_path": str(baseline), "parent_eligible": True, "pareto_eligible": True},
            {"experiment_id": f"{campaign_id}-attempt-01", "campaign_id": campaign_id,
             "parent_experiment_id": "baseline", "source_sha256": candidate_sha,
             "source_path": str(candidate), "parent_eligible": True, "pareto_eligible": True},
        ]
        (source / campaign_id / "source_registry.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in registry)
        )
        (source / campaign_id / "train.py").write_text("candidate one\n")
        result = source / campaign_id / f"position-{first['global_position']:03d}-{campaign_id}.json"
        write_json(result, {
            "status": "completed", "campaign_id": campaign_id, "contract_id": "T",
            "attempt": first["campaign_attempt"], "experiment_id": f"{campaign_id}-attempt-01",
            "parent_experiment_id": "baseline", "candidate_source_before_sha256": baseline_sha,
            "candidate_source_after_sha256": candidate_sha,
        })
        positions[str(first["global_position"])] = {"status": "completed", "result_file": str(result)}
        next_position = int(second["global_position"])
        pending = source / campaign_id / "pending" / f"position-{next_position:03d}"
        pending.mkdir(parents=True)
        next_candidate = pending / "train.py"
        next_candidate.write_text("prepared attempt two\n")
        next_sha = sha256_file(next_candidate)
        write_json(pending / "registered_metadata.json", {
            "candidate_source_after_sha256": next_sha,
            "candidate_source_before_sha256": candidate_sha,
            "contract_parameters": {"paired_baseline_checkpoint": "/old/t.pt"},
            "evaluator_hashes": {"evaluator": "a" * 64},
        })
        positions[str(next_position)] = {
            "status": "reconciliation_required",
            "campaign_id": campaign_id,
            "campaign_attempt": second["campaign_attempt"],
            "reconciliation_failure": {"type": "RemoteInfrastructureFailure", "receipt": {"status": "FAILED"}},
            "execution": {
                "schema": "autopareto_pending_execution_v1",
                "remote_run_id": f"old-position-{next_position:03d}",
                "output_name": f"position-{next_position:03d}-{campaign_id}.json",
                "invocation_id": f"pair-a-{next_position:03d}-oldinvocation",
                "parent_experiment_id": f"{campaign_id}-attempt-01",
                "parent_source_path": str(candidate),
                "candidate_path": str(next_candidate),
                "metadata_path": str(pending / "registered_metadata.json"),
                "candidate_source_before_sha256": candidate_sha,
                "candidate_source_after_sha256": next_sha,
                "metadata_sha256": sha256_file(pending / "registered_metadata.json"),
                "reservation": {"campaign_id": campaign_id, "campaign_attempt": second["campaign_attempt"],
                                "a40": {"gpu_uuid": "GPU-old", "manifest_sha256": "1" * 64}},
                "scientist": {"handoff": {"proposal": {"hypothesis": "fixture"}}},
                "dispatch_status": "launch_requested",
            },
        }
        state = {
            "schema": "autopareto_pair_a_controller_v1",
            "study_id": "autopareto-contracts-2026-pair-a",
            "a40_uuid": "GPU-old",
            "manifest_sha256": "1" * 64,
            "schedule": schedule,
            "positions": positions,
        }
        if keep_active_contracts:
            state["active_contracts"] = ["T"]
        write_json(source / "controller_state.json", state)
        return source, next_position

    def test_single_arm_continuation_skips_inactive_positions_and_rearms_the_exact_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source, next_position = self.make_single_arm_source(root)
            source_state_sha = sha256_file(source / "controller_state.json")
            audit = audit_completed_prefix(source, completed_positions=next_position - 1)
            self.assertEqual([row["campaign_attempt"] for row in audit["completed_results"]], [1])
            self.assertEqual(audit["next_position"], next_position)
            manifest = root / "manifest.json"
            manifest.write_text('{"manifest":"fixture"}\n')
            destination = root / "continued"
            prepare_continuation_state(
                source_root=source,
                destination_root=destination,
                manifest_source=manifest,
                manifest_relative_path=pathlib.PurePosixPath("provenance/final.json"),
                completed_positions=next_position - 1,
                new_gpu_uuid="GPU-new",
                new_remote_state_root="/var/lib/autopareto/continued",
                paired_baseline_n_checkpoint="/var/lib/autopareto/n.pt",
                paired_baseline_t_checkpoint="/var/lib/autopareto/t.pt",
                evaluator_hashes={"evaluator": "b" * 64},
                remote_run_namespace="continuation-test",
            )
            self.assertEqual(sha256_file(source / "controller_state.json"), source_state_sha)
            state = json.loads((destination / "controller_state.json").read_text())
            statuses = [state["positions"][str(i)]["status"] for i in range(1, 51)]
            self.assertEqual(statuses.count("completed"), 1)
            self.assertEqual(statuses.count("running"), 1)
            self.assertEqual(statuses.count("not_in_study"), 25)
            self.assertEqual(statuses.count("pending"), 23)
            self.assertEqual(state["active_contracts"], ["T"])
            self.assertEqual(state["a40_uuid"], "GPU-new")
            execution = state["positions"][str(next_position)]["execution"]
            self.assertEqual(execution["dispatch_status"], "prepared")
            self.assertEqual(execution["candidate_source_after_sha256"], audit["next_candidate_sha256"])
            self.assertNotEqual(execution["invocation_id"], f"pair-a-{next_position:03d}-oldinvocation")
            metadata = json.loads(pathlib.Path(execution["metadata_path"]).read_text())
            self.assertEqual(metadata["contract_parameters"]["paired_baseline_checkpoint"], "/var/lib/autopareto/t.pt")
            abandoned = json.loads((destination / "migration" / f"abandoned_position_{next_position:03d}.json").read_text())
            self.assertEqual(abandoned["old_position_record"]["reconciliation_failure"]["receipt"]["status"], "FAILED")
            self.assertEqual(sorted(p.name for p in destination.iterdir() if p.is_dir()),
                             sorted(["migration", "provenance", execution["reservation"]["campaign_id"]]))
            # Arming must pick the re-armed studied position, not the first not_in_study one.
            transport = FakeTransport()
            armed = arm_continuation_payload(state_root=destination, transport=transport)
            self.assertEqual(armed["status"], "armed")
            self.assertEqual(transport.transfers, 1)
            self.assertEqual(transport.payload.name, f"position-{next_position:03d}")
            state = json.loads((destination / "controller_state.json").read_text())
            armed_execution = state["positions"][str(next_position)]["execution"]
            self.assertEqual(armed_execution["dispatch_status"], "launch_requested")
            self.assertEqual(armed_execution["remote_payload_dir"], f"/remote/{execution['remote_run_id']}")
            self.assertEqual([state["positions"][str(i)]["status"] for i in range(1, 51)], statuses)
            armed_again = arm_continuation_payload(state_root=destination, transport=transport)
            self.assertEqual(armed_again["status"], "already_armed")
            self.assertEqual(transport.transfers, 1)

    def test_inactive_positions_are_rejected_without_a_single_arm_declaration(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, next_position = self.make_single_arm_source(pathlib.Path(tmp), keep_active_contracts=False)
            with self.assertRaisesRegex(RuntimeError, "is not completed"):
                audit_completed_prefix(source, completed_positions=next_position - 1)


if __name__ == "__main__":
    unittest.main()
