import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

from study.contracts_2026 import campaign_controller
from study.contracts_2026.bridge import (
    A40RemoteConfig,
    ScientistHandoff,
    build_registered_metadata,
    canonicalize_scientist_proposal,
    stage_remote_payload,
    update_campaign_view,
    validate_scientist_proposal,
)
from study.contracts_2026.scientist_config import ScientistConfig


class BridgeTests(unittest.TestCase):
    def test_scientist_payload_rejects_remote_access(self):
        with self.assertRaises(ValueError):
            validate_scientist_proposal({"hypothesis": "x", "intended_change": "y", "change_category": "z", "ssh": "no"})

    def test_remote_payload_contains_only_candidate_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            candidate = root / "train.py"
            candidate.write_text("candidate")
            payload = stage_remote_payload(
                destination=root / "payload",
                candidate=candidate,
                metadata={"schema": "autopareto_a40_payload_v1", "candidate_source_after_sha256": "x"},
            )
            self.assertEqual(sorted(path.name for path in payload.iterdir()), ["registered_metadata.json", "train.py"])

    def test_registered_metadata_has_no_transport_configuration(self):
        request = {
            "model_identifier": "test-model",
            "reasoning_setting": "high",
            "api_configuration_available": False,
        }
        handoff = ScientistHandoff(
            "autopareto_trusted_bridge_v1", "campaign", "T", 1,
            {"hypothesis": "h", "intended_change": "i", "change_category": "c"},
            "a" * 64, "b" * 64, "c" * 64, {"train.py": "b" * 64}, request,
        )
        metadata = build_registered_metadata(
            handoff=handoff,
            study_id="study",
            training_seed=3,
            starting_source_hash="d" * 64,
            evaluator_hashes={},
            dataset_hashes={},
            tokenizer_hash="e" * 64,
            contract_parameters={"time_seconds": 300.0},
            protected_prepare_sha256="f" * 64,
            study_runtime_sha256="0" * 64,
        )
        self.assertNotIn("ssh", json.dumps(metadata).lower())
        self.assertNotIn("remote_repo", json.dumps(metadata).lower())
        self.assertIn("metadata_sha256", metadata)

    def test_campaign_view_accepts_only_same_campaign_protected_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "campaign_view.json"
            path.write_text(json.dumps({
                "campaign_id": "campaign",
                "contract_id": "T",
                "baseline": {
                    "record_scope": "same_seed_baseline", "campaign_id": "campaign",
                    "protected_validation_bpb": 2.0,
                    "inference_result": {
                        "inference_status": "stable", "normalized_decode_throughput": 1.0,
                        "normalized_decode_throughput_ci_95pct": [1.0, 1.0],
                        "peak_allocated_inference_memory_bytes": 20,
                    },
                },
                "frontier": [],
            }))
            updated = update_campaign_view(
                view_path=path,
                result={
                    "campaign_id": "campaign",
                    "attempt": 1,
                    "status": "completed",
                    "protected_validation_bpb": 1.5,
                    "inference_result": {
                        "inference_status": "stable",
                        "cached_decode_tokens_per_second_median": 10.0,
                        "normalized_decode_throughput": 1.0,
                        "normalized_decode_throughput_ci_95pct": [1.0, 1.0],
                        "peak_allocated_inference_memory_bytes": 20,
                    },
                    "pareto_decision": {"decision": "retained"},
                },
            )
            self.assertEqual(updated["frontier"][0]["campaign_id"], "campaign")
            with self.assertRaises(ValueError):
                update_campaign_view(view_path=path, result={"campaign_id": "other", "attempt": 2})

    def test_campaign_view_drops_dominated_records_from_derived_frontier(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "campaign_view.json"
            path.write_text(json.dumps({
                "campaign_id": "campaign",
                "contract_id": "T",
                "baseline": {
                    "record_scope": "same_seed_baseline", "campaign_id": "campaign",
                    "protected_validation_bpb": 2.0,
                    "inference_result": {
                        "inference_status": "stable", "normalized_decode_throughput": 1.0,
                        "normalized_decode_throughput_ci_95pct": [1.0, 1.0],
                        "peak_allocated_inference_memory_bytes": 20,
                    },
                },
                "frontier": [{
                    "experiment_id": "old-dominated",
                    "campaign_id": "campaign",
                    "pareto_decision": {"decision": "dominated"},
                }],
            }))
            updated = update_campaign_view(
                view_path=path,
                result={
                    "campaign_id": "campaign",
                    "attempt": 2,
                    "experiment_id": "retained",
                    "status": "completed",
                    "protected_validation_bpb": 1.5,
                    "inference_result": {
                        "inference_status": "stable",
                        "cached_decode_tokens_per_second_median": 10.0,
                        "normalized_decode_throughput": 1.1,
                        "normalized_decode_throughput_ci_95pct": [1.1, 1.1],
                        "peak_allocated_inference_memory_bytes": 20,
                    },
                    "pareto_decision": {"decision": "retained"},
                },
            )
            self.assertEqual([r["experiment_id"] for r in updated["frontier"]], ["retained"])

    def test_later_dominator_retires_prior_frontier_and_parent_entry(self):
        def stable(speed, bpb, memory):
            return {
                "inference_status": "stable",
                "normalized_decode_throughput": speed,
                "normalized_decode_throughput_ci_95pct": [speed, speed],
                "peak_allocated_inference_memory_bytes": memory,
            }

        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "campaign_view.json"
            path.write_text(json.dumps({
                "campaign_id": "campaign", "contract_id": "T",
                "baseline": {
                    "record_scope": "same_seed_baseline", "campaign_id": "campaign",
                    "protected_validation_bpb": 2.0, "inference_result": stable(1.0, 2.0, 10),
                },
                "frontier": [], "same_arm_evidence": [], "parent_pool": [], "failures": [],
            }))
            first = {
                "campaign_id": "campaign", "attempt": 1, "experiment_id": "t-01", "status": "completed",
                "protected_validation_bpb": 1.9, "inference_result": stable(1.0, 1.9, 10),
                "pareto_decision": {"decision": "retained"},
            }
            update_campaign_view(view_path=path, result=first)
            updated = update_campaign_view(view_path=path, result={
                "campaign_id": "campaign", "attempt": 2, "experiment_id": "t-02", "status": "completed",
                "protected_validation_bpb": 1.8, "inference_result": stable(1.1, 1.8, 9),
                "pareto_decision": {"decision": "retained"},
            })
            self.assertEqual([r["experiment_id"] for r in updated["frontier"]], ["t-02"])
            self.assertEqual([r["experiment_id"] for r in updated["parent_pool"]], ["t-02"])
            prior = updated["same_arm_evidence"][0]
            self.assertEqual(prior["pareto_decision"]["decision"], "dominated")
            self.assertFalse(prior["parent_eligible"])

    def test_campaign_view_retains_full_nonsealed_result_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "campaign_view.json"
            path.write_text(json.dumps({
                "campaign_id": "campaign",
                "contract_id": "T",
                "baseline": {
                    "record_scope": "same_seed_baseline", "campaign_id": "campaign",
                    "protected_validation_bpb": 2.0,
                    "inference_result": {
                        "inference_status": "stable", "normalized_decode_throughput": 1.0,
                        "normalized_decode_throughput_ci_95pct": [1.0, 1.0],
                        "peak_allocated_inference_memory_bytes": 20,
                    },
                },
                "frontier": [],
                "same_arm_evidence": [],
                "parent_pool": [],
                "failures": [],
            }))
            updated = update_campaign_view(
                view_path=path,
                result={
                    "campaign_id": "campaign",
                    "attempt": 1,
                    "experiment_id": "t-01",
                    "parent_experiment_id": "baseline",
                    "hypothesis": "h",
                    "candidate_patch": "diff",
                    "status": "completed",
                    "protected_validation_bpb": 1.2,
                    "num_params": 123,
                    "total_tokens": 456,
                    "optimizer_updates": 7,
                    "inference_result": {
                        "inference_status": "stable",
                        "measurement_windows": [1, 2, 3, 4, 5, 6, 7, 8],
                        "normalized_pairwise_ratios": [1.0] * 8,
                        "normalized_range_drift": 0.01,
                        "normalized_decode_throughput": 1.0,
                        "normalized_decode_throughput_ci_95pct": [1.0, 1.0],
                        "peak_allocated_inference_memory_bytes": 20,
                    },
                    "pareto_decision": {"decision": "retained"},
                },
            )
            evidence = updated["same_arm_evidence"][0]
            self.assertEqual(evidence["num_params"], 123)
            self.assertEqual(evidence["total_tokens"], 456)
            self.assertEqual(evidence["inference"]["measurement_windows"], list(range(1, 9)))
            self.assertEqual(evidence["inference"]["normalized_pairwise_ratios"], [1.0] * 8)

    def test_campaign_view_retains_failed_candidate_lineage_without_parent_eligibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "campaign_view.json"
            path.write_text(json.dumps({
                "campaign_id": "campaign",
                "contract_id": "T",
                "baseline": {"record_scope": "same_seed_baseline", "campaign_id": "campaign"},
                "frontier": [],
                "same_arm_evidence": [],
                "parent_pool": [],
                "failures": [],
            }))
            updated = update_campaign_view(
                view_path=path,
                result={
                    "campaign_id": "campaign",
                    "attempt": 1,
                    "experiment_id": "t-01",
                    "parent_experiment_id": "baseline",
                    "hypothesis": "failed",
                    "candidate_patch": "diff",
                    "status": "scientific_failure",
                    "failure_class": "candidate",
                    "error": "NaN",
                    "quality_status": "invalid",
                    "parent_eligible": False,
                    "pareto_eligible": False,
                    "pareto_decision": {"decision": "failed"},
                },
            )
            self.assertEqual(updated["same_arm_evidence"][0]["hypothesis"], "failed")
            self.assertEqual(updated["same_arm_evidence"][0]["error"], "NaN")
            self.assertFalse(updated["lineage"][1]["parent_eligible"])

    def test_remote_defaults_are_orchestrator_only(self):
        config = A40RemoteConfig()
        command = config.ssh_base()
        self.assertIn("ssh", command[0])
        self.assertIn("ServerAliveInterval=10", command)
        self.assertIn("ServerAliveCountMax=3", command)
        self.assertNotIn("ssh", json.dumps({"program": "only-local"}))

    def test_transfer_bundles_exact_payload_in_one_verified_remote_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            payload = root / "payload"
            payload.mkdir()
            (payload / "train.py").write_text("candidate")
            (payload / "registered_metadata.json").write_text('{"schema":"x"}\n')
            config = A40RemoteConfig(host="example.invalid", payload_root="/runs")
            from study.contracts_2026.bridge import A40Transport, sha256_file

            transport = A40Transport(config)
            expected = {
                "train.py": sha256_file(payload / "train.py"),
                "registered_metadata.json": sha256_file(payload / "registered_metadata.json"),
            }
            result = json.dumps({
                name: {"bytes": (payload / name).stat().st_size, "sha256": digest}
                for name, digest in expected.items()
            }, sort_keys=True, separators=(",", ":")).encode()
            import base64
            completed = type("Completed", (), {"stdout": (
                "root@a40:~# python3 -c 'print(\\\"noise\\\")'\n"
                "AUTOPARETO_TRANSFER_RESULT:" + base64.b64encode(result).decode("ascii") + "\n"
                "root@a40:~# exit\n"
            ).encode()})()
            with patch.object(transport, "_run_pty_lines", return_value=completed) as run:
                remote = transport.transfer(payload=payload, remote_run_id="attempt-001")
            self.assertEqual(remote, "/runs/attempt-001")
            self.assertEqual(run.call_count, 1)

    def test_transfer_rejects_missing_or_ambiguous_framed_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            payload = root / "payload"
            payload.mkdir()
            (payload / "train.py").write_text("candidate")
            (payload / "registered_metadata.json").write_text('{"schema":"x"}\n')
            from study.contracts_2026.bridge import A40Transport

            transport = A40Transport(A40RemoteConfig(host="example.invalid", payload_root="/runs"))
            malformed = type("Completed", (), {"stdout": b"AUTOPARETO_TRANSFER_RESULT:not-base64!!!\n"})()
            with patch.object(transport, "_run_pty_lines", return_value=malformed):
                with self.assertRaisesRegex(RuntimeError, "transfer result sentinel"):
                    transport.transfer(payload=payload, remote_run_id="attempt-002")

    def test_pty_marker_loss_is_classified_as_retryable_transport_framing(self):
        from study.contracts_2026.bridge import A40Transport, A40TransportFrameError

        transport = A40Transport(A40RemoteConfig(host="example.invalid"))
        missing_markers = type("Completed", (), {"stdout": b"login prompt\npartial command output\n", "stderr": b"", "returncode": 0})()
        with patch("study.contracts_2026.bridge.shutil.which", return_value="/usr/bin/script"), \
                patch("study.contracts_2026.bridge.subprocess.run", return_value=missing_markers):
            with self.assertRaises(A40TransportFrameError):
                transport._run_pty_lines(["printf ready"], operation="regression_marker_loss")

    def test_pty_ssh_failure_preserves_transport_error_and_stderr(self):
        from study.contracts_2026.bridge import A40Transport, A40TransportCommandError

        transport = A40Transport(A40RemoteConfig(host="example.invalid"))
        failed = type("Completed", (), {
            "stdout": b"login prompt\npartial output\n",
            "stderr": b"Permission denied (publickey).\n",
            "returncode": 255,
        })()
        with patch("study.contracts_2026.bridge.shutil.which", return_value="/usr/bin/script"), \
                patch("study.contracts_2026.bridge.subprocess.run", return_value=failed):
            with self.assertRaisesRegex(A40TransportCommandError, "Permission denied"):
                transport._run_pty_lines(["printf ready"], operation="regression_ssh_failure")

    def test_cli_a40_cache_reaches_remote_execution_environment(self):
        cache = "/workspace/pair_a_registered_cache"
        with patch.object(campaign_controller, "run_campaign", return_value={"status": "ok"}) as run_campaign, patch.object(
            campaign_controller,
            "resolve_scientist_config",
            return_value=ScientistConfig(
                model="gpt-5.6-sol",
                reasoning="high",
                scientist_bin=sys.executable,
                codex_version="test",
                execution_backend="host",
                sandbox_mode="workspace-write",
            ),
        ):
            result = campaign_controller.main([
                "run",
                "--execute",
                "--a40-cache",
                cache,
                "--a40-host",
                "example.invalid",
                "--manifest",
                "manifest.json",
                "--manifest-sha256",
                "a" * 64,
                "--model",
                "gpt-5.6-sol",
                "--reasoning",
                "high",
            ])
        self.assertEqual(result, 0)
        transport = run_campaign.call_args.kwargs["transport"]
        self.assertEqual(transport.config.registered_cache, cache)

        with patch.object(transport, "_run") as run_remote, patch.object(
            transport, "_fetch_durable_record", return_value={}
        ):
            transport.launch(
                remote_dir="/tmp/payload",
                invocation_id="cache-test-12345678",
                invocation_metadata={},
            )
        command = run_remote.call_args.args[0]
        self.assertIn("AUTOPARETO_REGISTERED_CACHE=/workspace/pair_a_registered_cache", command)
        self.assertIn("HF_HOME=/workspace/pair_a_registered_cache/huggingface", command)
        self.assertIn("HF_HUB_CACHE=/workspace/pair_a_registered_cache/huggingface/hub", command)
        self.assertIn("HF_HUB_OFFLINE=1", command)
        self.assertNotIn("/opt/autopareto_candidate_inputs/cache/autoresearch", command)

    def test_trusted_hash_verification_accepts_match_and_rejects_mismatch(self):
        config = A40RemoteConfig(host="example.invalid", repository="/opt/autopareto_trusted")
        transport = __import__("study.contracts_2026.bridge", fromlist=["A40Transport"]).A40Transport(config)
        expected = {"study/contracts_2026/bridge.py": "a" * 64}
        import base64
        import hashlib
        nonce = "0123456789abcdef0123456789abcdef"

        def frame(value):
            payload = json.dumps(value, separators=(",", ":")).encode()
            encoded = base64.b64encode(payload).decode()
            return "\n".join([
                "root@a40:~# noise",
                f"AUTOPARETO_TRUSTED_FILES_BEGIN:{nonce}",
                f"AUTOPARETO_TRUSTED_FILES_META:{len(payload)}:{hashlib.sha256(payload).hexdigest()}",
                *[f"AUTOPARETO_TRUSTED_FILES_DATA:{encoded[index:index + 32]}" for index in range(0, len(encoded), 32)],
                f"AUTOPARETO_TRUSTED_FILES_END:{nonce}",
                "root@a40:~# exit",
            ])

        matching = type("Completed", (), {"stdout": (frame(expected) + "\n").encode()})()
        with patch("study.contracts_2026.bridge.uuid.uuid4", return_value=type("UUID", (), {"hex": nonce})()), \
                patch.object(transport, "_run", return_value=matching):
            self.assertEqual(transport.verify_trusted_files(expected), expected)
        mismatching_payload = {"study/contracts_2026/bridge.py": "b" * 64}
        mismatching = type("Completed", (), {"stdout": (frame(mismatching_payload) + "\n").encode()})()
        with patch("study.contracts_2026.bridge.uuid.uuid4", return_value=type("UUID", (), {"hex": nonce})()), \
                patch.object(transport, "_run", return_value=mismatching):
            with self.assertRaisesRegex(RuntimeError, "trusted deployment hash mismatch"):
                transport.verify_trusted_files(expected)

    def test_trusted_hash_verification_rejects_duplicate_framed_result(self):
        import base64
        import hashlib
        expected = {"study/contracts_2026/bridge.py": "a" * 64}
        transport = __import__("study.contracts_2026.bridge", fromlist=["A40Transport"]).A40Transport(
            A40RemoteConfig(host="example.invalid", repository="/opt/autopareto_trusted")
        )
        nonce = "0123456789abcdef0123456789abcdef"
        payload = json.dumps(expected, separators=(",", ":")).encode()
        encoded = base64.b64encode(payload).decode()
        framed = "\n".join([
            f"AUTOPARETO_TRUSTED_FILES_BEGIN:{nonce}",
            f"AUTOPARETO_TRUSTED_FILES_META:{len(payload)}:{hashlib.sha256(payload).hexdigest()}",
            *[f"AUTOPARETO_TRUSTED_FILES_DATA:{encoded[index:index + 32]}" for index in range(0, len(encoded), 32)],
            f"AUTOPARETO_TRUSTED_FILES_END:{nonce}",
        ])
        duplicate = type("Completed", (), {"stdout": (framed + "\n" + framed + "\n").encode()})()
        with patch("study.contracts_2026.bridge.uuid.uuid4", return_value=type("UUID", (), {"hex": nonce})()), \
                patch.object(transport, "_run", return_value=duplicate):
            with self.assertRaisesRegex(RuntimeError, "trusted deployment hash frame"):
                transport.verify_trusted_files(expected)

    def test_remote_payload_verification_accepts_noise_and_rejects_hash_mismatch(self):
        import base64
        import hashlib
        from study.contracts_2026.bridge import A40Transport, sha256_file

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            candidate = root / "train.py"
            metadata = root / "registered_metadata.json"
            candidate.write_text("candidate\n")
            metadata.write_text('{"schema":"x"}\n')
            expected = {
                "train.py": {"bytes": candidate.stat().st_size, "sha256": sha256_file(candidate)},
                "registered_metadata.json": {"bytes": metadata.stat().st_size, "sha256": sha256_file(metadata)},
            }
            nonce = "0123456789abcdef0123456789abcdef"

            def framed(value):
                payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
                encoded = base64.b64encode(payload).decode()
                return "\n".join([
                    "root@a40:~# ignored PTY noise",
                    f"AUTOPARETO_PAYLOAD_VERIFY_BEGIN:{nonce}",
                    f"AUTOPARETO_PAYLOAD_VERIFY_META:{len(payload)}:{hashlib.sha256(payload).hexdigest()}",
                    *[f"AUTOPARETO_PAYLOAD_VERIFY_DATA:{encoded[index:index + 32]}" for index in range(0, len(encoded), 32)],
                    f"AUTOPARETO_PAYLOAD_VERIFY_END:{nonce}",
                    "root@a40:~# exit",
                ]) + "\n"

            transport = A40Transport(A40RemoteConfig(host="example.invalid"))
            completed = type("Completed", (), {"stdout": framed(expected).encode()})()
            with patch("study.contracts_2026.bridge.uuid.uuid4", return_value=type("UUID", (), {"hex": nonce})()), \
                    patch.object(transport, "_run", return_value=completed):
                self.assertEqual(
                    transport.verify_payload(remote_dir="/remote/payload", candidate=candidate, metadata=metadata),
                    expected,
                )
            bad = json.loads(json.dumps(expected))
            bad["train.py"]["sha256"] = "0" * 64
            completed = type("Completed", (), {"stdout": framed(bad).encode()})()
            with patch("study.contracts_2026.bridge.uuid.uuid4", return_value=type("UUID", (), {"hex": nonce})()), \
                    patch.object(transport, "_run", return_value=completed):
                with self.assertRaisesRegex(RuntimeError, "remote payload verification mismatch"):
                    transport.verify_payload(remote_dir="/remote/payload", candidate=candidate, metadata=metadata)

    def test_run_cli_propagates_explicit_paired_baseline_checkpoints(self):
        with patch.object(campaign_controller, "run_campaign", return_value={"status": "ok"}) as run_campaign, patch.object(
            campaign_controller,
            "resolve_scientist_config",
            return_value=ScientistConfig(
                model="gpt-5.6-sol",
                reasoning="high",
                scientist_bin=sys.executable,
                codex_version="test",
                execution_backend="host",
                sandbox_mode="workspace-write",
            ),
        ):
            campaign_controller.main([
                "run", "--execute",
                "--manifest", "manifest.json", "--manifest-sha256", "a" * 64,
                "--model", "gpt-5.6-sol", "--reasoning", "high",
                "--a40-paired-baseline-t-checkpoint", "/state/t.pt",
                "--a40-paired-baseline-n-checkpoint", "/state/n.pt",
            ])
        config = run_campaign.call_args.kwargs["transport"].config
        self.assertEqual(config.paired_baseline_t_checkpoint, "/state/t.pt")
        self.assertEqual(config.paired_baseline_n_checkpoint, "/state/n.pt")

    def test_durable_result_readback_reassembles_noisy_framed_chunks(self):
        import base64
        import hashlib
        import zlib
        from study.contracts_2026.bridge import A40Transport

        payload = json.dumps({"schema": "autopareto_a40_result_v1", "status": "complete"}, sort_keys=True).encode()
        digest = hashlib.sha256(payload).hexdigest()
        nonce = "0123456789abcdef0123456789abcdef"
        compressed = zlib.compress(payload, 6)
        encoded = base64.b64encode(compressed).decode()
        framed = "\n".join([
            "root@a40:~# noisy prompt",
            f"AUTOPARETO_DURABLE_RESULT_BEGIN:{nonce}",
            f"AUTOPARETO_DURABLE_RESULT_META:{len(compressed)}:{hashlib.sha256(compressed).hexdigest()}",
            *[f"AUTOPARETO_DURABLE_RESULT_DATA:{encoded[index:index + 32]}" for index in range(0, len(encoded), 32)],
            f"AUTOPARETO_DURABLE_RESULT_END:{nonce}",
            "root@a40:~# exit",
        ])
        transport = A40Transport(A40RemoteConfig(host="example.invalid"))
        record = {"status": "COMPLETE", "result_path": "/state/result.json", "result_bytes": len(payload), "result_sha256": digest}
        completed = type("Completed", (), {"stdout": ("\x1b]0;root@a40\x07" + framed + "\n").encode()})()
        with patch("study.contracts_2026.bridge.uuid.uuid4", return_value=type("UUID", (), {"hex": nonce})()), \
                patch.object(transport, "_run", return_value=completed):
            self.assertEqual(transport.fetch_durable_json(record), json.loads(payload))

    def test_durable_result_readback_retries_one_lost_outer_pty_frame(self):
        import base64
        import hashlib
        import zlib
        from study.contracts_2026.bridge import A40Transport, A40TransportFrameError

        payload = json.dumps({"schema": "autopareto_a40_result_v1", "status": "complete"}, sort_keys=True).encode()
        digest = hashlib.sha256(payload).hexdigest()
        nonce = "0123456789abcdef0123456789abcdef"
        compressed = zlib.compress(payload, 6)
        encoded = base64.b64encode(compressed).decode()
        framed = "\n".join([
            f"AUTOPARETO_DURABLE_RESULT_BEGIN:{nonce}",
            f"AUTOPARETO_DURABLE_RESULT_META:{len(compressed)}:{hashlib.sha256(compressed).hexdigest()}",
            *[f"AUTOPARETO_DURABLE_RESULT_DATA:{encoded[index:index + 32]}" for index in range(0, len(encoded), 32)],
            f"AUTOPARETO_DURABLE_RESULT_END:{nonce}",
        ])
        transport = A40Transport(A40RemoteConfig(host="example.invalid"))
        record = {"status": "COMPLETE", "result_path": "/state/result.json", "result_bytes": len(payload), "result_sha256": digest}
        malformed = type("Completed", (), {"stdout": b"partial PTY output\n"})()
        complete = type("Completed", (), {"stdout": framed.encode()})()
        with patch("study.contracts_2026.bridge.uuid.uuid4", return_value=type("UUID", (), {"hex": nonce})()), \
                patch.object(transport, "_run", side_effect=[A40TransportFrameError("outer frame lost"), complete]) as run:
            self.assertEqual(transport.fetch_durable_json(record), json.loads(payload))
        self.assertEqual(run.call_count, 2)

    def test_durable_result_readback_rejects_unframed_payload_without_retrying_mutation(self):
        from study.contracts_2026.bridge import A40Transport

        transport = A40Transport(A40RemoteConfig(host="example.invalid"))
        record = {"status": "COMPLETE", "result_path": "/state/result.json", "result_bytes": 2, "result_sha256": "0" * 64}
        malformed = type("Completed", (), {"stdout": b"AUTOPARETO_DURABLE_RESULT_BEGIN:not-the-requested-nonce\n"})()
        with patch.object(transport, "_run", return_value=malformed) as run:
            with self.assertRaisesRegex(RuntimeError, "durable result frame"):
                transport.fetch_durable_json(record)
        self.assertEqual(run.call_count, 1)

    def test_durable_result_readback_rejects_legacy_wrapped_or_malformed_output(self):
        import base64
        import hashlib
        from study.contracts_2026.bridge import A40Transport

        payload = b'{"schema":"autopareto_a40_result_v1"}'
        digest = hashlib.sha256(payload).hexdigest()
        encoded = base64.b64encode(payload).decode()
        malformed = f"AUTOPARETO_DURABLE_RESULT:{len(payload)}:{digest}:{encoded[:8]}\n{encoded[8:]}\n"
        transport = A40Transport(A40RemoteConfig(host="example.invalid"))
        record = {"status": "COMPLETE", "result_path": "/state/result.json", "result_bytes": len(payload), "result_sha256": digest}
        completed = type("Completed", (), {"stdout": malformed.encode()})()
        with patch("study.contracts_2026.bridge.uuid.uuid4", return_value=type("UUID", (), {"hex": "0123456789abcdef0123456789abcdef"})()), \
                patch.object(transport, "_run", return_value=completed):
            with self.assertRaisesRegex(RuntimeError, "durable result frame"):
                transport.fetch_durable_json(record)

    def test_durable_record_readback_reassembles_bounded_noisy_frame(self):
        import base64
        import hashlib
        from study.contracts_2026.bridge import A40Transport

        nonce = "0123456789abcdef0123456789abcdef"
        expected = {
            "record": {"invocation_id": "pair-a-002-12345678", "status": "COMPLETE"},
            "process_alive": False,
        }
        payload = json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()
        encoded = base64.b64encode(payload).decode()
        framed = "\n".join([
            "root@a40:~# shell noise",
            f"AUTOPARETO_DURABLE_RECORD_BEGIN:{nonce}",
            f"AUTOPARETO_DURABLE_RECORD_META:{len(payload)}:{hashlib.sha256(payload).hexdigest()}",
            *[f"AUTOPARETO_DURABLE_RECORD_DATA:{encoded[index:index + 32]}" for index in range(0, len(encoded), 32)],
            f"AUTOPARETO_DURABLE_RECORD_END:{nonce}",
            "root@a40:~# exit",
        ])
        completed = type("Completed", (), {"stdout": ("\x1b]0;root@a40\x07" + framed + "\n").encode()})()
        transport = A40Transport(A40RemoteConfig(host="example.invalid"))
        with patch("study.contracts_2026.bridge.uuid.uuid4", return_value=type("UUID", (), {"hex": nonce})()), \
                patch.object(transport, "_run", return_value=completed):
            self.assertEqual(transport._fetch_durable_record("pair-a-002-12345678"), expected)

    def test_durable_record_readback_rejects_the_legacy_wrapped_line(self):
        import base64
        from study.contracts_2026.bridge import A40Transport

        encoded = base64.b64encode(b'{"record":{},"process_alive":false}').decode()
        # This reproduces the prior failure mode: a large one-line record is
        # split by PTY output handling and must never be parsed as a receipt.
        wrapped_legacy = (
            "AUTOPARETO_DURABLE_RECORD:" + encoded[:12] + "\n" + encoded[12:] + "\n"
        )
        transport = A40Transport(A40RemoteConfig(host="example.invalid"))
        completed = type("Completed", (), {"stdout": wrapped_legacy.encode()})()
        with patch.object(transport, "_run", return_value=completed):
            with self.assertRaisesRegex(RuntimeError, "durable record frame"):
                transport._fetch_durable_record("pair-a-002-12345678")

    def test_gpu_identity_requires_exact_framed_match(self):
        expected_uuid = "GPU-0188884d-bcea-4416-6590-96303960308c"
        transport = __import__("study.contracts_2026.bridge", fromlist=["A40Transport"]).A40Transport(
            A40RemoteConfig(host="example.invalid")
        )
        matching = type("Completed", (), {"stdout": ("prompt\nAUTOPARETO_GPU_IDENTITY:" + expected_uuid + "\n").encode()})()
        with patch.object(transport, "_run", return_value=matching):
            self.assertEqual(transport.verify_gpu_identity(expected_uuid), expected_uuid)

    def test_model_outputs_share_one_canonical_proposal_schema(self):
        sol = canonicalize_scientist_proposal(
            {"hypothesis": "h", "intended_change": "i", "change_category": "c"},
            phase="edit_child",
            expected_parent_experiment_id="baseline",
        )
        terra = canonicalize_scientist_proposal(
            {"hypothesis": "h", "intended_change": "i", "change_category": "c"},
            phase="edit_child",
            expected_parent_experiment_id="baseline",
        )
        self.assertEqual(sol, terra)
        self.assertEqual(sol["parent_experiment_id"], "baseline")

    def test_scientist_command_cannot_wait_for_approval(self):
        selected_plan = {
            "hypothesis": "h",
            "intended_change": "i",
            "change_category": "architecture",
        }
        command = campaign_controller._scientist_command(
            pathlib.Path("/tmp/context"),
            "campaign",
            1,
            ScientistConfig(
                model="gpt-5.6-sol",
                reasoning="high",
                scientist_bin="/Applications/ChatGPT.app/Contents/Resources/codex",
            ),
            "edit_child",
            "registered-baseline",
            selected_plan,
        )
        approval_index = command.index("-c")
        self.assertEqual(command[approval_index + 1], 'approval_policy="never"')
        self.assertIn("--ephemeral", command)
        self.assertIn("--ignore-user-config", command)
        self.assertIn(json.dumps(selected_plan, sort_keys=True), command[-1])

    def test_child_edit_cannot_substitute_controller_selected_plan(self):
        selected_plan = {
            "hypothesis": "h",
            "intended_change": "i",
            "change_category": "architecture",
        }
        campaign_controller._validate_child_plan(selected_plan, selected_plan)
        with self.assertRaisesRegex(RuntimeError, "controller-selected scientific plan"):
            campaign_controller._validate_child_plan(
                {**selected_plan, "intended_change": "different edit"}, selected_plan
            )

    def test_canonical_proposal_rejects_missing_decision_and_parent_override(self):
        with self.assertRaisesRegex(ValueError, "missing fields"):
            canonicalize_scientist_proposal(
                {"hypothesis": "h", "intended_change": "i"},
                phase="edit_child",
                expected_parent_experiment_id="baseline",
            )
        with self.assertRaisesRegex(ValueError, "cannot override"):
            canonicalize_scientist_proposal(
                {"hypothesis": "h", "intended_change": "i", "change_category": "c", "parent_experiment_id": "other"},
                phase="edit_child",
                expected_parent_experiment_id="baseline",
            )
        with self.assertRaises(ValueError):
            canonicalize_scientist_proposal(
                {"hypothesis": "h", "intended_change": "i", "change_category": "c", "source_sha256": "wrong"},
                phase="edit_child",
                expected_parent_experiment_id="baseline",
            )

    def test_proposal_envelope_parser_rejects_malformed_or_conflicting_output(self):
        from study.contracts_2026.bridge import _read_json_line

        wrapped = json.dumps({"type": "item.completed", "item": {"text": json.dumps({
            "hypothesis": "h", "intended_change": "i", "change_category": "c"
        })}})
        self.assertEqual(_read_json_line(wrapped)["hypothesis"], "h")
        with self.assertRaises(ValueError):
            _read_json_line("not-json")
        conflicting = "\n".join([
            json.dumps({"hypothesis": "h1", "intended_change": "i", "change_category": "c"}),
            json.dumps({"hypothesis": "h2", "intended_change": "i", "change_category": "c"}),
        ])
        with self.assertRaises(ValueError):
            _read_json_line(conflicting)

    def test_proposal_envelope_parser_rejects_the_observed_empty_hypothesis_key(self):
        from study.contracts_2026.bridge import _read_json_line

        observed = json.dumps({
            "": "A scientifically meaningful hypothesis was placed under the wrong key.",
            "intended_change": "change",
            "change_category": "architecture",
            "parent_experiment_id": "pair-a-fixed-tokens-seed3-attempt-03",
        })
        with self.assertRaisesRegex(ValueError, "did not emit a proposal"):
            _read_json_line(observed)

    def test_proposal_envelope_parser_accepts_pretty_printed_agent_message(self):
        from study.contracts_2026.bridge import _read_json_line

        pretty = json.dumps({
            "hypothesis": "h",
            "intended_change": "i",
            "change_category": "c",
        }, indent=2)
        envelope = json.dumps({
            "type": "item.completed",
            "item": {"type": "agent_message", "text": pretty},
        })
        self.assertEqual(_read_json_line(envelope)["change_category"], "c")

    def test_terminal_codex_event_selects_only_the_final_schema_proposal(self):
        from study.contracts_2026.bridge import _read_json_line

        def message(proposal):
            return json.dumps({
                "type": "item.completed",
                "item": {"type": "agent_message", "text": json.dumps(proposal)},
            })

        stream = "\n".join([
            message({
                "hypothesis": "intermediate", "intended_change": "inspect", "change_category": "progress",
                "parent_experiment_id": "pending",
            }),
            message({
                "hypothesis": "final", "intended_change": "edit", "change_category": "architecture",
                "parent_experiment_id": "registered-parent",
            }),
            json.dumps({"type": "turn.completed", "usage": {}}),
        ])
        proposal = _read_json_line(stream)
        self.assertEqual(proposal["hypothesis"], "final")
        self.assertEqual(proposal["parent_experiment_id"], "registered-parent")

    def test_terminal_codex_event_rejects_duplicate_or_nonterminal_completion(self):
        from study.contracts_2026.bridge import _read_json_line

        final = json.dumps({
            "type": "item.completed",
            "item": {"type": "agent_message", "text": json.dumps({
                "hypothesis": "h", "intended_change": "i", "change_category": "c",
            })},
        })
        with self.assertRaisesRegex(ValueError, "terminal Codex event framing"):
            _read_json_line("\n".join([final, json.dumps({"type": "turn.completed"}), json.dumps({"type": "turn.completed"})]))
