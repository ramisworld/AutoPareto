import pathlib
import platform
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from study.contracts_2026.context import build_campaign_view
from study.contracts_2026.isolation import (
    SCIENTIST_PARENT_SELECTION_SCHEMA,
    SandboxUnavailableError,
    _sandbox_profile,
    run_agent_command,
)


class ContextTests(unittest.TestCase):
    def test_native_codex_backend_scopes_execution_to_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            context = pathlib.Path(tmp) / "context"
            context.mkdir()
            command = [
                "/Applications/ChatGPT.app/Contents/Resources/codex", "exec",
                "-C", str(context), "-s", "workspace-write", "--ephemeral",
                "--output-schema", str(SCIENTIST_PARENT_SELECTION_SCHEMA),
            ]
            completed = subprocess.CompletedProcess(command, 0, "", "")
            with patch("study.contracts_2026.isolation.subprocess.run", return_value=completed) as run:
                observed = run_agent_command(command, context, capture_output=True, text=True)
            self.assertIs(observed, completed)
            run.assert_called_once_with(command, cwd=context.resolve(), check=False, capture_output=True, text=True)

    def test_native_codex_backend_rejects_widening_or_bypass(self):
        with tempfile.TemporaryDirectory() as tmp:
            context = pathlib.Path(tmp) / "context"
            context.mkdir()
            base = ["codex", "exec", "-C", str(context), "-s", "workspace-write"]
            with self.assertRaises(SandboxUnavailableError):
                run_agent_command(base + ["--add-dir", "/tmp/other"], context)
            with self.assertRaises(SandboxUnavailableError):
                run_agent_command(base + ["--dangerously-bypass-approvals-and-sandbox"], context)
            with self.assertRaises(SandboxUnavailableError):
                run_agent_command(["codex", "exec", "-C", str(context), "-s", "read-only"], context)

    def test_scientist_sandbox_allows_only_codex_https_network(self):
        if platform.system() != "Darwin":
            self.skipTest("the legacy macOS sandbox profile exists only on macOS")
        profile = _sandbox_profile(__import__("pathlib").Path("/tmp/scientist-context"))
        self.assertIn('(allow network-outbound', profile)
        self.assertIn('(literal "/private/var/run/mDNSResponder")', profile)
        self.assertIn('(remote tcp "*:443")', profile)
        self.assertNotIn('(deny network-outbound)', profile)
        self.assertIn('(deny network-inbound)', profile)

    def test_downstream_is_sealed_and_scope_is_local(self):
        view = build_campaign_view(
            campaign_id="t",
            contract_id="T",
            baseline={"record_scope": "same_seed_baseline", "validation_bpb": 2.0},
            frontier=[{"campaign_id": "t", "validation_bpb": 1.9}],
            attempt_number=1,
        )
        self.assertNotIn("held_out_bpb", view["baseline"])
        with self.assertRaises(ValueError):
            build_campaign_view(
                campaign_id="t",
                contract_id="T",
                baseline={"record_scope": "same_seed_baseline"},
                frontier=[{"campaign_id": "n"}],
                attempt_number=1,
            )

    def test_same_arm_evidence_contains_complete_nonsealed_measurements(self):
        prior = {
            "campaign_id": "t", "attempt": 1, "experiment_id": "t-01",
            "parent_experiment_id": "baseline",
            "hypothesis": "h", "intended_change": "i",
            "change_category": "architecture",
            "candidate_patch": "@@ -1 +1 @@\n-old\n+new\n",
            "candidate_source_after_sha256": "a" * 64,
            "protected_validation_bpb": 1.9,
            "parent_eligible": True, "pareto_eligible": False,
            "status": "completed",
            "num_params": 123,
            "total_tokens": 456,
            "optimizer_updates": 7,
            "training_throughput_tokens_per_second": 89.0,
            "inference_result": {
                "inference_status": "unstable", "correctness": {"passed": True},
                "absolute_candidate_tok_s": 100.0,
                "absolute_baseline_tok_s": 99.0,
                "baseline_before_window_tok_s": [98.0] * 8,
                "baseline_after_window_tok_s": [100.0] * 8,
                "interpolated_baseline_window_tok_s": [99.0] * 8,
                "normalized_pairwise_ratios": [1.01] * 8,
                "normalized_decode_throughput": 1.0,
                "normalized_decode_throughput_ci_95pct": [0.99, 1.02],
                "normalized_coefficient_of_variation": 0.01,
                "normalized_range_drift": 0.02,
                "peak_allocated_inference_memory_bytes": 10,
                "measurement_windows": ["all eight windows"],
                "held_out_bpb": 0.1,
            },
        }
        view = build_campaign_view(
            campaign_id="t", contract_id="T",
            baseline={"record_scope": "same_seed_baseline", "campaign_id": "t"},
            frontier=[], attempt_number=2, evidence=[prior], parent_pool=[prior],
        )
        self.assertEqual(view["same_arm_evidence"][0]["hypothesis"], "h")
        self.assertEqual(view["same_arm_evidence"][0]["validation_bpb"], 1.9)
        self.assertEqual(view["same_arm_evidence"][0]["protected_validation_bpb"], 1.9)
        self.assertIn("candidate_patch", view["same_arm_evidence"][0])
        self.assertEqual(view["same_arm_evidence"][0]["inference"]["measurement_windows"], ["all eight windows"])
        self.assertEqual(view["same_arm_evidence"][0]["inference"]["normalized_pairwise_ratios"], [1.01] * 8)
        self.assertNotIn("held_out_bpb", view["same_arm_evidence"][0]["inference"])
        self.assertEqual(view["same_arm_evidence"][0]["num_params"], 123)
        self.assertEqual(view["same_arm_evidence"][0]["optimizer_updates"], 7)
        self.assertEqual(view["parent_pool"][0]["experiment_id"], "t-01")
        self.assertEqual(view["lineage"][1]["parent_experiment_id"], "baseline")
        with self.assertRaises(ValueError):
            build_campaign_view(
                campaign_id="t",
                contract_id="T",
                baseline={"record_scope": "same_seed_baseline", "LAMBADA": 1.0},
                frontier=[],
                attempt_number=1,
            )

    def test_controller_archive_preserves_nonsealed_metrics_for_scientist_context(self):
        baseline = {
            "record_scope": "same_seed_baseline",
            "campaign_id": "t",
            "protected_validation_bpb": 2.0,
            "inference_result": {
                "inference_status": "stable",
                "normalized_decode_throughput": 1.0,
                "peak_allocated_inference_memory_bytes": 10,
                "measurement_windows": list(range(1000)),
            },
        }
        frontier = [{
            "campaign_id": "t",
            "attempt": 1,
            "experiment_id": "t-01",
            "protected_validation_bpb": 1.9,
            "inference_result": {
                "inference_status": "stable",
                "normalized_decode_throughput": 1.1,
                "peak_allocated_inference_memory_bytes": 9,
                "measurement_windows": list(range(1000)),
            },
        }]
        view = build_campaign_view(
            campaign_id="t", contract_id="T", baseline=baseline,
            frontier=frontier, attempt_number=2,
        )
        self.assertEqual(view["baseline"]["inference_result"]["normalized_decode_throughput"], 1.0)
        self.assertEqual(view["baseline"]["inference_result"]["measurement_windows"], list(range(1000)))
        self.assertEqual(view["frontier"][0]["inference_result"]["measurement_windows"], list(range(1000)))
        self.assertEqual(view["frontier"][0]["protected_validation_bpb"], 1.9)

    def test_lineage_is_same_arm_and_includes_failed_attempts(self):
        failed = {
            "campaign_id": "t",
            "attempt": 2,
            "experiment_id": "t-02",
            "parent_experiment_id": "t-01",
            "candidate_source_after_sha256": None,
            "hypothesis": "failed hypothesis",
            "intended_change": "failed change",
            "change_category": "architecture",
            "candidate_patch": "@@ failure diff",
            "status": "scientific_failure",
            "quality_status": "invalid",
            "parent_eligible": False,
            "pareto_eligible": False,
            "error": "OOM",
        }
        view = build_campaign_view(
            campaign_id="t", contract_id="T",
            baseline={"record_scope": "same_seed_baseline", "campaign_id": "t"},
            frontier=[], attempt_number=3, evidence=[failed], parent_pool=[],
        )
        self.assertEqual(view["same_arm_evidence"][0]["status"], "scientific_failure")
        self.assertEqual(view["same_arm_evidence"][0]["error"], "OOM")
        self.assertEqual(view["lineage"][1]["experiment_id"], "t-02")
        self.assertFalse(view["lineage"][1]["parent_eligible"])
        with self.assertRaises(ValueError):
            build_campaign_view(
                campaign_id="t", contract_id="T",
                baseline={"record_scope": "same_seed_baseline", "campaign_id": "t"},
                frontier=[], attempt_number=1,
                evidence=[{**failed, "campaign_id": "n"}], parent_pool=[],
            )
