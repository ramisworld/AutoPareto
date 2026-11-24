import pathlib
import json
import tempfile
import unittest

from study.contracts_2026.campaign import (
    append_frontier,
    append_local_history,
    campaign_history_path,
    candidate_disposition,
    decide_candidate,
)
from study.contracts_2026.campaign_controller import _ensure_unique_candidate_source
from study.contracts_2026.contracts import ContractId, ResourceContract
from study.contracts_2026.runner import assert_study_inference_command, inference_command
from study.contracts_2026.benchmark_policy import PROTOCOL_VERSION


class CampaignTests(unittest.TestCase):
    def test_histories_are_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            append_local_history(root, "pair-a-fixed-time-seed3", {"campaign_id": "pair-a-fixed-time-seed3", "attempt": 1})
            self.assertTrue(campaign_history_path(root, "pair-a-fixed-time-seed3").exists())
            self.assertFalse(campaign_history_path(root, "pair-a-fixed-tokens-seed3").exists())
            with self.assertRaises(ValueError):
                append_local_history(root, "pair-a-fixed-tokens-seed3", {"campaign_id": "pair-a-fixed-time-seed3", "attempt": 1})
            with self.assertRaises(ValueError):
                append_frontier(root, "pair-a-fixed-tokens-seed3", {"campaign_id": "pair-a-fixed-time-seed3", "attempt": 1})

    def test_byte_identical_candidate_is_rejected_before_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = pathlib.Path(tmp) / "source_registry.jsonl"
            registry.write_text(json.dumps({"source_sha256": "a" * 64}) + "\n")
            with self.assertRaisesRegex(RuntimeError, "byte-identical"):
                _ensure_unique_candidate_source(registry, "a" * 64)
            _ensure_unique_candidate_source(registry, "b" * 64)

    def test_legacy_benchmark_is_unreachable(self):
        command = inference_command(
            context=pathlib.Path("runs/x/context"),
            checkpoint="runs/x/checkpoint.pt",
            output=pathlib.Path("runs/x/inference.json"),
        )
        self.assertIn(PROTOCOL_VERSION, " ".join(command))
        self.assertNotIn("autopareto_inference_stabilized_v3", " ".join(command))
        with self.assertRaises(ValueError):
            assert_study_inference_command(["python", "inference_benchmark.py", "--legacy-diagnostic"])

    def test_pareto_uses_only_protected_inference_metrics(self):
        baseline = {
            "record_scope": "same_seed_baseline",
            "inference_result": {"inference_status": "stable", "cached_decode_tokens_per_second_median": 100.0, "peak_allocated_inference_memory_bytes": 10},
            "protected_validation_bpb": 2.0,
        }
        candidate = {
            "inference_result": {"inference_status": "stable", "normalized_decode_throughput": 1.01, "normalized_decode_throughput_ci_95pct": [1.0, 1.02], "cached_decode_tokens_per_second_median": 101.0, "peak_allocated_inference_memory_bytes": 10},
            "protected_validation_bpb": 1.9,
            "validation_bpb": 999.0,
            "metrics": {"validation_bpb": 999.0, "cached_decode_tokens_per_second": 0.0, "peak_allocated_inference_memory_bytes": 999},
        }
        result = decide_candidate(candidate, [baseline])
        self.assertEqual(result["decision"], "retained")

    def test_pareto_rejects_unprotected_validation_fallback(self):
        candidate = {
            "validation_bpb": 1.0,
            "inference_result": {
                "inference_status": "stable",
                "normalized_decode_throughput": 1.0,
                "cached_decode_tokens_per_second_median": 100.0,
                "peak_allocated_inference_memory_bytes": 10,
            },
        }
        with self.assertRaises(ValueError):
            decide_candidate(candidate, [])

    def test_quality_valid_speed_uncertain_is_parent_eligible_only_at_baseline_quality(self):
        candidate = {
            "protected_validation_bpb": 1.9,
            "inference_result": {
                "inference_status": "unstable",
                "correctness": {"passed": True},
                "cached_decode_tokens_per_second_median": 100.0,
                "peak_allocated_inference_memory_bytes": 10,
            },
        }
        disposition = candidate_disposition(candidate, quality_reference_bpb=2.0)
        self.assertEqual(disposition["quality_status"], "quality_valid_speed_uncertain")
        self.assertTrue(disposition["parent_eligible"])
        self.assertFalse(disposition["pareto_eligible"])
        decision = decide_candidate(candidate, [{"record_scope": "same_seed_baseline", "protected_validation_bpb": 2.0}])
        self.assertEqual(decision["decision"], "quality_valid_speed_uncertain")
        self.assertTrue(decision["parent_eligible"])
        self.assertFalse(decision["pareto_eligible"])

    def test_quality_valid_speed_uncertain_worse_than_baseline_is_not_parent(self):
        candidate = {
            "protected_validation_bpb": 2.1,
            "inference_result": {
                "inference_status": "unstable",
                "correctness": {"passed": True},
            },
        }
        decision = decide_candidate(
            candidate,
            [{"record_scope": "same_seed_baseline", "protected_validation_bpb": 2.0}],
        )
        self.assertEqual(decision["quality_status"], "quality_valid_speed_uncertain")
        self.assertFalse(decision["parent_eligible"])
        self.assertFalse(decision["pareto_eligible"])

    def test_dominated_stable_candidate_is_not_a_parent(self):
        baseline = {
            "record_scope": "same_seed_baseline",
            "protected_validation_bpb": 2.0,
            "inference_result": {
                "inference_status": "stable",
                "normalized_decode_throughput": 1.0,
                "normalized_decode_throughput_ci_95pct": [1.0, 1.0],
                "peak_allocated_inference_memory_bytes": 10,
            },
        }
        candidate = {
            "protected_validation_bpb": 2.1,
            "inference_result": {
                "inference_status": "stable",
                "normalized_decode_throughput": 1.0,
                "normalized_decode_throughput_ci_95pct": [1.0, 1.0],
                "peak_allocated_inference_memory_bytes": 10,
            },
        }
        decision = decide_candidate(candidate, [baseline])
        self.assertEqual(decision["decision"], "dominated")
        self.assertEqual(decision["quality_status"], "dominated_valid")
        self.assertFalse(decision["parent_eligible"])
        self.assertFalse(decision["pareto_eligible"])

    def test_cache_correctness_failure_never_becomes_a_parent(self):
        decision = decide_candidate({
            "protected_validation_bpb": 1.9,
            "inference_result": {
                "inference_status": "unstable",
                "correctness": {"passed": False},
            },
        }, [])
        self.assertEqual(decision["decision"], "invalid")
        self.assertFalse(decision["parent_eligible"])
