import unittest
import json
import contextlib
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from study.contracts_2026 import inference_benchmark
from study.contracts_2026.benchmark_policy import PROTOCOL_VERSION
from study.contracts_2026.inference_benchmark import (
    PROMPT_TOKEN_HASHES,
    PROMPT_TOKEN_IDS,
    PROTECTED_BOS_TOKEN_ID,
    WORKLOAD,
    finalize_payload,
    validate_workload,
)


class InferenceIntegrationTests(unittest.TestCase):
    def test_bracketed_evaluator_records_canonical_protocol_version(self):
        self.assertEqual(inference_benchmark.PROTOCOL_VERSION, PROTOCOL_VERSION)
        self.assertEqual(
            inference_benchmark.ACTIVE_INFERENCE_CONFIG["measurement"]["paired_baseline"]["protocol"],
            "baseline_before_candidate_baseline_after_for_each_window",
        )
        warmup = inference_benchmark.ACTIVE_INFERENCE_CONFIG["warmup"]
        self.assertEqual(warmup["mode"], "fixed_discarded_cached_decode_duration_v1")
        self.assertFalse(warmup["adaptive_steady_state_gate"])

    def test_workload_and_prompt_invariants_are_frozen(self):
        validate_workload()
        self.assertEqual(len(PROMPT_TOKEN_IDS), WORKLOAD["protected_prompt_count"])
        self.assertEqual(len(PROMPT_TOKEN_HASHES), 8)
        self.assertEqual({len(prompt) for prompt in PROMPT_TOKEN_IDS}, {1024})
        self.assertTrue(all(prompt[0] == PROTECTED_BOS_TOKEN_ID for prompt in PROMPT_TOKEN_IDS))
        self.assertEqual(len(set(PROMPT_TOKEN_IDS)), 8)

    def test_protected_inference_payload_exposes_pareto_metrics_and_raw_repetitions(self):
        blocks = [100.0, 99.0, 101.0, 100.5, 99.5, 100.0, 100.2, 99.8]
        result = finalize_payload(
            {
                "correctness_passed": True,
                "decode_blocks": blocks,
                "peak_allocated_cuda_memory_bytes": 123,
                "reserved_cuda_memory_bytes": 456,
                "kv_cache_bytes": 78,
            }
        )
        self.assertEqual(result["inference_status"], "stable")
        self.assertEqual(result["peak_allocated_inference_memory_bytes"], 123)
        self.assertEqual(result["reserved_cuda_memory_bytes"], 456)
        self.assertEqual(result["kv_cache_bytes"], 78)
        self.assertEqual(result["decode_block_velocities"], blocks)

    def test_invalid_correctness_has_no_search_scalar(self):
        result = finalize_payload(
            {
                "correctness_passed": False,
                "decode_blocks": [],
                "peak_allocated_cuda_memory_bytes": 1,
                "reserved_cuda_memory_bytes": 2,
                "kv_cache_bytes": 3,
            }
        )
        self.assertEqual(result["inference_status"], "invalid")
        self.assertIsNone(result["decode_velocity_point_estimate"])

    def test_fixed_warmup_is_discarded_before_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "result.json"
            calls = []

            def fake_repetition(*_args):
                calls.append(True)
                return (
                    {
                        "decode_tokens_per_second": 100.0,
                        "decode_seconds": 4.0,
                        "peak_allocated_cuda_memory_mb": 1.0,
                        "peak_reserved_cuda_memory_mb": 2.0,
                        "final_kv_cache_memory_mb": 3.0,
                        "prefill_milliseconds": 1.0,
                        "first_token_milliseconds": 1.0,
                    },
                    None,
                )

            argv = [
                "inference_benchmark.py",
                "--context", tmp,
                "--checkpoint", str(Path(tmp) / "checkpoint.pt"),
                "--output", str(output),
                "--protocol", PROTOCOL_VERSION,
            ]
            fake_module = SimpleNamespace()
            fake_module.Tokenizer = SimpleNamespace()
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(inference_benchmark, "load_candidate_definitions", return_value=fake_module), \
                    mock.patch.object(inference_benchmark, "load_checkpoint_file", return_value={}), \
                    mock.patch.object(inference_benchmark, "load_checkpoint_model", return_value=object()), \
                    mock.patch.object(inference_benchmark, "build_adapter", return_value=object()), \
                    mock.patch.object(inference_benchmark, "_load_tokenizer", return_value=object()), \
                    mock.patch.object(inference_benchmark, "fixed_input", return_value=object()), \
                    mock.patch.object(inference_benchmark, "protected_prompts", return_value=[object()] * 8), \
                    mock.patch.object(inference_benchmark, "precision_aware_correctness", return_value={"passed": True}), \
                    mock.patch.object(inference_benchmark, "one_measured_repetition", side_effect=fake_repetition):
                self.assertEqual(inference_benchmark.main(), 0)

            result = json.loads(output.read_text())
            self.assertEqual(result["warmup_mode"], "fixed_discarded_cached_decode_duration_v1")
            self.assertEqual(result["warmup_decode_seconds"], 20.0)
            self.assertEqual(result["warmup_repetition_count"], 5)
            self.assertEqual(result["warmup_blocks"], [])
            self.assertEqual(result["warmup_repetitions"], [])
            self.assertEqual(len(result["measurement_windows"]), 8)
            self.assertEqual(result["inference_status"], "stable")
            self.assertGreater(len(calls), 10)

    def test_bracketed_measurement_runs_after_fixed_warmup_without_stability_gate(self):
        candidate_adapter = object()
        baseline_adapter = object()
        calls = []
        memory_calls = []

        def fake_window(_adapter, _prompts, index, _offset):
            phase = len(calls) % 3
            calls.append((index, phase))
            base = index * 10_000
            started = base + phase * 2_000
            row = {
                "decode_tokens_per_second": 100.0,
                "started_at_unix_ns": started,
                "completed_at_unix_ns": started + 1_000,
                "peak_allocated_cuda_memory_mb": 1.0,
                "peak_reserved_cuda_memory_mb": 2.0,
                "final_kv_cache_memory_mb": 3.0,
                "prefill_milliseconds": 1.0,
                "first_token_milliseconds": 1.0,
            }
            return {**row, "repetitions": [row]}

        warmup = {
            "warmup_mode": "fixed_discarded_cached_decode_duration_v1",
            "warmup_decode_seconds": 20.0,
            "warmup_repetition_count": 5,
            "warmup_repetitions": [],
            "warmup_blocks": [],
        }
        def fake_memory_repetition(adapter, _prompt, repetition):
            memory_calls.append((adapter, repetition))
            return ({
                "peak_allocated_cuda_memory_mb": 7.0,
                "peak_reserved_cuda_memory_mb": 8.0,
                "final_kv_cache_memory_mb": 3.0,
            }, None)
        with mock.patch.object(inference_benchmark, "load_candidate_definitions", return_value=SimpleNamespace()), \
                mock.patch.object(inference_benchmark, "load_checkpoint_file", return_value={}), \
                mock.patch.object(inference_benchmark, "load_checkpoint_model", return_value=object()), \
                mock.patch.object(inference_benchmark, "build_adapter", return_value=baseline_adapter), \
                mock.patch.object(inference_benchmark, "fixed_input", return_value=object()), \
                mock.patch.object(inference_benchmark, "precision_aware_correctness", return_value={"passed": True}), \
                mock.patch.object(inference_benchmark, "_run_fixed_decode_warmup", side_effect=[warmup, warmup]), \
                mock.patch.object(inference_benchmark, "_run_official_window", side_effect=fake_window), \
                mock.patch.object(inference_benchmark, "one_measured_repetition", side_effect=fake_memory_repetition), \
                mock.patch.object(inference_benchmark.torch.cuda, "empty_cache"), \
                mock.patch.object(
                    inference_benchmark.torch.amp, "autocast", return_value=contextlib.nullcontext()
                ) as autocast:
            result = inference_benchmark._bracketed_inference_result(
                candidate_module=SimpleNamespace(),
                candidate_checkpoint={},
                candidate_model=object(),
                candidate_adapter=candidate_adapter,
                candidate_train=Path("candidate.train.py"),
                baseline_context=Path("baseline-context"),
                baseline_checkpoint_path=Path("baseline-checkpoint.pt"),
                tokenizer=object(),
                prompts=[object()] * 8,
                device=object(),
            )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["inference_status"], "stable")
        self.assertEqual(len(calls), 24)
        self.assertEqual(result["warmup_repetitions"], [])
        self.assertEqual(memory_calls, [(candidate_adapter, -1)])
        autocast.assert_called_once_with(device_type="cuda", dtype=inference_benchmark.torch.bfloat16)
        self.assertEqual(result["peak_allocated_inference_memory_bytes"], 7 * 1048576)
        self.assertIn("candidate_only_memory_measurement", result)

    def test_tokenizer_uses_authoritative_registered_cache_without_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "train.py"
            candidate.write_text("# candidate\n")
            context_prepare = root / "prepare.py"
            context_prepare.write_text(
                "class Tokenizer:\n"
                "    @classmethod\n"
                "    def from_directory(cls, path):\n"
                "        return path\n"
            )
            registered = root / "registered"
            tokenizer = registered / "tokenizer" / "tokenizer.pkl"
            tokenizer.parent.mkdir(parents=True)
            tokenizer.write_bytes(b"registered-tokenizer")
            old_cache = os.environ.get("AUTOPARETO_REGISTERED_CACHE")
            sys.modules.pop("prepare", None)
            try:
                os.environ["AUTOPARETO_REGISTERED_CACHE"] = str(registered)
                loaded = inference_benchmark._load_tokenizer(candidate)
                self.assertEqual(Path(loaded).resolve(), (registered / "tokenizer").resolve())
                missing_root = root / "missing"
                missing_root.mkdir()
                os.environ["AUTOPARETO_REGISTERED_CACHE"] = str(missing_root)
                with self.assertRaisesRegex(RuntimeError, "registered tokenizer"):
                    inference_benchmark._load_tokenizer(candidate)
            finally:
                sys.modules.pop("prepare", None)
                if old_cache is None:
                    os.environ.pop("AUTOPARETO_REGISTERED_CACHE", None)
                else:
                    os.environ["AUTOPARETO_REGISTERED_CACHE"] = old_cache

    def test_flash_attention_resolution_is_reused_after_first_known_good_load(self):
        good_module = SimpleNamespace(flash_attn_interface=object())
        broken_second_module = SimpleNamespace()
        calls = []

        def upstream_get_kernel(*_args, **_kwargs):
            calls.append(True)
            return good_module if len(calls) == 1 else broken_second_module

        fake_kernels = SimpleNamespace(get_kernel=upstream_get_kernel)
        with mock.patch.object(inference_benchmark, "_KERNEL_CACHE_MARKER", "_test_kernel_cache_marker"), \
                mock.patch.dict(sys.modules, {"kernels": fake_kernels}):
            inference_benchmark._install_kernel_resolution_cache(fake_kernels)
            first = fake_kernels.get_kernel("kernels-community/flash-attn3")
            second = fake_kernels.get_kernel("kernels-community/flash-attn3")
        self.assertIs(first, good_module)
        self.assertIs(second, good_module)
        self.assertEqual(len(calls), 1)
        self.assertTrue(hasattr(second, "flash_attn_interface"))

    def test_candidate_adapter_contract_is_supported_when_torch_is_available(self):
        try:
            from protected.inference_adapter import build_adapter
        except ModuleNotFoundError:
            self.skipTest("PyTorch is not installed in the local audit environment")

        class Adapter:
            def prefill(self, input_ids):
                return None, None

            def decode_one(self, token, state):
                return None, state

            def cache_nbytes(self, state):
                return 0

        class Module:
            @staticmethod
            def build_inference_adapter(model):
                return Adapter()

        adapter = build_adapter(Module, object())
        self.assertTrue(all(callable(getattr(adapter, name)) for name in ("prefill", "decode_one", "cache_nbytes")))


if __name__ == "__main__":
    unittest.main()
