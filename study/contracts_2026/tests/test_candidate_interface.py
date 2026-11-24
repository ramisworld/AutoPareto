import inspect
import pathlib
import tempfile
import types
import unittest
from unittest import mock

from study.contracts_2026 import candidate_process
from study.contracts_2026 import candidate_sandbox
from study.contracts_2026 import remote_runner
from study.contracts_2026 import train_launcher
from study.contracts_2026.bridge import REMOTE_RESULT_SCHEMA


class _Observer:
    def __init__(self, contract_id, n_star):
        self.contract_id = contract_id
        self.n_star = n_star
        self.progress_calls = []
        self.update_calls = 0

    def note_progress(self, value):
        self.progress_calls.append(value)
        return 0.25

    def after_optimizer_update(self):
        self.update_calls += 1


def _runtime_module():
    def contract_progress(contract_id, measured_training_time, total_tokens, n_star=None):
        return 0.0

    def contract_update_completed():
        return None

    return types.SimpleNamespace(
        contract_progress=contract_progress,
        contract_update_completed=contract_update_completed,
    )


class CandidateInterfaceTests(unittest.TestCase):
    def test_fixed_token_contract_propagates_n_star_into_candidate_environment(self):
        metadata = {"contract_id": "N", "contract_parameters": {"n_star": 88_604_672}}
        self.assertEqual(remote_runner._candidate_contract_environment(metadata), {"N_STAR": "88604672"})

        time_metadata = {"contract_id": "T", "contract_parameters": {"n_star": None}}
        self.assertEqual(remote_runner._candidate_contract_environment(time_metadata), {})

    def test_fixed_token_contract_rejects_missing_or_invalid_n_star_environment(self):
        for metadata in (
            {"contract_id": "N", "contract_parameters": {}},
            {"contract_id": "N", "contract_parameters": {"n_star": 0}},
            {"contract_id": "N", "contract_parameters": {"n_star": "88604672"}},
        ):
            with self.subTest(metadata=metadata):
                with self.assertRaisesRegex(ValueError, "N_STAR"):
                    remote_runner._candidate_contract_environment(metadata)

    def test_candidate_sandbox_preserves_valid_n_star(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            with mock.patch.dict("os.environ", {"N_STAR": "88604672"}, clear=False):
                environment = candidate_sandbox._sanitized_candidate_environment(root, root, root / "home")
            self.assertEqual(environment["N_STAR"], "88604672")

    def test_registered_baseline_is_candidate_compatible_and_hash_registered(self):
        baseline = (
            pathlib.Path(__file__).resolve().parents[1]
            / "historical_reference"
            / "baseline_source"
            / "train.py"
        )
        source = baseline.read_text()
        self.assertNotIn("evaluate_bpb", source)
        candidate_process.validate_candidate_contract_api(baseline, require_progress=False)
        self.assertEqual(
            candidate_process._sha256(baseline),
            candidate_process.REGISTERED_BASELINE_SOURCE_SHA256,
        )

    def test_protected_adapter_preserves_staged_public_signature_for_n_and_t(self):
        for contract_id, n_star, call in (
            ("N", 88_604_672, lambda runtime: runtime.contract_progress("N", 12.0, 32_768, 88_604_672)),
            ("T", None, lambda runtime: runtime.contract_progress("T", 12.0, 32_768)),
        ):
            with self.subTest(contract_id=contract_id):
                runtime = _runtime_module()
                staged_signature = inspect.signature(runtime.contract_progress)
                observer = _Observer(contract_id, n_star)
                candidate_process._install_contract_api(runtime, observer)
                self.assertEqual(inspect.signature(runtime.contract_progress), staged_signature)
                self.assertEqual(call(runtime), 0.25)
                self.assertEqual(observer.progress_calls, [0.0])
                runtime.contract_update_completed()
                self.assertEqual(observer.update_calls, 1)

    def test_wrong_static_contract_hook_arity_is_rejected_before_cuda(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = pathlib.Path(tmp) / "train.py"
            candidate.write_text(
                "from study_runtime import contract_progress\n"
                "contract_progress('N', 1.0)\n"
            )
            with self.assertRaisesRegex(
                candidate_process.CandidateInterfaceCompatibilityError,
                "contract_progress must receive",
            ):
                candidate_process.validate_candidate_contract_api(candidate)

    def test_candidate_side_validation_is_rejected_before_cuda(self):
        cases = (
            "from prepare import evaluate_bpb\nevaluate_bpb(model, tokenizer, 1)\n",
            "import prepare\nprepare.evaluate_bpb(model, tokenizer, 1)\n",
        )
        for source in cases:
            with self.subTest(source=source), tempfile.TemporaryDirectory() as tmp:
                candidate = pathlib.Path(tmp) / "train.py"
                candidate.write_text(source)
                with self.assertRaisesRegex(
                    candidate_process.CandidateInterfaceCompatibilityError,
                    "candidate-side validation/evaluation is forbidden",
                ):
                    candidate_process.validate_candidate_contract_api(candidate)

    def test_candidate_without_validation_call_passes_static_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = pathlib.Path(tmp) / "train.py"
            candidate.write_text(
                "from prepare import make_dataloader\n"
                "from study_runtime import contract_progress\n"
                "make_dataloader\n"
                "contract_progress('N', 0.0, 32768, 88604672)\n"
            )
            candidate_process.validate_candidate_contract_api(candidate)

    def test_missing_progress_hook_is_rejected_before_remote_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = pathlib.Path(tmp) / "train.py"
            candidate.write_text("from prepare import make_dataloader\nmake_dataloader\n")
            with self.assertRaisesRegex(
                candidate_process.CandidateInterfaceCompatibilityError,
                "must call contract_progress",
            ):
                candidate_process.validate_candidate_contract_api(candidate)

    def test_fixed_token_progress_literal_must_match_protected_n_star(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = pathlib.Path(tmp) / "train.py"
            candidate.write_text(
                "from study_runtime import contract_progress\n"
                "contract_progress('N', 0.0, 32768, 88604672)\n"
            )
            with self.assertRaisesRegex(
                candidate_process.CandidateInterfaceCompatibilityError,
                "disagrees with protected N_STAR",
            ):
                candidate_process.validate_candidate_contract_api(
                    candidate,
                    expected_n_star=84_934_656,
                )

    def test_fixed_token_progress_matching_literal_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = pathlib.Path(tmp) / "train.py"
            candidate.write_text(
                "from study_runtime import contract_progress\n"
                "contract_progress('N', 0.0, 32768, 84934656)\n"
            )
            candidate_process.validate_candidate_contract_api(
                candidate,
                expected_n_star=84_934_656,
            )

    def test_registered_baseline_is_allowed_only_in_baseline_mode(self):
        baseline = (
            pathlib.Path(__file__).resolve().parents[1]
            / "historical_reference"
            / "baseline_source"
            / "train.py"
        )
        with self.assertRaisesRegex(
            candidate_process.CandidateInterfaceCompatibilityError,
            "must call contract_progress",
        ):
            candidate_process.validate_candidate_contract_api(baseline)
        candidate_process.validate_candidate_contract_api(baseline, require_progress=False)

    def test_exact_pre_repair_note_progress_failures_are_harness_failures(self):
        metadata = {"campaign_id": "pair-a-fixed-tokens-seed3", "contract_id": "N", "attempt_number": 1}
        for arity in (4, 5):
            with self.subTest(arity=arity):
                payload = {
                    "status": "error",
                    "failure_origin": "harness",
                    "error": (
                        "TypeError: CandidateObserver.note_progress() takes 2 positional arguments "
                        f"but {arity} were given"
                    ),
                }
                result = remote_runner._classify_worker_failure(payload, metadata)
                self.assertEqual(result["schema"], REMOTE_RESULT_SCHEMA)
                self.assertEqual(result["status"], "harness_failure")
                self.assertEqual(result["failure_class"], "harness")

    def test_source_python_error_is_candidate_failure_but_unknown_is_not_guessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = pathlib.Path(tmp) / "train.py"
            candidate.write_text("raise NameError('candidate bug')\n")
            try:
                exec(compile(candidate.read_text(), str(candidate), "exec"), {})
            except BaseException as exc:
                self.assertEqual(candidate_process._failure_origin(exc, candidate), "candidate")
        metadata = {"campaign_id": "pair-a-fixed-tokens-seed3", "contract_id": "N", "attempt_number": 1}
        with self.assertRaisesRegex(RuntimeError, "unknown"):
            remote_runner._classify_worker_failure(
                {"status": "error", "failure_origin": "unknown", "error": "opaque"}, metadata
            )

    def test_post_training_missing_gated_value_embedding_is_candidate_failure(self):
        exc = RuntimeError(
            "bracketed inference failed: RuntimeError: "
            "canonical value embedding is missing gated layer 0"
        )
        self.assertEqual(train_launcher._post_training_failure_origin(exc), "candidate")

    def test_post_training_cache_size_misreport_is_candidate_failure(self):
        # The exact launcher error of the Opus arm's attempt 9 (2026-11-05): the
        # candidate's CUDA-graph adapter left its static I/O tensors out of
        # cache_nbytes, and the protected traversal of its own state caught it.
        exc = RuntimeError(
            "bracketed inference did not produce /opt/autopareto-candidate-attempts/x/candidate-output/"
            "position-018-pair-a-fixed-time-seed3.json.inference.json: returncode=1; stderr=Traceback "
            "(most recent call last):\n  File \"/opt/autopareto_trusted/protected/inference_adapter.py\", "
            "line 340, in validated_cache_nbytes\n    raise RuntimeError(\nRuntimeError: candidate "
            "cache_nbytes reported 7864320 bytes but protected state traversal found 7907344 bytes\n"
        )
        self.assertEqual(train_launcher._post_training_failure_origin(exc), "candidate")

    def test_protected_cache_size_check_message_matches_the_marker(self):
        # The marker must track the message the protected check actually raises.
        import torch

        from protected.inference_adapter import validated_cache_nbytes

        class UndercountingAdapter:
            def cache_nbytes(self, state):
                return int(state["k"].untyped_storage().nbytes())

        state = {"k": torch.zeros(4, dtype=torch.float32), "io": torch.zeros(2, dtype=torch.float32)}
        with self.assertRaises(RuntimeError) as raised:
            validated_cache_nbytes(UndercountingAdapter(), state)
        self.assertIn("candidate cache_nbytes reported 16 bytes but protected state traversal found 24 bytes",
                      str(raised.exception))
        wrapped = RuntimeError(f"bracketed inference did not produce out.json: returncode=1; stderr={raised.exception}")
        self.assertEqual(train_launcher._post_training_failure_origin(wrapped), "candidate")

    def test_unrecognized_post_training_failure_remains_unknown(self):
        self.assertEqual(
            train_launcher._post_training_failure_origin(RuntimeError("CUDA runtime failure")),
            "unknown",
        )

    def test_protected_inference_shape_failure_is_candidate_failure(self):
        error = RuntimeError(
            "bracketed inference did not produce output: "
            "Traceback: /opt/autopareto_trusted/protected/inference_adapter.py:216 "
            "value_embedding = value_embedding.view(batch, length, value_heads, attn.head_dim) "
            "RuntimeError: shape is invalid for input of size 16384"
        )
        self.assertEqual(train_launcher._post_training_failure_origin(error), "candidate")

    def test_candidate_protected_inference_exception_is_explicitly_classified(self):
        error = train_launcher.CandidateProtectedInferenceError(
            "protected inference rejected candidate architecture"
        )
        self.assertEqual(train_launcher._post_training_failure_origin(error), "candidate")

    def test_new_runtime_generation_materializes_changed_trusted_worker_source(self):
        """A changed protected worker must use a new immutable runtime directory."""
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source = root / "repository" / "study" / "contracts_2026"
            source.mkdir(parents=True)
            (source / "__init__.py").write_text("")
            worker = source / "candidate_process.py"
            worker.write_text("VERSION = 1\n")
            runtime_root = root / "runtime"
            with mock.patch.dict(
                "os.environ", {"AUTOPARETO_CANDIDATE_RUNTIME_ROOT": str(runtime_root)}, clear=False
            ), mock.patch.object(candidate_sandbox, "SANDBOX_POLICY_VERSION", "generation-one"):
                first = candidate_sandbox._prepare_process_runtime(
                    repository=root / "repository", study_files=("__init__.py", "candidate_process.py"), protected_files=()
                )
            worker.write_text("VERSION = 2\n")
            with mock.patch.dict(
                "os.environ", {"AUTOPARETO_CANDIDATE_RUNTIME_ROOT": str(runtime_root)}, clear=False
            ), mock.patch.object(candidate_sandbox, "SANDBOX_POLICY_VERSION", "generation-two"):
                second = candidate_sandbox._prepare_process_runtime(
                    repository=root / "repository", study_files=("__init__.py", "candidate_process.py"), protected_files=()
                )
            self.assertNotEqual(first, second)
            self.assertEqual((second / "study/contracts_2026/candidate_process.py").read_text(), "VERSION = 2\n")


if __name__ == "__main__":
    unittest.main()
