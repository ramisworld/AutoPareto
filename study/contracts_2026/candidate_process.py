"""Untrusted candidate execution host.

This module is intentionally the only process that executes candidate train.py.
It contains no trusted evaluator, campaign history, Pareto logic, or protected
parent observer.  All messages sent to the parent are untrusted attestations;
the parent validates their sequence and derives the accepted counters.
"""

from __future__ import annotations

import argparse
import ast
import functools
import hashlib
import json
import os
import pathlib
import sys
import time
import traceback
import types
from typing import Any


class CandidateStop(RuntimeError):
    """The trusted parent closed the candidate's training boundary."""


class CandidateWarmupStop(CandidateStop):
    """The trusted parent ended disposable compilation warm-up."""


class CandidateExitedBeforeStop(RuntimeError):
    """The candidate's own training loop returned before the protected stop."""


class CandidateInterfaceCompatibilityError(RuntimeError):
    """The protected worker and staged candidate API disagree.

    This is a harness error, never evidence about a proposed architecture. The
    public ``study_runtime`` signatures are part of the registered instrument
    and must remain callable after the protected observer is installed.
    """


def validate_candidate_contract_api(
    candidate: pathlib.Path,
    *,
    require_progress: bool = True,
    expected_n_star: int | None = None,
) -> None:
    """Fail early for statically provable misuse of staged contract hooks.

    The check intentionally covers only direct, unambiguous calls. Dynamic
    Python remains governed by the protected observer so the validator itself
    does not narrow the frozen candidate search space.
    """
    try:
        tree = ast.parse(candidate.read_text(), filename=str(candidate))
    except SyntaxError as exc:
        raise CandidateInterfaceCompatibilityError(
            f"candidate source does not parse: {exc.msg}"
        ) from exc

    progress_names = {"contract_progress"}
    update_names = {"contract_update_completed"}
    runtime_modules: set[str] = set()
    prepare_modules: set[str] = set()
    evaluation_names: set[str] = set()
    progress_call_found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "study_runtime":
            for imported in node.names:
                if imported.name == "contract_progress":
                    progress_names.add(imported.asname or imported.name)
                elif imported.name == "contract_update_completed":
                    update_names.add(imported.asname or imported.name)
        elif isinstance(node, ast.Import):
            for imported in node.names:
                if imported.name == "study_runtime":
                    runtime_modules.add(imported.asname or imported.name)
                elif imported.name == "prepare":
                    prepare_modules.add(imported.asname or imported.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "prepare":
            for imported in node.names:
                if imported.name == "evaluate_bpb":
                    evaluation_names.add(imported.asname or imported.name)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in evaluation_names | {"evaluate_bpb"}
        ) or (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in prepare_modules
            and node.func.attr == "evaluate_bpb"
        ):
            raise CandidateInterfaceCompatibilityError(
                "candidate-side validation/evaluation is forbidden; "
                "the protected parent evaluator owns validation"
            )
        name: str | None = None
        if isinstance(node.func, ast.Name) and node.func.id in progress_names:
            name = "contract_progress"
        elif (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in runtime_modules
            and node.func.attr == "contract_progress"
        ):
            name = "contract_progress"
        elif isinstance(node.func, ast.Name) and node.func.id in update_names:
            name = "contract_update_completed"
        elif (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in runtime_modules
            and node.func.attr == "contract_update_completed"
        ):
            name = "contract_update_completed"
        if name == "contract_progress":
            progress_call_found = True
            if not 3 <= len(node.args) <= 4:
                raise CandidateInterfaceCompatibilityError(
                    "contract_progress must receive contract_id, measured_training_time, "
                    "total_tokens, and optional n_star"
                )
            if any(keyword.arg not in {"n_star"} for keyword in node.keywords):
                raise CandidateInterfaceCompatibilityError(
                    "contract_progress accepts only the optional n_star keyword"
                )
            if expected_n_star is not None:
                n_star_node = node.args[3] if len(node.args) == 4 else next(
                    (keyword.value for keyword in node.keywords if keyword.arg == "n_star"),
                    None,
                )
                if n_star_node is None:
                    raise CandidateInterfaceCompatibilityError(
                        "Contract N contract_progress must pass the protected N_STAR"
                    )
                if isinstance(n_star_node, ast.Constant) and n_star_node.value != expected_n_star:
                    raise CandidateInterfaceCompatibilityError(
                        f"candidate N_STAR literal {n_star_node.value!r} disagrees with protected "
                        f"N_STAR {expected_n_star}"
                    )
        elif name == "contract_update_completed" and (node.args or node.keywords):
            raise CandidateInterfaceCompatibilityError(
                "contract_update_completed takes no arguments"
            )
    if require_progress and not progress_call_found:
        raise CandidateInterfaceCompatibilityError(
            "candidate must call contract_progress before protected optimizer updates"
        )


def _failure_origin(exc: BaseException, candidate: pathlib.Path) -> str:
    """Classify worker exceptions without turning harness faults into science.

    Candidate-source exceptions are the documented policy for ordinary Python
    mistakes: they are candidate failures. Protected API, environment, and
    unknown exceptions are never silently consumed as research attempts.
    """
    message = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, CandidateExitedBeforeStop):
        return "candidate"
    if isinstance(exc, CandidateInterfaceCompatibilityError):
        return "harness"
    # Regression guard for the pre-repair API swap. These exact arity failures
    # prove that the protected observer exposed the wrong public signature.
    if "CandidateObserver.note_progress() takes 2 positional arguments but" in message:
        return "harness"
    lowered = message.lower()
    if any(marker in lowered for marker in (
        "cudagetdevicecount", "cuda driver", "no cuda gpus are available",
        "flash-attn", "flash_attn", "pyarrow", "tokenizer.pkl", "permission denied",
    )):
        return "infrastructure"
    candidate_path = candidate.resolve()
    try:
        frames = traceback.extract_tb(exc.__traceback__)
        if any(pathlib.Path(frame.filename).resolve() == candidate_path for frame in frames):
            return "candidate"
    except (OSError, RuntimeError):
        pass
    return "unknown"


def _install_contract_api(runtime_module: Any, observer: "CandidateObserver") -> None:
    """Install a signature-preserving bridge over candidate-visible hooks.

    The candidate supplies progress-call placement only. Resource totals and
    timing remain derived from the protected observer, so reported numeric
    values cannot affect either Contract T or Contract N.
    """
    public_progress = runtime_module.contract_progress
    public_update = runtime_module.contract_update_completed

    @functools.wraps(public_progress)
    def protected_contract_progress(
        contract_id: str,
        measured_training_time: float,
        total_tokens: int,
        n_star: int | None = None,
    ) -> float:
        del measured_training_time, total_tokens
        if contract_id != observer.contract_id:
            raise CandidateInterfaceCompatibilityError(
                f"candidate declared Contract {contract_id!r}; expected {observer.contract_id!r}"
            )
        if observer.contract_id == "N" and n_star != observer.n_star:
            raise CandidateInterfaceCompatibilityError(
                "candidate N_STAR disagrees with the protected fixed-token contract"
            )
        return observer.note_progress(0.0)

    @functools.wraps(public_update)
    def protected_contract_update_completed() -> None:
        observer.after_optimizer_update()

    runtime_module.contract_progress = protected_contract_progress
    runtime_module.contract_update_completed = protected_contract_update_completed


REGISTERED_BASELINE_SOURCE_SHA256 = "b2fb8788f02517af8bc2fcfa059fb799a6974c9050d00e322168ec414fac4fab"


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_value(namespace: dict[str, Any], name: str) -> Any:
    value = namespace.get(name)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _checkpoint_payload(namespace: dict[str, Any], seed: int, candidate: pathlib.Path) -> dict[str, Any]:
    model = namespace.get("model")
    config = namespace.get("config")
    if model is None or config is None:
        raise RuntimeError("candidate must expose model and config for protected checkpointing")
    unwrapped = getattr(model, "_orig_mod", model)
    if not hasattr(unwrapped, "state_dict") or not hasattr(config, "__dict__"):
        raise RuntimeError("candidate model/config are not checkpoint-compatible")
    config_payload = dict(vars(config))
    return {
        "format_version": 2,
        "seed": int(seed),
        "config": config_payload,
        "config_sha256": hashlib.sha256(
            (json.dumps(config_payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        ).hexdigest(),
        "state_dict": {
            key: value.detach().cpu() for key, value in unwrapped.state_dict().items()
        },
        "train_file": candidate.name,
        "source_sha256": _sha256(candidate),
    }


class CandidateObserver:
    """Local instrumentation with no reference to trusted parent objects."""

    def __init__(self, *, contract_id: str, n_star: int | None, event_fd: int, control_fd: int, require_progress: bool) -> None:
        self.contract_id = contract_id
        self.n_star = n_star
        self.event_stream = open(event_fd, "w", buffering=1, encoding="utf-8")
        self.control_stream = open(control_fd, "r", buffering=1, encoding="utf-8")
        self.require_progress = require_progress
        self.boundary_started = False
        self.boundary_closed = False
        self.total_tokens = 0
        self.microbatches = 0
        self.optimizer_updates = 0
        self._pending_tokens = 0
        self._pending_microbatches = 0
        self._batch: tuple[int, int] | None = None
        self._progress_calls = 0
        self.completed_timing_seconds = 0.0
        self.synchronize = None
        self.phase = "warmup"
        self.warmup_initial_state_sha256: str | None = None
        self.scientific_initial_state_sha256: str | None = None

    @staticmethod
    def _state_sha256(module: Any) -> str:
        """Hash pristine parameters/buffers without retaining model state."""
        target = getattr(module, "_orig_mod", module)
        import torch

        digest = hashlib.sha256()
        for name, value in list(target.named_parameters()) + list(target.named_buffers()):
            tensor = value.detach().to(device="cpu", copy=True).contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(repr(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def record_initial_state(self, module: Any) -> None:
        if self.phase == "warmup" and self.warmup_initial_state_sha256 is None:
            self.warmup_initial_state_sha256 = self._state_sha256(module)
        elif self.phase == "scientific" and self.scientific_initial_state_sha256 is None:
            self.scientific_initial_state_sha256 = self._state_sha256(module)

    def reset_for_scientific(self) -> None:
        """Drop all warm-up accounting while retaining this worker/process."""
        self.phase = "scientific"
        self.boundary_started = False
        self.boundary_closed = False
        self.total_tokens = 0
        self.microbatches = 0
        self.optimizer_updates = 0
        self._pending_tokens = 0
        self._pending_microbatches = 0
        self._batch = None
        self._progress_calls = 0
        self.completed_timing_seconds = 0.0
        self._last_update_total_tokens = 0
        self._last_update_microbatches = 0

    def event(self, kind: str, **payload: Any) -> None:
        self.event_stream.write(json.dumps({
            "protocol": "autopareto_candidate_event_v1",
            "kind": kind,
            "monotonic_ns": time.perf_counter_ns(),
            **payload,
        }, sort_keys=True) + "\n")
        self.event_stream.flush()

    def _read_control(self) -> str:
        value = self.control_stream.readline().strip()
        if not value:
            raise CandidateStop("trusted parent closed the control stream")
        return value

    def before_first_training_forward(self) -> None:
        if self.boundary_started:
            return
        self.event("boundary_ready", phase=self.phase, pid=os.getpid())
        command = self._read_control()
        if not command.startswith("START "):
            raise RuntimeError("trusted parent did not open the candidate boundary")
        self.boundary_started = True
        self.event("boundary_started")

    def observe_microbatch_start(self, input_ids: Any, targets: Any) -> tuple[int, int, int]:
        if self.boundary_closed:
            raise CandidateStop("candidate boundary is closed")
        if not self.boundary_started:
            self.before_first_training_forward()
        shape = getattr(input_ids, "shape", None)
        target_shape = getattr(targets, "shape", None)
        ndim = getattr(input_ids, "ndim", None)
        if shape is None or target_shape is None or ndim != 2 or tuple(shape) != tuple(target_shape):
            raise RuntimeError("training forwards must receive matching [B, 2048] tensors")
        batch_size, sequence_length = int(shape[0]), int(shape[1])
        if batch_size <= 0 or sequence_length != 2048:
            raise RuntimeError("protected training sequence length must be exactly 2048")
        tokens = batch_size * sequence_length
        if self.contract_id == "N" and self.n_star is not None and self.total_tokens + tokens > self.n_star:
            raise RuntimeError("fixed-token candidate attempted a partial final microbatch")
        if self._batch is None:
            self._batch = (batch_size, sequence_length)
        elif self._batch != (batch_size, sequence_length):
            raise RuntimeError("device batch or sequence length changed")
        self._pending_tokens += tokens
        self._pending_microbatches += 1
        self.event(
            "microbatch_started",
            device_batch_size=batch_size,
            sequence_length=sequence_length,
            tokens=tokens,
        )
        return batch_size, sequence_length, tokens

    def observe_microbatch_completed(self, observed: tuple[int, int, int]) -> None:
        del observed
        if self._pending_microbatches <= 0:
            raise RuntimeError("microbatch completion without a start")
        self.total_tokens += self._pending_tokens // self._pending_microbatches
        self.microbatches += 1
        tokens = self._pending_tokens // self._pending_microbatches
        self.event(
            "microbatch_completed",
            device_batch_size=self._batch[0] if self._batch else None,
            sequence_length=self._batch[1] if self._batch else None,
            tokens=tokens,
            total_tokens=self.total_tokens,
            microbatches=self.microbatches,
        )
        self._pending_tokens -= tokens
        self._pending_microbatches -= 1

    def note_progress(self, _value: float) -> float:
        self._progress_calls += 1
        if self.contract_id == "T":
            value = min(max(self.completed_timing_seconds, 0.0) / 300.0, 1.0)
        elif self.n_star:
            value = min(self.total_tokens / self.n_star, 1.0)
        else:
            raise RuntimeError("invalid contract")
        self.event("progress", contract=self.contract_id, value=value, total_tokens=self.total_tokens)
        return value

    def before_optimizer_update(self) -> None:
        if self._pending_microbatches:
            raise RuntimeError("optimizer update occurred before all microbatches completed")
        if self.microbatches <= getattr(self, "_last_update_microbatches", 0):
            raise RuntimeError("optimizer update had no observed microbatch")
        if self.require_progress and self._progress_calls <= 0:
            raise RuntimeError("candidate did not call protected contract_progress before update")

    def after_optimizer_update(self) -> None:
        # The observer is installed both at the optimizer hook and at the
        # candidate's optional contract_update_completed() API.  Treat the
        # second notification for the same completed update as a harmless
        # duplicate so candidates remain free to use that API explicitly.
        if self.microbatches <= getattr(self, "_last_update_microbatches", 0):
            return
        if self.synchronize is not None:
            self.synchronize()
        if self._pending_microbatches != 0:
            raise RuntimeError("optimizer update occurred before all microbatches completed")
        if self._batch is None:
            raise RuntimeError("optimizer update has no observed batch")
        self.optimizer_updates += 1
        update_tokens = self.total_tokens - getattr(self, "_last_update_total_tokens", 0)
        if update_tokens not in (32768, 65536, 131072, 262144, 524288):
            raise RuntimeError("tokens per optimizer update are outside the common allowed set")
        self._last_update_total_tokens = self.total_tokens
        self._last_update_microbatches = self.microbatches
        self.event(
            "optimizer_update_completed",
            optimizer_updates=self.optimizer_updates,
            microbatches=self.microbatches,
            update_tokens=update_tokens,
            total_tokens=self.total_tokens,
            device_batch_size=self._batch[0],
            sequence_length=self._batch[1],
            accumulation_steps=update_tokens // (self._batch[0] * self._batch[1]),
        )
        self._progress_calls = 0
        decision = self._read_control()
        if decision in {"STOP", "WARMUP_STOP"}:
            self.boundary_closed = True
            if decision == "WARMUP_STOP":
                raise CandidateWarmupStop("disposable compilation warm-up complete")
            raise CandidateStop("trusted parent closed the candidate boundary")
        if decision.startswith("CONTINUE_TIMED "):
            self.completed_timing_seconds = float(decision.split(maxsplit=1)[1])
            return
        if decision in {"CONTINUE", "CONTINUE_UNTIMED"}:
            return
        raise RuntimeError("trusted parent returned an invalid boundary decision")

    def summary(self) -> dict[str, Any]:
        if self._batch is None or self.optimizer_updates <= 0:
            raise RuntimeError("no complete optimizer update was observed")
        if self._pending_microbatches:
            raise RuntimeError("training ended with an incomplete optimizer update")
        last_update_total = getattr(self, "_last_update_total_tokens", 0)
        return {
            "total_tokens": int(last_update_total),
            "microbatches": int(self.microbatches),
            "optimizer_updates": int(self.optimizer_updates),
            "sequence_length": int(self._batch[1]),
            "device_batch_size": int(self._batch[0]),
            "total_batch_size": int(last_update_total // self.optimizer_updates),
            "accumulation_steps": int(last_update_total // self.optimizer_updates // (self._batch[0] * self._batch[1])),
        }


def execute(args: argparse.Namespace) -> int:
    context = args.context.resolve()
    candidate = (context / args.candidate).resolve()
    if candidate.parent != context or candidate.name != "train.py" or not candidate.is_file() or candidate.is_symlink():
        raise SystemExit("candidate must be the regular train.py in the isolated context")
    if args.baseline_calibration and args.contract != "T" and not args.baseline_smoke:
        raise SystemExit("baseline calibration is valid only for Contract T")
    if args.baseline_smoke and args.contract != "N":
        raise SystemExit("baseline smoke is valid only for Contract N")
    if args.baseline_calibration and args.baseline_smoke:
        raise SystemExit("baseline calibration and baseline smoke are mutually exclusive")
    if (args.baseline_calibration or args.baseline_smoke) and _sha256(candidate) != REGISTERED_BASELINE_SOURCE_SHA256:
        raise SystemExit("baseline mode is restricted to the registered baseline source")
    observer = CandidateObserver(
        contract_id=args.contract,
        n_star=args.n_star,
        event_fd=args.event_fd,
        control_fd=args.control_fd,
        require_progress=not (args.baseline_calibration or args.baseline_smoke),
    )
    previous_main = sys.modules.get("__main__")
    candidate_module = types.ModuleType("__main__")
    candidate_module.__file__ = str(candidate)
    candidate_namespace = candidate_module.__dict__
    sys.modules["__main__"] = candidate_module
    seed_originals = None
    original_module_call = None
    original_optimized_module_call = None
    optimized_module_class = None
    original_optimizer_init = None
    original_contract_progress = None
    original_contract_update_completed = None
    runtime_module = None
    call_depth = 0
    stopped = False
    warmup_stopped = False
    validation_attempted = False
    warmup_lifecycle = {
        "same_worker_process": not args.warmup_only,
        "warmup_model_discarded": False,
        "scientific_state_reconstructed": False,
    }
    try:
        # This happens before importing torch or opening a CUDA context. It
        # catches unambiguous contract-hook misuse without spending protected
        # GPU time and emits a structured harness receipt on rejection.
        validate_candidate_contract_api(
            candidate,
            require_progress=observer.require_progress,
            expected_n_star=args.n_star if args.contract == "N" else None,
        )
        import importlib
        import torch

        # Bind the CUDA synchronization function before candidate source is
        # executed.  Completion notifications are emitted only after all CUDA
        # work for the update is complete; parent timing is then authoritative
        # over a completed-update boundary rather than an enqueue boundary.
        observer.synchronize = torch.cuda.synchronize

        seed_originals = (
            torch.manual_seed,
            torch.cuda.manual_seed,
            torch.cuda.manual_seed_all,
        )
        torch.manual_seed = lambda _ignored: seed_originals[0](args.seed)
        torch.cuda.manual_seed = lambda _ignored: seed_originals[1](args.seed)
        torch.cuda.manual_seed_all = lambda _ignored: seed_originals[2](args.seed)

        sys.path.insert(0, str(context))
        prepare_module = importlib.import_module("prepare")
        original_evaluate = getattr(prepare_module, "evaluate_bpb", None)

        def forbidden_evaluate(*_values: Any, **_kwargs: Any) -> float:
            nonlocal validation_attempted
            validation_attempted = True
            raise RuntimeError("candidate-side validation is forbidden; parent evaluator owns validation")

        if original_evaluate is not None:
            prepare_module.evaluate_bpb = forbidden_evaluate

        runtime_module = importlib.import_module("study_runtime")
        original_contract_progress = runtime_module.contract_progress
        original_contract_update_completed = runtime_module.contract_update_completed
        _install_contract_api(runtime_module, observer)

        original_module_call = torch.nn.Module.__call__

        def observed_module_call(module: Any, *module_args: Any, **module_kwargs: Any) -> Any:
            nonlocal call_depth
            top_level = call_depth == 0
            looks_like_training = (
                top_level
                and len(module_args) >= 2
                and all(isinstance(value, torch.Tensor) and value.ndim == 2 for value in module_args[:2])
                and tuple(module_args[0].shape) == tuple(module_args[1].shape)
            )
            if looks_like_training:
                observer.record_initial_state(module)
            observed = observer.observe_microbatch_start(module_args[0], module_args[1]) if looks_like_training else None
            call_depth += 1
            try:
                result = original_module_call(module, *module_args, **module_kwargs)
            finally:
                call_depth -= 1
            if observed is not None:
                observer.observe_microbatch_completed(observed)
            return result

        torch.nn.Module.__call__ = observed_module_call

        # torch.compile returns torch._dynamo.eval_frame.OptimizedModule,
        # which overrides Module.__call__.  Observe its outer call as the
        # protected microbatch boundary while its nested original-module calls
        # remain suppressed by call_depth.
        try:
            from torch._dynamo.eval_frame import OptimizedModule

            optimized_module_class = OptimizedModule
            original_optimized_module_call = OptimizedModule.__call__

            def observed_optimized_module_call(module: Any, *module_args: Any, **module_kwargs: Any) -> Any:
                nonlocal call_depth
                top_level = call_depth == 0
                looks_like_training = (
                    top_level
                    and len(module_args) >= 2
                    and all(isinstance(value, torch.Tensor) and value.ndim == 2 for value in module_args[:2])
                    and tuple(module_args[0].shape) == tuple(module_args[1].shape)
                )
                if looks_like_training:
                    observer.record_initial_state(module)
                observed = observer.observe_microbatch_start(module_args[0], module_args[1]) if looks_like_training else None
                call_depth += 1
                try:
                    # The compiled wrapper calls the original module through
                    # Module.__call__.  Avoid wrapping that nested path again:
                    # it can introduce graph breaks and retain extra
                    # activation references without adding any accounting.
                    torch.nn.Module.__call__ = original_module_call
                    result = original_optimized_module_call(module, *module_args, **module_kwargs)
                finally:
                    torch.nn.Module.__call__ = observed_module_call
                    call_depth -= 1
                if observed is not None:
                    observer.observe_microbatch_completed(observed)
                return result

            OptimizedModule.__call__ = observed_optimized_module_call
        except (ImportError, AttributeError):
            pass

        original_optimizer_init = torch.optim.Optimizer.__init__

        def observed_optimizer_init(optimizer: Any, *optimizer_args: Any, **optimizer_kwargs: Any) -> None:
            original_optimizer_init(optimizer, *optimizer_args, **optimizer_kwargs)
            original_step = optimizer.step
            if getattr(optimizer, "_autopareto_candidate_step_hook", False):
                return

            def observed_step(*step_args: Any, **step_kwargs: Any) -> Any:
                observer.before_optimizer_update()
                result = original_step(*step_args, **step_kwargs)
                observer.after_optimizer_update()
                return result

            optimizer.step = observed_step
            optimizer._autopareto_candidate_step_hook = True

        torch.optim.Optimizer.__init__ = observed_optimizer_init

        source = candidate.read_text()
        setup_marker = "# Setup: tokenizer, model, optimizer, dataloader"
        marker_index = source.find(setup_marker)
        if marker_index < 0:
            raise RuntimeError("candidate is missing the registered setup marker")
        # Compile definitions once and execute the setup/training body twice.
        # Reusing the same code objects/classes lets torch.compile retain its
        # process-local graph cache while each pass constructs fresh state.
        definitions_source = source[:marker_index]
        training_source = source[marker_index:]
        definitions_code = compile(definitions_source, str(candidate), "exec")
        training_code = compile(training_source, str(candidate), "exec")
        exec(definitions_code, candidate_namespace, candidate_namespace)
        while True:
            try:
                exec(training_code, candidate_namespace, candidate_namespace)
            except CandidateWarmupStop:
                if args.warmup_only:
                    warmup_stopped = True
                    break
                if warmup_lifecycle["warmup_model_discarded"]:
                    raise RuntimeError("candidate requested warm-up stop more than once")
                # Clear every candidate-global reference before recreating the
                # source namespace.  We intentionally do not reset torch/CUDA
                # caches or call torch._dynamo.reset().
                old_namespace = candidate_namespace
                # Do not clear the module dictionary: TorchDynamo keeps weak
                # references to its internal builtins/compiler entries and
                # expects their keys to remain valid while cached graphs retire.
                # Release the candidate-owned state explicitly instead.
                for name in (
                    "model", "optimizer", "train_loader", "x", "y", "epoch",
                    "tokenizer", "config", "autocast_ctx", "device",
                ):
                    if name in old_namespace:
                        old_namespace[name] = None
                observer.reset_for_scientific()
                import gc

                gc.collect()
                if seed_originals is not None:
                    seed_originals[0](args.seed)
                    seed_originals[1](args.seed)
                    seed_originals[2](args.seed)
                warmup_lifecycle["warmup_model_discarded"] = True
                warmup_lifecycle["scientific_state_reconstructed"] = True
                observer.event("scientific_state_reset", pid=os.getpid())
                continue
            except CandidateStop:
                stopped = True
                break
            else:
                # Re-executing the body after a normal return would rebuild an
                # untrained model and checkpoint it (the 2026-09 Pair-A runs
                # scored at BPB 3.2159), so an early return is a candidate failure.
                raise CandidateExitedBeforeStop(
                    "candidate training body returned before the protected stop"
                )
    except BaseException as exc:
        payload = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "failure_origin": _failure_origin(exc, candidate),
            "candidate_reported": {
                key: _candidate_value(candidate_namespace, key)
                for key in (
                    "total_training_time", "total_tokens", "optimizer_updates",
                    "microbatches", "step", "TOTAL_BATCH_SIZE", "DEVICE_BATCH_SIZE",
                    "grad_accum_steps", "accumulation_steps",
                )
            },
        }
        pathlib.Path(args.worker_output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return 1
    finally:
        if previous_main is not None:
            sys.modules["__main__"] = previous_main
        if runtime_module is not None and original_contract_progress is not None:
            runtime_module.contract_progress = original_contract_progress
        if runtime_module is not None and original_contract_update_completed is not None:
            runtime_module.contract_update_completed = original_contract_update_completed
        if original_module_call is not None:
            import torch
            torch.nn.Module.__call__ = original_module_call
        if optimized_module_class is not None and original_optimized_module_call is not None:
            optimized_module_class.__call__ = original_optimized_module_call
        if original_optimizer_init is not None:
            import torch
            torch.optim.Optimizer.__init__ = original_optimizer_init
        if str(context) in sys.path:
            sys.path.remove(str(context))
        if seed_originals is not None:
            import torch
            torch.manual_seed, torch.cuda.manual_seed, torch.cuda.manual_seed_all = seed_originals

    if warmup_stopped:
        payload = {"status": "warmup_complete", "observed": observer.summary()}
        pathlib.Path(args.worker_output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return 0
    if not stopped:
        raise RuntimeError("candidate exited without a trusted terminal decision")
    observed = observer.summary()
    checkpoint = pathlib.Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_checkpoint_payload(candidate_namespace, args.seed, candidate), checkpoint)
    payload = {
        "status": "completed",
        "observed": observed,
        "warmup_lifecycle": {
            **warmup_lifecycle,
            "warmup_initial_state_sha256": observer.warmup_initial_state_sha256,
            "scientific_initial_state_sha256": observer.scientific_initial_state_sha256,
            "initial_state_match": (
                observer.warmup_initial_state_sha256 is not None
                and observer.warmup_initial_state_sha256 == observer.scientific_initial_state_sha256
            ),
        },
        "candidate_reported": {
            key: _candidate_value(candidate_namespace, key)
            for key in (
                "total_training_time", "total_tokens", "optimizer_updates",
                "microbatches", "step", "TOTAL_BATCH_SIZE", "DEVICE_BATCH_SIZE",
                "grad_accum_steps", "accumulation_steps", "progress",
            )
        },
        "candidate_validation_attempted": validation_attempted,
        "num_params": _parameter_count(candidate_namespace["model"]),
        "checkpoint_sha256": _sha256(checkpoint),
    }
    pathlib.Path(args.worker_output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def _parameter_count(model: Any) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=pathlib.Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--contract", choices=("T", "N"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--event-fd", type=int, required=True)
    parser.add_argument("--control-fd", type=int, required=True)
    parser.add_argument("--worker-output", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--n-star", type=int)
    parser.add_argument("--baseline-calibration", action="store_true")
    parser.add_argument("--baseline-smoke", action="store_true")
    parser.add_argument("--warmup-only", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    raise SystemExit(execute(build_parser()))
