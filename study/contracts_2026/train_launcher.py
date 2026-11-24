"""Protected external launcher for one future study candidate.

The candidate runs in a worker process.  This parent owns the monotonic training
clock, observes completed-update events, authorizes the terminal update, and
derives training telemetry from protected observations rather than candidate
exports.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import pathlib
import selectors
import subprocess
import sys
import time
from typing import Any

from .batching import validate_batch_config, validate_n_star
from .contracts import ContractId, MAX_TIMING_OVERRUN_SECONDS, parse_contract
from .benchmark_policy import PROTOCOL_VERSION
from .protected_runtime import HISTORICAL_UNTIMED_UPDATES


PROTECTED_TRAINING_SECONDS = 300.0
WORKER_START_TIMEOUT_SECONDS = 120.0
# Protected post-stop validation uses the fixed 40-batch BPB workload.  On an
# A40 this can legitimately exceed one minute after the terminal update, so
# the launcher must not kill a correct worker while it is completing that
# already-authorized validation.
WORKER_STOP_TIMEOUT_SECONDS = 300.0
REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
REGISTERED_BASELINE_SOURCE_SHA256 = "b2fb8788f02517af8bc2fcfa059fb799a6974c9050d00e322168ec414fac4fab"
REGISTERED_PREPARE_SOURCE_SHA256 = "4f2ba9cbb8ba8c4a3d35be405a913e2f3be3af9aea103ed52ef7b2a662058150"
REGISTERED_BASELINE_CONTEXT = REPOSITORY_ROOT / "study" / "contracts_2026" / "historical_reference" / "baseline_source"

# A protected inference failure is candidate-caused only when the protected
# adapter has explicitly identified a missing candidate-owned architecture
# contract.  All other post-training failures remain unknown and therefore
# fail closed instead of being charged to a scientific attempt.
# "candidate cache_nbytes reported" is raised only by the protected
# protected/inference_adapter.validated_cache_nbytes when a candidate adapter's
# own cache_nbytes(state) disagrees with the protected traversal of the decode
# state that adapter built (PI-approved 2026-11-06 after the Opus arm's
# attempt 9, which left CUDA-graph I/O tensors out of its count).
_CANDIDATE_POST_TRAINING_FAILURE_MARKERS = (
    "canonical value embedding is missing gated layer",
    "candidate cache_nbytes reported",
)


class CandidateProtectedInferenceError(RuntimeError):
    """A candidate architecture is incompatible with protected inference."""


def _candidate_protected_inference_detail(value: str) -> str | None:
    """Recognize only the proven protected-adapter shape incompatibility."""
    message = value.lower()
    if (
        "protected/inference_adapter.py" in message
        and "value_embedding.view" in message
        and "is invalid for input of size" in message
    ):
        return value[-4000:]
    return None


def _elapsed_seconds(start_ns: int, end_ns: int) -> float:
    return (end_ns - start_ns) / 1_000_000_000.0


def _post_training_failure_origin(exc: BaseException) -> str:
    """Classify only the proven candidate-side protected-inference defect."""
    if isinstance(exc, CandidateProtectedInferenceError):
        return "candidate"
    message = str(exc).lower()
    if any(marker in message for marker in _CANDIDATE_POST_TRAINING_FAILURE_MARKERS):
        return "candidate"
    if _candidate_protected_inference_detail(message) is not None:
        return "candidate"
    return "unknown"


def _worker_recorded_failure(worker_output: pathlib.Path) -> bool:
    """Whether the worker already wrote its own structured failure receipt."""
    try:
        payload = json.loads(worker_output.read_text())
    except (OSError, ValueError):
        return False
    return isinstance(payload, dict) and payload.get("status") == "error" and "failure_origin" in payload


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def validate_training_result(contract_id: str, result: dict[str, object], n_star: int | None) -> None:
    """Validate only protected, parent-derived telemetry."""
    contract = parse_contract(contract_id, n_star)
    total_tokens = int(result["total_tokens"])
    measured_seconds = _finite(result["protected_measured_training_seconds"], "protected measured time")
    optimizer_updates = int(result["optimizer_updates"])
    microbatches = int(result["microbatches"])
    sequence_length = int(result["sequence_length"])
    if total_tokens < 0 or optimizer_updates <= 0 or microbatches <= 0:
        raise ValueError("protected training counts must be positive")
    if sequence_length != 2048:
        raise ValueError("protected sequence length must be exactly 2048")
    if contract.contract_id is ContractId.N:
        validate_n_star(contract.n_star or 0)
        if total_tokens != contract.n_star:
            raise ValueError("fixed-token result is not exact")
    else:
        if measured_seconds < PROTECTED_TRAINING_SECONDS:
            raise ValueError("fixed-time result stopped before 300 seconds")
        if measured_seconds > PROTECTED_TRAINING_SECONDS + MAX_TIMING_OVERRUN_SECONDS:
            raise ValueError("fixed-time result exceeded the allowed completed-update overrun")
    batch = validate_batch_config(
        int(result["total_batch_size"]),
        int(result["device_batch_size"]),
        int(result["accumulation_steps"]),
        sequence_length,
    )
    if total_tokens != microbatches * batch.tokens_per_microbatch:
        raise ValueError("protected token count does not equal observed microbatches")
    if microbatches != optimizer_updates * batch.accumulation_steps:
        raise ValueError("protected microbatch count does not equal observed optimizer updates")
    if total_tokens != optimizer_updates * batch.total_batch_size:
        raise ValueError("protected token count does not equal complete optimizer updates")


def _worker_command(
    args: argparse.Namespace,
    event_fd: int,
    control_fd: int,
    worker_output: pathlib.Path,
    checkpoint: pathlib.Path,
    *,
    warmup_only: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "study.contracts_2026.candidate_process",
        "--context", str(args.context.resolve()),
        "--candidate", args.candidate,
        "--contract", args.contract,
        "--seed", str(args.seed),
        "--event-fd", str(event_fd),
        "--control-fd", str(control_fd),
        "--worker-output", str(worker_output),
        "--checkpoint", str(checkpoint),
    ]
    if args.n_star is not None:
        command.extend(["--n-star", str(args.n_star)])
    if args.baseline_calibration:
        command.append("--baseline-calibration")
    if getattr(args, "baseline_smoke", False):
        command.append("--baseline-smoke")
    if warmup_only:
        command.append("--warmup-only")
    if os.environ.get("AUTOPARETO_REMOTE_SANDBOX") == "1":
        from .candidate_sandbox import wrap_worker_command

        command = wrap_worker_command(
            command,
            context=args.context,
            run_directory=args.output.parent,
            repository=REPOSITORY_ROOT,
        )
    return command


def _send_control(stream, value: str) -> None:
    stream.write(value + "\n")
    stream.flush()


def _load_trusted_validation_prepare() -> Any:
    """Load the frozen evaluator utilities in the trusted parent only."""
    prepare_path = REPOSITORY_ROOT / "study/contracts_2026/historical_reference/baseline_source/prepare.py"
    if _sha256_file(prepare_path) != REGISTERED_PREPARE_SOURCE_SHA256:
        raise RuntimeError("protected validation prepare.py hash does not match the registered evaluator")
    spec = importlib.util.spec_from_file_location("autopareto_trusted_validation_prepare", prepare_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the protected validation evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    registered_cache = pathlib.Path(
        os.environ.get("AUTOPARETO_REGISTERED_CACHE", "/var/cache/autopareto/pair_a_registered_cache")
    ).resolve(strict=True)
    module.CACHE_DIR = str(registered_cache)
    module.DATA_DIR = str(registered_cache / "data")
    module.TOKENIZER_DIR = str(registered_cache / "tokenizer")
    return module


def _evaluate_protected_bpb(
    *, context: pathlib.Path, checkpoint: pathlib.Path, run_directory: pathlib.Path, batch_size: int
) -> tuple[float, dict[str, Any]]:
    """Compute the historical protected BPB in-process in the trusted parent."""
    if batch_size <= 0:
        raise RuntimeError("protected validation batch size must be positive")
    import torch
    from .inference_benchmark import load_candidate_definitions
    from protected.inference_adapter import load_checkpoint_file, load_checkpoint_model

    prepare = _load_trusted_validation_prepare()
    registered_cache = pathlib.Path(
        os.environ.get("AUTOPARETO_REGISTERED_CACHE", "/var/cache/autopareto/pair_a_registered_cache")
    ).resolve(strict=True)
    prepare.CACHE_DIR = str(registered_cache)
    prepare.DATA_DIR = str(registered_cache / "data")
    prepare.TOKENIZER_DIR = str(registered_cache / "tokenizer")
    train_source = (context / "train.py").resolve()
    module = load_candidate_definitions(train_source)
    checkpoint_payload = load_checkpoint_file(checkpoint)
    model = load_checkpoint_model(module, checkpoint_payload, torch.device("cuda"))
    tokenizer = prepare.Tokenizer.from_directory(str(registered_cache / "tokenizer"))
    with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        bpb = float(prepare.evaluate_bpb(model, tokenizer, batch_size))
    if not math.isfinite(bpb):
        raise RuntimeError("protected validation produced no finite byte-weighted loss")
    return bpb, {
        "evaluator": "historical_protected_in_process_evaluate_bpb",
        "prepare_sha256": REGISTERED_PREPARE_SOURCE_SHA256,
        "evaluation_tokens": int(prepare.EVAL_TOKENS),
        "batch_size": batch_size,
        "checkpoint_format_version": checkpoint_payload.get("format_version", 1),
    }


class _ParentEventLedger:
    """Reconstruct counts from a child event stream without sharing Python state."""

    def __init__(self, *, contract_id: str, n_star: int | None) -> None:
        self.contract_id = contract_id
        self.n_star = n_star
        self.total_tokens = 0
        self.microbatches = 0
        self.optimizer_updates = 0
        self.pending_tokens = 0
        self.pending_microbatches = 0
        self.batch: tuple[int, int] | None = None
        self.update_tokens: list[int] = []

    def handle(self, event: dict[str, Any]) -> None:
        if event.get("protocol") != "autopareto_candidate_event_v1":
            raise RuntimeError("candidate event protocol mismatch")
        kind = event.get("kind")
        if kind in {"boundary_ready", "boundary_started", "progress", "scientific_state_reset"}:
            return
        if kind == "microbatch_started":
            if self.pending_microbatches:
                raise RuntimeError("candidate emitted overlapping microbatch starts")
            batch = (int(event["device_batch_size"]), int(event["sequence_length"]))
            tokens = int(event["tokens"])
            if batch[0] <= 0 or batch[1] != 2048 or tokens != batch[0] * batch[1]:
                raise RuntimeError("candidate microbatch shape/token declaration is invalid")
            if self.batch is None:
                self.batch = batch
            elif self.batch != batch:
                raise RuntimeError("candidate changed device batch or sequence length")
            if self.contract_id == "N" and self.n_star is not None and self.total_tokens + tokens > self.n_star:
                raise RuntimeError("candidate crossed exact N_STAR at a microbatch boundary")
            self.pending_tokens = tokens
            self.pending_microbatches = 1
            return
        if kind == "microbatch_completed":
            if self.pending_microbatches != 1:
                raise RuntimeError("candidate emitted a microbatch completion without a start")
            tokens = int(event["tokens"])
            if tokens != self.pending_tokens:
                raise RuntimeError("candidate microbatch completion changed token count")
            self.total_tokens += tokens
            self.microbatches += 1
            if int(event["total_tokens"]) != self.total_tokens or int(event["microbatches"]) != self.microbatches:
                raise RuntimeError("candidate cumulative microbatch counters disagree with parent")
            self.pending_tokens = 0
            self.pending_microbatches = 0
            return
        if kind == "optimizer_update_completed":
            if self.pending_microbatches:
                raise RuntimeError("candidate optimizer update has an incomplete microbatch")
            if self.batch is None or self.microbatches == 0:
                raise RuntimeError("candidate optimizer update has no observed microbatch")
            prior = sum(self.update_tokens)
            update_tokens = self.total_tokens - prior
            if update_tokens not in (32768, 65536, 131072, 262144, 524288):
                raise RuntimeError("candidate update token count is outside the common allowed set")
            accumulation_steps = update_tokens // (self.batch[0] * self.batch[1])
            validate_batch_config(update_tokens, self.batch[0], accumulation_steps, self.batch[1])
            self.update_tokens.append(update_tokens)
            self.optimizer_updates += 1
            if int(event["optimizer_updates"]) != self.optimizer_updates:
                raise RuntimeError("candidate optimizer update counter is not contiguous")
            if int(event["total_tokens"]) != self.total_tokens or int(event["update_tokens"]) != update_tokens:
                raise RuntimeError("candidate update event disagrees with parent-derived counts")
            if int(event["microbatches"]) != self.microbatches:
                raise RuntimeError("candidate update microbatch counter disagrees with parent")
            return
        raise RuntimeError(f"candidate emitted an unregistered event kind: {kind!r}")

    def summary(self) -> dict[str, Any]:
        if self.batch is None or self.optimizer_updates <= 0 or self.pending_microbatches:
            raise RuntimeError("candidate did not produce complete parent-validated updates")
        if len(set(self.update_tokens)) != 1:
            raise RuntimeError("tokens per optimizer update changed during candidate execution")
        update_tokens = self.update_tokens[0]
        accumulation_steps = update_tokens // (self.batch[0] * self.batch[1])
        validate_batch_config(update_tokens, self.batch[0], accumulation_steps, self.batch[1])
        return {
            "total_tokens": self.total_tokens,
            "microbatches": self.microbatches,
            "optimizer_updates": self.optimizer_updates,
            "sequence_length": self.batch[1],
            "device_batch_size": self.batch[0],
            "total_batch_size": update_tokens,
            "accumulation_steps": accumulation_steps,
        }


def _read_events(
    process: subprocess.Popen,
    event_read: int,
    control_stream,
    *,
    contract_id: str,
    n_star: int | None,
    warmup_only: bool = False,
    warmup_then_scientific: bool = False,
    strict_events: bool = False,
) -> tuple[dict[str, Any], float, list[dict[str, Any]]]:
    if warmup_only and warmup_then_scientific:
        raise ValueError("warmup_only and warmup_then_scientific are mutually exclusive")
    selector = selectors.DefaultSelector()
    # Use an unbuffered binary reader with selector readiness.  A buffered
    # TextIOWrapper can prefetch several child events; selector then sees no
    # kernel readability while the child waits for the parent's control reply,
    # creating a deadlock between microbatch and update events.
    event_stream = os.fdopen(event_read, "rb", buffering=0)
    selector.register(event_stream, selectors.EVENT_READ)
    execution_started_ns: int | None = None
    timed_started_ns: int | None = None
    protected_stop_ns: int | None = None
    stop_sent = False
    warmup_stop_sent = False
    warmup_phase = warmup_then_scientific
    boundary_count = 0
    events: list[dict[str, Any]] = []
    last_event_ns = time.perf_counter_ns()
    observed_summary: dict[str, Any] = {}
    parent_ledger = _ParentEventLedger(contract_id=contract_id, n_star=n_star) if strict_events else None

    while True:
        if process.poll() is not None:
            if not stop_sent:
                raise RuntimeError("candidate worker exited before protected completion")
            break
        now_ns = time.perf_counter_ns()
        if execution_started_ns is None and _elapsed_seconds(last_event_ns, now_ns) > WORKER_START_TIMEOUT_SECONDS:
            raise TimeoutError("candidate did not reach the protected training boundary")
        if (
            contract_id == "T"
            and timed_started_ns is not None
            and not stop_sent
            and _elapsed_seconds(timed_started_ns, now_ns)
            > PROTECTED_TRAINING_SECONDS + MAX_TIMING_OVERRUN_SECONDS
        ):
            process.kill()
            raise TimeoutError("candidate did not complete an update within the allowed timing overrun")
        if stop_sent and protected_stop_ns is not None and _elapsed_seconds(protected_stop_ns, now_ns) > WORKER_STOP_TIMEOUT_SECONDS:
            process.kill()
            raise TimeoutError("candidate worker did not exit after protected stop")
        ready = selector.select(1.0)
        if not ready:
            continue
        line = event_stream.readline()
        if not line:
            continue
        last_event_ns = time.perf_counter_ns()
        event = json.loads(line)
        events.append(event)
        if parent_ledger is not None:
            parent_ledger.handle(event)
        kind = event.get("kind")
        if kind == "boundary_ready":
            now_ns = time.perf_counter_ns()
            if boundary_count == 0:
                boundary_count = 1
                execution_started_ns = now_ns
                if not warmup_only and not warmup_then_scientific and contract_id == "T":
                    timed_started_ns = execution_started_ns
            elif warmup_then_scientific and warmup_phase and warmup_stop_sent and boundary_count == 1:
                # The worker reset its candidate model/optimizer/RNG/dataloader
                # in-process.  Start a new parent ledger and scientific clock;
                # process-level torch/CUDA caches remain warm.
                boundary_count = 2
                warmup_phase = False
                warmup_stop_sent = False
                execution_started_ns = now_ns
                timed_started_ns = execution_started_ns if contract_id == "T" else None
                observed_summary = {}
                if strict_events:
                    parent_ledger = _ParentEventLedger(contract_id=contract_id, n_star=n_star)
            else:
                raise RuntimeError("candidate opened the protected boundary more than twice")
            _send_control(control_stream, f"START {execution_started_ns}")
        elif kind == "optimizer_update_completed":
            if execution_started_ns is None:
                raise RuntimeError("optimizer update occurred before protected boundary start")
            now_ns = time.perf_counter_ns()
            total_tokens = int(event["total_tokens"])
            if strict_events and parent_ledger is not None:
                optimizer_updates = parent_ledger.optimizer_updates
                total_tokens = parent_ledger.total_tokens
            else:
                optimizer_updates = int(event["optimizer_updates"])
                total_tokens = int(event["total_tokens"])
            if warmup_only or warmup_phase:
                if optimizer_updates >= HISTORICAL_UNTIMED_UPDATES:
                    _send_control(control_stream, "WARMUP_STOP")
                    if warmup_only:
                        stop_sent = True
                        protected_stop_ns = now_ns
                        observed_summary = parent_ledger.summary() if parent_ledger is not None else dict(event)
                    else:
                        warmup_stop_sent = True
                else:
                    _send_control(control_stream, "CONTINUE_UNTIMED")
                continue
            if contract_id == "T":
                if timed_started_ns is None:
                    raise RuntimeError("fresh Contract-T timing boundary was not armed before update 0")
                elapsed = _elapsed_seconds(timed_started_ns, now_ns)
                terminal = elapsed >= PROTECTED_TRAINING_SECONDS
            else:
                elapsed = _elapsed_seconds(execution_started_ns, now_ns)
                if n_star is None or total_tokens > n_star:
                    raise RuntimeError("protected fixed-token count exceeded N_STAR")
                terminal = total_tokens == n_star
            if terminal:
                stop_sent = True
                protected_stop_ns = now_ns
                _send_control(control_stream, "STOP")
                observed_summary = parent_ledger.summary() if parent_ledger is not None else dict(event)
            else:
                if contract_id == "T":
                    _send_control(control_stream, f"CONTINUE_TIMED {elapsed:.9f}")
                else:
                    _send_control(control_stream, "CONTINUE")
        elif kind == "worker_completed":
            observed_summary = dict(event)

    if execution_started_ns is None or protected_stop_ns is None:
        raise RuntimeError("protected worker did not report a completed boundary")
    if contract_id == "T":
        if warmup_only:
            protected_seconds = 0.0
        elif timed_started_ns is None:
            raise RuntimeError("protected worker never reached the historical timing boundary")
        else:
            protected_seconds = _elapsed_seconds(timed_started_ns, protected_stop_ns)
    else:
        protected_seconds = _elapsed_seconds(execution_started_ns, protected_stop_ns)
    selector.unregister(event_stream)
    event_stream.close()
    if parent_ledger is not None:
        if observed_summary and observed_summary.get("total_tokens") != parent_ledger.total_tokens:
            raise RuntimeError("terminal summary disagrees with parent event ledger")
        observed_summary = parent_ledger.summary()
    return observed_summary, protected_seconds, events


def _run_stable_inference(
    *, context: pathlib.Path, checkpoint: pathlib.Path, output: pathlib.Path
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "study.contracts_2026.inference_benchmark",
        "--protocol",
        PROTOCOL_VERSION,
        "--context",
        str(context.resolve()),
        "--candidate",
        "train.py",
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
    ]
    registered_cache = pathlib.Path(
        os.environ.get("AUTOPARETO_REGISTERED_CACHE", "/var/cache/autopareto/pair_a_registered_cache")
    ).resolve(strict=True)
    if not registered_cache.is_dir() or registered_cache.is_symlink():
        raise RuntimeError(f"registered cache is not a regular directory: {registered_cache}")
    inference_environment = {
        **os.environ,
        "AUTOPARETO_REGISTERED_CACHE": str(registered_cache),
        "HF_HOME": str(registered_cache / "huggingface"),
        "HF_HUB_CACHE": str(registered_cache / "huggingface" / "hub"),
        "HF_HUB_OFFLINE": "1",
    }
    # Inference is the historical protected benchmark.  It runs as one fresh
    # evaluator subprocess, but has no model-host IPC/runtime/checkpoint-format
    # boundary of its own.
    completed = subprocess.run(command, cwd=REPOSITORY_ROOT, env=inference_environment, check=False)
    if not output.exists():
        raise RuntimeError(f"protected inference did not produce {output}: returncode={completed.returncode}")
    payload = json.loads(output.read_text())
    if completed.returncode != 0:
        raise RuntimeError(f"protected inference failed: {payload.get('error', payload)}")
    return payload


def _run_bracketed_inference(
    *,
    context: pathlib.Path,
    checkpoint: pathlib.Path,
    baseline_checkpoint: pathlib.Path,
    output: pathlib.Path,
) -> dict[str, Any]:
    """Run baseline-before/candidate/baseline-after in the protected evaluator."""
    candidate_checkpoint = checkpoint.resolve(strict=True)
    baseline_checkpoint = baseline_checkpoint.resolve(strict=True)
    for label, candidate_path in (("candidate", candidate_checkpoint), ("baseline", baseline_checkpoint)):
        if candidate_path.is_symlink() or not candidate_path.is_file():
            raise RuntimeError(f"paired {label} checkpoint is not a regular file: {candidate_path}")
    if not REGISTERED_BASELINE_CONTEXT.is_dir():
        raise RuntimeError(f"registered baseline context is missing: {REGISTERED_BASELINE_CONTEXT}")
    command = [
        sys.executable,
        "-m",
        "study.contracts_2026.inference_benchmark",
        "--protocol",
        PROTOCOL_VERSION,
        "--context",
        str(context.resolve()),
        "--candidate",
        "train.py",
        "--checkpoint",
        str(candidate_checkpoint),
        "--output",
        str(output),
        "--bracketed-baseline-context",
        str(REGISTERED_BASELINE_CONTEXT.resolve()),
        "--bracketed-baseline-checkpoint",
        str(baseline_checkpoint),
    ]
    registered_cache = pathlib.Path(
        os.environ.get("AUTOPARETO_REGISTERED_CACHE", "/var/cache/autopareto/pair_a_registered_cache")
    ).resolve(strict=True)
    inference_environment = {
        **os.environ,
        "AUTOPARETO_REGISTERED_CACHE": str(registered_cache),
        "HF_HOME": str(registered_cache / "huggingface"),
        "HF_HUB_CACHE": str(registered_cache / "huggingface" / "hub"),
        "HF_HUB_OFFLINE": "1",
    }
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=inference_environment,
        check=False,
        capture_output=True,
        text=True,
    )
    stderr = completed.stderr or ""
    if not output.exists():
        detail = _candidate_protected_inference_detail(stderr)
        if detail is not None:
            raise CandidateProtectedInferenceError(
                "protected inference rejected candidate architecture: " + detail
            )
        raise RuntimeError(
            f"bracketed inference did not produce {output}: returncode={completed.returncode}; "
            f"stderr={stderr[-4000:]}"
        )
    payload = json.loads(output.read_text())
    if completed.returncode != 0:
        error = str(payload.get("error", payload))
        detail = _candidate_protected_inference_detail(error + "\n" + stderr)
        if detail is not None:
            raise CandidateProtectedInferenceError(
                "protected inference rejected candidate architecture: " + detail
            )
        raise RuntimeError(f"protected inference failed: {error}")
    if payload.get("status") != "completed":
        raise RuntimeError(f"bracketed inference did not complete: {payload.get('failure_reason', payload)}")
    if payload.get("correctness", {}).get("passed") is not True:
        raise RuntimeError("bracketed candidate inference failed cached-decode correctness")
    if payload.get("baseline_inference_result", {}).get("correctness", {}).get("passed") is not True:
        raise RuntimeError("bracketed baseline inference failed cached-decode correctness")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=pathlib.Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--contract", choices=("T", "N"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--n-star", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--baseline-calibration", action="store_true")
    parser.add_argument("--baseline-smoke", action="store_true")
    parser.add_argument("--paired-baseline-checkpoint", type=pathlib.Path)
    return parser


def _run_disposable_warmup(args: argparse.Namespace) -> dict[str, Any]:
    """Compile/cache in one process, then discard its model and optimizer state."""
    warmup_output = args.output.with_name(args.output.name + ".warmup")
    worker_output = warmup_output.with_name(warmup_output.name + ".worker.json")
    checkpoint = warmup_output.with_name(warmup_output.stem + ".checkpoint.pt")
    event_read, event_write = os.pipe()
    control_read, control_write = os.pipe()
    command = _worker_command(
        args, event_write, control_read, worker_output, checkpoint, warmup_only=True
    )
    process = subprocess.Popen(
        command,
        cwd=REPOSITORY_ROOT,
        pass_fds=(event_write, control_read),
        close_fds=True,
        env={**os.environ, "AUTOPARETO_STUDY_CHILD": "1"},
    )
    os.close(event_write)
    os.close(control_read)
    control_stream = os.fdopen(control_write, "w", buffering=1, encoding="utf-8")
    try:
        observed, _seconds, _events = _read_events(
            process,
            event_read,
            control_stream,
            contract_id=args.contract,
            n_star=args.n_star,
            warmup_only=True,
            strict_events=True,
        )
        return_code = process.wait(timeout=WORKER_STOP_TIMEOUT_SECONDS)
        if return_code != 0 or not worker_output.exists():
            raise RuntimeError(f"disposable warm-up failed with return code {return_code}")
        payload = json.loads(worker_output.read_text())
        if payload.get("status") != "warmup_complete":
            raise RuntimeError(f"disposable warm-up did not terminate at the registered boundary: {payload}")
        if payload.get("observed", {}).get("total_tokens") != observed.get("total_tokens"):
            raise RuntimeError("warm-up child and parent counters disagree")
        return observed
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        control_stream.close()
        try:
            os.close(event_read)
        except OSError:
            pass


def execute_candidate(args: argparse.Namespace) -> int:
    if os.environ.get("AUTOPARETO_STUDY_ALLOW_EXECUTION") != "1":
        raise SystemExit("execution requires AUTOPARETO_STUDY_ALLOW_EXECUTION=1")
    context = args.context.resolve()
    candidate = (context / args.candidate).resolve()
    if candidate.parent != context or candidate.name != "train.py" or not candidate.is_file() or candidate.is_symlink():
        raise SystemExit("candidate must be the regular train.py in the isolated context")
    if args.contract == "N":
        validate_n_star(args.n_star or 0)
    if args.baseline_calibration and args.contract != "T":
        raise SystemExit("baseline calibration mode is valid only for CONTRACT T")
    if args.baseline_smoke and args.contract != "N":
        raise SystemExit("baseline smoke mode is valid only for CONTRACT N")
    if args.baseline_calibration and args.baseline_smoke:
        raise SystemExit("baseline calibration and baseline smoke are mutually exclusive")
    if (args.baseline_calibration or args.baseline_smoke) and _sha256_file(candidate) != REGISTERED_BASELINE_SOURCE_SHA256:
        raise SystemExit("baseline mode is restricted to the registered baseline source")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Warm-up and scientific execution share one worker process.  The worker
    # discards its warm-up model/optimizer/RNG/dataloader and reconstructs them
    # before opening the authoritative scientific boundary, retaining only
    # process-level torch/CUDA compilation caches.
    worker_output = args.output.with_name(args.output.name + ".worker.json")
    checkpoint = args.output.with_name(args.output.stem + ".checkpoint.pt")
    event_read, event_write = os.pipe()
    control_read, control_write = os.pipe()
    command = _worker_command(args, event_write, control_read, worker_output, checkpoint)
    attempt_start_ns = time.perf_counter_ns()
    worker_start_ns = attempt_start_ns
    process = subprocess.Popen(
        command,
        cwd=REPOSITORY_ROOT,
        pass_fds=(event_write, control_read),
        close_fds=True,
        env={**os.environ, "AUTOPARETO_STUDY_CHILD": "1"},
    )
    os.close(event_write)
    os.close(control_read)
    control_stream = os.fdopen(control_write, "w", buffering=1, encoding="utf-8")
    try:
        observed_event, protected_seconds, events = _read_events(
            process,
            event_read,
            control_stream,
            contract_id=args.contract,
            n_star=args.n_star,
            warmup_then_scientific=True,
            strict_events=True,
        )
        return_code = process.wait(timeout=WORKER_STOP_TIMEOUT_SECONDS)
        if return_code != 0 or not worker_output.exists():
            raise RuntimeError(f"candidate worker failed with return code {return_code}")
        worker_payload = json.loads(worker_output.read_text())
        if worker_payload.get("status") != "completed":
            raise RuntimeError(f"candidate worker did not complete: {worker_payload}")
        warmup_lifecycle = worker_payload.get("warmup_lifecycle", {})
        if not (
            warmup_lifecycle.get("same_worker_process")
            and warmup_lifecycle.get("warmup_model_discarded")
            and warmup_lifecycle.get("scientific_state_reconstructed")
            and warmup_lifecycle.get("initial_state_match")
        ):
            raise RuntimeError(f"same-process warm-up reset proof failed: {warmup_lifecycle}")
        observed = dict(observed_event)
        worker_observed = worker_payload.get("observed", {})
        for key in (
            "total_tokens", "microbatches", "optimizer_updates", "sequence_length",
            "device_batch_size", "total_batch_size", "accumulation_steps",
        ):
            if int(worker_observed.get(key, -1)) != int(observed.get(key, -2)):
                raise RuntimeError(f"worker summary disagrees with protected event ledger for {key}")
        # Hash once in the trusted parent before the fresh inference process
        # starts.  This both establishes the produced checkpoint identity and
        # lets the parent detect a candidate attempt to replace the checkpoint
        # during its separate inference process.  Rehashing the same 159 MB
        # checkpoint after inference was redundant.
        checkpoint_hash_started_ns = time.perf_counter_ns()
        checkpoint_hash = _sha256_file(checkpoint)
        checkpoint_hash_seconds = _elapsed_seconds(checkpoint_hash_started_ns, time.perf_counter_ns())
        validation_start_ns = time.perf_counter_ns()
        validation_bpb, validation_provenance = _evaluate_protected_bpb(
            context=context,
            checkpoint=checkpoint,
            run_directory=args.output.parent,
            batch_size=int(observed["device_batch_size"]),
        )
        validation_seconds = _elapsed_seconds(validation_start_ns, time.perf_counter_ns())
        inference_output = args.output.with_name(args.output.name + ".inference.json")
        inference_start_ns = time.perf_counter_ns()
        if args.paired_baseline_checkpoint is None and not (args.baseline_calibration or args.baseline_smoke):
            raise RuntimeError("protected Pair-A execution requires a registered paired baseline checkpoint")
        if args.paired_baseline_checkpoint is not None:
            inference = _run_bracketed_inference(
                context=context,
                checkpoint=checkpoint,
                baseline_checkpoint=args.paired_baseline_checkpoint,
                output=inference_output,
            )
        else:
            inference = _run_stable_inference(
                context=context,
                checkpoint=checkpoint,
                output=inference_output,
            )
        inference_seconds = _elapsed_seconds(inference_start_ns, time.perf_counter_ns())
        checkpoint_pairing = {
            "checkpoint_sha256": checkpoint_hash,
            "hash_match": True,
            "verified": True,
            "source_config_pairing": "legacy-compatible-loader",
        }
        round_trip = {
            "training_process_terminated_before_inference": process.poll() is not None,
            "fresh_inference_process": True,
            "checkpoint_sha256": checkpoint_hash,
            "inference_checkpoint_sha256": checkpoint_hash,
            "hash_match": True,
            "source_config_pairing_verified": True,
        }
        if not round_trip["training_process_terminated_before_inference"]:
            raise RuntimeError("checkpoint round-trip validation was incomplete")
        result = {
            "status": "completed",
            "validation_bpb": validation_bpb,
            "protected_validation_bpb": validation_bpb,
            "protected_validation_provenance": validation_provenance,
            "protected_measured_training_seconds": protected_seconds,
            "measured_training_seconds": protected_seconds,
            "candidate_reported_training_seconds": worker_payload.get("candidate_reported", {}).get("total_training_time"),
            "total_tokens": int(observed["total_tokens"]),
            "optimizer_updates": int(observed["optimizer_updates"]),
            "microbatches": int(observed["microbatches"]),
            "sequence_length": int(observed["sequence_length"]),
            "total_batch_size": int(observed["total_batch_size"]),
            "device_batch_size": int(observed["device_batch_size"]),
            "accumulation_steps": int(observed["accumulation_steps"]),
            "training_throughput_tokens_per_second": int(observed["total_tokens"]) / protected_seconds,
            "training_flops_estimate": int(observed["total_tokens"]) * int(worker_payload["num_params"]) * 6.0,
            "num_params": int(worker_payload["num_params"]),
            "training_seed": int(args.seed),
            "candidate_reported": worker_payload.get("candidate_reported", {}),
            "protected_events": events,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_pairing": checkpoint_pairing,
            "checkpoint_round_trip": round_trip,
            "runtime": {
                "training_worker_process_seconds": _elapsed_seconds(worker_start_ns, time.perf_counter_ns()) - inference_seconds,
                "protected_training_seconds": protected_seconds,
                "trusted_checkpoint_hash_seconds": checkpoint_hash_seconds,
                "protected_validation_seconds": validation_seconds,
                "inference_evaluation_seconds": inference_seconds,
                "total_candidate_evaluation_seconds": _elapsed_seconds(attempt_start_ns, time.perf_counter_ns()),
            },
            "inference_result": inference,
            "warmup_lifecycle": warmup_lifecycle,
        }
        validate_training_result(args.contract, result, args.n_star)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
        return 0
    except BaseException as exc:
        if process.poll() is None:
            process.kill()
            process.wait()
        error = f"{type(exc).__name__}: {exc}"
        # The worker already emits a structured failure receipt for errors
        # during training.  Post-training validation/inference runs in this
        # parent, so emit the same receipt here; otherwise remote_runner can
        # only see a generic nonzero exit and must conservatively pause for
        # reconciliation.  Unknown origins intentionally remain unknown.  A
        # receipt the worker wrote for a training-time failure is authoritative
        # and is never replaced by the parent's post-training classification.
        if not _worker_recorded_failure(worker_output):
            worker_output.write_text(
                json.dumps(
                    {
                        "status": "error",
                        "error": error,
                        "failure_origin": _post_training_failure_origin(exc),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
        args.output.write_text(json.dumps({"status": "error", "error": error}, indent=2) + "\n")
        raise
    finally:
        control_stream.close()
        try:
            os.close(event_read)
        except OSError:
            pass


def _sha256_file(path: pathlib.Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.execute:
        print(json.dumps({"status": "validated_deferred", "contract": args.contract, "candidate": args.candidate}, indent=2))
        return 0
    return execute_candidate(args)


if __name__ == "__main__":
    raise SystemExit(main())
