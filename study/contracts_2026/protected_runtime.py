"""Protected runtime primitives used by the study candidate worker.

The candidate process may emit observations, but it does not supply the values
used for contract validation.  The parent launcher owns the authoritative
monotonic clock and tells the worker when a completed update is the terminal
update.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from typing import Any, Callable, TextIO

from .batching import ALLOWED_TOTAL_BATCH_SIZES, validate_batch_config


PROTECTED_RUNTIME_PROTOCOL = "autopareto_protected_runtime_v3"
SEQUENCE_LENGTH = 2048
HISTORICAL_UNTIMED_UPDATES = 11


class ProtectedInstrumentationError(RuntimeError):
    """The candidate did not satisfy an observable protected boundary."""


class ProtectedStop(RuntimeError):
    """Raised only after the protected parent approves a completed update."""


@dataclass(frozen=True)
class ObservedBatch:
    device_batch_size: int
    sequence_length: int
    tokens: int


class AuthoritativeObserver:
    """Observe actual tensor forwards and optimizer calls in the worker.

    The observer deliberately has no public result counters for candidates to
    export.  Its summary is consumed by the protected worker and launcher.
    """

    def __init__(
        self,
        *,
        contract_id: str,
        n_star: int | None,
        event_stream: TextIO,
        control_stream: TextIO,
        send_event: Callable[[dict[str, Any]], None] | None = None,
        require_progress: bool = True,
    ) -> None:
        self.contract_id = contract_id
        self.n_star = n_star
        self.event_stream = event_stream
        self.control_stream = control_stream
        self._send_event = send_event or self._write_event
        self.require_progress = require_progress
        self.boundary_started = False
        self.boundary_closed = False
        self.stop_requested = False
        self.total_tokens = 0
        self.microbatches = 0
        self.optimizer_updates = 0
        self._microbatches_since_update = 0
        self._tokens_since_update = 0
        self._batch: ObservedBatch | None = None
        self._update_batches: list[int] = []
        self._progress_calls_since_update = 0
        self.progress_values: list[float] = []
        self.worker_start_perf_counter_ns = time.perf_counter_ns()
        self.timing_boundary_started = contract_id != "T"
        self.completed_timing_seconds = 0.0

    def _write_event(self, payload: dict[str, Any]) -> None:
        self.event_stream.write(json.dumps(payload, sort_keys=True) + "\n")
        self.event_stream.flush()

    def event(self, kind: str, **payload: Any) -> None:
        self._send_event(
            {
                "protocol": PROTECTED_RUNTIME_PROTOCOL,
                "kind": kind,
                "monotonic_ns": time.perf_counter_ns(),
                **payload,
            }
        )

    def before_first_training_forward(self) -> None:
        if self.boundary_started:
            return
        self.event("boundary_ready")
        command = self.control_stream.readline().strip()
        if not command.startswith("START "):
            raise ProtectedInstrumentationError("parent did not open the protected training boundary")
        self.boundary_started = True
        self.worker_start_perf_counter_ns = time.perf_counter_ns()
        self.event("boundary_started")

    def observe_microbatch_start(self, input_ids: Any, targets: Any) -> ObservedBatch:
        if self.boundary_closed:
            raise ProtectedStop("protected training boundary is closed")
        if not self.boundary_started:
            self.before_first_training_forward()
        shape = getattr(input_ids, "shape", None)
        target_shape = getattr(targets, "shape", None)
        ndim = getattr(input_ids, "ndim", None)
        if shape is None or target_shape is None or ndim != 2 or tuple(shape) != tuple(target_shape):
            raise ProtectedInstrumentationError("training forwards must receive matching [B, 2048] tensors")
        batch_size, sequence_length = (int(shape[0]), int(shape[1]))
        if batch_size <= 0 or sequence_length != SEQUENCE_LENGTH:
            raise ProtectedInstrumentationError("protected training sequence length must be exactly 2048")
        tokens = batch_size * sequence_length
        if self.contract_id == "N" and self.n_star is not None and self.total_tokens + tokens > self.n_star:
            raise ProtectedInstrumentationError("fixed-token candidate attempted a partial final microbatch")
        observed = ObservedBatch(batch_size, sequence_length, tokens)
        if self._batch is None:
            self._batch = observed
        elif self._batch != observed:
            raise ProtectedInstrumentationError("device batch or sequence length changed during training")
        self.event(
            "microbatch_started",
            device_batch_size=batch_size,
            sequence_length=sequence_length,
            tokens=tokens,
        )
        return observed

    def observe_microbatch_completed(self, observed: ObservedBatch) -> None:
        self.total_tokens += observed.tokens
        self.microbatches += 1
        self._microbatches_since_update += 1
        self._tokens_since_update += observed.tokens
        self.event(
            "microbatch_completed",
            device_batch_size=observed.device_batch_size,
            sequence_length=observed.sequence_length,
            tokens=observed.tokens,
            total_tokens=self.total_tokens,
            microbatches=self.microbatches,
        )

    def note_progress(self, value: float) -> float:
        if self.contract_id == "T":
            # Match the registered loop: schedules see only durations from
            # completed timed updates.  Updates 0..10 remain real updates but
            # contribute no Contract-T progress.
            value = min(max(self.completed_timing_seconds, 0.0) / 300.0, 1.0)
        elif self.contract_id == "N" and self.n_star:
            value = min(self.total_tokens / self.n_star, 1.0)
        else:
            raise ProtectedInstrumentationError("invalid protected contract progress request")
        self._progress_calls_since_update += 1
        self.progress_values.append(value)
        self.event("progress", contract=self.contract_id, value=value, total_tokens=self.total_tokens)
        return value

    def before_optimizer_update(self) -> None:
        if self.boundary_closed:
            raise ProtectedStop("protected training boundary is closed")
        if self._microbatches_since_update <= 0:
            raise ProtectedInstrumentationError("optimizer update had no observed training microbatch")
        if self.require_progress and self._progress_calls_since_update <= 0:
            raise ProtectedInstrumentationError("candidate did not call protected contract_progress before update")

    def after_optimizer_update(self) -> None:
        if self._batch is None:
            raise ProtectedInstrumentationError("optimizer update has no observed batch")
        if self._tokens_since_update not in ALLOWED_TOTAL_BATCH_SIZES:
            raise ProtectedInstrumentationError("observed tokens per update are outside the common allowed set")
        accumulation_steps = self._microbatches_since_update
        validate_batch_config(
            self._tokens_since_update,
            self._batch.device_batch_size,
            accumulation_steps,
            self._batch.sequence_length,
        )
        self.optimizer_updates += 1
        self._update_batches.append(self._tokens_since_update)
        self.event(
            "optimizer_update_completed",
            optimizer_updates=self.optimizer_updates,
            microbatches=self.microbatches,
            update_tokens=self._tokens_since_update,
            total_tokens=self.total_tokens,
            device_batch_size=self._batch.device_batch_size,
            sequence_length=self._batch.sequence_length,
            accumulation_steps=accumulation_steps,
        )
        self._microbatches_since_update = 0
        self._tokens_since_update = 0
        self._progress_calls_since_update = 0
        decision = self.control_stream.readline().strip()
        if decision == "STOP":
            self.stop_requested = True
            self.boundary_closed = True
            raise ProtectedStop("parent closed the protected training boundary")
        if decision == "CONTINUE_UNTIMED":
            if self.contract_id != "T" or self.optimizer_updates >= HISTORICAL_UNTIMED_UPDATES:
                raise ProtectedInstrumentationError("parent returned an invalid untimed update decision")
            return
        if decision.startswith("CONTINUE_TIMED "):
            if self.contract_id != "T" or self.optimizer_updates < HISTORICAL_UNTIMED_UPDATES:
                raise ProtectedInstrumentationError("parent returned an invalid timed update decision")
            try:
                completed_seconds = float(decision.split(maxsplit=1)[1])
            except (IndexError, ValueError) as exc:
                raise ProtectedInstrumentationError("parent returned invalid completed timing") from exc
            if completed_seconds < self.completed_timing_seconds:
                raise ProtectedInstrumentationError("parent timing moved backwards")
            first_timed_boundary = not self.timing_boundary_started
            self.timing_boundary_started = True
            self.completed_timing_seconds = completed_seconds
            if first_timed_boundary:
                self.event(
                    "timing_boundary_started",
                    excluded_optimizer_updates=HISTORICAL_UNTIMED_UPDATES,
                    first_timed_zero_based_step_index=HISTORICAL_UNTIMED_UPDATES,
                )
            return
        if decision != "CONTINUE":
            raise ProtectedInstrumentationError("parent returned an invalid update decision")

    def elapsed_worker_seconds(self) -> float:
        if not self.boundary_started:
            return 0.0
        return max(
            0.0,
            (time.perf_counter_ns() - self.worker_start_perf_counter_ns) / 1_000_000_000.0,
        )

    def summary(self) -> dict[str, Any]:
        if self._batch is None:
            raise ProtectedInstrumentationError("no training microbatches were observed")
        if self._microbatches_since_update:
            raise ProtectedInstrumentationError("training ended with an incomplete optimizer update")
        if len(set(self._update_batches)) != 1:
            raise ProtectedInstrumentationError("tokens per optimizer update changed during training")
        batch = validate_batch_config(
            self._update_batches[0],
            self._batch.device_batch_size,
            self.microbatches // self.optimizer_updates,
            self._batch.sequence_length,
        )
        return {
            "total_tokens": self.total_tokens,
            "microbatches": self.microbatches,
            "optimizer_updates": self.optimizer_updates,
            "sequence_length": batch.sequence_length,
            "total_batch_size": batch.total_batch_size,
            "device_batch_size": batch.device_batch_size,
            "accumulation_steps": batch.accumulation_steps,
            "worker_observed_training_seconds": self.elapsed_worker_seconds(),
            "progress_values": list(self.progress_values),
        }


def open_event_stream(fd: int) -> TextIO:
    return os.fdopen(fd, "w", buffering=1, encoding="utf-8")


def open_control_stream(fd: int) -> TextIO:
    return os.fdopen(fd, "r", buffering=1, encoding="utf-8")
