"""Contract runtime primitives used by protected future training launchers."""

from __future__ import annotations

from dataclasses import dataclass
import time

from .batching import BatchConfig, accounted_tokens, validate_batch_config, validate_n_star
from .contracts import ContractId, ResourceContract


@dataclass
class TrainingLedger:
    """Count only completed microbatches and optimizer updates."""

    batch: BatchConfig
    contract: ResourceContract
    total_tokens: int = 0
    optimizer_updates: int = 0
    microbatches: int = 0

    def __post_init__(self) -> None:
        self.batch.validate()
        if self.contract.contract_id is ContractId.N:
            validate_n_star(self.contract.n_star or 0)

    def before_microbatch(self) -> None:
        if self.contract.contract_id is ContractId.N:
            target = self.contract.n_star or 0
            if self.total_tokens + self.batch.tokens_per_microbatch > target:
                raise RuntimeError("a partial fixed-token microbatch is forbidden")

    def after_microbatch(self) -> None:
        self.microbatches += 1
        self.total_tokens += self.batch.tokens_per_microbatch

    def after_optimizer_update(self) -> None:
        expected_tokens = self.batch.tokens_per_optimizer_update
        if self.total_tokens != (self.optimizer_updates + 1) * expected_tokens:
            raise RuntimeError("optimizer update was recorded without its complete microbatches")
        self.optimizer_updates += 1

    @property
    def progress(self) -> float:
        return self.contract.progress(measured_training_time=0.0, total_tokens=self.total_tokens)

    def finish(self, measured_training_seconds: float) -> dict[str, int | float]:
        self.contract.validate_completion(
            measured_training_time=measured_training_seconds,
            total_tokens=self.total_tokens,
        )
        return {
            "total_tokens": self.total_tokens,
            "optimizer_updates": self.optimizer_updates,
            "microbatches": self.microbatches,
            "total_batch_size": self.batch.total_batch_size,
            "device_batch_size": self.batch.device_batch_size,
            "accumulation_steps": self.batch.accumulation_steps,
            "progress": self.contract.progress(
                measured_training_time=measured_training_seconds,
                total_tokens=self.total_tokens,
            ),
        }


class MeasuredTrainingClock:
    """The protected clock uses a monotonic timer and excludes post-loop work."""

    def __init__(self, budget_seconds: float = 300.0):
        if budget_seconds != 300.0:
            raise ValueError("this instrument uses exactly 300.000 seconds")
        self.budget_seconds = budget_seconds
        self._started: float | None = None
        self._stopped: float | None = None

    def start(self) -> None:
        if self._started is not None:
            raise RuntimeError("training clock already started")
        self._started = time.perf_counter()

    def elapsed(self) -> float:
        if self._started is None:
            raise RuntimeError("training clock has not started")
        return (self._stopped or time.perf_counter()) - self._started

    def should_stop_after_completed_update(self) -> bool:
        return self.elapsed() >= self.budget_seconds

    def stop(self) -> float:
        if self._started is None:
            raise RuntimeError("training clock has not started")
        if self._stopped is None:
            self._stopped = time.perf_counter()
        return self.elapsed()


def estimate_training_flops(total_tokens: int, num_params: int, multiplier: float = 6.0) -> float:
    if total_tokens < 0 or num_params < 0 or multiplier <= 0:
        raise ValueError("FLOP inputs must be non-negative with a positive multiplier")
    return float(total_tokens) * float(num_params) * multiplier
