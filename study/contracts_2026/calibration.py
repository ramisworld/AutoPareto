"""Deterministic N_STAR calibration math and deferred execution plan."""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics

from .batching import ALLOWED_TOTAL_BATCH_SIZES, COMMON_TOKEN_QUANTUM, validate_n_star


@dataclass(frozen=True)
class CalibrationRun:
    run_index: int
    measured_training_seconds: float
    total_tokens: int
    optimizer_updates: int
    training_throughput_tokens_per_second: float | None
    validation_bpb: float | None
    error: str | None = None

    def validate(self) -> None:
        if self.run_index not in (1, 2, 3):
            raise ValueError("calibration runs are numbered 1, 2, 3")
        if self.total_tokens < 0 or self.optimizer_updates < 0:
            raise ValueError("calibration counts cannot be negative")
        if self.error is None and self.total_tokens == 0:
            raise ValueError("a successful calibration run must process tokens")


@dataclass(frozen=True)
class CalibrationChoice:
    median_observed_tokens: float
    n_star: int
    token_quantum: int
    source_runs: tuple[CalibrationRun, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "median_observed_tokens": self.median_observed_tokens,
            "n_star": self.n_star,
            "token_quantum": self.token_quantum,
            "source_runs": [run.__dict__ for run in self.source_runs],
            "selection_rule": "floor(median_observed_tokens / token_quantum) * token_quantum",
        }


def choose_n_star(token_counts: list[int] | tuple[int, ...], allowed_batches=ALLOWED_TOTAL_BATCH_SIZES) -> int:
    if len(token_counts) != 3:
        raise ValueError("N_STAR requires exactly three independent calibration runs")
    if any(not isinstance(value, int) or value <= 0 for value in token_counts):
        raise ValueError("successful calibration token counts must be positive integers")
    quantum = math.lcm(*allowed_batches)
    median_tokens = statistics.median(token_counts)
    n_star = math.floor(median_tokens / quantum) * quantum
    if n_star <= 0:
        raise ValueError("the median calibration run is smaller than one common token quantum")
    return validate_n_star(n_star)


def finalize_calibration(runs: list[CalibrationRun] | tuple[CalibrationRun, ...]) -> CalibrationChoice:
    if len(runs) != 3:
        raise ValueError("exactly three calibration runs are required")
    ordered = tuple(sorted(runs, key=lambda run: run.run_index))
    if tuple(run.run_index for run in ordered) != (1, 2, 3):
        raise ValueError("calibration run indexes must be unique and equal to 1, 2, 3")
    for run in ordered:
        run.validate()
        if run.error:
            raise ValueError(f"calibration run {run.run_index} failed: {run.error}")
    n_star = choose_n_star([run.total_tokens for run in ordered])
    return CalibrationChoice(
        median_observed_tokens=statistics.median(run.total_tokens for run in ordered),
        n_star=n_star,
        token_quantum=COMMON_TOKEN_QUANTUM,
        source_runs=ordered,
    )


def calibration_command(*, seed: int, baseline_source: str, output: str) -> list[str]:
    """Return the deferred command; the caller must explicitly opt into execution."""
    return [
        "python",
        "-m",
        "study.contracts_2026.runner",
        "calibrate",
        "--contract",
        "T",
        "--repetitions",
        "3",
        "--seed",
        str(seed),
        "--baseline-source",
        baseline_source,
        "--output",
        output,
        "--execute",
    ]
