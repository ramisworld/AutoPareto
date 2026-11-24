"""Exact token accounting and the common batching policy."""

from __future__ import annotations

from dataclasses import dataclass
import math


MAX_SEQ_LEN = 2048
# These are the distinct TOTAL_BATCH_SIZE values actually used by the registered
# AutoPareto campaigns (seed 0, seed 1, and seed 42).  They are not inferred from
# a paper claim: the source declarations were statically inspected in queue/ and
# results/candidates/ before this harness was created.
HISTORICAL_AUTOPARETO_TOTAL_BATCH_SIZES = (32768, 65536, 131072, 262144, 524288)
ALLOWED_TOTAL_BATCH_SIZES = HISTORICAL_AUTOPARETO_TOTAL_BATCH_SIZES
COMMON_TOKEN_QUANTUM = math.lcm(*ALLOWED_TOTAL_BATCH_SIZES)


@dataclass(frozen=True)
class BatchConfig:
    total_batch_size: int
    device_batch_size: int
    accumulation_steps: int
    sequence_length: int = MAX_SEQ_LEN

    @property
    def tokens_per_microbatch(self) -> int:
        return self.device_batch_size * self.sequence_length

    @property
    def microbatches_per_update(self) -> int:
        return self.accumulation_steps

    @property
    def tokens_per_optimizer_update(self) -> int:
        return self.total_batch_size

    def validate(self) -> None:
        validate_batch_config(
            self.total_batch_size,
            self.device_batch_size,
            self.accumulation_steps,
            self.sequence_length,
        )


def validate_batch_config(
    total_batch_size: int,
    device_batch_size: int,
    accumulation_steps: int,
    sequence_length: int = MAX_SEQ_LEN,
) -> BatchConfig:
    if total_batch_size not in ALLOWED_TOTAL_BATCH_SIZES:
        raise ValueError(f"TOTAL_BATCH_SIZE {total_batch_size} is not in the common allowed set")
    if device_batch_size <= 0 or accumulation_steps <= 0 or sequence_length <= 0:
        raise ValueError("batch and sequence dimensions must be positive")
    expected = device_batch_size * sequence_length * accumulation_steps
    if expected != total_batch_size:
        raise ValueError(
            "TOTAL_BATCH_SIZE must equal DEVICE_BATCH_SIZE * sequence_length * accumulation_steps"
        )
    return BatchConfig(total_batch_size, device_batch_size, accumulation_steps, sequence_length)


def validate_n_star(n_star: int) -> int:
    if not isinstance(n_star, int) or n_star <= 0:
        raise ValueError("N_STAR must be a positive integer")
    if n_star % COMMON_TOKEN_QUANTUM:
        raise ValueError(f"N_STAR must be divisible by {COMMON_TOKEN_QUANTUM}")
    return n_star


def exact_optimizer_updates(n_star: int, total_batch_size: int) -> int:
    validate_n_star(n_star)
    if total_batch_size not in ALLOWED_TOTAL_BATCH_SIZES:
        raise ValueError("illegal TOTAL_BATCH_SIZE")
    if n_star % total_batch_size:
        raise ValueError("N_STAR is not exactly divisible by TOTAL_BATCH_SIZE")
    return n_star // total_batch_size


def accounted_tokens(*, optimizer_updates: int, total_batch_size: int) -> int:
    if optimizer_updates < 0 or total_batch_size <= 0:
        raise ValueError("counts must be non-negative and batch size positive")
    return optimizer_updates * total_batch_size


def legal_device_batch_options(total_batch_size: int, sequence_length: int = MAX_SEQ_LEN) -> tuple[tuple[int, int], ...]:
    """Return (DEVICE_BATCH_SIZE, accumulation_steps) pairs for a legal total."""
    if total_batch_size not in ALLOWED_TOTAL_BATCH_SIZES:
        raise ValueError("illegal TOTAL_BATCH_SIZE")
    pairs = []
    tokens_per_example = sequence_length
    for device_batch_size in range(1, total_batch_size // tokens_per_example + 1):
        remainder = total_batch_size % (device_batch_size * tokens_per_example)
        if remainder == 0:
            pairs.append((device_batch_size, total_batch_size // (device_batch_size * tokens_per_example)))
    return tuple(pairs)
