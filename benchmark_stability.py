"""Pure, testable stability rules for final GPU inference measurements.

Discovery measurements used a fixed warm-up count.  Final paper measurements use
this adaptive protocol and begin only after two consecutive timing blocks are stable.
The functions in this module deliberately know nothing about CUDA so their policy can
be exhaustively tested without a GPU.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import statistics
from typing import Any, Callable


STABILIZED_PROTOCOL_VERSION = "autopareto_inference_stabilized_v2"


@dataclass(frozen=True)
class StabilityConfig:
    block_size: int = 5
    required_stable_blocks: int = 2
    max_warmups: int = 30
    median_relative_tolerance: float = 0.02
    coefficient_of_variation_limit: float = 0.05

    def validate(self) -> None:
        if self.block_size < 2:
            raise ValueError("stability block size must be at least two")
        if self.required_stable_blocks != 2:
            raise ValueError("the frozen protocol requires exactly two stable blocks")
        minimum = self.block_size * self.required_stable_blocks
        if self.max_warmups < minimum or self.max_warmups % self.block_size:
            raise ValueError("max warm-ups must be a block-aligned multiple of at least two blocks")
        if not 0 < self.median_relative_tolerance < 1:
            raise ValueError("median tolerance must be between zero and one")
        if not 0 < self.coefficient_of_variation_limit < 1:
            raise ValueError("coefficient-of-variation limit must be between zero and one")


class StabilityError(RuntimeError):
    def __init__(self, result: dict[str, Any]):
        super().__init__(result["failure_reason"])
        self.result = result


def coefficient_of_variation(values: list[float]) -> float:
    if len(values) < 2:
        raise ValueError("coefficient of variation requires at least two values")
    mean = statistics.mean(values)
    if mean <= 0:
        raise ValueError("timing throughput must be positive")
    return statistics.stdev(values) / mean


def summarize_block(rows: list[dict[str, Any]], block_index: int) -> dict[str, Any]:
    speeds = [float(row["decode_tokens_per_second"]) for row in rows]
    if any(value <= 0 for value in speeds):
        raise ValueError("warm-up throughput must be positive")
    return {
        "block_index": block_index,
        "sample_count": len(speeds),
        "median_decode_tokens_per_second": statistics.median(speeds),
        "mean_decode_tokens_per_second": statistics.mean(speeds),
        "coefficient_of_variation": coefficient_of_variation(speeds),
        "minimum_decode_tokens_per_second": min(speeds),
        "maximum_decode_tokens_per_second": max(speeds),
    }


def compare_blocks(
    previous: dict[str, Any],
    current: dict[str, Any],
    config: StabilityConfig,
) -> dict[str, Any]:
    previous_median = float(previous["median_decode_tokens_per_second"])
    current_median = float(current["median_decode_tokens_per_second"])
    relative_drift = abs(current_median - previous_median) / previous_median
    stable = (
        relative_drift <= config.median_relative_tolerance
        and float(previous["coefficient_of_variation"])
        <= config.coefficient_of_variation_limit
        and float(current["coefficient_of_variation"])
        <= config.coefficient_of_variation_limit
    )
    return {
        "previous_block_index": int(previous["block_index"]),
        "current_block_index": int(current["block_index"]),
        "median_relative_drift": relative_drift,
        "median_relative_tolerance": config.median_relative_tolerance,
        "coefficient_of_variation_limit": config.coefficient_of_variation_limit,
        "stable": stable,
    }


def adaptive_warmup(
    measure_once: Callable[[int, int], dict[str, Any]],
    config: StabilityConfig,
) -> dict[str, Any]:
    """Run complete untimed inference repetitions until throughput plateaus."""
    config.validate()
    rows: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for block_index in range(config.max_warmups // config.block_size):
        block_rows = []
        for _ in range(config.block_size):
            index = len(rows)
            row = dict(measure_once(index, block_index))
            row.update({
                "is_warmup": True,
                "stability_block_index": block_index,
                "warmup_index": index,
            })
            rows.append(row)
            block_rows.append(row)
        block = summarize_block(block_rows, block_index)
        blocks.append(block)
        if len(blocks) >= 2:
            comparison = compare_blocks(blocks[-2], blocks[-1], config)
            comparisons.append(comparison)
            if comparison["stable"]:
                return {
                    "protocol_version": STABILIZED_PROTOCOL_VERSION,
                    "passed": True,
                    "warmup_count": len(rows),
                    "config": asdict(config),
                    "blocks": blocks,
                    "comparisons": comparisons,
                    "warmup_repetitions": rows,
                    "failure_reason": None,
                }
    result = {
        "protocol_version": STABILIZED_PROTOCOL_VERSION,
        "passed": False,
        "warmup_count": len(rows),
        "config": asdict(config),
        "blocks": blocks,
        "comparisons": comparisons,
        "warmup_repetitions": rows,
        "failure_reason": (
            f"inference throughput did not stabilize after {len(rows)} warm-ups"
        ),
    }
    raise StabilityError(result)


def summarize_measurement_pass(rows: list[dict[str, Any]], pass_index: int) -> dict[str, Any]:
    speeds = [float(row["decode_tokens_per_second"]) for row in rows]
    if len(speeds) < 2:
        raise ValueError("measurement pass requires at least two repetitions")
    return {
        "measurement_pass": pass_index,
        "sample_count": len(speeds),
        "median_decode_tokens_per_second": statistics.median(speeds),
        "mean_decode_tokens_per_second": statistics.mean(speeds),
        "sample_standard_deviation": statistics.stdev(speeds),
        "coefficient_of_variation": coefficient_of_variation(speeds),
        "minimum_decode_tokens_per_second": min(speeds),
        "maximum_decode_tokens_per_second": max(speeds),
    }
