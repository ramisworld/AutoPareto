"""Frozen Design-B search-time inference statistics."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import statistics
from typing import Iterable


PROTOCOL_VERSION = "autopareto_search_inference_bracketed_v2"
DECODE_BLOCKS = 8
DECODE_BLOCK_SECONDS = 8.0
PREFILL_BLOCKS = 5
WARMUP_SECONDS = 20.0
BOOTSTRAP_RESAMPLES = 10_000
UNCERTAINTY_RELATIVE_HALF_WIDTH_LIMIT = 0.03
PRACTICAL_EQUIVALENCE_MARGIN = 0.02
MAX_RELATIVE_DRIFT = 0.02
MAX_BLOCK_CV = 0.05


@dataclass(frozen=True)
class StabilityConfig:
    """Compatibility surface for callers that previously imported this class."""

    decode_blocks: int = DECODE_BLOCKS
    decode_block_seconds: float = DECODE_BLOCK_SECONDS
    prefill_blocks: int = PREFILL_BLOCKS
    warmup_seconds: float = WARMUP_SECONDS
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES


DEFAULT_STABILITY = StabilityConfig()


def _positive(values: Iterable[float], name: str) -> list[float]:
    result = [float(value) for value in values]
    if not result or any(not math.isfinite(value) or value <= 0 for value in result):
        raise ValueError(f"{name} must contain finite positive values")
    return result


def coefficient_of_variation(values: Iterable[float]) -> float:
    values = _positive(values, "values")
    return statistics.pstdev(values) / statistics.fmean(values)


def classify_measurement_stability(
    decode_blocks: Iterable[float],
    *,
    max_relative_drift: float = MAX_RELATIVE_DRIFT,
    max_cv: float = MAX_BLOCK_CV,
) -> dict[str, object]:
    """Apply the registered final measured-window stability gates."""
    blocks = _positive(decode_blocks, "decode blocks")
    mean = statistics.fmean(blocks)
    standard_deviation = statistics.pstdev(blocks)
    cv = standard_deviation / mean
    median = statistics.median(blocks)
    quartiles = statistics.quantiles(blocks, n=4, method="inclusive")
    # Retain the registered 2% limit, but use robust interquartile drift for
    # measured windows. Keep raw range as a diagnostic so one outlier cannot
    # decide a run.
    drift = (quartiles[2] - quartiles[0]) / median
    range_drift = (max(blocks) - min(blocks)) / min(blocks)
    reasons = []
    if drift > max_relative_drift:
        reasons.append(f"measured drift {drift} > {max_relative_drift}")
    if cv > max_cv:
        reasons.append(f"measured CV {cv} > {max_cv}")
    return {
        "inference_status": "unstable" if reasons else "stable",
        "cached_decode_coefficient_of_variation": cv,
        "decode_block_relative_drift": drift,
        "decode_block_range_drift": range_drift,
        "stability_failure_reason": "; ".join(reasons) if reasons else None,
    }


def paired_decode_throughput(
    candidate_blocks: Iterable[float],
    baseline_blocks: Iterable[float],
    *,
    bootstrap_seed: int = 0,
) -> dict[str, object]:
    """Summarize one contemporaneous eight-window candidate/baseline pair.

    Absolute throughput is retained for reporting.  The search scalar is the
    median of the same-index candidate/baseline ratios, so common temporal
    device drift is removed without selecting individual windows.
    """
    candidate = _positive(candidate_blocks, "candidate decode blocks")
    baseline = _positive(baseline_blocks, "baseline decode blocks")
    if len(candidate) != DECODE_BLOCKS or len(baseline) != DECODE_BLOCKS:
        raise ValueError(f"paired throughput requires exactly {DECODE_BLOCKS} windows per arm")
    ratios = [candidate_value / baseline_value for candidate_value, baseline_value in zip(candidate, baseline)]
    candidate_point, candidate_lower, candidate_upper = bootstrap_median_ci(
        candidate, seed=bootstrap_seed
    )
    baseline_point, baseline_lower, baseline_upper = bootstrap_median_ci(
        baseline, seed=bootstrap_seed + 1
    )
    ratio_point, ratio_lower, ratio_upper = bootstrap_median_ci(
        ratios, seed=bootstrap_seed + 2
    )
    ratio_stability = classify_measurement_stability(ratios)
    return {
        "absolute_candidate_tok_s": candidate_point,
        "absolute_candidate_ci_95pct": [candidate_lower, candidate_upper],
        "absolute_baseline_tok_s": baseline_point,
        "absolute_baseline_ci_95pct": [baseline_lower, baseline_upper],
        "candidate_window_tok_s": candidate,
        "baseline_window_tok_s": baseline,
        "normalized_pairwise_ratios": ratios,
        "normalized_decode_throughput": ratio_point,
        "normalized_decode_throughput_ci_95pct": [ratio_lower, ratio_upper],
        "normalized_coefficient_of_variation": ratio_stability["cached_decode_coefficient_of_variation"],
        "normalized_relative_drift": ratio_stability["decode_block_relative_drift"],
        "normalized_range_drift": ratio_stability["decode_block_range_drift"],
        "normalized_inference_status": ratio_stability["inference_status"],
        "normalized_stability_failure_reason": ratio_stability["stability_failure_reason"],
    }


def bracketed_decode_throughput(
    candidate_windows: Iterable[dict[str, object]],
    baseline_before_windows: Iterable[dict[str, object]],
    baseline_after_windows: Iterable[dict[str, object]],
    *,
    bootstrap_seed: int = 0,
) -> dict[str, object]:
    """Normalize eight candidate windows against interpolated bracket baselines.

    Each row must contain ``decode_tokens_per_second``,
    ``started_at_unix_ns``, and ``completed_at_unix_ns``.  For every index the
    baseline is measured before and after the candidate, then linearly
    interpolated at the candidate window midpoint.  The existing stability
    thresholds are applied unchanged to the eight normalized ratios.
    """

    candidate = list(candidate_windows)
    before = list(baseline_before_windows)
    after = list(baseline_after_windows)
    if not (len(candidate) == len(before) == len(after) == DECODE_BLOCKS):
        raise ValueError(f"bracketed throughput requires exactly {DECODE_BLOCKS} windows per arm")

    def row_values(rows: list[dict[str, object]], label: str) -> list[float]:
        values = []
        for row in rows:
            try:
                values.append(float(row["decode_tokens_per_second"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{label} row is missing finite throughput") from exc
        return _positive(values, f"{label} decode windows")

    def midpoint(row: dict[str, object], label: str) -> float:
        try:
            started = int(row["started_at_unix_ns"])
            completed = int(row["completed_at_unix_ns"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{label} row is missing timestamps") from exc
        if started < 0 or completed <= started:
            raise ValueError(f"{label} row has invalid timestamps")
        return (started + completed) / 2.0

    candidate_values = row_values(candidate, "candidate")
    before_values = row_values(before, "baseline-before")
    after_values = row_values(after, "baseline-after")
    interpolated = []
    for index, (candidate_row, before_row, after_row) in enumerate(zip(candidate, before, after)):
        candidate_time = midpoint(candidate_row, f"candidate[{index}]")
        before_time = midpoint(before_row, f"baseline-before[{index}]")
        after_time = midpoint(after_row, f"baseline-after[{index}]")
        if not before_time < candidate_time < after_time:
            raise ValueError(f"bracket timestamps are not ordered for window {index}")
        weight = (candidate_time - before_time) / (after_time - before_time)
        interpolated.append(before_values[index] + weight * (after_values[index] - before_values[index]))

    ratios = [candidate_value / baseline_value for candidate_value, baseline_value in zip(candidate_values, interpolated)]
    candidate_point, candidate_lower, candidate_upper = bootstrap_median_ci(
        candidate_values, seed=bootstrap_seed
    )
    baseline_point, baseline_lower, baseline_upper = bootstrap_median_ci(
        interpolated, seed=bootstrap_seed + 1
    )
    ratio_point, ratio_lower, ratio_upper = bootstrap_median_ci(
        ratios, seed=bootstrap_seed + 2
    )
    candidate_stability = classify_measurement_stability(candidate_values)
    before_stability = classify_measurement_stability(before_values)
    after_stability = classify_measurement_stability(after_values)
    ratio_stability = classify_measurement_stability(ratios)
    return {
        "absolute_candidate_tok_s": candidate_point,
        "absolute_candidate_ci_95pct": [candidate_lower, candidate_upper],
        "absolute_baseline_tok_s": baseline_point,
        "absolute_baseline_ci_95pct": [baseline_lower, baseline_upper],
        "absolute_baseline_before_tok_s": statistics.median(before_values),
        "absolute_baseline_after_tok_s": statistics.median(after_values),
        "candidate_window_tok_s": candidate_values,
        "baseline_before_window_tok_s": before_values,
        "baseline_after_window_tok_s": after_values,
        "interpolated_baseline_window_tok_s": interpolated,
        "normalized_pairwise_ratios": ratios,
        "normalized_decode_throughput": ratio_point,
        "normalized_decode_throughput_ci_95pct": [ratio_lower, ratio_upper],
        "normalized_coefficient_of_variation": ratio_stability["cached_decode_coefficient_of_variation"],
        "normalized_relative_drift": ratio_stability["decode_block_relative_drift"],
        "normalized_range_drift": ratio_stability["decode_block_range_drift"],
        "normalized_inference_status": ratio_stability["inference_status"],
        "normalized_stability_failure_reason": ratio_stability["stability_failure_reason"],
        "candidate_absolute_diagnostics": candidate_stability,
        "baseline_before_absolute_diagnostics": before_stability,
        "baseline_after_absolute_diagnostics": after_stability,
    }


def bootstrap_median_ci(
    block_values: Iterable[float],
    *,
    seed: int = 0,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float, float]:
    """Return (median, lower, upper) from a whole-block percentile bootstrap."""
    values = _positive(block_values, "block values")
    if len(values) < 2:
        raise ValueError("at least two blocks are required for uncertainty")
    if resamples < 100:
        raise ValueError("bootstrap requires at least 100 resamples")
    rng = random.Random(seed)
    medians = [statistics.median(rng.choices(values, k=len(values))) for _ in range(resamples)]
    medians.sort()
    lower = medians[int(0.025 * (len(medians) - 1))]
    upper = medians[int(0.975 * (len(medians) - 1))]
    return statistics.median(values), lower, upper


def combined_log_ci(
    block_values: Iterable[float],
    *,
    process_log_sd: float = 0.0,
    seed: int = 0,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, float]:
    """Combine block uncertainty with a multiplicative fresh-process variance floor."""
    values = _positive(block_values, "block values")
    if process_log_sd < 0 or not math.isfinite(process_log_sd):
        raise ValueError("process_log_sd must be finite and non-negative")
    point = statistics.median(values)
    rng = random.Random(seed)
    log_medians = [
        statistics.median(math.log(value) for value in rng.choices(values, k=len(values)))
        for _ in range(resamples)
    ]
    within_sd = statistics.stdev(log_medians) if len(log_medians) > 1 else 0.0
    total_sd = math.sqrt(within_sd * within_sd + process_log_sd * process_log_sd)
    half = 1.959963984540054 * total_sd
    return {
        "point": point,
        "lower": math.exp(math.log(point) - half),
        "upper": math.exp(math.log(point) + half),
        "within_process_log_sd": within_sd,
        "process_log_sd": process_log_sd,
        "combined_log_sd": total_sd,
    }


def estimate_process_log_sd(model_runs: dict[str, Iterable[float]]) -> dict[str, object]:
    """Estimate a frozen process floor from one point per fresh process."""
    per_model: dict[str, float] = {}
    counts: dict[str, int] = {}
    for model, runs in model_runs.items():
        values = _positive(runs, f"{model} process runs")
        if len(values) < 3:
            raise ValueError("at least three fresh processes per development model are required")
        logs = [math.log(value) for value in values]
        per_model[model] = statistics.stdev(logs)
        counts[model] = len(values)
    return {
        "policy": "max_model_sample_sd_log_velocity",
        "process_log_sd": max(per_model.values(), default=0.0),
        "per_model_log_sd": per_model,
        "counts": counts,
    }


def speed_relation(
    candidate_velocity: float,
    candidate_interval: tuple[float, float],
    incumbent_velocity: float,
    incumbent_interval: tuple[float, float],
    *,
    practical_margin: float = PRACTICAL_EQUIVALENCE_MARGIN,
) -> dict[str, object]:
    """Conservatively classify a pairwise speed relation using log intervals."""
    if candidate_velocity <= 0 or incumbent_velocity <= 0:
        raise ValueError("velocities must be positive")
    if not 0 <= practical_margin < 1:
        raise ValueError("practical margin must be in [0, 1)")
    lower = math.log(candidate_interval[0]) - math.log(incumbent_interval[1])
    upper = math.log(candidate_interval[1]) - math.log(incumbent_interval[0])
    threshold = math.log1p(practical_margin)
    relation = "faster" if lower > threshold else "slower" if upper < -threshold else "tied"
    return {
        "relation": relation,
        "log_ratio_point": math.log(candidate_velocity / incumbent_velocity),
        "log_ratio_interval": [lower, upper],
        "practical_equivalence_margin": practical_margin,
    }


def accept_measurement(
    *,
    correctness_passed: bool,
    decode_blocks: Iterable[float] | None = None,
    measured_passes: Iterable[Iterable[float]] | None = None,
    warmup_blocks: Iterable[Iterable[float]] | None = None,
    config: StabilityConfig | None = None,
    process_log_sd: float = 0.0,
    bootstrap_seed: int = 0,
    invalid_reason: str | None = None,
) -> dict[str, object]:
    """Finalize a valid Design-B measurement without noise-based rejection."""
    del warmup_blocks, config
    if decode_blocks is None:
        # Compatibility for pre-Design-B CPU fixtures; real evaluators always pass
        # the eight measured block rates explicitly.
        decode_blocks = [statistics.fmean(group) for group in (measured_passes or [])]
    decode_blocks = list(decode_blocks)
    if not correctness_passed and not decode_blocks:
        decode_blocks = [1.0]
    blocks = _positive(decode_blocks, "decode blocks")
    invalid = invalid_reason or (None if correctness_passed else "correctness failure")
    result: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "correctness_passed": bool(correctness_passed),
        "decode_block_count": len(blocks),
        "decode_block_velocities": blocks,
        "coefficient_variation_blocks": coefficient_of_variation(blocks),
        "status": "invalid" if invalid else "valid",
        "inference_status": "invalid" if invalid else "stable",
        "invalid_reason": invalid,
    }
    if invalid:
        result.update({"decode_velocity_point_estimate": None, "decode_velocity_95pct_interval": None})
        return result
    interval = combined_log_ci(blocks, process_log_sd=process_log_sd, seed=bootstrap_seed)
    result.update(
        {
            "decode_velocity_point_estimate": interval["point"],
            "decode_velocity_95pct_interval": [interval["lower"], interval["upper"]],
            "uncertainty": interval,
            "cached_decode_tokens_per_second_median": interval["point"],
        }
    )
    return result
