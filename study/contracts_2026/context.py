"""Construct the only history view that can be shown to a study agent."""

from __future__ import annotations

from typing import Any

CAMPAIGN_VIEW_SCHEMA = "autopareto_campaign_view_v3"

_VISIBLE_INFERENCE_FIELDS = (
    # Headline values and the complete bracketed measurement evidence.
    "inference_status",
    "absolute_candidate_tok_s",
    "absolute_baseline_tok_s",
    "absolute_candidate_ci_95pct",
    "absolute_baseline_ci_95pct",
    "absolute_baseline_before_tok_s",
    "absolute_baseline_after_tok_s",
    "baseline_before_window_tok_s",
    "baseline_after_window_tok_s",
    "interpolated_baseline_window_tok_s",
    "measurement_windows",
    "normalized_pairwise_ratios",
    "normalized_decode_throughput",
    "normalized_decode_throughput_ci_95pct",
    "normalized_coefficient_of_variation",
    "normalized_range_drift",
    "normalized_relative_drift",
    "normalized_stability_failure_reason",
    "stability_failure_reason",
    # Diagnostic throughput and memory fields used to explain a trade-off.
    "decode_block_velocities",
    "decode_block_range_drift",
    "decode_block_relative_drift",
    "decode_velocity_point_estimate",
    "decode_velocity_95pct_interval",
    "cached_decode_tokens_per_second_median",
    "cached_decode_coefficient_of_variation",
    "peak_allocated_inference_memory_bytes",
    "reserved_cuda_memory_bytes",
    "autoregressive_state_bytes",
    "kv_cache_bytes",
    "candidate_only_memory_measurement",
    # Correctness/protocol identity, not hidden evaluation data.
    "correctness",
    "paired_measurement_protocol",
    "protocol_version",
    "evaluator_sha256",
    "prompt_set_sha256",
    "precision",
    "inference_batch_size",
    "input_tokens",
    "generated_tokens",
    "warmup_mode",
    "warmup_blocks",
    "warmup_repetitions",
    "warmup_repetition_count",
    "warmup_decode_seconds",
    "prefill_milliseconds",
    "first_token_milliseconds",
)

_VISIBLE_TRAINING_FIELDS = (
    "contract_id",
    "training_seed",
    "total_tokens",
    "measured_training_seconds",
    "protected_measured_training_seconds",
    "optimizer_updates",
    "microbatches",
    "num_params",
    "training_throughput_tokens_per_second",
    "sequence_length",
    "total_batch_size",
    "device_batch_size",
    "accumulation_steps",
    "training_flops_estimate",
    "warmup_lifecycle",
)

# Direct result fields retained in the campaign-local view. This is deliberately
# separate from the full durable result so provider logs, checkpoints, and
# protected downstream artifacts cannot enter Scientist context.
VIEW_RESULT_FIELDS = (
    "campaign_id",
    "attempt",
    "experiment_id",
    "parent_experiment_id",
    "candidate_source_before_sha256",
    "candidate_source_after_sha256",
    "validation_bpb",
    "protected_validation_bpb",
    "hypothesis",
    "intended_change",
    "change_category",
    "candidate_patch",
    "status",
    "failure_class",
    "error",
    "quality_status",
    "parent_eligible",
    "pareto_eligible",
    "pareto_decision",
    *_VISIBLE_TRAINING_FIELDS,
)

_CONTROLLER_ARCHIVE_FIELDS = (
    "record_scope",
    "campaign_id",
    "attempt",
    "experiment_id",
    "baseline_experiment_id",
    "parent_experiment_id",
    "candidate_source_before_sha256",
    "candidate_source_after_sha256",
    "hypothesis",
    "intended_change",
    "change_category",
    "candidate_patch",
    "protected_validation_bpb",
    "validation_bpb",
    "status",
    "failure_class",
    "error",
    "quality_status",
    "parent_eligible",
    "pareto_eligible",
    "pareto_decision",
    *_VISIBLE_TRAINING_FIELDS,
)


def _scientist_evidence(record: dict[str, Any]) -> dict[str, Any]:
    """Project one same-arm result into complete non-sealed Scientist evidence."""
    evidence = {
        key: record[key]
        for key in (
            "campaign_id",
            "attempt",
            "experiment_id",
            "baseline_experiment_id",
            "parent_experiment_id",
            "candidate_source_before_sha256",
            "candidate_source_after_sha256",
            "hypothesis",
            "intended_change",
            "change_category",
            "candidate_patch",
            "protected_validation_bpb",
            "validation_bpb",
            "status",
            "failure_class",
            "error",
            "quality_status",
            "parent_eligible",
            "pareto_eligible",
            "pareto_decision",
        )
        if key in record
    }
    for key in _VISIBLE_TRAINING_FIELDS:
        if key in record:
            evidence[key] = record[key]
    # The trusted evaluator's authoritative field is protected_validation_bpb,
    # while the Scientist-facing contract names the metric validation_bpb.
    # Expose the same value under the canonical Scientist-facing name without
    # discarding the provenance-specific field above.
    if "validation_bpb" not in evidence and "protected_validation_bpb" in evidence:
        evidence["validation_bpb"] = evidence["protected_validation_bpb"]
    inference = record.get("inference_result", record.get("inference"))
    if isinstance(inference, dict):
        evidence["inference"] = {
            key: inference[key] for key in _VISIBLE_INFERENCE_FIELDS if key in inference
        }
    return evidence


def _controller_archive_record(record: dict[str, Any]) -> dict[str, Any]:
    """Project a result for controller arithmetic without sealed metrics."""
    compact = {key: record[key] for key in _CONTROLLER_ARCHIVE_FIELDS if key in record}
    inference = record.get("inference_result", record.get("inference"))
    if isinstance(inference, dict):
        compact["inference_result"] = {
            key: inference[key] for key in _VISIBLE_INFERENCE_FIELDS if key in inference
        }
    return compact


def build_campaign_view(
    *,
    campaign_id: str,
    contract_id: str,
    baseline: dict[str, Any],
    frontier: list[dict[str, Any]],
    attempt_number: int,
    failures: list[dict[str, Any]] | None = None,
    evidence: list[dict[str, Any]] | None = None,
    parent_pool: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return an allowlisted, campaign-local context with no sealed metrics."""
    if attempt_number < 1 or attempt_number > 25:
        raise ValueError("attempt number must be in 1..25")
    if baseline.get("record_scope") != "same_seed_baseline":
        raise ValueError("baseline is not marked as the registered same-seed baseline")
    evidence = list(evidence if evidence is not None else frontier)
    parent_pool = list(parent_pool if parent_pool is not None else frontier)
    for record in (baseline, *frontier, *evidence, *parent_pool):
        if record.get("campaign_id") != campaign_id:
            if record is baseline and record.get("record_scope") == "same_seed_baseline":
                pass
            else:
                raise ValueError("campaign view contains a cross-campaign record")
        if any(key in record for key in ("held_out_bpb", "BLiMP", "PIQA", "ARC-Easy", "LAMBADA")):
            raise ValueError("downstream metric entered search context")
    lineage = [
        {
            "experiment_id": baseline.get("baseline_experiment_id", "registered_baseline"),
            "parent_experiment_id": None,
            "source_sha256": baseline.get("candidate_source_after_sha256")
            or baseline.get("starting_source_hash")
            or baseline.get("source_sha256"),
            "status": "registered_baseline",
            "parent_eligible": True,
            "pareto_eligible": True,
        }
    ]
    lineage.extend(
        {
            "experiment_id": record.get("experiment_id"),
            "parent_experiment_id": record.get("parent_experiment_id"),
            "source_sha256": record.get("candidate_source_after_sha256"),
            "status": record.get("status", "completed"),
            "quality_status": record.get("quality_status"),
            "parent_eligible": bool(record.get("parent_eligible", False)),
            "pareto_eligible": bool(record.get("pareto_eligible", False)),
        }
        for record in evidence
        if record.get("experiment_id")
    )
    return {
        "schema": CAMPAIGN_VIEW_SCHEMA,
        "campaign_id": campaign_id,
        "contract_id": contract_id,
        "attempt_number": attempt_number,
        # Only non-sealed, search-relevant evidence is mounted. Full stdout,
        # provider diagnostics, checkpoints, and protected downstream metrics
        # remain outside the Scientist context.
        "baseline": _controller_archive_record(baseline),
        "frontier": [_controller_archive_record(record) for record in frontier],
        "parent_pool": [_scientist_evidence(record) for record in parent_pool],
        "same_arm_evidence": [_scientist_evidence(record) for record in evidence],
        "lineage": lineage,
        "failures": list(failures or []),
        "visible_metrics": [
            "validation_bpb",
            "absolute_candidate_tok_s",
            "absolute_baseline_tok_s",
            "absolute_candidate_ci_95pct",
            "absolute_baseline_ci_95pct",
            "baseline_before_window_tok_s",
            "baseline_after_window_tok_s",
            "interpolated_baseline_window_tok_s",
            "measurement_windows",
            "normalized_decode_throughput",
            "normalized_decode_throughput_ci_95pct",
            "normalized_pairwise_ratios",
            "normalized_coefficient_of_variation",
            "normalized_range_drift",
            "normalized_relative_drift",
            "decode_velocity_point_estimate",
            "decode_velocity_95pct_interval",
            "decode_block_velocities",
            "peak_allocated_inference_memory_bytes",
            "reserved_cuda_memory_bytes",
            "autoregressive_state_bytes",
            "kv_cache_bytes",
            "candidate_only_memory_measurement",
            *_VISIBLE_TRAINING_FIELDS,
            "pareto_decision",
            "quality_status",
            "parent_eligible",
            "pareto_eligible",
        ],
        "sealed_metrics": ["held_out_bpb", "BLiMP", "PIQA", "ARC-Easy", "LAMBADA"],
    }
