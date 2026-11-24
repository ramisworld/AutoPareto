"""Campaign registration, local histories, and the three-objective frontier."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import pathlib
from typing import Any

from .contracts import ContractId, ResourceContract
from .benchmark_policy import speed_relation


@dataclass(frozen=True)
class CampaignRegistration:
    campaign_id: str
    contract: ResourceContract
    attempts: int
    seed: int
    baseline_experiment_id: str
    starting_source_hash: str
    agent_model: str
    reasoning_level: str
    evaluator_hashes: dict[str, str]
    history_path: str

    def validate(self) -> None:
        if self.attempts != 25:
            raise ValueError("Pair A campaigns must have exactly 25 attempts")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not self.campaign_id:
            raise ValueError("campaign ID is required")
        history_parts = pathlib.PurePosixPath(self.history_path).parts
        if self.history_path.startswith("/") or ".." in history_parts or history_parts[:2] != ("runs", self.campaign_id):
            raise ValueError("history path must be portable and campaign-local")
        if self.contract.contract_id is ContractId.N and self.contract.n_star is None:
            raise ValueError("fixed-token registration is not runnable until N_STAR is frozen")

    def as_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "contract_id": self.contract.contract_id.value,
            "attempts": self.attempts,
            "seed": self.seed,
            "baseline_experiment_id": self.baseline_experiment_id,
            "starting_source_hash": self.starting_source_hash,
            "agent_model": self.agent_model,
            "reasoning_level": self.reasoning_level,
            "evaluator_hashes": self.evaluator_hashes,
            "history_path": self.history_path,
            "n_star": self.contract.n_star,
        }


def campaign_history_path(root: pathlib.Path, campaign_id: str) -> pathlib.Path:
    if not campaign_id or "/" in campaign_id or ".." in pathlib.PurePosixPath(campaign_id).parts:
        raise ValueError("unsafe campaign ID")
    return root / "runs" / campaign_id / "history.jsonl"


def append_local_history(root: pathlib.Path, campaign_id: str, record: dict[str, Any]) -> pathlib.Path:
    path = campaign_history_path(root, campaign_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if record.get("campaign_id") != campaign_id:
        raise ValueError("a result cannot enter another campaign history")
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def append_frontier(root: pathlib.Path, campaign_id: str, candidate: dict[str, Any]) -> pathlib.Path:
    """Append only a result carrying the same campaign ID to its own frontier."""
    if candidate.get("campaign_id") != campaign_id:
        raise ValueError("a candidate cannot enter another campaign frontier")
    path = root / "runs" / campaign_id / "frontier.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(candidate, sort_keys=True) + "\n")
    return path


def dominates(left: dict[str, float], right: dict[str, float]) -> bool:
    """Return whether left is no worse in all objectives and better in one."""
    better_or_equal = (
        left["validation_bpb"] <= right["validation_bpb"]
        and left["normalized_decode_throughput"] >= right["normalized_decode_throughput"]
        and left["peak_allocated_inference_memory_bytes"] <= right["peak_allocated_inference_memory_bytes"]
    )
    strictly_better = (
        left["validation_bpb"] < right["validation_bpb"]
        or left["normalized_decode_throughput"] > right["normalized_decode_throughput"]
        or left["peak_allocated_inference_memory_bytes"] < right["peak_allocated_inference_memory_bytes"]
    )
    return better_or_equal and strictly_better


def protected_search_metrics(record: dict[str, Any]) -> dict[str, float]:
    """Extract the only three metrics permitted to enter the search frontier."""
    inference = record.get("inference_result")
    if not isinstance(inference, dict) or inference.get("inference_status") != "stable":
        raise ValueError("only a protected stable inference result can enter the frontier")
    validation_bpb = record.get("protected_validation_bpb")
    throughput = inference.get("normalized_decode_throughput")
    if throughput is None and record.get("record_scope") == "same_seed_baseline":
        # The registered baseline is the normalization reference.  Its own
        # absolute throughput remains in the record; normalized baseline speed
        # is exactly one by definition.
        throughput = 1.0
    memory = inference.get("peak_allocated_inference_memory_bytes", record.get("peak_allocated_inference_memory_bytes"))
    if validation_bpb is None or throughput is None or memory is None:
        raise ValueError("protected search metrics are incomplete")
    return {
        "validation_bpb": float(validation_bpb),
        "normalized_decode_throughput": float(throughput),
        "peak_allocated_inference_memory_bytes": float(memory),
    }


def candidate_disposition(
    record: dict[str, Any],
    *,
    quality_reference_bpb: float | None = None,
) -> dict[str, Any]:
    """Separate valid scientific source evidence from Pareto eligibility.

    A result with valid protected quality and cache correctness may remain a
    same-arm parent when throughput is uncertain.  It cannot enter the
    three-objective archive until the protected throughput measurement is
    valid.  Cache-correctness or validation failures are never parent eligible.
    """
    inference = record.get("inference_result")
    validation_bpb = record.get("protected_validation_bpb")
    if not isinstance(inference, dict):
        return {
            "quality_status": "invalid",
            "parent_eligible": False,
            "pareto_eligible": False,
            "reason": "missing protected inference result",
        }
    if inference.get("inference_status") == "stable":
        # Preserve the established fail-closed invariant for records claiming
        # a stable protected measurement: all three protected search metrics
        # must actually be present, rather than being silently downgraded.
        protected_search_metrics(record)
    try:
        quality_valid = math.isfinite(float(validation_bpb))
    except (TypeError, ValueError):
        quality_valid = False
    correctness = inference.get("correctness")
    # Stable historical records predate the explicit correctness field. Their
    # stable status is the registered proof of the protected correctness gate;
    # an uncertain-speed record must carry the explicit current proof.
    correctness_valid = (
        inference.get("inference_status") == "stable"
        or (isinstance(correctness, dict) and correctness.get("passed") is True)
    )
    if not quality_valid or not correctness_valid:
        return {
            "quality_status": "invalid",
            "parent_eligible": False,
            "pareto_eligible": False,
            "reason": "protected validation or cached-decode correctness is invalid",
        }
    if inference.get("inference_status") != "stable":
        # Uncertain speed cannot justify carrying a candidate whose quality is
        # already worse than the registered baseline. Keep the evidence, but
        # fail closed on parent eligibility until speed is valid. Stable
        # measurements are adjudicated by Pareto arithmetic and may trade
        # quality for speed or memory.
        parent_eligible = (
            quality_reference_bpb is not None
            and float(validation_bpb) <= float(quality_reference_bpb)
        )
        return {
            "quality_status": "quality_valid_speed_uncertain",
            "parent_eligible": parent_eligible,
            "pareto_eligible": False,
            "reason": (
                "quality and cache correctness passed; protected throughput is uncertain"
                if parent_eligible
                else "quality and cache correctness passed, but uncertain speed cannot justify quality worse than baseline"
            ),
        }
    return {
        "quality_status": "quality_valid_speed_valid",
        "parent_eligible": True,
        "pareto_eligible": True,
        "reason": "protected validation, cache correctness, and throughput are valid",
    }


def _speed_interval(inference: dict[str, Any], point: float) -> tuple[float, float]:
    """Read the frozen interval, retaining point-only compatibility for old fixtures."""
    raw = inference.get("normalized_decode_throughput_ci_95pct")
    if raw is None:
        raw = inference.get("decode_velocity_95pct_interval")
    if raw is None:
        return point, point
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError("decode velocity uncertainty interval must contain two bounds")
    lower, upper = float(raw[0]), float(raw[1])
    if not (0 < lower <= upper):
        raise ValueError("decode velocity uncertainty interval is invalid")
    return lower, upper


def decide_candidate(candidate: dict[str, Any], archive: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = next(
        (record for record in archive if record.get("record_scope") == "same_seed_baseline"),
        None,
    )
    baseline_bpb = None if baseline is None else baseline.get("protected_validation_bpb")
    try:
        baseline_bpb = None if baseline_bpb is None else float(baseline_bpb)
    except (TypeError, ValueError):
        baseline_bpb = None
    disposition = candidate_disposition(candidate, quality_reference_bpb=baseline_bpb)
    if not disposition["parent_eligible"]:
        return {"decision": "invalid", **disposition}
    if not disposition["pareto_eligible"]:
        return {"decision": "quality_valid_speed_uncertain", **disposition}
    metrics = protected_search_metrics(candidate)
    for prior in archive:
        if prior.get("inference_result", {}).get("inference_status") != "stable":
            continue
        prior_metrics = protected_search_metrics(prior)
        candidate_interval = _speed_interval(candidate["inference_result"], metrics["normalized_decode_throughput"])
        prior_interval = _speed_interval(prior["inference_result"], prior_metrics["normalized_decode_throughput"])
        relation = speed_relation(
            prior_metrics["normalized_decode_throughput"], prior_interval,
            metrics["normalized_decode_throughput"], candidate_interval,
        )
        speed_no_worse = relation["relation"] != "slower"
        strict_speed = relation["relation"] == "faster"
        quality_no_worse = prior_metrics["validation_bpb"] <= metrics["validation_bpb"]
        memory_no_worse = prior_metrics["peak_allocated_inference_memory_bytes"] <= metrics["peak_allocated_inference_memory_bytes"]
        strict_other = (
            prior_metrics["validation_bpb"] < metrics["validation_bpb"]
            or prior_metrics["peak_allocated_inference_memory_bytes"] < metrics["peak_allocated_inference_memory_bytes"]
        )
        if quality_no_worse and memory_no_worse and speed_no_worse and (strict_other or strict_speed):
            return {
                "decision": "dominated",
                "reason": "dominated by registered same-campaign archive member",
                "speed_relation": relation,
                **disposition,
                "quality_status": "dominated_valid",
                "parent_eligible": False,
                "pareto_eligible": False,
            }
    return {
        "decision": "retained",
        "reason": "non-dominated within baseline plus this campaign",
        **disposition,
    }


def reconcile_same_arm_archive(
    *, baseline: dict[str, Any], evidence: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Recompute the derived Pareto frontier and parent pool from full evidence.

    The durable evidence is append-only.  Frontier and parent eligibility are
    derived views, so a later candidate that dominates an earlier one removes
    that earlier candidate from both views without deleting its lineage.
    """
    complete = [record for record in evidence if record.get("status") == "completed"]
    reconciled: list[dict[str, Any]] = []
    for record in evidence:
        updated = dict(record)
        # Campaign views expose inference to the Scientist under ``inference``;
        # controller arithmetic uses ``inference_result``.  A persisted view is
        # subsequently read back through this function, so normalize that one
        # deliberate presentation boundary before recomputing eligibility.
        if "inference_result" not in updated and isinstance(updated.get("inference"), dict):
            updated["inference_result"] = updated["inference"]
        if record.get("status") == "completed":
            peers = [baseline]
            for other in complete:
                if other is record:
                    continue
                peer = dict(other)
                if "inference_result" not in peer and isinstance(peer.get("inference"), dict):
                    peer["inference_result"] = peer["inference"]
                peers.append(peer)
            decision = decide_candidate(updated, peers)
            updated["pareto_decision"] = decision
            updated["quality_status"] = decision["quality_status"]
            updated["parent_eligible"] = bool(decision["parent_eligible"])
            updated["pareto_eligible"] = bool(decision["pareto_eligible"])
        else:
            updated["parent_eligible"] = False
            updated["pareto_eligible"] = False
        reconciled.append(updated)
    frontier = [
        record for record in reconciled
        if record.get("pareto_eligible") and (record.get("pareto_decision") or {}).get("decision") == "retained"
    ]
    parent_pool = [record for record in reconciled if record.get("parent_eligible")]
    return reconciled, frontier, parent_pool
