#!/usr/bin/env python3
"""Apply one audited, non-scientific control-plane amendment to a stopped campaign.

This tool is intentionally narrow: it can requeue only a position that failed
before remote execution, while preserving the completed prefix and every failure
receipt.  It never edits candidate sources, histories, Pareto decisions, resource
contracts, or scientific measurements.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[3]
SCHEMA = "autopareto_control_plane_amendment_v1"
NEW_RUNTIME_PATHS = (
    "study/contracts_2026/scientist_parent_selection_schema.json",
    "study/contracts_2026/scientist_child_edit_schema.json",
)


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json_write(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def amend(
    *,
    prior_manifest: pathlib.Path,
    output_manifest: pathlib.Path,
    state_root: pathlib.Path,
    effective_position: int,
    remote_absence_verified_at: str,
    reason: str,
    scientist_timeout_seconds: float | None = None,
    retry_existing_execution: bool = False,
) -> dict[str, Any]:
    prior_manifest = prior_manifest.resolve(strict=True)
    output_manifest = output_manifest.resolve()
    state_root = state_root.resolve(strict=True)
    state_path = state_root / "controller_state.json"
    if output_manifest.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {output_manifest}")
    if effective_position < 1:
        raise ValueError("effective position must be positive")
    prior_sha = _sha256(prior_manifest)
    state = _load_json(state_path)
    if state.get("manifest_sha256") != prior_sha:
        raise RuntimeError("state does not reference the supplied prior manifest")
    positions = state.get("positions")
    if not isinstance(positions, dict):
        raise RuntimeError("controller state has no position map")
    # A single-arm study (state records active_contracts) marks the other arm's
    # schedule positions "not_in_study"; they are never executed, so they neither
    # break the completed prefix nor count as future work. Two-arm states are
    # checked exactly as before.
    inactive = {"not_in_study"} if state.get("active_contracts") else set()
    for position in range(1, effective_position):
        status = positions.get(str(position), {}).get("status")
        if status in inactive:
            continue
        if status != "completed":
            raise RuntimeError(f"completed prefix is not intact at position {position}")
    target = positions.get(str(effective_position))
    if not isinstance(target, dict) or target.get("status") != "reconciliation_required":
        raise RuntimeError("target position is not a reconciliation-required pre-remote failure")
    failure = target.get("reconciliation_failure")
    execution = target.get("execution")
    if retry_existing_execution:
        if not isinstance(execution, dict) or execution.get("schema") != "autopareto_pending_execution_v1":
            raise RuntimeError("post-remote retry requires the original pending execution record")
        receipt = failure.get("receipt") if isinstance(failure, dict) else None
        if not isinstance(receipt, dict) or receipt.get("status") != "FAILED":
            raise RuntimeError("post-remote retry requires a durable FAILED receipt")
        if receipt.get("invocation_id") != execution.get("invocation_id"):
            raise RuntimeError("reconciliation receipt does not match the pending invocation")
        if execution.get("dispatch_status") != "launch_requested":
            raise RuntimeError("post-remote retry requires the original launch_requested state")
        if not isinstance(execution.get("remote_payload_dir"), str) or not execution["remote_payload_dir"]:
            raise RuntimeError("post-remote retry requires a verified remote payload directory")
    else:
        if "execution" in target:
            raise RuntimeError("target position has execution state and cannot be requeued by this tool")
        if not isinstance(failure, dict) or failure.get("phase") != "pre_remote_execution":
            raise RuntimeError("target failure is not proven pre-remote")
    for position in range(effective_position + 1, len(positions) + 1):
        status = positions.get(str(position), {}).get("status")
        if status in inactive:
            continue
        if status != "pending":
            raise RuntimeError(f"future position {position} is not pending")

    manifest = copy.deepcopy(_load_json(prior_manifest))
    if scientist_timeout_seconds is not None and scientist_timeout_seconds <= 0:
        raise ValueError("Scientist timeout must be positive")
    state_scientist_config = state.get("scientist_config")
    manifest_scientist_config = (
        manifest.get("scientist", {}).get("configuration")
        if isinstance(manifest.get("scientist"), dict)
        else None
    )
    if scientist_timeout_seconds is not None:
        if not isinstance(state_scientist_config, dict) or not isinstance(manifest_scientist_config, dict):
            raise RuntimeError("cannot amend Scientist timeout without resolved ScientistConfig in state and manifest")
        prior_state_timeout = state_scientist_config.get("timeout_seconds")
        prior_manifest_timeout = manifest_scientist_config.get("timeout_seconds")
        if prior_state_timeout != prior_manifest_timeout:
            raise RuntimeError("state and manifest Scientist timeout disagree")
        manifest_scientist_config["timeout_seconds"] = float(scientist_timeout_seconds)
    immutable = manifest.get("immutable_deployment")
    if not isinstance(immutable, dict):
        raise RuntimeError("prior manifest has no immutable deployment closure")
    original_hashes = dict(immutable)
    for relative in (*immutable, *NEW_RUNTIME_PATHS):
        source = ROOT / relative
        if source.is_symlink() or not source.is_file():
            raise FileNotFoundError(f"immutable deployment source missing: {relative}")
        immutable[relative] = _sha256(source)
    changed = {
        relative: {"before": original_hashes.get(relative), "after": immutable[relative]}
        for relative in immutable
        if original_hashes.get(relative) != immutable[relative]
    }
    allowed_changed = {
        "study/contracts_2026/isolation.py",
        "study/contracts_2026/campaign_controller.py",
        "study/contracts_2026/bridge.py",
        "study/contracts_2026/context.py",
        "protected/inference_adapter.py",
        *NEW_RUNTIME_PATHS,
    }
    if set(changed) - allowed_changed:
        raise RuntimeError(f"unexpected immutable deployment drift: {sorted(set(changed) - allowed_changed)}")
    amendment = {
        "schema": SCHEMA,
        "effective_global_position": effective_position,
        "prior_manifest_path": str(prior_manifest),
        "prior_manifest_sha256": prior_sha,
        "remote_absence_verified_at": remote_absence_verified_at,
        "reason": reason,
        "changed_immutable_paths": changed,
        "scientific_contract_changed": False,
        "scientific_components_unchanged": [
            "training", "N_STAR", "T_300s", "evaluation", "inference", "Pareto",
            "lineage", "T/N isolation", "schedule", "dataset", "tokenizer",
        ],
    }
    if retry_existing_execution:
        amendment.update({
            "retry_existing_execution": True,
            "prior_invocation_id": execution["invocation_id"],
            "remote_payload_dir": execution["remote_payload_dir"],
            "remote_result_not_accepted": True,
        })
    if scientist_timeout_seconds is not None:
        amendment["scientist_timeout_seconds"] = {
            "before": prior_manifest_timeout,
            "after": float(scientist_timeout_seconds),
        }
    prior_amendments = manifest.pop("operational_amendments", None)
    if prior_amendments is None:
        prior = manifest.pop("operational_amendment", None)
        prior_amendments = [prior] if isinstance(prior, dict) else []
    if not isinstance(prior_amendments, list) or not all(isinstance(item, dict) for item in prior_amendments):
        raise RuntimeError("manifest operational amendment history is malformed")
    manifest["operational_amendments"] = [*prior_amendments, amendment]
    manifest["manifest_sha256"] = None

    amended_state = copy.deepcopy(state)
    amended_target = amended_state["positions"][str(effective_position)]
    event = dict(amended_target.pop("reconciliation_failure"))
    event.update({
        "reconciled_by": "control_plane_amendment",
        "remote_absence_verified_at": remote_absence_verified_at,
    })
    amended_target.setdefault("non_consuming_infrastructure_events", []).append(event)
    if retry_existing_execution:
        old_execution = copy.deepcopy(amended_target["execution"])
        event["prior_execution"] = old_execution
        retry_digest = hashlib.sha256(
            f"{old_execution['invocation_id']}:{immutable['protected/inference_adapter.py']}:{effective_position}".encode()
        ).hexdigest()
        retry_execution = amended_target["execution"]
        retry_execution.update({
            "invocation_id": f"pair-a-{effective_position:03d}-retry-{retry_digest[:32]}",
            "remote_run_id": f"{old_execution['remote_run_id']}-retry-{retry_digest[:12]}",
            "output_name": f"position-{effective_position:03d}-adapterfix-{retry_digest[:12]}.json",
            "dispatch_status": "launch_requested",
            "retry_of_invocation_id": old_execution["invocation_id"],
        })
        reservation = retry_execution.get("reservation")
        if isinstance(reservation, dict):
            reservation["manifest_path"] = str(output_manifest)
            reservation["manifest_sha256"] = "pending"
            a40 = reservation.get("a40")
            if isinstance(a40, dict):
                a40["manifest_sha256"] = "pending"
        amended_target["status"] = "running"
    else:
        amended_target["status"] = "pending"
    if scientist_timeout_seconds is not None:
        amended_state["scientist_config"]["timeout_seconds"] = float(scientist_timeout_seconds)
    amended_state["manifest_sha256"] = None
    amended_state.setdefault("operational_manifest_amendments", []).append({
        "schema": SCHEMA,
        "effective_global_position": effective_position,
        "prior_manifest_sha256": prior_sha,
        "manifest_path": str(output_manifest),
        "remote_absence_verified_at": remote_absence_verified_at,
        "scientific_contract_changed": False,
        "retry_existing_execution": retry_existing_execution,
    })

    _atomic_json_write(output_manifest, manifest)
    amended_state["manifest_sha256"] = _sha256(output_manifest)
    if retry_existing_execution:
        reservation = amended_target["execution"].get("reservation")
        if isinstance(reservation, dict):
            reservation["manifest_sha256"] = amended_state["manifest_sha256"]
            a40 = reservation.get("a40")
            if isinstance(a40, dict):
                a40["manifest_sha256"] = amended_state["manifest_sha256"]
    amended_state["operational_manifest_amendments"][-1]["manifest_sha256"] = amended_state["manifest_sha256"]
    _atomic_json_write(state_path, amended_state)
    return {
        "manifest": str(output_manifest),
        "manifest_sha256": amended_state["manifest_sha256"],
        "effective_global_position": effective_position,
        "requeued_status": amended_target["status"],
        "changed_immutable_paths": sorted(changed),
        "retry_existing_execution": retry_existing_execution,
        "new_invocation_id": amended_target.get("execution", {}).get("invocation_id") if retry_existing_execution else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--output-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--state-root", type=pathlib.Path, required=True)
    parser.add_argument("--effective-position", type=int, required=True)
    parser.add_argument("--remote-absence-verified-at", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--scientist-timeout-seconds", type=float)
    parser.add_argument(
        "--retry-existing-execution",
        action="store_true",
        help="re-arm one verified post-remote harness failure with the same candidate and payload",
    )
    args = parser.parse_args(argv)
    print(json.dumps(amend(
        prior_manifest=args.prior_manifest,
        output_manifest=args.output_manifest,
        state_root=args.state_root,
        effective_position=args.effective_position,
        remote_absence_verified_at=args.remote_absence_verified_at,
        reason=args.reason,
        scientist_timeout_seconds=args.scientist_timeout_seconds,
        retry_existing_execution=args.retry_existing_execution,
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
