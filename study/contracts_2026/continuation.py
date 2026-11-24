"""Fail-closed preparation of a Pair-A campaign continuation.

This module never launches a scientific process.  It preserves a completed
campaign prefix, records an abandoned infrastructure invocation, and prepares
the exact already-selected next candidate for a replacement A40 deployment.
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import os
import pathlib
import shutil
import tempfile
from typing import Any, Mapping


STATE_SCHEMA = "autopareto_pair_a_controller_v1"
CONTINUATION_SCHEMA = "autopareto_pair_a_cross_a40_continuation_v1"
MIGRATION_SCHEMA = "autopareto_pair_a_state_migration_v1"
FROZEN_N_STAR = 88_604_672


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json_write(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        temporary = pathlib.Path(handle.name)
    os.replace(temporary, path)


def _inactive_statuses(state: Mapping[str, Any]) -> set[str]:
    """Statuses that a single-arm study gives the other arm's schedule positions.

    A single-arm state records ``active_contracts``; its inactive positions are
    ``not_in_study``, are never executed, and the inactive campaign has no state
    directory.  Two-arm states have no inactive positions and are checked as before.
    """
    return {"not_in_study"} if state.get("active_contracts") else set()


def _studied_campaign_ids(state: Mapping[str, Any]) -> list[str]:
    inactive = _inactive_statuses(state)
    positions = state.get("positions", {})
    return sorted({
        str(row["campaign_id"])
        for row in state["schedule"]
        if positions.get(str(row["global_position"]), {}).get("status") not in inactive
    })


def _regular_file_within(path: pathlib.Path, root: pathlib.Path) -> pathlib.Path:
    if path.is_symlink():
        raise RuntimeError(f"campaign artifact is missing, symlinked, or outside state root: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or not resolved.is_relative_to(root.resolve()):
        raise RuntimeError(f"campaign artifact is missing, symlinked, or outside state root: {path}")
    return resolved


def audit_completed_prefix(
    source_root: pathlib.Path,
    *,
    completed_positions: int,
    expected_next_status: str = "reconciliation_required",
) -> dict[str, Any]:
    """Verify the immutable completed prefix and abandoned next invocation."""
    source_root = source_root.resolve(strict=True)
    state_path = _regular_file_within(source_root / "controller_state.json", source_root)
    state = json.loads(state_path.read_text())
    if state.get("schema") != STATE_SCHEMA:
        raise RuntimeError("source controller state has an unexpected schema")
    positions = state.get("positions")
    schedule = state.get("schedule")
    if not isinstance(positions, dict) or not isinstance(schedule, list) or len(schedule) != 50:
        raise RuntimeError("source controller state has an invalid schedule")
    if completed_positions <= 0 or completed_positions >= len(schedule):
        raise ValueError("completed prefix must leave at least one scheduled position")

    inactive = _inactive_statuses(state)
    result_rows: list[dict[str, Any]] = []
    for row in schedule:
        global_position = int(row["global_position"])
        position = positions.get(str(global_position))
        if not isinstance(position, dict):
            raise RuntimeError(f"missing controller position {global_position}")
        if position.get("status") in inactive:
            continue
        if global_position <= completed_positions:
            if position.get("status") != "completed":
                raise RuntimeError(f"position {global_position} is not completed")
            result_file = _regular_file_within(pathlib.Path(str(position.get("result_file", ""))), source_root)
            result = json.loads(result_file.read_text())
            if (
                result.get("campaign_id") != row.get("campaign_id")
                or int(result.get("attempt", -1)) != int(row.get("campaign_attempt", -2))
                or result.get("contract_id") != row.get("contract_id")
            ):
                raise RuntimeError(f"position {global_position} result identity mismatch")
            failed = result.get("status") == "scientific_failure"
            if failed != (position.get("outcome") == "scientific_failure"):
                raise RuntimeError(f"position {global_position} outcome disagrees with its result status")
            result_rows.append({
                "global_position": global_position,
                "campaign_id": row["campaign_id"],
                "campaign_attempt": row["campaign_attempt"],
                "contract_id": row["contract_id"],
                "path": str(result_file.relative_to(source_root)),
                "bytes": result_file.stat().st_size,
                "sha256": sha256_file(result_file),
                "status": result.get("status"),
                "experiment_id": result.get("experiment_id"),
                "parent_experiment_id": result.get("parent_experiment_id"),
                "candidate_source_sha256": result.get("candidate_source_after_sha256"),
            })
        elif global_position == completed_positions + 1:
            if position.get("status") != expected_next_status:
                raise RuntimeError(
                    f"next position status is {position.get('status')!r}, expected {expected_next_status!r}"
                )
        elif position.get("status") != "pending":
            raise RuntimeError(f"future position {global_position} is not pending")

    registry_rows: list[dict[str, Any]] = []
    for campaign_id in _studied_campaign_ids(state):
        registry = _regular_file_within(source_root / campaign_id / "source_registry.jsonl", source_root)
        for line_number, line in enumerate(registry.read_text().splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            source = _regular_file_within(pathlib.Path(str(record.get("source_path", ""))), source_root)
            observed = sha256_file(source)
            if observed != record.get("source_sha256"):
                raise RuntimeError(f"source registry hash mismatch at {registry}:{line_number}")
            registry_rows.append({
                "campaign_id": campaign_id,
                "experiment_id": record.get("experiment_id"),
                "parent_experiment_id": record.get("parent_experiment_id"),
                "source_sha256": observed,
                "source_path": str(source.relative_to(source_root)),
                "parent_eligible": bool(record.get("parent_eligible")),
                "pareto_eligible": bool(record.get("pareto_eligible")),
            })

    by_experiment = {(row["campaign_id"], row["experiment_id"]): row for row in registry_rows}
    for result in result_rows:
        registered = by_experiment.get((result["campaign_id"], result["experiment_id"]))
        if result["status"] == "scientific_failure":
            # The controller keeps a candidate-caused failure in the history and the Scientist's view, but never
            # registers its source, so it can never become a parent.
            if registered is not None or result["candidate_source_sha256"] is not None:
                raise RuntimeError(f"failed position {result['global_position']} is registered as a source")
        elif registered is None or registered["source_sha256"] != result["candidate_source_sha256"]:
            raise RuntimeError(f"completed position {result['global_position']} has no matching source registry row")
        parent = by_experiment.get((result["campaign_id"], result["parent_experiment_id"]))
        if parent is None or not parent["parent_eligible"]:
            raise RuntimeError(f"completed position {result['global_position']} has an invalid parent lineage")

    next_position = copy.deepcopy(positions[str(completed_positions + 1)])
    execution = next_position.get("execution")
    if not isinstance(execution, dict) or execution.get("schema") != "autopareto_pending_execution_v1":
        raise RuntimeError("next position has no durable pending execution record")
    candidate = _regular_file_within(pathlib.Path(str(execution.get("candidate_path", ""))), source_root)
    metadata = _regular_file_within(pathlib.Path(str(execution.get("metadata_path", ""))), source_root)
    if sha256_file(candidate) != execution.get("candidate_source_after_sha256"):
        raise RuntimeError("next candidate hash mismatch")
    if sha256_file(metadata) != execution.get("metadata_sha256"):
        raise RuntimeError("next metadata file hash mismatch")
    next_parent = by_experiment.get((str(execution.get("reservation", {}).get("campaign_id")), execution.get("parent_experiment_id")))
    if (
        next_parent is None
        or not next_parent["parent_eligible"]
        or next_parent["source_sha256"] != execution.get("candidate_source_before_sha256")
    ):
        raise RuntimeError("next position parent is not an eligible exact same-arm source")

    return {
        "source_root": str(source_root),
        "controller_state_sha256": sha256_file(state_path),
        "manifest_sha256": state.get("manifest_sha256"),
        "a40_uuid": state.get("a40_uuid"),
        "completed_positions": completed_positions,
        "completed_results": result_rows,
        "source_registry": registry_rows,
        "next_position": completed_positions + 1,
        "abandoned_invocation_id": execution.get("invocation_id"),
        "next_campaign_id": execution.get("reservation", {}).get("campaign_id"),
        "next_campaign_attempt": execution.get("reservation", {}).get("campaign_attempt"),
        "next_parent_experiment_id": execution.get("parent_experiment_id"),
        "next_parent_source_sha256": execution.get("candidate_source_before_sha256"),
        "next_candidate_sha256": execution.get("candidate_source_after_sha256"),
        "next_candidate_path": str(candidate),
        "next_metadata_path": str(metadata),
        "next_position_record": next_position,
    }


def reconcile_uncommitted_position(
    source_root: pathlib.Path,
    *,
    global_position: int,
    reason: str,
    remote_process_verified_absent: bool,
    durable_receipt_verified_absent: bool,
) -> dict[str, Any]:
    """Record an abandoned, uncommitted position without consuming it.

    This is intentionally a separate, fail-closed transition: callers must
    prove that no remote process or durable completion receipt remains before
    a prepared continuation can requeue the position.
    """
    if not remote_process_verified_absent or not durable_receipt_verified_absent:
        raise RuntimeError("cannot reconcile a position without both remote absence proofs")
    source_root = source_root.resolve(strict=True)
    state_path = _regular_file_within(source_root / "controller_state.json", source_root)
    state = json.loads(state_path.read_text())
    position = state.get("positions", {}).get(str(global_position))
    if not isinstance(position, dict) or position.get("status") != "running":
        raise RuntimeError("only an uncommitted running position may be reconciled")
    if position.get("result_file") is not None:
        raise RuntimeError("position has a local result and cannot be requeued")
    execution = position.get("execution")
    if not isinstance(execution, dict) or execution.get("schema") != "autopareto_pending_execution_v1":
        raise RuntimeError("running position has no durable pending execution record")
    position["status"] = "reconciliation_required"
    position["reconciliation_failure"] = {
        "type": "infrastructure_ambiguity_reconciled_for_continuation",
        "phase": "pre_commit",
        "message": reason,
        "remote_process_verified_absent": True,
        "durable_receipt_verified_absent": True,
        "scientific_position_consumed": False,
        "invocation_id": execution.get("invocation_id"),
    }
    _atomic_json_write(state_path, state)
    return {
        "status": "reconciliation_required",
        "global_position": global_position,
        "invocation_id": execution.get("invocation_id"),
        "controller_state_sha256": sha256_file(state_path),
    }


def build_continuation_manifest(
    *,
    prior_manifest_path: pathlib.Path,
    source_state_root: pathlib.Path,
    completed_positions: int,
    repository_root: pathlib.Path,
    new_gpu_uuid: str,
    new_ssh_host: str,
    new_local_state_root: str,
    new_remote_state_root: str,
    registered_cache: str,
    remote_python: str,
    study_venv: str,
    paired_baseline_checkpoints: Mapping[str, Mapping[str, Any]],
    calibration_result: Mapping[str, Any],
    calibration_path: str,
    calibration_sha256: str,
    registered_cache_runtime_provenance: Mapping[str, Any],
    extra_immutable_paths: tuple[str, ...] = (),
    amendment_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a manifest that explicitly links the inherited and replacement-A40 epochs.

    ``amendment_record`` describes a continuation that is not a hardware move (for example a reviewed
    protected-harness fix on the same A40). See :func:`validate_amendment_record`.
    """
    if amendment_record is not None:
        validate_amendment_record(amendment_record)
    prior_manifest_path = prior_manifest_path.resolve(strict=True)
    repository_root = repository_root.resolve(strict=True)
    audit = audit_completed_prefix(source_state_root, completed_positions=completed_positions)
    prior = json.loads(prior_manifest_path.read_text())
    prior_n_star = (
        prior.get("a40", {}).get("calibration", {}).get("n_star")
        if isinstance(prior.get("a40"), dict)
        and isinstance(prior.get("a40", {}).get("calibration"), dict)
        else None
    )
    frozen_n_star = prior_n_star if isinstance(prior_n_star, int) and prior_n_star > 0 else FROZEN_N_STAR
    choice = calibration_result.get("choice")
    if not isinstance(choice, dict):
        raise RuntimeError("replacement-A40 calibration has no valid choice")
    calibrated_n_star = choice.get("n_star")
    token_quantum = choice.get("token_quantum")
    if not isinstance(calibrated_n_star, int) or not isinstance(token_quantum, int) or token_quantum <= 0:
        raise RuntimeError("replacement-A40 calibration has invalid token accounting")
    if frozen_n_star % token_quantum != 0 or calibrated_n_star % token_quantum != 0:
        raise RuntimeError("replacement-A40 calibration is not aligned to the legal token quantum")
    calibration_delta = calibrated_n_star - frozen_n_star
    if abs(calibration_delta) > token_quantum:
        raise RuntimeError("replacement-A40 calibration differs from frozen N_STAR by more than one legal update")
    source_runs = choice.get("source_runs")
    if not isinstance(source_runs, list) or len(source_runs) != 3:
        raise RuntimeError("replacement-A40 calibration must contain exactly three source runs")
    observed_tokens: list[int] = []
    for run in source_runs:
        if (
            not isinstance(run, dict)
            or run.get("error") is not None
            or not isinstance(run.get("total_tokens"), int)
            or run.get("total_tokens") <= 0
            or not isinstance(run.get("optimizer_updates"), int)
            or not isinstance(run.get("measured_training_seconds"), (int, float))
        ):
            raise RuntimeError("replacement-A40 calibration contains an invalid source run")
        observed_tokens.append(int(run["total_tokens"]))
    # A valid three-run calibration freezes the floor of the median observed
    # count.  Individual successful runs may differ by a legal update quantum
    # (the predecessor evidence contains exactly that case).
    if (sorted(observed_tokens)[1] // token_quantum) * token_quantum != calibrated_n_star:
        raise RuntimeError("replacement-A40 calibration choice does not match the median source run")

    manifest = copy.deepcopy(prior)
    manifest["schema"] = CONTINUATION_SCHEMA
    manifest["frozen_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    manifest["a40"] = {
        **dict(manifest.get("a40", {})),
        "gpu_model": "NVIDIA A40",
        "gpu_uuid": new_gpu_uuid,
        "ssh_host": new_ssh_host,
        "trusted_repository": "/opt/autopareto_trusted",
        "rootfs_runtime": remote_python,
        "study_venv": study_venv,
        "paired_baseline_checkpoints": {
            arm: dict(value) for arm, value in sorted(paired_baseline_checkpoints.items())
        },
    }
    manifest.setdefault("runtime", {})["python"] = remote_python
    manifest["runtime"]["study_venv"] = study_venv
    persistent_paths = manifest.setdefault("persistent_paths", {})
    persistent_paths.update({
        "local_official_state": new_local_state_root,
        "registered_cache": registered_cache,
        "trusted_state_root": new_remote_state_root,
        "candidate_workspace_root": "/opt/autopareto_candidate_contexts",
        "candidate_attempt_root": "/opt/autopareto-candidate-attempts",
        "candidate_output_root": "/var/lib/autopareto/candidate_attempts",
        "candidate_runtime_root": "/opt/autopareto_candidate_runtime",
        "candidate_input_root": "/opt/autopareto_candidate_inputs/per_attempts",
    })
    manifest.setdefault("registered_artifacts", {})["registered_cache"] = registered_cache
    manifest["registered_artifacts"]["runtime_cache_closure"] = copy.deepcopy(
        dict(registered_cache_runtime_provenance)
    )

    immutable = dict(manifest.get("immutable_deployment", {}))
    for relative in sorted(set(immutable) | set(extra_immutable_paths)):
        path = repository_root / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"immutable deployment file is missing: {relative}")
        immutable[relative] = sha256_file(path)
    manifest["immutable_deployment"] = dict(sorted(immutable.items()))
    _sync_pinned_hashes(manifest, immutable, amendment_record)

    manifest["continuation"] = {
        "schema": MIGRATION_SCHEMA,
        "predecessor_manifest": {
            "path": str(prior_manifest_path),
            "sha256": sha256_file(prior_manifest_path),
        },
        "predecessor_controller_state": {
            "path": str(source_state_root.resolve()),
            "sha256": audit["controller_state_sha256"],
        },
        "hardware_transition_after_global_position": completed_positions,
        "old_gpu_uuid": audit["a40_uuid"],
        "new_gpu_uuid": new_gpu_uuid,
        "inherited_completed_positions": audit["completed_results"],
        "inherited_source_registry": audit["source_registry"],
        "abandoned_unconsumed_invocation": {
            "global_position": audit["next_position"],
            "invocation_id": audit["abandoned_invocation_id"],
            "candidate_source_sha256": audit["next_candidate_sha256"],
            "parent_experiment_id": audit["next_parent_experiment_id"],
            "parent_source_sha256": audit["next_parent_source_sha256"],
            "reason": "source pod lost its GPU/rootfs durable state before a verifiable result was committed",
            "remote_process_verified_absent": True,
            "durable_receipt_verified_absent": True,
            "scientific_position_consumed": False,
        },
        "replacement_a40_calibration": {
            "path": calibration_path,
            "sha256": calibration_sha256,
            "choice": copy.deepcopy(choice),
            "calibrated_n_star": calibrated_n_star,
            "frozen_continuation_n_star": frozen_n_star,
            "delta_tokens": calibration_delta,
            "delta_legal_updates": calibration_delta // token_quantum,
            "decision": "preserve the pre-registered N_STAR across the hardware epoch; do not redefine the scientific contract mid-study",
        },
        "scientific_closure_changed": False,
        "operational_closure_refrozen": True,
        "next_global_position": completed_positions + 1,
    }
    if amendment_record is not None:
        continuation = manifest["continuation"]
        continuation["transition_kind"] = amendment_record["transition_kind"]
        continuation["amendments"] = copy.deepcopy(list(amendment_record["amendments"]))
        continuation["abandoned_unconsumed_invocation"].update(
            copy.deepcopy(dict(amendment_record.get("abandoned_invocation", {})))
        )
        if "scientific_closure_changed" in amendment_record:
            continuation["scientific_closure_changed"] = bool(amendment_record["scientific_closure_changed"])
    manifest["official_pair_a_attempts"] = completed_positions
    manifest["scientific_execution_launched"] = True
    manifest["manifest_sha256"] = None
    return manifest


def _sync_pinned_hashes(
    manifest: dict[str, Any], immutable: Mapping[str, str], amendment_record: Mapping[str, Any] | None
) -> None:
    """Keep the manifest's other file pins (evaluator, inference config, prompts, schedule) consistent.

    A pinned file whose recomputed hash changed may only move with an amendment record that lists it;
    otherwise the continuation fails here instead of producing a manifest the controller rejects.
    """
    inference = manifest.get("inference") if isinstance(manifest.get("inference"), dict) else {}
    prompt = inference.get("protected_prompt_artifact") if isinstance(inference.get("protected_prompt_artifact"), dict) else {}
    schedule = manifest.get("schedule") if isinstance(manifest.get("schedule"), dict) else {}
    pins = ((inference, "evaluator", "evaluator_sha256"), (inference, "active_config", "active_config_sha256"),
            (prompt, "path", "sha256"), (schedule, "path", "sha256"))
    amended = {str(item.get("file")) for item in (amendment_record or {}).get("amendments", [])}
    for container, path_key, hash_key in pins:
        relative = container.get(path_key)
        if relative not in immutable or container.get(hash_key) == immutable[relative]:
            continue
        if relative not in amended:
            raise RuntimeError(f"{relative} changed but is pinned as {hash_key}; only a listed amendment may change it")
        container[hash_key] = immutable[relative]


_AMENDABLE_ABANDONMENT_FIELDS = frozenset({
    "reason", "remote_process_verified_absent", "durable_receipt_verified_absent", "durable_receipt",
})


def validate_amendment_record(record: Mapping[str, Any]) -> None:
    """Check a continuation amendment record: what changed, and the true state of the abandoned invocation."""
    if not isinstance(record.get("transition_kind"), str) or not record["transition_kind"]:
        raise ValueError("amendment record needs a transition_kind")
    amendments = record.get("amendments")
    if not isinstance(amendments, list) or not amendments or not all(isinstance(a, dict) for a in amendments):
        raise ValueError("amendment record must list its amendments")
    abandoned = record.get("abandoned_invocation", {})
    if not isinstance(abandoned, dict) or set(abandoned) - _AMENDABLE_ABANDONMENT_FIELDS:
        raise ValueError(f"abandoned_invocation may only set {sorted(_AMENDABLE_ABANDONMENT_FIELDS)}")


def prepare_continuation_state(
    *,
    source_root: pathlib.Path,
    destination_root: pathlib.Path,
    manifest_source: pathlib.Path,
    manifest_relative_path: pathlib.PurePosixPath,
    completed_positions: int,
    new_gpu_uuid: str,
    new_remote_state_root: str,
    paired_baseline_n_checkpoint: str,
    paired_baseline_t_checkpoint: str | None = None,
    evaluator_hashes: Mapping[str, str],
    remote_run_namespace: str,
    abandoned_invocation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Copy the completed prefix and prepare the exact old next candidate.

    The destination is created atomically.  The returned position remains in
    ``prepared`` dispatch state until :func:`arm_continuation_payload` performs
    a SHA-verified transfer.
    """
    if abandoned_invocation is not None and set(abandoned_invocation) - _AMENDABLE_ABANDONMENT_FIELDS:
        raise ValueError(f"abandoned_invocation may only set {sorted(_AMENDABLE_ABANDONMENT_FIELDS)}")
    source_root = source_root.resolve(strict=True)
    destination_root = destination_root.resolve()
    if destination_root.exists():
        raise FileExistsError(f"continuation destination already exists: {destination_root}")
    if destination_root.is_relative_to(source_root) or source_root.is_relative_to(destination_root):
        raise ValueError("continuation destination must be separate from source state")
    audit = audit_completed_prefix(source_root, completed_positions=completed_positions)
    manifest_source = manifest_source.resolve(strict=True)
    manifest_sha = sha256_file(manifest_source)
    old_position = audit["next_position_record"]
    old_execution = old_position["execution"]
    old_metadata = json.loads(pathlib.Path(audit["next_metadata_path"]).read_text())
    old_candidate = pathlib.Path(audit["next_candidate_path"])
    next_position = completed_positions + 1

    staging = destination_root.parent / f".{destination_root.name}.staging-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"continuation staging path already exists: {staging}")
    try:
        shutil.copytree(source_root, staging, symlinks=False)
        final_manifest = staging / pathlib.Path(manifest_relative_path.as_posix())
        final_manifest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_source, final_manifest)

        state_path = staging / "controller_state.json"
        state = json.loads(state_path.read_text())
        state["a40_uuid"] = new_gpu_uuid
        state["manifest_sha256"] = manifest_sha
        state["continuation"] = {
            "schema": MIGRATION_SCHEMA,
            "source_state_root": str(source_root),
            "source_controller_state_sha256": audit["controller_state_sha256"],
            "inherited_completed_positions": completed_positions,
            "old_gpu_uuid": audit["a40_uuid"],
            "new_gpu_uuid": new_gpu_uuid,
            "new_remote_state_root": new_remote_state_root,
            "manifest": str(destination_root / manifest_relative_path),
            "manifest_sha256": manifest_sha,
            "next_global_position": next_position,
        }

        inactive = _inactive_statuses(state)
        for index in range(1, completed_positions + 1):
            position = state["positions"][str(index)]
            if position.get("status") in inactive:
                continue
            old_result = pathlib.Path(position["result_file"]).resolve()
            position["result_file"] = str(destination_root / old_result.relative_to(source_root))

        for campaign_id in _studied_campaign_ids(state):
            registry_path = staging / campaign_id / "source_registry.jsonl"
            rewritten: list[str] = []
            for line in registry_path.read_text().splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                old_source = pathlib.Path(str(record["source_path"])).resolve()
                record["source_path"] = str(destination_root / old_source.relative_to(source_root))
                rewritten.append(json.dumps(record, sort_keys=True, separators=(",", ":")))
            registry_path.write_text("\n".join(rewritten) + "\n")

        # A chained continuation inherits its predecessor's migration records; keep them, never overwrite one.
        migration_dir = staging / "migration"
        migration_dir.mkdir(exist_ok=True)
        if (migration_dir / f"abandoned_position_{next_position:03d}.json").exists():
            raise FileExistsError(f"a migration record for position {next_position} already exists")
        _atomic_json_write(migration_dir / f"abandoned_position_{next_position:03d}.json", {
            "schema": MIGRATION_SCHEMA,
            "global_position": next_position,
            "old_position_record": old_position,
            "old_candidate_sha256": audit["next_candidate_sha256"],
            "old_metadata_file_sha256": old_execution["metadata_sha256"],
            "scientific_position_consumed": False,
            "remote_process_verified_absent": True,
            "durable_receipt_verified_absent": True,
            **copy.deepcopy(dict(abandoned_invocation or {})),
        })

        pending = staging / str(old_execution["reservation"]["campaign_id"]) / "pending" / f"position-{next_position:03d}"
        if pending.exists():
            shutil.rmtree(pending)
        pending.mkdir(parents=True)
        candidate_path = pending / "train.py"
        shutil.copy2(old_candidate, candidate_path)

        metadata = copy.deepcopy(old_metadata)
        metadata["evaluator_hashes"] = dict(sorted(evaluator_hashes.items()))
        scheduled_contract = str(state["schedule"][next_position - 1]["contract_id"])
        if scheduled_contract == "T":
            if paired_baseline_t_checkpoint is None:
                raise RuntimeError("continuation T position requires a paired T baseline checkpoint")
            paired_baseline_checkpoint = paired_baseline_t_checkpoint
        elif scheduled_contract == "N":
            paired_baseline_checkpoint = paired_baseline_n_checkpoint
        else:
            raise RuntimeError(f"unknown continuation contract: {scheduled_contract}")
        metadata["contract_parameters"]["paired_baseline_checkpoint"] = paired_baseline_checkpoint
        metadata.pop("metadata_sha256", None)
        metadata["metadata_sha256"] = _json_hash(metadata)
        metadata_path = pending / "registered_metadata.json"
        _atomic_json_write(metadata_path, metadata)
        candidate_sha = sha256_file(candidate_path)
        metadata_file_sha = sha256_file(metadata_path)
        if candidate_sha != audit["next_candidate_sha256"]:
            raise RuntimeError("continuation candidate bytes changed during staging")

        parent_id = str(audit["next_parent_experiment_id"])
        campaign_id = str(old_execution["reservation"]["campaign_id"])
        registry = [
            json.loads(line) for line in (staging / campaign_id / "source_registry.jsonl").read_text().splitlines()
            if line.strip()
        ]
        parent_rows = [row for row in registry if row.get("experiment_id") == parent_id]
        if len(parent_rows) != 1 or parent_rows[0].get("source_sha256") != audit["next_parent_source_sha256"]:
            raise RuntimeError("continuation parent registry identity mismatch")

        row = state["schedule"][next_position - 1]
        invocation_digest = hashlib.sha256(
            f"{state['study_id']}:{next_position}:{campaign_id}:{candidate_sha}:{metadata_file_sha}".encode()
        ).hexdigest()
        invocation_id = f"pair-a-{next_position:03d}-{invocation_digest[:32]}"
        reservation = copy.deepcopy(old_execution["reservation"])
        reservation.update({
            "status": "running",
            "a40": {"gpu_uuid": new_gpu_uuid, "manifest_sha256": manifest_sha},
            "manifest_path": str(destination_root / manifest_relative_path),
            "manifest_sha256": manifest_sha,
            "parent_source_path": str(destination_root / campaign_id / "sources" / pathlib.Path(parent_rows[0]["source_path"]).name),
        })
        execution = {
            "schema": "autopareto_pending_execution_v1",
            "remote_run_id": f"{remote_run_namespace}-position-{next_position:03d}-{campaign_id}",
            "output_name": f"position-{next_position:03d}-{campaign_id}.json",
            "invocation_id": invocation_id,
            "parent_experiment_id": parent_id,
            "parent_source_path": parent_rows[0]["source_path"],
            "candidate_path": str(destination_root / campaign_id / "pending" / f"position-{next_position:03d}" / "train.py"),
            "metadata_path": str(destination_root / campaign_id / "pending" / f"position-{next_position:03d}" / "registered_metadata.json"),
            "candidate_source_before_sha256": audit["next_parent_source_sha256"],
            "candidate_source_after_sha256": candidate_sha,
            "metadata_sha256": metadata_file_sha,
            "reservation": reservation,
            "scientist": copy.deepcopy(old_execution["scientist"]),
            "dispatch_status": "prepared",
        }
        state["positions"][str(next_position)] = {
            "status": "running",
            "campaign_id": row["campaign_id"],
            "campaign_attempt": row["campaign_attempt"],
            "execution": execution,
        }
        for index in range(next_position + 1, 51):
            if state["positions"][str(index)].get("status") in inactive:
                continue
            if state["positions"][str(index)].get("status") != "pending":
                raise RuntimeError(f"future position {index} changed during continuation preparation")
        _atomic_json_write(state_path, state)
        os.replace(staging, destination_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "status": "prepared",
        "destination_root": str(destination_root),
        "manifest": str(destination_root / manifest_relative_path),
        "manifest_sha256": manifest_sha,
        "completed_positions": completed_positions,
        "next_global_position": next_position,
        "invocation_id": invocation_id,
        "remote_run_id": execution["remote_run_id"],
        "candidate_sha256": candidate_sha,
        "metadata_sha256": metadata_file_sha,
    }


def arm_continuation_payload(*, state_root: pathlib.Path, transport: Any) -> dict[str, Any]:
    """Transfer the prepared candidate and atomically arm it for durable launch."""
    state_root = state_root.resolve(strict=True)
    state_path = state_root / "controller_state.json"
    state = json.loads(state_path.read_text())
    finished = {"completed"} | _inactive_statuses(state)
    incomplete = [row for row in state["schedule"] if state["positions"][str(row["global_position"])]["status"] not in finished]
    if not incomplete:
        raise RuntimeError("continuation campaign is already complete")
    row = incomplete[0]
    position = state["positions"][str(row["global_position"])]
    if position.get("status") != "running" or not isinstance(position.get("execution"), dict):
        raise RuntimeError("next continuation position is not a prepared running position")
    execution = position["execution"]
    pending = state_root / row["campaign_id"] / "pending" / f"position-{int(row['global_position']):03d}"
    if execution.get("dispatch_status") == "launch_requested":
        observed = transport.verify_payload(
            remote_dir=str(execution["remote_payload_dir"]),
            candidate=pending / "train.py",
            metadata=pending / "registered_metadata.json",
        )
        return {"status": "already_armed", "execution": execution, "remote_hashes": observed}
    if execution.get("dispatch_status") != "prepared":
        raise RuntimeError("continuation payload has an invalid dispatch state")
    remote_dir = transport.transfer(payload=pending, remote_run_id=str(execution["remote_run_id"]))
    observed = transport.verify_payload(
        remote_dir=remote_dir,
        candidate=pending / "train.py",
        metadata=pending / "registered_metadata.json",
    )
    execution["remote_payload_dir"] = remote_dir
    execution["dispatch_status"] = "launch_requested"
    _atomic_json_write(state_path, state)
    return {"status": "armed", "execution": execution, "remote_hashes": observed}
