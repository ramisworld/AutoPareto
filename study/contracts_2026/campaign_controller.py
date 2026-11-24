"""Thin production controller for the frozen Pair-A study.

This module only composes the registered Scientist bridge and A40 transport.  It
does not own any scientific policy: contracts, evaluation, warm-up, inference,
and Pareto arithmetic remain in the frozen study modules.

The controller deliberately treats a position as single-use.  It writes a
``running`` reservation before invoking anything external and never retries a
reserved or failed position automatically.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import pathlib
import re
import shutil
import stat
import sys
import tempfile
from typing import Any, Mapping

from .bridge import (
    A40RemoteConfig,
    A40Transport,
    build_registered_metadata,
    run_local_scientist,
    sha256_file,
    stage_remote_payload,
    update_campaign_view,
)
from .candidate_process import validate_candidate_contract_api
from .campaign import decide_candidate, protected_search_metrics
from .context import CAMPAIGN_VIEW_SCHEMA, build_campaign_view
from .contracts import ContractId, ResourceContract, json_contract
from .isolation import (
    AGENT_VISIBLE_PATHS,
    AGENT_WRITABLE_PATHS,
    CLAUDE_SCIENTIST_TOOLS,
    SCIENTIST_CHILD_EDIT_SCHEMA,
    SCIENTIST_PARENT_SELECTION_SCHEMA,
    claude_output_schema,
    stage_agent_context,
    validate_context_paths,
)
from .scheduling import SCHEDULE_SEED, ScheduledAttempt, build_schedule, validate_schedule
from .scientist_config import DEFAULT_SCIENTIST_BIN, ScientistConfig, resolve_scientist_config


ROOT = pathlib.Path(__file__).resolve().parents[2]
STUDY_ROOT = ROOT / "study" / "contracts_2026"
BASELINE_SOURCE = STUDY_ROOT / "historical_reference" / "baseline_source" / "train.py"
PROTECTED_PREPARE = STUDY_ROOT / "historical_reference" / "baseline_source" / "prepare.py"
STUDY_RUNTIME = STUDY_ROOT / "study_runtime.py"
SCHEDULE_PATH = STUDY_ROOT / "provenance" / "pair_a_interleaved_same_index_schedule_20260902.json"
DATA_MANIFEST_PATH = STUDY_ROOT / "provenance" / "data_manifest.json"

STUDY_ID = "autopareto-contracts-2026-pair-a"
TRAINING_SEED = 3
N_STAR = 85_458_944
A40_UUID = "GPU-98579d60-2f4d-8840-cadb-847d43886326"
FINAL_MANIFEST_PATH = ROOT / "runs" / "pair-a-official-20260914-new-a40" / "provenance" / "final_pre_study_manifest_pair_a_20260914_new_a40.json"
DEFAULT_BASELINE_RECORD_PATH = ROOT / "runs" / "pair-a-seed3-baseline-registration.json"
DEFAULT_A40_HOST = "<pod-id>-<suffix>@ssh.runpod.io"
DEFAULT_A40_IDENTITY = "~/.ssh/id_ed25519"
DEFAULT_A40_REPOSITORY = "/opt/autopareto_trusted"
FROZEN_SCHEDULE_SHA256 = "ccd498d8697b35ebfbd63482b5a7b55cde08324abfce59f8f9f8f24dbeea1233"

CAMPAIGNS: dict[str, dict[str, Any]] = {
    "T": {
        "campaign_id": "pair-a-fixed-time-seed3",
        "program": STUDY_ROOT / "programs" / "program_fixed_time.md",
        "contract": ResourceContract(ContractId.T),
        "history_name": "history.jsonl",
    },
    "N": {
        "campaign_id": "pair-a-fixed-tokens-seed3",
        "program": STUDY_ROOT / "programs" / "program_fixed_tokens.md",
        "contract": ResourceContract(ContractId.N, time_seconds=None, n_star=N_STAR),
        "history_name": "history.jsonl",
    },
}


def _atomic_json_write(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _schedule_from_frozen_file() -> list[ScheduledAttempt]:
    if not SCHEDULE_PATH.is_file():
        raise FileNotFoundError(f"frozen schedule is missing: {SCHEDULE_PATH}")
    if sha256_file(SCHEDULE_PATH) != FROZEN_SCHEDULE_SHA256:
        raise RuntimeError("frozen schedule hash does not match the registered schedule")
    payload = _load_json(SCHEDULE_PATH)
    rows = [
        ScheduledAttempt(
            int(row["global_position"]),
            str(row["contract_id"]),
            str(row["campaign_id"]),
            int(row["campaign_attempt"]),
        )
        for row in payload.get("blocks", [])
    ]
    validate_schedule(rows)
    generated = build_schedule(seed=SCHEDULE_SEED)
    if [row.as_dict() for row in rows] != [row.as_dict() for row in generated]:
        raise RuntimeError("frozen schedule file disagrees with the registered schedule implementation")
    return rows


def _campaign_for(row: ScheduledAttempt) -> dict[str, Any]:
    try:
        campaign = CAMPAIGNS[row.contract_id]
    except KeyError as exc:
        raise RuntimeError(f"unregistered contract in schedule: {row.contract_id}") from exc
    if campaign["campaign_id"] != row.campaign_id:
        raise RuntimeError("schedule campaign identity does not match controller registration")
    return campaign


def _manifest_a40_uuid(payload: Mapping[str, Any]) -> str:
    """Return the A40 identity frozen by a manifest.

    Continuation manifests store the identity under ``a40.gpu_uuid``.  The
    legacy top-level form is accepted only for the small fixture manifests
    used by unit tests; production manifests must use the nested form.
    """
    a40 = payload.get("a40")
    if isinstance(a40, Mapping) and isinstance(a40.get("gpu_uuid"), str):
        value = str(a40["gpu_uuid"])
    else:
        value = str(payload.get("a40_uuid", ""))
    if not re.fullmatch(r"GPU-[A-Za-z0-9-]+", value):
        raise RuntimeError("manifest is missing a valid frozen A40 GPU UUID")
    return value


def _manifest_contains_expected_identity(payload: Any) -> bool:
    """Accept the registered manifest's schema while checking its frozen identity."""
    if not isinstance(payload, Mapping):
        return False
    try:
        _manifest_a40_uuid(payload)
    except RuntimeError:
        return False
    contracts = payload.get("contracts")
    return isinstance(contracts, Mapping) and contracts.get("N", {}).get("active_n_star") == N_STAR


def _manifest_hashes(payload: Mapping[str, Any]) -> dict[str, str]:
    """Return the complete manifest-relative immutable deployment hash set."""
    entries: dict[str, str] = {}

    def add(relative: Any, digest: Any) -> None:
        path = pathlib.PurePosixPath(str(relative))
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"manifest deployment path is not relative: {relative}")
        value = str(digest)
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise RuntimeError(f"manifest deployment hash is malformed: {relative}")
        normalized = path.as_posix()
        previous = entries.get(normalized)
        if previous is not None and previous != value:
            raise RuntimeError(f"manifest contains conflicting hashes for {normalized}")
        entries[normalized] = value

    deployment = payload.get("immutable_deployment")
    if not isinstance(deployment, Mapping):
        raise RuntimeError("manifest is missing immutable deployment hashes")
    for relative, digest in deployment.items():
        add(relative, digest)

    inference = payload.get("inference")
    if not isinstance(inference, Mapping):
        raise RuntimeError("manifest is missing inference provenance")
    add(inference.get("evaluator"), inference.get("evaluator_sha256"))
    add(inference.get("active_config"), inference.get("active_config_sha256"))
    prompt = inference.get("protected_prompt_artifact")
    if not isinstance(prompt, Mapping):
        raise RuntimeError("manifest is missing protected prompt provenance")
    add(prompt.get("path"), prompt.get("sha256"))

    schedule = payload.get("schedule")
    if not isinstance(schedule, Mapping):
        raise RuntimeError("manifest is missing schedule provenance")
    add(schedule.get("path"), schedule.get("sha256"))

    baseline = payload.get("registered_artifacts", {}).get("baseline_registration")
    if not isinstance(baseline, Mapping):
        raise RuntimeError("manifest is missing baseline provenance")
    add("study/contracts_2026/historical_reference/baseline_source/train.py", baseline.get("source_sha256"))

    required = {
        "study/contracts_2026/bridge.py",
        "study/contracts_2026/benchmark_policy.py",
        "study/contracts_2026/batching.py",
        "study/contracts_2026/campaign.py",
        "study/contracts_2026/candidate_process.py",
        "study/contracts_2026/candidate_sandbox.py",
        "study/contracts_2026/campaign_controller.py",
        "study/contracts_2026/contracts.py",
        "study/contracts_2026/correctness_policy.py",
        "study/contracts_2026/context.py",
        "study/contracts_2026/inference_ipc.py",
        "study/contracts_2026/inference_model_host.py",
        "study/contracts_2026/historical_reference/baseline_source/prepare.py",
        "study/contracts_2026/historical_reference/baseline_source/train.py",
        "study/contracts_2026/historical_reference/benchmark_config.py",
        "study/contracts_2026/inference_benchmark.py",
        "study/contracts_2026/isolation.py",
        "study/contracts_2026/provenance/pair_a_active_inference_config_20260904.json",
        "study/contracts_2026/provenance/data_manifest.json",
        "study/contracts_2026/provenance/pair_a_interleaved_same_index_schedule_20260902.json",
        "study/contracts_2026/provenance/pair_a_protected_prompt_tokens_v2.json",
        "study/contracts_2026/programs/program_fixed_time.md",
        "study/contracts_2026/programs/program_fixed_tokens.md",
        "study/contracts_2026/remote_runner.py",
        "study/contracts_2026/scientist_config.py",
        "study/contracts_2026/scheduling.py",
        "study/contracts_2026/study_runtime.py",
        "study/contracts_2026/train_launcher.py",
        "protected/inference_adapter.py",
        "protected/phase1_runtime.py",
        "event_log.py",
    }
    missing = sorted(required - set(entries))
    if missing:
        raise RuntimeError(f"manifest immutable deployment is incomplete: {missing}")
    return entries


def _verify_manifest_hashes(payload: Mapping[str, Any]) -> dict[str, str]:
    entries = _manifest_hashes(payload)
    mismatches: dict[str, dict[str, str]] = {}
    for relative, expected in entries.items():
        path = ROOT / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"manifest deployment file is missing or not regular: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            mismatches[relative] = {"expected": expected, "actual": actual}
    if mismatches:
        raise RuntimeError(f"live Mac deployment hashes do not match manifest: {mismatches}")
    return entries


def _manifest_scientist_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    scientist = payload.get("scientist")
    if not isinstance(scientist, Mapping):
        raise RuntimeError("manifest is missing Scientist configuration")
    configuration = scientist.get("configuration")
    if not isinstance(configuration, Mapping):
        raise RuntimeError("manifest is missing resolved ScientistConfig")
    required = {
        "model", "reasoning", "scientist_bin", "codex_version", "execution_backend",
        "container_engine", "container_image", "container_image_digest", "sandbox_mode",
        "approval_policy", "timeout_seconds",
    }
    missing = sorted(required - set(configuration))
    if missing:
        raise RuntimeError(f"manifest ScientistConfig is incomplete: {missing}")
    return dict(configuration)


def _verify_final_manifest(
    path: pathlib.Path,
    scientist_config: ScientistConfig,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"registered repaired manifest is missing: {path}")
    digest = sha256_file(path)
    if expected_manifest_sha256 is not None and digest != expected_manifest_sha256:
        raise RuntimeError(f"manifest SHA mismatch: {digest}")
    payload = _load_json(path)
    if not _manifest_contains_expected_identity(payload):
        raise RuntimeError("repaired manifest does not contain the registered A40 UUID and N_STAR")
    frozen_config = _manifest_scientist_config(payload)
    if frozen_config != scientist_config.as_dict():
        raise RuntimeError("manifest ScientistConfig does not match the requested launch configuration")
    contracts = payload.get("contracts")
    if not isinstance(contracts, Mapping) or contracts.get("N", {}).get("active_n_star") != N_STAR:
        raise RuntimeError("manifest N_STAR does not match the registered run")
    _verify_manifest_hashes(payload)
    return payload


def _data_hashes() -> tuple[dict[str, str], str]:
    manifest = _load_json(DATA_MANIFEST_PATH)
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("data manifest has no file list")
    dataset_hashes: dict[str, str] = {}
    tokenizer_rows: list[dict[str, str]] = []
    for row in files:
        if not isinstance(row, dict) or "path" not in row or "sha256" not in row:
            raise ValueError("data manifest contains an invalid file row")
        path = str(row["path"])
        digest = str(row["sha256"])
        if str(row.get("split")) == "tokenizer":
            tokenizer_rows.append({"path": path, "sha256": digest})
        else:
            dataset_hashes[path] = digest
    tokenizer_hash = hashlib.sha256(
        (json.dumps(sorted(tokenizer_rows, key=lambda row: row["path"]), sort_keys=True) + "\n").encode()
    ).hexdigest()
    dataset_hashes["aggregate_manifest_sha256"] = str(manifest["aggregate_manifest_sha256"])
    return dataset_hashes, tokenizer_hash


def _paired_baseline_checkpoint_hashes(manifest: Mapping[str, Any], active_contracts: tuple[str, ...]) -> dict[str, str]:
    """The frozen paired baseline checkpoints the active arms bracket against."""
    checkpoints = manifest.get("a40", {}).get("paired_baseline_checkpoints")
    if not isinstance(checkpoints, Mapping):
        raise RuntimeError("manifest is missing paired baseline checkpoint configuration")
    hashes: dict[str, str] = {}
    for contract_id in active_contracts:
        entry = checkpoints.get(contract_id)
        if not isinstance(entry, Mapping) or not entry.get("path") or not entry.get("sha256"):
            raise RuntimeError(f"manifest has no hashed paired baseline checkpoint for contract {contract_id}")
        hashes[str(entry["path"])] = str(entry["sha256"])
    return hashes


def _registered_cache_file_hashes(cache_root: str) -> dict[str, str]:
    """Map every frozen data-manifest entry to its authoritative remote path."""
    root = pathlib.PurePosixPath(cache_root)
    if not root.is_absolute() or ".." in root.parts:
        raise ValueError("registered cache root must be an absolute safe path")
    manifest = _load_json(DATA_MANIFEST_PATH)
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("registered data manifest has no files")
    expected: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise RuntimeError("registered data manifest contains an invalid row")
        relative = pathlib.PurePosixPath(str(row.get("path", "")))
        digest = str(row.get("sha256", ""))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise RuntimeError(f"registered cache path is unsafe: {relative}")
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise RuntimeError(f"registered cache hash is malformed: {relative}")
        expected[str(root / relative)] = digest
    return expected


def _static_hashes() -> dict[str, str]:
    return {
        "benchmark_policy.py": sha256_file(STUDY_ROOT / "benchmark_policy.py"),
        "inference_benchmark.py": sha256_file(STUDY_ROOT / "inference_benchmark.py"),
        "pair_a_active_inference_config.json": sha256_file(STUDY_ROOT / "provenance" / "pair_a_active_inference_config_20260904.json"),
        "pair_a_protected_prompt_tokens.json": sha256_file(STUDY_ROOT / "provenance" / "pair_a_protected_prompt_tokens_v2.json"),
        "campaign.py": sha256_file(STUDY_ROOT / "campaign.py"),
        "train_launcher.py": sha256_file(STUDY_ROOT / "train_launcher.py"),
    }


def _set_context_modes(context: pathlib.Path) -> None:
    """Make the five visible files read-only except for the candidate train.py."""
    validate_context_paths(tuple(path.name for path in context.iterdir()))
    context.chmod(stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    for path in context.iterdir():
        path.chmod(0o644 if path.name == "train.py" else 0o444)


def _writable_files(context: pathlib.Path) -> list[str]:
    return sorted(
        path.name
        for path in context.iterdir()
        if path.is_file() and path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    )


def _scientist_command(
    context: pathlib.Path,
    campaign_id: str,
    attempt: int,
    scientist_config: ScientistConfig,
    phase: str,
    parent_experiment_id: str | None = None,
    selected_plan: Mapping[str, str] | None = None,
) -> list[str]:
    # This is operational plumbing only; the frozen program.md remains the
    # Scientist's scientific instruction and is the only contract prompt file.
    phase_instruction = (
        "Select exactly one eligible parent_experiment_id from campaign_view.json and do not change train.py yet."
        if phase == "select_parent"
        else (
            "Edit only the selected parent source; the controller supplies "
            f"parent_experiment_id={parent_experiment_id}. "
            "The controller-selected scientific plan is "
            f"{json.dumps(dict(selected_plan or {}), sort_keys=True)}. "
            "Implement that exact plan. Your final hypothesis, intended_change, and "
            "change_category fields must reproduce the controller-selected strings exactly; "
            "do not substitute a different experiment."
        )
    )
    proposal_schema = SCIENTIST_PARENT_SELECTION_SCHEMA if phase == "select_parent" else SCIENTIST_CHILD_EDIT_SCHEMA
    contract_id = "N" if "fixed-tokens" in campaign_id else "T"
    contract_instruction = (
        f"The protected contract is N with exact N_STAR={N_STAR}; every contract_progress call "
        "must pass that exact integer. "
        if contract_id == "N"
        else "The protected contract is T with its fixed 300-second budget. "
    )
    prompt = (
        "Read program.md completely FIRST and treat it as the authoritative research protocol. "
        "Then read campaign_view.json. Then inspect the remaining supplied files as needed. "
        "Modify only train.py. Before returning, verify that train.py contains a direct "
        "study_runtime.contract_progress (or imported contract_progress) call before every "
        "optimizer update; this protected contract hook is mandatory and candidates missing "
        "it are rejected before A40 execution. Do not replace it with validation/evaluation. "
        + contract_instruction
        +
        f"This is campaign {campaign_id}, attempt {attempt}, phase {phase}. {phase_instruction} "
        "Your final response MUST be one JSON object with nonempty string fields \"hypothesis\", "
        "\"intended_change\", and \"change_category\". Parent selection MUST also include a nonempty "
        "\"parent_experiment_id\". Do not emit controller bookkeeping "
        "such as source hashes, IDs, or configuration. Treat validation quality, cached-decode throughput, and "
        "inference memory as co-equal search objectives. Use the complete same-arm evidence, exact prior diffs, "
        "and recorded failures; do not repeat a mechanism unless the proposed source-level distinction and "
        "expected trade-off are explicit."
    )
    if scientist_config.agent_backend == "claude":
        # Same prompt and phase schema; only the agent CLI differs. The working
        # directory is the staged context (run_agent_command uses cwd=context).
        return [
            scientist_config.scientist_bin,
            "-p",
            prompt,
            "--model",
            scientist_config.model,
            "--effort",
            scientist_config.reasoning,
            "--output-format",
            "stream-json",
            "--verbose",
            "--json-schema",
            json.dumps(claude_output_schema(proposal_schema), separators=(",", ":"), sort_keys=True),
            "--no-session-persistence",
            "--safe-mode",
            "--setting-sources",
            "local",
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--tools",
            CLAUDE_SCIENTIST_TOOLS,
            "--permission-mode",
            scientist_config.approval_policy,
        ]
    return [
        scientist_config.scientist_bin,
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--skip-git-repo-check",
        "-c",
        f'approval_policy="{scientist_config.approval_policy}"',
        "--json",
        "--output-schema",
        str(proposal_schema),
        "-C",
        str(context),
        "-s",
        scientist_config.sandbox_mode,
        "-m",
        scientist_config.model,
        "-c",
        f"model_reasoning_effort={scientist_config.reasoning}",
        prompt,
    ]


def _campaign_paths(state_root: pathlib.Path, campaign: Mapping[str, Any]) -> dict[str, pathlib.Path]:
    campaign_root = state_root / str(campaign["campaign_id"])
    return {
        "root": campaign_root,
        "train": campaign_root / "train.py",
        "view": campaign_root / "campaign_view.json",
        "history": campaign_root / str(campaign["history_name"]),
        "source_registry": campaign_root / "source_registry.jsonl",
        "source_dir": campaign_root / "sources",
    }


def _initial_baseline(path: pathlib.Path) -> dict[str, Any]:
    baseline = _load_json(path)
    if isinstance(baseline.get("baseline"), dict):
        baseline = baseline["baseline"]
    if baseline.get("record_scope") != "same_seed_baseline":
        raise ValueError("baseline record must be marked same_seed_baseline")
    return baseline


def _initialize_state(
    state_root: pathlib.Path,
    baseline_path: pathlib.Path | None,
    scientist_config: ScientistConfig,
    manifest_sha256: str,
    expected_a40_uuid: str | None = None,
    active_contracts: tuple[str, ...] = ("T", "N"),
) -> dict[str, Any]:
    scientist_config.validate()
    expected_a40_uuid = expected_a40_uuid or A40_UUID
    schedule = _schedule_from_frozen_file()
    state_path = state_root / "controller_state.json"
    if state_path.exists():
        state = _load_json(state_path)
        if state.get("schema") != "autopareto_pair_a_controller_v1":
            raise RuntimeError("controller state has an unexpected schema")
        if (
            state.get("n_star") != N_STAR
            or state.get("a40_uuid") != expected_a40_uuid
            or state.get("schedule_sha256") != FROZEN_SCHEDULE_SHA256
            or state.get("scientist_config") != scientist_config.as_dict()
            or state.get("manifest_sha256") != manifest_sha256
            or tuple(state.get("active_contracts", ("T", "N"))) != active_contracts
        ):
            raise RuntimeError("controller state does not match the frozen study")
        if state.get("schedule") != [row.as_dict() for row in schedule]:
            raise RuntimeError("controller state schedule differs from the frozen schedule")
        return state
    if baseline_path is None or not baseline_path.is_file():
        raise FileNotFoundError(
            "new campaign state requires --baseline-record containing the registered same-seed baseline"
        )
    baseline = _initial_baseline(baseline_path)
    for contract_id, campaign in CAMPAIGNS.items():
        if contract_id not in active_contracts:
            continue
        paths = _campaign_paths(state_root, campaign)
        if any(path.exists() for path in paths.values()):
            raise FileExistsError(f"partial campaign state exists: {paths['root']}")
        paths["root"].mkdir(parents=True, exist_ok=False)
        paths["source_dir"].mkdir()
        shutil.copy2(BASELINE_SOURCE, paths["train"])
        baseline_snapshot = paths["source_dir"] / "baseline.train.py"
        shutil.copy2(BASELINE_SOURCE, baseline_snapshot)
        _append_history(
            paths["source_registry"],
            {
                "experiment_id": baseline["baseline_experiment_id"],
                "campaign_id": campaign["campaign_id"],
                "source_sha256": sha256_file(BASELINE_SOURCE),
                "source_path": str(baseline_snapshot),
                "parent_eligible": True,
                "pareto_eligible": True,
                "record_scope": "same_seed_baseline",
            },
        )
        view = build_campaign_view(
            campaign_id=str(campaign["campaign_id"]),
            contract_id=str(campaign["contract"].contract_id.value),
            baseline=baseline,
            frontier=[],
            attempt_number=1,
        )
        _atomic_json_write(paths["view"], view)
    state = {
        "schema": "autopareto_pair_a_controller_v1",
        "study_id": STUDY_ID,
        "n_star": N_STAR,
        "a40_uuid": expected_a40_uuid,
        "manifest_sha256": manifest_sha256,
        "scientist_config": scientist_config.as_dict(),
        "schedule_sha256": FROZEN_SCHEDULE_SHA256,
        "schedule": [row.as_dict() for row in schedule],
        "positions": {
            str(row.global_position): {"status": "pending" if row.contract_id in active_contracts else "not_in_study"}
            for row in schedule
        },
    }
    if active_contracts != ("T", "N"):
        # Single-arm study: inactive positions are never executed or counted.
        state["active_contracts"] = list(active_contracts)
    _atomic_json_write(state_path, state)
    return state


def _validate_existing_campaign_state(state_root: pathlib.Path, row: ScheduledAttempt) -> dict[str, pathlib.Path]:
    campaign = _campaign_for(row)
    paths = _campaign_paths(state_root, campaign)
    for name in ("train", "view"):
        path = paths[name]
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"campaign state is missing its regular {name}: {path}")
    view = _load_json(paths["view"])
    if view.get("schema") != CAMPAIGN_VIEW_SCHEMA:
        raise RuntimeError(
            "campaign view schema is stale; refusing to expose incomplete Scientist history"
        )
    if view.get("campaign_id") != row.campaign_id or view.get("contract_id") != row.contract_id:
        raise RuntimeError("campaign view identity does not match the scheduled position")
    if int(view.get("attempt_number", -1)) != row.campaign_attempt:
        raise RuntimeError("campaign view is not positioned at the scheduled attempt")
    return paths


def _provenance_base(
    *,
    manifest_path: pathlib.Path,
    row: ScheduledAttempt,
    campaign: Mapping[str, Any],
    scientist_config: ScientistConfig,
    a40_uuid: str = A40_UUID,
) -> dict[str, Any]:
    return {
        "study_id": STUDY_ID,
        "global_position": row.global_position,
        "campaign_id": row.campaign_id,
        "contract_id": row.contract_id,
        "campaign_attempt": row.campaign_attempt,
        "training_seed": TRAINING_SEED,
        "agent_model_identifier": scientist_config.model,
        "reasoning_setting": scientist_config.reasoning,
        "contract": json_contract(campaign["contract"]),
        "a40": {"gpu_uuid": a40_uuid, "manifest_sha256": sha256_file(manifest_path)},
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "schedule_sha256": FROZEN_SCHEDULE_SHA256,
        "program_sha256": sha256_file(campaign["program"]),
        "prepare_sha256": sha256_file(PROTECTED_PREPARE),
        "study_runtime_sha256": sha256_file(STUDY_RUNTIME),
        "starting_source_sha256": sha256_file(BASELINE_SOURCE),
    }


def _candidate_patch(before: pathlib.Path, after: pathlib.Path) -> str:
    before_lines = before.read_text().splitlines(keepends=True)
    after_lines = after.read_text().splitlines(keepends=True)
    return "".join(difflib.unified_diff(before_lines, after_lines, fromfile="train.py.before", tofile="train.py.after"))


def _ensure_unique_candidate_source(source_registry: pathlib.Path, candidate_sha256: str) -> None:
    """Reject a byte-identical source before creating any remote invocation."""
    if not source_registry.is_file():
        return
    for line in source_registry.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("source_sha256") == candidate_sha256:
            raise RuntimeError(
                "candidate source is byte-identical to an existing campaign source; "
                "no duplicate scientific attempt will be dispatched"
            )


def _validate_parent_selection(
    *, paths: Mapping[str, pathlib.Path], row: ScheduledAttempt, proposal: Mapping[str, Any],
) -> tuple[str, pathlib.Path, str]:
    parent_id = proposal.get("parent_experiment_id")
    if parent_id is None:
        raise RuntimeError("Scientist must select an explicit parent_experiment_id")
    records = []
    registry = paths["source_registry"]
    if registry.is_file():
        records = [json.loads(line) for line in registry.read_text().splitlines() if line.strip()]
    matches = [record for record in records if record.get("experiment_id") == str(parent_id)]
    if len(matches) != 1:
        raise RuntimeError("Scientist selected an unknown or ambiguous parent experiment")
    parent = matches[0]
    if parent.get("campaign_id") != row.campaign_id or not parent.get("parent_eligible"):
        raise RuntimeError("Scientist selected a cross-arm or non-parent-eligible source")
    source_path = pathlib.Path(str(parent.get("source_path", ""))).resolve()
    source_dir = paths["source_dir"].resolve()
    if source_path.parent != source_dir or source_path.is_symlink() or not source_path.is_file():
        raise RuntimeError("selected parent source is missing or outside the campaign source registry")
    source_sha256 = sha256_file(source_path)
    if parent.get("source_sha256") != source_sha256:
        raise RuntimeError("selected parent source hash does not match its registry record")
    return str(parent_id), source_path, source_sha256


def _validate_child_plan(
    proposal: Mapping[str, Any], selected_plan: Mapping[str, str]
) -> None:
    """Reject a child edit that silently substitutes the selected experiment."""
    observed = {
        key: proposal.get(key)
        for key in ("hypothesis", "intended_change", "change_category")
    }
    if observed != dict(selected_plan):
        raise RuntimeError("child Scientist changed the controller-selected scientific plan")


def _append_history(path: pathlib.Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(dict(record), indent=None, sort_keys=True, default=str) + "\n")


def _reserve_position(state_path: pathlib.Path, state: dict[str, Any], row: ScheduledAttempt) -> None:
    position = state["positions"].get(str(row.global_position))
    if not isinstance(position, dict) or position.get("status") != "pending":
        raise RuntimeError(
            f"schedule position {row.global_position} is not pending; refusing any retry or duplicate execution"
        )
    active = state.get("active_contracts", ["T", "N"])
    earlier = [
        state["positions"][str(item.global_position)]["status"]
        for item in _schedule_from_frozen_file()
        if item.global_position < row.global_position and item.contract_id in active
    ]
    if any(status != "completed" for status in earlier):
        raise RuntimeError("frozen schedule cannot advance before every earlier position is completed")
    position.update({"status": "running", "campaign_id": row.campaign_id, "campaign_attempt": row.campaign_attempt})
    _atomic_json_write(state_path, state)


def _finish_position(state_path: pathlib.Path, state: dict[str, Any], row: ScheduledAttempt, status: str, **extra: Any) -> None:
    state["positions"][str(row.global_position)].update({"status": status, **extra})
    _atomic_json_write(state_path, state)


def _pending_root(paths: Mapping[str, pathlib.Path], row: ScheduledAttempt) -> pathlib.Path:
    return paths["root"] / "pending" / f"position-{row.global_position:03d}"


def _invocation_id(*, row: ScheduledAttempt, candidate_sha256: str, metadata_sha256: str) -> str:
    digest = hashlib.sha256(
        f"{STUDY_ID}:{row.global_position}:{row.campaign_id}:{candidate_sha256}:{metadata_sha256}".encode()
    ).hexdigest()
    return f"pair-a-{row.global_position:03d}-{digest[:32]}"


def _persist_pending_execution(
    *,
    paths: Mapping[str, pathlib.Path],
    row: ScheduledAttempt,
    state: dict[str, Any],
    state_path: pathlib.Path,
    parent_experiment_id: str,
    parent_source: pathlib.Path,
    candidate: pathlib.Path,
    metadata: Mapping[str, Any],
    reservation: Mapping[str, Any],
    scientist: Mapping[str, Any],
    remote_run_namespace: str = "pair-a",
) -> dict[str, Any]:
    root = _pending_root(paths, row)
    if root.exists():
        raise RuntimeError("pending execution directory already exists for the scheduled position")
    root.mkdir(parents=True)
    candidate_path = root / "train.py"
    metadata_path = root / "registered_metadata.json"
    shutil.copy2(candidate, candidate_path)
    _atomic_json_write(metadata_path, dict(metadata))
    candidate_sha256 = sha256_file(candidate_path)
    metadata_sha256 = sha256_file(metadata_path)
    if candidate_sha256 != metadata.get("candidate_source_after_sha256"):
        raise RuntimeError("persisted child source hash disagrees with registered metadata")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", remote_run_namespace):
        raise ValueError("remote run namespace is malformed")
    execution = {
        "schema": "autopareto_pending_execution_v1",
        "remote_run_id": f"{remote_run_namespace}-position-{row.global_position:03d}-{row.campaign_id}",
        "output_name": f"position-{row.global_position:03d}-{row.campaign_id}.json",
        "invocation_id": _invocation_id(
            row=row, candidate_sha256=candidate_sha256, metadata_sha256=metadata_sha256
        ),
        "parent_experiment_id": parent_experiment_id,
        "parent_source_path": str(parent_source),
        "candidate_path": str(candidate_path),
        "metadata_path": str(metadata_path),
        "candidate_source_before_sha256": metadata["candidate_source_before_sha256"],
        "candidate_source_after_sha256": candidate_sha256,
        "metadata_sha256": metadata_sha256,
        "reservation": dict(reservation),
        "scientist": dict(scientist),
        "dispatch_status": "prepared",
    }
    position = state["positions"][str(row.global_position)]
    position["execution"] = execution
    _atomic_json_write(state_path, state)
    return execution


def _load_pending_execution(
    *, paths: Mapping[str, pathlib.Path], row: ScheduledAttempt, position: Mapping[str, Any]
) -> tuple[dict[str, Any], pathlib.Path, dict[str, Any]]:
    execution = position.get("execution")
    if not isinstance(execution, dict) or execution.get("schema") != "autopareto_pending_execution_v1":
        raise RuntimeError("running position has no durable pending execution record")
    candidate = pathlib.Path(str(execution.get("candidate_path", ""))).resolve()
    metadata_path = pathlib.Path(str(execution.get("metadata_path", ""))).resolve()
    root = _pending_root(paths, row).resolve()
    if candidate.parent != root or metadata_path.parent != root or not candidate.is_file() or not metadata_path.is_file():
        raise RuntimeError("pending execution artifacts are missing or outside their campaign state directory")
    metadata = _load_json(metadata_path)
    if sha256_file(candidate) != execution.get("candidate_source_after_sha256"):
        raise RuntimeError("pending candidate source hash does not match controller state")
    if sha256_file(metadata_path) != execution.get("metadata_sha256"):
        raise RuntimeError("pending metadata hash does not match controller state")
    if metadata.get("candidate_source_after_sha256") != execution.get("candidate_source_after_sha256"):
        raise RuntimeError("pending metadata candidate hash does not match controller state")
    return dict(execution), candidate, metadata


def _mark_reconciliation_required(
    *, state: dict[str, Any], state_path: pathlib.Path, row: ScheduledAttempt, reason: Mapping[str, Any]
) -> None:
    position = state["positions"][str(row.global_position)]
    position.update({"status": "reconciliation_required", "reconciliation_failure": dict(reason)})
    _atomic_json_write(state_path, state)


def _dispatch_and_wait(
    *,
    state: dict[str, Any],
    state_path: pathlib.Path,
    row: ScheduledAttempt,
    paths: Mapping[str, pathlib.Path],
    transport: A40Transport,
    study_scope: str = "official_pair_a",
) -> tuple[dict[str, Any], dict[str, Any]]:
    position = state["positions"][str(row.global_position)]
    execution, candidate, metadata = _load_pending_execution(paths=paths, row=row, position=position)
    if execution.get("dispatch_status") == "prepared":
        try:
            with tempfile.TemporaryDirectory(prefix=f"autopareto-resume-{row.global_position:03d}-") as temporary:
                payload = stage_remote_payload(
                    destination=pathlib.Path(temporary) / "payload", candidate=candidate, metadata=metadata
                )
                remote_dir = transport.transfer(payload=payload, remote_run_id=str(execution["remote_run_id"]))
        except BaseException as exc:
            _mark_reconciliation_required(
                state=state,
                state_path=state_path,
                row=row,
                reason={"type": type(exc).__name__, "message": str(exc), "phase": "payload_transfer"},
            )
            raise
        execution["remote_payload_dir"] = remote_dir
        execution["dispatch_status"] = "payload_transferred"
        position["execution"] = execution
        _atomic_json_write(state_path, state)
    if execution.get("dispatch_status") == "payload_transferred":
        execution["dispatch_status"] = "launch_requested"
        position["execution"] = execution
        _atomic_json_write(state_path, state)
    try:
        if execution.get("dispatch_status") == "launch_requested":
            transport.launch(
                remote_dir=str(execution["remote_payload_dir"]),
                output_name=str(execution["output_name"]),
                invocation_id=str(execution["invocation_id"]),
                invocation_metadata={
                    "study_id": STUDY_ID,
                    "study_scope": study_scope,
                    "global_position": row.global_position,
                    "campaign_id": row.campaign_id,
                    "contract_id": row.contract_id,
                    "campaign_attempt": row.campaign_attempt,
                    "parent_experiment_id": execution["parent_experiment_id"],
                    "parent_source_sha256": execution["candidate_source_before_sha256"],
                    "candidate_source_sha256": execution["candidate_source_after_sha256"],
                },
            )
        observed = transport.wait_for_invocation(str(execution["invocation_id"]))
    except BaseException as exc:
        # The durable remote receipt is the authority.  A local transport loss
        # leaves this position running for reconciliation; it is never consumed.
        position["last_transport_error"] = {"type": type(exc).__name__, "message": str(exc)}
        _atomic_json_write(state_path, state)
        raise
    record = observed["record"]
    if record.get("status") == "COMPLETE":
        return execution, transport.fetch_durable_result(record)
    if record.get("status") == "RUNNING" and not observed.get("process_alive"):
        _mark_reconciliation_required(
            state=state,
            state_path=state_path,
            row=row,
            reason={"type": "RemoteInvocationAmbiguous", "message": "RUNNING receipt has no matching live worker"},
        )
        raise RuntimeError("remote invocation is ambiguous; manual reconciliation is required")
    if record.get("status") == "FAILED":
        _mark_reconciliation_required(
            state=state,
            state_path=state_path,
            row=row,
            reason={"type": "RemoteInfrastructureFailure", "receipt": record},
        )
        raise RuntimeError("remote invocation failed before a valid scientific result; manual reconciliation is required")
    _mark_reconciliation_required(
        state=state,
        state_path=state_path,
        row=row,
        reason={"type": "RemoteInvocationIntegrityFailure", "receipt": record},
    )
    raise RuntimeError("remote invocation receipt has an invalid state")


def _commit_pending_result(
    *,
    state: dict[str, Any],
    state_path: pathlib.Path,
    row: ScheduledAttempt,
    paths: Mapping[str, pathlib.Path],
    remote_result: dict[str, Any],
) -> dict[str, Any]:
    position = state["positions"][str(row.global_position)]
    execution, candidate, metadata = _load_pending_execution(paths=paths, row=row, position=position)
    parent_experiment_id, parent_source, source_before_sha256 = _validate_parent_selection(
        paths=paths, row=row, proposal={"parent_experiment_id": execution["parent_experiment_id"]}
    )
    if source_before_sha256 != execution["candidate_source_before_sha256"]:
        raise RuntimeError("committed parent SHA disagrees with the persisted pending execution")
    candidate_after_sha256 = sha256_file(candidate)
    if candidate_after_sha256 != execution["candidate_source_after_sha256"]:
        raise RuntimeError("pending child source changed after dispatch")
    if remote_result.get("campaign_id") != row.campaign_id or int(remote_result.get("attempt", -1)) != row.campaign_attempt:
        raise RuntimeError("retrieved A40 result does not match the scheduled campaign position")
    reservation = dict(execution["reservation"])
    provenance = dict(reservation.get("provenance", reservation))
    if remote_result.get("status") == "harness_failure" and remote_result.get("failure_class") == "harness":
        # A protected API/runtime defect must never be silently charged to a
        # scientific round. Preserve the durable result for diagnosis, leave
        # both campaign histories/frontiers untouched, and stop before retry.
        _mark_reconciliation_required(
            state=state,
            state_path=state_path,
            row=row,
            reason={
                "type": "ProtectedHarnessFailure",
                "error": str(remote_result.get("error", "protected harness failure")),
                "remote_result": remote_result,
            },
        )
        raise RuntimeError("protected harness failure; the scheduled position was not consumed")
    if remote_result.get("status") == "scientific_failure" and remote_result.get("failure_class") == "candidate":
        # Preserve the failed candidate's same-arm lineage and research
        # evidence in campaign_view.json. It remains non-parent-eligible, but
        # the next Scientist must be able to see which hypothesis/source failed
        # and avoid repeating it blindly.
        remote_result.update(
            {
                "experiment_id": f"{row.campaign_id}-attempt-{row.campaign_attempt:02d}",
                "parent_experiment_id": parent_experiment_id,
                "candidate_source_before_sha256": source_before_sha256,
                "hypothesis": metadata["hypothesis"],
                "intended_change": metadata["intended_change"],
                "change_category": metadata["change_category"],
                "quality_status": "invalid",
                "parent_eligible": False,
                "pareto_eligible": False,
                "candidate_patch": _candidate_patch(parent_source, candidate),
            }
        )
        failure_record = {
            **reservation,
            "status": "scientific_failure",
            "candidate_source_after_sha256": None,
            "failure": {
                "type": "CandidateScientificFailure",
                "message": str(remote_result.get("error", "candidate failure")),
            },
            "result": remote_result,
            "provenance": provenance,
        }
        remote_result["pareto_decision"] = {
            "decision": "failed",
            "reason": "candidate-caused scientific failure",
        }
        result_path = paths["root"] / str(execution["output_name"])
        _atomic_json_write(result_path, remote_result)
        update_campaign_view(view_path=paths["view"], result=remote_result)
        _append_history(paths["history"], failure_record)
        _finish_position(
            state_path, state, row, "completed", outcome="scientific_failure", result_file=str(result_path)
        )
        return failure_record
    if remote_result.get("status") != "completed":
        raise RuntimeError(f"protected A40 attempt failed: {remote_result.get('error', remote_result)}")
    current_view = _load_json(paths["view"])
    decision = decide_candidate(
        remote_result, [current_view["baseline"], *current_view.get("frontier", [])]
    )
    remote_result["pareto_decision"] = decision
    experiment_id = f"{row.campaign_id}-attempt-{row.campaign_attempt:02d}"
    remote_result.update({
        "experiment_id": experiment_id,
        "parent_experiment_id": parent_experiment_id,
        "candidate_source_before_sha256": source_before_sha256,
        "candidate_source_after_sha256": candidate_after_sha256,
        "hypothesis": metadata["hypothesis"],
        "intended_change": metadata["intended_change"],
        "change_category": metadata["change_category"],
        "quality_status": decision["quality_status"],
        "parent_eligible": bool(decision["parent_eligible"]),
        "pareto_eligible": bool(decision["pareto_eligible"]),
    })
    candidate_patch = _candidate_patch(parent_source, candidate)
    remote_result["candidate_patch"] = candidate_patch
    result_path = paths["root"] / str(execution["output_name"])
    _atomic_json_write(result_path, remote_result)
    source_snapshot = paths["source_dir"] / f"{experiment_id}.train.py"
    shutil.copy2(candidate, source_snapshot)
    _append_history(
        paths["source_registry"],
        {
            "experiment_id": experiment_id,
            "campaign_id": row.campaign_id,
            "parent_experiment_id": parent_experiment_id,
            "source_sha256": candidate_after_sha256,
            "source_path": str(source_snapshot),
            "parent_eligible": bool(decision["parent_eligible"]),
            "pareto_eligible": bool(decision["pareto_eligible"]),
            "quality_status": decision["quality_status"],
        },
    )
    update_campaign_view(view_path=paths["view"], result=remote_result)
    record = {
        **reservation,
        "status": "completed",
        "candidate_source_after_sha256": candidate_after_sha256,
        "candidate_patch": candidate_patch,
        "hypothesis": metadata["hypothesis"],
        "intended_change": metadata["intended_change"],
        "change_category": metadata["change_category"],
        "scientist": execution["scientist"],
        "result": remote_result,
        "pareto_decision": decision,
        "provenance": provenance,
    }
    _append_history(paths["history"], record)
    _finish_position(state_path, state, row, "completed", result_file=str(result_path))
    return record


def _resume_running_position(
    *,
    state_root: pathlib.Path,
    state: dict[str, Any],
    state_path: pathlib.Path,
    row: ScheduledAttempt,
    transport: A40Transport,
) -> dict[str, Any]:
    paths = _validate_existing_campaign_state(state_root, row)
    execution, _candidate, _metadata = _load_pending_execution(
        paths=paths, row=row, position=state["positions"][str(row.global_position)]
    )
    if execution.get("dispatch_status") != "launch_requested":
        raise RuntimeError("running position was interrupted before durable remote launch; manual reconciliation is required")
    _execution, remote_result = _dispatch_and_wait(
        state=state, state_path=state_path, row=row, paths=paths, transport=transport
    )
    return _commit_pending_result(
        state=state, state_path=state_path, row=row, paths=paths, remote_result=remote_result
    )


def dry_run_position(
    position: int,
    output_root: pathlib.Path | None = None,
    *,
    manifest_path: pathlib.Path = FINAL_MANIFEST_PATH,
    baseline_path: pathlib.Path = DEFAULT_BASELINE_RECORD_PATH,
    scientist_config: ScientistConfig,
    a40_config: A40RemoteConfig | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    scientist_config.validate()
    _verify_final_manifest(manifest_path, scientist_config, expected_manifest_sha256)
    baseline = _initial_baseline(baseline_path)
    protected_search_metrics({"inference_result": baseline.get("inference_result"), **baseline})
    if a40_config is None:
        a40_config = A40RemoteConfig(
            host=DEFAULT_A40_HOST,
            identity_file=DEFAULT_A40_IDENTITY,
            repository=DEFAULT_A40_REPOSITORY,
        )
    schedule = _schedule_from_frozen_file()
    if not 1 <= position <= len(schedule):
        raise ValueError("dry-run position must be in 1..50")
    row = schedule[position - 1]
    campaign = _campaign_for(row)
    if output_root is None:
        output_root = pathlib.Path(tempfile.mkdtemp(prefix=f"autopareto-dry-run-{position:03d}-"))
    else:
        output_root = output_root.resolve()
        if output_root.exists():
            raise FileExistsError(f"refusing to overwrite dry-run output: {output_root}")
        output_root.mkdir(parents=True)
    view_path = output_root / "campaign_view.json"
    view = build_campaign_view(
        campaign_id=row.campaign_id,
        contract_id=row.contract_id,
        baseline=baseline,
        frontier=[],
        attempt_number=row.campaign_attempt,
    )
    _atomic_json_write(view_path, view)
    context = output_root / "scientist-context"
    stage_agent_context(
        destination=context,
        baseline_train=BASELINE_SOURCE,
        baseline_prepare=PROTECTED_PREPARE,
        study_runtime=STUDY_RUNTIME,
        program=campaign["program"],
        campaign_view=view_path,
    )
    _set_context_modes(context)
    observed = sorted(path.name for path in context.iterdir())
    validate_context_paths(tuple(observed))
    writable = _writable_files(context)
    if writable != list(AGENT_WRITABLE_PATHS):
        raise RuntimeError(f"dry-run writable allowlist mismatch: {writable}")
    result = {
        "status": "dry_run_passed",
        "scientist_invoked": False,
        "a40_invoked": False,
        "schedule_position": row.as_dict(),
        "context": str(context),
        "visible_files": observed,
        "writable_files": writable,
        "program": str(campaign["program"]),
        "n_star": N_STAR,
        "a40_uuid": A40_UUID,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "baseline_record": str(baseline_path),
        "scientist_config": scientist_config.as_dict(),
        "a40": {
            "host": a40_config.host,
            "port": a40_config.port,
            "identity_file": a40_config.identity_file,
            "repository": a40_config.repository,
        },
    }
    _atomic_json_write(output_root / "dry_run.json", result)
    return result


def _run_position(
    *,
    state_root: pathlib.Path,
    state: dict[str, Any],
    state_path: pathlib.Path,
    row: ScheduledAttempt,
    manifest_path: pathlib.Path,
    transport: A40Transport,
    scientist_config: ScientistConfig,
    study_scope: str = "official_pair_a",
    remote_run_namespace: str = "pair-a",
) -> dict[str, Any]:
    campaign = _campaign_for(row)
    paths = _validate_existing_campaign_state(state_root, row)
    provenance = _provenance_base(
        manifest_path=manifest_path,
        row=row,
        campaign=campaign,
        scientist_config=scientist_config,
        a40_uuid=str(state["a40_uuid"]),
    )
    reservation = {
        **provenance,
        "status": "running",
    }
    _reserve_position(state_path, state, row)
    try:
        with tempfile.TemporaryDirectory(prefix=f"autopareto-position-{row.global_position:03d}-") as temporary:
            root = pathlib.Path(temporary)
            selection_context = root / "parent-selection-context"
            # Parent selection sees the registered baseline plus concise
            # same-arm evidence. The child-edit phase stages only the exact
            # controller-validated parent, so no implicit latest source can
            # override an explicit parent selection.
            stage_agent_context(
                destination=selection_context,
                baseline_train=paths["source_dir"] / "baseline.train.py",
                baseline_prepare=PROTECTED_PREPARE,
                study_runtime=STUDY_RUNTIME,
                program=campaign["program"],
                campaign_view=paths["view"],
            )
            _set_context_modes(selection_context)
            observed = sorted(path.name for path in selection_context.iterdir())
            validate_context_paths(tuple(observed))
            if _writable_files(selection_context) != list(AGENT_WRITABLE_PATHS):
                raise RuntimeError("Scientist context writable allowlist mismatch")

            selection_command = _scientist_command(
                selection_context,
                row.campaign_id,
                row.campaign_attempt,
                scientist_config,
                "select_parent",
            )
            selection_handoff, selection_process = run_local_scientist(
                command=selection_command,
                context=selection_context,
                campaign_id=row.campaign_id,
                contract_id=row.contract_id,
                attempt_number=row.campaign_attempt,
                phase="select_parent",
                scientist_config=scientist_config,
                diagnostic_root=paths["root"] / "scientist_diagnostics",
            )
            parent_experiment_id, parent_source, source_before_sha256 = _validate_parent_selection(
                paths=paths,
                row=row,
                proposal=selection_handoff.proposal,
            )
            selected_plan = {
                key: str(selection_handoff.proposal[key])
                for key in ("hypothesis", "intended_change", "change_category")
            }
            context = root / "scientist-context"
            stage_agent_context(
                destination=context,
                baseline_train=parent_source,
                baseline_prepare=PROTECTED_PREPARE,
                study_runtime=STUDY_RUNTIME,
                program=campaign["program"],
                campaign_view=paths["view"],
            )
            _set_context_modes(context)
            if sorted(path.name for path in context.iterdir()) != observed or _writable_files(context) != list(AGENT_WRITABLE_PATHS):
                raise RuntimeError("Scientist child context allowlist mismatch")
            if sha256_file(context / "train.py") != source_before_sha256:
                raise RuntimeError("selected parent SHA does not equal the source staged for child editing")
            child_command = _scientist_command(
                context,
                row.campaign_id,
                row.campaign_attempt,
                scientist_config,
                "edit_child",
                parent_experiment_id,
                selected_plan,
            )
            handoff, scientist_process = run_local_scientist(
                command=child_command,
                context=context,
                campaign_id=row.campaign_id,
                contract_id=row.contract_id,
                attempt_number=row.campaign_attempt,
                phase="edit_child",
                expected_parent_experiment_id=parent_experiment_id,
                scientist_config=scientist_config,
                diagnostic_root=paths["root"] / "scientist_diagnostics",
            )
            if handoff.proposal.get("parent_experiment_id") != parent_experiment_id:
                raise RuntimeError("child Scientist changed the controller-validated parent_experiment_id")
            _validate_child_plan(handoff.proposal, selected_plan)
            if handoff.candidate_source_before_sha256 != source_before_sha256:
                raise RuntimeError("child Scientist handoff source SHA does not match selected parent source SHA")
            # Reject only statically provable contract-hook misuse before a
            # durable A40 invocation is created. The runtime performs the same
            # check independently at the candidate boundary.
            validate_candidate_contract_api(
                context / "train.py",
                expected_n_star=N_STAR if row.contract_id == "N" else None,
            )
            candidate_after_sha256 = sha256_file(context / "train.py")
            _ensure_unique_candidate_source(paths["source_registry"], candidate_after_sha256)
            reservation["candidate_source_before_sha256"] = source_before_sha256
            reservation["parent_source_path"] = str(parent_source)
            reservation["parent_experiment_id"] = parent_experiment_id
            dataset_hashes, tokenizer_hash = _data_hashes()
            metadata = build_registered_metadata(
                handoff=handoff,
                study_id=STUDY_ID,
                training_seed=TRAINING_SEED,
                starting_source_hash=sha256_file(BASELINE_SOURCE),
                evaluator_hashes=_static_hashes(),
                dataset_hashes=dataset_hashes,
                tokenizer_hash=tokenizer_hash,
                contract_parameters={
                    "time_seconds": 300.0 if row.contract_id == "T" else None,
                    "n_star": N_STAR if row.contract_id == "N" else None,
                    "paired_baseline_checkpoint": transport.config.paired_baseline_checkpoint(row.contract_id),
                },
                protected_prepare_sha256=sha256_file(PROTECTED_PREPARE),
                study_runtime_sha256=sha256_file(STUDY_RUNTIME),
            )
            scientist_record = {
                "parent_selection_handoff": selection_handoff.as_dict(),
                "parent_selection_returncode": selection_process.returncode,
                "parent_selection_stdout_sha256": hashlib.sha256(selection_process.stdout.encode()).hexdigest(),
                "parent_selection_stderr_sha256": hashlib.sha256(selection_process.stderr.encode()).hexdigest(),
                "handoff": handoff.as_dict(),
                "returncode": scientist_process.returncode,
                "stdout_sha256": hashlib.sha256(scientist_process.stdout.encode()).hexdigest(),
                "stderr_sha256": hashlib.sha256(scientist_process.stderr.encode()).hexdigest(),
                "process_diagnostics": getattr(scientist_process, "scientist_diagnostics", None),
                "visible_files": observed,
                "writable_files": list(AGENT_WRITABLE_PATHS),
            }
            _persist_pending_execution(
                paths=paths,
                row=row,
                state=state,
                state_path=state_path,
                parent_experiment_id=parent_experiment_id,
                parent_source=parent_source,
                candidate=context / "train.py",
                metadata=metadata,
                reservation=reservation,
                scientist=scientist_record,
                remote_run_namespace=remote_run_namespace,
            )
            _execution, remote_result = _dispatch_and_wait(
                state=state,
                state_path=state_path,
                row=row,
                paths=paths,
                transport=transport,
                study_scope=study_scope,
            )
            return _commit_pending_result(
                state=state, state_path=state_path, row=row, paths=paths, remote_result=remote_result
            )
    except BaseException as exc:
        position = state["positions"][str(row.global_position)]
        execution = position.get("execution")
        if position.get("status") == "running" and isinstance(execution, dict) and execution.get("dispatch_status") == "launch_requested":
            # A durable remote receipt may still be RUNNING or COMPLETE.  Do
            # not consume, fail, or duplicate this scientific position merely
            # because the local transport disappeared.
            position["last_transport_error"] = {"type": type(exc).__name__, "message": str(exc)}
            _atomic_json_write(state_path, state)
        elif position.get("status") == "running":
            _mark_reconciliation_required(
                state=state,
                state_path=state_path,
                row=row,
                reason={"type": type(exc).__name__, "message": str(exc), "phase": "pre_remote_execution"},
            )
        raise


def run_campaign(
    *,
    state_root: pathlib.Path,
    baseline_path: pathlib.Path | None,
    manifest_path: pathlib.Path,
    transport: A40Transport,
    scientist_config: ScientistConfig,
    expected_manifest_sha256: str | None = None,
    active_contracts: tuple[str, ...] = ("T", "N"),
) -> dict[str, Any]:
    scientist_config.validate()
    manifest = _verify_final_manifest(manifest_path, scientist_config, expected_manifest_sha256)
    _verify_transport_config(manifest, transport.config)
    _verify_local_state_config(manifest, state_root)
    # A derived manifest freezes its study arms and a remote namespace that keeps
    # its payloads apart from earlier campaigns restored on the same A40.
    derivation = manifest.get("derivation") or {}
    if "active_contracts" in derivation and tuple(derivation["active_contracts"]) != tuple(active_contracts):
        raise RuntimeError(
            f"manifest freezes contracts {derivation['active_contracts']}, not the requested {list(active_contracts)}"
        )
    remote_run_namespace = str(derivation.get("remote_run_namespace", "pair-a"))
    manifest_sha256 = sha256_file(manifest_path)
    expected_a40_uuid = _manifest_a40_uuid(manifest)
    transport.verify_trusted_files(_manifest_hashes(manifest))
    transport.verify_gpu_identity(expected_a40_uuid)
    absolute_files = _registered_cache_file_hashes(str(transport.config.registered_cache))
    absolute_files.update(_paired_baseline_checkpoint_hashes(manifest, active_contracts))
    transport.verify_absolute_files(absolute_files)
    state = _initialize_state(
        state_root,
        baseline_path,
        scientist_config,
        manifest_sha256,
        expected_a40_uuid=expected_a40_uuid,
        active_contracts=active_contracts,
    )
    state_path = state_root / "controller_state.json"
    schedule = _schedule_from_frozen_file()
    # Refuse to consume a study position unless every active campaign view
    # already contains the registered, protected same-seed baseline.  The dry
    # run uses a deliberately synthetic baseline and never reaches this gate.
    for contract_id, campaign in CAMPAIGNS.items():
        if contract_id not in active_contracts:
            continue
        paths = _campaign_paths(state_root, campaign)
        baseline = _load_json(paths["view"])["baseline"]
        protected_search_metrics({"inference_result": baseline.get("inference_result"), **baseline})
    completed: list[dict[str, Any]] = []
    for row in schedule:
        if row.contract_id not in active_contracts:
            continue
        position = state["positions"][str(row.global_position)]
        if position.get("status") == "completed":
            continue
        if position.get("status") == "running":
            completed.append(
                _resume_running_position(
                    state_root=state_root,
                    state=state,
                    state_path=state_path,
                    row=row,
                    transport=transport,
                )
            )
            continue
        if position.get("status") != "pending":
            raise RuntimeError(f"campaign is paused at schedule position {row.global_position}: {position}")
        completed.append(
            _run_position(
                state_root=state_root,
                state=state,
                state_path=state_path,
                row=row,
                manifest_path=manifest_path,
                transport=transport,
                scientist_config=scientist_config,
                remote_run_namespace=remote_run_namespace,
            )
        )
    return {"status": "completed", "attempts": len(completed), "schedule_positions": len(schedule)}


def _verify_transport_config(manifest: Mapping[str, Any], config: A40RemoteConfig) -> None:
    a40 = manifest.get("a40")
    persistent = manifest.get("persistent_paths")
    if not isinstance(a40, Mapping) or not isinstance(persistent, Mapping):
        raise RuntimeError("manifest is missing A40 transport or persistent-path configuration")
    checkpoints = a40.get("paired_baseline_checkpoints")
    if not isinstance(checkpoints, Mapping):
        raise RuntimeError("manifest is missing paired baseline checkpoint configuration")
    expected = {
        "host": a40.get("ssh_host"),
        "repository": a40.get("trusted_repository"),
        "python_path": a40.get("rootfs_runtime"),
        "study_venv": a40.get("study_venv"),
        "registered_cache": persistent.get("registered_cache"),
        "trusted_state_root": persistent.get("trusted_state_root"),
        "paired_baseline_t_checkpoint": checkpoints.get("T", {}).get("path") if isinstance(checkpoints.get("T"), Mapping) else None,
        "paired_baseline_n_checkpoint": checkpoints.get("N", {}).get("path") if isinstance(checkpoints.get("N"), Mapping) else None,
    }
    observed = {
        "host": config.host,
        "repository": config.repository,
        "python_path": config.python_path,
        "study_venv": config.study_venv,
        "registered_cache": config.registered_cache,
        "trusted_state_root": config.trusted_state_root,
        "paired_baseline_t_checkpoint": config.paired_baseline_t_checkpoint,
        "paired_baseline_n_checkpoint": config.paired_baseline_n_checkpoint,
    }
    if expected != observed:
        raise RuntimeError(f"launch transport configuration does not match manifest: {observed} != {expected}")
    candidate_paths = {
        "candidate_workspace_root": "/opt/autopareto_candidate_contexts",
        "candidate_attempt_root": "/opt/autopareto-candidate-attempts",
        "candidate_output_root": "/var/lib/autopareto/candidate_attempts",
        "candidate_runtime_root": "/opt/autopareto_candidate_runtime",
        "candidate_input_root": "/opt/autopareto_candidate_inputs/per_attempts",
    }
    observed_candidate_paths = {name: persistent.get(name) for name in candidate_paths}
    if observed_candidate_paths != candidate_paths:
        raise RuntimeError(
            f"manifest candidate-runtime paths do not match the production launcher: "
            f"{observed_candidate_paths} != {candidate_paths}"
        )


def _verify_local_state_config(manifest: Mapping[str, Any], state_root: pathlib.Path) -> None:
    persistent = manifest.get("persistent_paths")
    if not isinstance(persistent, Mapping):
        raise RuntimeError("manifest is missing persistent-path configuration")
    frozen = pathlib.Path(str(persistent.get("local_official_state", ""))).resolve()
    if frozen != state_root.resolve():
        raise RuntimeError(f"launch state root does not match manifest: {state_root.resolve()} != {frozen}")


def preflight_campaign(
    *,
    state_root: pathlib.Path,
    manifest_path: pathlib.Path,
    transport: A40Transport,
    scientist_config: ScientistConfig,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Exercise the official launch gates without creating or launching an attempt."""
    state_path = state_root / "controller_state.json"
    if not state_path.is_file():
        raise FileNotFoundError("preflight requires an already-prepared campaign state")
    scientist_config.validate()
    manifest = _verify_final_manifest(manifest_path, scientist_config, expected_manifest_sha256)
    _verify_transport_config(manifest, transport.config)
    _verify_local_state_config(manifest, state_root)
    expected_a40_uuid = _manifest_a40_uuid(manifest)
    transport.verify_trusted_files(_manifest_hashes(manifest))
    transport.verify_gpu_identity(expected_a40_uuid)
    checkpoint_rows = manifest["a40"]["paired_baseline_checkpoints"]
    checkpoint_hashes = {
        str(checkpoint_rows[arm]["path"]): str(checkpoint_rows[arm]["sha256"])
        for arm in ("T", "N")
    }
    transport.verify_absolute_files(checkpoint_hashes)
    state = _initialize_state(
        state_root,
        None,
        scientist_config,
        sha256_file(manifest_path),
        expected_a40_uuid=expected_a40_uuid,
    )
    schedule = _schedule_from_frozen_file()
    completed = 0
    next_row: ScheduledAttempt | None = None
    for row in schedule:
        position = state["positions"][str(row.global_position)]
        if position.get("status") == "completed":
            if next_row is not None:
                raise RuntimeError("completed schedule position follows an incomplete position")
            completed += 1
            continue
        if next_row is None:
            next_row = row
        elif position.get("status") != "pending":
            raise RuntimeError("more than one schedule position is active")
    if next_row is None:
        return {"status": "preflight_passed", "completed": 50, "remaining": 0, "next_position": None}
    position = state["positions"][str(next_row.global_position)]
    if position.get("status") != "running":
        raise RuntimeError(f"next continuation position is not armed: {position}")
    paths = _validate_existing_campaign_state(state_root, next_row)
    execution, candidate, metadata = _load_pending_execution(paths=paths, row=next_row, position=position)
    validate_candidate_contract_api(
        candidate,
        expected_n_star=N_STAR if next_row.contract_id == "N" else None,
    )
    _parent_id, _parent_source, parent_sha256 = _validate_parent_selection(
        paths=paths,
        row=next_row,
        proposal={"parent_experiment_id": execution["parent_experiment_id"]},
    )
    if parent_sha256 != execution.get("candidate_source_before_sha256"):
        raise RuntimeError("continuation parent source SHA does not match its registered lineage")
    if metadata.get("evaluator_hashes") != _static_hashes():
        raise RuntimeError("continuation evaluator hashes do not match the live protected evaluator")
    contract_parameters = metadata.get("contract_parameters")
    if not isinstance(contract_parameters, Mapping):
        raise RuntimeError("continuation contract parameters are missing")
    if contract_parameters.get("paired_baseline_checkpoint") != transport.config.paired_baseline_checkpoint(next_row.contract_id):
        raise RuntimeError("continuation paired baseline checkpoint does not match the frozen transport")
    if next_row.contract_id == "N" and contract_parameters.get("n_star") != N_STAR:
        raise RuntimeError("continuation N contract does not preserve N_STAR")
    if next_row.contract_id == "T" and float(contract_parameters.get("time_seconds", -1)) != 300.0:
        raise RuntimeError("continuation T contract does not preserve 300 seconds")
    if execution.get("dispatch_status") != "launch_requested":
        raise RuntimeError("next continuation position is not armed for durable launch")
    remote_hashes = transport.verify_payload(
        remote_dir=str(execution.get("remote_payload_dir", "")),
        candidate=candidate,
        metadata=pathlib.Path(str(execution["metadata_path"])).resolve(),
    )
    return {
        "status": "preflight_passed",
        "completed": completed,
        "remaining": len(schedule) - completed,
        "next_position": next_row.as_dict(),
        "invocation_id": execution["invocation_id"],
        "candidate_source_sha256": execution["candidate_source_after_sha256"],
        "parent_experiment_id": execution["parent_experiment_id"],
        "parent_source_sha256": execution["candidate_source_before_sha256"],
        "remote_payload_dir": execution["remote_payload_dir"],
        "remote_payload_hashes": remote_hashes,
        "official_attempt_launched": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python3 -m study.contracts_2026.campaign_controller")
    sub = parser.add_subparsers(dest="command", required=True)
    dry = sub.add_parser("dry-run", help="stage and verify one context without invoking Scientist or A40")
    dry.add_argument("--position", type=int, default=1)
    dry.add_argument("--output-root", type=pathlib.Path)
    dry.add_argument("--manifest", type=pathlib.Path, default=FINAL_MANIFEST_PATH)
    dry.add_argument("--baseline-record", type=pathlib.Path, default=DEFAULT_BASELINE_RECORD_PATH)
    dry.add_argument("--scientist-bin", default=DEFAULT_SCIENTIST_BIN)
    dry.add_argument("--model", required=True)
    dry.add_argument("--reasoning", required=True)
    dry.add_argument("--scientist-timeout-seconds", type=float, default=600.0)
    dry.add_argument("--agent-backend", choices=("codex", "claude"), default="codex")
    dry.add_argument("--manifest-sha256")
    dry.add_argument("--a40-host", default=DEFAULT_A40_HOST)
    dry.add_argument("--a40-identity", default=DEFAULT_A40_IDENTITY)
    dry.add_argument("--a40-port", type=int, default=None)
    dry.add_argument("--a40-repository", default=DEFAULT_A40_REPOSITORY)
    dry.add_argument("--a40-python", default=None)
    dry.add_argument("--a40-study-venv", default=None)
    dry.add_argument("--a40-cache", default=None)
    dry.add_argument("--a40-state-root", default="/var/lib/autopareto/pair_a_state")
    dry.add_argument("--a40-paired-baseline-t-checkpoint", default=None)
    dry.add_argument("--a40-paired-baseline-n-checkpoint", default=None)
    preflight = sub.add_parser("preflight", help="verify the exact prepared continuation without launching it")
    preflight.add_argument("--state-root", type=pathlib.Path, required=True)
    preflight.add_argument("--manifest", type=pathlib.Path, required=True)
    preflight.add_argument("--manifest-sha256", required=True)
    preflight.add_argument("--scientist-bin", default=DEFAULT_SCIENTIST_BIN)
    preflight.add_argument("--model", required=True)
    preflight.add_argument("--reasoning", required=True)
    preflight.add_argument("--scientist-timeout-seconds", type=float, default=600.0)
    preflight.add_argument("--agent-backend", choices=("codex", "claude"), default="codex")
    preflight.add_argument("--a40-host", default=DEFAULT_A40_HOST)
    preflight.add_argument("--a40-identity", default=DEFAULT_A40_IDENTITY)
    preflight.add_argument("--a40-port", type=int, default=None)
    preflight.add_argument("--a40-repository", default=DEFAULT_A40_REPOSITORY)
    preflight.add_argument("--a40-python", required=True)
    preflight.add_argument("--a40-study-venv", required=True)
    preflight.add_argument("--a40-cache", required=True)
    preflight.add_argument("--a40-state-root", required=True)
    preflight.add_argument("--a40-paired-baseline-t-checkpoint", required=True)
    preflight.add_argument("--a40-paired-baseline-n-checkpoint", required=True)
    run = sub.add_parser("run", help="run the frozen 50-position campaign")
    run.add_argument("--execute", action="store_true", required=True)
    run.add_argument("--state-root", type=pathlib.Path, default=pathlib.Path("runs/pair-a"))
    run.add_argument("--baseline-record", type=pathlib.Path, default=DEFAULT_BASELINE_RECORD_PATH)
    run.add_argument("--manifest", type=pathlib.Path, required=True)
    run.add_argument("--manifest-sha256", required=True)
    run.add_argument("--scientist-bin", default=DEFAULT_SCIENTIST_BIN)
    run.add_argument("--model", required=True)
    run.add_argument("--reasoning", required=True)
    run.add_argument("--scientist-timeout-seconds", type=float, default=600.0)
    run.add_argument("--agent-backend", choices=("codex", "claude"), default="codex")
    run.add_argument(
        "--contracts", choices=("T,N", "T"), default="T,N",
        help="T runs only the 25 fixed-time positions as a single-arm study",
    )
    run.add_argument("--a40-host", default=DEFAULT_A40_HOST)
    run.add_argument("--a40-identity", default=DEFAULT_A40_IDENTITY)
    run.add_argument("--a40-port", type=int, default=None)
    run.add_argument("--a40-repository", default=DEFAULT_A40_REPOSITORY)
    run.add_argument("--a40-python", default=None)
    run.add_argument("--a40-study-venv", default=None)
    run.add_argument("--a40-cache", default=None)
    run.add_argument("--a40-state-root", default="/var/lib/autopareto/pair_a_state")
    run.add_argument("--a40-paired-baseline-t-checkpoint", default=None)
    run.add_argument("--a40-paired-baseline-n-checkpoint", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "agent_backend", "codex") == "claude" and args.scientist_bin == DEFAULT_SCIENTIST_BIN:
        args.scientist_bin = shutil.which("claude") or "claude"
    if args.command == "dry-run":
        scientist_config = resolve_scientist_config(
            model=args.model,
            reasoning=args.reasoning,
            scientist_bin=args.scientist_bin,
            timeout_seconds=args.scientist_timeout_seconds,
            agent_backend=args.agent_backend,
        )
        config = A40RemoteConfig(
            host=args.a40_host,
            port=args.a40_port,
            identity_file=args.a40_identity,
            repository=args.a40_repository,
            python_path=args.a40_python,
            study_venv=args.a40_study_venv,
            registered_cache=args.a40_cache,
            trusted_state_root=args.a40_state_root,
            paired_baseline_t_checkpoint=args.a40_paired_baseline_t_checkpoint,
            paired_baseline_n_checkpoint=args.a40_paired_baseline_n_checkpoint,
        )
        print(
            json.dumps(
                dry_run_position(
                    args.position,
                    args.output_root,
                    manifest_path=args.manifest.resolve(),
                    baseline_path=args.baseline_record.resolve(),
                    scientist_config=scientist_config,
                    a40_config=config,
                    expected_manifest_sha256=args.manifest_sha256,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "preflight":
        scientist_config = resolve_scientist_config(
            model=args.model,
            reasoning=args.reasoning,
            scientist_bin=args.scientist_bin,
            timeout_seconds=args.scientist_timeout_seconds,
            agent_backend=args.agent_backend,
        )
        config = A40RemoteConfig(
            host=args.a40_host,
            port=args.a40_port,
            identity_file=args.a40_identity,
            repository=args.a40_repository,
            python_path=args.a40_python,
            study_venv=args.a40_study_venv,
            registered_cache=args.a40_cache,
            trusted_state_root=args.a40_state_root,
            paired_baseline_t_checkpoint=args.a40_paired_baseline_t_checkpoint,
            paired_baseline_n_checkpoint=args.a40_paired_baseline_n_checkpoint,
        )
        result = preflight_campaign(
            state_root=args.state_root.resolve(),
            manifest_path=args.manifest.resolve(),
            transport=A40Transport(config),
            scientist_config=scientist_config,
            expected_manifest_sha256=args.manifest_sha256,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "run":
        scientist_config = resolve_scientist_config(
            model=args.model,
            reasoning=args.reasoning,
            scientist_bin=args.scientist_bin,
            timeout_seconds=args.scientist_timeout_seconds,
            agent_backend=args.agent_backend,
        )
        config = A40RemoteConfig(
            host=args.a40_host,
            port=args.a40_port,
            identity_file=args.a40_identity,
            repository=args.a40_repository,
            python_path=args.a40_python,
            study_venv=args.a40_study_venv,
            registered_cache=args.a40_cache,
            trusted_state_root=args.a40_state_root,
            paired_baseline_t_checkpoint=args.a40_paired_baseline_t_checkpoint,
            paired_baseline_n_checkpoint=args.a40_paired_baseline_n_checkpoint,
        )
        result = run_campaign(
            state_root=args.state_root.resolve(),
            baseline_path=args.baseline_record.resolve(),
            manifest_path=args.manifest.resolve(),
            transport=A40Transport(config),
            scientist_config=scientist_config,
            expected_manifest_sha256=args.manifest_sha256,
            active_contracts=tuple(args.contracts.split(",")),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
