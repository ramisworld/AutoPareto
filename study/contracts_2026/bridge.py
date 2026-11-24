"""Trusted local Scientist-to-A40 orchestration bridge.

The Scientist side of the study never receives this module, SSH configuration, or
any remote path.  It runs in the local macOS sandbox and emits a small proposal;
the trusted parent records the proposal, stages only ``train.py`` plus registered
metadata, and owns the A40 transport and protected execution.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import zlib
from typing import Any, Mapping, Sequence

from .agent_request import AgentRequest
from .context import VIEW_RESULT_FIELDS, build_campaign_view
from .scheduling import ATTEMPTS_PER_CONDITION
from .campaign import reconcile_same_arm_archive
from .isolation import AGENT_VISIBLE_PATHS, CLAUDE_SCIENTIST_TOOLS, run_agent_command, validate_context_paths
from .scientist_config import ScientistConfig


BRIDGE_SCHEMA = "autopareto_trusted_bridge_v1"
REMOTE_PAYLOAD_SCHEMA = "autopareto_a40_payload_v1"
REMOTE_RESULT_SCHEMA = "autopareto_a40_result_v1"
DEFAULT_A40_HOST = "<pod-id>-<suffix>@ssh.runpod.io"
DEFAULT_A40_IDENTITY = "~/.ssh/id_ed25519"
DEFAULT_A40_REPOSITORY = "/opt/autopareto_trusted"
DEFAULT_PAIRED_BASELINE_T_CHECKPOINT = "/var/lib/autopareto/pair_a_state/nonstudy_calibration/calibration/run1/result.checkpoint.pt"
DEFAULT_PAIRED_BASELINE_N_CHECKPOINT = "/var/lib/autopareto/nonstudy_pair_a_n_final_20260905/nonstudy_tests/n_smoke_artifacts/result.checkpoint.pt"

_FORBIDDEN_AGENT_ENV_PREFIXES = (
    "SSH_",
    "AUTOPARETO_REMOTE",
    "RUNPOD_",
)
_FORBIDDEN_AGENT_TOKENS = {"ssh", "scp", "sftp", "runpod"}
_ANSI_SEQUENCE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\)|[@-_])")
_CANONICAL_STATUS = re.compile(r"(?:0|[1-9][0-9]*)")
_FILE_RESULT = re.compile(r"AUTOPARETO_FILE_RESULT:([0-9]+):([0-9a-f]{64})")
_TRANSFER_RESULT = re.compile(r"AUTOPARETO_TRANSFER_RESULT:([A-Za-z0-9+/=]+)")
_DURABLE_RESULT_BEGIN = re.compile(r"AUTOPARETO_DURABLE_RESULT_BEGIN:([0-9a-f]{32})")
_DURABLE_RESULT_META = re.compile(r"AUTOPARETO_DURABLE_RESULT_META:((?:0|[1-9][0-9]*)):([0-9a-f]{64})")
_DURABLE_RESULT_DATA = re.compile(r"AUTOPARETO_DURABLE_RESULT_DATA:([A-Za-z0-9+/=]+)")
_DURABLE_RESULT_END = re.compile(r"AUTOPARETO_DURABLE_RESULT_END:([0-9a-f]{32})")
_DURABLE_RECORD_BEGIN = re.compile(r"AUTOPARETO_DURABLE_RECORD_BEGIN:([0-9a-f]{32})")
_DURABLE_RECORD_META = re.compile(r"AUTOPARETO_DURABLE_RECORD_META:((?:0|[1-9][0-9]*)):([0-9a-f]{64})")
_DURABLE_RECORD_DATA = re.compile(r"AUTOPARETO_DURABLE_RECORD_DATA:([A-Za-z0-9+/=]+)")
_DURABLE_RECORD_END = re.compile(r"AUTOPARETO_DURABLE_RECORD_END:([0-9a-f]{32})")
_GPU_IDENTITY_RESULT = re.compile(r"AUTOPARETO_GPU_IDENTITY:(GPU-[A-Za-z0-9-]+)")
_PTY_FRAME_CHUNK_CHARS = 32
_DURABLE_READBACK_ATTEMPTS = 3
_DURABLE_READBACK_TIMEOUT_SECONDS = 60.0
_METADATA_KEYS = {
    "schema",
    "study_id",
    "campaign_id",
    "contract_id",
    "attempt_number",
    "training_seed",
    "parent_experiment_id",
    "hypothesis",
    "intended_change",
    "change_category",
    "candidate_source_before_sha256",
    "candidate_source_after_sha256",
    "candidate_patch",
    "starting_source_hash",
    "evaluator_hashes",
    "dataset_hashes",
    "tokenizer_hash",
    "program_sha256",
    "agent_context_hashes",
    "agent_model_identifier",
    "reasoning_setting",
    "api_configuration_available",
    "contract_parameters",
    "protected_prepare_sha256",
    "study_runtime_sha256",
}


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    return sha256_bytes(path.read_bytes())


def _normalize_pty_output(value: str) -> str:
    """Remove only recognized ANSI sequences and C0 terminal controls."""
    value = _ANSI_SEQUENCE.sub("", value)
    return "".join(character for character in value if character in "\n\t" or ord(character) >= 0x20)


class A40TransportFrameError(RuntimeError):
    """A machine-readable PTY frame was not recovered unambiguously."""


class A40TransportCommandError(RuntimeError):
    """The SSH/script transport failed before a valid PTY frame was recovered."""


class A40TransportTimeout(RuntimeError):
    """A bounded PTY command exceeded its transport timeout."""


def _json_hash(value: Mapping[str, Any]) -> str:
    return sha256_bytes((json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode())


def _portable_relative(value: str) -> str:
    path = pathlib.PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"bridge path must be relative: {value}")
    return path.as_posix()


def _proposal_objects(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            try:
                return _proposal_objects(json.loads(stripped))
            except json.JSONDecodeError:
                pass
        objects: list[dict[str, Any]] = []
        for line in value.splitlines():
            try:
                parsed = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            objects.extend(_proposal_objects(parsed))
        return objects
    if isinstance(value, Mapping):
        if {"hypothesis", "intended_change", "change_category"}.issubset(value):
            return [dict(value)]
        objects = []
        for key in ("item", "message", "text", "output", "content"):
            if key in value:
                objects.extend(_proposal_objects(value[key]))
        return objects
    if isinstance(value, list):
        objects: list[dict[str, Any]] = []
        for item in value:
            objects.extend(_proposal_objects(item))
        return objects
    return []


def _read_json_line(output: str) -> dict[str, Any]:
    # Codex --json is an event stream.  During a turn it can contain several
    # schema-shaped progress messages, but the proposal is only the terminal
    # agent message immediately preceding the one terminal turn.completed
    # receipt.  Parsing every message as a proposal would make normal progress
    # look ambiguous; accepting a non-terminal message would be worse.  Use
    # the durable event boundary when it is present and retain the legacy
    # strict envelope parser for non-Codex callers/tests.
    events: list[dict[str, Any]] = []
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, Mapping):
            events.append(dict(event))
    terminal_indexes = [
        index for index, event in enumerate(events)
        if event.get("type") == "turn.completed"
    ]
    if terminal_indexes:
        if len(terminal_indexes) != 1 or terminal_indexes[0] != len(events) - 1:
            raise ValueError("Scientist emitted malformed or ambiguous terminal Codex event framing")
        terminal_index = terminal_indexes[0]
        if terminal_index == 0:
            raise ValueError("Scientist terminal Codex event has no proposal message")
        message_event = events[terminal_index - 1]
        item = message_event.get("item")
        if (
            message_event.get("type") != "item.completed"
            or not isinstance(item, Mapping)
            or item.get("type") != "agent_message"
            or not isinstance(item.get("text"), str)
        ):
            raise ValueError("Scientist terminal Codex proposal message is malformed")
        candidates = _proposal_objects(item["text"])
        if len(candidates) != 1:
            raise ValueError("Scientist terminal Codex proposal is missing or ambiguous")
        return validate_scientist_proposal(candidates[0])

    candidates = _proposal_objects(output)
    if not candidates:
        raise ValueError("Scientist did not emit a proposal object in the Codex response")
    canonical = [validate_scientist_proposal(candidate) for candidate in candidates]
    encoded = {json.dumps(item, sort_keys=True, separators=(",", ":")) for item in canonical}
    if len(encoded) != 1:
        raise ValueError("Scientist emitted conflicting proposal objects")
    return canonical[0]


def _reject_remote_secrets(value: Any, path: str = "proposal") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in ("ssh", "runpod", "identity_file", "remote_repo", "private_key")):
                raise ValueError(f"remote-access field is forbidden in Scientist payload: {path}.{key}")
            _reject_remote_secrets(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_remote_secrets(child, f"{path}[{index}]")


def validate_scientist_proposal(proposal: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the only semantic fields the local Scientist may return."""
    _reject_remote_secrets(proposal)
    required = {"hypothesis", "intended_change", "change_category"}
    missing = sorted(required - set(proposal))
    if missing:
        raise ValueError(f"Scientist proposal is missing fields: {missing}")
    allowed = required | {"parent_experiment_id"}
    extra = sorted(set(proposal) - allowed)
    if extra:
        raise ValueError(f"Scientist proposal contains unregistered fields: {extra}")
    result = {key: str(proposal[key]) for key in required}
    if "parent_experiment_id" in proposal:
        result["parent_experiment_id"] = (
            None if proposal["parent_experiment_id"] is None else str(proposal["parent_experiment_id"])
        )
    return result


def canonicalize_scientist_proposal(
    proposal: Mapping[str, Any],
    *,
    phase: str,
    expected_parent_experiment_id: str | None = None,
) -> dict[str, Any]:
    """Keep semantic decisions model-owned and fill bookkeeping deterministically."""
    semantic = validate_scientist_proposal(proposal)
    if phase == "select_parent":
        parent = semantic.get("parent_experiment_id")
        if not parent:
            raise ValueError("parent selection proposal must include parent_experiment_id")
        return semantic
    if phase == "edit_child":
        if not expected_parent_experiment_id:
            raise ValueError("child edit requires the controller-selected parent")
        supplied = semantic.get("parent_experiment_id")
        if supplied is not None and supplied != expected_parent_experiment_id:
            raise ValueError("Scientist cannot override the controller-selected parent")
        return {**semantic, "parent_experiment_id": expected_parent_experiment_id}
    if phase == "semantic":
        return semantic
    raise ValueError(f"unknown Scientist proposal phase: {phase}")


def _scientist_environment(
    context: pathlib.Path, extra: Mapping[str, str] | None
) -> tuple[dict[str, str], pathlib.Path]:
    """Build a minimal environment with no inherited SSH or remote credentials."""
    source_codex_home = pathlib.Path(os.environ.get("CODEX_HOME") or pathlib.Path.home() / ".codex").expanduser().resolve()
    source_auth = source_codex_home / "auth.json"
    if not source_auth.is_file():
        raise FileNotFoundError(f"Codex authentication file is missing: {source_auth}")
    relay = pathlib.Path(tempfile.mkdtemp(prefix="autopareto-codex-home-", dir="/tmp"))
    (relay / "auth.json").symlink_to(source_auth)
    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": str(context.resolve()),
        "CODEX_HOME": str(relay),
        "TMPDIR": "/tmp",
        "PYTHONNOUSERSITE": "1",
        "LC_ALL": "C",
        "AUTOPARETO_SCIENTIST_SANDBOX": "1",
    }
    for key, value in (extra or {}).items():
        if any(key.upper().startswith(prefix) for prefix in _FORBIDDEN_AGENT_ENV_PREFIXES):
            raise ValueError(f"forbidden remote-access environment variable: {key}")
        env[key] = str(value)
    return env, relay


# Inherited provider variables could silently switch Claude Code to API-key
# billing, another endpoint or another provider, so none are passed through.
_CLAUDE_FORBIDDEN_ENV_PREFIXES = _FORBIDDEN_AGENT_ENV_PREFIXES + ("ANTHROPIC_", "CLAUDE_", "AWS_", "GOOGLE_", "VERTEX_")
# The campaign never runs on paid overage: it halts once any seat usage window
# reaches this fraction, so the next call cannot spill into extra usage.
CLAUDE_USAGE_HALT_UTILIZATION = 0.95


def _claude_scientist_environment(
    scientist_config: ScientistConfig, extra: Mapping[str, str] | None
) -> dict[str, str]:
    """Minimal Claude Code environment: the user's own login, no inherited credentials."""
    env = {
        "PATH": f"{pathlib.Path(scientist_config.scientist_bin).parent}:/usr/bin:/bin:/usr/local/bin",
        "HOME": str(pathlib.Path.home()),
        "USER": os.environ.get("USER", ""),
        "LOGNAME": os.environ.get("LOGNAME", os.environ.get("USER", "")),
        "TMPDIR": "/tmp",
        "LC_ALL": "C",
        "TERM": "dumb",
        "DISABLE_AUTOUPDATER": "1",
        # Refusal, rate-limit and compaction fallbacks would otherwise let
        # another model take over part of the run; with this set they fail.
        "CLAUDE_CODE_NO_MODEL_FALLBACK": "1",
        "AUTOPARETO_SCIENTIST_SANDBOX": "1",
    }
    for key, value in (extra or {}).items():
        if any(key.upper().startswith(prefix) for prefix in _CLAUDE_FORBIDDEN_ENV_PREFIXES):
            raise ValueError(f"forbidden Claude Code Scientist environment variable: {key}")
        env[key] = str(value)
    return env


def _read_claude_stream(output: str, scientist_config: ScientistConfig) -> dict[str, Any]:
    """Accept a Claude Code proposal only if the run used exactly the registered agent."""
    events: list[dict[str, Any]] = []
    # stream-json is newline-delimited; str.splitlines would also split inside
    # a result string at U+2028/U+0085 and lose the result event.
    for line in output.split("\n"):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, Mapping):
            events.append(dict(event))
    inits = [event for event in events if event.get("type") == "system" and event.get("subtype") == "init"]
    results = [event for event in events if event.get("type") == "result"]
    if len(inits) != 1 or len(results) != 1 or events[-1] is not results[0]:
        raise ValueError("Claude Code stream must contain one init event and end with one result event")
    init, result = inits[0], results[0]
    if init.get("model") != scientist_config.model:
        raise ValueError(f"Claude Code ran model {init.get('model')!r}, not {scientist_config.model!r}")
    if init.get("permissionMode") != scientist_config.approval_policy:
        raise ValueError(f"Claude Code permission mode was {init.get('permissionMode')!r}")
    # --json-schema makes Claude Code add one tool of its own, through which the
    # model returns the schema-validated proposal.
    allowed_tools = set(CLAUDE_SCIENTIST_TOOLS.split(",")) | {"StructuredOutput"}
    unexpected_tools = sorted(set(init.get("tools") or []) - allowed_tools)
    if unexpected_tools:
        raise ValueError(f"Claude Code exposed unregistered tools: {unexpected_tools}")
    if init.get("mcp_servers"):
        raise ValueError("Claude Code loaded MCP servers")
    if init.get("apiKeySource") != "none":
        raise ValueError(f"Claude Code must use the subscription login, not {init.get('apiKeySource')!r}")
    frozen_version = scientist_config.codex_version.split(" ", 1)[0]
    if init.get("claude_code_version") != frozen_version:
        raise ValueError(f"Claude Code version {init.get('claude_code_version')!r} is not the frozen {frozen_version!r}")
    for event in events:
        message = event.get("message")
        if event.get("type") == "assistant" and isinstance(message, Mapping) and message.get("model") != scientist_config.model:
            raise ValueError(f"Claude Code answered with model {message.get('model')!r}")
        if event.get("type") == "rate_limit_event":
            info = event.get("rate_limit_info") or {}
            if info.get("isUsingOverage") or info.get("overageInUse"):
                raise ValueError("Claude Code call used paid overage; the campaign does not run on overage")
            if info.get("status") not in ("allowed", "allowed_warning"):
                raise ValueError(f"Claude seat usage limit status is {info.get('status')!r}")
            for window, state in (info.get("unifiedWindows") or {}).items():
                utilization = float(state.get("utilization", 0)) if isinstance(state, Mapping) else 0.0
                if utilization >= CLAUDE_USAGE_HALT_UTILIZATION:
                    raise ValueError(f"Claude seat {window} usage is at {utilization:.0%}; halting before overage")
    if result.get("subtype") != "success" or result.get("is_error"):
        raise ValueError(f"Claude Code run did not succeed: {result.get('subtype')!r}")
    usage = result.get("modelUsage") or {}
    if scientist_config.model not in usage:
        raise ValueError("Claude Code reported no usage for the requested model")
    for model, counts in usage.items():
        if model != scientist_config.model and (not isinstance(counts, Mapping) or counts.get("outputTokens", 1) != 0):
            raise ValueError(f"Claude Code used another model to produce output: {model!r}")
    structured = result.get("structured_output")
    if structured is None:
        raise ValueError("Claude Code returned no schema-validated structured output")
    candidates = _proposal_objects(structured)
    if len(candidates) != 1:
        raise ValueError("Claude Code proposal is missing or ambiguous")
    return validate_scientist_proposal(candidates[0])


def _guard_scientist_command(command: Sequence[str]) -> None:
    for token in command:
        lowered = pathlib.PurePosixPath(str(token)).name.lower()
        if lowered in _FORBIDDEN_AGENT_TOKENS or any(word in str(token).lower() for word in ("runpod", "ssh://")):
            raise ValueError("Scientist command contains a forbidden remote-access token")


def _context_hashes(context: pathlib.Path) -> dict[str, str]:
    validate_context_paths(tuple(path.name for path in sorted(context.iterdir()) if path.is_file()))
    return {name: sha256_file(context / name) for name in AGENT_VISIBLE_PATHS}


@dataclass(frozen=True)
class ScientistHandoff:
    bridge_schema: str
    campaign_id: str
    contract_id: str
    attempt_number: int
    proposal: dict[str, Any]
    candidate_source_before_sha256: str
    candidate_source_after_sha256: str
    program_sha256: str
    agent_context_hashes: dict[str, str]
    agent_request: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "bridge_schema": self.bridge_schema,
            "campaign_id": self.campaign_id,
            "contract_id": self.contract_id,
            "attempt_number": self.attempt_number,
            "proposal": self.proposal,
            "candidate_source_before_sha256": self.candidate_source_before_sha256,
            "candidate_source_after_sha256": self.candidate_source_after_sha256,
            "program_sha256": self.program_sha256,
            "agent_context_hashes": self.agent_context_hashes,
            "agent_request": self.agent_request,
        }


def run_local_scientist(
    *,
    command: Sequence[str],
    context: pathlib.Path,
    campaign_id: str,
    contract_id: str,
    attempt_number: int,
    model_identifier: str | None = None,
    reasoning_setting: str | None = None,
    api_configuration: Mapping[str, Any] | None = None,
    system_context_text: str | None = None,
    extra_environment: Mapping[str, str] | None = None,
    phase: str = "semantic",
    expected_parent_experiment_id: str | None = None,
    scientist_config: ScientistConfig | None = None,
    diagnostic_root: pathlib.Path | None = None,
) -> tuple[ScientistHandoff, subprocess.CompletedProcess[str]]:
    """Run one harmless/local Scientist interaction and record all visible hashes."""
    _guard_scientist_command(command)
    context = context.resolve()
    if not context.is_dir():
        raise ValueError(f"Scientist context is not a directory: {context}")
    runtime_command = list(command)
    if scientist_config is None:
        if model_identifier is None or reasoning_setting is None:
            raise ValueError("ScientistConfig or model/reasoning values are required")
        scientist_config = ScientistConfig(
            model=model_identifier,
            reasoning=reasoning_setting,
            scientist_bin=str(runtime_command[0]),
        )
    scientist_config.validate()
    if model_identifier is not None and model_identifier != scientist_config.model:
        raise ValueError("Scientist model argument disagrees with ScientistConfig")
    if reasoning_setting is not None and reasoning_setting != scientist_config.reasoning:
        raise ValueError("Scientist reasoning argument disagrees with ScientistConfig")
    model_identifier = scientist_config.model
    reasoning_setting = scientist_config.reasoning
    before_hashes = _context_hashes(context)
    program_text = (context / "program.md").read_text()
    context_text = {name: (context / name).read_text() for name in AGENT_VISIBLE_PATHS}
    request = AgentRequest(
        campaign_id=campaign_id,
        contract_id=contract_id,
        attempt_number=attempt_number,
        model_identifier=model_identifier,
        reasoning_setting=reasoning_setting,
        program_path="program.md",
        program_text=program_text,
        context_files=context_text,
        api_configuration=dict(api_configuration) if api_configuration is not None else None,
        system_context_text=system_context_text,
    ).as_dict()
    if scientist_config.agent_backend == "claude":
        environment = _claude_scientist_environment(scientist_config, extra_environment)
        codex_home_relay = None
    else:
        environment, codex_home_relay = _scientist_environment(context, extra_environment)
    if runtime_command[:1] != [scientist_config.scientist_bin]:
        raise ValueError("Scientist command executable disagrees with ScientistConfig")
    if diagnostic_root is None:
        # Production callers pass the campaign state root.  The fallback keeps
        # legacy non-study preflights self-contained in their temporary tree.
        diagnostic_root = context.parent / "scientist-diagnostics"
    try:
        completed = run_agent_command(
            runtime_command,
            context,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            env=environment,
            scientist_config=scientist_config,
            timeout_seconds=scientist_config.timeout_seconds,
            diagnostic_root=diagnostic_root,
            invocation_metadata={
                "campaign_id": campaign_id,
                "contract_id": contract_id,
                "attempt_number": attempt_number,
                "phase": phase,
                "model": scientist_config.model,
                "reasoning": scientist_config.reasoning,
                "codex_version": scientist_config.codex_version,
            },
        )
    finally:
        if codex_home_relay is not None:
            shutil.rmtree(codex_home_relay, ignore_errors=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Scientist sandbox command failed: {completed.returncode}: {completed.stderr}")
    proposal = canonicalize_scientist_proposal(
        _read_claude_stream(completed.stdout, scientist_config)
        if scientist_config.agent_backend == "claude"
        else _read_json_line(completed.stdout),
        phase=phase,
        expected_parent_experiment_id=expected_parent_experiment_id,
    )
    after_hashes = _context_hashes(context)
    if any(before_hashes[name] != after_hashes[name] for name in AGENT_VISIBLE_PATHS if name != "train.py"):
        raise RuntimeError("Scientist modified a non-train.py context file")
    handoff = ScientistHandoff(
        bridge_schema=BRIDGE_SCHEMA,
        campaign_id=campaign_id,
        contract_id=contract_id,
        attempt_number=attempt_number,
        proposal=proposal,
        candidate_source_before_sha256=before_hashes["train.py"],
        candidate_source_after_sha256=after_hashes["train.py"],
        program_sha256=before_hashes["program.md"],
        agent_context_hashes=after_hashes,
        agent_request=request,
    )
    return handoff, completed


def build_registered_metadata(
    *,
    handoff: ScientistHandoff,
    study_id: str,
    training_seed: int,
    starting_source_hash: str,
    evaluator_hashes: Mapping[str, str],
    dataset_hashes: Mapping[str, str],
    tokenizer_hash: str,
    contract_parameters: Mapping[str, Any],
    protected_prepare_sha256: str,
    study_runtime_sha256: str,
) -> dict[str, Any]:
    metadata = {
        "schema": REMOTE_PAYLOAD_SCHEMA,
        "study_id": study_id,
        "campaign_id": handoff.campaign_id,
        "contract_id": handoff.contract_id,
        "attempt_number": handoff.attempt_number,
        "training_seed": int(training_seed),
        "parent_experiment_id": handoff.proposal.get("parent_experiment_id"),
        "hypothesis": handoff.proposal["hypothesis"],
        "intended_change": handoff.proposal["intended_change"],
        "change_category": handoff.proposal["change_category"],
        "candidate_source_before_sha256": handoff.candidate_source_before_sha256,
        "candidate_source_after_sha256": handoff.candidate_source_after_sha256,
        "starting_source_hash": starting_source_hash,
        "evaluator_hashes": dict(sorted(evaluator_hashes.items())),
        "dataset_hashes": dict(sorted(dataset_hashes.items())),
        "tokenizer_hash": tokenizer_hash,
        "program_sha256": handoff.program_sha256,
        "agent_context_hashes": dict(sorted(handoff.agent_context_hashes.items())),
        "agent_model_identifier": handoff.agent_request["model_identifier"],
        "reasoning_setting": handoff.agent_request["reasoning_setting"],
        "api_configuration_available": bool(handoff.agent_request["api_configuration_available"]),
        "contract_parameters": dict(contract_parameters),
        "protected_prepare_sha256": protected_prepare_sha256,
        "study_runtime_sha256": study_runtime_sha256,
    }
    if set(metadata) - _METADATA_KEYS:
        raise AssertionError("bridge metadata schema drift")
    metadata["metadata_sha256"] = _json_hash(metadata)
    return metadata


def stage_remote_payload(*, destination: pathlib.Path, candidate: pathlib.Path, metadata: Mapping[str, Any]) -> pathlib.Path:
    """Create a transfer directory containing exactly candidate plus metadata."""
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite remote payload staging directory: {destination}")
    candidate = candidate.resolve()
    if candidate.name != "train.py" or candidate.is_symlink() or not candidate.is_file():
        raise ValueError("only a regular train.py may be staged for transfer")
    destination.mkdir(parents=True)
    shutil.copy2(candidate, destination / "train.py")
    (destination / "registered_metadata.json").write_text(
        json.dumps(dict(metadata), indent=2, sort_keys=True) + "\n"
    )
    if sorted(path.name for path in destination.iterdir()) != ["registered_metadata.json", "train.py"]:
        raise AssertionError("remote payload contains an unregistered file")
    return destination


@dataclass(frozen=True)
class A40RemoteConfig:
    host: str = DEFAULT_A40_HOST
    port: int | None = None
    identity_file: str = DEFAULT_A40_IDENTITY
    repository: str = DEFAULT_A40_REPOSITORY
    payload_root: str = "/opt/autopareto_trusted/payloads"
    python_path: str | None = None
    registered_cache: str | None = None
    study_venv: str | None = None
    trusted_state_root: str | None = None
    paired_baseline_t_checkpoint: str | None = DEFAULT_PAIRED_BASELINE_T_CHECKPOINT
    paired_baseline_n_checkpoint: str | None = DEFAULT_PAIRED_BASELINE_N_CHECKPOINT

    def paired_baseline_checkpoint(self, contract_id: str) -> str:
        value = self.paired_baseline_t_checkpoint if contract_id == "T" else self.paired_baseline_n_checkpoint
        if not value:
            raise ValueError(f"no registered paired baseline checkpoint configured for contract {contract_id}")
        return value

    def ssh_base(self) -> list[str]:
        command = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=20",
            "-o",
            "ServerAliveInterval=10",
            "-o",
            "ServerAliveCountMax=3",
            "-i",
            os.path.expanduser(self.identity_file),
        ]
        if self.port is not None:
            if not (1 <= int(self.port) <= 65535):
                raise ValueError("A40 SSH port must be in the valid range")
            command.extend(("-p", str(int(self.port))))
        command.append(self.host)
        return command


class A40Transport:
    """Orchestrator-owned A40 transport; never serialized into Scientist context."""

    def __init__(self, config: A40RemoteConfig, runner_module: str = "study.contracts_2026.remote_runner"):
        self.config = config
        self.runner_module = runner_module

    def _record_pty_diagnostic(
        self,
        *,
        operation: str,
        nonce: str,
        raw_output: bytes,
        normalized_output: str,
        begin: str,
        end: str,
        completed: subprocess.CompletedProcess[bytes] | None = None,
    ) -> None:
        """Persist bounded transport metadata without persisting result payloads."""
        try:
            root = pathlib.Path(tempfile.gettempdir()) / "autopareto-pty-diagnostics"
            root.mkdir(mode=0o700, parents=True, exist_ok=True)
            normalized_bytes = normalized_output.encode()
            payload = {
                "schema": "autopareto_pty_diagnostic_v1",
                "operation": operation,
                "nonce": nonce,
                "stdout_bytes": len(raw_output),
                "normalized_stdout_bytes": len(normalized_bytes),
                "stdout_sha256": sha256_bytes(raw_output),
                "normalized_stdout_sha256": sha256_bytes(normalized_bytes),
                "begin_occurrences": normalized_output.count(begin),
                "end_occurrences": normalized_output.count(end),
                "returncode": completed.returncode if completed is not None else None,
                "stderr_bytes": len(completed.stderr or b"") if completed is not None else None,
                "stderr_sha256": sha256_bytes(completed.stderr or b"") if completed is not None else None,
                "normalized_head": normalized_output[:512],
                "normalized_tail": normalized_output[-512:],
            }
            destination = root / f"{int(time.time() * 1_000_000)}-{nonce}.json"
            temporary = destination.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
            os.replace(temporary, destination)
        except OSError:
            # Diagnostics must never alter transport classification or campaign state.
            return

    def _run_pty_lines(
        self,
        lines: Sequence[str],
        *,
        operation: str = "pty_command",
        timeout_seconds: float | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        """Run shell lines through the endpoint's required PTY and recover output."""
        if not shutil.which("script"):
            raise RuntimeError("PTY-required A40 endpoint needs the local script utility")
        nonce = uuid.uuid4().hex
        begin = f"AUTOPARETO_BEGIN:{nonce}"
        end = f"AUTOPARETO_END:{nonce}"
        shell_input = (
            # The RunPod endpoint echoes input by default.  Disable echo only
            # inside this disposable PTY session so echoed shell commands can
            # never be mistaken for transport framing.  The setting disappears
            # with the session even if the command itself fails.
            "stty -echo\n"
            f"printf '%s\\n' {shlex.quote(begin)}\n"
            + "\n".join(lines)
            + "\nremote_status=$?\n"
            + f"printf '%s:%s\\n' {shlex.quote(end)} \"$remote_status\"\n"
            + "stty echo\n"
            + "exit\n"
        )
        # script owns the PTY; ssh -tt makes the RunPod login shell accept stdin.
        ssh_command = ["script", "-q", "/dev/null", "ssh", "-tt", *self.config.ssh_base()[1:]]
        try:
            completed = subprocess.run(
                ssh_command,
                input=shell_input.encode(),
                capture_output=True,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raw_output = exc.output if isinstance(exc.output, bytes) else b""
            normalized = _normalize_pty_output(raw_output.decode(errors="replace"))
            self._record_pty_diagnostic(
                operation=operation,
                nonce=nonce,
                raw_output=raw_output,
                normalized_output=normalized,
                begin=begin,
                end=end,
            )
            raise A40TransportTimeout(
                f"A40 PTY command timed out after {timeout_seconds}s: {operation}"
            ) from exc
        normalized = _normalize_pty_output(completed.stdout.decode(errors="replace"))
        output_lines = normalized.splitlines()
        begin_positions = [index for index, line in enumerate(output_lines) if line == begin]
        end_positions = [
            index for index, line in enumerate(output_lines)
            if line.startswith(end + ":") and _CANONICAL_STATUS.fullmatch(line[len(end) + 1:])
        ]
        if completed.returncode != 0 and not end_positions:
            stderr = completed.stderr.decode(errors="replace") if completed.stderr else ""
            detail = (stderr or normalized)[-2000:]
            self._record_pty_diagnostic(
                operation=operation,
                nonce=nonce,
                raw_output=completed.stdout,
                normalized_output=normalized,
                begin=begin,
                end=end,
                completed=completed,
            )
            raise A40TransportCommandError(
                f"A40 PTY/SSH command failed before a valid frame: returncode={completed.returncode}; "
                f"detail={detail}"
            )
        if len(begin_positions) != 1 or len(end_positions) != 1 or end_positions[0] <= begin_positions[0]:
            self._record_pty_diagnostic(
                operation=operation,
                nonce=nonce,
                raw_output=completed.stdout,
                normalized_output=normalized,
                begin=begin,
                end=end,
                completed=completed,
            )
            raise A40TransportFrameError("A40 PTY command markers were missing, duplicated, or malformed")
        end_index = end_positions[0]
        payload = "\n".join(output_lines[begin_positions[0] + 1:end_index])
        status_text = output_lines[end_index][len(end) + 1:]
        remote_returncode = int(status_text)
        result = subprocess.CompletedProcess(ssh_command, remote_returncode, payload.encode(), payload.encode())
        if remote_returncode != 0:
            raise RuntimeError(f"A40 SSH command failed ({remote_returncode}): {payload}")
        return result

    def _run(
        self,
        remote_command: str,
        *,
        input_text: str | None = None,
        operation: str = "pty_command",
        timeout_seconds: float | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        if input_text is not None:
            raise RuntimeError("raw stdin transfer is disabled for the PTY-required A40 transport")
        return self._run_pty_lines(
            [remote_command], operation=operation, timeout_seconds=timeout_seconds
        )

    def _run_read_only_pty(self, remote_command: str, *, operation: str) -> subprocess.CompletedProcess[bytes]:
        """Retry only idempotent receipt/result reads after transport loss."""
        last_error: Exception | None = None
        for attempt in range(_DURABLE_READBACK_ATTEMPTS):
            try:
                return self._run(
                    remote_command,
                    operation=operation,
                    timeout_seconds=_DURABLE_READBACK_TIMEOUT_SECONDS,
                )
            except (A40TransportCommandError, A40TransportFrameError, A40TransportTimeout) as exc:
                last_error = exc
                if attempt + 1 < _DURABLE_READBACK_ATTEMPTS:
                    time.sleep(attempt + 1)
        assert last_error is not None
        raise last_error

    def _send_file(self, local_path: pathlib.Path, remote_path: str) -> None:
        raise RuntimeError("_send_file is not used by the PTY-safe chunked transport")

    def verify_trusted_files(self, expected_hashes: Mapping[str, str]) -> dict[str, str]:
        """Verify the trusted checkout against manifest-relative SHA-256 values."""
        if not expected_hashes:
            raise ValueError("trusted deployment hash set must not be empty")
        normalized: dict[str, str] = {}
        for relative, expected in expected_hashes.items():
            safe = _portable_relative(str(relative))
            if not re.fullmatch(r"[0-9a-f]{64}", str(expected)):
                raise ValueError(f"trusted deployment hash is malformed: {relative}")
            normalized[safe] = str(expected)
        root = pathlib.PurePosixPath(self.config.repository)
        request = json.dumps(sorted(normalized), separators=(",", ":"))
        nonce = uuid.uuid4().hex
        checker = (
            "import base64,hashlib,json,pathlib,sys; root=pathlib.Path(sys.argv[1]); "
            "paths=json.loads(sys.argv[2]); result={}; "
            "exec(\"for rel in paths:\\n p=root/rel\\n "
            "if p.is_symlink() or not p.is_file(): raise SystemExit('trusted file missing: '+rel)\\n "
            "b=p.read_bytes(); result[rel]=hashlib.sha256(b).hexdigest()\") ; "
            "b=json.dumps(result,sort_keys=True,separators=(',',':')).encode(); "
            "encoded=base64.b64encode(b).decode('ascii'); nonce=sys.argv[3]; "
            "print('AUTOPARETO_TRUSTED_FILES_BEGIN:'+nonce); "
            "print('AUTOPARETO_TRUSTED_FILES_META:%d:%s' % (len(b),hashlib.sha256(b).hexdigest())); "
            "[print('AUTOPARETO_TRUSTED_FILES_DATA:'+encoded[i:i+32]) for i in range(0,len(encoded),32)]; "
            "print('AUTOPARETO_TRUSTED_FILES_END:'+nonce)")
        completed = self._run(
            f"python3 -c {shlex.quote(checker)} {shlex.quote(str(root))} {shlex.quote(request)} {shlex.quote(nonce)}"
        )
        try:
            observed = json.loads(self._read_framed_base64_payload(
                completed,
                label="trusted deployment hash",
                nonce=nonce,
                begin_prefix="AUTOPARETO_TRUSTED_FILES_BEGIN",
                meta_prefix="AUTOPARETO_TRUSTED_FILES_META",
                data_prefix="AUTOPARETO_TRUSTED_FILES_DATA",
                end_prefix="AUTOPARETO_TRUSTED_FILES_END",
            ))
        except json.JSONDecodeError as exc:
            raise RuntimeError("trusted deployment hash frame payload was not valid JSON") from exc
        if not isinstance(observed, dict) or observed != normalized:
            raise RuntimeError(f"trusted deployment hash mismatch: {observed} != {normalized}")
        return {str(path): str(digest) for path, digest in observed.items()}

    def verify_gpu_identity(self, expected_uuid: str) -> str:
        """Verify the live remote GPU UUID before official state creation."""
        if not re.fullmatch(r"GPU-[A-Za-z0-9-]+", expected_uuid):
            raise ValueError("GPU UUID is malformed")
        command = (
            "import subprocess; value=subprocess.check_output([\"nvidia-smi\",\"--query-gpu=uuid\",\"--format=csv,noheader,nounits\"], "
            "text=True).strip(); print(\"AUTOPARETO_GPU_IDENTITY:\"+value)"
        )
        completed = self._run(f"python3 -c {shlex.quote(command)}")
        normalized_output = _normalize_pty_output(completed.stdout.decode(errors="replace"))
        matches = [
            match for match in (_GPU_IDENTITY_RESULT.fullmatch(line) for line in normalized_output.splitlines())
            if match is not None
        ]
        if len(matches) != 1 or matches[0].group(1) != expected_uuid:
            observed = matches[0].group(1) if len(matches) == 1 else None
            raise RuntimeError(f"live GPU UUID mismatch: {observed!r} != {expected_uuid!r}")
        return expected_uuid

    def transfer(self, *, payload: pathlib.Path, remote_run_id: str) -> str:
        """Transfer the two registered payload files through PTY-safe hex chunks."""
        if pathlib.PurePosixPath(remote_run_id).is_absolute() or ".." in pathlib.PurePosixPath(remote_run_id).parts:
            raise ValueError("unsafe remote run ID")
        remote_dir = f"{self.config.payload_root.rstrip('/')}/{remote_run_id}"
        files: dict[str, dict[str, str]] = {}
        for name in ("train.py", "registered_metadata.json"):
            path = payload / name
            content = path.read_bytes()
            files[name] = {
                "content_base64": base64.b64encode(content).decode("ascii"),
                "sha256": sha256_bytes(content),
            }
        stage_root = f"{remote_dir}/.autopareto-transfer-{uuid.uuid4().hex}"
        lines = [
            f"mkdir -p {shlex.quote(remote_dir)}",
            f"rm -rf {shlex.quote(stage_root)}",
            f"mkdir {shlex.quote(stage_root)}",
            f"trap {shlex.quote(f'rm -rf {stage_root}')} EXIT",
        ]
        verification_items = []
        for name in ("train.py", "registered_metadata.json"):
            item = files[name]
            temporary = f"{stage_root}/{name}.partial"
            destination = f"{remote_dir}/{name}"
            lines.append(f"rm -f {shlex.quote(temporary)}")
            content = base64.b64decode(item["content_base64"])
            for offset in range(0, len(content), 2048):
                chunk = content[offset:offset + 2048].hex()
                writer = (
                    "import pathlib,sys; p=pathlib.Path(sys.argv[1]); "
                    "p.open('ab').write(bytes.fromhex(sys.argv[2]))"
                )
                lines.append(
                    f"python3 -c {shlex.quote(writer)} {shlex.quote(temporary)} {shlex.quote(chunk)}"
                )
            verification_items.append({
                "name": name,
                "temp": temporary,
                "dest": destination,
                "size": len(content),
                "sha256": item["sha256"],
            })
        verifier = (
            "import hashlib,json,pathlib,sys; items=json.loads(sys.argv[1]); result={}; "
            "exec(\"for item in items:\\n p=pathlib.Path(item['temp']); b=p.read_bytes()\\n "
            "if len(b)!=item['size'] or hashlib.sha256(b).hexdigest()!=item['sha256']: raise SystemExit('staged file integrity mismatch: '+item['name'])\\n "
            "p.replace(item['dest']); result[item['name']]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}\") ; "
            "payload=json.dumps(result,sort_keys=True,separators=(',',':')).encode(); "
            "import base64; print('AUTOPARETO_TRANSFER_RESULT:'+base64.b64encode(payload).decode('ascii'))"
        )
        lines.append(f"python3 -c {shlex.quote(verifier)} {shlex.quote(json.dumps(verification_items, sort_keys=True, separators=(',', ':')))}")
        completed = self._run_pty_lines(lines)
        normalized = _normalize_pty_output(completed.stdout.decode(errors="replace"))
        result_lines = [line for line in normalized.splitlines() if line.startswith("AUTOPARETO_TRANSFER_RESULT:")]
        if len(result_lines) != 1:
            raise RuntimeError("A40 transfer result sentinel was missing, duplicated, or malformed")
        result_match = _TRANSFER_RESULT.fullmatch(result_lines[0])
        if result_match is None:
            raise RuntimeError("A40 transfer result sentinel was malformed")
        try:
            remote_hashes = json.loads(base64.b64decode(result_match.group(1), validate=True))
        except (ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("A40 transfer result payload was not valid base64 JSON") from exc
        if not isinstance(remote_hashes, dict):
            raise RuntimeError("A40 transfer result payload has an unexpected shape")
        expected_hashes = {name: item["sha256"] for name, item in files.items()}
        if set(remote_hashes) != set(expected_hashes) or any(
            not isinstance(item, dict)
            or set(item) != {"bytes", "sha256"}
            or not isinstance(item["bytes"], int)
            or item["bytes"] < 0
            or not isinstance(item["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None
            for item in remote_hashes.values()
        ):
            raise RuntimeError("A40 transfer result payload has an unexpected shape")
        observed_hashes = {name: item["sha256"] for name, item in remote_hashes.items()}
        observed_sizes = {name: item["bytes"] for name, item in remote_hashes.items()}
        expected_sizes = {name: len(base64.b64decode(item["content_base64"])) for name, item in files.items()}
        if observed_hashes != expected_hashes or observed_sizes != expected_sizes:
            raise RuntimeError("A40 payload transfer hash verification failed")
        return remote_dir

    def verify_payload(
        self,
        *,
        remote_dir: str,
        candidate: pathlib.Path,
        metadata: pathlib.Path,
    ) -> dict[str, dict[str, Any]]:
        """Read-only verification of an already-staged two-file payload."""
        root = pathlib.PurePosixPath(remote_dir)
        if not root.is_absolute() or ".." in root.parts:
            raise ValueError("remote payload directory must be an absolute safe path")
        local = {"train.py": candidate, "registered_metadata.json": metadata}
        expected: dict[str, dict[str, Any]] = {}
        for name, path in local.items():
            content = path.read_bytes()
            expected[name] = {"bytes": len(content), "sha256": sha256_bytes(content)}
        nonce = uuid.uuid4().hex
        checker = (
            "import base64,hashlib,json,pathlib,sys; root=pathlib.Path(sys.argv[1]); "
            "names=['train.py','registered_metadata.json']; result={}; "
            "exec(\"for name in names:\\n p=root/name\\n "
            "if p.is_symlink() or not p.is_file(): raise SystemExit('payload file missing: '+name)\\n "
            "b=p.read_bytes(); result[name]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}\"); "
            "b=json.dumps(result,sort_keys=True,separators=(',',':')).encode(); "
            "encoded=base64.b64encode(b).decode('ascii'); nonce=sys.argv[2]; "
            "print('AUTOPARETO_PAYLOAD_VERIFY_BEGIN:'+nonce); "
            "print('AUTOPARETO_PAYLOAD_VERIFY_META:%d:%s' % (len(b),hashlib.sha256(b).hexdigest())); "
            "[print('AUTOPARETO_PAYLOAD_VERIFY_DATA:'+encoded[i:i+32]) for i in range(0,len(encoded),32)]; "
            "print('AUTOPARETO_PAYLOAD_VERIFY_END:'+nonce)"
        )
        completed = self._run(
            f"python3 -c {shlex.quote(checker)} {shlex.quote(str(root))} {shlex.quote(nonce)}"
        )
        try:
            observed = json.loads(self._read_framed_base64_payload(
                completed,
                label="remote payload verification",
                nonce=nonce,
                begin_prefix="AUTOPARETO_PAYLOAD_VERIFY_BEGIN",
                meta_prefix="AUTOPARETO_PAYLOAD_VERIFY_META",
                data_prefix="AUTOPARETO_PAYLOAD_VERIFY_DATA",
                end_prefix="AUTOPARETO_PAYLOAD_VERIFY_END",
            ))
        except json.JSONDecodeError as exc:
            raise RuntimeError("remote payload verification was not valid JSON") from exc
        if observed != expected:
            raise RuntimeError(f"remote payload verification mismatch: {observed} != {expected}")
        return observed

    def verify_absolute_files(self, expected_hashes: Mapping[str, str]) -> dict[str, str]:
        """Verify trusted absolute remote files without modifying them."""
        if not expected_hashes:
            raise ValueError("absolute remote hash set must not be empty")
        normalized: dict[str, str] = {}
        for raw_path, digest in expected_hashes.items():
            path = pathlib.PurePosixPath(str(raw_path))
            if not path.is_absolute() or ".." in path.parts:
                raise ValueError(f"remote file path is not absolute and safe: {raw_path}")
            if re.fullmatch(r"[0-9a-f]{64}", str(digest)) is None:
                raise ValueError(f"remote file hash is malformed: {raw_path}")
            normalized[str(path)] = str(digest)
        request = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        nonce = uuid.uuid4().hex
        checker = (
            "import base64,hashlib,json,pathlib,sys; expected=json.loads(sys.argv[1]); result={}; "
            "exec(\"for name in sorted(expected):\\n p=pathlib.Path(name)\\n "
            "if p.is_symlink() or not p.is_file(): raise SystemExit('remote file missing: '+name)\\n "
            "result[name]=hashlib.sha256(p.read_bytes()).hexdigest()\"); "
            "b=json.dumps(result,sort_keys=True,separators=(',',':')).encode(); "
            "encoded=base64.b64encode(b).decode('ascii'); nonce=sys.argv[2]; "
            "print('AUTOPARETO_ABSOLUTE_FILES_BEGIN:'+nonce); "
            "print('AUTOPARETO_ABSOLUTE_FILES_META:%d:%s' % (len(b),hashlib.sha256(b).hexdigest())); "
            "[print('AUTOPARETO_ABSOLUTE_FILES_DATA:'+encoded[i:i+32]) for i in range(0,len(encoded),32)]; "
            "print('AUTOPARETO_ABSOLUTE_FILES_END:'+nonce)"
        )
        completed = self._run(
            f"python3 -c {shlex.quote(checker)} {shlex.quote(request)} {shlex.quote(nonce)}"
        )
        try:
            observed = json.loads(self._read_framed_base64_payload(
                completed,
                label="absolute remote file verification",
                nonce=nonce,
                begin_prefix="AUTOPARETO_ABSOLUTE_FILES_BEGIN",
                meta_prefix="AUTOPARETO_ABSOLUTE_FILES_META",
                data_prefix="AUTOPARETO_ABSOLUTE_FILES_DATA",
                end_prefix="AUTOPARETO_ABSOLUTE_FILES_END",
            ))
        except json.JSONDecodeError as exc:
            raise RuntimeError("absolute remote file verification was not valid JSON") from exc
        if observed != normalized:
            raise RuntimeError(f"absolute remote file hash mismatch: {observed} != {normalized}")
        return observed

    def install_file(self, *, local_path: pathlib.Path, remote_path: str) -> dict[str, Any]:
        """Install one approved file using verified same-filesystem PTY staging."""
        destination = pathlib.PurePosixPath(remote_path)
        if not destination.is_absolute() or ".." in destination.parts:
            raise ValueError("trusted destination must be an absolute safe path")
        content = local_path.read_bytes()
        expected = {"bytes": len(content), "sha256": sha256_bytes(content)}
        stage_root = f"{destination.parent}/.autopareto-install-{uuid.uuid4().hex}"
        temporary = f"{stage_root}/{destination.name}.partial"
        lines = [
            f"mkdir -p {shlex.quote(str(destination.parent))}",
            f"rm -rf {shlex.quote(stage_root)}",
            f"mkdir {shlex.quote(stage_root)}",
            f"trap {shlex.quote(f'rm -rf {stage_root}')} EXIT",
            f"rm -f {shlex.quote(temporary)}",
        ]
        for offset in range(0, len(content), 2048):
            chunk = content[offset:offset + 2048].hex()
            writer = (
                "import pathlib,sys; p=pathlib.Path(sys.argv[1]); "
                "p.open('ab').write(bytes.fromhex(sys.argv[2]))"
            )
            lines.append(f"python3 -c {shlex.quote(writer)} {shlex.quote(temporary)} {shlex.quote(chunk)}")
        verifier = (
            "import hashlib,pathlib,sys; p=pathlib.Path(sys.argv[1]); b=p.read_bytes(); "
            "expected_size=int(sys.argv[3]); expected_sha=sys.argv[4]; "
            "actual_sha=hashlib.sha256(b).hexdigest(); "
            "(len(b)==expected_size and actual_sha==expected_sha) or (_ for _ in ()).throw(SystemExit('staged file integrity mismatch')); "
            "p.replace(sys.argv[2]); print('AUTOPARETO_'+'FILE_RESULT:%d:%s' % (len(b),actual_sha))"
        )
        lines.append(
            f"python3 -c {shlex.quote(verifier)} {shlex.quote(temporary)} {shlex.quote(str(destination))} "
            f"{expected['bytes']} {expected['sha256']}"
        )
        completed = self._run_pty_lines(lines)
        normalized = _normalize_pty_output(completed.stdout.decode(errors="replace"))
        result_lines = [line for line in normalized.splitlines() if "AUTOPARETO_FILE_RESULT" in line]
        if len(result_lines) != 1:
            raise RuntimeError("trusted file result sentinel was missing or duplicated")
        result_match = _FILE_RESULT.fullmatch(result_lines[0])
        if result_match is None:
            raise RuntimeError("trusted file result sentinel was malformed")
        observed = {"bytes": int(result_match.group(1)), "sha256": result_match.group(2)}
        if observed != expected:
            raise RuntimeError(f"trusted file integrity mismatch after PTY install: {observed} != {expected}")
        return observed

    def _trusted_state_root(self) -> pathlib.PurePosixPath:
        return pathlib.PurePosixPath(
            self.config.trusted_state_root
            or os.environ.get("AUTOPARETO_TRUSTED_STATE_ROOT", "/var/lib/autopareto/pair_a_state")
        )

    def _read_framed_base64_payload(
        self,
        completed: subprocess.CompletedProcess[bytes],
        *,
        label: str,
        nonce: str,
        begin_prefix: str,
        meta_prefix: str,
        data_prefix: str,
        end_prefix: str,
    ) -> bytes:
        """Strictly recover one bounded, SHA-verified payload from noisy PTY output."""
        normalized = _normalize_pty_output(completed.stdout.decode(errors="replace"))
        lines = normalized.splitlines()
        begin = f"{begin_prefix}:{nonce}"
        end = f"{end_prefix}:{nonce}"
        begin_positions = [index for index, line in enumerate(lines) if line == begin]
        end_positions = [index for index, line in enumerate(lines) if line == end]
        if len(begin_positions) != 1 or len(end_positions) != 1 or end_positions[0] <= begin_positions[0]:
            raise RuntimeError(f"{label} frame was missing, duplicated, or malformed")
        framed_lines = lines[begin_positions[0] + 1:end_positions[0]]
        meta_pattern = re.compile(
            re.escape(meta_prefix) + r":((?:0|[1-9][0-9]*)):([0-9a-f]{64})"
        )
        data_pattern = re.compile(re.escape(data_prefix) + r":([A-Za-z0-9+/=]+)")
        metadata = [match.groups() for match in (meta_pattern.fullmatch(line) for line in framed_lines) if match]
        chunks = [match.group(1) for match in (data_pattern.fullmatch(line) for line in framed_lines) if match]
        if len(metadata) != 1 or not chunks or len(metadata) + len(chunks) != len(framed_lines):
            raise RuntimeError(f"{label} frame was missing, duplicated, or malformed")
        size_text, expected_sha = metadata[0]
        try:
            content = base64.b64decode("".join(chunks), validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise RuntimeError(f"{label} frame payload was not valid base64") from exc
        if int(size_text) != len(content) or sha256_bytes(content) != expected_sha:
            raise RuntimeError(f"{label} frame payload size or SHA-256 did not match")
        return content

    def _fetch_durable_record(self, invocation_id: str) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{7,127}", invocation_id):
            raise ValueError("durable invocation ID is malformed")
        record_path = self._trusted_state_root() / "invocations" / f"{invocation_id}.json"
        nonce = uuid.uuid4().hex
        reader = (
            "import base64,hashlib,json,os,pathlib,sys; p=pathlib.Path(sys.argv[1]); "
            "payload=json.loads(p.read_text()); pid=payload.get('worker_pid'); alive=False; "
            "cmd=''; "
            "\ntry:\n os.kill(pid,0) if isinstance(pid,int) and pid>0 else (_ for _ in ()).throw(OSError())\n "
            "cmd=pathlib.Path('/proc/%d/cmdline' % pid).read_bytes().decode(errors='replace')\n "
            "alive=('--durable-worker' in cmd and payload.get('invocation_id','') in cmd)\n"
            "except OSError:\n pass\n"
            "b=json.dumps({'record':payload,'process_alive':alive},sort_keys=True,separators=(',',':')).encode(); "
            "encoded=base64.b64encode(b).decode('ascii'); nonce=sys.argv[2]; "
            "print('AUTOPARETO_DURABLE_RECORD_BEGIN:'+nonce); "
            "print('AUTOPARETO_DURABLE_RECORD_META:%d:%s' % (len(b),hashlib.sha256(b).hexdigest())); "
            "[print('AUTOPARETO_DURABLE_RECORD_DATA:'+encoded[i:i+32]) for i in range(0,len(encoded),32)]; "
            "print('AUTOPARETO_DURABLE_RECORD_END:'+nonce)"
        )
        completed = self._run_read_only_pty(
            f"python3 -c {shlex.quote(reader)} {shlex.quote(str(record_path))} {shlex.quote(nonce)}",
            operation=f"durable_record_read:{invocation_id}",
        )
        try:
            value = json.loads(self._read_framed_base64_payload(
                completed,
                label="durable record",
                nonce=nonce,
                begin_prefix="AUTOPARETO_DURABLE_RECORD_BEGIN",
                meta_prefix="AUTOPARETO_DURABLE_RECORD_META",
                data_prefix="AUTOPARETO_DURABLE_RECORD_DATA",
                end_prefix="AUTOPARETO_DURABLE_RECORD_END",
            ))
        except json.JSONDecodeError as exc:
            raise RuntimeError("durable record frame payload was not valid JSON") from exc
        if not isinstance(value, dict) or not isinstance(value.get("record"), dict):
            raise RuntimeError("durable record payload has an unexpected shape")
        return value

    def launch(
        self,
        *,
        remote_dir: str,
        output_name: str = "protected_result.json",
        invocation_id: str,
        invocation_metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        if pathlib.PurePosixPath(output_name).name != output_name:
            raise ValueError("output name must be a plain filename")
        repository = pathlib.PurePosixPath(self.config.repository)
        workspace = repository.parent
        remote_python = pathlib.PurePosixPath(
            self.config.python_path or "/usr/local/bin/python"
        )
        registered_cache = pathlib.PurePosixPath(
            self.config.registered_cache or "/var/cache/autopareto/pair_a_registered_cache"
        )
        study_venv = pathlib.PurePosixPath(
            self.config.study_venv or "/usr/local"
        )
        trusted_state = self._trusted_state_root()
        if pathlib.PurePosixPath(output_name).name != output_name:
            raise ValueError("output name must be a plain filename")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{7,127}", invocation_id):
            raise ValueError("durable invocation ID is malformed")
        output = trusted_state / "bridge_results" / output_name
        command = (
            f"cd {shlex.quote(self.config.repository)} && "
            f"HOME={shlex.quote(str(workspace))} "
            "LD_LIBRARY_PATH=$(find /usr/local/lib/python3.12/dist-packages/nvidia -type d -name lib -print | paste -sd:) "
            f"AUTOPARETO_REGISTERED_CACHE={shlex.quote(str(registered_cache))} "
            f"HF_HOME={shlex.quote(str(registered_cache / 'huggingface'))} "
            f"HF_HUB_CACHE={shlex.quote(str(registered_cache / 'huggingface' / 'hub'))} "
            "HF_HUB_OFFLINE=1 "
            f"AUTOPARETO_STUDY_VENV={shlex.quote(str(study_venv))} "
            f"AUTOPARETO_TRUSTED_ROOT={shlex.quote(str(repository))} "
            f"AUTOPARETO_TRUSTED_STATE_ROOT={shlex.quote(str(trusted_state))} "
            f"AUTOPARETO_CANDIDATE_ROOT=/var/lib/autopareto/candidate_attempts "
            f"AUTOPARETO_CANDIDATE_RUNTIME_ROOT=/opt/autopareto_candidate_runtime "
            f"AUTOPARETO_CANDIDATE_INPUT_ROOT=/opt/autopareto_candidate_inputs/per_attempts "
            f"AUTOPARETO_APPROVED_INPUT_ROOT=/opt/autopareto_candidate_inputs "
            f"{shlex.quote(str(remote_python))} -m {shlex.quote(self.runner_module)} "
            f"--payload-dir {shlex.quote(remote_dir)} --output {shlex.quote(str(output))} "
            f"--repository {shlex.quote(self.config.repository)} --durable-launch "
            f"--durable-invocation-id {shlex.quote(invocation_id)} "
            f"--durable-record-root {shlex.quote(str(trusted_state / 'invocations'))} "
            f"--durable-metadata-json {shlex.quote(json.dumps(dict(invocation_metadata), sort_keys=True, separators=(',', ':')))}"
        )
        self._run(command)
        return self._fetch_durable_record(invocation_id)

    def launch_trusted_command(
        self,
        *,
        command_argv: Sequence[str],
        output_path: str,
        invocation_id: str,
        invocation_metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Launch one trusted, non-Scientist command with the durable receipt protocol.

        This is deliberately limited to outputs inside the persistent trusted-state
        root.  It is used for non-study validation such as the frozen evaluator,
        whose completion must not depend on an SSH/PTTY session either.
        """
        if not command_argv or any(not isinstance(item, str) or not item for item in command_argv):
            raise ValueError("trusted durable command must be a non-empty argv string list")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{7,127}", invocation_id):
            raise ValueError("durable invocation ID is malformed")
        trusted_state = self._trusted_state_root()
        output = pathlib.PurePosixPath(output_path)
        if not output.is_absolute() or trusted_state not in output.parents:
            raise ValueError("trusted durable command output must be inside trusted state")
        repository = pathlib.PurePosixPath(self.config.repository)
        workspace = repository.parent
        remote_python = pathlib.PurePosixPath(self.config.python_path or "/usr/local/bin/python")
        registered_cache = pathlib.PurePosixPath(
            self.config.registered_cache or "/var/cache/autopareto/pair_a_registered_cache"
        )
        study_venv = pathlib.PurePosixPath(self.config.study_venv or "/usr/local")
        argv_json = json.dumps(list(command_argv), separators=(",", ":"))
        command = (
            f"cd {shlex.quote(self.config.repository)} && "
            f"HOME=/root "
            "LD_LIBRARY_PATH=$(find /usr/local/lib/python3.12/dist-packages/nvidia -type d -name lib -print | paste -sd:) "
            f"AUTOPARETO_REGISTERED_CACHE={shlex.quote(str(registered_cache))} "
            f"AUTOPARETO_STUDY_VENV={shlex.quote(str(study_venv))} "
            f"AUTOPARETO_TRUSTED_ROOT={shlex.quote(str(repository))} "
            f"AUTOPARETO_TRUSTED_STATE_ROOT={shlex.quote(str(trusted_state))} "
            f"HF_HOME=/var/cache/autopareto/pair_a_registered_cache/huggingface "
            f"HF_HUB_CACHE=/var/cache/autopareto/pair_a_registered_cache/huggingface/hub "
            f"HF_HUB_OFFLINE=1 "
            f"{shlex.quote(str(remote_python))} -m {shlex.quote(self.runner_module)} "
            f"--payload-dir {shlex.quote(self.config.repository)} --output {shlex.quote(str(output))} "
            f"--repository {shlex.quote(self.config.repository)} --durable-launch "
            f"--durable-invocation-id {shlex.quote(invocation_id)} "
            f"--durable-record-root {shlex.quote(str(trusted_state / 'invocations'))} "
            f"--durable-metadata-json {shlex.quote(json.dumps(dict(invocation_metadata), sort_keys=True, separators=(',', ':')))} "
            f"--durable-command-json {shlex.quote(argv_json)}"
        )
        self._run(command)
        return self._fetch_durable_record(invocation_id)

    def reconcile_invocation(self, invocation_id: str) -> dict[str, Any]:
        """Read the durable remote receipt; transport output is never authoritative."""
        return self._fetch_durable_record(invocation_id)

    def wait_for_invocation(self, invocation_id: str, *, poll_seconds: float = 5.0) -> dict[str, Any]:
        if poll_seconds <= 0 or poll_seconds > 60:
            raise ValueError("durable invocation poll interval must be in (0, 60]")
        while True:
            observed = self.reconcile_invocation(invocation_id)
            record = observed["record"]
            status = record.get("status")
            if status in {"COMPLETE", "FAILED"}:
                return observed
            if status != "RUNNING" or not observed.get("process_alive"):
                return observed
            time.sleep(poll_seconds)

    def fetch_durable_json(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Retrieve a receipt-bound JSON artifact without trusting the PTY output."""
        if record.get("status") != "COMPLETE":
            raise RuntimeError("cannot retrieve a non-complete durable invocation")
        remote_path = record.get("result_path")
        expected_size = record.get("result_bytes")
        expected_sha = record.get("result_sha256")
        if not isinstance(remote_path, str) or not isinstance(expected_size, int) or not re.fullmatch(r"[0-9a-f]{64}", str(expected_sha)):
            raise RuntimeError("durable completion record has an invalid result receipt")
        nonce = uuid.uuid4().hex
        reader = (
            "import base64,hashlib,pathlib,sys,zlib; "
            "p=pathlib.Path(sys.argv[1]); b=p.read_bytes(); compressed=zlib.compress(b,6); "
            "encoded=base64.b64encode(compressed).decode('ascii'); nonce=sys.argv[2]; "
            "print('AUTOPARETO_DURABLE_RESULT_BEGIN:'+nonce); "
            "print('AUTOPARETO_DURABLE_RESULT_META:%d:%s' % (len(compressed),hashlib.sha256(compressed).hexdigest())); "
            "[print('AUTOPARETO_DURABLE_RESULT_DATA:'+encoded[i:i+32]) for i in range(0,len(encoded),32)]; "
            "print('AUTOPARETO_DURABLE_RESULT_END:'+nonce)"
        )
        completed = self._run_read_only_pty(
            f"python3 -c {shlex.quote(reader)} {shlex.quote(remote_path)} {shlex.quote(nonce)}",
            operation=f"durable_result_read:{remote_path}",
        )
        compressed = self._read_framed_base64_payload(
            completed,
            label="durable result",
            nonce=nonce,
            begin_prefix="AUTOPARETO_DURABLE_RESULT_BEGIN",
            meta_prefix="AUTOPARETO_DURABLE_RESULT_META",
            data_prefix="AUTOPARETO_DURABLE_RESULT_DATA",
            end_prefix="AUTOPARETO_DURABLE_RESULT_END",
        )
        try:
            content = zlib.decompress(compressed)
        except zlib.error as exc:
            raise RuntimeError("durable result compressed payload was invalid") from exc
        if len(content) != expected_size or sha256_bytes(content) != expected_sha:
            raise RuntimeError("durable result receipt hash or size mismatch")
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError("durable result is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("durable result is not a JSON object")
        return payload

    def fetch_durable_result(self, record: Mapping[str, Any]) -> dict[str, Any]:
        payload = self.fetch_durable_json(record)
        if payload.get("schema") != REMOTE_RESULT_SCHEMA:
            raise RuntimeError("A40 result has an unexpected bridge schema")
        return payload

    def fetch_result(self, remote_path: str) -> dict[str, Any]:
        result = self._run(f"cat {shlex.quote(remote_path)}")
        payload = json.loads(result.stdout.decode())
        if payload.get("schema") != REMOTE_RESULT_SCHEMA:
            raise ValueError("A40 result has an unexpected bridge schema")
        return payload


def update_campaign_view(*, view_path: pathlib.Path, result: Mapping[str, Any]) -> dict[str, Any]:
    """Return and persist the next campaign-local view, excluding sealed metrics."""
    view = json.loads(view_path.read_text())
    campaign_id = view.get("campaign_id")
    if result.get("campaign_id") != campaign_id:
        raise ValueError("A40 result belongs to a different campaign")
    forbidden = {"held_out_bpb", "BLiMP", "PIQA", "ARC-Easy", "LAMBADA"}
    if forbidden.intersection(result):
        raise ValueError("sealed downstream metric entered campaign view")
    record = {key: result[key] for key in VIEW_RESULT_FIELDS if key in result}
    if "inference_result" in result:
        record["inference_result"] = result["inference_result"]
    baseline = view["baseline"]
    decision = result.get("pareto_decision", {})
    evidence = [*view.get("same_arm_evidence", [])]
    if result.get("status") in {"completed", "scientific_failure"}:
        evidence.append(record)
    evidence, frontier, parent_pool = reconcile_same_arm_archive(
        baseline=baseline, evidence=evidence
    )
    failures = [*view.get("failures", [])]
    if result.get("status") == "scientific_failure":
        failures.append({
            "attempt": int(result["attempt"]),
            "failure_class": result.get("failure_class", "candidate"),
            "error": str(result.get("error", "candidate scientific failure")),
        })
    updated = build_campaign_view(
        campaign_id=campaign_id,
        contract_id=view["contract_id"],
        baseline=baseline,
        frontier=frontier,
        # After the final scheduled attempt the view is terminal (no Scientist
        # reads it); "attempt 26" used to raise and strand attempt 25 uncommitted.
        attempt_number=min(int(result["attempt"]) + 1, ATTEMPTS_PER_CONDITION),
        failures=failures,
        evidence=evidence,
        parent_pool=parent_pool,
    )
    view_path.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n")
    return updated


__all__ = [
    "A40RemoteConfig",
    "A40Transport",
    "BRIDGE_SCHEMA",
    "REMOTE_PAYLOAD_SCHEMA",
    "ScientistHandoff",
    "build_registered_metadata",
    "run_local_scientist",
    "sha256_file",
    "stage_remote_payload",
    "update_campaign_view",
    "validate_scientist_proposal",
]
