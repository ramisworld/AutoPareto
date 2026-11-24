"""Trusted A40-side entry point for the local Scientist bridge.

Only the orchestrator may invoke this module.  Its payload directory is expected
to contain exactly ``train.py`` and ``registered_metadata.json``; protected
``prepare.py`` and ``study_runtime.py`` are copied from the preinstalled A40
study checkout, never transferred from the Scientist context.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping

from .bridge import REMOTE_RESULT_SCHEMA, sha256_file


DEFAULT_TRUSTED_ROOT = "/opt/autopareto_trusted"
DEFAULT_TRUSTED_STATE_ROOT = "/var/lib/autopareto/pair_a_state"
DEFAULT_CANDIDATE_ROOT = "/var/lib/autopareto/candidate_attempts"
DEFAULT_CANDIDATE_ATTEMPT_ROOT = "/opt/autopareto-candidate-attempts"
MIN_CANDIDATE_FREE_BYTES = 8 * 1024**3
DURABLE_INVOCATION_SCHEMA = "autopareto_durable_invocation_v1"
_MODULE_NAME = "study.contracts_2026.remote_runner"
_INVOCATION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{7,127}")


def _utc_timestamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json_write(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _prepare_result_output(output: pathlib.Path, trusted_state_root: pathlib.Path) -> pathlib.Path:
    """Validate and prepare the trusted result destination before execution."""
    output = output.resolve()
    trusted_state_root = trusted_state_root.resolve()
    if not output.is_relative_to(trusted_state_root):
        raise ValueError("trusted runner output must be persisted below AUTOPARETO_TRUSTED_STATE_ROOT")
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    return output


def _remove_disposable_attempt_tree(path: pathlib.Path, parent: pathlib.Path) -> None:
    """Remove exactly one completed attempt tree after trusted persistence."""
    resolved_parent = parent.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if path.is_symlink() or resolved.parent != resolved_parent:
        raise RuntimeError(f"refusing unsafe disposable-attempt cleanup: {path}")
    shutil.rmtree(resolved)


def _require_candidate_disk_capacity(
    attempt_root: pathlib.Path,
    trusted_state_root: pathlib.Path,
    *,
    minimum_free_bytes: int = MIN_CANDIDATE_FREE_BYTES,
) -> None:
    """Fail before GPU work unless every backing filesystem has safe headroom."""
    if minimum_free_bytes <= 0:
        raise ValueError("minimum candidate free-space requirement must be positive")
    checked_devices: set[int] = set()
    for label, path in (("candidate attempt", attempt_root), ("trusted state", trusted_state_root)):
        resolved = path.resolve(strict=True)
        device = resolved.stat().st_dev
        if device in checked_devices:
            continue
        checked_devices.add(device)
        free_bytes = shutil.disk_usage(resolved).free
        if free_bytes < minimum_free_bytes:
            raise RuntimeError(
                f"insufficient {label} filesystem capacity before candidate execution: "
                f"{free_bytes} bytes free, {minimum_free_bytes} required"
            )


def _safe_invocation_id(value: str) -> str:
    if not _INVOCATION_ID.fullmatch(value):
        raise ValueError("durable invocation ID is malformed")
    return value


def _canonical_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()


def _invocation_path(record_root: pathlib.Path, invocation_id: str) -> pathlib.Path:
    return record_root / f"{_safe_invocation_id(invocation_id)}.json"


def _prepare_durable_state_root(args: argparse.Namespace) -> tuple[pathlib.Path, pathlib.Path]:
    """Create the trusted durable root before a candidate boundary is entered.

    Fresh state namespaces are created by the durable launcher, not by a
    candidate.  They must therefore be private to the trusted launcher before
    the candidate sandbox checks its rootfs permission boundary.
    """
    record_root = args.durable_record_root.resolve()
    trusted_state_root = record_root.parent
    output = args.output.resolve()
    if record_root.name != "invocations" or not output.is_relative_to(trusted_state_root):
        raise ValueError("durable receipt and output must be below one trusted state root")
    trusted_state_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if trusted_state_root.is_symlink() or not trusted_state_root.is_dir():
        raise RuntimeError("trusted durable state root is not a regular directory")
    if trusted_state_root.stat().st_uid != os.geteuid():
        raise RuntimeError("trusted durable state root is not owned by the launcher")
    trusted_state_root.chmod(0o700)
    if trusted_state_root.stat().st_mode & 0o077:
        raise RuntimeError("trusted durable state root is not private")
    record_root.mkdir(mode=0o700, exist_ok=True)
    if record_root.is_symlink() or not record_root.is_dir():
        raise RuntimeError("durable invocation directory is not a regular directory")
    record_root.chmod(0o700)
    return trusted_state_root, record_root


def _invocation_identity(args: argparse.Namespace) -> dict[str, Any]:
    durable_command_json = getattr(args, "durable_command_json", None)
    if durable_command_json:
        command = json.loads(durable_command_json)
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(part, str) or not part for part in command)
        ):
            raise ValueError("durable command must be a non-empty argv string list")
        return {
            "kind": "trusted_command",
            "argv": command,
            "output": str(args.output.resolve()),
            "repository": str(args.repository.resolve()),
        }
    payload_dir = args.payload_dir.resolve(strict=True)
    metadata = payload_dir / "registered_metadata.json"
    candidate = payload_dir / "train.py"
    return {
        "payload_dir": str(payload_dir),
        "registered_metadata_sha256": sha256_file(metadata),
        "candidate_source_sha256": sha256_file(candidate),
        "output": str(args.output.resolve()),
        "repository": str(args.repository.resolve()),
    }


def _read_invocation(path: pathlib.Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or payload.get("schema") != DURABLE_INVOCATION_SCHEMA:
        raise RuntimeError("durable invocation record has an unexpected schema")
    return payload


def _process_matches(record: dict[str, Any]) -> bool:
    if record.get("status") != "RUNNING":
        return False
    pid = record.get("worker_pid")
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        command = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
    except OSError:
        return False
    return "--durable-worker" in command and str(record.get("invocation_id", "")) in command


def _terminal_record(
    *, path: pathlib.Path, record: dict[str, Any], status: str, exit_status: int, **extra: Any
) -> None:
    if status not in {"COMPLETE", "FAILED"}:
        raise ValueError("terminal durable invocation status is invalid")
    record.update({
        "status": status,
        "exit_status": int(exit_status),
        "completed_at": _utc_timestamp(),
        **extra,
    })
    _atomic_json_write(path, record)


def _run_durable_worker(args: argparse.Namespace) -> int:
    _trusted_state_root, record_root = _prepare_durable_state_root(args)
    invocation_id = _safe_invocation_id(args.durable_invocation_id)
    record_path = _invocation_path(record_root, invocation_id)
    record = _read_invocation(record_path)
    identity = _invocation_identity(args)
    if record.get("invocation_id") != invocation_id or record.get("command_identity") != identity:
        raise RuntimeError("durable worker invocation identity disagrees with its record")
    if record.get("status") != "RUNNING":
        raise RuntimeError("durable worker cannot execute a terminal invocation")
    record["worker_pid"] = os.getpid()
    record["worker_started_at"] = _utc_timestamp()
    _atomic_json_write(record_path, record)
    try:
        if getattr(args, "durable_command_json", None):
            command = json.loads(args.durable_command_json)
            if (
                not isinstance(command, list)
                or not command
                or any(not isinstance(part, str) or not part for part in command)
            ):
                raise ValueError("durable command must be a non-empty argv string list")
            subprocess.run(command, cwd=args.repository.resolve(), check=True)
            result = {"status": "completed"}
        else:
            result = execute(
                payload_dir=args.payload_dir.resolve(), output=args.output.resolve(), repository=args.repository.resolve()
            )
        result_path = args.output.resolve()
        if not result_path.is_file() or result_path.is_symlink():
            raise RuntimeError("durable invocation completed without a regular result file")
        _terminal_record(
            path=record_path,
            record=_read_invocation(record_path),
            status="COMPLETE",
            exit_status=0,
            result_path=str(result_path),
            result_bytes=result_path.stat().st_size,
            result_sha256=sha256_file(result_path),
            result_status=result.get("status"),
        )
        return 0
    except BaseException as exc:
        _terminal_record(
            path=record_path,
            record=_read_invocation(record_path),
            status="FAILED",
            exit_status=1,
            failure={"type": type(exc).__name__, "message": str(exc)},
        )
        raise


def _launch_durable_invocation(args: argparse.Namespace) -> int:
    _trusted_state_root, record_root = _prepare_durable_state_root(args)
    invocation_id = _safe_invocation_id(args.durable_invocation_id)
    record_path = _invocation_path(record_root, invocation_id)
    identity = _invocation_identity(args)
    metadata = json.loads(args.durable_metadata_json) if args.durable_metadata_json else {}
    if not isinstance(metadata, dict):
        raise ValueError("durable invocation metadata must be an object")
    if record_path.exists():
        record = _read_invocation(record_path)
        if record.get("invocation_id") != invocation_id or record.get("command_identity") != identity:
            raise RuntimeError("durable invocation ID conflicts with an existing command identity")
        print(json.dumps({"invocation_id": invocation_id, "status": record.get("status")}, sort_keys=True))
        return 0
    record = {
        "schema": DURABLE_INVOCATION_SCHEMA,
        "invocation_id": invocation_id,
        "status": "RUNNING",
        "created_at": _utc_timestamp(),
        "command_identity": identity,
        "command_sha256": _canonical_sha256(identity),
        "metadata": metadata,
        "worker_pid": None,
        "log_path": str(record_root / f"{invocation_id}.log"),
    }
    _atomic_json_write(record_path, record)
    command = [
        sys.executable, "-m", _MODULE_NAME,
        "--payload-dir", str(args.payload_dir.resolve()),
        "--output", str(args.output.resolve()),
        "--repository", str(args.repository.resolve()),
        "--durable-worker",
        "--durable-invocation-id", invocation_id,
        "--durable-record-root", str(record_root),
    ]
    if getattr(args, "durable_command_json", None):
        command.extend(("--durable-command-json", args.durable_command_json))
    log_path = pathlib.Path(record["log_path"])
    try:
        with log_path.open("ab", buffering=0) as log_handle:
            child = subprocess.Popen(
                command,
                cwd=args.repository.resolve(),
                env=os.environ.copy(),
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                close_fds=True,
                start_new_session=True,
            )
    except BaseException as exc:
        _terminal_record(
            path=record_path,
            record=_read_invocation(record_path),
            status="FAILED",
            exit_status=1,
            failure={"type": type(exc).__name__, "message": str(exc)},
        )
        raise
    latest = _read_invocation(record_path)
    if latest.get("status") == "RUNNING" and latest.get("worker_pid") is None:
        latest["launcher_pid"] = child.pid
        _atomic_json_write(record_path, latest)
    print(json.dumps({"invocation_id": invocation_id, "status": "RUNNING"}, sort_keys=True))
    return 0


def _durable_status(args: argparse.Namespace) -> int:
    record = _read_invocation(
        _invocation_path(args.durable_record_root.resolve(), args.durable_invocation_id)
    )
    print(json.dumps({"record": record, "process_alive": _process_matches(record)}, sort_keys=True))
    return 0


def _require_metadata(metadata: dict[str, Any], key: str) -> Any:
    if key not in metadata:
        raise ValueError(f"registered metadata is missing {key}")
    return metadata[key]


def _safe_payload(payload_dir: pathlib.Path) -> None:
    if not payload_dir.is_dir():
        raise ValueError("payload directory is not a directory")
    names = sorted(path.name for path in payload_dir.iterdir() if not path.name.startswith("."))
    if names != ["registered_metadata.json", "train.py"]:
        raise ValueError(f"payload must contain exactly train.py and registered_metadata.json: {names}")


def _candidate_contract_environment(metadata: Mapping[str, Any]) -> dict[str, str]:
    """Return contract values that must cross the trusted-to-candidate boundary."""
    if str(metadata.get("contract_id")) != "N":
        return {}
    contract_parameters = metadata.get("contract_parameters")
    if not isinstance(contract_parameters, Mapping):
        raise ValueError("fixed-token payload is missing contract parameters")
    n_star = contract_parameters.get("n_star")
    if not isinstance(n_star, int) or n_star <= 0:
        raise ValueError("fixed-token payload is missing a positive N_STAR")
    return {"N_STAR": str(n_star)}
    train = payload_dir / "train.py"
    if train.is_symlink() or not train.is_file():
        raise ValueError("payload train.py must be a regular file")


def _candidate_prepare_source(protected_prepare: pathlib.Path) -> str:
    """Stage the complete registered prepare.py dependency.

    Candidate-side validation remains forbidden by the candidate-process
    observer, which replaces ``evaluate_bpb`` after import.  Keeping the full
    module preserves the historical source dependency graph and avoids the
    incomplete temporary context that caused import/runtime drift.
    """
    source = protected_prepare.read_text()
    if not source.strip():
        raise ValueError("protected prepare.py is empty")
    return source


def _classify_worker_failure(worker_payload: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    """Translate a structured candidate-worker failure without guessing.

    Only a failure explicitly attributed to candidate source is a scientific
    outcome. Harness and infrastructure faults remain durable evidence but are
    returned as non-consumable failures so the controller stops for
    reconciliation. Unknown provenance is an integrity failure and must not be
    converted into either category.
    """
    origin = worker_payload.get("failure_origin")
    if origin == "candidate":
        return {
            "schema": REMOTE_RESULT_SCHEMA,
            "status": "scientific_failure",
            "failure_class": "candidate",
            "campaign_id": metadata["campaign_id"],
            "contract_id": metadata["contract_id"],
            "attempt": metadata["attempt_number"],
            "error": worker_payload.get("error", "candidate worker failed"),
            "worker_payload": worker_payload,
        }
    if origin == "harness":
        return {
            "schema": REMOTE_RESULT_SCHEMA,
            "status": "harness_failure",
            "failure_class": "harness",
            "campaign_id": metadata["campaign_id"],
            "contract_id": metadata["contract_id"],
            "attempt": metadata["attempt_number"],
            "error": worker_payload.get("error", "protected candidate interface failed"),
            "worker_payload": worker_payload,
        }
    if origin == "infrastructure":
        raise RuntimeError(
            "candidate worker reported infrastructure failure; manual reconciliation is required"
        )
    raise RuntimeError(
        "candidate worker failure origin is unknown; manual reconciliation is required"
    )


def execute(*, payload_dir: pathlib.Path, output: pathlib.Path, repository: pathlib.Path) -> dict[str, Any]:
    _safe_payload(payload_dir)
    trusted_root = pathlib.Path(
        os.environ.get("AUTOPARETO_TRUSTED_ROOT", str(repository))
    ).resolve(strict=True)
    trusted_state_root = pathlib.Path(
        os.environ.get("AUTOPARETO_TRUSTED_STATE_ROOT", DEFAULT_TRUSTED_STATE_ROOT)
    ).resolve()
    trusted_state_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    output = _prepare_result_output(output, trusted_state_root)
    if trusted_root.stat().st_uid != 0 or trusted_root.stat().st_mode & 0o077:
        raise ValueError("trusted runner root must be root-owned and inaccessible to the candidate UID")
    metadata = json.loads((payload_dir / "registered_metadata.json").read_text())
    if metadata.get("schema") != "autopareto_a40_payload_v1":
        raise ValueError("unexpected registered payload schema")
    if sha256_file(payload_dir / "train.py") != _require_metadata(metadata, "candidate_source_after_sha256"):
        raise ValueError("candidate source hash does not match registered metadata")

    protected_prepare = trusted_root / "study/contracts_2026/historical_reference/baseline_source/prepare.py"
    study_runtime = trusted_root / "study/contracts_2026/study_runtime.py"
    if not protected_prepare.is_file() or not study_runtime.is_file():
        raise FileNotFoundError("preinstalled protected prepare.py or study_runtime.py is missing")
    registered_prepare_hash = metadata.get("protected_prepare_sha256")
    if registered_prepare_hash and sha256_file(protected_prepare) != registered_prepare_hash:
        raise ValueError("protected prepare.py hash does not match registered metadata")
    registered_runtime_hash = metadata.get("study_runtime_sha256")
    if registered_runtime_hash and sha256_file(study_runtime) != registered_runtime_hash:
        raise ValueError("study_runtime.py hash does not match registered metadata")

    attempt_root_base = pathlib.Path(
        os.environ.get("AUTOPARETO_CANDIDATE_ATTEMPT_ROOT", DEFAULT_CANDIDATE_ATTEMPT_ROOT)
    ).resolve()
    attempt_root_base.mkdir(mode=0o711, parents=True, exist_ok=True)
    _require_candidate_disk_capacity(attempt_root_base, trusted_state_root)
    attempt_root = attempt_root_base / f"{payload_dir.name}-{secrets.token_hex(12)}"
    attempt_root.mkdir(mode=0o711)
    # The payload directory and all trusted metadata stay outside the clean room.
    # Only the five registered Scientist-visible files are staged under the
    # candidate-visible root; the candidate output/checkpoint stays in its own
    # disposable directory and is copied to trusted state after verification.
    candidate_root = pathlib.Path(
        os.environ.get("AUTOPARETO_CANDIDATE_ROOT", DEFAULT_CANDIDATE_ROOT)
    ).resolve()
    context_root = pathlib.Path(
        os.environ.get("AUTOPARETO_CANDIDATE_CONTEXT_ROOT", "/opt/autopareto_candidate_contexts")
    ).resolve()
    context_root.mkdir(mode=0o755, parents=True, exist_ok=True)
    context = pathlib.Path(tempfile.mkdtemp(prefix=f"{payload_dir.name}-", dir=context_root))
    context.chmod(0o555)
    try:
        shutil.copy2(payload_dir / "train.py", context / "train.py")
        (context / "prepare.py").write_text(_candidate_prepare_source(protected_prepare))
        shutil.copy2(study_runtime, context / "study_runtime.py")
        # The trusted parent owns the context.  The candidate process receives
        # read-only copies; only the unique candidate-output directory is
        # writable by the dedicated non-root identity.
        context.chmod(0o555)
        for path in context.iterdir():
            path.chmod(0o555 if path.is_dir() else 0o444)
        candidate_output = attempt_root / "candidate-output"
        candidate_output.mkdir(mode=0o700)
        candidate_result = candidate_output / output.name
        command = [
            sys.executable,
            "-m",
            "study.contracts_2026.train_launcher",
            "--context",
            str(context),
            "--candidate",
            "train.py",
            "--contract",
            str(_require_metadata(metadata, "contract_id")),
            "--seed",
            str(_require_metadata(metadata, "training_seed")),
            "--output",
            str(candidate_result),
            "--execute",
        ]
        contract_parameters = metadata.get("contract_parameters", {})
        if contract_parameters.get("n_star") is not None:
            command.extend(["--n-star", str(contract_parameters["n_star"])])
        if contract_parameters.get("paired_baseline_checkpoint"):
            command.extend(["--paired-baseline-checkpoint", str(contract_parameters["paired_baseline_checkpoint"])])
        if metadata.get("process_log_sd") is not None:
            command.extend(["--process-log-sd", str(float(metadata["process_log_sd"]))])
        # Do not inherit pod credentials or campaign/history variables into the
        # candidate launcher.  The process boundary applies a second whitelist,
        # but sanitizing before the wrapper is still required defense in depth.
        inherited_names = (
            "PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
            "CUDA_HOME", "VIRTUAL_ENV", "PYTHONNOUSERSITE",
        )
        child_env = {name: os.environ[name] for name in inherited_names if os.environ.get(name)}
        child_env.update({
            # ``prepare.py`` is a frozen legacy dependency which resolves its
            # registered tokenizer/data through ``~/.cache/autoresearch``.
            # This is the trusted parent environment only: the candidate
            # sandbox replaces HOME with its isolated, read-only input home
            # immediately before exec.
            "HOME": os.environ.get("AUTOPARETO_TRUSTED_EVALUATOR_HOME", "/root"),
            "HF_HOME": os.environ.get(
                "HF_HOME",
                str(pathlib.Path(os.environ.get("AUTOPARETO_APPROVED_INPUT_ROOT", "/opt/autopareto_candidate_inputs")) / "cache" / "huggingface"),
            ),
            "HF_HUB_CACHE": os.environ.get(
                "HF_HUB_CACHE",
                str(pathlib.Path(os.environ.get("AUTOPARETO_APPROVED_INPUT_ROOT", "/opt/autopareto_candidate_inputs")) / "cache" / "huggingface" / "hub"),
            ),
            "AUTOPARETO_REGISTERED_CACHE": os.environ.get(
                "AUTOPARETO_REGISTERED_CACHE",
                str(pathlib.Path(os.environ.get("AUTOPARETO_APPROVED_INPUT_ROOT", "/opt/autopareto_candidate_inputs")) / "cache" / "autoresearch"),
            ),
            "AUTOPARETO_STUDY_VENV": os.environ.get(
                "AUTOPARETO_STUDY_VENV",
                str(pathlib.Path(os.environ.get("AUTOPARETO_STUDY_VENV", "/usr/local"))),
            ),
            "AUTOPARETO_TRUSTED_ROOT": str(trusted_root),
            "AUTOPARETO_TRUSTED_STATE_ROOT": str(trusted_state_root),
            "AUTOPARETO_TRUSTED_MIRROR_ROOT": os.environ.get(
                "AUTOPARETO_TRUSTED_MIRROR_ROOT", "/var/lib/autopareto/pair_a_trusted_backup"
            ),
            "AUTOPARETO_CANDIDATE_ROOT": str(candidate_root),
            "AUTOPARETO_CANDIDATE_RUNTIME_ROOT": os.environ.get(
                "AUTOPARETO_CANDIDATE_RUNTIME_ROOT", "/opt/autopareto_candidate_runtime"
            ),
            "AUTOPARETO_CANDIDATE_INPUT_ROOT": os.environ.get(
                "AUTOPARETO_CANDIDATE_INPUT_ROOT", "/opt/autopareto_candidate_inputs/per_attempts"
            ),
            "AUTOPARETO_STUDY_ALLOW_EXECUTION": "1",
            "AUTOPARETO_REMOTE_SANDBOX": "1",
        })
        child_env.update(_candidate_contract_environment(metadata))
        completed = subprocess.run(
            command,
            cwd=trusted_root,
            env=child_env,
            check=False,
        )
        if completed.returncode != 0 or not candidate_result.is_file():
            worker_failure = candidate_result.with_name(candidate_result.name + ".worker.json")
            if worker_failure.is_file():
                try:
                    worker_payload = json.loads(worker_failure.read_text())
                except (OSError, json.JSONDecodeError) as exc:
                    raise RuntimeError("candidate worker emitted an unreadable failure record") from exc
                if worker_payload.get("status") == "error":
                    result = _classify_worker_failure(worker_payload, metadata)
                    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
                    return result
            raise RuntimeError(f"protected A40 runner failed with return code {completed.returncode}")
        result = json.loads(candidate_result.read_text())
    finally:
        # Preserve the attempt-local checkpoint/timing artifacts for trusted
        # provenance.  The candidate cannot modify them after its process exits.
        context.chmod(0o555)
    # Persist a root-owned copy of the candidate result and all post-training
    # artifacts before returning it to the outer bridge.  The candidate-visible
    # attempt directory remains disposable and is never the authoritative store.
    trusted_attempt = trusted_state_root / "attempts" / attempt_root.name
    trusted_attempt.mkdir(mode=0o700, parents=True, exist_ok=True)
    persisted: dict[str, str] = {}
    checkpoint_source = pathlib.Path(str(result.get("checkpoint", ""))).resolve()
    expected_artifacts = [
        candidate_result,
        candidate_result.with_name(candidate_result.name + ".worker.json"),
        checkpoint_source,
        candidate_result.with_name(candidate_result.stem + ".inference.json"),
    ]
    for source in expected_artifacts:
        if source.parent != candidate_output:
            continue
        name = source.name
        if source.is_file():
            target = trusted_attempt / name
            shutil.copy2(source, target)
            target.chmod(0o400)
            persisted[name] = str(target)
    persisted_checkpoint = persisted.get(checkpoint_source.name)
    if persisted_checkpoint is not None:
        result["checkpoint"] = persisted_checkpoint
    result.update(
        {
            "schema": REMOTE_RESULT_SCHEMA,
            "campaign_id": metadata["campaign_id"],
            "contract_id": metadata["contract_id"],
            "attempt": metadata["attempt_number"],
            "registered_metadata_sha256": hashlib.sha256(
                (json.dumps(metadata, sort_keys=True) + "\n").encode()
            ).hexdigest(),
            "trusted_persistence": {
                "root": str(trusted_state_root),
                "attempt": str(trusted_attempt),
                "artifacts": persisted,
            },
        }
    )
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _remove_disposable_attempt_tree(attempt_root, attempt_root_base)
    _remove_disposable_attempt_tree(context, context_root)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--repository", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[2])
    parser.add_argument("--durable-launch", action="store_true")
    parser.add_argument("--durable-worker", action="store_true")
    parser.add_argument("--durable-status", action="store_true")
    parser.add_argument("--durable-invocation-id")
    parser.add_argument("--durable-record-root", type=pathlib.Path)
    parser.add_argument("--durable-metadata-json")
    parser.add_argument("--durable-command-json")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    durable_actions = sum((args.durable_launch, args.durable_worker, args.durable_status))
    if durable_actions > 1:
        raise SystemExit("only one durable invocation action may be selected")
    if durable_actions:
        if not args.durable_invocation_id or args.durable_record_root is None:
            raise SystemExit("durable invocation action requires ID and record root")
        if args.durable_launch:
            raise SystemExit(_launch_durable_invocation(args))
        if args.durable_worker:
            raise SystemExit(_run_durable_worker(args))
        raise SystemExit(_durable_status(args))
    payload = execute(payload_dir=args.payload_dir.resolve(), output=args.output.resolve(), repository=args.repository.resolve())
    print(json.dumps(payload, sort_keys=True))
