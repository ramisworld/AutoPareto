"""Campaign-local agent visibility and isolated context staging."""

from __future__ import annotations

import json
import hashlib
import os
import pathlib
import platform
import signal
import subprocess
import shutil
import sys
import tempfile
import threading
import time
import uuid


AGENT_VISIBLE_PATHS = (
    "train.py",
    "prepare.py",
    "study_runtime.py",
    "program.md",
    "campaign_view.json",
)
AGENT_WRITABLE_PATHS = ("train.py",)
SCIENTIST_PARENT_SELECTION_SCHEMA = pathlib.Path(__file__).with_name("scientist_parent_selection_schema.json").resolve()
SCIENTIST_CHILD_EDIT_SCHEMA = pathlib.Path(__file__).with_name("scientist_child_edit_schema.json").resolve()
SCIENTIST_PROPOSAL_SCHEMAS = frozenset((SCIENTIST_PARENT_SELECTION_SCHEMA, SCIENTIST_CHILD_EDIT_SCHEMA))
FORBIDDEN_PATH_PREFIXES = (
    ".git",
    "results",
    "queue",
    "worktrees",
    "docs",
    "protocol",
    "README.md",
    "PROJECT_STATE.md",
    "FORENSIC_RESEARCH_AUDIT.md",
    "AUTOPARETO_TRAJECTORY_AUDIT.md",
)


class SandboxUnavailableError(RuntimeError):
    """The host has no enforceable allowlisted filesystem sandbox."""


class ScientistExecutionTimeout(TimeoutError):
    """Codex exceeded its bounded local Scientist deadline."""


def _is_codex_exec(command: list[str]) -> bool:
    """Identify the headless Codex entry point that owns its native sandbox."""
    return len(command) >= 2 and pathlib.Path(str(command[0])).name in {"codex", "codex-cli"} and command[1] == "exec"


# Claude Code Scientist: file tools only. No Bash, web, task or MCP tools.
CLAUDE_SCIENTIST_TOOLS = "Read,Edit,Write,Glob,Grep"
_CLAUDE_REQUIRED_FLAGS = {
    "--output-format": "stream-json",
    "--permission-mode": "acceptEdits",
    "--setting-sources": "local",
    "--tools": CLAUDE_SCIENTIST_TOOLS,
}
_CLAUDE_REQUIRED_SWITCHES = (
    "--verbose", "--no-session-persistence", "--safe-mode", "--strict-mcp-config", "--disable-slash-commands",
)
_CLAUDE_FORBIDDEN_TOKENS = frozenset({
    "--dangerously-skip-permissions", "--allow-dangerously-skip-permissions", "--add-dir",
    "--fallback-model", "--resume", "-r", "--continue", "-c", "--fork-session", "--mcp-config",
    "--plugin-dir", "--plugin-url", "--agents", "--agent", "--settings", "--chrome", "--bg",
    "--background", "--remote-control", "--worktree", "-w", "--bare", "--allowedTools",
    "--allowed-tools", "--system-prompt", "--append-system-prompt", "--system-prompt-file",
    "--append-system-prompt-file", "--permission-prompt-tool", "--plugin-dir-no-mcp",
})


def _is_claude_print(command: list[str]) -> bool:
    """Identify a headless Claude Code CLI (print-mode) Scientist call."""
    return len(command) >= 2 and pathlib.Path(str(command[0])).name == "claude" and command[1] == "-p"


def claude_output_schema(path: pathlib.Path) -> dict:
    """A registered proposal schema in the form Claude Code's --json-schema accepts.

    Claude Code's validator does not load the draft 2020-12 meta-schema that the
    registered files declare, so only the `$schema` declaration is dropped; the
    schemas use only keywords that mean the same in draft-07.
    """
    schema = json.loads(path.read_text())
    schema.pop("$schema", None)
    return schema


def _validate_claude_command(command: list[str], scientist_config) -> None:
    """Require the registered, permission-checked Claude Code invocation."""
    forbidden = sorted(set(command) & _CLAUDE_FORBIDDEN_TOKENS)
    if forbidden:
        raise SandboxUnavailableError(f"unsafe or unregistered Claude Code Scientist flags: {forbidden}")
    values = {command[index]: command[index + 1] for index in range(len(command) - 1) if command[index].startswith("--")}
    for flag, expected in _CLAUDE_REQUIRED_FLAGS.items():
        if values.get(flag) != expected:
            raise SandboxUnavailableError(f"Claude Code Scientist requires {flag} {expected}")
    for switch in _CLAUDE_REQUIRED_SWITCHES:
        if switch not in command:
            raise SandboxUnavailableError(f"Claude Code Scientist requires {switch}")
    if scientist_config is None or values.get("--model") != scientist_config.model:
        raise SandboxUnavailableError("Claude Code Scientist model must equal the resolved ScientistConfig model")
    if values.get("--effort") != scientist_config.reasoning:
        raise SandboxUnavailableError("Claude Code Scientist effort must equal the resolved ScientistConfig effort")
    try:
        schema = json.loads(values.get("--json-schema", ""))
    except json.JSONDecodeError as exc:
        raise SandboxUnavailableError("Claude Code Scientist requires a JSON output schema") from exc
    registered = [claude_output_schema(path) for path in sorted(SCIENTIST_PROPOSAL_SCHEMAS)]
    if schema not in registered:
        raise SandboxUnavailableError("Claude Code Scientist requires a registered phase-specific proposal schema")


def _validate_native_codex_command(
    command: list[str], context: pathlib.Path, required_sandbox: str = "workspace-write"
) -> None:
    """Require Codex's scoped workspace-write sandbox for Scientist execution."""
    if any(token in {"--dangerously-bypass-approvals-and-sandbox", "--oss"} for token in command):
        raise SandboxUnavailableError("unsafe or unsupported Codex execution mode")
    if "--add-dir" in command:
        raise SandboxUnavailableError("native Codex Scientist sandbox cannot widen its writable root")
    sandbox = None
    for index, token in enumerate(command[:-1]):
        if token in {"-s", "--sandbox"}:
            sandbox = command[index + 1]
            break
    if sandbox != required_sandbox:
        raise SandboxUnavailableError(f"Codex Scientist execution requires sandbox={required_sandbox}")
    working_root = None
    for index, token in enumerate(command[:-1]):
        if token in {"-C", "--cd"}:
            working_root = pathlib.Path(command[index + 1]).resolve()
            break
    if working_root != context.resolve():
        raise SandboxUnavailableError("Codex Scientist working root is not the staged context")
    schema_values = [command[index + 1] for index, token in enumerate(command[:-1]) if token == "--output-schema"]
    if len(schema_values) != 1 or pathlib.Path(schema_values[0]).resolve() not in SCIENTIST_PROPOSAL_SCHEMAS:
        raise SandboxUnavailableError("Codex Scientist execution requires a registered phase-specific proposal output schema")


def _docker_codex_command(
    command: list[str],
    context: pathlib.Path,
    *,
    scientist_config,
    environment: dict[str, str] | None,
) -> list[str]:
    """Place the exact Codex command in a disposable, non-root container."""
    _validate_native_codex_command(command, context, scientist_config.sandbox_mode)
    if not isinstance(environment, dict):
        raise ValueError("Docker Scientist execution requires the isolated environment")
    codex_home = environment.get("CODEX_HOME")
    if not codex_home:
        raise ValueError("Docker Scientist execution requires CODEX_HOME")
    auth_file = (pathlib.Path(codex_home) / "auth.json").resolve()
    if not auth_file.is_file() or auth_file.is_symlink():
        raise ValueError("Docker Scientist execution requires a regular read-only auth file")
    translated = list(command)
    for index, token in enumerate(translated[:-1]):
        if token in {"-C", "--cd"}:
            translated[index + 1] = "/workspace"
            break
    schema_index = translated.index("--output-schema")
    schema_path = pathlib.Path(translated[schema_index + 1]).resolve()
    translated[schema_index + 1] = f"/run/codex/{schema_path.name}"
    uid = os.getuid()
    gid = os.getgid()
    return [
        scientist_config.container_engine,
        "run",
        "--rm",
        "--init",
        "--read-only",
        "--network=bridge",
        "--user",
        f"{uid}:{gid}",
        "--cap-drop=ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "256",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,mode=1777",
        "--mount",
        f"type=bind,src={context.resolve()},dst=/workspace",
        "--mount",
        f"type=bind,src={auth_file},dst=/run/codex/auth.json,ro",
        "--mount",
        f"type=bind,src={schema_path},dst=/run/codex/{schema_path.name},ro",
        "--workdir",
        "/workspace",
        "--env",
        "HOME=/tmp/scientist-home",
        "--env",
        "CODEX_HOME=/run/codex",
        "--env",
        "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "--env",
        "TMPDIR=/tmp",
        "--env",
        "PYTHONNOUSERSITE=1",
        "--env",
        "LC_ALL=C",
        "--env",
        "AUTOPARETO_SCIENTIST_SANDBOX=1",
        scientist_config.container_image,
        *translated[1:],
    ]


def _safe_relative(path: pathlib.Path) -> str:
    value = path.as_posix()
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe context path: {path}")
    return value


def path_is_forbidden(relative_path: str) -> bool:
    value = pathlib.PurePosixPath(relative_path).as_posix()
    return any(value == prefix or value.startswith(prefix + "/") for prefix in FORBIDDEN_PATH_PREFIXES)


def validate_context_paths(paths: list[str] | tuple[str, ...]) -> None:
    expected = set(AGENT_VISIBLE_PATHS)
    observed = set(paths)
    if observed != expected:
        raise ValueError(f"agent context must equal the frozen allowlist; observed={sorted(observed)}")
    if any(path_is_forbidden(path) for path in paths):
        raise ValueError("forbidden path entered the agent context")


def write_allowlist(path: pathlib.Path) -> None:
    payload = {
        "schema": "autopareto_agent_allowlist_v1",
        "visible_paths": list(AGENT_VISIBLE_PATHS),
        "writable_paths": list(AGENT_WRITABLE_PATHS),
        "forbidden_prefixes": list(FORBIDDEN_PATH_PREFIXES),
        "notes": [
            "The agent receives only its program snapshot and campaign-local view.",
            "No existing result, database, paper, audit, downstream, or other campaign path is staged.",
            "Protected evaluator and orchestration internals are not agent-visible.",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def stage_agent_context(
    *,
    destination: pathlib.Path,
    baseline_train: pathlib.Path,
    baseline_prepare: pathlib.Path,
    study_runtime: pathlib.Path,
    program: pathlib.Path,
    campaign_view: pathlib.Path,
) -> pathlib.Path:
    """Stage exactly five files into a new context; never copy a whole worktree."""
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite an existing context: {destination}")
    destination.mkdir(parents=True)
    sources = {
        "train.py": baseline_train,
        "prepare.py": baseline_prepare,
        "study_runtime.py": study_runtime,
        "program.md": program,
        "campaign_view.json": campaign_view,
    }
    validate_context_paths(tuple(sources))
    for relative, source in sources.items():
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"context source must be a regular file: {source}")
        target = destination / relative
        shutil.copy2(source, target)
    return destination


def _sandbox_profile(context: pathlib.Path, codex_home: str | None = None) -> str:
    """Return a deny-by-default macOS sandbox profile for one context."""
    if platform.system() != "Darwin":
        raise SandboxUnavailableError("macOS sandbox profile requested on a non-macOS host")
    context = context.resolve()
    python_prefix = pathlib.Path(sys.prefix).resolve()
    profile_paths = [context, python_prefix, pathlib.Path(sys.executable).resolve()]
    profile_literal_reads: list[pathlib.Path] = []
    if codex_home:
        codex_home_path = pathlib.Path(codex_home).expanduser().resolve()
        profile_paths.append(codex_home_path)
        auth_link = codex_home_path / "auth.json"
        if auth_link.is_symlink():
            profile_literal_reads.append(auth_link.resolve())

    def quote(path: pathlib.Path) -> str:
        return '"' + str(path).replace("\\", "\\\\").replace('"', '\\"') + '"'

    lines = [
        "(version 1)",
        "(deny default)",
        "(allow process-fork)",
        "(allow process-exec)",
        "(allow signal (target self))",
        "(allow sysctl-read)",
        "(allow mach-lookup)",
        # Codex's headless ChatGPT transport requires HTTPS and DNS.  Keep the
        # candidate sandbox restricted to DNS resolution plus TCP/443; SSH and
        # other remote-access executables remain explicitly denied below.
        "(allow network-outbound",
        "    (literal \"/private/var/run/mDNSResponder\")",
        "    (remote tcp \"*:443\"))",
        "(deny network-inbound)",
        "(deny process-exec (literal \"/usr/bin/ssh\"))",
        "(deny process-exec (literal \"/usr/bin/scp\"))",
        "(deny process-exec (literal \"/usr/bin/sftp\"))",
        "(deny process-exec (literal \"/usr/local/bin/ssh\"))",
        "(deny process-exec (literal \"/usr/local/bin/scp\"))",
        "(deny process-exec (literal \"/usr/local/bin/sftp\"))",
        # Python needs broad system-library reads.  The final two rules make
        # /Users deny-by-default while restoring access only to this context.
        "(allow file-read* (subpath \"/\"))",
        "(deny file-read* (subpath \"/Users\"))",
        "(allow file-read* (literal \"/dev/null\"))",
        "(allow file-read* (literal \"/dev/urandom\"))",
        "(allow file-write* (literal \"/dev/null\"))",
    ]
    lines.extend(f"(allow file-read* (subpath {quote(path)}))" for path in profile_paths)
    lines.extend(f"(allow file-read* (literal {quote(path)}))" for path in profile_literal_reads)
    lines.append(f"(allow file-write* (literal {quote(context / 'train.py')}))")
    if codex_home:
        lines.append(f"(allow file-write* (subpath {quote(pathlib.Path(codex_home).resolve())}))")
    return "\n".join(lines) + "\n"


def sandbox_command(
    command: list[str], context: pathlib.Path, codex_home: str | None = None
) -> tuple[list[str], pathlib.Path | None]:
    """Wrap an agent command in a deny-by-default filesystem sandbox.

    The returned temporary profile path is caller-owned and may be removed after
    the subprocess exits.  No unsandboxed fallback is provided.
    """
    context = context.resolve()
    if not context.is_dir():
        raise ValueError(f"agent context is not a directory: {context}")
    if platform.system() == "Darwin" and shutil.which("sandbox-exec") and _sandbox_exec_available():
        profile_fd, profile_name = tempfile.mkstemp(prefix="autopareto-sandbox-", suffix=".sb")
        os.close(profile_fd)
        profile = pathlib.Path(profile_name)
        profile.write_text(_sandbox_profile(context, codex_home))
        return ["sandbox-exec", "-f", str(profile), *command], profile
    if shutil.which("bwrap"):
        command_args = [
            "bwrap", "--die-with-parent", "--unshare-all", "--new-session",
            "--ro-bind", "/usr", "/usr",
            "--ro-bind", "/bin", "/bin",
            "--ro-bind", "/lib", "/lib",
            "--ro-bind", "/lib64", "/lib64",
            "--ro-bind", "/etc", "/etc",
            "--proc", "/proc", "--dev", "/dev",
            "--ro-bind", str(context), "/context",
        ]
        train_file = context / "train.py"
        if train_file.is_file() and not train_file.is_symlink():
            command_args.extend(("--bind", str(train_file), "/context/train.py"))
        command_args.extend(("--chdir", "/context", "--", *command))
        return command_args, None
    raise SandboxUnavailableError("no deny-by-default sandbox-exec or bubblewrap runtime is available")


def _sandbox_exec_available() -> bool:
    """Probe once because managed/macOS hosts may expose but disable sandbox-exec."""
    try:
        result = subprocess.run(
            ["sandbox-exec", "-p", "(version 1) (allow default)", "/usr/bin/true"],
            capture_output=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _atomic_json_write(path: pathlib.Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _supervised_codex(
    command: list[str],
    context: pathlib.Path,
    *,
    timeout_seconds: float,
    diagnostic_root: pathlib.Path,
    invocation_metadata: dict[str, object] | None,
    **kwargs,
) -> subprocess.CompletedProcess[str]:
    if kwargs.pop("capture_output", False) is not True or kwargs.pop("text", False) is not True:
        raise ValueError("supervised Scientist execution requires text capture of stdout and stderr")
    if kwargs.pop("stdin", None) != subprocess.DEVNULL:
        raise ValueError("supervised Scientist execution requires stdin=DEVNULL")
    environment = kwargs.pop("env", None)
    if kwargs:
        raise ValueError(f"unsupported supervised Scientist subprocess options: {sorted(kwargs)}")
    invocation_id = f"scientist-{time.time_ns()}-{uuid.uuid4().hex[:12]}"
    root = diagnostic_root.resolve() / invocation_id
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    stdout_path = root / "stdout.jsonl"
    stderr_path = root / "stderr.log"
    record_path = root / "invocation.json"
    started_at = time.time()
    runtime_command = list(command)
    docker_cidfile: pathlib.Path | None = None
    if len(runtime_command) >= 2 and runtime_command[1] == "run" and pathlib.Path(runtime_command[0]).name == "docker":
        docker_cidfile = root / "docker.cid"
        runtime_command[2:2] = ["--cidfile", str(docker_cidfile)]
    record: dict[str, object] = {
        "schema": "autopareto_scientist_invocation_v1",
        "invocation_id": invocation_id,
        "status": "RUNNING",
        "started_at_epoch": started_at,
        "pid": None,
        "command_sha256": hashlib.sha256("\0".join(runtime_command).encode()).hexdigest(),
        "command_summary": runtime_command[:-1],
        "context": str(context.resolve()),
        "timeout_seconds": float(timeout_seconds),
        "metadata": dict(invocation_metadata or {}),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    if docker_cidfile is not None:
        record["container_id_path"] = str(docker_cidfile)
    _atomic_json_write(record_path, record)
    try:
        process = subprocess.Popen(
            runtime_command,
            cwd=context,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            close_fds=True,
            env=environment,
        )
    except BaseException as exc:
        record.update({
            "status": "FAILED",
            "failure_class": "infrastructure",
            "failure": f"Scientist process could not start: {type(exc).__name__}: {exc}",
            "completed_at_epoch": time.time(),
            "returncode": None,
            "stdout_bytes": 0,
            "stderr_bytes": 0,
        })
        _atomic_json_write(record_path, record)
        raise
    record["pid"] = process.pid
    _atomic_json_write(record_path, record)
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    def drain(pipe, path: pathlib.Path, chunks: list[str]) -> None:
        with path.open("w") as handle:
            for line in iter(pipe.readline, ""):
                chunks.append(line)
                handle.write(line)
                handle.flush()
            pipe.close()

    stdout_thread = threading.Thread(target=drain, args=(process.stdout, stdout_path, stdout_chunks), daemon=True)
    stderr_thread = threading.Thread(target=drain, args=(process.stderr, stderr_path, stderr_chunks), daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    timed_out = threading.Event()
    process_finished = threading.Event()

    def signal_process_group(signal_number: int) -> None:
        try:
            os.killpg(process.pid, signal_number)
        except ProcessLookupError:
            pass

    def deadline_watchdog() -> None:
        if process_finished.wait(timeout=float(timeout_seconds)):
            return
        if process.poll() is not None:
            return
        timed_out.set()
        signal_process_group(signal.SIGTERM)
        if process_finished.wait(timeout=5.0) or process.poll() is not None:
            return
        if docker_cidfile is not None and docker_cidfile.is_file():
            container_id = docker_cidfile.read_text().strip()
            if container_id:
                try:
                    subprocess.run(
                        [runtime_command[0], "kill", container_id],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=5,
                        check=False,
                    )
                except (OSError, subprocess.SubprocessError):
                    pass
        signal_process_group(signal.SIGKILL)

    watchdog = threading.Thread(target=deadline_watchdog, name=f"autopareto-watchdog-{invocation_id}", daemon=True)
    watchdog.start()
    try:
        returncode = process.wait()
    finally:
        process_finished.set()
        watchdog.join(timeout=6)
    if timed_out.is_set():
        record["status"] = "FAILED"
        record["failure_class"] = "infrastructure"
        record["failure"] = "Scientist execution exceeded its bounded timeout"
    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)
    record["completed_at_epoch"] = time.time()
    record["returncode"] = int(returncode)
    record["stdout_bytes"] = stdout_path.stat().st_size if stdout_path.exists() else 0
    record["stderr_bytes"] = stderr_path.stat().st_size if stderr_path.exists() else 0
    if not timed_out.is_set():
        record["status"] = "COMPLETE" if returncode == 0 else "FAILED"
        if returncode != 0:
            record["failure_class"] = "infrastructure"
    _atomic_json_write(record_path, record)
    completed = subprocess.CompletedProcess(command, returncode, "".join(stdout_chunks), "".join(stderr_chunks))
    setattr(completed, "scientist_diagnostics", record)
    if timed_out.is_set():
        raise ScientistExecutionTimeout(
            f"Scientist execution timed out after {timeout_seconds:g}s; diagnostics={record_path}"
        )
    return completed


def run_agent_command(
    command: list[str],
    context: pathlib.Path,
    *,
    timeout_seconds: float | None = None,
    diagnostic_root: pathlib.Path | None = None,
    invocation_metadata: dict[str, object] | None = None,
    scientist_config=None,
    **kwargs,
) -> subprocess.CompletedProcess:
    """Run an agent with native Codex sandboxing and optional supervision."""
    context = context.resolve()
    if _is_claude_print(command):
        if scientist_config is None or getattr(scientist_config, "agent_backend", "codex") != "claude":
            raise SandboxUnavailableError("Claude Code Scientist execution requires a claude ScientistConfig")
        _validate_claude_command(command, scientist_config)
        if diagnostic_root is None:
            raise ValueError("supervised Scientist execution requires a persistent diagnostic root")
        return _supervised_codex(
            command,
            context,
            timeout_seconds=scientist_config.timeout_seconds if timeout_seconds is None else timeout_seconds,
            diagnostic_root=diagnostic_root,
            invocation_metadata=invocation_metadata,
            **kwargs,
        )
    if _is_codex_exec(command):
        if scientist_config is not None:
            if scientist_config.execution_backend == "docker":
                if scientist_config.sandbox_mode != "danger-full-access":
                    raise SandboxUnavailableError(
                        "Docker Scientist execution requires danger-full-access inside the outer container"
                    )
            elif scientist_config.sandbox_mode != "workspace-write":
                raise SandboxUnavailableError("host Scientist execution requires native workspace-write sandbox")
            if scientist_config.approval_policy != "never":
                raise SandboxUnavailableError("Scientist config requires approval_policy=never")
            timeout_seconds = scientist_config.timeout_seconds if timeout_seconds is None else timeout_seconds
            if scientist_config.execution_backend == "docker":
                command = _docker_codex_command(
                    command,
                    context,
                    scientist_config=scientist_config,
                    environment=kwargs.get("env"),
                )
            else:
                _validate_native_codex_command(command, context)
        else:
            _validate_native_codex_command(command, context)
        if timeout_seconds is not None:
            if diagnostic_root is None:
                raise ValueError("supervised Scientist execution requires a persistent diagnostic root")
            return _supervised_codex(
                command,
                context,
                timeout_seconds=timeout_seconds,
                diagnostic_root=diagnostic_root,
                invocation_metadata=invocation_metadata,
                **kwargs,
            )
        return subprocess.run(command, cwd=context, check=False, **kwargs)
    environment = kwargs.get("env")
    codex_home = environment.get("CODEX_HOME") if isinstance(environment, dict) else None
    wrapped, profile = sandbox_command(command, context, codex_home)
    try:
        return subprocess.run(wrapped, cwd=context, check=False, **kwargs)
    finally:
        if profile is not None:
            try:
                profile.unlink()
            except FileNotFoundError:
                pass
