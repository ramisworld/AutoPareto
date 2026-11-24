"""Resolved configuration for one local Scientist provider (Codex CLI or Claude Code CLI)."""

from __future__ import annotations

from dataclasses import dataclass
import pathlib
import re
import shutil
import subprocess


DEFAULT_SCIENTIST_BIN = "/usr/local/bin/codex"
DEFAULT_EXECUTION_BACKEND = "docker"
DEFAULT_CONTAINER_ENGINE = "docker"
DEFAULT_CONTAINER_IMAGE = "autopareto-scientist-codex:0.153.0"
DEFAULT_SANDBOX_MODE = "danger-full-access"
HOST_SANDBOX_MODE = "workspace-write"
DEFAULT_APPROVAL_POLICY = "never"
DEFAULT_TIMEOUT_SECONDS = 600.0
AGENT_BACKENDS = frozenset({"codex", "claude"})
# Claude Code runs on the host. Its isolation is the CLI's own tool permissions
# (file tools only, edits limited to the staged context) plus the bridge's
# existing context-hash checks; these labels record that in the manifest.
CLAUDE_SANDBOX_MODE = "claude-file-tools-accept-edits"
CLAUDE_APPROVAL_POLICY = "acceptEdits"
CLAUDE_EFFORT_LEVELS = frozenset({"low", "medium", "high", "xhigh", "max"})
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_IMAGE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]*")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


@dataclass(frozen=True)
class ScientistConfig:
    """The single resolved Scientist configuration passed through the bridge."""

    model: str
    reasoning: str
    scientist_bin: str = DEFAULT_SCIENTIST_BIN
    codex_version: str = "unknown"
    execution_backend: str = DEFAULT_EXECUTION_BACKEND
    container_engine: str = DEFAULT_CONTAINER_ENGINE
    container_image: str = DEFAULT_CONTAINER_IMAGE
    container_image_digest: str = "unknown"
    sandbox_mode: str = DEFAULT_SANDBOX_MODE
    approval_policy: str = DEFAULT_APPROVAL_POLICY
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    # "codex_version" records the agent CLI version for either backend; the field
    # name is kept so existing Codex manifests verify unchanged.
    agent_backend: str = "codex"

    def validate(self) -> None:
        if not _TOKEN.fullmatch(self.model) or not _TOKEN.fullmatch(self.reasoning):
            raise ValueError("Scientist model and reasoning must be simple non-empty identifiers")
        if self.agent_backend not in AGENT_BACKENDS:
            raise ValueError(f"unsupported Scientist agent backend: {self.agent_backend}")
        if self.agent_backend == "claude":
            if self.execution_backend != "host":
                raise ValueError("Claude Code Scientist execution runs on the host")
            if not pathlib.Path(self.scientist_bin).is_file():
                raise FileNotFoundError(f"Scientist executable is missing: {self.scientist_bin}")
            if self.reasoning not in CLAUDE_EFFORT_LEVELS:
                raise ValueError(f"Claude Code effort must be one of {sorted(CLAUDE_EFFORT_LEVELS)}")
            if self.sandbox_mode != CLAUDE_SANDBOX_MODE or self.approval_policy != CLAUDE_APPROVAL_POLICY:
                raise ValueError("Claude Code Scientist requires file-tools-only acceptEdits permissions")
            if self.timeout_seconds <= 0:
                raise ValueError("Scientist timeout must be positive")
            return
        if self.execution_backend not in {"docker", "host"}:
            raise ValueError(f"unsupported Scientist execution backend: {self.execution_backend}")
        if self.execution_backend == "host" and not pathlib.Path(self.scientist_bin).is_file():
            raise FileNotFoundError(f"Scientist executable is missing: {self.scientist_bin}")
        if self.execution_backend == "docker":
            if shutil.which(self.container_engine) is None:
                raise FileNotFoundError(f"Scientist container engine is missing: {self.container_engine}")
            if not _IMAGE.fullmatch(self.container_image):
                raise ValueError(f"Scientist container image is malformed: {self.container_image}")
            if self.container_image_digest != "unknown" and not _DIGEST.fullmatch(self.container_image_digest):
                raise ValueError("Scientist container image digest is malformed")
        if self.execution_backend == "docker" and self.sandbox_mode != DEFAULT_SANDBOX_MODE:
            raise ValueError("Docker Scientist execution requires Codex danger-full-access inside the outer container")
        if self.execution_backend == "host" and self.sandbox_mode != HOST_SANDBOX_MODE:
            raise ValueError("host Scientist execution requires Codex workspace-write sandbox")
        if self.approval_policy != DEFAULT_APPROVAL_POLICY:
            raise ValueError("Scientist execution requires approval_policy=never")
        if self.timeout_seconds <= 0:
            raise ValueError("Scientist timeout must be positive")

    def as_dict(self) -> dict[str, object]:
        values: dict[str, object] = {
            "model": self.model,
            "reasoning": self.reasoning,
            "scientist_bin": self.scientist_bin,
            "codex_version": self.codex_version,
            "execution_backend": self.execution_backend,
            "container_engine": self.container_engine,
            "container_image": self.container_image,
            "container_image_digest": self.container_image_digest,
            "sandbox_mode": self.sandbox_mode,
            "approval_policy": self.approval_policy,
            "timeout_seconds": self.timeout_seconds,
        }
        if self.agent_backend != "codex":
            values["agent_backend"] = self.agent_backend
        return values


def resolve_codex_version(scientist_bin: str) -> str:
    """Read the selected CLI version without loading user configuration."""
    result = subprocess.run(
        [scientist_bin, "--version"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=10,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Scientist executable version probe failed: {result.returncode}")
    value = result.stdout.strip()
    if not value:
        raise RuntimeError("Scientist executable returned an empty version")
    return value


def resolve_container_identity(
    *,
    container_engine: str,
    container_image: str,
    scientist_bin: str,
) -> tuple[str, str]:
    """Resolve the pinned Codex version and local image ID from Docker."""
    engine = shutil.which(container_engine)
    if engine is None:
        raise FileNotFoundError(f"Scientist container engine is missing: {container_engine}")
    image_result = subprocess.run(
        [engine, "image", "inspect", "--format", "{{.Id}}", container_image],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=15,
        check=False,
    )
    if image_result.returncode != 0 or not _DIGEST.fullmatch(image_result.stdout.strip()):
        raise RuntimeError(f"Scientist container image is unavailable or unpinned: {container_image}")
    version_result = subprocess.run(
        [engine, "run", "--rm", "--read-only", "--entrypoint", scientist_bin, container_image, "--version"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=30,
        check=False,
    )
    if version_result.returncode != 0 or not version_result.stdout.strip():
        raise RuntimeError("Scientist Codex version probe failed inside the pinned container")
    return version_result.stdout.strip(), image_result.stdout.strip()


def resolve_scientist_config(
    *,
    model: str,
    reasoning: str,
    scientist_bin: str = DEFAULT_SCIENTIST_BIN,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    execution_backend: str = DEFAULT_EXECUTION_BACKEND,
    container_engine: str = DEFAULT_CONTAINER_ENGINE,
    container_image: str = DEFAULT_CONTAINER_IMAGE,
    agent_backend: str = "codex",
) -> ScientistConfig:
    if agent_backend == "claude":
        config = ScientistConfig(
            model=model,
            reasoning=reasoning,
            scientist_bin=scientist_bin,
            codex_version=resolve_codex_version(scientist_bin),
            execution_backend="host",
            container_engine="none",
            container_image="none",
            sandbox_mode=CLAUDE_SANDBOX_MODE,
            approval_policy=CLAUDE_APPROVAL_POLICY,
            timeout_seconds=float(timeout_seconds),
            agent_backend="claude",
        )
        config.validate()
        return config
    if execution_backend == "docker":
        codex_version, image_digest = resolve_container_identity(
            container_engine=container_engine,
            container_image=container_image,
            scientist_bin=scientist_bin,
        )
    else:
        codex_version = resolve_codex_version(scientist_bin)
        image_digest = "unknown"
    config = ScientistConfig(
        model=model,
        reasoning=reasoning,
        scientist_bin=scientist_bin,
        codex_version=codex_version,
        execution_backend=execution_backend,
        container_engine=container_engine,
        container_image=container_image,
        container_image_digest=image_digest,
        timeout_seconds=float(timeout_seconds),
    )
    config.validate()
    return config


__all__ = [
    "ScientistConfig",
    "resolve_codex_version",
    "resolve_container_identity",
    "resolve_scientist_config",
]
