"""Portable, explicit provenance records for trusted study attempts.

This package is the sole public provenance API.  Keeping the implementation
here avoids the former ``provenance.py``/``provenance/`` import collision,
where Python selected the package but silently hid the record definitions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import pathlib
from typing import Any


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: str | pathlib.Path) -> str:
    value = pathlib.PurePosixPath(str(path).replace("\\", "/"))
    if value.is_absolute() or ".." in value.parts:
        raise ValueError(f"paths in study manifests must be portable relative paths: {path}")
    return value.as_posix()


@dataclass(frozen=True)
class ProgramSnapshot:
    path: str
    sha256: str
    size_bytes: int
    content: str

    @classmethod
    def from_file(cls, path: pathlib.Path, relative_path: str) -> "ProgramSnapshot":
        data = path.read_bytes()
        return cls(portable_path(relative_path), sha256_bytes(data), len(data), data.decode())


@dataclass(frozen=True)
class AttemptManifest:
    schema: str
    study_id: str
    campaign_id: str
    contract_id: str
    attempt_number: int
    training_seed: int
    parent_experiment_id: str | None
    hypothesis: str
    intended_change: str
    change_category: str
    candidate_source_before_sha256: str
    candidate_source_after_sha256: str
    candidate_patch: str | None
    starting_source_hash: str
    evaluator_hashes: dict[str, str]
    dataset_hashes: dict[str, str]
    tokenizer_hash: str
    program: ProgramSnapshot
    agent_context_files: dict[str, str]
    system_context_text: str | None
    agent_model_identifier: str
    reasoning_setting: str
    api_configuration_available: bool
    api_configuration: dict[str, Any] | None
    training_environment: dict[str, Any]
    gpu_identity: dict[str, Any]
    command: list[str]
    training_result: dict[str, Any]
    inference_result: dict[str, Any]
    inference_raw_repetitions: list[float]
    pareto_decision: dict[str, Any]
    checkpoint_hash: str | None

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["program"]["path"] = portable_path(result["program"]["path"])
        result["command"] = [str(part) for part in self.command]
        return result


REQUIRED_ATTEMPT_FIELDS = tuple(AttemptManifest.__dataclass_fields__.keys())


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


__all__ = [
    "AttemptManifest",
    "ProgramSnapshot",
    "REQUIRED_ATTEMPT_FIELDS",
    "portable_path",
    "sha256_bytes",
    "sha256_file",
    "write_json",
]
