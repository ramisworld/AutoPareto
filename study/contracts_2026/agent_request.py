"""Provider-neutral request envelope for the accessible agent context."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import pathlib
from typing import Any


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class AgentRequest:
    campaign_id: str
    contract_id: str
    attempt_number: int
    model_identifier: str
    reasoning_setting: str
    program_path: str
    program_text: str
    context_files: dict[str, str]
    api_configuration: dict[str, Any] | None
    system_context_text: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "autopareto_agent_request_v1",
            "campaign_id": self.campaign_id,
            "contract_id": self.contract_id,
            "attempt_number": self.attempt_number,
            "model_identifier": self.model_identifier,
            "reasoning_setting": self.reasoning_setting,
            "program": {
                "path": self.program_path,
                "sha256": _sha(self.program_text),
                "content": self.program_text,
            },
            "context_files": {
                path: {"sha256": _sha(content), "content": content}
                for path, content in sorted(self.context_files.items())
            },
            "api_configuration_available": self.api_configuration is not None,
            "api_configuration": self.api_configuration,
            "system_context_available": self.system_context_text is not None,
            "system_context_text": self.system_context_text,
            "hidden_provider_internals": "not recorded; unavailable to the orchestration layer",
        }


def write_request(path: pathlib.Path, request: AgentRequest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(request.as_dict(), indent=2, sort_keys=True) + "\n")
