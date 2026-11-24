"""End-to-end dry run of a single-arm (Contract T) campaign with the Claude Code backend.

No model call, GPU, network or RunPod access: the agent is the offline fake CLI
in ``fake_claude_cli/`` and the A40 is an in-process mock that returns clearly
labelled synthetic results. It exercises the real controller loop: parent
selection, child edit, contract validation, durable pending execution, result
commit, archive and view updates, failure recording, and resume after a
transport interruption.

Usage (from the repository root):
    python3 -m study.contracts_2026.tools.dry_run_claude_campaign --state-root <new empty dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any
from unittest import mock

from study.contracts_2026 import campaign_controller as controller
from study.contracts_2026.remote_runner import REMOTE_RESULT_SCHEMA
from study.contracts_2026.scientist_config import (
    CLAUDE_APPROVAL_POLICY,
    CLAUDE_SANDBOX_MODE,
    ScientistConfig,
)

FAKE_CLAUDE = pathlib.Path(__file__).resolve().with_name("fake_claude_cli") / "claude"
DRY_RUN_NAMESPACE = "claude-dry-run"
DRY_RUN_MANIFEST = {
    "dry_run": True,
    "derivation": {"active_contracts": ["T"], "remote_run_namespace": DRY_RUN_NAMESPACE},
    "a40": {"paired_baseline_checkpoints": {"T": {"path": "/dry-run/T/baseline.pt", "sha256": "0" * 64}}},
}


class _DryRunConfig:
    registered_cache = "/dev/null/dry-run-cache"

    @staticmethod
    def paired_baseline_checkpoint(contract_id: str) -> str:
        return f"dry-run/{contract_id}/baseline.pt"


class DryRunTransport:
    """Mock A40: deterministic synthetic results, optional failures and interruptions."""

    def __init__(self, *, fail_attempts=(), interrupt_attempts=()):
        self.config = _DryRunConfig()
        self.fail_attempts = set(fail_attempts)
        self.interrupt_attempts = set(interrupt_attempts)
        self.payloads: dict[str, pathlib.Path] = {}
        self.invocations: dict[str, dict[str, Any]] = {}
        self.launch_requests = 0
        self.remote_run_ids: list[str] = []

    def verify_trusted_files(self, _hashes) -> None:
        return None

    def verify_gpu_identity(self, _uuid) -> None:
        return None

    def verify_absolute_files(self, hashes) -> None:
        self.verified_absolute_files = dict(hashes)

    def transfer(self, *, payload: pathlib.Path, remote_run_id: str) -> str:
        self.remote_run_ids.append(remote_run_id)
        remote_dir = f"dry-run://{remote_run_id}"
        self.payloads[remote_dir] = json.loads((payload / "registered_metadata.json").read_text())
        self.payloads[remote_dir]["_source_sha256"] = hashlib.sha256((payload / "train.py").read_bytes()).hexdigest()
        return remote_dir

    def launch(self, *, remote_dir: str, output_name: str, invocation_id: str, invocation_metadata) -> None:
        # Like the real remote runner, a repeated request for an existing
        # invocation id is idempotent and never starts a second training run.
        self.launch_requests += 1
        self.invocations.setdefault(invocation_id, {"remote_dir": remote_dir, "metadata": dict(invocation_metadata)})

    def wait_for_invocation(self, invocation_id: str) -> dict[str, Any]:
        attempt = int(self.invocations[invocation_id]["metadata"]["campaign_attempt"])
        if attempt in self.interrupt_attempts:
            self.interrupt_attempts.discard(attempt)
            raise ConnectionError(f"dry-run transport interruption at attempt {attempt}")
        return {"record": {"status": "COMPLETE", "invocation_id": invocation_id}, "process_alive": False}

    def fetch_durable_result(self, record: dict[str, Any]) -> dict[str, Any]:
        invocation = self.invocations[record["invocation_id"]]
        meta = invocation["metadata"]
        attempt = int(meta["campaign_attempt"])
        common = {
            "schema": REMOTE_RESULT_SCHEMA,
            "campaign_id": meta["campaign_id"],
            "contract_id": meta["contract_id"],
            "attempt": attempt,
            "dry_run": True,
        }
        if attempt in self.fail_attempts:
            return {**common, "status": "scientific_failure", "failure_class": "candidate",
                    "error": "CandidateExitedBeforeStop: candidate training body returned before the protected stop"}
        digest = int(self.payloads[invocation["remote_dir"]]["_source_sha256"][:8], 16)
        bpb = 1.20 - 0.002 * attempt + (digest % 100) * 1e-5
        speed = 1.0 + 0.01 * (digest % 7)
        return {
            **common,
            "status": "completed",
            "protected_validation_bpb": bpb,
            "total_tokens": 85_000_000,
            "optimizer_updates": 163,
            "measured_training_seconds": 300.0,
            "inference_result": {
                "inference_status": "stable",
                "correctness": {"passed": True},
                "normalized_decode_throughput": speed,
                "normalized_decode_throughput_ci_95pct": [speed * 0.99, speed * 1.01],
                "peak_allocated_inference_memory_bytes": 260_000_000 + digest % 1000,
            },
        }


def dry_run_baseline() -> dict[str, Any]:
    return {
        "record_scope": "same_seed_baseline",
        "baseline_experiment_id": "dry-run-baseline",
        "protected_validation_bpb": 1.2,
        "dry_run": True,
        "inference_result": {
            "inference_status": "stable",
            "correctness": {"passed": True},
            "normalized_decode_throughput": 1.0,
            "peak_allocated_inference_memory_bytes": 260_000_000,
        },
    }


def dry_run_config(model: str = "claude-opus-5-5", effort: str = "xhigh") -> ScientistConfig:
    return ScientistConfig(
        model=model,
        reasoning=effort,
        scientist_bin=str(FAKE_CLAUDE),
        codex_version="0.0.0 (offline fake for AutoPareto dry runs)",
        execution_backend="host",
        container_engine="none",
        container_image="none",
        sandbox_mode=CLAUDE_SANDBOX_MODE,
        approval_policy=CLAUDE_APPROVAL_POLICY,
        timeout_seconds=60.0,
        agent_backend="claude",
    )


def run_dry_campaign(
    state_root: pathlib.Path,
    *,
    fail_attempts=(3,),
    interrupt_attempts=(4,),
) -> dict[str, Any]:
    """Run all 25 fixed-time positions of the single-arm study, then report the state."""
    state_root.mkdir(parents=True, exist_ok=False)
    baseline_path = state_root.parent / f"{state_root.name}.baseline.json"
    baseline_path.write_text(json.dumps(dry_run_baseline()))
    manifest_path = state_root.parent / f"{state_root.name}.manifest.json"
    manifest_path.write_text(json.dumps(DRY_RUN_MANIFEST))
    transport = DryRunTransport(fail_attempts=fail_attempts, interrupt_attempts=interrupt_attempts)
    config = dry_run_config()
    patches = [
        mock.patch.object(controller, "_verify_final_manifest", lambda *_a, **_k: DRY_RUN_MANIFEST),
        mock.patch.object(controller, "_verify_transport_config", lambda *_a, **_k: None),
        mock.patch.object(controller, "_verify_local_state_config", lambda *_a, **_k: None),
        mock.patch.object(controller, "_manifest_a40_uuid", lambda *_a, **_k: "GPU-DRY-RUN"),
        mock.patch.object(controller, "_manifest_hashes", lambda *_a, **_k: {}),
        mock.patch.object(controller, "_registered_cache_file_hashes", lambda *_a, **_k: {}),
    ]
    events: list[str] = []
    for patcher in patches:
        patcher.start()
    try:
        for _ in range(len(interrupt_attempts) + 1):
            try:
                controller.run_campaign(
                    state_root=state_root, baseline_path=baseline_path, manifest_path=manifest_path,
                    transport=transport, scientist_config=config, active_contracts=("T",),
                )
                events.append("run_completed")
                break
            except ConnectionError as exc:
                events.append(f"interrupted: {exc}; resuming")
    finally:
        for patcher in reversed(patches):
            patcher.stop()
    state = json.loads((state_root / "controller_state.json").read_text())
    history_path = state_root / "pair-a-fixed-time-seed3" / "history.jsonl"
    history = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
    return {
        "events": events,
        "launch_requests": transport.launch_requests,
        "unique_invocations": len(transport.invocations),
        "remote_run_ids": sorted(set(transport.remote_run_ids)),
        "verified_absolute_files": sorted(getattr(transport, "verified_absolute_files", {})),
        "active_contracts": state.get("active_contracts"),
        "positions": {key: value.get("status") for key, value in state["positions"].items() if value.get("status") != "not_in_study" and value.get("status") != "pending"},
        "not_in_study": sum(value.get("status") == "not_in_study" for value in state["positions"].values()),
        "history": [
            {
                "attempt": record.get("campaign_attempt", record.get("attempt")),
                "status": record.get("status"),
                "parent": record.get("parent_experiment_id"),
                "agent_model": record.get("agent_model_identifier"),
                "effort": record.get("reasoning_setting"),
                "bpb": record.get("protected_validation_bpb") or (record.get("result") or {}).get("protected_validation_bpb"),
                "decision": (record.get("pareto_decision") or (record.get("result") or {}).get("pareto_decision") or {}).get("decision"),
            }
            for record in history
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state-root", type=pathlib.Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_dry_campaign(args.state_root.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
