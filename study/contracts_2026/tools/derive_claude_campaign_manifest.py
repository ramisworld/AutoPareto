#!/usr/bin/env python3
"""Derive the frozen manifest for a single-arm Claude Code Scientist campaign.

The base is the last executed Pair-A Codex manifest. Everything scientific is
inherited unchanged: contracts, schedule, data, tokenizer, inference protocol,
baseline registration, calibration and paired baseline checkpoints. Only the
Scientist configuration, the hashes of the files the Claude backend change
touched, the local and remote state roots, and (after a pod restore) the A40
SSH host and GPU UUID are replaced. Every replaced field is recorded under
``derivation``. The written manifest is then checked with the controller's own
launch gate, so a manifest this tool accepts is one ``run`` will accept on the Mac.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import os
import pathlib
from typing import Any

from study.contracts_2026 import campaign_controller as controller
from study.contracts_2026.scientist_config import resolve_scientist_config

ROOT = controller.ROOT
SCHEMA = "autopareto_pair_a_claude_scientist_v1"
# The only deployed files the Claude backend and harness fixes may change.
EXPECTED_CHANGES = {
    "study/contracts_2026/bridge.py",
    "study/contracts_2026/campaign_controller.py",
    "study/contracts_2026/candidate_process.py",
    "study/contracts_2026/isolation.py",
    "study/contracts_2026/scientist_config.py",
    "study/contracts_2026/train_launcher.py",
}
CODEX_CAMPAIGN_HISTORY = ("continuation", "operational_amendments")


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive(args: argparse.Namespace) -> dict[str, Any]:
    base = json.loads(args.base.read_text())
    config = resolve_scientist_config(
        model=args.model,
        reasoning=args.reasoning,
        scientist_bin=args.scientist_bin,
        timeout_seconds=args.scientist_timeout_seconds,
        agent_backend="claude",
    )
    config.validate()
    manifest = copy.deepcopy(base)
    for key in CODEX_CAMPAIGN_HISTORY:
        manifest.pop(key, None)

    changed: dict[str, dict[str, str]] = {}
    for relative, before in base["immutable_deployment"].items():
        after = _sha256(ROOT / relative)
        if after != before:
            changed[relative] = {"before": before, "after": after}
            manifest["immutable_deployment"][relative] = after
    unexpected = sorted(set(changed) - EXPECTED_CHANGES)
    if unexpected:
        raise RuntimeError(f"deployed files changed outside the Claude backend amendment: {unexpected}")

    replaced: dict[str, dict[str, Any]] = {}

    def replace(section: str, key: str, value: Any) -> None:
        if value is None or manifest[section].get(key) == value:
            return
        replaced[f"{section}.{key}"] = {"before": manifest[section].get(key), "after": value}
        manifest[section][key] = value

    replace("persistent_paths", "local_official_state", str(args.state_root.resolve()))
    replace("persistent_paths", "trusted_state_root", args.trusted_state_root)
    replace("a40", "ssh_host", args.a40_ssh_host)
    replace("a40", "gpu_uuid", args.a40_gpu_uuid)
    for contract_id, value in (("T", args.a40_paired_baseline_t), ("N", args.a40_paired_baseline_n)):
        if value is None:
            continue
        path, _, digest = value.rpartition(":")
        if not path.startswith("/") or len(digest) != 64:
            raise SystemExit(f"--a40-paired-baseline-{contract_id.lower()} must be /absolute/path:sha256")
        before = manifest["a40"]["paired_baseline_checkpoints"].get(contract_id)
        after = {"path": path, "sha256": digest}
        if before != after:
            replaced[f"a40.paired_baseline_checkpoints.{contract_id}"] = {"before": before, "after": after}
            manifest["a40"]["paired_baseline_checkpoints"][contract_id] = after

    manifest.update(
        schema=SCHEMA,
        status="FROZEN_PRE_STUDY_CLAUDE_T_ONLY_FRESH_0_OF_25",
        frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        scientific_execution_launched=False,
        official_pair_a_attempts=0,
        manifest_sha256=None,
        supersedes=None,
        scientist={
            "provider": "headless_claude_code_cli",
            "model": config.model,
            "reasoning": config.reasoning,
            "codex_version": config.codex_version,
            "execution_backend": config.execution_backend,
            "configuration": config.as_dict(),
            "fresh_ephemeral_invocations": True,
            "campaign_histories_shared": False,
        },
        derivation={
            "base_manifest": {"path": str(args.base.resolve()), "sha256": _sha256(args.base)},
            "active_contracts": ["T"],
            "remote_run_namespace": args.remote_run_namespace,
            "removed_codex_campaign_history": [key for key in CODEX_CAMPAIGN_HISTORY if key in base],
            "changed_immutable_paths": changed,
            "replaced_fields": replaced,
            "scientist_before": base["scientist"]["configuration"],
            "reason": "Claude Code CLI Scientist backend and early-return checkpoint fix; "
                      "contracts, schedule, data, inference protocol and baseline are inherited unchanged",
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)

    digest = _sha256(args.output)
    controller._verify_final_manifest(args.output, config, digest)
    controller._verify_local_state_config(json.loads(args.output.read_text()), args.state_root)
    return {"manifest": str(args.output), "sha256": digest, "changed": changed, "replaced": replaced}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--state-root", type=pathlib.Path, required=True)
    parser.add_argument("--trusted-state-root", required=True)
    parser.add_argument("--scientist-bin", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reasoning", required=True)
    parser.add_argument("--scientist-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--a40-ssh-host")
    parser.add_argument("--a40-gpu-uuid")
    parser.add_argument("--remote-run-namespace", default="claude-opus55")
    parser.add_argument("--a40-paired-baseline-t", help="re-registered T checkpoint as /path:sha256")
    parser.add_argument("--a40-paired-baseline-n", help="re-registered N checkpoint as /path:sha256")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite an existing manifest: {args.output}")
    print(json.dumps(derive(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
