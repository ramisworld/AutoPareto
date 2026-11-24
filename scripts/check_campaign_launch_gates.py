#!/usr/bin/env python3
"""Run the campaign controller's launch gates read-only, without creating state or calling the agent.

``campaign_controller run`` verifies the manifest, transport and remote files before it creates
``controller_state.json``; ``preflight`` needs an existing state. This script calls the same
controller functions, in the same order, and stops before ``_initialize_state``. It also checks the
registered baseline record and that the local state root holds nothing but the frozen manifest.

It writes nothing locally or remotely. The remote checks run over the controller's own transport
(the SSH proxy), so they also prove that path works.

Usage (repository root; same values as the run command):
    PYTHONPATH=$PWD python3 -B scripts/check_campaign_launch_gates.py --manifest M --manifest-sha256 S \
        --state-root R --baseline-record B --a40-host H ... (see --help)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from study.contracts_2026 import campaign_controller as controller  # noqa: E402
from study.contracts_2026.bridge import A40RemoteConfig, A40Transport  # noqa: E402
from study.contracts_2026.scientist_config import resolve_scientist_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("manifest", "state-root", "baseline-record"):
        parser.add_argument(f"--{name}", type=pathlib.Path, required=True)
    for name in ("manifest-sha256", "scientist-bin", "model", "reasoning", "a40-host", "a40-identity",
                 "a40-repository", "a40-python", "a40-study-venv", "a40-cache", "a40-state-root",
                 "a40-paired-baseline-t-checkpoint", "a40-paired-baseline-n-checkpoint"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--scientist-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--contracts", default="T")
    args = parser.parse_args()
    active = tuple(args.contracts.split(","))
    report: dict[str, object] = {}

    config = resolve_scientist_config(model=args.model, reasoning=args.reasoning, scientist_bin=args.scientist_bin,
                                      timeout_seconds=args.scientist_timeout_seconds, agent_backend="claude")
    config.validate()
    report["scientist"] = {"model": config.model, "reasoning": config.reasoning, "cli": config.codex_version}
    manifest_path = args.manifest.resolve()
    state_root = args.state_root.resolve()
    manifest = controller._verify_final_manifest(manifest_path, config, args.manifest_sha256)
    report["manifest_gate"] = "passed"
    transport = A40Transport(A40RemoteConfig(
        host=args.a40_host, port=None, identity_file=args.a40_identity, repository=args.a40_repository,
        python_path=args.a40_python, study_venv=args.a40_study_venv, registered_cache=args.a40_cache,
        trusted_state_root=args.a40_state_root,
        paired_baseline_t_checkpoint=args.a40_paired_baseline_t_checkpoint,
        paired_baseline_n_checkpoint=args.a40_paired_baseline_n_checkpoint,
    ))
    controller._verify_transport_config(manifest, transport.config)
    controller._verify_local_state_config(manifest, state_root)
    derivation = manifest.get("derivation") or {}
    if tuple(derivation.get("active_contracts", ())) != active:
        raise SystemExit(f"manifest freezes {derivation.get('active_contracts')}, not {list(active)}")
    report["transport_and_state_config"] = "match manifest"

    hashes = controller._manifest_hashes(manifest)
    transport.verify_trusted_files(hashes)
    report["remote_trusted_files"] = f"{len(hashes)} files match the manifest"
    uuid = controller._manifest_a40_uuid(manifest)
    transport.verify_gpu_identity(uuid)
    report["remote_gpu"] = uuid
    absolute = controller._registered_cache_file_hashes(str(transport.config.registered_cache))
    absolute.update(controller._paired_baseline_checkpoint_hashes(manifest, active))
    for arm in ("T", "N"):  # the N checkpoint is not used by a T-only study, but preflight checks it
        row = manifest["a40"]["paired_baseline_checkpoints"][arm]
        absolute[str(row["path"])] = str(row["sha256"])
    transport.verify_absolute_files(absolute)
    report["remote_absolute_files"] = {"count": len(absolute),
                                       "paired_baseline_checkpoints": manifest["a40"]["paired_baseline_checkpoints"]}

    baseline = controller._initial_baseline(args.baseline_record.resolve())
    controller.protected_search_metrics({"inference_result": baseline.get("inference_result"), **baseline})
    report["baseline"] = {key: baseline.get(key) for key in ("baseline_experiment_id", "protected_validation_bpb",
                                                             "record_scope")}

    leftovers = sorted(str(path.relative_to(state_root)) for path in state_root.rglob("*") if path.is_file())
    if leftovers != [str(manifest_path.relative_to(state_root))]:
        raise SystemExit(f"state root is not fresh: {leftovers}")
    report["local_state_root"] = "fresh: only the frozen manifest"
    report["status"] = "launch_gates_passed_read_only"
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
