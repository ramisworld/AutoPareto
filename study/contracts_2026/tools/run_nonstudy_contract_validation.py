"""Run N1 then T1 through the production controller in isolated non-study state.

This intentionally uses the same Scientist, parent/source staging, transfer,
durable A40 execution, validation, inference, and result-commit code as Pair A.
Only its explicit local and remote state roots differ, so it can never create an
official scheduled attempt or contribute to an official campaign history.
"""

from __future__ import annotations

import argparse
import json
import pathlib

from study.contracts_2026 import campaign_controller as controller
from study.contracts_2026.bridge import A40RemoteConfig, A40Transport, sha256_file
from study.contracts_2026.scientist_config import resolve_scientist_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", type=pathlib.Path, required=True)
    parser.add_argument("--baseline-record", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reasoning", required=True)
    parser.add_argument("--scientist-bin", required=True)
    parser.add_argument("--scientist-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--a40-host", required=True)
    parser.add_argument("--a40-identity", required=True)
    parser.add_argument("--a40-port", type=int, default=None)
    parser.add_argument("--a40-repository", required=True)
    parser.add_argument("--a40-python", default=None)
    parser.add_argument("--a40-study-venv", default=None)
    parser.add_argument("--a40-cache", required=True)
    parser.add_argument("--a40-state-root", required=True)
    parser.add_argument("--a40-paired-baseline-t-checkpoint", default=None)
    parser.add_argument("--a40-paired-baseline-n-checkpoint", default=None)
    parser.add_argument("--remote-run-namespace", required=True)
    args = parser.parse_args(argv)

    state_root = args.state_root.resolve()
    manifest = args.manifest.resolve(strict=True)
    baseline = args.baseline_record.resolve(strict=True)
    if state_root.exists():
        raise SystemExit("non-study state root must not already exist")
    scientist_config = resolve_scientist_config(
        model=args.model,
        reasoning=args.reasoning,
        scientist_bin=args.scientist_bin,
        timeout_seconds=args.scientist_timeout_seconds,
    )
    remote_config = dict(
        host=args.a40_host,
        port=args.a40_port,
        identity_file=args.a40_identity,
        repository=args.a40_repository,
        python_path=args.a40_python,
        study_venv=args.a40_study_venv,
        registered_cache=args.a40_cache,
        trusted_state_root=args.a40_state_root,
    )
    if args.a40_paired_baseline_t_checkpoint is not None:
        remote_config["paired_baseline_t_checkpoint"] = args.a40_paired_baseline_t_checkpoint
    if args.a40_paired_baseline_n_checkpoint is not None:
        remote_config["paired_baseline_n_checkpoint"] = args.a40_paired_baseline_n_checkpoint
    transport = A40Transport(A40RemoteConfig(**remote_config))
    transport.verify_gpu_identity(controller.A40_UUID)
    state = controller._initialize_state(
        state_root, baseline, scientist_config, sha256_file(manifest)
    )
    state_path = state_root / "controller_state.json"
    rows = controller._schedule_from_frozen_file()[:2]
    if [(row.contract_id, row.campaign_attempt) for row in rows] != [("N", 1), ("T", 1)]:
        raise RuntimeError("frozen schedule does not begin N1 then T1")
    completed = []
    for row in rows:
        completed.append(controller._run_position(
            state_root=state_root,
            state=state,
            state_path=state_path,
            row=row,
            manifest_path=manifest,
            transport=transport,
            scientist_config=scientist_config,
            study_scope="nonstudy_validation",
            remote_run_namespace=args.remote_run_namespace,
        ))
    print(json.dumps({
        "schema": "autopareto_nonstudy_contract_validation_v1",
        "status": "completed",
        "state_root": str(state_root),
        "remote_state_root": args.a40_state_root,
        "positions": [1, 2],
        "results": completed,
    }, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
