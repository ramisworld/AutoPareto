#!/usr/bin/env python3
"""Prepare, but never launch, a verified Pair-A cross-A40 continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import tempfile

from study.contracts_2026 import campaign_controller
from study.contracts_2026.continuation import (
    build_continuation_manifest,
    prepare_continuation_state,
    sha256_file,
    validate_amendment_record,
)


ROOT = pathlib.Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-state", type=pathlib.Path, required=True)
    parser.add_argument("--destination-state", type=pathlib.Path, required=True)
    parser.add_argument("--prior-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--completed-positions", type=int, required=True)
    parser.add_argument("--calibration-result", type=pathlib.Path, required=True)
    parser.add_argument("--registered-cache-runtime-provenance", type=pathlib.Path, required=True)
    parser.add_argument("--new-gpu-uuid", required=True)
    parser.add_argument("--new-a40-host", required=True)
    parser.add_argument("--new-remote-state", required=True)
    parser.add_argument("--registered-cache", required=True)
    parser.add_argument("--remote-python", required=True)
    parser.add_argument("--study-venv", required=True)
    parser.add_argument("--paired-baseline-t-checkpoint", required=True)
    parser.add_argument("--paired-baseline-t-sha256", required=True)
    parser.add_argument("--paired-baseline-n-checkpoint", required=True)
    parser.add_argument("--paired-baseline-n-sha256", required=True)
    parser.add_argument("--manifest-name", default="final_pair_a_continuation_manifest_20260907.json")
    parser.add_argument("--remote-run-namespace", default="pair-a-continuation-20260907")
    parser.add_argument(
        "--amendment-record", type=pathlib.Path, default=None,
        help="JSON record for a continuation that is not a hardware move (see continuation.validate_amendment_record)",
    )
    args = parser.parse_args(argv)
    amendment_record = None
    if args.amendment_record is not None:
        amendment_path = args.amendment_record.resolve(strict=True)
        amendment_record = json.loads(amendment_path.read_text())
        validate_amendment_record(amendment_record)

    calibration_path = args.calibration_result.resolve(strict=True)
    calibration = json.loads(calibration_path.read_text())
    cache_provenance_path = args.registered_cache_runtime_provenance.resolve(strict=True)
    cache_provenance = json.loads(cache_provenance_path.read_text())
    cache_provenance["record_path"] = str(cache_provenance_path)
    cache_provenance["record_sha256"] = sha256_file(cache_provenance_path)
    manifest_relative = pathlib.PurePosixPath("provenance") / args.manifest_name
    extras = (
        "study/contracts_2026/__init__.py",
        "study/contracts_2026/agent_request.py",
        "study/contracts_2026/calibration.py",
        "study/contracts_2026/continuation.py",
        "study/contracts_2026/crossover.py",
        "study/contracts_2026/protected_runtime.py",
        "study/contracts_2026/provenance/__init__.py",
        "study/contracts_2026/runner.py",
        "study/contracts_2026/scientist_runtime.Dockerfile",
        "study/contracts_2026/tools/prepare_campaign_continuation.py",
    )
    manifest = build_continuation_manifest(
        prior_manifest_path=args.prior_manifest,
        source_state_root=args.source_state,
        completed_positions=args.completed_positions,
        repository_root=ROOT,
        new_gpu_uuid=args.new_gpu_uuid,
        new_ssh_host=args.new_a40_host,
        new_local_state_root=str(args.destination_state.resolve()),
        new_remote_state_root=args.new_remote_state,
        registered_cache=args.registered_cache,
        remote_python=args.remote_python,
        study_venv=args.study_venv,
        paired_baseline_checkpoints={
            "T": {"path": args.paired_baseline_t_checkpoint, "sha256": args.paired_baseline_t_sha256},
            "N": {"path": args.paired_baseline_n_checkpoint, "sha256": args.paired_baseline_n_sha256},
        },
        calibration_result=calibration,
        calibration_path=str(calibration_path),
        calibration_sha256=sha256_file(calibration_path),
        registered_cache_runtime_provenance=cache_provenance,
        extra_immutable_paths=extras,
        amendment_record=amendment_record,
    )
    manifest["continuation"]["preparation_tool"] = {
        "path": "study/contracts_2026/tools/prepare_campaign_continuation.py",
        "sha256": sha256_file(pathlib.Path(__file__).resolve()),
    }
    if amendment_record is not None:
        manifest["continuation"]["amendment_record_file"] = {
            "path": str(amendment_path), "sha256": sha256_file(amendment_path),
        }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary_manifest = pathlib.Path(handle.name)
    try:
        prepared = prepare_continuation_state(
            source_root=args.source_state,
            destination_root=args.destination_state,
            manifest_source=temporary_manifest,
            manifest_relative_path=manifest_relative,
            completed_positions=args.completed_positions,
            new_gpu_uuid=args.new_gpu_uuid,
            new_remote_state_root=args.new_remote_state,
            paired_baseline_t_checkpoint=args.paired_baseline_t_checkpoint,
            paired_baseline_n_checkpoint=args.paired_baseline_n_checkpoint,
            evaluator_hashes=campaign_controller._static_hashes(),
            remote_run_namespace=args.remote_run_namespace,
            abandoned_invocation=(amendment_record or {}).get("abandoned_invocation"),
        )
    finally:
        temporary_manifest.unlink(missing_ok=True)
    print(json.dumps(prepared, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
