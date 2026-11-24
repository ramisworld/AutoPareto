"""Diagnostic-only four-sequence timing test on registered frozen workloads."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time
from typing import Any

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from study.contracts_2026.benchmark_policy import DEFAULT_STABILITY, accept_measurement, adaptive_warmup
from study.contracts_2026.inference_benchmark import (
    WORKLOAD,
    _fixed_input,
    _inference_autocast,
    _load_candidate_definitions,
    correctness_gate,
    timed_aggregated_cached_decode,
)


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except OSError as exc:
        return {"command": command, "returncode": None, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}"}


def telemetry() -> dict[str, Any]:
    return {
        "captured_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "gpu": command([
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,temperature.gpu,power.draw,clocks.sm,clocks.mem,pstate,utilization.gpu,utilization.memory,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]),
        "compute_processes": command([
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]),
        "process": command(["ps", "-o", "pid=,stat=,etime=,cmd=", "-p", str(os.getpid())]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    context = args.context.resolve()
    checkpoint_path = args.checkpoint.resolve()
    module, candidate_path = _load_candidate_definitions(context, "train.py")
    from protected.inference_adapter import build_adapter, load_checkpoint_file, load_checkpoint_model

    checkpoint = load_checkpoint_file(checkpoint_path)
    device = torch.device("cuda")
    model = load_checkpoint_model(module, checkpoint, device)
    adapter = build_adapter(module, model)
    fp32_model = load_checkpoint_model(module, checkpoint, device).float()
    fp32_adapter = build_adapter(module, fp32_model)
    tokenizer = module.Tokenizer.from_directory()
    input_ids = _fixed_input(tokenizer, torch, device)
    process_start = {"pid": os.getpid(), "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "telemetry": telemetry()}
    with torch.inference_mode(), _inference_autocast(torch):
        correctness = correctness_gate(
            adapter,
            lambda ids: model(ids),
            input_ids,
            torch,
            model=model,
            fp32_model=fp32_model,
            fp32_adapter=fp32_adapter,
        )

    def timed():
        before = telemetry()
        row = timed_aggregated_cached_decode(
            adapter,
            input_ids,
            torch,
            WORKLOAD["generated_tokens"],
            WORKLOAD["independent_sequences_per_repetition"],
        )
        row["telemetry_before"] = before
        row["telemetry_after"] = telemetry()
        row["wall_tokens_per_second"] = row["cached_decode_tokens_per_second"]
        row["cuda_event_tokens_per_second"] = row["decoded_tokens"] / row["cuda_event_elapsed_seconds"]
        return row

    warmup_blocks = []
    warmup_details = []
    decision = None
    for _ in range(DEFAULT_STABILITY.max_warmup_blocks):
        details = [timed() for _ in range(DEFAULT_STABILITY.block_size)]
        warmup_details.append(details)
        warmup_blocks.append([row["wall_tokens_per_second"] for row in details])
        decision = adaptive_warmup(warmup_blocks, DEFAULT_STABILITY)
        if decision["stable"]:
            break
    measured_details = []
    measured_values = []
    if decision and decision["stable"]:
        for _ in range(DEFAULT_STABILITY.measured_passes):
            details = [timed() for _ in range(DEFAULT_STABILITY.repetitions_per_pass)]
            measured_details.append(details)
            measured_values.append([row["wall_tokens_per_second"] for row in details])
    measured_decision = accept_measurement(
        correctness_passed=bool(correctness["passed"]),
        warmup_blocks=warmup_blocks,
        measured_passes=measured_values,
        config=DEFAULT_STABILITY,
    ) if measured_values else None
    all_rows = [row for group in warmup_details + measured_details for row in group]
    return {
        "schema": "autopareto_contracts_2026_aggregate_timing_diagnostic_v1",
        "status": "stable" if measured_decision and measured_decision["stable"] else "measurement-unstable",
        "session_id": args.session_id,
        "model_id": args.model_id,
        "order_index": args.order_index,
        "synthetic_prompt": False,
        "prompt_source": "registered_tokenizer_fixed_input_text",
        "candidate_source_sha256": sha256_file(candidate_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "adapter_type": type(adapter).__name__,
        "correctness": correctness,
        "warmup_blocks": warmup_details,
        "warmup_decision": decision,
        "measured_passes": measured_details,
        "measured_decision": measured_decision,
        "allocated_cuda_memory_bytes": max(row["allocated_cuda_memory_bytes"] for row in all_rows),
        "reserved_cuda_memory_bytes": max(row["reserved_cuda_memory_bytes"] for row in all_rows),
        "kv_cache_bytes": max(row["kv_cache_bytes"] for row in all_rows),
        "process_start_state": process_start,
        "telemetry_at_finish": telemetry(),
        "workload": WORKLOAD,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--order-index", type=int, required=True)
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = run(args)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
        print(json.dumps({"model_id": args.model_id, "output": str(args.output), "status": payload["status"]}, sort_keys=True))
        return 0
    except BaseException as exc:
        payload = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
