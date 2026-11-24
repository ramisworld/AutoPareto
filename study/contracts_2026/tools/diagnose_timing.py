"""Diagnostic-only fresh-process inference timing with A40 telemetry.

This module does not alter the frozen search benchmark.  It repeats the same
workload and stability policy while recording synchronized host timing, CUDA
event timing, raw warm-up/measurement repetitions, and GPU/process telemetry.
It is intentionally limited to pre-existing frozen checkpoints.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import statistics
import subprocess
import time
from typing import Any

import torch

from study.contracts_2026.benchmark_policy import (
    DEFAULT_STABILITY,
    PROTOCOL_VERSION,
    accept_measurement,
    adaptive_warmup,
)
from study.contracts_2026.inference_benchmark import (
    WORKLOAD,
    _fixed_input,
    _inference_autocast,
    _last_logits,
    _load_candidate_definitions,
    _validated_cache_nbytes,
)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _command(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except OSError as exc:
        return {"command": command, "returncode": None, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}"}


def _process_start_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "pid": os.getpid(),
        "monotonic_start_seconds": time.monotonic(),
    }
    stat_path = pathlib.Path(f"/proc/{os.getpid()}/stat")
    uptime_path = pathlib.Path("/proc/uptime")
    if stat_path.is_file():
        fields = stat_path.read_text().split()
        if len(fields) >= 22:
            state["proc_start_ticks"] = int(fields[21])
    if uptime_path.is_file():
        state["system_uptime_seconds"] = float(uptime_path.read_text().split()[0])
    return state


def telemetry() -> dict[str, Any]:
    gpu = _command([
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,temperature.gpu,power.draw,clocks.sm,clocks.mem,pstate,utilization.gpu,utilization.memory,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ])
    compute = _command([
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    return {
        "captured_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "gpu": gpu,
        "other_compute_processes": compute,
        "process": _command(["ps", "-o", "pid=,stat=,etime=,cmd=", "-p", str(os.getpid())]),
    }


def _timed_decode(adapter: Any, input_ids: torch.Tensor, generated_tokens: int) -> dict[str, Any]:
    torch.cuda.reset_peak_memory_stats()
    before = telemetry()
    with torch.inference_mode(), _inference_autocast(torch):
        _, state = adapter.prefill(input_ids)
        token = input_ids[:, -1:]
        torch.cuda.synchronize()
        event_start = torch.cuda.Event(enable_timing=True)
        event_end = torch.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        event_start.record()
        for _ in range(generated_tokens):
            logits, state = adapter.decode_one(token, state)
            token = torch.argmax(_last_logits(logits), dim=-1)
        event_end.record()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        event_milliseconds = float(event_start.elapsed_time(event_end))
    after = telemetry()
    if elapsed <= 0 or event_milliseconds <= 0:
        raise RuntimeError("non-positive synchronized timing")
    return {
        "wall_seconds": elapsed,
        "cuda_event_milliseconds": event_milliseconds,
        "wall_tokens_per_second": generated_tokens / elapsed,
        "cuda_event_tokens_per_second": generated_tokens / (event_milliseconds / 1000.0),
        "allocated_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "reserved_cuda_memory_bytes": int(torch.cuda.max_memory_reserved()),
        "kv_cache_bytes": int(_validated_cache_nbytes(adapter, state)),
        "telemetry_before": before,
        "telemetry_after": after,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    context = args.context.resolve()
    checkpoint_path = args.checkpoint.resolve()
    module, candidate_path = _load_candidate_definitions(context, "train.py")
    from protected.inference_adapter import build_adapter, check_cached_correctness, load_checkpoint_file, load_checkpoint_model

    checkpoint = load_checkpoint_file(checkpoint_path)
    model = load_checkpoint_model(module, checkpoint, torch.device("cuda"))
    adapter = build_adapter(module, model)
    tokenizer = module.Tokenizer.from_directory()
    input_ids = _fixed_input(tokenizer, torch, torch.device("cuda"))
    start_state = _process_start_state()
    try:
        with torch.inference_mode(), _inference_autocast(torch):
            correctness = check_cached_correctness(
                model,
                adapter,
                input_ids,
                steps=32,
                mean_atol=0.03,
                raise_on_failure=False,
            )
        if not correctness["passed"]:
            return {
                "schema": "autopareto_contracts_2026_timing_diagnostic_v1",
                "status": "correctness-rejected",
                "protocol_version": PROTOCOL_VERSION,
                "session_id": args.session_id,
                "order_index": args.order_index,
                "model_id": args.model_id,
                "candidate_source_sha256": sha256_file(candidate_path),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "adapter_type": type(adapter).__name__,
                "correctness": correctness,
                "process_start_state": start_state,
                "telemetry_at_rejection": telemetry(),
                "workload": WORKLOAD,
            }

        warmup_blocks: list[list[float]] = []
        warmup_details: list[list[dict[str, Any]]] = []
        warmup_decision: dict[str, Any] | None = None
        for block_index in range(DEFAULT_STABILITY.max_warmup_blocks):
            details = [
                _timed_decode(adapter, input_ids, WORKLOAD["generated_tokens"])
                for _ in range(DEFAULT_STABILITY.block_size)
            ]
            warmup_details.append(details)
            warmup_blocks.append([row["wall_tokens_per_second"] for row in details])
            warmup_decision = adaptive_warmup(warmup_blocks, DEFAULT_STABILITY)
            if warmup_decision["stable"]:
                break

        if not warmup_decision or not warmup_decision["stable"]:
            all_details = [row for block in warmup_details for row in block]
            return {
                "schema": "autopareto_contracts_2026_timing_diagnostic_v1",
                "status": "measurement-unstable",
                "protocol_version": PROTOCOL_VERSION,
                "session_id": args.session_id,
                "order_index": args.order_index,
                "model_id": args.model_id,
                "candidate_source_sha256": sha256_file(candidate_path),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "adapter_type": type(adapter).__name__,
                "correctness": correctness,
                "warmup_blocks": warmup_details,
                "warmup_decision": warmup_decision,
                "measured_passes": [],
                "measured_decision": None,
                "allocated_cuda_memory_bytes": max(row["allocated_cuda_memory_bytes"] for row in all_details),
                "reserved_cuda_memory_bytes": max(row["reserved_cuda_memory_bytes"] for row in all_details),
                "kv_cache_bytes": max(row["kv_cache_bytes"] for row in all_details),
                "process_start_state": start_state,
                "telemetry_at_finish": telemetry(),
                "workload": WORKLOAD,
            }

        measured_details: list[list[dict[str, Any]]] = []
        measured_values: list[list[float]] = []
        for _ in range(DEFAULT_STABILITY.measured_passes):
            details = [
                _timed_decode(adapter, input_ids, WORKLOAD["generated_tokens"])
                for _ in range(DEFAULT_STABILITY.repetitions_per_pass)
            ]
            measured_details.append(details)
            measured_values.append([row["wall_tokens_per_second"] for row in details])
        decision = accept_measurement(
            correctness_passed=True,
            warmup_blocks=warmup_blocks,
            measured_passes=measured_values,
            config=DEFAULT_STABILITY,
        )
        all_details = [row for block in warmup_details + measured_details for row in block]
        decision["cuda_event_median_tokens_per_second"] = statistics.median(
            row["cuda_event_tokens_per_second"] for row in all_details if row in [item for group in measured_details for item in group]
        ) if measured_details else None
        return {
            "schema": "autopareto_contracts_2026_timing_diagnostic_v1",
            "status": "stable" if decision["stable"] else "measurement-unstable",
            "protocol_version": PROTOCOL_VERSION,
            "session_id": args.session_id,
            "order_index": args.order_index,
            "model_id": args.model_id,
            "candidate_source_sha256": sha256_file(candidate_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "adapter_type": type(adapter).__name__,
            "correctness": correctness,
            "warmup_blocks": warmup_details,
            "warmup_decision": warmup_decision,
            "measured_passes": measured_details,
            "measured_decision": decision,
            "allocated_cuda_memory_bytes": max(row["allocated_cuda_memory_bytes"] for row in all_details),
            "reserved_cuda_memory_bytes": max(row["reserved_cuda_memory_bytes"] for row in all_details),
            "kv_cache_bytes": max(row["kv_cache_bytes"] for row in all_details),
            "process_start_state": start_state,
            "telemetry_at_finish": telemetry(),
            "workload": WORKLOAD,
        }
    finally:
        del adapter, model
        torch.cuda.empty_cache()


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
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"output": str(args.output), "status": payload["status"], "model_id": args.model_id}, sort_keys=True))
        return 0 if payload["status"] != "error" else 1
    except BaseException as exc:
        args.output.write_text(json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"}, indent=2) + "\n")
        print(json.dumps({"output": str(args.output), "status": "error", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
