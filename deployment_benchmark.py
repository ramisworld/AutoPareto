#!/usr/bin/env python3
"""Supporting cached-inference workloads that never replace headline metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any

import torch

from benchmark_config import FIXED_INPUT_TEXT
from benchmark_stability import (
    STABILIZED_PROTOCOL_VERSION,
    StabilityConfig,
    StabilityError,
    adaptive_warmup,
    compare_blocks,
    summarize_block,
    summarize_measurement_pass,
)
from inference_benchmark import (
    file_sha256,
    generate_cached,
    git_commit,
    load_candidate_definitions,
    precision_aware_correctness,
    synchronize,
)
from prepare import Tokenizer
from protected.inference_adapter import (
    build_adapter,
    load_checkpoint_file,
    load_checkpoint_model,
    validated_cache_nbytes,
)


def prompt(tokenizer: Tokenizer, count: int, batch_size: int, device: torch.device) -> torch.Tensor:
    seed = tokenizer.encode(FIXED_INPUT_TEXT, prepend=tokenizer.get_bos_token_id())
    ids = (seed * ((count + len(seed) - 1) // len(seed)))[:count]
    return torch.tensor([ids], dtype=torch.long, device=device).expand(batch_size, -1).contiguous()


def atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def gpu_telemetry() -> dict[str, Any]:
    """Capture GPU state outside the timed region without hiding unavailable fields."""
    fields = (
        "name,uuid,driver_version,pstate,clocks.current.sm,clocks.current.memory,"
        "temperature.gpu,power.draw,power.limit,memory.used,memory.total"
    )
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            text=True,
            capture_output=True,
            check=True,
            timeout=10,
        )
        rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if len(rows) != 1:
            raise RuntimeError(f"expected one GPU telemetry row, found {len(rows)}")
        values = [value.strip() for value in rows[0].split(",")]
        names = fields.split(",")
        if len(values) != len(names):
            raise RuntimeError("nvidia-smi telemetry column mismatch")
        return {"status": "available", **dict(zip(names, values))}
    except Exception as exc:
        return {
            "status": "unavailable",
            "failure_reason": f"{type(exc).__name__}: {exc}",
        }


def cuda_events() -> tuple[torch.cuda.Event, torch.cuda.Event]:
    return (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )


@torch.inference_mode()
def measure(
    adapter,
    input_ids: torch.Tensor,
    generated_tokens: int,
    repetition: int,
    *,
    measurement_pass: int = 0,
) -> dict:
    torch.cuda.reset_peak_memory_stats()
    prefill_start_event, prefill_stop_event = cuda_events()
    synchronize()
    started = time.perf_counter()
    prefill_start_event.record()
    logits, state = adapter.prefill(input_ids)
    prefill_stop_event.record()
    synchronize()
    prefill_seconds = time.perf_counter() - started
    prefill_cuda_seconds = prefill_start_event.elapsed_time(prefill_stop_event) / 1000
    initial_cache = validated_cache_nbytes(adapter, state)
    token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    decode_start_event, decode_stop_event = cuda_events()
    synchronize()
    started = time.perf_counter()
    decode_start_event.record()
    for _ in range(generated_tokens - 1):
        logits, state = adapter.decode_one(token, state)
        token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    decode_stop_event.record()
    synchronize()
    decode_seconds = time.perf_counter() - started
    decode_cuda_seconds = decode_start_event.elapsed_time(decode_stop_event) / 1000
    final_cache = validated_cache_nbytes(adapter, state)
    batch = input_ids.shape[0]
    measured_tokens = batch * (generated_tokens - 1)
    return {
        "repetition_index": repetition,
        "measurement_pass": measurement_pass,
        "metric_mode": "cached_decode_workload",
        "timing_source": "synchronized_wall_clock_headline_with_cuda_event_diagnostic",
        "prefill_milliseconds": prefill_seconds * 1000,
        "prefill_cuda_milliseconds": prefill_cuda_seconds * 1000,
        "prefill_tokens_per_second": input_ids.numel() / prefill_seconds,
        "decode_seconds": decode_seconds,
        "decode_cuda_seconds": decode_cuda_seconds,
        "decode_tokens_per_second": measured_tokens / decode_seconds,
        "decode_cuda_tokens_per_second": measured_tokens / decode_cuda_seconds,
        "peak_allocated_cuda_memory_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
        "peak_reserved_cuda_memory_mb": torch.cuda.max_memory_reserved() / 1024 / 1024,
        "prefill_kv_cache_memory_mb": initial_cache / 1024 / 1024,
        "incremental_kv_cache_memory_mb": (final_cache - initial_cache) / 1024 / 1024,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--input-tokens", required=True, type=int)
    parser.add_argument("--generated-tokens", type=int, default=128)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--protocol-version",
        choices=("legacy_fixed", STABILIZED_PROTOCOL_VERSION),
        default="legacy_fixed",
    )
    parser.add_argument("--measurement-passes", type=int, default=1)
    parser.add_argument("--warmup-block-size", type=int, default=5)
    parser.add_argument("--max-warmups", type=int, default=30)
    parser.add_argument("--stability-median-tolerance", type=float, default=0.02)
    parser.add_argument("--stability-cv-limit", type=float, default=0.05)
    args = parser.parse_args()
    if args.input_tokens < 1 or args.generated_tokens < 2 or args.batch_size < 1:
        raise ValueError("workload sizes must be positive and generated tokens must be at least two")
    if args.repetitions < 2 or args.measurement_passes < 1:
        raise ValueError("measurement passes must be positive and each pass needs two repetitions")
    if (
        args.protocol_version == STABILIZED_PROTOCOL_VERSION
        and args.repetitions != 2 * args.warmup_block_size
    ):
        raise ValueError(
            "stabilized measurement requires two measured blocks per pass"
        )

    device = torch.device("cuda")
    module = load_candidate_definitions(args.train)
    checkpoint = load_checkpoint_file(args.checkpoint)
    model = load_checkpoint_model(module, checkpoint, device)
    if args.input_tokens + args.generated_tokens > model.config.sequence_len:
        raise ValueError(
            "input plus generated tokens exceed the candidate sequence length"
        )
    adapter = build_adapter(module, model)
    tokenizer = Tokenizer.from_directory()
    input_ids = prompt(tokenizer, args.input_tokens, args.batch_size, device)
    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    correctness = precision_aware_correctness(
        module,
        checkpoint,
        model,
        adapter,
        [(args.input_tokens, input_ids)],
        device,
        steps=min(16, args.generated_tokens),
    )
    evaluator = Path(__file__).resolve()
    telemetry: list[dict[str, Any]] = []
    warmups: list[dict[str, Any]] = []
    repetitions: list[dict[str, Any]] = []
    stability_passes: list[dict[str, Any]] = []
    measurement_summaries: list[dict[str, Any]] = []
    status = "completed"
    failure_reason = None

    if args.protocol_version == STABILIZED_PROTOCOL_VERSION:
        stability_config = StabilityConfig(
            block_size=args.warmup_block_size,
            max_warmups=args.max_warmups,
            median_relative_tolerance=args.stability_median_tolerance,
            coefficient_of_variation_limit=args.stability_cv_limit,
        )
        try:
            with autocast:
                for pass_index in range(args.measurement_passes):
                    telemetry.append({
                        "measurement_pass": pass_index,
                        "position": "before_stabilization",
                        **gpu_telemetry(),
                    })

                    def warmup_once(index: int, _block: int) -> dict[str, Any]:
                        return measure(
                            adapter,
                            input_ids,
                            args.generated_tokens,
                            index,
                            measurement_pass=pass_index,
                        )

                    stability = adaptive_warmup(warmup_once, stability_config)
                    warmup_offset = len(warmups)
                    for row in stability["warmup_repetitions"]:
                        row["repetition_index"] = warmup_offset + int(row["warmup_index"])
                        row["measurement_pass"] = pass_index
                    warmups.extend(stability["warmup_repetitions"])
                    stability_passes.append({
                        key: value
                        for key, value in stability.items()
                        if key != "warmup_repetitions"
                    } | {"measurement_pass": pass_index})
                    telemetry.append({
                        "measurement_pass": pass_index,
                        "position": "after_stabilization",
                        **gpu_telemetry(),
                    })
                    pass_rows = []
                    for local_index in range(args.repetitions):
                        global_index = pass_index * args.repetitions + local_index
                        row = measure(
                            adapter,
                            input_ids,
                            args.generated_tokens,
                            global_index,
                            measurement_pass=pass_index,
                        )
                        row["is_warmup"] = False
                        pass_rows.append(row)
                        repetitions.append(row)
                    measurement_summaries.append(
                        summarize_measurement_pass(pass_rows, pass_index)
                    )
                    first_block = summarize_block(
                        pass_rows[: stability_config.block_size], 0
                    )
                    second_block = summarize_block(
                        pass_rows[stability_config.block_size :], 1
                    )
                    measured_stability = compare_blocks(
                        first_block, second_block, stability_config
                    )
                    measurement_summaries[-1]["first_block"] = first_block
                    measurement_summaries[-1]["second_block"] = second_block
                    measurement_summaries[-1]["stability"] = measured_stability
                    telemetry.append({
                        "measurement_pass": pass_index,
                        "position": "after_measurement",
                        **gpu_telemetry(),
                    })
                    if not measured_stability["stable"]:
                        status = "unstable_environment"
                        failure_reason = (
                            f"measured throughput drifted during pass {pass_index}"
                        )
                        break
        except StabilityError as exc:
            status = "unstable_environment"
            failure_reason = str(exc)
            partial = exc.result
            warmup_offset = len(warmups)
            pass_index = len(stability_passes)
            for row in partial["warmup_repetitions"]:
                row["repetition_index"] = warmup_offset + int(row["warmup_index"])
                row["measurement_pass"] = pass_index
            warmups.extend(partial["warmup_repetitions"])
            stability_passes.append({
                key: value for key, value in partial.items() if key != "warmup_repetitions"
            } | {"measurement_pass": pass_index})
    else:
        with autocast:
            for index in range(args.warmups):
                generate_cached(adapter, input_ids, args.generated_tokens)
                warmups.append({
                    "repetition_index": index,
                    "measurement_pass": 0,
                    "is_warmup": True,
                    "completed": True,
                })
            synchronize()
            repetitions = [
                measure(adapter, input_ids, args.generated_tokens, index)
                for index in range(args.repetitions)
            ]
            for row in repetitions:
                row["is_warmup"] = False
            measurement_summaries = [summarize_measurement_pass(repetitions, 0)]

    decode_speeds = [row["decode_tokens_per_second"] for row in repetitions]
    metrics = {
        "status": status,
        "failure_reason": failure_reason,
        "protocol_version": args.protocol_version,
        "workload_name": f"context{args.input_tokens}-batch{args.batch_size}-decode{args.generated_tokens}",
        "input_tokens": args.input_tokens,
        "generated_tokens": args.generated_tokens,
        "inference_batch_size": args.batch_size,
        "warmup_repetitions": warmups,
        "inference_repetitions": repetitions,
        "stability_passes": stability_passes,
        "measurement_pass_summaries": measurement_summaries,
        "gpu_telemetry": telemetry,
        "correctness": correctness,
        "median_cached_decode_tokens_per_second": (
            statistics.median(decode_speeds) if decode_speeds else None
        ),
        "peak_inference_cuda_memory_mb": (
            max(row["peak_allocated_cuda_memory_mb"] for row in repetitions)
            if repetitions else None
        ),
        "prompt_sha256": hashlib.sha256(input_ids.cpu().numpy().tobytes()).hexdigest(),
        "evaluator_git_commit": git_commit(evaluator.parent),
        "evaluator_sha256": file_sha256(evaluator),
    }
    atomic_json_write(args.output, metrics)
    print(json.dumps(metrics, sort_keys=True))
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
