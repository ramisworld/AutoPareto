#!/usr/bin/env python3
"""Protected, quality-agnostic cached inference benchmark for all candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types
from typing import Any

import torch

from benchmark_config import (
    CORRECTNESS_DECODE_STEPS,
    CORRECTNESS_PREFIX_TOKENS,
    FIXED_INPUT_TEXT,
    GENERATED_TOKENS,
    GENERATION_MODE,
    INPUT_TOKENS,
    LEGACY_INFERENCE_METRIC_MODE,
    LOGIT_CORRECTNESS_MEAN_ATOL,
    MEASURED_INFERENCE_REPETITIONS,
    OFFICIAL_INFERENCE_METRIC_MODE,
    PRECISION,
    WARMUP_ITERATIONS,
)
from event_log import append_event, make_event
from prepare import Tokenizer
from protected.inference_adapter import (
    build_adapter,
    check_cached_correctness,
    load_checkpoint_file,
    load_checkpoint_model,
    validated_cache_nbytes,
)

SETUP_MARKER = "# Setup: tokenizer, model, optimizer, dataloader"


def load_candidate_definitions(train_path: Path):
    source = train_path.read_text()
    if SETUP_MARKER not in source:
        raise RuntimeError(f"candidate train.py removed required marker: {SETUP_MARKER}")
    definitions = source.split(SETUP_MARKER, 1)[0]
    module_name = "autopareto_candidate_train"
    module = types.ModuleType(module_name)
    module.__file__ = str(train_path)
    sys.modules[module_name] = module
    exec(compile(definitions, str(train_path), "exec"), module.__dict__)
    return module


def fixed_input(
    tokenizer: Tokenizer,
    device: torch.device,
    input_tokens: int = INPUT_TOKENS,
) -> torch.Tensor:
    seed_ids = tokenizer.encode(FIXED_INPUT_TEXT, prepend=tokenizer.get_bos_token_id())
    if not seed_ids:
        raise RuntimeError("protected benchmark text produced no tokens")
    token_ids = (seed_ids * ((input_tokens + len(seed_ids) - 1) // len(seed_ids)))[:input_tokens]
    return torch.tensor([token_ids], dtype=torch.long, device=device)


@torch.inference_mode()
def generate_uncached(model, prompt: torch.Tensor, count: int, capture_logits: bool = False):
    tokens = prompt
    next_logits: list[torch.Tensor] = []
    for _ in range(count):
        logits = model(tokens)[:, -1, :]
        if capture_logits:
            next_logits.append(logits.detach().float().cpu())
        next_token = logits.argmax(dim=-1, keepdim=True)
        tokens = torch.cat((tokens, next_token), dim=1)
    return tokens, next_logits


@torch.inference_mode()
def generate_cached(adapter, prompt: torch.Tensor, count: int, capture_logits: bool = False):
    logits, state = adapter.prefill(prompt)
    step_logits = logits[:, -1, :]
    captured: list[torch.Tensor] = []
    generated: list[torch.Tensor] = []
    for index in range(count):
        if capture_logits:
            captured.append(step_logits.detach().float().cpu())
        token = step_logits.argmax(dim=-1, keepdim=True)
        generated.append(token)
        if index + 1 < count:
            logits, state = adapter.decode_one(token, state)
            step_logits = logits[:, -1, :]
    return torch.cat((prompt, torch.cat(generated, dim=1)), dim=1), captured, state


def correctness_check(model, adapter, prompt: torch.Tensor) -> dict[str, Any]:
    return check_cached_correctness(
        model, adapter, prompt, CORRECTNESS_DECODE_STEPS,
        LOGIT_CORRECTNESS_MEAN_ATOL,
    )


def correctness_suite(
    model,
    adapter,
    tokenizer: Tokenizer,
    device: torch.device,
) -> dict[str, Any]:
    """Check the candidate cache on multiple protected prefix lengths."""
    checks = []
    for prefix_tokens in CORRECTNESS_PREFIX_TOKENS:
        result = correctness_check(
            model, adapter, fixed_input(tokenizer, device, prefix_tokens)
        )
        checks.append({"prefix_tokens": prefix_tokens, **result})
    return {
        "passed": True,
        "greedy_tokens_match": all(row["greedy_tokens_match"] for row in checks),
        "logits_match": all(row["logits_match"] for row in checks),
        "max_logit_absolute_error": max(row["max_logit_absolute_error"] for row in checks),
        "max_mean_logit_absolute_error": max(
            row["max_mean_logit_absolute_error"] for row in checks
        ),
        "mean_atol": LOGIT_CORRECTNESS_MEAN_ATOL,
        "checked_decode_steps": CORRECTNESS_DECODE_STEPS,
        "prefix_mode": "teacher_forced_multiple_common_prefixes",
        "prefix_tokens": list(CORRECTNESS_PREFIX_TOKENS),
        "checks": checks,
    }


def synchronize() -> None:
    torch.cuda.synchronize()


def one_measured_repetition(adapter, prompt: torch.Tensor, repetition: int) -> tuple[dict[str, Any], torch.Tensor]:
    torch.cuda.reset_peak_memory_stats()
    synchronize()
    prefill_started = time.perf_counter()
    logits, state = adapter.prefill(prompt)
    synchronize()
    prefill_seconds = time.perf_counter() - prefill_started
    prefill_cache_bytes = validated_cache_nbytes(adapter, state)

    first = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [first]
    step_token = first
    # The first generated token is produced by prefill. Standard decode throughput
    # therefore measures the remaining 255 one-token cached decode operations.
    synchronize()
    decode_started = time.perf_counter()
    for _ in range(GENERATED_TOKENS - 1):
        logits, state = adapter.decode_one(step_token, state)
        step_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(step_token)
    synchronize()
    decode_seconds = time.perf_counter() - decode_started
    final_cache_bytes = validated_cache_nbytes(adapter, state)
    output = torch.cat((prompt, torch.cat(generated, dim=1)), dim=1)
    measured_decode_tokens = GENERATED_TOKENS - 1
    result = {
        "repetition_index": repetition,
        "metric_mode": OFFICIAL_INFERENCE_METRIC_MODE,
        "prefill_milliseconds": prefill_seconds * 1000,
        "prefill_tokens_per_second": INPUT_TOKENS / prefill_seconds,
        "decode_seconds": decode_seconds,
        "decode_measured_tokens": measured_decode_tokens,
        "decode_tokens_per_second": measured_decode_tokens / decode_seconds,
        "end_to_end_tokens_per_second": GENERATED_TOKENS / (prefill_seconds + decode_seconds),
        "peak_allocated_cuda_memory_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
        "peak_reserved_cuda_memory_mb": torch.cuda.max_memory_reserved() / 1024 / 1024,
        "prefill_kv_cache_memory_mb": prefill_cache_bytes / 1024 / 1024,
        "incremental_kv_cache_memory_mb": (final_cache_bytes - prefill_cache_bytes) / 1024 / 1024,
        "final_kv_cache_memory_mb": final_cache_bytes / 1024 / 1024,
    }
    return result, output


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--event-log", type=Path)
    parser.add_argument("--campaign-id")
    parser.add_argument("--experiment-id", type=int)
    parser.add_argument("--training-run-id", type=int)
    parser.add_argument("--legacy-diagnostic", action="store_true")
    args = parser.parse_args()

    if PRECISION != "bfloat16" or GENERATION_MODE != "greedy":
        raise RuntimeError("protected inference configuration changed unexpectedly")
    device = torch.device("cuda")
    module = load_candidate_definitions(args.train)
    checkpoint = load_checkpoint_file(args.checkpoint)
    model = load_checkpoint_model(module, checkpoint, device)
    adapter = build_adapter(module, model)
    tokenizer = Tokenizer.from_directory()
    prompt = fixed_input(tokenizer, device)
    prompt_hash = hashlib.sha256(prompt.detach().cpu().numpy().tobytes()).hexdigest()
    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    warmup_repetitions = []
    with autocast:
        correctness = correctness_suite(model, adapter, tokenizer, device)
        for warmup_index in range(WARMUP_ITERATIONS):
            generate_cached(adapter, prompt, GENERATED_TOKENS)
            warmup = {
                "repetition_index": warmup_index,
                "is_warmup": True,
                "metric_mode": OFFICIAL_INFERENCE_METRIC_MODE,
                "generated_tokens": GENERATED_TOKENS,
                "completed": True,
            }
            warmup_repetitions.append(warmup)
            if args.event_log:
                append_event(args.event_log, make_event(
                    "inference_warmup_completed", campaign_id=args.campaign_id,
                    experiment_id=args.experiment_id, training_run_id=args.training_run_id,
                    status="running", payload=warmup,
                ))
    synchronize()

    repetitions: list[dict[str, Any]] = []
    final_output = None
    with autocast:
        for index in range(MEASURED_INFERENCE_REPETITIONS):
            repetition, final_output = one_measured_repetition(adapter, prompt, index)
            repetition["correctness_passed"] = True
            repetition["max_logit_absolute_error"] = correctness["max_logit_absolute_error"]
            repetitions.append(repetition)
            if args.event_log:
                append_event(args.event_log, make_event(
                    "inference_repetition_completed",
                    campaign_id=args.campaign_id,
                    experiment_id=args.experiment_id,
                    training_run_id=args.training_run_id,
                    status="running",
                    payload=repetition,
                ))

    if final_output is None or final_output[:, INPUT_TOKENS:].shape[1] != GENERATED_TOKENS:
        raise RuntimeError("benchmark did not generate the protected token count")

    decode_speeds = [row["decode_tokens_per_second"] for row in repetitions]
    prefill_ms = [row["prefill_milliseconds"] for row in repetitions]
    median_decode = statistics.median(decode_speeds)
    speed_variation = statistics.stdev(decode_speeds)
    generated = final_output[:, INPUT_TOKENS:]
    args.sample.parent.mkdir(parents=True, exist_ok=True)
    args.sample.write_text(tokenizer.decode(generated[0].tolist()))

    legacy = None
    if args.legacy_diagnostic:
        torch.cuda.reset_peak_memory_stats()
        synchronize()
        started = time.perf_counter()
        with autocast:
            legacy_output, _ = generate_uncached(model, prompt, GENERATED_TOKENS)
        synchronize()
        elapsed = time.perf_counter() - started
        if not torch.equal(final_output, legacy_output):
            raise RuntimeError("legacy diagnostic tokens differ from cached output")
        legacy = {
            "metric_mode": LEGACY_INFERENCE_METRIC_MODE,
            "tokens_per_second": GENERATED_TOKENS / elapsed,
            "elapsed_seconds": elapsed,
            "peak_allocated_cuda_memory_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
        }

    evaluator_path = Path(__file__).resolve()
    metrics = {
        "inference_metric_mode": OFFICIAL_INFERENCE_METRIC_MODE,
        "median_cached_decode_tokens_per_second": median_decode,
        "cached_decode_variation_tokens_per_second": speed_variation,
        "cached_decode_min_tokens_per_second": min(decode_speeds),
        "cached_decode_max_tokens_per_second": max(decode_speeds),
        "prefill_milliseconds": statistics.median(prefill_ms),
        "prefill_tokens_per_second": statistics.median(row["prefill_tokens_per_second"] for row in repetitions),
        "peak_inference_cuda_memory_mb": max(row["peak_allocated_cuda_memory_mb"] for row in repetitions),
        "peak_reserved_inference_cuda_memory_mb": max(row["peak_reserved_cuda_memory_mb"] for row in repetitions),
        "incremental_kv_cache_memory_mb": max(row["incremental_kv_cache_memory_mb"] for row in repetitions),
        "inference_seconds": statistics.median(row["decode_seconds"] for row in repetitions),
        "inference_repetitions": repetitions,
        "warmup_repetitions": warmup_repetitions,
        "correctness": correctness,
        "legacy_diagnostic": legacy,
        "input_tokens": INPUT_TOKENS,
        "generated_tokens": GENERATED_TOKENS,
        "prompt_sha256": prompt_hash,
        "generation_mode": GENERATION_MODE,
        "precision": PRECISION,
        "warmup_iterations": WARMUP_ITERATIONS,
        "measured_repetitions": MEASURED_INFERENCE_REPETITIONS,
        "evaluator_git_commit": git_commit(evaluator_path.parent),
        "evaluator_sha256": file_sha256(evaluator_path),
        "sample": str(args.sample),
    }
    # Compatibility alias: future rows store the official cached median here while
    # legacy rows remain explicitly marked uncached_legacy.
    metrics["inference_tokens_per_second"] = median_decode
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
