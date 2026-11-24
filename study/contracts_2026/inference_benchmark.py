#!/usr/bin/env python3
"""Protected, quality-agnostic cached inference benchmark for all candidates."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types
import math
import random
from typing import Any

import torch

from .historical_reference.benchmark_config import (
    CORRECTNESS_DECODE_STEPS,
    CORRECTNESS_PREFIX_TOKENS,
    FIXED_INPUT_TEXT,
    FP32_LOGIT_CORRECTNESS_MAX_ATOL,
    FP32_LOGIT_CORRECTNESS_MEAN_ATOL,
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
try:
    from event_log import append_event, make_event
except ModuleNotFoundError:
    def append_event(*_args, **_kwargs):
        return None

    def make_event(event_type, **kwargs):
        return {"event_type": event_type, **kwargs}

from protected.inference_adapter import (
    CanonicalGPTAdapter,
    build_adapter,
    check_cached_correctness,
    load_checkpoint_file,
    load_checkpoint_model,
    validated_cache_nbytes,
)
from .benchmark_policy import PROTOCOL_VERSION, classify_measurement_stability

SETUP_MARKER = "# Setup: tokenizer, model, optimizer, dataloader"

# Compatibility surface for the study runner and readiness tests.  These values
# intentionally mirror the last known-good benchmark configuration.
ACTIVE_INFERENCE_CONFIG_PATH = Path(__file__).resolve().parent / "provenance" / "pair_a_active_inference_config_20260904.json"
ACTIVE_INFERENCE_CONFIG = json.loads(ACTIVE_INFERENCE_CONFIG_PATH.read_text())
_ACTIVE_PROMPT_CONFIG = ACTIVE_INFERENCE_CONFIG["protected_prompt_token_artifact"]
WORKLOAD = {
    "input_tokens": ACTIVE_INFERENCE_CONFIG["input_tokens"],
    "generated_tokens": ACTIVE_INFERENCE_CONFIG["generated_tokens"],
    "cached_decode_transitions": ACTIVE_INFERENCE_CONFIG["cached_decode_transitions"],
    "decoding": ACTIVE_INFERENCE_CONFIG["decoding"],
    "precision": ACTIVE_INFERENCE_CONFIG["precision"],
    "batch_size": ACTIVE_INFERENCE_CONFIG["batch_size"],
    "repetitions": ACTIVE_INFERENCE_CONFIG["measurement"]["windows"],
    "protected_prompt_count": ACTIVE_INFERENCE_CONFIG["prompt_count"],
}

FROZEN_PROMPT_TEXT_SET_SHA256 = None
FROZEN_PROMPT_SET_SHA256 = _ACTIVE_PROMPT_CONFIG["prompt_set_token_sha256"]
FROZEN_PROMPT_TOKEN_HASHES = tuple(_ACTIVE_PROMPT_CONFIG["prompt_token_hashes"])
_KERNEL_CACHE_MARKER = "_autopareto_original_get_kernel"

# Retained only for callers of the retired Design-B readiness fixtures.  The
# authoritative benchmark uses ``fixed_input`` and the historical tokenizer
# rather than these synthetic prompt IDs.
PROTECTED_BOS_TOKEN_ID = 8188
PROMPT_TOKEN_IDS = tuple(
    tuple([PROTECTED_BOS_TOKEN_ID] + [(index + prompt_index) % 8188 for index in range(1, 1024)])
    for prompt_index in range(8)
)
PROMPT_TOKEN_HASHES = tuple(
    hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    for ids in PROMPT_TOKEN_IDS
)


def validate_workload() -> None:
    if WORKLOAD["input_tokens"] != 1024 or WORKLOAD["generated_tokens"] != 128:
        raise RuntimeError("registered Design-B inference workload changed")
    if WORKLOAD["cached_decode_transitions"] != WORKLOAD["generated_tokens"] - 1:
        raise RuntimeError("historical cached transition count changed")
    if WORKLOAD["precision"] != "bfloat16" or WORKLOAD["decoding"] != "greedy":
        raise RuntimeError("registered inference precision or decoding changed")


validate_workload()


def _install_kernel_resolution_cache(kernels_module: Any | None = None) -> None:
    """Reuse the known-good kernel module within one protected process."""
    if kernels_module is None:
        import kernels as kernels_module
    if hasattr(kernels_module, _KERNEL_CACHE_MARKER):
        return
    original = kernels_module.get_kernel
    cache: dict[tuple[object, object, object], Any] = {}

    def cached_get_kernel(repo_id, revision=None, version=None, user_agent=None):
        key = (repo_id, revision, version)
        if key not in cache:
            cache[key] = original(
                repo_id,
                revision=revision,
                version=version,
                user_agent=user_agent,
            )
        return cache[key]

    setattr(kernels_module, _KERNEL_CACHE_MARKER, original)
    kernels_module.get_kernel = cached_get_kernel


def load_candidate_definitions(train_path: Path):
    source = train_path.read_text()
    if SETUP_MARKER not in source:
        raise RuntimeError(f"candidate train.py removed required marker: {SETUP_MARKER}")
    definitions = source.split(SETUP_MARKER, 1)[0]
    module_name = "autopareto_candidate_train"
    module = types.ModuleType(module_name)
    module.__file__ = str(train_path)
    sys.modules[module_name] = module
    _install_kernel_resolution_cache()
    candidate_root = str(train_path.resolve().parent)
    if candidate_root not in sys.path:
        sys.path.insert(0, candidate_root)
    exec(compile(definitions, str(train_path), "exec"), module.__dict__)
    return module


def _load_candidate_definitions(context_or_train: Path, candidate: str | None = None):
    """Compatibility entry point; inference is no longer model-host IPC."""
    if candidate is not None:
        raise RuntimeError("model-host IPC boundary is retired; use direct protected inference")
    return load_candidate_definitions(Path(context_or_train))


def _load_tokenizer(train_path: Path) -> Any:
    """Load the complete registered prepare.py dependency beside train.py."""
    candidate_root = str(train_path.resolve().parent)
    if candidate_root not in sys.path:
        sys.path.insert(0, candidate_root)
    prepare = importlib.import_module("prepare")
    registered_cache = Path(
        os.environ.get("AUTOPARETO_REGISTERED_CACHE", "/var/cache/autopareto/pair_a_registered_cache")
    ).resolve(strict=True)
    if not registered_cache.is_dir() or registered_cache.is_symlink():
        raise RuntimeError(f"registered cache is not a regular directory: {registered_cache}")
    tokenizer_dir = registered_cache / "tokenizer"
    tokenizer_path = tokenizer_dir / "tokenizer.pkl"
    if not tokenizer_path.is_file() or tokenizer_path.is_symlink():
        raise RuntimeError(f"registered tokenizer is missing or not a regular file: {tokenizer_path}")
    prepare.CACHE_DIR = str(registered_cache)
    prepare.DATA_DIR = str(registered_cache / "data")
    prepare.TOKENIZER_DIR = str(tokenizer_dir)
    return prepare.Tokenizer.from_directory(str(tokenizer_dir))


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


def protected_prompts(tokenizer: Any, device: torch.device) -> list[torch.Tensor]:
    """Return the registered eight prompts, failing closed if their source is absent."""
    source = Path(__file__).resolve().parent / "provenance" / Path(_ACTIVE_PROMPT_CONFIG["path"]).name
    if not source.is_file():
        raise RuntimeError(
            f"registered protected prompt token source is missing: {source}; refusing to substitute prompts"
        )
    if file_sha256(source) != _ACTIVE_PROMPT_CONFIG["sha256"]:
        raise RuntimeError("active protected prompt artifact file hash does not match its active configuration")
    payload = json.loads(source.read_text())
    if payload.get("status") != "ACTIVE_PAIR_A":
        raise RuntimeError("protected prompt artifact is not active for Pair A")
    rows = payload.get("prompt_token_ids")
    if not isinstance(rows, list) or len(rows) != 8:
        raise RuntimeError("registered prompt source must contain exactly eight prompt_token_ids")
    prompts = []
    hashes = []
    for row in rows:
        if not isinstance(row, list) or len(row) != 1024 or any(not isinstance(token, int) for token in row):
            raise RuntimeError("registered prompt source contains an invalid 1024-token prompt")
        hashes.append(hashlib.sha256(json.dumps(row, separators=(",", ":")).encode()).hexdigest())
        prompts.append(torch.tensor([row], dtype=torch.long, device=device))
    if tuple(hashes) != tuple(payload.get("prompt_token_hashes", [])):
        raise RuntimeError("registered prompt token source hash does not match the frozen prompt manifest")
    if hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest() != FROZEN_PROMPT_SET_SHA256:
        raise RuntimeError("protected prompt token set hash does not match the active configuration")
    return prompts


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


def correctness_check(
    model,
    adapter,
    prompt: torch.Tensor,
    *,
    mean_atol: float = LOGIT_CORRECTNESS_MEAN_ATOL,
    max_atol: float | None = None,
    raise_on_failure: bool = True,
) -> dict[str, Any]:
    return check_cached_correctness(
        model, adapter, prompt, CORRECTNESS_DECODE_STEPS,
        mean_atol, max_atol=max_atol, raise_on_failure=raise_on_failure,
    )


def correctness_suite_for_prompts(
    model,
    adapter,
    prompts: list[tuple[int, torch.Tensor]],
    *,
    steps: int = CORRECTNESS_DECODE_STEPS,
    mean_atol: float = LOGIT_CORRECTNESS_MEAN_ATOL,
    max_atol: float | None = None,
    raise_on_failure: bool = True,
) -> dict[str, Any]:
    """Measure cached correctness on one or more protected common prefixes."""
    checks = []
    for prefix_tokens, input_ids in prompts:
        result = check_cached_correctness(
            model, adapter, input_ids, steps, mean_atol,
            max_atol=max_atol, raise_on_failure=False,
        )
        checks.append({"prefix_tokens": prefix_tokens, **result})
    result = {
        "passed": all(row["passed"] for row in checks),
        "greedy_tokens_match": all(row["greedy_tokens_match"] for row in checks),
        "logits_match": all(row["logits_match"] for row in checks),
        "max_logit_absolute_error": max(row["max_logit_absolute_error"] for row in checks),
        "max_mean_logit_absolute_error": max(
            row["max_mean_logit_absolute_error"] for row in checks
        ),
        "mean_atol": mean_atol,
        "max_atol": max_atol,
        "checked_decode_steps": steps,
        "prefix_mode": "teacher_forced_multiple_common_prefixes",
        "prefix_tokens": [row[0] for row in prompts],
        "checks": checks,
    }
    if raise_on_failure and not result["passed"]:
        raise RuntimeError(
            "cached/full correctness failed: "
            f"greedy_tokens_match={result['greedy_tokens_match']} "
            f"mean_logits_match={result['logits_match']} "
            f"max_mean_abs_error={result['max_mean_logit_absolute_error']:.6g} "
            f"max_abs_error={result['max_logit_absolute_error']:.6g}"
        )
    return result


def correctness_suite(
    model,
    adapter,
    tokenizer: Tokenizer,
    device: torch.device,
    *,
    raise_on_failure: bool = True,
) -> dict[str, Any]:
    prompts = [
        (prefix_tokens, fixed_input(tokenizer, device, prefix_tokens))
        for prefix_tokens in CORRECTNESS_PREFIX_TOKENS
    ]
    return correctness_suite_for_prompts(
        model, adapter, prompts, raise_on_failure=raise_on_failure,
    )


_FLASH_PERFORMANCE_ONLY_OPTIONS = ("num_splits", "pack_gqa", "deterministic", "sm_margin")


def fp32_flash_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    """FlashAttention's ``flash_attn_func`` forward semantics, computed with SDPA in the input dtype.

    Layout (B, T, H, D) in and out; GQA by contiguous head groups; default scale 1/sqrt(D); the causal mask
    and the sliding window ``[i + Tk - Tq - left, i + Tk - Tq + right]`` (``-1`` = unbounded, and ``right``
    is 0 when causal) are aligned to the bottom-right corner; a query with no visible key returns zeros.
    """
    batch, q_len, heads, head_dim = q.shape
    k_len, kv_heads = k.shape[1], k.shape[2]
    if heads % kv_heads:
        raise RuntimeError("query heads must be a multiple of key/value heads")
    left, right = (int(window_size[0]), int(window_size[1]))
    if causal:
        right = 0
    rows = torch.arange(q_len, device=q.device)[:, None] + (k_len - q_len)
    cols = torch.arange(k_len, device=q.device)[None, :]
    allowed = torch.ones(q_len, k_len, dtype=torch.bool, device=q.device)
    if right >= 0:
        allowed &= cols <= rows + right
    if left >= 0:
        allowed &= cols >= rows - left
    q_t, k_t, v_t = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    if kv_heads != heads:
        k_t = k_t.repeat_interleave(heads // kv_heads, dim=1)
        v_t = v_t.repeat_interleave(heads // kv_heads, dim=1)
    scale = float(softmax_scale) if softmax_scale is not None else head_dim ** -0.5
    out = torch.nn.functional.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=allowed, scale=scale)
    out = out.masked_fill(~allowed.any(dim=-1)[None, None, :, None], 0.0)
    return out.transpose(1, 2)


class _FP32ReferenceFlashAttention:
    """A candidate's FlashAttention interface, with FP32 inputs computed by the exact SDPA stand-in.

    FlashAttention rejects FP32 inputs. Every other dtype and attribute goes to the original kernel interface
    unchanged; FP32 options that change the math and are not reproduced here are refused, never ignored.
    """

    def __init__(self, original: Any) -> None:
        self._original = original
        self.fp32_calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)

    def flash_attn_func(self, q, k, v, softmax_scale=None, causal=False, window_size=(-1, -1), **options):
        if q.dtype != torch.float32:
            return self._original.flash_attn_func(
                q, k, v, softmax_scale=softmax_scale, causal=causal, window_size=window_size, **options
            )
        unsupported = {
            name: value for name, value in options.items()
            if name not in _FLASH_PERFORMANCE_ONLY_OPTIONS and value not in (None, 0, 0.0, False)
        }
        if unsupported:
            raise RuntimeError(f"FP32 FlashAttention reference does not reproduce options {sorted(unsupported)}")
        self.fp32_calls += 1
        return fp32_flash_attention_reference(
            q, k, v, softmax_scale=softmax_scale, causal=causal, window_size=window_size
        )


@contextlib.contextmanager
def fp32_reference_attention(module: Any, adapter: Any):
    """During FP32 adjudication of a candidate-supplied adapter, let the candidate forward run in FP32.

    The reference for a candidate-supplied adapter is the candidate's own ``GPT.forward``. When that forward
    calls FlashAttention (the registered baseline does), FP32 inputs would be rejected. Each module global that
    is a FlashAttention interface, or its ``flash_attn_func``, is replaced by the exact stand-in for the
    duration of the adjudication and restored afterwards. The canonical adapter's reference is its own SDPA
    prefill and is untouched.
    """
    if isinstance(adapter, CanonicalGPTAdapter):
        yield {"mode": "protected_canonical_sdpa_prefill"}
        return
    namespace = vars(module)
    replacements: dict[str, Any] = {}
    shims: list[_FP32ReferenceFlashAttention] = []
    for name, value in list(namespace.items()):
        if isinstance(value, _FP32ReferenceFlashAttention):
            continue
        if callable(getattr(value, "flash_attn_func", None)) and not isinstance(value, type):
            shim = _FP32ReferenceFlashAttention(value)
            shims.append(shim)
            replacements[name] = shim
        elif callable(value) and getattr(value, "__name__", "") == "flash_attn_func":
            shim = _FP32ReferenceFlashAttention(types.SimpleNamespace(flash_attn_func=value))
            shims.append(shim)
            replacements[name] = shim.flash_attn_func
    originals = {name: namespace[name] for name in replacements}
    record = {
        "mode": (
            "candidate_forward_with_fp32_flash_attention_reference"
            if replacements else "candidate_forward_unchanged"
        ),
        "replaced_globals": sorted(replacements),
    }
    namespace.update(replacements)
    try:
        yield record
    finally:
        namespace.update(originals)
        record["fp32_flash_attention_calls"] = sum(shim.fp32_calls for shim in shims)


def precision_aware_correctness(
    module: Any,
    checkpoint: dict[str, Any],
    model,
    adapter,
    prompts: list[tuple[int, torch.Tensor]],
    device: torch.device,
    *,
    steps: int = CORRECTNESS_DECODE_STEPS,
) -> dict[str, Any]:
    """Use cheap BF16 validation and strict FP32 for every BF16 failure."""
    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    with autocast:
        bf16 = correctness_suite_for_prompts(
            model, adapter, prompts, steps=steps, raise_on_failure=False,
        )
    if not _requires_fp32_adjudication(bf16):
        return {
            **bf16,
            "verification_mode": "bf16_fast_path",
            "bf16": bf16,
            "fp32": None,
        }

    fp32_model = None
    fp32_adapter = None
    try:
        fp32_model = load_checkpoint_model(module, checkpoint, device).float()
        fp32_adapter = build_adapter(module, fp32_model)
        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]), \
                fp32_reference_attention(module, fp32_adapter) as reference_attention:
            fp32 = correctness_suite_for_prompts(
                fp32_model,
                fp32_adapter,
                prompts,
                steps=steps,
                mean_atol=FP32_LOGIT_CORRECTNESS_MEAN_ATOL,
                max_atol=FP32_LOGIT_CORRECTNESS_MAX_ATOL,
                raise_on_failure=False,
            )
        fp32 = {**fp32, "fp32_reference_attention": reference_attention}
    finally:
        del fp32_adapter
        del fp32_model
        torch.cuda.empty_cache()

    if not fp32["passed"]:
        raise RuntimeError(
            "cached/full correctness failed after float32 adjudication: "
            f"greedy_tokens_match={fp32['greedy_tokens_match']} "
            f"max_mean_abs_error={fp32['max_mean_logit_absolute_error']:.6g} "
            f"max_abs_error={fp32['max_logit_absolute_error']:.6g}"
        )
    return {
        **bf16,
        "passed": True,
        "logits_match": True,
        "greedy_tokens_match": True,
        "verification_mode": "fp32_adjudicated",
        "bf16_logit_tolerance_passed": False,
        "fp32_logit_tolerance_passed": True,
        "bf16": bf16,
        "fp32": fp32,
    }


def _requires_fp32_adjudication(bf16: dict[str, Any]) -> bool:
    """Adjudicate any BF16 semantic or numerical correctness failure in FP32."""
    return not (
        bf16["greedy_tokens_match"]
        and bf16["logits_match"]
        and bf16["passed"]
    )


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

    first_token_started = time.perf_counter()
    first = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    synchronize()
    first_token_seconds = time.perf_counter() - first_token_started
    generated = [first]
    step_token = first
    # The first generated token is produced by prefill. Design-B throughput
    # therefore measures the remaining 127 one-token cached decode operations.
    synchronize()
    decode_started = time.perf_counter()
    for _ in range(127):
        logits, state = adapter.decode_one(step_token, state)
        step_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(step_token)
    synchronize()
    decode_seconds = time.perf_counter() - decode_started
    final_cache_bytes = validated_cache_nbytes(adapter, state)
    output = torch.cat((prompt, torch.cat(generated, dim=1)), dim=1)
    measured_decode_tokens = 127
    result = {
        "repetition_index": repetition,
        "metric_mode": OFFICIAL_INFERENCE_METRIC_MODE,
        "prefill_milliseconds": prefill_seconds * 1000,
        "first_token_milliseconds": first_token_seconds * 1000,
        "prefill_tokens_per_second": 1024 / prefill_seconds,
        "decode_seconds": decode_seconds,
        "decode_measured_tokens": measured_decode_tokens,
        "decode_tokens_per_second": measured_decode_tokens / decode_seconds,
        "end_to_end_tokens_per_second": 128 / (prefill_seconds + decode_seconds),
        "peak_allocated_cuda_memory_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
        "peak_reserved_cuda_memory_mb": torch.cuda.max_memory_reserved() / 1024 / 1024,
        "prefill_kv_cache_memory_mb": prefill_cache_bytes / 1024 / 1024,
        "incremental_kv_cache_memory_mb": (final_cache_bytes - prefill_cache_bytes) / 1024 / 1024,
        "final_kv_cache_memory_mb": final_cache_bytes / 1024 / 1024,
    }
    return result, output


def _summarize_block(rows: list[dict[str, Any]], block_index: int) -> dict[str, Any]:
    speeds = [float(row["decode_tokens_per_second"]) for row in rows]
    if len(speeds) < 2 or any(not math.isfinite(value) or value <= 0 for value in speeds):
        raise RuntimeError("inference block contains insufficient/non-finite throughput")
    mean = statistics.fmean(speeds)
    cv = statistics.pstdev(speeds) / mean
    return {
        "block_index": block_index,
        "sample_count": len(speeds),
        "median_decode_tokens_per_second": statistics.median(speeds),
        "mean_decode_tokens_per_second": mean,
        "standard_deviation_decode_tokens_per_second": statistics.pstdev(speeds),
        "coefficient_of_variation": cv,
        "minimum_decode_tokens_per_second": min(speeds),
        "maximum_decode_tokens_per_second": max(speeds),
    }


def _run_fixed_decode_warmup(adapter, prompts: list[torch.Tensor]) -> dict[str, Any]:
    """Run the frozen, discarded 20-second cached-decode warm-up.

    This deliberately establishes runtime state without selecting a throughput
    regime.  Only the subsequent eight bracketed windows are scientific
    throughput observations.
    """
    repetition_count = 0
    decode_seconds = 0.0
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        while decode_seconds < 20.0:
            repetition, _ = one_measured_repetition(
                adapter, prompts[repetition_count % len(prompts)], repetition_count
            )
            decode_seconds += float(repetition["decode_seconds"])
            repetition_count += 1
    return {
        "warmup_mode": "fixed_discarded_cached_decode_duration_v1",
        "warmup_decode_seconds": decode_seconds,
        "warmup_repetition_count": repetition_count,
        # Preserve the result schema while deliberately discarding every
        # warm-up throughput observation.
        "warmup_repetitions": [],
        "warmup_blocks": [],
    }


def _run_official_window(
    adapter,
    prompts: list[torch.Tensor],
    window_index: int,
    repetition_offset: int,
) -> dict[str, Any]:
    """Run one fresh >=8-second official cached-decode timing window."""
    rows = []
    decode_seconds = 0.0
    started_at_unix_ns = time.time_ns()
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        while decode_seconds < 8.0:
            repetition, _ = one_measured_repetition(
                adapter, prompts[(window_index + len(rows)) % len(prompts)], repetition_offset + len(rows)
            )
            repetition.update({"window_index": window_index, "is_warmup": False})
            rows.append(repetition)
            decode_seconds += float(repetition["decode_seconds"])
    completed_at_unix_ns = time.time_ns()
    window = _summarize_block(rows, window_index)
    window.update({
        "decode_seconds": decode_seconds,
        "decode_measured_tokens": 127 * len(rows),
        "decode_tokens_per_second": 127 * len(rows) / decode_seconds,
        "minimum_decode_seconds": 8.0,
        "started_at_unix_ns": started_at_unix_ns,
        "completed_at_unix_ns": completed_at_unix_ns,
        "repetitions": rows,
    })
    return window


def _bracketed_inference_result(
    *,
    candidate_module: Any,
    candidate_checkpoint: dict[str, Any],
    candidate_model: Any,
    candidate_adapter: Any,
    candidate_train: Path,
    baseline_context: Path,
    baseline_checkpoint_path: Path,
    tokenizer: Any,
    prompts: list[torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    """Run baseline-before/candidate/baseline-after brackets in one process."""
    from .benchmark_policy import bracketed_decode_throughput

    baseline_train = (baseline_context / "train.py").resolve()
    baseline_module = load_candidate_definitions(baseline_train)
    baseline_checkpoint = load_checkpoint_file(baseline_checkpoint_path)
    baseline_model = load_checkpoint_model(baseline_module, baseline_checkpoint, device)
    baseline_adapter = build_adapter(baseline_module, baseline_model)
    correctness_prompts = [
        (prefix_tokens, fixed_input(tokenizer, device, prefix_tokens))
        for prefix_tokens in CORRECTNESS_PREFIX_TOKENS
    ]
    candidate_correctness = precision_aware_correctness(
        candidate_module, candidate_checkpoint, candidate_model, candidate_adapter,
        correctness_prompts, device,
    )
    baseline_correctness = precision_aware_correctness(
        baseline_module, baseline_checkpoint, baseline_model, baseline_adapter,
        correctness_prompts, device,
    )
    candidate_warmup = _run_fixed_decode_warmup(candidate_adapter, prompts)
    baseline_warmup = _run_fixed_decode_warmup(baseline_adapter, prompts)

    before_windows = []
    candidate_windows = []
    after_windows = []
    for window_index in range(8):
        before_windows.append(
            _run_official_window(baseline_adapter, prompts, window_index, window_index * 3)
        )
        candidate_windows.append(
            _run_official_window(candidate_adapter, prompts, window_index, window_index * 3)
        )
        after_windows.append(
            _run_official_window(baseline_adapter, prompts, window_index, window_index * 3)
        )
    paired = bracketed_decode_throughput(candidate_windows, before_windows, after_windows)
    candidate_rows = [row for window in candidate_windows for row in window["repetitions"]]
    baseline_rows = [
        row
        for window in before_windows + after_windows
        for row in window["repetitions"]
    ]
    candidate_diagnostics = paired["candidate_absolute_diagnostics"]
    baseline_before_diagnostics = paired["baseline_before_absolute_diagnostics"]
    baseline_after_diagnostics = paired["baseline_after_absolute_diagnostics"]
    # The bracketed speed comparison necessarily keeps the baseline model live
    # until the final baseline-after window.  It must not contaminate the
    # candidate's memory objective, however.  Release it completely, then run
    # one identical candidate decode solely to obtain candidate-only peak
    # allocation.  This diagnostic is outside the timed speed windows.
    del baseline_adapter
    del baseline_model
    gc.collect()
    torch.cuda.empty_cache()
    # Match the BF16 execution context used by every official decode window.
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        candidate_memory_row, _ = one_measured_repetition(candidate_adapter, prompts[0], -1)

    candidate_result = {
        "status": "completed",
        "inference_status": paired["normalized_inference_status"],
        "absolute_inference_status": candidate_diagnostics["inference_status"],
        "protocol_version": PROTOCOL_VERSION,
        "inference_metric_mode": "cached_decode",
        "input_tokens": 1024,
        "generated_tokens": 128,
        "cached_decode_transitions": 127,
        "inference_batch_size": 1,
        "protected_prompt_count": 8,
        "prompt_set_sha256": FROZEN_PROMPT_SET_SHA256,
        "prompt_token_hashes": list(FROZEN_PROMPT_TOKEN_HASHES),
        "decode_block_velocities": paired["candidate_window_tok_s"],
        "decode_velocity_point_estimate": paired["absolute_candidate_tok_s"],
        "decode_velocity_95pct_interval": paired["absolute_candidate_ci_95pct"],
        "cached_decode_tokens_per_second_median": paired["absolute_candidate_tok_s"],
        "cached_decode_coefficient_of_variation": candidate_diagnostics["cached_decode_coefficient_of_variation"],
        "decode_block_relative_drift": candidate_diagnostics["decode_block_relative_drift"],
        "decode_block_range_drift": candidate_diagnostics["decode_block_range_drift"],
        "stability_failure_reason": candidate_diagnostics["stability_failure_reason"],
        "warmup_repetitions": candidate_warmup["warmup_repetitions"],
        "warmup_blocks": candidate_warmup["warmup_blocks"],
        "warmup_mode": candidate_warmup["warmup_mode"],
        "warmup_decode_seconds": candidate_warmup["warmup_decode_seconds"],
        "warmup_repetition_count": candidate_warmup["warmup_repetition_count"],
        "inference_repetitions": candidate_rows,
        "measurement_windows": candidate_windows,
        "correctness": candidate_correctness,
        "peak_allocated_inference_memory_bytes": int(
            candidate_memory_row["peak_allocated_cuda_memory_mb"] * 1048576
        ),
        "reserved_cuda_memory_bytes": int(
            candidate_memory_row["peak_reserved_cuda_memory_mb"] * 1048576
        ),
        "candidate_only_memory_measurement": candidate_memory_row,
        "kv_cache_bytes": max(int(row["final_kv_cache_memory_mb"] * 1048576) for row in candidate_rows),
        "prefill_milliseconds": statistics.median(row["prefill_milliseconds"] for row in candidate_rows),
        "first_token_milliseconds": statistics.median(row["first_token_milliseconds"] for row in candidate_rows),
        "generation_mode": "greedy",
        "precision": "bfloat16",
        "evaluator_git_commit": git_commit(Path(__file__).resolve().parent),
        "evaluator_sha256": file_sha256(Path(__file__).resolve()),
        "sample": str(candidate_train),
        "baseline_inference_result": {
            "inference_status": "stable" if (
                baseline_before_diagnostics["inference_status"] == "stable"
                and baseline_after_diagnostics["inference_status"] == "stable"
            ) else "unstable",
            "correctness": baseline_correctness,
            "warmup_repetitions": baseline_warmup["warmup_repetitions"],
            "warmup_blocks": baseline_warmup["warmup_blocks"],
            "warmup_mode": baseline_warmup["warmup_mode"],
            "warmup_decode_seconds": baseline_warmup["warmup_decode_seconds"],
            "warmup_repetition_count": baseline_warmup["warmup_repetition_count"],
            "measurement_windows_before": before_windows,
            "measurement_windows_after": after_windows,
            "inference_repetitions": baseline_rows,
            "baseline_before_absolute_diagnostics": baseline_before_diagnostics,
            "baseline_after_absolute_diagnostics": baseline_after_diagnostics,
        },
        "paired_measurement_protocol": "baseline_before_candidate_baseline_after_interpolated_v1",
        "baseline_before_window_tok_s": paired["baseline_before_window_tok_s"],
        "baseline_after_window_tok_s": paired["baseline_after_window_tok_s"],
        "interpolated_baseline_window_tok_s": paired["interpolated_baseline_window_tok_s"],
        "absolute_baseline_before_tok_s": paired["absolute_baseline_before_tok_s"],
        "absolute_baseline_after_tok_s": paired["absolute_baseline_after_tok_s"],
        "absolute_candidate_tok_s": paired["absolute_candidate_tok_s"],
        "absolute_candidate_ci_95pct": paired["absolute_candidate_ci_95pct"],
        "absolute_baseline_tok_s": paired["absolute_baseline_tok_s"],
        "absolute_baseline_ci_95pct": paired["absolute_baseline_ci_95pct"],
        "normalized_pairwise_ratios": paired["normalized_pairwise_ratios"],
        "normalized_decode_throughput": paired["normalized_decode_throughput"],
        "normalized_decode_throughput_ci_95pct": paired["normalized_decode_throughput_ci_95pct"],
        "normalized_coefficient_of_variation": paired["normalized_coefficient_of_variation"],
        "normalized_relative_drift": paired["normalized_relative_drift"],
        "normalized_range_drift": paired["normalized_range_drift"],
        "normalized_inference_status": paired["normalized_inference_status"],
        "normalized_stability_failure_reason": paired["normalized_stability_failure_reason"],
    }
    return candidate_result


def _bootstrap_median_ci(values: list[float], *, seed: int = 0, resamples: int = 10000) -> tuple[float, float, float]:
    if len(values) < 2:
        raise RuntimeError("at least two decode windows are required for bootstrap")
    rng = random.Random(seed)
    medians = [statistics.median(rng.choices(values, k=len(values))) for _ in range(resamples)]
    medians.sort()
    return statistics.median(values), medians[int(0.025 * (len(medians) - 1))], medians[int(0.975 * (len(medians) - 1))]


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Compatibility normalizer for existing readiness fixtures."""
    blocks = list(payload.get("decode_blocks", []))
    passed = bool(payload.get("correctness_passed", False))
    result = dict(payload)
    result["inference_status"] = "stable" if passed and blocks else "invalid"
    result["peak_allocated_inference_memory_bytes"] = payload.get(
        "peak_allocated_cuda_memory_bytes"
    )
    result["reserved_cuda_memory_bytes"] = payload.get("reserved_cuda_memory_bytes")
    result["kv_cache_bytes"] = payload.get("kv_cache_bytes")
    result["decode_block_velocities"] = blocks
    result["decode_velocity_point_estimate"] = statistics.median(blocks) if blocks and passed else None
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path)
    parser.add_argument("--context", type=Path)
    parser.add_argument("--candidate", default="train.py")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--protocol")
    parser.add_argument("--process-log-sd", type=float, default=0.0)
    parser.add_argument("--event-log", type=Path)
    parser.add_argument("--campaign-id")
    parser.add_argument("--experiment-id", type=int)
    parser.add_argument("--training-run-id", type=int)
    parser.add_argument("--legacy-diagnostic", action="store_true")
    parser.add_argument("--bracketed-baseline-context", type=Path)
    parser.add_argument("--bracketed-baseline-checkpoint", type=Path)
    args = parser.parse_args()

    if args.train is None:
        if args.context is None:
            parser.error("one of --train or --context is required")
        args.train = args.context / args.candidate
    args.train = args.train.resolve()
    if args.sample is None:
        args.sample = args.output.with_suffix(".sample.txt")

    if PRECISION != "bfloat16" or GENERATION_MODE != "greedy":
        raise RuntimeError("protected inference configuration changed unexpectedly")
    if args.protocol not in (None, PROTOCOL_VERSION):
        raise RuntimeError("only the registered bracketed evaluator is permitted during Pair A")
    device = torch.device("cuda")
    module = load_candidate_definitions(args.train)
    checkpoint = load_checkpoint_file(args.checkpoint)
    model = load_checkpoint_model(module, checkpoint, device)
    adapter = build_adapter(module, model)
    tokenizer = _load_tokenizer(args.train)
    prompts = protected_prompts(tokenizer, device)
    if (args.bracketed_baseline_context is None) != (args.bracketed_baseline_checkpoint is None):
        parser.error("bracketed baseline context and checkpoint must be supplied together")
    if args.bracketed_baseline_context is not None:
        result = _bracketed_inference_result(
            candidate_module=module,
            candidate_checkpoint=checkpoint,
            candidate_model=model,
            candidate_adapter=adapter,
            candidate_train=args.train,
            baseline_context=args.bracketed_baseline_context.resolve(strict=True),
            baseline_checkpoint_path=args.bracketed_baseline_checkpoint.resolve(strict=True),
            tokenizer=tokenizer,
            prompts=prompts,
            device=device,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, sort_keys=True))
        return 0
    prompt_hash = FROZEN_PROMPT_SET_SHA256
    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    correctness_prompts = [
        (prefix_tokens, fixed_input(tokenizer, device, prefix_tokens))
        for prefix_tokens in CORRECTNESS_PREFIX_TOKENS
    ]
    correctness = precision_aware_correctness(
        module, checkpoint, model, adapter, correctness_prompts, device,
    )
    warmup = _run_fixed_decode_warmup(adapter, prompts)

    windows: list[dict[str, Any]] = []
    repetitions: list[dict[str, Any]] = []
    for window_index in range(8):
        rows = []
        decode_seconds = 0.0
        with autocast:
            while decode_seconds < 8.0:
                repetition, final_output = one_measured_repetition(
                    adapter, prompts[(window_index + len(rows)) % len(prompts)], len(repetitions)
                )
                repetition.update({"window_index": window_index, "is_warmup": False})
                repetitions.append(repetition)
                rows.append(repetition)
                decode_seconds += float(repetition["decode_seconds"])
        window_tokens = 127 * len(rows)
        window = _summarize_block(rows, window_index)
        window.update({
            "decode_seconds": decode_seconds,
            "decode_measured_tokens": window_tokens,
            "decode_tokens_per_second": window_tokens / decode_seconds,
            "minimum_decode_seconds": 8.0,
        })
        windows.append(window)

    block_speeds = [float(row["decode_tokens_per_second"]) for row in windows]
    point, lower, upper = _bootstrap_median_ci(block_speeds)
    mean = statistics.fmean(block_speeds)
    std = statistics.pstdev(block_speeds)
    drift = (max(block_speeds) - min(block_speeds)) / min(block_speeds)
    stability = classify_measurement_stability(block_speeds)
    metrics = {
        "status": "completed",
        "inference_status": stability["inference_status"],
        "protocol_version": PROTOCOL_VERSION,
        "inference_metric_mode": "cached_decode",
        "input_tokens": 1024,
        "generated_tokens": 128,
        "cached_decode_transitions": 127,
        "inference_batch_size": 1,
        "protected_prompt_count": 8,
        "prompt_set_sha256": prompt_hash,
        "prompt_token_hashes": list(FROZEN_PROMPT_TOKEN_HASHES),
        "decode_block_velocities": block_speeds,
        "decode_velocity_point_estimate": point,
        "decode_velocity_95pct_interval": [lower, upper],
        "cached_decode_tokens_per_second_median": point,
        "cached_decode_mean_tokens_per_second": mean,
        "cached_decode_standard_deviation_tokens_per_second": std,
        "cached_decode_coefficient_of_variation": stability["cached_decode_coefficient_of_variation"],
        "cached_decode_min_tokens_per_second": min(block_speeds),
        "cached_decode_max_tokens_per_second": max(block_speeds),
        "decode_block_relative_drift": stability["decode_block_relative_drift"],
        "decode_block_range_drift": stability["decode_block_range_drift"],
        "stability_failure_reason": stability["stability_failure_reason"],
        "warmup_repetitions": warmup["warmup_repetitions"],
        "warmup_blocks": warmup["warmup_blocks"],
        "warmup_mode": warmup["warmup_mode"],
        "warmup_decode_seconds": warmup["warmup_decode_seconds"],
        "warmup_repetition_count": warmup["warmup_repetition_count"],
        "inference_repetitions": repetitions,
        "measurement_windows": windows,
        "correctness": correctness,
        "peak_allocated_inference_memory_bytes": max(int(row["peak_allocated_cuda_memory_mb"] * 1048576) for row in repetitions),
        "reserved_cuda_memory_bytes": max(int(row["peak_reserved_cuda_memory_mb"] * 1048576) for row in repetitions),
        "kv_cache_bytes": max(int(row["final_kv_cache_memory_mb"] * 1048576) for row in repetitions),
        "prefill_milliseconds": statistics.median(row["prefill_milliseconds"] for row in repetitions),
        "first_token_milliseconds": statistics.median(row["first_token_milliseconds"] for row in repetitions),
        "generation_mode": "greedy",
        "precision": "bfloat16",
        "evaluator_git_commit": git_commit(Path(__file__).resolve().parent),
        "evaluator_sha256": file_sha256(Path(__file__).resolve()),
        "sample": str(args.sample),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
