#!/usr/bin/env python3
"""Phase 6 unified inference benchmark: one checkpoint, one mode, one record.

The headline numbers come from the registered Pair-A protocol
``autopareto_search_inference_bracketed_v2`` (1,024 input tokens, 128 generated,
batch 1, bfloat16 autocast, greedy; 20 s discarded warm-up; eight rounds of
baseline-before / candidate / baseline-after windows of at least 8 s of decode
each). That protocol is called unchanged: ``_bracketed_inference_result`` from
``study.contracts_2026.inference_benchmark`` is imported, not copied.

Around it, this script records what the registered protocol does not:

* **Full-workload correctness.** All eight protected 1,024-token prompts, all 128
  generated tokens, checked against an independent reference: the candidate's own
  uncached ``model.forward`` (the function that training optimised and BPB
  scores), run once, teacher-forced, over prompt + generated tokens. This covers
  the whole 1,024-token context, positions 1,024-1,151 and any ring-cache
  wrap-around. A step whose token differs from the reference argmax is a *hard
  mismatch* unless the reference prefers its own argmax by at most
  ``NEAR_TIE_LOGITS`` (a bfloat16 rounding tie).
* **Timed-path outputs.** ``one_measured_repetition`` (the timed function) is
  run once per prompt and its generated tokens must equal the validated tokens.
* **Cold start.** Definition load, checkpoint read, model build, adapter build,
  first prefill and the first decode calls (CUDA-graph capture, compilation),
  each timed separately with a fresh compiler cache.
* **Footprint.** Parameter count, weight bytes as stored on the GPU, peak
  allocated and peak reserved memory (the protocol's candidate-only pass).

Modes:
  native     ``build_adapter``: the candidate's own ``build_inference_adapter``,
             else the protected canonical adapter (the agent-system comparison).
  canonical  the protected ``CanonicalGPTAdapter``, forced (architecture-only
             control; valid only where it reproduces ``model.forward``).
  uncached   no K/V cache: every step recomputes ``model.forward`` over the whole
             prefix (model-neutral architecture-only control).

A candidate whose outputs fail the full-workload check is recorded and **not
timed**: fast but wrong outputs are not accepted.

    python scripts/phase6_unified_inference.py --train <ctx>/train.py --checkpoint <ck> \
        --mode native --label <name> --output <out.json> \
        --bracketed-baseline-context <ctx> --bracketed-baseline-checkpoint <ck>
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FORMAT = "autopareto_phase6_unified_inference_v1"
NEAR_TIE_LOGITS = 0.25
WORKLOAD = {"input_tokens": 1024, "generated_tokens": 128, "batch": 1}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tokens_sha256(tensor) -> str:
    return hashlib.sha256(json.dumps(tensor.detach().cpu().tolist(), separators=(",", ":")).encode()).hexdigest()


class UncachedReferenceAdapter:
    """No K/V cache: each call recomputes ``model.forward`` over the full prefix."""

    def __init__(self, model):
        self.model = model

    def prefill(self, input_ids):
        return self.model(input_ids), {"tokens": input_ids}

    def decode_one(self, token_id, state):
        tokens = __import__("torch").cat((state["tokens"], token_id), dim=1)
        return self.model(tokens)[:, -1:, :], {"tokens": tokens}

    def cache_nbytes(self, state) -> int:
        tokens = state["tokens"]
        return int(tokens.untyped_storage().nbytes())


def environment() -> dict[str, Any]:
    import torch

    info = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_name": torch.cuda.get_device_name(0),
        "hostname_hash": hashlib.sha256(platform.node().encode()).hexdigest()[:12],
    }
    try:
        import triton

        info["triton"] = triton.__version__
    except Exception:  # pragma: no cover - recorded, not required
        info["triton"] = None
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version,clocks.max.sm,clocks.max.mem,power.limit",
             "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout.strip()
        info["nvidia_smi"] = smi
    except Exception as exc:  # pragma: no cover
        info["nvidia_smi"] = f"unavailable: {exc}"
    return info


def footprint(model) -> dict[str, Any]:
    params = list(model.parameters())
    buffers = list(model.buffers())
    by_dtype: dict[str, int] = {}
    for tensor in params + buffers:
        by_dtype[str(tensor.dtype)] = by_dtype.get(str(tensor.dtype), 0) + tensor.numel() * tensor.element_size()
    return {
        "parameter_count": sum(p.numel() for p in params),
        "parameter_bytes": sum(p.numel() * p.element_size() for p in params),
        "buffer_bytes": sum(b.numel() * b.element_size() for b in buffers),
        "weight_bytes_by_dtype": by_dtype,
    }


def cold_start(ib, train_path: Path, checkpoint_path: Path, mode: str, prompt, device) -> tuple[dict, Any, Any, Any, Any]:
    """Load and build with each phase timed; then one generation with per-call timing."""
    import torch
    from protected.inference_adapter import CanonicalGPTAdapter, build_adapter, _validate_adapter

    sync = torch.cuda.synchronize
    record: dict[str, Any] = {}
    t = time.perf_counter()
    module = ib.load_candidate_definitions(train_path)
    record["definitions_load_seconds"] = time.perf_counter() - t
    t = time.perf_counter()
    checkpoint = ib.load_checkpoint_file(checkpoint_path)
    record["checkpoint_read_seconds"] = time.perf_counter() - t
    sync()
    t = time.perf_counter()
    model = ib.load_checkpoint_model(module, checkpoint, device)
    sync()
    record["model_build_seconds"] = time.perf_counter() - t
    t = time.perf_counter()
    try:
        if mode == "native":
            adapter = build_adapter(module, model)
        elif mode == "canonical":
            adapter = _validate_adapter(CanonicalGPTAdapter(model))
        else:
            adapter = _validate_adapter(UncachedReferenceAdapter(model))
    except Exception as exc:  # e.g. the canonical adapter refuses a non-canonical architecture
        record["adapter_build_error"] = f"{type(exc).__name__}: {exc}"
        return record, module, checkpoint, model, None
    sync()
    record["adapter_build_seconds"] = time.perf_counter() - t
    record["adapter_class"] = type(adapter).__name__
    record["adapter_is_candidate_supplied"] = mode == "native" and callable(getattr(module, "build_inference_adapter", None))
    decode_calls = []
    with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        sync()
        t = time.perf_counter()
        logits, state = adapter.prefill(prompt)
        sync()
        record["first_prefill_milliseconds"] = (time.perf_counter() - t) * 1000
        token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        for _ in range(127):
            sync()
            t = time.perf_counter()
            logits, state = adapter.decode_one(token, state)
            sync()
            decode_calls.append((time.perf_counter() - t) * 1000)
            token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        sync()
        t = time.perf_counter()
        logits, state = adapter.prefill(prompt)
        sync()
        record["second_prefill_milliseconds"] = (time.perf_counter() - t) * 1000
    record["first_three_decode_milliseconds"] = decode_calls[:3]
    record["median_decode_call_milliseconds_first_generation"] = statistics.median(decode_calls)
    record["first_generation_decode_milliseconds_total"] = sum(decode_calls)
    del logits, state
    return record, module, checkpoint, model, adapter


def full_workload_correctness(ib, model, adapter, prompts) -> dict[str, Any]:
    """Every prompt, every generated token, against teacher-forced uncached model.forward."""
    import torch

    per_prompt = []
    outputs = []
    with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for index, prompt in enumerate(prompts):
            output, captured, _ = ib.generate_cached(adapter, prompt, WORKLOAD["generated_tokens"], capture_logits=True)
            outputs.append(output)
            reference = model(output[:, :-1])[0].float()
            start = prompt.shape[1] - 1
            reference = reference[start:start + WORKLOAD["generated_tokens"]]
            candidate = torch.stack([row[0] for row in captured]).to(reference.device)
            generated = output[0, prompt.shape[1]:]
            chosen = reference.gather(1, generated[:, None])[:, 0]
            best = reference.max(dim=-1).values
            deficit = (best - chosen).clamp_min(0)
            agree = reference.argmax(dim=-1) == generated
            hard = (~agree) & (deficit > NEAR_TIE_LOGITS)
            error = (candidate - reference).abs()
            finite = bool(torch.isfinite(candidate).all().item() and torch.isfinite(reference).all().item())
            mean_err = error.mean(dim=-1)
            per_prompt.append({
                "prompt_index": index,
                "generated_tokens_sha256": tokens_sha256(generated),
                "token_agreement": int(agree.sum().item()),
                "near_tie_disagreements": int(((~agree) & ~hard).sum().item()),
                "hard_mismatches": int(hard.sum().item()),
                "hard_mismatch_steps": [int(i) for i in torch.nonzero(hard).flatten().tolist()],
                "max_reference_deficit": float(deficit.max().item()),
                "max_abs_logit_error": float(error.max().item()),
                "max_step_mean_abs_logit_error": float(mean_err.max().item()),
                "mean_abs_logit_error": float(error.mean().item()),
                "mean_abs_logit_error_steps_0_31": float(mean_err[:32].mean().item()),
                "mean_abs_logit_error_steps_96_127": float(mean_err[96:].mean().item()),
                "finite": finite,
            })
    hard_total = sum(row["hard_mismatches"] for row in per_prompt)
    return {
        "reference": "candidate model.forward, uncached, teacher-forced on the generated sequence, bfloat16 autocast",
        "prompts": len(prompts),
        "positions_checked": [prompts[0].shape[1], prompts[0].shape[1] + WORKLOAD["generated_tokens"] - 1],
        "near_tie_logits": NEAR_TIE_LOGITS,
        "tokens_checked": len(prompts) * WORKLOAD["generated_tokens"],
        "token_agreement": sum(row["token_agreement"] for row in per_prompt),
        "near_tie_disagreements": sum(row["near_tie_disagreements"] for row in per_prompt),
        "hard_mismatches": hard_total,
        "max_abs_logit_error": max(row["max_abs_logit_error"] for row in per_prompt),
        "max_step_mean_abs_logit_error": max(row["max_step_mean_abs_logit_error"] for row in per_prompt),
        "all_finite": all(row["finite"] for row in per_prompt),
        "passed": hard_total == 0 and all(row["finite"] for row in per_prompt),
        "per_prompt": per_prompt,
    }, outputs


def timed_path_outputs(ib, adapter, prompts, validated) -> dict[str, Any]:
    import torch

    rows = []
    with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for index, prompt in enumerate(prompts):
            repetition, output = ib.one_measured_repetition(adapter, prompt, index)
            rows.append({
                "prompt_index": index,
                "equal_to_validated": bool(torch.equal(output, validated[index])),
                "generated_tokens_sha256": tokens_sha256(output[0, prompt.shape[1]:]),
                "prefill_milliseconds": repetition["prefill_milliseconds"],
                "decode_tokens_per_second": repetition["decode_tokens_per_second"],
            })
    return {"all_equal": all(row["equal_to_validated"] for row in rows), "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=("native", "canonical", "uncached"), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bracketed-baseline-context", type=Path, required=True)
    parser.add_argument("--bracketed-baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--correctness-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    for key in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        if os.environ.get(key):
            raise SystemExit(f"{key} is set; the registered protocol runs with the default allocator")
    compile_cache = Path(tempfile.mkdtemp(prefix="phase6_cold_"))
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(compile_cache / "inductor")
    os.environ["TRITON_CACHE_DIR"] = str(compile_cache / "triton")

    import torch
    from study.contracts_2026 import inference_benchmark as ib

    device = torch.device("cuda")
    started = time.time()
    train_path = args.train.resolve(strict=True)
    tokenizer = ib._load_tokenizer(train_path)
    prompts = ib.protected_prompts(tokenizer, device)
    record: dict[str, Any] = {
        "format": FORMAT,
        "label": args.label,
        "mode": args.mode,
        "workload": WORKLOAD,
        "protocol": ib.PROTOCOL_VERSION,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "inputs": {
            "train": str(train_path),
            "train_sha256": sha256(train_path),
            "prepare_sha256": sha256(train_path.parent / "prepare.py") if (train_path.parent / "prepare.py").is_file() else None,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256(args.checkpoint),
            "baseline_train_sha256": sha256(args.bracketed_baseline_context / "train.py"),
            "baseline_checkpoint_sha256": sha256(args.bracketed_baseline_checkpoint),
            "script_sha256": sha256(Path(__file__)),
            "evaluator_sha256": sha256(Path(ib.__file__)),
            "prompt_set_sha256": ib.FROZEN_PROMPT_SET_SHA256,
        },
        "environment": environment(),
    }
    torch.cuda.reset_peak_memory_stats()
    cold, module, checkpoint, model, adapter = cold_start(ib, train_path, args.checkpoint, args.mode, prompts[0], device)
    record["cold_start"] = cold
    record["footprint"] = footprint(model)
    if adapter is None:
        record["status"] = "incompatible_adapter"
        correctness = {"hard_mismatches": None, "token_agreement": None}
    else:
        correctness, validated = full_workload_correctness(ib, model, adapter, prompts)
        record["full_workload_correctness"] = correctness
        record["timed_path_outputs"] = timed_path_outputs(ib, adapter, prompts, validated)
    if adapter is None:
        pass
    elif not correctness["passed"] or not record["timed_path_outputs"]["all_equal"]:
        record["status"] = "incorrect_not_timed"
    elif args.correctness_only:
        record["status"] = "correctness_only"
    else:
        try:
            result = ib._bracketed_inference_result(
                candidate_module=module,
                candidate_checkpoint=checkpoint,
                candidate_model=model,
                candidate_adapter=adapter,
                candidate_train=train_path,
                baseline_context=args.bracketed_baseline_context.resolve(strict=True),
                baseline_checkpoint_path=args.bracketed_baseline_checkpoint.resolve(strict=True),
                tokenizer=tokenizer,
                prompts=prompts,
                device=device,
            )
        except Exception as exc:  # recorded, never hidden
            record["status"] = "protocol_error"
            record["protocol_error"] = f"{type(exc).__name__}: {exc}"
        else:
            record["status"] = "completed"
            record["registered_protocol_result"] = result
            rows = result["inference_repetitions"]
            prefill = [row["prefill_milliseconds"] for row in rows]
            decode = [row["decode_seconds"] * 1000 / row["decode_measured_tokens"] for row in rows]
            e2e = [row["prefill_milliseconds"] + row["first_token_milliseconds"] + row["decode_seconds"] * 1000 for row in rows]
            record["summary"] = {
                "decode_tokens_per_second": result["absolute_candidate_tok_s"],
                "decode_tokens_per_second_ci95": result["absolute_candidate_ci_95pct"],
                "speedup_vs_bracketing_baseline": result["normalized_decode_throughput"],
                "speedup_ci95": result["normalized_decode_throughput_ci_95pct"],
                "bracketing_baseline_tokens_per_second": result["absolute_baseline_tok_s"],
                "normalized_inference_status": result["normalized_inference_status"],
                "prefill_milliseconds_median": statistics.median(prefill),
                "prefill_tokens_per_second_median": 1024 / (statistics.median(prefill) / 1000),
                "decode_milliseconds_per_token_median": statistics.median(decode),
                "end_to_end_milliseconds_median": statistics.median(e2e),
                "end_to_end_tokens_per_second_median": 128 / (statistics.median(e2e) / 1000),
                "end_to_end_milliseconds_iqr": [statistics.quantiles(e2e, n=4)[0], statistics.quantiles(e2e, n=4)[2]],
                "peak_allocated_mib": result["peak_allocated_inference_memory_bytes"] / 1048576,
                "peak_reserved_mib": result["reserved_cuda_memory_bytes"] / 1048576,
                "kv_cache_mib": result["kv_cache_bytes"] / 1048576,
                "weight_mib": (record["footprint"]["parameter_bytes"] + record["footprint"]["buffer_bytes"]) / 1048576,
                "parameter_count": record["footprint"]["parameter_count"],
                "repetitions": len(rows),
                "cold_first_prefill_milliseconds": cold["first_prefill_milliseconds"],
                "cold_overhead_prefill_milliseconds": cold["first_prefill_milliseconds"] - statistics.median(prefill),
                "cold_first_decode_call_milliseconds": cold["first_three_decode_milliseconds"][0],
                "cold_overhead_first_generation_decode_milliseconds":
                    cold["first_generation_decode_milliseconds_total"] - statistics.median(decode) * 127,
                "load_seconds": cold["definitions_load_seconds"] + cold["checkpoint_read_seconds"]
                    + cold["model_build_seconds"] + cold["adapter_build_seconds"],
            }
    record["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record["wall_seconds"] = time.time() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".partial")
    tmp.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    tmp.rename(args.output)
    shutil.rmtree(compile_cache, ignore_errors=True)
    summary = record.get("summary", {})
    print(json.dumps({
        "label": args.label, "mode": args.mode, "status": record["status"],
        "hard_mismatches": correctness["hard_mismatches"], "token_agreement": correctness["token_agreement"],
        "decode_tok_s": summary.get("decode_tokens_per_second"), "speedup": summary.get("speedup_vs_bracketing_baseline"),
        "peak_mib": summary.get("peak_allocated_mib"), "wall_s": round(record["wall_seconds"], 1),
    }), flush=True)
    return 0 if record["status"] in ("completed", "correctness_only") else 3


if __name__ == "__main__":
    raise SystemExit(main())
