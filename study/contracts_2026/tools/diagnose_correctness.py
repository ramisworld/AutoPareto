"""Trace protected cached/full-prefix correctness without training."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Any

from study.contracts_2026.inference_benchmark import _fixed_input, _load_candidate_definitions


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def trace(
    model: Any,
    adapter: Any,
    prompt: Any,
    torch_module: Any,
    *,
    steps: int,
    use_autocast: bool,
    reference_mode: str = "protected_sdpa",
) -> dict[str, Any]:
    from protected.inference_adapter import CanonicalGPTAdapter

    autocast = torch_module.amp.autocast(device_type="cuda", dtype=torch_module.bfloat16)
    context = autocast if use_autocast else _NullContext()
    records: list[dict[str, Any]] = []
    with torch_module.inference_mode(), context:
        cached_logits, state = adapter.prefill(prompt)
        common_tokens = prompt
        for index in range(steps):
            if reference_mode == "candidate_forward":
                reference_logits = model(common_tokens)
                reference = reference_logits[:, -1, :].detach().float()
            elif isinstance(adapter, CanonicalGPTAdapter):
                reference_logits, _ = adapter.prefill(common_tokens)
                reference = reference_logits[:, -1, :].detach().float()
            else:
                reference = model(common_tokens)[:, -1, :].detach().float()
            cached = cached_logits[:, -1, :].detach().float()
            difference = (reference - cached).abs()
            reference_token = reference.argmax(dim=-1, keepdim=True)
            cached_token = cached.argmax(dim=-1, keepdim=True)
            records.append({
                "step": index,
                "mean_abs_error": float(difference.mean().item()),
                "max_abs_error": float(difference.max().item()),
                "greedy_token_match": bool(torch_module.equal(reference_token, cached_token)),
            })
            common_tokens = torch_module.cat((common_tokens, reference_token), dim=1)
            if index + 1 < steps:
                cached_logits, state = adapter.decode_one(reference_token, state)
    return {
        "steps": records,
        "max_mean_abs_error": max(row["mean_abs_error"] for row in records),
        "max_abs_error": max(row["max_abs_error"] for row in records),
        "all_greedy_tokens_match": all(row["greedy_token_match"] for row in records),
        "reference_mode": reference_mode,
    }


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--steps", type=int, default=32)
    args = parser.parse_args(argv)

    import torch
    from protected.inference_adapter import build_adapter, load_checkpoint_file, load_checkpoint_model

    candidate = args.context / "train.py"
    module, _ = _load_candidate_definitions(args.context, "train.py")
    checkpoint = load_checkpoint_file(args.checkpoint)
    device = torch.device("cuda")
    model = load_checkpoint_model(module, checkpoint, device)
    adapter = build_adapter(module, model)
    tokenizer = module.Tokenizer.from_directory()
    prompt = _fixed_input(tokenizer, torch, device)

    bf16 = trace(model, adapter, prompt, torch, steps=args.steps, use_autocast=True)
    bf16_candidate = trace(
        model, adapter, prompt, torch, steps=args.steps, use_autocast=True, reference_mode="candidate_forward"
    )
    fp32_model = load_checkpoint_model(module, checkpoint, device).float()
    fp32_adapter = build_adapter(module, fp32_model)
    fp32 = trace(fp32_model, fp32_adapter, prompt, torch, steps=args.steps, use_autocast=False)
    try:
        fp32_candidate = trace(
            fp32_model, fp32_adapter, prompt, torch, steps=args.steps, use_autocast=False, reference_mode="candidate_forward"
        )
        fp32_candidate_error = None
    except BaseException as exc:
        fp32_candidate = None
        fp32_candidate_error = f"{type(exc).__name__}: {exc}"

    payload = {
        "schema": "autopareto_contracts_2026_correctness_trace_v1",
        "candidate": "train.py",
        "candidate_source_sha256": sha256_file(candidate),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_seed": checkpoint.get("seed"),
        "checkpoint_train_file": checkpoint.get("train_file"),
        "checkpoint_config": checkpoint.get("config"),
        "adapter_type": type(adapter).__name__,
        "steps": args.steps,
        "bf16_autocast": bf16,
        "bf16_candidate_forward": bf16_candidate,
        "fp32": fp32,
        "fp32_candidate_forward": fp32_candidate,
        "fp32_candidate_forward_error": fp32_candidate_error,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "adapter_type": payload["adapter_type"],
        "bf16_max_mean_abs_error": bf16["max_mean_abs_error"],
        "bf16_candidate_forward_max_mean_abs_error": bf16_candidate["max_mean_abs_error"],
        "fp32_max_mean_abs_error": fp32["max_mean_abs_error"],
        "fp32_candidate_forward_max_mean_abs_error": None if fp32_candidate is None else fp32_candidate["max_mean_abs_error"],
        "bf16_all_greedy_tokens_match": bf16["all_greedy_tokens_match"],
        "fp32_all_greedy_tokens_match": fp32["all_greedy_tokens_match"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
