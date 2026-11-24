#!/usr/bin/env python3
"""Phase 5B diagnostic: GPU kernels and host launch calls per cached decode token.

This is not a timing measurement; the throughput numbers come from the protected
benchmark. For one context and checkpoint it loads the model and adapter exactly
as the protected benchmark does (same loader, same ``build_adapter``, the first
registered 1,024-token prompt), prefills, decodes 8 warm-up tokens, then
profiles 32 ``decode_one`` calls with ``torch.profiler``. It reports, per token:
- the number of GPU kernels executed;
- host ``cudaLaunchKernel``-family calls;
- ``cudaGraphLaunch`` calls;
- profiler-observed host time.

Run from the trusted tree:
    cd /opt/autopareto_trusted && <runtime python> <this file> --context <dir> --checkpoint <pt> --output <json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokens", type=int, default=32)
    args = parser.parse_args()
    sys.path.insert(0, str(Path.cwd()))
    import torch
    from torch.profiler import ProfilerActivity, profile

    from study.contracts_2026 import inference_benchmark as bench

    device = torch.device("cuda")
    train = (args.context / "train.py").resolve()
    module = bench.load_candidate_definitions(train)
    checkpoint = bench.load_checkpoint_file(args.checkpoint)
    model = bench.load_checkpoint_model(module, checkpoint, device)
    adapter = bench.build_adapter(module, model)
    tokenizer = bench._load_tokenizer(train)
    prompt = bench.protected_prompts(tokenizer, device)[0]
    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    with torch.inference_mode(), autocast:
        logits, state = adapter.prefill(prompt)
        token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        for _ in range(8):
            logits, state = adapter.decode_one(token, state)
            token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(args.tokens):
                logits, state = adapter.decode_one(token, state)
                token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            torch.cuda.synchronize()
    kernels = 0
    launches = 0
    graph_launches = 0
    host_us = 0.0
    names: dict[str, int] = {}
    for event in prof.events():
        device_type = str(getattr(event, "device_type", ""))
        if device_type.endswith("CUDA"):
            kernels += 1
            names[event.name] = names.get(event.name, 0) + 1
        elif event.name in ("cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel", "cuLaunchKernelEx"):
            launches += 1
        elif event.name in ("cudaGraphLaunch", "cuGraphLaunch"):
            graph_launches += 1
    totals = prof.key_averages()
    host_us = sum(row.self_cpu_time_total for row in totals)
    top = sorted(names.items(), key=lambda item: -item[1])[:12]
    result = {
        "format": "autopareto_phase5b_kernel_census_v1",
        "context": str(args.context),
        "train_py": str(train),
        "checkpoint": str(args.checkpoint),
        "profiled_decode_tokens": args.tokens,
        "gpu_kernels_per_token": kernels / args.tokens,
        "host_kernel_launch_calls_per_token": launches / args.tokens,
        "host_graph_launch_calls_per_token": graph_launches / args.tokens,
        "profiler_host_self_time_us_per_token": host_us / args.tokens,
        "top_kernels": [{"name": name[:120], "count_per_token": count / args.tokens} for name, count in top],
        "torch": torch.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "top_kernels"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
