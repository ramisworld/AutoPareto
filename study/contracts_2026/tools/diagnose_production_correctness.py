"""Run the frozen production cache-correctness gate without timing or training."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
    import torch

    from study.contracts_2026.inference_benchmark import (
        _fixed_input,
        _load_candidate_definitions,
        correctness_gate,
    )
    from protected.inference_adapter import build_adapter, load_checkpoint_file, load_checkpoint_model

    module, candidate_path = _load_candidate_definitions(args.context, "train.py")
    payload = load_checkpoint_file(args.checkpoint)
    device = torch.device("cuda")
    model = load_checkpoint_model(module, payload, device)
    adapter = build_adapter(module, model)
    fp32_model = load_checkpoint_model(module, payload, device).float()
    fp32_adapter = build_adapter(module, fp32_model)
    tokenizer = module.Tokenizer.from_directory()
    input_ids = _fixed_input(tokenizer, torch, device)
    try:
        result = correctness_gate(
            adapter,
            lambda ids: model(ids),
            input_ids,
            torch,
            model=model,
            fp32_model=fp32_model,
            fp32_adapter=fp32_adapter,
        )
        status = "passed"
        error = None
    except BaseException as exc:
        result = None
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
    output = {
        "status": status,
        "error": error,
        "candidate_source_sha256": sha256_file(candidate_path),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_config": payload.get("config"),
        "checkpoint_seed": payload.get("seed"),
        "adapter_kind": type(adapter).__name__,
        "fp32_adapter_kind": type(fp32_adapter).__name__,
        "result": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=str) + "\n")
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
