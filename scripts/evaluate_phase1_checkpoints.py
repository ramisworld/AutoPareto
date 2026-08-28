#!/usr/bin/env python3
"""Evaluate Phase 1 milestone checkpoints without entering the training loop."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(candidate: Path, manifest_path: Path, output: Path, heldout: bool, suite: Path) -> int:
    import torch

    from inference_benchmark import load_candidate_definitions
    from prepare import Tokenizer, evaluate_bpb
    from protected.inference_adapter import load_checkpoint_file, load_checkpoint_model

    candidate = candidate.resolve()
    payload = json.loads(manifest_path.read_text())
    module = load_candidate_definitions(candidate)
    tokenizer = Tokenizer.from_directory()
    rows = []
    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    for row in payload["checkpoints"]:
        checkpoint = Path(row["checkpoint_path"]).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        actual = sha256(checkpoint)
        if actual != row["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {checkpoint}")
        checkpoint_payload = load_checkpoint_file(checkpoint)
        model = load_checkpoint_model(module, checkpoint_payload, torch.device("cuda"))
        model.eval()
        with torch.inference_mode(), autocast:
            val_bpb = float(evaluate_bpb(model, tokenizer, int(payload.get("device_batch_size", 64))))
        rows.append({
            **row,
            "validation_bpb": val_bpb,
            "evaluator": "prepare.evaluate_bpb",
        })
        del model, checkpoint_payload
        torch.cuda.empty_cache()

    result = {
        "format": "autopareto_phase1_checkpoint_evaluations_v1",
        "candidate_path": str(candidate),
        "candidate_sha256": sha256(candidate),
        "milestone_manifest": str(manifest_path.resolve()),
        "milestone_manifest_sha256": sha256(manifest_path),
        "checkpoints": rows,
    }
    if heldout:
        from downstream_evaluator import ModelScorer, evaluate_heldout_bpb, load_suite

        loaded_suite = load_suite(suite)
        final = rows[-1]
        scorer = ModelScorer(
            candidate=candidate,
            checkpoint=Path(final["checkpoint_path"]),
            max_length=int(loaded_suite["runtime"]["max_sequence_tokens"]),
            batch_size=int(loaded_suite["runtime"]["scoring_batch_size"]),
            module=module,
        )
        result["heldout_bpb"] = evaluate_heldout_bpb(
            scorer,
            loaded_suite["heldout_bpb"],
            random_seed=int(loaded_suite["runtime"]["random_seed"]),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "ok", "checkpoints": len(rows), "heldout": heldout}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--milestone-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--heldout", action="store_true")
    parser.add_argument("--suite", type=Path, default=ROOT / "evaluation/downstream_suite.json")
    args = parser.parse_args()
    return evaluate(args.candidate, args.milestone_manifest, args.output, args.heldout, args.suite)


if __name__ == "__main__":
    raise SystemExit(main())
