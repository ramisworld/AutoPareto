#!/usr/bin/env python3
"""Verify Phase 1's exact training-token stream without model/GPU training.

Static mode proves that all frozen candidates call the same deterministic loader.
Runtime mode additionally consumes the exact CPU loader stream for both optimizer
batch regimes at seeds 3, 4, and 5 and records SHA-256 digests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "evaluation/phase1_equal_token_preflight.json"
OUTPUT = ROOT / "evaluation/phase1_token_stream_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_plan(path: Path) -> dict:
    payload = json.loads(path.read_text())
    expected = {
        "baseline", "autoresearch_compatible", "tpe_quality", "motpe_quality",
        "autopareto_292", "autopareto_299", "autopareto_294",
    }
    actual = {model["model_id"] for model in payload["models"]}
    if actual != expected or len(payload["models"]) != 7:
        raise RuntimeError(f"token-stream audit requires the frozen seven-model panel: {actual}")
    if any(model.get("optional") for model in payload["models"]):
        raise RuntimeError("token-stream audit found an optional model; all seven must be Stage B")
    return payload


def static_audit(plan: dict) -> dict:
    loader_call = 'make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")'
    source_rows = []
    for model in plan["models"]:
        source = ROOT / model["source_path"]
        source_text = source.read_text()
        actual_source_sha256 = sha256(source)
        if actual_source_sha256 != model["source_sha256"]:
            raise RuntimeError(f"frozen source hash mismatch for {source}")
        if source_text.count(loader_call) != 1:
            raise RuntimeError(f"unexpected loader call count in {source}")
        source_rows.append({
            "model_id": model["model_id"],
            "source_path": model["source_path"],
            "source_sha256": actual_source_sha256,
            "loader_call_count": source_text.count(loader_call),
            "device_batch_size": model["device_batch_size"],
            "sequence_length": model["sequence_length"],
            "gradient_accumulation": model["gradient_accumulation"],
        })

    prepare_path = ROOT / "prepare.py"
    prepare_text = prepare_path.read_text()
    if "def make_dataloader(" not in prepare_text:
        raise RuntimeError("prepare.py does not expose make_dataloader")
    if "sorted(f for f in os.listdir(DATA_DIR) if f.endswith(\".parquet\")" not in prepare_text:
        raise RuntimeError("prepare.py shard ordering implementation changed")
    return {
        "format": "autopareto_phase1_token_stream_audit_v1",
        "static_status": "ANALYTICALLY_ESTABLISHED",
        "exact_sequence_for_fixed_cache": True,
        "conditional_on": "identical tokenizer.pkl, token_bytes.pt, and training parquet bytes",
        "prepare_path": "prepare.py",
        "prepare_sha256": sha256(prepare_path),
        "loader_function": "make_dataloader",
        "loader_call": loader_call,
        "source_rows": source_rows,
        "microbatch_tokens": 64 * 2048,
        "total_microbatches": plan["token_budget"] // (64 * 2048),
        "baseline_regime": {
            "gradient_accumulation": 4,
            "optimizer_steps": plan["token_budget"] // 524288,
            "microbatches": plan["token_budget"] // (64 * 2048),
        },
        "single_microbatch_regime": {
            "gradient_accumulation": 1,
            "optimizer_steps": plan["token_budget"] // 131072,
            "microbatches": plan["token_budget"] // (64 * 2048),
        },
        "determinism_reasons": [
            "sorted training shard paths, excluding pinned validation shard",
            "deterministic parquet row-group and row order",
            "deterministic tokenizer encoding and best-fit document packing",
            "deterministic shortest-document cropping when a document does not fit",
            "no RNG operation in make_dataloader",
            "all seven candidates use the same loader call and default buffer size",
        ],
        "runtime_verification": {
            "required": True,
            "output": "results/phase1/token_stream_verification.json",
            "observed": False,
        },
    }


def content_fingerprint(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def metadata_fingerprint(path: Path) -> dict:
    # Do not reread a potentially very large training corpus merely to prove
    # provenance; the exact consumed token stream is hashed below.
    return {"path": str(path), "bytes": path.stat().st_size}


def runtime_hash(plan: dict, audit: dict) -> dict:
    # Imports are intentionally delayed: static/offline audit must not require the
    # dataset or instantiate any CUDA tensor.
    import torch

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from prepare import DATA_DIR, TOKENIZER_DIR, VAL_FILENAME, Tokenizer, make_dataloader

    target_microbatches = audit["total_microbatches"]
    results = []
    for seed in plan["seeds"]:
        regime_hashes = {}
        for regime_name, grad_accumulation in (("baseline", 4), ("single_microbatch", 1)):
            torch.manual_seed(int(seed))
            loader = make_dataloader(
                Tokenizer.from_directory(), 64, 2048, "train", device="cpu"
            )
            digest = hashlib.sha256()
            microbatch_count = 0
            optimizer_steps = target_microbatches // grad_accumulation
            first_digest = None
            last_digest = None
            for _ in range(optimizer_steps):
                for _ in range(grad_accumulation):
                    inputs, targets, _epoch = next(loader)
                    if tuple(inputs.shape) != (64, 2048) or tuple(targets.shape) != (64, 2048):
                        raise RuntimeError(f"unexpected CPU loader shape: {inputs.shape}, {targets.shape}")
                    if inputs.device.type != "cpu" or targets.device.type != "cpu":
                        raise RuntimeError("CPU token-stream audit received a non-CPU tensor")
                    chunk = hashlib.sha256(
                        inputs.contiguous().numpy().tobytes(order="C")
                        + targets.contiguous().numpy().tobytes(order="C")
                    ).digest()
                    if first_digest is None:
                        first_digest = chunk.hex()
                    last_digest = chunk.hex()
                    digest.update(struct.pack("<Q", microbatch_count))
                    digest.update(chunk)
                    microbatch_count += 1
            if microbatch_count != target_microbatches:
                raise RuntimeError("CPU loader consumed the wrong number of microbatches")
            regime_hashes[regime_name] = {
                "gradient_accumulation": grad_accumulation,
                "optimizer_steps": optimizer_steps,
                "microbatches": microbatch_count,
                "tokens": microbatch_count * 64 * 2048,
                "stream_sha256": digest.hexdigest(),
                "first_microbatch_sha256": first_digest,
                "last_microbatch_sha256": last_digest,
            }
        if regime_hashes["baseline"]["stream_sha256"] != regime_hashes["single_microbatch"]["stream_sha256"]:
            raise RuntimeError(f"accumulation regimes differ for seed {seed}")
        results.append({"seed": seed, "regimes": regime_hashes})

    tokenizer_files = [
        Path(TOKENIZER_DIR) / "tokenizer.pkl",
        Path(TOKENIZER_DIR) / "token_bytes.pt",
    ]
    training_files = sorted(
        path for path in Path(DATA_DIR).glob("shard_*.parquet")
        if path.name != VAL_FILENAME
    )
    for path in tokenizer_files + training_files:
        if not path.is_file():
            raise FileNotFoundError(path)

    stream_hashes = [row["regimes"]["baseline"]["stream_sha256"] for row in results]
    if len(set(stream_hashes)) != 1:
        raise RuntimeError("same-seed-independent loader stream hashes differ across seeds")
    return {
        **audit,
        "status": "RUNTIME_VERIFIED",
        "runtime_verification": {
            "required": True,
            "output": "results/phase1/token_stream_verification.json",
            "observed": True,
            "tokenizer_files": [content_fingerprint(path) for path in tokenizer_files],
            "training_files": [metadata_fingerprint(path) for path in training_files],
            "seeds": results,
            "all_seed_stream_hashes_equal": True,
            "both_accumulation_regimes_equal": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=PLAN)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--runtime-hash", action="store_true")
    args = parser.parse_args()
    plan = load_plan(args.manifest)
    audit = static_audit(plan)
    payload = runtime_hash(plan, audit) if args.runtime_hash else audit
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload.get("status", payload.get("static_status")), "output": str(args.output), "runtime_hash": args.runtime_hash}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
