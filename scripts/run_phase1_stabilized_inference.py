#!/usr/bin/env python3
"""Run the registered Phase 1 stabilized inference-only rebenchmark.

This wrapper performs no training. It validates the seven frozen source/checkpoint
artifacts, then invokes the repository's protected deployment_benchmark.py with the
already-frozen adaptive timing policy at the Phase 1 headline workload.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "evaluation/phase1_stabilized_inference_rebenchmark.json"
PLAN = ROOT / "evaluation/phase1_equal_token_preflight.json"
EVALUATOR = ROOT / "deployment_benchmark.py"
STABILITY = ROOT / "benchmark_stability.py"
RESULT_ROOT = ROOT / "results/phase1"
DEFAULT_RUN_ID = "phase1-stabilized-inference-a40-20260824"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_prefix() -> list[str]:
    if shutil.which("uv"):
        return ["uv", "run", "--locked", "python"]
    return [sys.executable]


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text())
    if config.get("status") != "REGISTERED_BEFORE_REBENCHMARK":
        raise RuntimeError("stabilized rebenchmark registration is not frozen")
    return config


def validate(config: dict[str, Any]) -> None:
    if not EVALUATOR.is_file() or not STABILITY.is_file() or not PLAN.is_file():
        raise RuntimeError("required Phase 1 or stabilized evaluator file is missing")
    expected_hashes = config["implementation_hashes"]
    actual_evaluator = sha256(EVALUATOR)
    if actual_evaluator != expected_hashes["deployment_benchmark.py"]:
        raise RuntimeError(
            f"deployment evaluator hash mismatch: {actual_evaluator} != "
            f"{expected_hashes['deployment_benchmark.py']}"
        )
    actual_stability = sha256(STABILITY)
    if actual_stability != expected_hashes["benchmark_stability.py"]:
        raise RuntimeError(
            f"stability policy hash mismatch: {actual_stability} != "
            f"{expected_hashes['benchmark_stability.py']}"
        )

    plan_models = {
        model["model_id"]: model for model in json.loads(PLAN.read_text())["models"]
    }
    expected_ids = set(plan_models)
    selected_ids = {model["model_id"] for model in config["models"]}
    if selected_ids != expected_ids or len(config["models"]) != 7:
        raise RuntimeError(f"registered model panel mismatch: {selected_ids}")

    for model in config["models"]:
        model_id = model["model_id"]
        plan_model = plan_models[model_id]
        if model["source_path"] != plan_model["source_path"]:
            raise RuntimeError(f"source path differs from Phase 1 manifest for {model_id}")
        if model["source_sha256"] != plan_model["source_sha256"]:
            raise RuntimeError(f"source hash differs from Phase 1 manifest for {model_id}")
        source = ROOT / model["source_path"]
        checkpoint = ROOT / model["checkpoint_path"]
        if not source.is_file() or sha256(source) != model["source_sha256"]:
            raise RuntimeError(f"frozen source mismatch for {model_id}: {source}")
        if not checkpoint.is_file():
            raise RuntimeError(f"selected checkpoint is unavailable for {model_id}: {checkpoint}")
        checkpoint_sha = sha256(checkpoint)
        if checkpoint_sha != model["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch for {model_id}: {checkpoint_sha}")
        if checkpoint.stat().st_size <= 0:
            raise RuntimeError(f"selected checkpoint is empty for {model_id}")
        if model["model_id"] in {"autoresearch_compatible", "motpe_quality"}:
            if "recovery" not in model["checkpoint_provenance"]:
                raise RuntimeError(f"recovery provenance missing for {model_id}")

    protocol = config["protocol"]
    if protocol["version"] != "autopareto_inference_stabilized_v2":
        raise RuntimeError("unexpected stabilized protocol version")
    if protocol["stability_policy"] != {
        "warmup_block_size": 5,
        "required_stable_adjacent_blocks": 2,
        "max_warmups_per_pass": 30,
        "adjacent_median_relative_tolerance": 0.02,
        "within_block_coefficient_of_variation_limit": 0.05,
    }:
        raise RuntimeError("stabilized policy differs from the frozen repository policy")
    if protocol["measurement_passes"] != 3 or protocol["repetitions_per_pass"] != 10:
        raise RuntimeError("stabilized measurement repetition count is not frozen")


def summarize_model(model: dict[str, Any], raw: dict[str, Any] | None, returncode: int,
                    output: Path, log: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model_id": model["model_id"],
        "role": model["role"],
        "source_experiment_id": model["source_experiment_id"],
        "checkpoint_experiment_id": model["checkpoint_experiment_id"],
        "checkpoint_provenance": model["checkpoint_provenance"],
        "historical_legacy_median_tok_per_s": model["historical_value"],
        "historical_value_provenance": model["historical_value_provenance"],
        "returncode": returncode,
        "raw_output": str(output.relative_to(ROOT)),
        "log": str(log.relative_to(ROOT)),
        "status": "unavailable",
        "standardized_median_tok_per_s": None,
        "partial_median_tok_per_s_diagnostic": None,
        "standardized_max_tok_per_s_diagnostic": None,
        "measurement_pass_medians_tok_per_s": [],
        "warmup_counts": [],
        "measurement_pass_stability": [],
        "cache_correctness_passed": None,
        "peak_allocated_vram_mb": None,
        "peak_reserved_vram_mb": None,
        "incremental_kv_cache_median_mb": None,
    }
    if raw is None:
        return row
    repetitions = raw.get("inference_repetitions", [])
    speeds = [float(item["decode_tokens_per_second"]) for item in repetitions]
    row.update({
        "status": raw.get("status", "unknown"),
        "standardized_median_tok_per_s": (
            raw.get("median_cached_decode_tokens_per_second")
            if raw.get("status") == "completed" else None
        ),
        "partial_median_tok_per_s_diagnostic": raw.get(
            "median_cached_decode_tokens_per_second"
        ),
        "standardized_max_tok_per_s_diagnostic": max(speeds) if speeds else None,
        "measurement_pass_medians_tok_per_s": [
            item.get("median_decode_tokens_per_second")
            for item in raw.get("measurement_pass_summaries", [])
        ],
        "warmup_counts": [item.get("warmup_count") for item in raw.get("stability_passes", [])],
        "measurement_pass_stability": [
            item.get("stability") for item in raw.get("measurement_pass_summaries", [])
        ],
        "cache_correctness_passed": raw.get("correctness", {}).get("passed"),
        "peak_allocated_vram_mb": max(
            (float(item["peak_allocated_cuda_memory_mb"]) for item in repetitions),
            default=None,
        ),
        "peak_reserved_vram_mb": max(
            (float(item["peak_reserved_cuda_memory_mb"]) for item in repetitions),
            default=None,
        ),
        "incremental_kv_cache_median_mb": statistics.median(
            [float(item["incremental_kv_cache_memory_mb"]) for item in repetitions]
        ) if repetitions else None,
    })
    return row


def write_tables(config: dict[str, Any], rows: list[dict[str, Any]], out_dir: Path) -> None:
    summary = {
        "format": "autopareto_phase1_stabilized_inference_rebenchmark_results_v1",
        "status": "COMPLETED_WITH_EXPLICIT_MODEL_STATUSES",
        "config": str(CONFIG.relative_to(ROOT)),
        "protocol": config["protocol"],
        "workload": config["workload"],
        "primary_metric": config["analysis_policy"]["primary_metric"],
        "maximum_throughput_policy": config["analysis_policy"]["maximum_throughput"],
        "models": rows,
    }
    (RESULT_ROOT / "phase1_stabilized_inference_rebenchmark.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    fields = [
        "model_id", "role", "source_experiment_id", "checkpoint_experiment_id",
        "checkpoint_provenance", "historical_legacy_median_tok_per_s", "status",
        "standardized_median_tok_per_s", "partial_median_tok_per_s_diagnostic",
        "standardized_max_tok_per_s_diagnostic",
        "cache_correctness_passed", "peak_allocated_vram_mb", "peak_reserved_vram_mb",
        "incremental_kv_cache_median_mb",
    ]
    with (RESULT_ROOT / "phase1_stabilized_inference_rebenchmark.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    lines = [
        "# Phase 1 stabilized inference rebenchmark",
        "",
        "Pre-Stage-B measurement-protocol correction. Historical values are preserved; the standardized headline is the median of 30 stabilized synchronized-wall-clock decode repetitions. Maximum throughput is diagnostic only.",
        "",
        "| Model | Source | Checkpoint | Provenance | Historical legacy tok/s | Stabilized median tok/s | Status | Correctness |",
        "|---|---:|---:|---|---:|---:|---|---|",
    ]
    for row in rows:
        stabilized = row["standardized_median_tok_per_s"]
        stabilized_text = "UNAVAILABLE" if stabilized is None else f"{stabilized:.4f}"
        historical_text = f"{row['historical_legacy_median_tok_per_s']:.4f}"
        lines.append(
            f"| {row['model_id']} | {row['source_experiment_id']} | {row['checkpoint_experiment_id']} | "
            f"{row['checkpoint_provenance']} | {historical_text} | {stabilized_text} | "
            f"{row['status']} | {row['cache_correctness_passed']} |"
        )
    (RESULT_ROOT / "phase1_stabilized_inference_rebenchmark.md").write_text("\n".join(lines) + "\n")


def execute(config: dict[str, Any], run_id: str) -> int:
    out_dir = RESULT_ROOT / "stabilized_inference" / run_id
    if out_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing inference session: {out_dir}")
    out_dir.mkdir(parents=True)
    rows = []
    workload = config["workload"]
    stability = config["protocol"]["stability_policy"]
    for index, model in enumerate(config["models"], start=1):
        output = out_dir / f"{index:02d}-{model['model_id']}.json"
        log = out_dir / f"{index:02d}-{model['model_id']}.log"
        command = command_prefix() + [
            str(EVALUATOR),
            "--train", str(ROOT / model["source_path"]),
            "--checkpoint", str(ROOT / model["checkpoint_path"]),
            "--output", str(output),
            "--input-tokens", str(workload["input_tokens"]),
            "--generated-tokens", str(workload["generated_tokens"]),
            "--batch-size", str(workload["batch_size"]),
            "--protocol-version", config["protocol"]["version"],
            "--measurement-passes", str(config["protocol"]["measurement_passes"]),
            "--repetitions", str(config["protocol"]["repetitions_per_pass"]),
            "--warmup-block-size", str(stability["warmup_block_size"]),
            "--max-warmups", str(stability["max_warmups_per_pass"]),
            "--stability-median-tolerance", str(stability["adjacent_median_relative_tolerance"]),
            "--stability-cv-limit", str(stability["within_block_coefficient_of_variation_limit"]),
        ]
        print(f"[{index}/7] running stabilized inference for {model['model_id']}", flush=True)
        with log.open("w") as stream:
            completed = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        raw = json.loads(output.read_text()) if output.is_file() else None
        rows.append(summarize_model(model, raw, completed.returncode, output, log))
        status = rows[-1]["status"]
        print(f"[{index}/7] {model['model_id']}: {status}; returncode={completed.returncode}", flush=True)
    write_tables(config, rows, out_dir)
    return 0 if all(row["status"] == "completed" for row in rows) else 2


def summarize_existing(config: dict[str, Any], run_id: str) -> int:
    """Regenerate the CPU-only summary without rerunning inference."""
    out_dir = RESULT_ROOT / "stabilized_inference" / run_id
    rows = []
    for index, model in enumerate(config["models"], start=1):
        output = out_dir / f"{index:02d}-{model['model_id']}.json"
        log = out_dir / f"{index:02d}-{model['model_id']}.log"
        raw = json.loads(output.read_text()) if output.is_file() else None
        returncode = 0 if raw and raw.get("status") == "completed" else 2 if raw else 1
        rows.append(summarize_model(model, raw, returncode, output, log))
    write_tables(config, rows, out_dir)
    print(json.dumps({"status": "summarized", "run_id": run_id}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--summarize-existing", action="store_true")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    args = parser.parse_args()
    config = load_config()
    validate(config)
    if args.summarize_existing:
        return summarize_existing(config, args.run_id)
    if args.validate_only or not args.execute:
        print(json.dumps({
            "status": "validated",
            "models": [model["model_id"] for model in config["models"]],
            "protocol": config["protocol"]["version"],
            "message": "No GPU inference launched; pass --execute for the registered inference-only rebenchmark.",
        }, sort_keys=True))
        return 0
    return execute(config, args.run_id)


if __name__ == "__main__":
    raise SystemExit(main())
