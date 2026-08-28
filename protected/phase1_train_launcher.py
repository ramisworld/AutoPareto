#!/usr/bin/env python3
"""Execute one frozen Phase 1 recipe at an exact token horizon.

This launcher is deliberately separate from the historical five-minute launcher.
It never changes the frozen candidate on disk: it verifies the candidate hash,
creates a derived execution-only source with the reviewed Phase 1 adapter, and
records both pure measured training time and end-to-end process time.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import runpy
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

import torch

from protected.phase1_source_transform import read_total_batch_size, transform_source
from protected.phase1_runtime import capture_gpu_state


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _force_seed(seed: int):
    original_manual_seed = torch.manual_seed
    original_cuda_manual_seed = torch.cuda.manual_seed
    original_cuda_manual_seed_all = torch.cuda.manual_seed_all

    def manual_seed(_ignored):
        return original_manual_seed(seed)

    def cuda_manual_seed(_ignored):
        return original_cuda_manual_seed(seed)

    def cuda_manual_seed_all(_ignored):
        return original_cuda_manual_seed_all(seed)

    torch.manual_seed = manual_seed
    torch.cuda.manual_seed = cuda_manual_seed
    torch.cuda.manual_seed_all = cuda_manual_seed_all
    return original_manual_seed, original_cuda_manual_seed, original_cuda_manual_seed_all


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _software_versions() -> dict[str, str | None]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", None),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "git_commit": _git_commit(),
    }


def _write_run_metadata(args: argparse.Namespace) -> Path:
    gpu_state = capture_gpu_state()
    metadata = {
        "format": "autopareto_phase1_run_metadata_v1",
        "recorded_utc": _utc_now_iso(),
        "pod_identifier": args.pod_identifier or os.environ.get("RUNPOD_POD_ID") or socket.gethostname(),
        "pod_start_time_utc": args.pod_start_time_utc or os.environ.get("AUTOPARETO_POD_START_TIME_UTC"),
        "seed": int(args.seed),
        "model_order_position": int(args.model_order_position) if args.model_order_position is not None else None,
        "model_identifier": args.model_id,
        "retried": bool(args.retried),
        "software_versions": _software_versions(),
        "gpu_state_before_execution": gpu_state,
        "gpu_uuid": gpu_state.get("gpu_uuid"),
    }
    path = args.run_dir / "run_metadata.json"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return path


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _validate_timing_contract(namespace: dict, token_budget: int, batch: int, telemetry_mode: str) -> dict:
    summary = namespace.get("phase1_step_timing_summary")
    if not isinstance(summary, dict):
        raise RuntimeError("derived candidate omitted Phase 1 timing summary")
    expected_steps, remainder = divmod(token_budget, batch)
    if remainder:
        raise RuntimeError("token budget is not exactly divisible by candidate batch")
    if summary.get("final_step_count") != expected_steps:
        raise RuntimeError("timing summary final step count does not match exact-token contract")
    if summary.get("run_token_budget") != token_budget:
        raise RuntimeError("timing summary token budget mismatch")
    if summary.get("steady_state_start_step_index") != 11:
        raise RuntimeError("historical steady-state boundary changed")
    timing_path = Path(summary["step_timing_path"])
    rows = _read_jsonl(timing_path)
    if len(rows) != expected_steps:
        raise RuntimeError(f"expected {expected_steps} per-step records, found {len(rows)}")
    for index, row in enumerate(rows):
        if row.get("zero_based_step_index") != index:
            raise RuntimeError("per-step records are not zero-based and contiguous")
        if row.get("optimizer_step_number") != index + 1:
            raise RuntimeError("optimizer step number is not one-based index + 1")
        if row.get("tokens_processed") != batch:
            raise RuntimeError("per-step token count changed")
        if row.get("cumulative_scientific_tokens") != (index + 1) * batch:
            raise RuntimeError("per-step cumulative token count changed")
        if row.get("step_wall_seconds", -1) < 0:
            raise RuntimeError("negative per-step wall duration")
        expected_steady_tokens = max(0, index + 1 - 11) * batch
        if row.get("cumulative_steady_state_tokens") != expected_steady_tokens:
            raise RuntimeError("steady-state token boundary changed")
        if index < 11 and row.get("cumulative_steady_state_seconds") != 0:
            raise RuntimeError("steps 0-10 were included in steady-state seconds")
    if rows[-1]["cumulative_scientific_tokens"] != token_budget:
        raise RuntimeError("per-step records do not reach exact token budget")
    telemetry = namespace.get("phase1_telemetry_payload")
    if not isinstance(telemetry, dict) or telemetry.get("format") != "autopareto_phase1_background_telemetry_v2":
        raise RuntimeError("background telemetry artifact is missing or has an unexpected format")
    if telemetry.get("sampling_outside_timed_critical_path") is not True:
        raise RuntimeError("telemetry was not declared outside the timed critical path")
    if telemetry.get("cuda_synchronization_called_by_sampler") is not False:
        raise RuntimeError("telemetry sampler synchronization contract failed")
    if telemetry.get("cuda_runtime_calls_by_sampler") is not False:
        raise RuntimeError("telemetry sampler must not call the CUDA runtime")
    if telemetry.get("telemetry_mode", "enabled") != ("disabled_control" if telemetry_mode == "disabled" else "enabled"):
        raise RuntimeError("telemetry mode metadata mismatch")
    if telemetry_mode == "enabled" and not telemetry.get("samples"):
        raise RuntimeError("background telemetry produced no samples")
    if any(row.get("cuda_stream_step_seconds") is not None for row in rows):
        raise RuntimeError("CUDA event timing must be disabled for scientific Phase 1 runs")
    if summary.get("cuda_event_profile") != "diagnostic-only; disabled in scientific Phase 1 timing":
        raise RuntimeError("scientific CUDA-event profile is not explicitly disabled")
    summed_seconds = sum(float(row["step_wall_seconds"]) for row in rows)
    if abs(summed_seconds - float(summary["total_scientific_step_wall_seconds"])) > 1e-9:
        raise RuntimeError("timing summary does not equal the sum of per-step wall durations")
    return summary


def _validate_contract(namespace: dict, token_budget: int, batch: int, protected_values: list[float], telemetry_mode: str) -> dict:
    required = {
        "model", "config", "total_tokens", "total_training_time", "step",
        "TOTAL_BATCH_SIZE", "peak_vram_mb", "num_params", "val_bpb",
    }
    missing = sorted(required - namespace.keys())
    if missing:
        raise RuntimeError(f"candidate omitted required outputs: {', '.join(missing)}")
    if len(protected_values) != 1:
        raise RuntimeError("candidate must call protected validation exactly once")
    if int(namespace["TOTAL_BATCH_SIZE"]) != batch:
        raise RuntimeError("candidate batch changed after source hash validation")
    expected_steps, remainder = divmod(token_budget, batch)
    if remainder:
        raise RuntimeError("token budget is not exactly divisible by candidate batch")
    if int(namespace["step"]) != expected_steps:
        raise RuntimeError(
            f"exact-token stop failed: step={namespace['step']} expected={expected_steps}"
        )
    if int(namespace["total_tokens"]) != token_budget:
        raise RuntimeError(
            f"exact-token accounting failed: total_tokens={namespace['total_tokens']} "
            f"expected={token_budget}"
        )
    if abs(float(namespace["val_bpb"]) - float(protected_values[0])) > 1e-12:
        raise RuntimeError("candidate val_bpb does not match protected evaluator")
    return _validate_timing_contract(namespace, token_budget, batch, telemetry_mode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--token-budget", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--transformed", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--milestone-dir", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--event-log", type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--model-order-position", type=int)
    parser.add_argument("--pod-identifier")
    parser.add_argument("--pod-start-time-utc")
    parser.add_argument("--retried", action="store_true")
    parser.add_argument("--telemetry-interval-seconds", required=True, type=float)
    parser.add_argument("--telemetry-mode", choices=("enabled", "disabled"), default="enabled")
    args = parser.parse_args()
    started_monotonic_ns = time.perf_counter_ns()

    source = args.source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    actual_source_sha256 = sha256(source)
    if actual_source_sha256 != args.source_sha256:
        raise RuntimeError(
            f"frozen source hash mismatch: {source} expected={args.source_sha256} "
            f"actual={actual_source_sha256}"
        )
    source_text = source.read_text()
    batch = read_total_batch_size(source_text)
    derived = transform_source(
        source_text,
        source_sha256=args.source_sha256,
        token_budget=args.token_budget,
        seed=args.seed,
        milestone_dir=args.milestone_dir.resolve(),
        final_checkpoint=args.checkpoint.resolve(),
        step_timing_path=(args.run_dir / "step_timing.jsonl").resolve(),
        timing_summary_path=(args.run_dir / "timing_summary.json").resolve(),
        telemetry_path=(args.run_dir / "telemetry.json").resolve(),
        telemetry_interval_seconds=args.telemetry_interval_seconds,
        telemetry_mode=args.telemetry_mode,
    )
    args.transformed.parent.mkdir(parents=True, exist_ok=True)
    args.transformed.write_text(derived)
    derived_sha256 = sha256(args.transformed)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.milestone_dir.mkdir(parents=True, exist_ok=True)
    run_metadata_path = _write_run_metadata(args)
    if args.event_log:
        os.environ["AUTOPARETO_PHASE1_EVENT_LOG"] = str(args.event_log.resolve())

    candidate_root = str(Path(__file__).resolve().parents[1])
    if candidate_root not in sys.path:
        sys.path.insert(0, candidate_root)
    transformed_root = str(args.transformed.parent)
    if transformed_root not in sys.path:
        sys.path.insert(0, transformed_root)

    originals = _force_seed(args.seed)
    protected_prepare = importlib.import_module("prepare")
    protected_values: list[float] = []
    original_evaluate_bpb = protected_prepare.evaluate_bpb

    def captured_protected_evaluation(model, tokenizer, batch_size):
        value = float(original_evaluate_bpb(model, tokenizer, batch_size))
        protected_values.append(value)
        return value

    protected_prepare.evaluate_bpb = captured_protected_evaluation
    try:
        namespace = runpy.run_path(str(args.transformed), run_name="__main__")
        timing_summary = _validate_contract(namespace, args.token_budget, batch, protected_values, args.telemetry_mode)
        milestone_manifest = args.milestone_dir / "manifest.json"
        if not milestone_manifest.is_file():
            raise RuntimeError("training completed without a Phase 1 milestone manifest")
        manifest = json.loads(milestone_manifest.read_text())
        final = [row for row in manifest["checkpoints"] if row["token_count"] == args.token_budget]
        if len(final) != 1 or final[0]["checkpoint_path"] != str(args.checkpoint.resolve()):
            raise RuntimeError("final checkpoint is missing from the milestone manifest")
        if not args.checkpoint.is_file() or sha256(args.checkpoint) != final[0]["checkpoint_sha256"]:
            raise RuntimeError("final checkpoint hash does not match milestone manifest")
        end_to_end_runtime_seconds = (time.perf_counter_ns() - started_monotonic_ns) / 1_000_000_000.0
        metrics = {
            "format": "autopareto_phase1_training_metrics_v3",
            "source_sha256": args.source_sha256,
            "derived_source_sha256": derived_sha256,
            "source_path": str(source),
            "derived_source_path": str(args.transformed),
            "training_seed": args.seed,
            "total_training_tokens": int(namespace["total_tokens"]),
            "optimizer_steps": int(namespace["step"]),
            "tokens_per_optimizer_step": batch,
            "val_bpb": float(protected_values[0]),
            # Continuous wall interval covering the complete scientific
            # optimizer-step workload, including steps 0–10.  The per-step sum
            # below remains the authoritative pure-workload decomposition.
            "total_scientific_wall_seconds": timing_summary["scientific_interval_seconds"],
            "total_scientific_step_wall_seconds": timing_summary["total_scientific_step_wall_seconds"],
            "steady_state_training_seconds": timing_summary["steady_state_training_seconds"],
            "steady_state_tokens": timing_summary["steady_state_tokens"],
            "steady_state_training_tokens_per_second": timing_summary["steady_state_training_tokens_per_second"],
            "steady_state_equivalent_seconds": timing_summary["steady_state_equivalent_seconds"],
            "common_token_boundary_start_step_index": timing_summary["common_token_boundary_start_step_index"],
            "common_token_boundary_excluded_tokens": timing_summary["common_token_boundary_excluded_tokens"],
            "common_token_boundary_seconds": timing_summary["common_token_boundary_seconds"],
            "common_token_boundary_tokens": timing_summary["common_token_boundary_tokens"],
            "common_token_boundary_tokens_per_second": timing_summary["common_token_boundary_tokens_per_second"],
            "common_token_boundary_equivalent_seconds": timing_summary["common_token_boundary_equivalent_seconds"],
            "primary_training_efficiency_metric": "steady_state_equivalent_seconds",
            "secondary_observed_training_metric": "total_scientific_step_wall_seconds",
            "cuda_event_profile": "disabled_scientific",
            "scientific_interval_seconds": timing_summary["scientific_interval_seconds"],
            "historical_timing_boundary": "Steps 0–10 are scientifically executed but excluded from the historical steady-state timing boundary. Steady-state timing begins at step index 11.",
            "excluded_initial_step_indices": list(range(11)),
            "excluded_initial_tokens": 11 * batch,
            "peak_training_cuda_memory_mb": float(namespace["peak_vram_mb"]),
            "parameter_count": int(namespace["num_params"]),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_bytes": args.checkpoint.stat().st_size,
            "checkpoint_sha256": sha256(args.checkpoint),
            "milestone_manifest": str(milestone_manifest.resolve()),
            "run_metadata": str(run_metadata_path.resolve()),
            "step_timing_artifact": str((args.run_dir / "step_timing.jsonl").resolve()),
            "timing_summary_artifact": str((args.run_dir / "timing_summary.json").resolve()),
            "telemetry_artifact": str((args.run_dir / "telemetry.json").resolve()),
            "telemetry_samples": len(namespace["phase1_telemetry_payload"].get("samples", [])),
            "telemetry_overhead_estimate_seconds": namespace["phase1_telemetry_payload"].get("sampler_thread_cpu_seconds"),
            "telemetry_mode": args.telemetry_mode,
            "end_to_end_runtime_seconds": end_to_end_runtime_seconds,
        }
        args.metrics.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        return 0
    except BaseException as exc:
        args.metrics.write_text(json.dumps({
            "format": "autopareto_phase1_training_metrics_v3",
            "source_sha256": args.source_sha256,
            "training_seed": args.seed,
            "run_metadata": str(run_metadata_path.resolve()),
            "end_to_end_runtime_seconds": (time.perf_counter_ns() - started_monotonic_ns) / 1_000_000_000.0,
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }, indent=2, sort_keys=True) + "\n")
        raise
    finally:
        protected_prepare.evaluate_bpb = original_evaluate_bpb
        torch.manual_seed, torch.cuda.manual_seed, torch.cuda.manual_seed_all = originals


if __name__ == "__main__":
    raise SystemExit(main())
