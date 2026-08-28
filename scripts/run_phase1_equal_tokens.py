#!/usr/bin/env python3
"""Staged Phase 1 executor; validation is the default and execution is explicit."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "evaluation/phase1_equal_token_preflight.json"
STABILIZED_CONFIG = ROOT / "evaluation/phase1_stabilized_inference_rebenchmark.json"
PHASE1_ROOT = ROOT / "results/phase1"
TOKEN_STREAM_VERIFICATION = PHASE1_ROOT / "token_stream_verification.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_plan() -> dict:
    payload = json.loads(PLAN.read_text())
    if payload.get("status") != "PREPARING / PREFLIGHT":
        raise RuntimeError("Phase 1 plan is not in preflight state")
    for model in payload["models"]:
        source = ROOT / model["source_path"]
        if not source.is_file() or sha256(source) != model["source_sha256"]:
            raise RuntimeError(f"frozen source mismatch: {source}")
    expected_order = {
        "3": ["baseline", "autoresearch_compatible", "tpe_quality", "motpe_quality", "autopareto_292", "autopareto_299", "autopareto_294"],
        "4": ["autopareto_292", "autopareto_294", "motpe_quality", "baseline", "autopareto_299", "autoresearch_compatible", "tpe_quality"],
        "5": ["autopareto_299", "tpe_quality", "autopareto_294", "autoresearch_compatible", "motpe_quality", "autopareto_292", "baseline"],
    }
    if payload.get("preregistered_model_order") != expected_order:
        raise RuntimeError("Phase 1 preregistered model order does not match the frozen amendment")
    for run in payload["runs"]:
        order = expected_order[str(run["seed"])]
        expected_position = order.index(run["model_id"]) + 1
        if run.get("model_order_position", expected_position) != expected_position:
            raise RuntimeError(f"run order position mismatch for {run['run_id']}")
    return payload


def load_stabilized_config() -> dict:
    payload = json.loads(STABILIZED_CONFIG.read_text())
    if payload.get("protocol", {}).get("version") != "autopareto_inference_stabilized_v2":
        raise RuntimeError("Phase 1 stabilized inference registration has an unexpected protocol")
    return payload


def command_prefix() -> list[str]:
    if shutil.which("uv"):
        return ["uv", "run", "--locked", "python"]
    return [sys.executable]


def validate_persistent_runtime() -> None:
    """Fail before reserving the GPU if runtime data/checkpoint storage is unsafe."""
    for script in (
        ROOT / "scripts/ensure_persistent_cache.py",
        ROOT / "scripts/ensure_persistent_checkpoint_storage.py",
    ):
        subprocess.run(command_prefix() + [str(script)], cwd=ROOT, check=True)


def require_token_stream_verification(plan: dict) -> None:
    if not TOKEN_STREAM_VERIFICATION.is_file():
        raise RuntimeError(
            "missing CPU token-stream verification; run scripts/verify_phase1_token_stream.py "
            "--runtime-hash before Stage B"
        )
    payload = json.loads(TOKEN_STREAM_VERIFICATION.read_text())
    runtime = payload.get("runtime_verification", {})
    if payload.get("status") != "RUNTIME_VERIFIED":
        raise RuntimeError("token-stream verification is not runtime-verified")
    if payload.get("prepare_sha256") != plan["data_equivalence"]["loader_sha256_at_preflight"]:
        raise RuntimeError("token-stream verification used a different prepare.py hash")
    if not runtime.get("observed") or not runtime.get("all_seed_stream_hashes_equal"):
        raise RuntimeError("token-stream verification did not establish equal seed streams")
    if not runtime.get("both_accumulation_regimes_equal"):
        raise RuntimeError("token-stream verification did not establish equal accumulation streams")
    verified_seeds = {row.get("seed") for row in runtime.get("seeds", [])}
    if verified_seeds != set(plan["seeds"]):
        raise RuntimeError(f"token-stream verification seeds mismatch: {verified_seeds}")


def integrity_gate(plan: dict) -> None:
    core = {m["model_id"] for m in plan["models"]}
    required = {"seed3"}
    runs = [r for r in plan["runs"] if r["stage"] == "B" and r["model_id"] in core]
    if len(runs) != len(core) or {r["seed_label"] for r in runs} != required or {r["model_id"] for r in runs} != core:
        raise RuntimeError("Phase 1 manifest does not contain the complete Stage B panel")
    for run in runs:
        model = next(item for item in plan["models"] if item["model_id"] == run["model_id"])
        metrics = PHASE1_ROOT / "runs" / run["run_id"] / "training_metrics.json"
        evaluations = PHASE1_ROOT / "runs" / run["run_id"] / "checkpoint_evaluations.json"
        if not metrics.is_file() or not evaluations.is_file():
            raise RuntimeError(f"Stage B integrity gate missing output for {run['run_id']}")
        data = json.loads(metrics.read_text())
        if data.get("total_training_tokens") != plan["token_budget"]:
            raise RuntimeError(f"Stage B token mismatch for {run['run_id']}")
        expected_steps = plan["token_budget"] // model["tokens_per_optimizer_step"]
        if data.get("optimizer_steps") != expected_steps:
            raise RuntimeError(f"Stage B optimizer-step mismatch for {run['run_id']}")
        if data.get("source_sha256") != model["source_sha256"]:
            raise RuntimeError(f"Stage B source hash mismatch for {run['run_id']}")
        if data.get("format") != "autopareto_phase1_training_metrics_v3":
            raise RuntimeError(f"Stage B timing metrics format mismatch for {run['run_id']}")
        expected_steady_tokens = max(0, expected_steps - 11) * model["tokens_per_optimizer_step"]
        if data.get("steady_state_tokens") != expected_steady_tokens:
            raise RuntimeError(f"Stage B steady-state token mismatch for {run['run_id']}")
        if data.get("primary_training_efficiency_metric") != "steady_state_equivalent_seconds":
            raise RuntimeError(f"Stage B primary timing metric mismatch for {run['run_id']}")
        if data.get("cuda_event_profile") != "disabled_scientific":
            raise RuntimeError(f"Stage B scientific CUDA-event profile mismatch for {run['run_id']}")
        for artifact_key in ("step_timing_artifact", "timing_summary_artifact", "telemetry_artifact", "run_metadata"):
            if not Path(data[artifact_key]).is_file():
                raise RuntimeError(f"Stage B artifact missing for {run['run_id']}: {artifact_key}")
        telemetry = json.loads(Path(data["telemetry_artifact"]).read_text())
        if telemetry.get("sampling_interval_seconds") != plan["telemetry_sampling_interval_seconds"]:
            raise RuntimeError(f"Stage B telemetry interval mismatch for {run['run_id']}")
        if not telemetry.get("samples"):
            raise RuntimeError(f"Stage B telemetry has no samples for {run['run_id']}")
        if not data.get("checkpoint_sha256") or data.get("checkpoint_bytes", 0) <= 0:
            raise RuntimeError(f"Stage B checkpoint metadata missing for {run['run_id']}")
        if data.get("checkpoint_sha256") != run["expected_checkpoint_sha256"]:
            # Scientific expected hash is initially null; this branch only detects
            # accidental reuse if a future amended manifest freezes a hash.
            if run["expected_checkpoint_sha256"] is not None:
                raise RuntimeError(f"Stage B checkpoint hash mismatch for {run['run_id']}")
        evals = json.loads(evaluations.read_text())
        if len(evals.get("checkpoints", [])) != 4:
            raise RuntimeError(f"Stage B milestone count mismatch for {run['run_id']}")
        expected_tokens = plan["token_accounting"]["common_milestone_tokens"]
        actual_tokens = [row.get("token_count") for row in evals["checkpoints"]]
        if actual_tokens != expected_tokens:
            raise RuntimeError(f"Stage B milestone token mismatch for {run['run_id']}")
        if "heldout_bpb" not in evals:
            raise RuntimeError(f"Stage B held-out BPB missing for {run['run_id']}")


def run_one(plan: dict, run: dict, *, evaluate: bool = False) -> None:
    model = next(item for item in plan["models"] if item["model_id"] == run["model_id"])
    run_dir = PHASE1_ROOT / "runs" / run["run_id"]
    milestone_dir = run_dir / "milestones"
    checkpoint = run_dir / "final.pt"
    transformed = run_dir / "phase1_train.py"
    metrics = run_dir / "training_metrics.json"
    event_log = run_dir / "events.jsonl"
    if (run_dir / "training_metrics.json").exists():
        raise RuntimeError(f"refusing to reuse an existing Phase 1 run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if event_log:
        env["AUTOPARETO_PHASE1_EVENT_LOG"] = str(event_log)
    compile_cache = run_dir / "compile_cache"
    compile_cache.mkdir(parents=True, exist_ok=True)
    env["TORCHINDUCTOR_CACHE_DIR"] = str(compile_cache / "torchinductor")
    env["TRITON_CACHE_DIR"] = str(compile_cache / "triton")
    env["CUDA_CACHE_PATH"] = str(compile_cache / "cuda")
    # Run the protected launcher as a module from the repository root.  File
    # execution would set sys.path[0] to protected/ and make the namespace
    # package `protected` unavailable to the launcher's top-level imports.
    launcher = command_prefix() + [
        "-m", "protected.phase1_train_launcher",
        "--source", str(ROOT / model["source_path"]),
        "--source-sha256", model["source_sha256"],
        "--token-budget", str(plan["token_budget"]),
        "--seed", str(run["seed"]),
        "--run-dir", str(run_dir),
        "--transformed", str(transformed),
        "--checkpoint", str(checkpoint),
        "--milestone-dir", str(milestone_dir),
        "--metrics", str(metrics),
        "--event-log", str(event_log),
        "--model-id", model["model_id"],
        "--telemetry-interval-seconds", str(plan["telemetry_sampling_interval_seconds"]),
        "--telemetry-mode", str(run.get("telemetry_mode", "enabled")),
    ]
    order_position = run.get("model_order_position")
    if order_position is None and run.get("seed") in plan.get("seeds", []):
        order_position = plan["preregistered_model_order"][str(run["seed"])].index(model["model_id"]) + 1
    if order_position is not None:
        launcher.extend(["--model-order-position", str(order_position)])
    if run.get("retried", False):
        launcher.append("--retried")
    log = run_dir / "training.log"
    with log.open("w") as stream:
        completed = subprocess.run(launcher, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise SystemExit(completed.returncode)

    if evaluate:
        evaluate_one(plan, run)


def evaluate_one(plan: dict, run: dict) -> None:
    model = next(item for item in plan["models"] if item["model_id"] == run["model_id"])
    run_dir = PHASE1_ROOT / "runs" / run["run_id"]
    transformed = run_dir / "phase1_train.py"
    milestone_dir = run_dir / "milestones"
    output = run_dir / "checkpoint_evaluations.json"
    evaluator = command_prefix() + [
        "-m", "scripts.evaluate_phase1_checkpoints",
        "--candidate", str(transformed),
        "--milestone-manifest", str(milestone_dir / "manifest.json"),
        "--output", str(output),
    ]
    if run.get("scientific", False):
        evaluator.append("--heldout")
    subprocess.run(evaluator, cwd=ROOT, check=True)


def inference_one(plan: dict, run: dict) -> None:
    stabilized = load_stabilized_config()
    protocol = stabilized["protocol"]
    workload = stabilized["workload"]
    stability = protocol["stability_policy"]
    run_dir = PHASE1_ROOT / "runs" / run["run_id"]
    transformed = run_dir / "phase1_train.py"
    checkpoint = run_dir / "final.pt"
    inference = command_prefix() + [
        str(ROOT / "deployment_benchmark.py"),
        "--train", str(transformed),
        "--checkpoint", str(checkpoint),
        "--output", str(run_dir / "final_inference.json"),
        "--input-tokens", str(workload["input_tokens"]),
        "--generated-tokens", str(workload["generated_tokens"]),
        "--batch-size", str(workload["batch_size"]),
        "--protocol-version", protocol["version"],
        "--measurement-passes", str(protocol["measurement_passes"]),
        "--repetitions", str(protocol["repetitions_per_pass"]),
        "--warmup-block-size", str(stability["warmup_block_size"]),
        "--max-warmups", str(stability["max_warmups_per_pass"]),
        "--stability-median-tolerance", str(stability["adjacent_median_relative_tolerance"]),
        "--stability-cv-limit", str(stability["within_block_coefficient_of_variation_limit"]),
    ]
    subprocess.run(inference, cwd=ROOT, check=True)


def ordered_runs(plan: dict, *, stage: str, seeds: set[int]) -> list[dict]:
    model_ids = {m["model_id"] for m in plan["models"]}
    runs = [
        r for r in plan["runs"]
        if r["model_id"] in model_ids and r["seed"] in seeds and r["stage"] == stage
    ]
    positions = {
        str(seed): {model_id: index for index, model_id in enumerate(
            plan["preregistered_model_order"][str(seed)], start=1
        )}
        for seed in plan["seeds"]
    }
    runs.sort(key=lambda row: (row["seed"], positions[str(row["seed"])][row["model_id"]]))
    if len(runs) != len(model_ids) * len(seeds):
        raise RuntimeError(f"incomplete staged run matrix for stage {stage}")
    return runs


def _smoke_overhead_artifact(plan: dict, runs: list[dict]) -> Path:
    """Compare telemetry-enabled/control timing using deterministic block bootstrap."""
    import random

    groups: dict[str, dict[str, list[dict]]] = {"baseline": {}, "autopareto_299": {}}
    for run in runs:
        model_id = run["model_id"]
        mode = run["telemetry_mode"]
        metrics_path = PHASE1_ROOT / "runs" / run["run_id"] / "training_metrics.json"
        data = json.loads(metrics_path.read_text())
        groups[model_id].setdefault(mode, []).append({"run": run, "metrics": data})

    result = {
        "format": "autopareto_phase1_smoke_overhead_v2",
        "scientific": False,
        "acceptance": plan["smoke"]["overhead_acceptance"],
        "models": {},
        "status": "pending",
    }
    rng = random.Random(0)
    gate_failures = []
    for model_id, modes in groups.items():
        if set(modes) != {"enabled", "disabled"} or any(len(rows) != 2 for rows in modes.values()):
            raise RuntimeError(f"smoke overhead requires two control and two instrumented runs for {model_id}")
        control_rates = []
        enabled_rates = []
        control_blocks = []
        enabled_blocks = []
        hashes_by_token: dict[str, set[str]] = {}
        for mode, rows in modes.items():
            for row in rows:
                data = row["metrics"]
                rate = float(data["steady_state_training_tokens_per_second"])
                (enabled_rates if mode == "enabled" else control_rates).append(rate)
                run_dir = PHASE1_ROOT / "runs" / row["run"]["run_id"]
                timing_rows = []
                with (run_dir / "step_timing.jsonl").open() as stream:
                    timing_rows = [json.loads(line) for line in stream if line.strip()]
                post = [float(item["step_wall_seconds"]) for item in timing_rows if item["zero_based_step_index"] >= 11]
                block_size = 4 if model_id == "baseline" else 16
                blocks = [post[index:index + block_size] for index in range(0, len(post), block_size)]
                blocks = [block for block in blocks if len(block) == block_size]
                tokens_per_step = next(
                    m["tokens_per_optimizer_step"] for m in plan["models"] if m["model_id"] == model_id
                )
                block_rates = [tokens_per_step * block_size / sum(block) for block in blocks]
                (enabled_blocks if mode == "enabled" else control_blocks).extend(block_rates)
                manifest = json.loads((run_dir / "milestones" / "manifest.json").read_text())
                for checkpoint in manifest["checkpoints"]:
                    checkpoint_path = Path(checkpoint["checkpoint_path"])
                    if not checkpoint_path.is_file():
                        raise RuntimeError(f"smoke checkpoint missing: {checkpoint_path}")
                    if sha256(checkpoint_path) != checkpoint["checkpoint_sha256"]:
                        raise RuntimeError(f"smoke checkpoint hash invalid: {checkpoint_path}")
                    hashes_by_token.setdefault(str(checkpoint["token_count"]), set()).add(checkpoint["checkpoint_sha256"])
        point_overhead = 1.0 - (statistics.median(enabled_rates) / statistics.median(control_rates))
        boot = []
        for _ in range(10000):
            c = [control_blocks[rng.randrange(len(control_blocks))] for _ in control_blocks]
            e = [enabled_blocks[rng.randrange(len(enabled_blocks))] for _ in enabled_blocks]
            boot.append(1.0 - statistics.median(e) / statistics.median(c))
        boot.sort()
        upper = boot[int(0.95 * len(boot))]
        result["models"][model_id] = {
            "control_rates_tokens_per_second": control_rates,
            "enabled_rates_tokens_per_second": enabled_rates,
            "relative_overhead": point_overhead,
            "block_bootstrap_upper_95_overhead": upper,
            "block_count": {"control": len(control_blocks), "enabled": len(enabled_blocks)},
            # Separate retraining runs are not required to be bitwise identical:
            # GPU training can use nondeterministic kernels even with the same
            # seed.  The smoke therefore verifies every hash against its own
            # checkpoint bytes and records cross-run differences diagnostically;
            # it does not turn bitwise reproducibility into an invalid telemetry
            # overhead criterion.
            "checkpoint_hashes_valid": True,
            "checkpoint_hashes_cross_run_identical": all(len(values) == 1 for values in hashes_by_token.values()),
            "checkpoint_hashes_by_token": {key: sorted(values) for key, values in hashes_by_token.items()},
        }
        if abs(point_overhead) > 0.01 or upper > 0.02:
            gate_failures.append(
                f"telemetry overhead gate failed for {model_id}: "
                f"point={point_overhead:.4%} upper={upper:.4%}"
            )
    output = PHASE1_ROOT / "smoke_overhead.json"
    result["status"] = "failed" if gate_failures else "passed"
    result["gate_failures"] = gate_failures
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if gate_failures:
        raise RuntimeError("; ".join(gate_failures))
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "B", "B-inference", "C", "C-inference"), default=None)
    parser.add_argument("--execute", action="store_true", help="required for any GPU work")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    plan = load_plan()
    if args.validate_only or not args.execute:
        print(json.dumps({
            "status": "validated",
            "plan": str(PLAN),
            "message": "No GPU work launched. Pass --stage and --execute only after human authorization.",
        }, sort_keys=True))
        return 0
    if args.stage is None:
        raise RuntimeError("--stage is required with --execute")
    validate_persistent_runtime()
    if args.stage == "smoke":
        smoke_models = [
            next(m for m in plan["models"] if m["model_id"] == model_id)
            for model_id in plan["smoke"]["model_ids"]
        ]
        smoke_plan = dict(plan)
        smoke_plan["token_budget"] = plan["smoke"]["token_budget"]
        smoke_runs = []
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        profile_order = plan["smoke"]["paired_order"][0]
        by_id = {m["model_id"]: m for m in smoke_models}
        with (PHASE1_ROOT / "gpu.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for index, label in enumerate(profile_order, start=1):
                model_id, profile = label.split(":", 1)
                run = {
                    "run_id": f"NON_SCIENTIFIC-smoke-{index:02d}-{model_id}-{profile}-{timestamp}",
                    "model_id": model_id,
                    "seed": 3,
                    "seed_label": "smoke",
                    "scientific": False,
                    "model_order_position": None,
                    "telemetry_mode": "disabled" if profile == "disabled" else "enabled",
                }
                run["source_path"] = by_id[model_id]["source_path"]
                smoke_runs.append(run)
                run_one(smoke_plan, run)
            for run in smoke_runs:
                evaluate_one(smoke_plan, run)
        artifact = _smoke_overhead_artifact(smoke_plan, smoke_runs)
        print(json.dumps({"status": "smoke_passed", "overhead_artifact": str(artifact), "runs": len(smoke_runs)}, sort_keys=True))
        return 0

    if args.stage in {"C", "C-inference"}:
        require_token_stream_verification(plan)
        integrity_gate(plan)
        seeds = {4, 5}
        stage = "C"
    elif args.stage in {"B", "B-inference"}:
        require_token_stream_verification(plan)
        seeds = {3}
        stage = "B"
    else:
        raise RuntimeError(f"unexpected stage: {args.stage}")
    runs = ordered_runs(plan, stage=stage, seeds=seeds)
    lock_path = PHASE1_ROOT / "gpu.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.stage.endswith("-inference"):
            for run in runs:
                inference_one(plan, run)
        else:
            for run in runs:
                run_one(plan, run)
            for run in runs:
                evaluate_one(plan, run)
    print(json.dumps({"status": "completed", "stage": args.stage, "runs": [r["run_id"] for r in runs]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
