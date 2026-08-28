#!/usr/bin/env python3
"""Freeze and validate the Phase 1 equal-token panel and complete run matrix."""

from __future__ import annotations

import ast
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from protected.phase1_source_transform import read_total_batch_size
DB = ROOT / "results/results.db"
OUTPUT = ROOT / "evaluation/phase1_equal_token_preflight.json"
TOKEN_BUDGET = 201_326_592
SEEDS = (3, 4, 5)
MICROBATCH_TOKENS = 64 * 2048
TOTAL_MICROBATCHES = TOKEN_BUDGET // MICROBATCH_TOKENS
MILESTONE_TOKENS = (50_331_648, 100_663_296, 150_994_944, TOKEN_BUDGET)
SMOKE_TOKEN_BUDGET = 16_777_216
# Background NVML/host telemetry is intentionally sparse enough not to perturb
# the short accumulation=1 smoke steps.  The smoke gate remains fixed at 1%
# point / 2% bootstrap upper overhead; this is an infrastructure parameter, not
# a model-recipe hyperparameter.
TELEMETRY_SAMPLING_INTERVAL_SECONDS = 2.0
MEASUREMENT_AMENDMENT_DOCUMENT = "docs/PHASE1_MEASUREMENT_AMENDMENT.md"
PREREGISTERED_MODEL_ORDER = {
    3: ("baseline", "autoresearch_compatible", "tpe_quality", "motpe_quality", "autopareto_292", "autopareto_299", "autopareto_294"),
    4: ("autopareto_292", "autopareto_294", "motpe_quality", "baseline", "autopareto_299", "autoresearch_compatible", "tpe_quality"),
    5: ("autopareto_299", "tpe_quality", "autopareto_294", "autoresearch_compatible", "motpe_quality", "autopareto_292", "baseline"),
}


EXPECTED_SOURCE_HASHES = {
    "baseline": "428e22f114040f9fa9dd2d4da8b1ad005d7aa0be4dd3b395931c4537ba82f6b8",
    "autoresearch_compatible": "b3a961eeea05dcec17c49cb1fa392dae70c1c87d520409da43d4f3fb9fd6c5b1",
    "tpe_quality": "8547b2789c6635c5814ed67c0bb60256bf32bb24af0fcb744604ab59e17c76b2",
    "motpe_quality": "62882a20b11c8c9b6f103f221dfc8621f43013a24c58c14da4ae5909e266e4b8",
    "autopareto_292": "55a7fd8bf32924f2a35ebc095c5206f3503e671a9d7a9646b94979c8ae49259a",
    "autopareto_299": "36e3ad91a62810e4a907c1465adfbd1f07da446d0e63570c9e6a944b6066f12b",
    "autopareto_294": "6d0d8d1d9c7a08eeca1cd8d15b20b0bfdf5d84ee5255f56cd8a8092715290d10",
}


RECIPES = (
    {
        "model_id": "baseline",
        "role": "untouched original train.py baseline",
        "method": "baseline",
        "source_experiment_id": 4,
        "discovery_source_experiment_id": None,
        "source_path": "results/candidates/baseline-a40-seed0/0002-train.py",
        "optional": False,
        "historical_row_id": 4,
        "checkpoint_history_ids": [4],
    },
    {
        "model_id": "autoresearch_compatible",
        "role": "strongest frozen AutoResearch compatible representative",
        "method": "autoresearch",
        "source_experiment_id": 68,
        "discovery_source_experiment_id": None,
        "source_path": "results/candidates/autoresearch-diagnostic5-seed42-20260724/0005-train.py",
        "optional": False,
        "historical_row_id": 68,
        "checkpoint_history_ids": [68, 77],
        "selection_note": (
            "Experiment 68 is the preregistered best-compatible AutoResearch recipe. "
            "Experiment 19 is a separate historical speed reference, not this role."
        ),
    },
    {
        "model_id": "tpe_quality",
        "role": "strongest frozen TPE quality representative",
        "method": "tpe",
        "source_experiment_id": 183,
        "discovery_source_experiment_id": None,
        "source_path": "results/candidates/tpe-seed42-20260805/0021-train.py",
        "optional": False,
        "historical_row_id": 183,
        "checkpoint_history_ids": [183],
    },
    {
        "model_id": "motpe_quality",
        "role": "strongest frozen MOTPE quality representative",
        "method": "motpe",
        "source_experiment_id": 234,
        "discovery_source_experiment_id": None,
        "source_path": "results/candidates/motpe-seed42-20260805/0022-train.py",
        "optional": False,
        "historical_row_id": 234,
        "checkpoint_history_ids": [234, 300],
        "selection_note": (
            "The original seed-42 checkpoint was lost to ephemeral storage; the exact "
            "recipe and measurements are frozen, and recovery experiment 300 is separate."
        ),
    },
    {
        "model_id": "autopareto_292",
        "role": "AutoPareto confirmed-quality recipe corresponding to confirmation 292",
        "method": "autopareto",
        "source_experiment_id": 292,
        "discovery_source_experiment_id": 134,
        "source_path": "results/candidates/autopareto-seed0-20260731/0022-train.py",
        "optional": False,
        "historical_row_id": 292,
        "checkpoint_history_ids": [134, 292],
    },
    {
        "model_id": "autopareto_299",
        "role": "AutoPareto shallow-wide / fast-quality recipe corresponding to confirmation 299",
        "method": "autopareto",
        "source_experiment_id": 299,
        "discovery_source_experiment_id": 137,
        "source_path": "results/candidates/autopareto-seed0-20260731/0025-train.py",
        "optional": False,
        "historical_row_id": 299,
        "checkpoint_history_ids": [137, 299],
    },
    {
        "model_id": "autopareto_294",
        "role": "AutoPareto efficiency-boundary recipe corresponding to confirmation 294",
        "method": "autopareto",
        "source_experiment_id": 294,
        "discovery_source_experiment_id": 136,
        "source_path": "results/candidates/autopareto-seed0-20260731/0024-train.py",
        "optional": False,
        "historical_row_id": 294,
        "checkpoint_history_ids": [136, 294],
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _value(node: ast.AST):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_value(item) for item in node.elts]
    if isinstance(node, ast.BinOp):
        left, right = _value(node.left), _value(node.right)
        return {
            ast.Add: left + right,
            ast.Sub: left - right,
            ast.Mult: left * right,
            ast.FloorDiv: left // right,
            ast.Pow: left ** right,
        }[type(node.op)]
    raise ValueError(f"unsupported recipe constant: {ast.dump(node)}")


def constants(source: str) -> dict:
    wanted = {
        "ASPECT_RATIO", "HEAD_DIM", "WINDOW_PATTERN", "KV_HEAD_RATIO",
        "TOTAL_BATCH_SIZE", "EMBEDDING_LR", "UNEMBEDDING_LR", "MATRIX_LR",
        "SCALAR_LR", "WEIGHT_DECAY", "ADAM_BETAS", "WARMUP_RATIO",
        "WARMDOWN_RATIO", "FINAL_LR_FRAC", "MUON_NS_STEPS", "DEPTH",
        "DEVICE_BATCH_SIZE",
    }
    values = {}
    for statement in ast.parse(source).body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    values[target.id] = _value(statement.value)
    if "MUON_NS_STEPS" not in values:
        match = re.search(r"ns_steps\s*=\s*(\d+)", source)
        values["MUON_NS_STEPS"] = int(match.group(1)) if match else 5
    return values


def architecture(values: dict) -> dict:
    width = ((values["DEPTH"] * values["ASPECT_RATIO"] + values["HEAD_DIM"] - 1) // values["HEAD_DIM"]) * values["HEAD_DIM"]
    query_heads = width // values["HEAD_DIM"]
    if "KV_HEAD_RATIO" in values:
        target = max(1, round(query_heads / values["KV_HEAD_RATIO"]))
        kv_heads = max(
            candidate for candidate in range(1, query_heads + 1)
            if query_heads % candidate == 0 and candidate <= target
        )
    else:
        # Frozen AutoPareto 136/137/134 source explicitly sets n_kv_head=n_head.
        kv_heads = query_heads
    return {
        "depth": values["DEPTH"],
        "width": width,
        "query_heads": query_heads,
        "kv_heads": kv_heads,
        "head_dimension": values["HEAD_DIM"],
    }


def checkpoint_provenance(db: sqlite3.Connection, ids: list[int]) -> list[dict]:
    rows = []
    for experiment_id in ids:
        row = db.execute(
            "SELECT id, campaign_id, experiment_number, checkpoint_path FROM experiments WHERE id=?",
            (experiment_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"missing checkpoint provenance experiment {experiment_id}")
        db_path = str(row["checkpoint_path"] or "")
        relative = db_path.replace("/workspace/autopareto/", "")
        local = ROOT / relative if relative else None
        if not local or not local.is_file():
            fallback = ROOT / "results" / "checkpoints" / row["campaign_id"] / f"{int(row['experiment_number']):04d}.pt"
            if fallback.is_file():
                local = fallback
        rows.append({
            "experiment_id": int(row["id"]),
            "campaign_id": row["campaign_id"],
            "experiment_number": int(row["experiment_number"]),
            "database_checkpoint_path": db_path or None,
            "local_checkpoint_path": str(local) if local and local.is_file() else None,
            "checkpoint_available_locally": bool(local and local.is_file()),
            "checkpoint_bytes": local.stat().st_size if local and local.is_file() else None,
            "checkpoint_sha256": sha256(local) if local and local.is_file() else None,
        })
    return rows


def row_metrics(db: sqlite3.Connection, experiment_id: int) -> dict:
    row = db.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone()
    if row is None:
        raise RuntimeError(f"missing source experiment {experiment_id}")
    fields = (
        "val_bpb", "training_tokens_per_second", "total_training_tokens", "training_seconds",
        "total_runtime_seconds", "parameter_count", "median_cached_decode_tokens_per_second",
        "peak_inference_cuda_memory_mb", "peak_reserved_inference_cuda_memory_mb",
        "incremental_kv_cache_memory_mb", "inference_metric_mode", "program_sha256",
        "git_commit", "checkpoint_path", "campaign_id", "training_seed",
    )
    return {field: row[field] for field in fields}


def build() -> dict:
    db = sqlite3.connect(DB)
    db.row_factory = sqlite3.Row
    models = []
    for recipe in RECIPES:
        source = ROOT / recipe["source_path"]
        if not source.is_file():
            raise FileNotFoundError(source)
        source_hash = sha256(source)
        if source_hash != EXPECTED_SOURCE_HASHES[recipe["model_id"]]:
            raise RuntimeError(f"frozen source hash mismatch for {recipe['model_id']}: {source_hash}")
        source_text = source.read_text()
        expected_loader_call = 'make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")'
        if source_text.count(expected_loader_call) != 1:
            raise RuntimeError(f"dataloader call mismatch for {recipe['model_id']}")
        values = constants(source_text)
        batch = read_total_batch_size(source_text)
        if values["TOTAL_BATCH_SIZE"] != batch:
            raise RuntimeError(f"batch parser mismatch for {recipe['model_id']}")
        grad_accum = batch // (values["DEVICE_BATCH_SIZE"] * 2048)
        if batch % (values["DEVICE_BATCH_SIZE"] * 2048):
            raise RuntimeError(f"microbatch does not divide total batch for {recipe['model_id']}")
        if TOKEN_BUDGET % batch:
            raise RuntimeError(f"token budget is not exact for {recipe['model_id']}")
        metrics = row_metrics(db, recipe["historical_row_id"])
        if metrics["parameter_count"] is None:
            raise RuntimeError(f"missing parameter count for {recipe['model_id']}")
        model = {
            **recipe,
            "source_sha256": source_hash,
            "source_bytes": source.stat().st_size,
            "architecture": architecture(values),
            "parameter_count": int(metrics["parameter_count"]),
            "device_batch_size": int(values["DEVICE_BATCH_SIZE"]),
            "sequence_length": 2048,
            "tokens_per_microbatch": int(values["DEVICE_BATCH_SIZE"] * 2048),
            "gradient_accumulation": int(grad_accum),
            "tokens_per_optimizer_step": int(batch),
            "optimizer": "MuonAdamW (Muon matrix groups + AdamW embedding/unembedding/scalars)",
            "recipe_constants": values,
            "schedule_audit": {
                "historical_progress_basis": "total_training_time / TIME_BUDGET",
                "historical_time_budget_seconds": 300,
                "phase1_progress_basis": "optimizer_step * TOTAL_BATCH_SIZE / TOKEN_BUDGET",
                "transformation": "same warmup/warmdown/final fraction shape over token progress; no ratio changes",
                "muon_momentum_basis": "optimizer step / 300, unchanged because it is not wall-clock LR progress",
                "weight_decay_basis": "same translated token progress as LR multiplier",
            },
            "historical_five_minute": metrics,
            "checkpoint_provenance": checkpoint_provenance(db, recipe["checkpoint_history_ids"]),
        }
        models.append(model)

    models_by_id = {model["model_id"]: model for model in models}
    runs = []
    for seed in SEEDS:
        for position, model_id in enumerate(PREREGISTERED_MODEL_ORDER[seed], start=1):
            model = models_by_id[model_id]
            label = f"seed{seed}"
            runs.append({
                "run_id": f"phase1-{model['model_id']}-{label}",
                "model_id": model["model_id"],
                "seed": seed,
                "seed_label": label,
                "model_order_position": position,
                "stage": "B" if seed == 3 else "C",
                "scientific": True,
                "retried": False,
                "expected_checkpoint_sha256": None,
            })
    source_artifacts = [
        "results/discovery-freeze-20260810.json",
        "results/autopareto_primary_confirmation_plan.json",
        "results/autopareto_primary_confirmation_analysis.json",
        "results/autopareto_confirmation_selection.json",
        "evaluation/seed42_case_study.json",
        "evaluation/tpe_seed42_case_study.json",
        "evaluation/motpe_seed42_case_study.json",
        "results/downstream_selection.json",
        "evaluation/phase1_token_stream_audit.json",
    ]
    implementation = {
        "source_transform": "protected/phase1_source_transform.py",
        "runtime_recorder": "protected/phase1_runtime.py",
        "token_launcher": "protected/phase1_train_launcher.py",
        "staged_executor": "scripts/run_phase1_equal_tokens.py",
        "checkpoint_evaluator": "scripts/evaluate_phase1_checkpoints.py",
        "preflight_generator": "scripts/prepare_phase1_equal_tokens.py",
        "token_stream_verifier": "scripts/verify_phase1_token_stream.py",
        "transform_policy": "hash-verify frozen source, derive execution-only source, preserve original source bytes",
    }
    return {
        "format": "autopareto_phase1_equal_token_preflight_v3",
        "status": "PREPARING / PREFLIGHT",
        "prepared_on": str(date.today()),
        "scientific_question": (
            "When representative baseline, AutoResearch, TPE, MOTPE, and AutoPareto "
            "recipes process exactly the same training tokens, compare quality and A40 time."
        ),
        "scope": {
            "gpu": "one NVIDIA A40",
            "training_jobs_launched": False,
            "gpu_compute_spent": False,
            "scientific_recipe_changes": False,
            "wall_clock_schedule_translation": True,
            "fixed_wall_clock_comparison_remains_valid": True,
        },
        "token_budget": TOKEN_BUDGET,
        "token_accounting": {
            "sequence_length": 2048,
            "tokens_per_microbatch": MICROBATCH_TOKENS,
            "total_microbatches": TOTAL_MICROBATCHES,
            "common_milestone_tokens": list(MILESTONE_TOKENS),
            "milestone_rule": "LCM-compatible common token counts; exact for both 524288 and 131072 batches",
        },
        "smoke": {
            "scientific": False,
            "label_prefix": "NON-SCIENTIFIC",
            "model_ids": ["baseline", "autopareto_299"],
            "token_budget": SMOKE_TOKEN_BUDGET,
            "regimes": {
                "baseline": {
                    "gradient_accumulation": 4,
                    "tokens_per_optimizer_step": 524_288,
                    "optimizer_steps": 32,
                },
                "autopareto_299": {
                    "gradient_accumulation": 1,
                    "tokens_per_optimizer_step": 131_072,
                    "optimizer_steps": 128,
                },
            },
            "milestone_count": 4,
            "profiles": ["disabled_control", "enabled_instrumented"],
            "paired_order": [
                ["baseline:disabled", "autopareto_299:enabled", "baseline:enabled", "autopareto_299:disabled",
                 "baseline:enabled", "autopareto_299:disabled", "baseline:disabled", "autopareto_299:enabled"],
            ],
            "overhead_acceptance": {
                "median_relative_overhead_limit": 0.01,
                "block_bootstrap_upper_limit": 0.02,
                "checkpoint_hashes_must_validate": True,
            },
            "excluded_from_scientific_analysis": True,
        },
        "data_equivalence": {
            "status": "ANALYTICALLY_ESTABLISHED_RUNTIME_HASH_REQUIRED",
            "exact_sequence_for_fixed_cache": True,
            "claim_scope": "same exact token-ID microbatch sequence for all seven models at each fixed seed, conditional on identical tokenizer/data cache bytes",
            "loader_path": "prepare.py",
            "loader_function": "make_dataloader",
            "loader_sha256_at_preflight": sha256(ROOT / "prepare.py"),
            "loader_call_signature": 'make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")',
            "loader_settings": {
                "device_batch_size": 64,
                "sequence_length": 2048,
                "microbatch_tokens": MICROBATCH_TOKENS,
                "microbatches_to_target": TOTAL_MICROBATCHES,
                "baseline_gradient_accumulation": 4,
                "other_gradient_accumulation": 1,
            },
            "determinism_basis": [
                "sorted training shard paths excluding pinned validation shard",
                "deterministic parquet row-group and row order",
                "deterministic tokenizer encoding of each document batch",
                "deterministic best-fit packing and shortest-document cropping",
                "no random operation or seed-dependent branch in make_dataloader",
                "all seven frozen sources use the identical loader call and defaults",
            ],
            "seed_effect": "seed changes model initialization and training RNG only; loader sequence is seed-independent",
            "microbatch_boundary_proof": "baseline consumes 4 consecutive microbatches per optimizer step; the other six consume 1; 384*4 = 1536*1 = 1536",
            "verification_script": "scripts/verify_phase1_token_stream.py",
            "runtime_hash_output": "results/phase1/token_stream_verification.json",
            "runtime_hashes_observed": False,
        },
        "analysis_policy": {
            "primary_quality_metric": "final held-out BPB at exactly 201326592 training tokens",
            "primary_training_efficiency_metric": "steady_state_equivalent_seconds for exactly 201326592 tokens, computed from synchronized post-step-11 throughput",
            "primary_deployment_metric": "final protected cached-decode throughput under the frozen headline workload/protocol",
            "secondary_training_time_metric": "total_scientific_step_wall_seconds, the observed sum from first through final optimizer step",
            "supporting_metrics": [
                "validation BPB at the four registered token milestones",
                "allocated and reserved VRAM",
                "incremental KV-cache memory",
                "cached-decode correctness",
                "time-to-quality only for a pre-declared threshold with no post-hoc cherry-picking",
            ],
            "amendment_rule": "do not change primary outcomes after seed-3 results without a recorded protocol amendment",
            "seed_aggregation": "report every seed, arithmetic mean, sample SD, and paired candidate-minus-baseline differences; no successful-run outlier deletion",
            "deployment_gate": "inference is a separate gate; unstable inference does not invalidate training or BPB, but deployment throughput remains unavailable",
        },
        "timing": {
            "duration_clock": "time.perf_counter_ns()",
            "steady_state_start_step_index": 11,
            "steady_state_boundary_text": "Steps 0–10 are scientifically executed but excluded from the historical steady-state timing boundary. Steady-state timing begins at step index 11.",
            "total_scientific_step_wall_seconds": "sum of synchronized per-step monotonic durations for all scientific optimizer steps",
            "steady_state_training_seconds": "sum of synchronized per-step monotonic durations for step indices 11 through final",
            "steady_state_tokens": "(final_step_count - 11) * tokens_per_optimizer_step",
            "steady_state_training_tokens_per_second": "steady_state_tokens / steady_state_training_seconds",
            "steady_state_equivalent_seconds": "201326592 / steady_state_training_tokens_per_second",
            "scientific_interval_seconds": "continuous monotonic interval from first step start through final step end; may include between-step bookkeeping",
            "cuda_stream_step_seconds": "diagnostic CUDA-event duration; does not include all CPU work or checkpoint I/O",
            "checkpoint_save_timing": "checkpoint writes occur after the synchronized per-step timing boundary and are excluded from total_scientific_step_wall_seconds",
            "end_to_end_seconds": "process monotonic runtime from launcher start through final training contract; evaluator runtimes are separate files",
            "common_token_boundary": "supporting sensitivity boundary excludes 5767168 tokens: 11 baseline optimizer steps or 44 accumulation-1 optimizer steps",
            "cuda_event_policy": "CUDA events disabled in scientific timing because per-step event instrumentation has unequal overhead; diagnostic-only artifacts may use events",
        },
        "measurement_amendment": {
            "version": "autopareto_phase1_measurement_infrastructure_v3",
            "scientific_execution_unchanged": True,
            "frozen_source_files_unchanged": True,
            "steady_state_start_step_index": 11,
            "telemetry_sampling_interval_seconds": TELEMETRY_SAMPLING_INTERVAL_SECONDS,
            "telemetry_implementation": "background NVML/host sampler outside timed critical path; sampler never imports torch or calls CUDA runtime APIs",
            "telemetry_calls_cuda_synchronize": False,
            "telemetry_calls_cuda_runtime": False,
            "scientific_cuda_events": False,
            "smoke_excluded_from_scientific_analysis": True,
        },
        "preregistered_model_order": {str(seed): list(order) for seed, order in PREREGISTERED_MODEL_ORDER.items()},
        "telemetry_sampling_interval_seconds": TELEMETRY_SAMPLING_INTERVAL_SECONDS,
        "seeds": list(SEEDS),
        "seed_audit": {
            "autopareto_selection_seeds": [0, 1, 2, 42],
            "phase1_seeds_used_to_select_autopareto_recipes": [],
            "database_autopareto_rows_on_phase1_seeds": 0,
            "verified_fresh_for_recipe_selection": True,
        },
        "panel_policy": {
            "core_model_count": 7,
            "core_scientific_runs": 21,
            "optional_model": None,
            "optional_total_runs": 0,
            "all_seven_models_frozen_before_stage_B": True,
            "no_treatment_changes_after_seed3": True,
        },
        "source_artifacts": {
            "database": "results/results.db",
            "database_sha256_at_preflight": sha256(DB),
            "artifact_sha256": {path: sha256(ROOT / path) for path in source_artifacts},
        },
        "documentation_sha256": {
            MEASUREMENT_AMENDMENT_DOCUMENT: sha256(ROOT / MEASUREMENT_AMENDMENT_DOCUMENT)
            if (ROOT / MEASUREMENT_AMENDMENT_DOCUMENT).is_file()
            else None,
            "docs/PHASE1_EQUAL_TOKEN_PREFLIGHT.md": sha256(ROOT / "docs/PHASE1_EQUAL_TOKEN_PREFLIGHT.md"),
        },
        "models": models,
        "runs": runs,
        "implementation": implementation,
        "implementation_sha256": {
            path: sha256(ROOT / path)
            for path in implementation.values()
            if path.endswith(".py")
        },
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    payload = build()
    if args.validate_only:
        print(json.dumps({"status": "ok", "models": len(payload["models"]), "runs": len(payload["runs"])}, sort_keys=True))
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "written", "path": str(args.output), "models": len(payload["models"]), "runs": len(payload["runs"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
