#!/usr/bin/env python3
"""Phase 5A executor: the three Opus frontier recipes at 201,326,592 tokens, seeds 6-8, paired with E1.

    python scripts/run_phase5a_opus_fixed_tokens.py                                   # validate only (default)
    python scripts/run_phase5a_opus_fixed_tokens.py --stage train --execute           # on the A40 pod
    python scripts/run_phase5a_opus_fixed_tokens.py --stage downstream --execute      # after train

The pre-registration is ``evaluation/PHASE5_PREREG.md``, with the machine-readable plan in
``evaluation/phase5a_opus_fixed_tokens_plan.json``. This file is deliberately thin:

* Training calls the E1 executor's code path unchanged: ``run_one`` from
  ``scripts/run_phase1_equal_tokens.py`` with a per-run plan view. That keeps the
  protected launcher, the source transform, the runtime recorders and the seed
  forcing byte for byte. Each Opus recipe's source is the committed output of
  ``scripts/opus_contract_t_to_phase1_source.py``, re-derived and byte-compared
  before launch.
* Each Opus run then exports deployed checkpoints
  (``scripts/phase5a_export_deployed_checkpoints.py``) and scores them with the
  unchanged Phase-1 checkpoint evaluator, which also scores held-out BPB.
* Every run is gated by the E1 executor's ``gate_token_run``. The one check it
  cannot apply, the manifest hash (the evaluator scores the deployed manifest),
  is replaced by explicit provenance checks on the deployed manifest.
* The drift control (AR-ref, 201.3M tokens, seed 6) runs through the E1
  executor's ``run_with_retry`` with the E1 plan, exactly as in E1.

Nothing under ``results/camera_ready_2026/amendment1`` or ``results/phase1`` is written.
"""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PLAN_PATH = ROOT / "evaluation/phase5a_opus_fixed_tokens_plan.json"
PLAN_FORMAT = "autopareto_phase5a_opus_fixed_tokens_preregistration_v1"


def load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


e1 = load_module("phase5a_e1_executor", ROOT / "scripts/run_camera_ready_e0_e1.py")
mapping = load_module("phase5a_opus_mapping", ROOT / "scripts/opus_contract_t_to_phase1_source.py")
phase1 = e1.phase1
sha256, read_json, write_json_new, append_jsonl, utc_now = e1.sha256, e1.read_json, e1.write_json_new, e1.append_jsonl, e1.utc_now


def load_plan() -> tuple[dict, str]:
    raw = PLAN_PATH.read_bytes()
    plan = json.loads(raw)
    if plan.get("format") != PLAN_FORMAT:
        raise RuntimeError(f"unexpected plan format {plan.get('format')}")
    return plan, hashlib.sha256(raw).hexdigest()


def runs_in_order(plan: dict) -> list[dict]:
    return sorted(plan["runs"], key=lambda row: row["order"])


def stage_root(plan: dict) -> Path:
    root = (ROOT / plan["output_root"]).resolve()
    for forbidden in (ROOT / "results/phase1", ROOT / "results/camera_ready_2026/amendment1"):
        forbidden = forbidden.resolve()
        if root == forbidden or forbidden in root.parents or root in forbidden.parents:
            raise RuntimeError(f"output root overlaps protected evidence: {root}")
    return root


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(plan: dict, plan_sha: str) -> list[dict]:
    rows: list[dict] = []

    def add(name: str, ok: bool, detail) -> None:
        rows.append({"check": name, "status": "ok" if ok else "FAIL", "detail": detail})

    e1_plan, e1_sha = e1.load_plan()
    add("e1_plan_sha256", e1_sha.startswith(plan["comparator_plan"]["sha256_prefix"]), e1_sha)
    mismatched = {
        rel: sha256(ROOT / rel) if (ROOT / rel).is_file() else None
        for rel, expected in plan["implementation_sha256"].items()
        if not (ROOT / rel).is_file() or sha256(ROOT / rel) != expected
    }
    add("phase5a_implementation_sha256", not mismatched, mismatched or f"{len(plan['implementation_sha256'])} files match")
    problems = []
    for key, recipe in plan["recipes"].items():
        original = ROOT / recipe["pair_a_source_path"]
        mapped = ROOT / recipe["source_path"]
        if sha256(original) != recipe["pair_a_source_sha256"]:
            problems.append(f"{key}: Pair-A source hash differs")
            continue
        if sha256(mapped) != recipe["source_sha256"]:
            problems.append(f"{key}: mapped source hash differs")
            continue
        if mapping.adapt(original.read_text()) != mapped.read_text():
            problems.append(f"{key}: mapping no longer reproduces the committed mapped source")
        if e1.read_total_batch_size(mapped.read_text()) != plan["TOTAL_BATCH_SIZE"]:
            problems.append(f"{key}: TOTAL_BATCH_SIZE differs from the plan")
        for seed in plan["seeds"]:
            reason = mapping._transform_accepts(mapped.read_text(), plan["token_budget"], seed)
            if reason:
                problems.append(f"{key} seed {seed}: transform refuses ({reason})")
    add("opus_sources_mapping_and_transform", not problems, problems or "3 recipes: hashes, byte-exact re-derivation, transform accepts seeds 6-8")
    e1_root = (ROOT / e1_plan["output_root"]).resolve() / "E1" / "runs"
    comparators = []
    for row in plan["comparator_runs"]:
        record = read_json(e1_root / row["run_id"] / e1.RECORD_NAME) or {}
        ok = (
            record.get("integrity", {}).get("passed") is True
            and record.get("seed") == row["seed"]
            and record.get("token_budget") == plan["token_budget"]
            and (record.get("metrics") or {}).get("checkpoint_sha256") == row["checkpoint_sha256"]
        )
        if not ok:
            comparators.append(row["run_id"])
    add("e1_comparator_runs", not comparators, comparators or f"{len(plan['comparator_runs'])} E1 201.3M runs passed, checkpoint hashes match")
    return rows


def environment_checks(e1_plan: dict, e1_sha: str) -> tuple[list[str], dict]:
    checks, info = e1.validate(e1_plan, e1_sha, strict=True, launcher_override=None)
    return checks.failures, {"checks": checks.rows, "environment": info}


# ---------------------------------------------------------------------------
# Opus runs
# ---------------------------------------------------------------------------

def opus_views(plan: dict, run: dict, dirname: str, attempt: int) -> tuple[dict, dict, dict]:
    recipe = plan["recipes"][run["recipe"]]
    plan_view = {
        "models": [{
            "model_id": recipe["phase1_model_id"],
            "source_path": recipe["source_path"],
            "source_sha256": recipe["source_sha256"],
        }],
        "token_budget": plan["token_budget"],
        "telemetry_sampling_interval_seconds": plan["telemetry_sampling_interval_seconds"],
        "seeds": [],
        "preregistered_model_order": {},
    }
    run_view = {
        "run_id": dirname,
        "model_id": recipe["phase1_model_id"],
        "seed": run["seed"],
        "telemetry_mode": "enabled",
        "model_order_position": run["order"],
        "retried": attempt > 1,
        "scientific": True,
    }
    e1_plan, _ = e1.load_plan()
    gate_view = {
        "batch": {"TOTAL_BATCH_SIZE": plan["TOTAL_BATCH_SIZE"], "DEVICE_BATCH_SIZE": plan["DEVICE_BATCH_SIZE"]},
        "cells": {run["cell"]: e1_plan["cells"][run["cell"]]},
        "recipes": {run["recipe"]: {"source_sha256": recipe["source_sha256"], "phase1_model_id": recipe["phase1_model_id"]}},
        "data_identity_gate": e1_plan["data_identity_gate"],
        "integrity_gates": e1_plan["integrity_gates"],
        "execution": {"token_cells": {"telemetry_sampling_interval_seconds": plan["telemetry_sampling_interval_seconds"]}},
        "implementation_sha256": {**e1_plan["implementation_sha256"], **plan["implementation_sha256"]},
    }
    return plan_view, run_view, gate_view


def deployed_provenance_failures(run_dir: Path, evals: dict | None) -> list[str]:
    failures = []
    training_manifest = run_dir / "milestones" / "manifest.json"
    deployed_manifest = run_dir / "deployed" / "manifest.json"
    report = read_json(run_dir / "deployed" / "deployment_report.json")
    deployed = read_json(deployed_manifest)
    training = read_json(training_manifest)
    if not report or not deployed or not training:
        return ["deployment report or manifests missing"]
    if report.get("failures"):
        failures.extend(f"deployment: {item}" for item in report["failures"])
    if deployed["deployment"]["training_manifest_sha256"] != sha256(training_manifest):
        failures.append("deployed manifest does not point at this run's training manifest")
    pairs = list(zip(training["checkpoints"], deployed["checkpoints"]))
    if len(pairs) != len(training["checkpoints"]) or len(deployed["checkpoints"]) != len(training["checkpoints"]):
        failures.append("deployed manifest row count differs")
    for train_row, dep_row in pairs:
        if dep_row.get("training_checkpoint_sha256") != train_row.get("checkpoint_sha256") or dep_row.get("token_count") != train_row.get("token_count"):
            failures.append(f"deployed row {dep_row.get('token_count')} does not derive from the training milestone")
        path = Path(dep_row["checkpoint_path"])
        if not path.is_file() or sha256(path) != dep_row["checkpoint_sha256"]:
            failures.append(f"deployed checkpoint {path.name} missing or hash differs")
    if evals is None or evals.get("milestone_manifest_sha256") != sha256(deployed_manifest):
        failures.append("evaluator did not score the deployed manifest")
    rows = (evals or {}).get("checkpoints", [])
    if [r.get("checkpoint_sha256") for r in rows] != [r["checkpoint_sha256"] for r in deployed["checkpoints"]]:
        failures.append("evaluated checkpoints are not the deployed checkpoints")
    return failures


def execute_opus_attempt(plan: dict, root: Path, run: dict, attempt: int) -> tuple[list[str], dict]:
    dirname = e1.attempt_dirname(run, attempt)
    run_dir = root / "runs" / dirname
    if run_dir.exists():
        raise RuntimeError(f"refusing to reuse run directory {run_dir}")
    recipe = plan["recipes"][run["recipe"]]
    plan_view, run_view, gate_view = opus_views(plan, run, dirname, attempt)
    failures: list[str] = []
    with e1.phase1_redirected(root):
        try:
            phase1.run_one(plan_view, run_view)
        except SystemExit as exc:
            failures.append(f"Phase-1 launcher exited with {exc.code} (see training.log)")
    if not failures:
        export = phase1.command_prefix() + [
            str(ROOT / "scripts/phase5a_export_deployed_checkpoints.py"),
            "--candidate", str(run_dir / "phase1_train.py"),
            "--original-source", str(ROOT / recipe["pair_a_source_path"]),
            "--milestone-manifest", str(run_dir / "milestones" / "manifest.json"),
            "--output-dir", str(run_dir / "deployed"),
        ]
        with (run_dir / "deployment.log").open("w") as stream:
            done = subprocess.run(export, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if done.returncode:
            failures.append(f"deployment export exited with {done.returncode}")
    if not failures:
        evaluator = phase1.command_prefix() + [
            "-m", "scripts.evaluate_phase1_checkpoints",
            "--candidate", str(run_dir / "phase1_train.py"),
            "--milestone-manifest", str(run_dir / "deployed" / "manifest.json"),
            "--output", str(run_dir / "checkpoint_evaluations.json"),
            "--heldout",
        ]
        with (run_dir / "evaluation.log").open("w") as stream:
            done = subprocess.run(evaluator, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if done.returncode:
            failures.append(f"Phase-1 evaluator exited with {done.returncode}")
    gate_failures, values = e1.gate_token_run(gate_view, run, run_dir)
    expected_substitution = "evaluator milestone manifest hash differs"
    gate_failures = [item for item in gate_failures if item != expected_substitution]
    if not failures:
        gate_failures.extend(deployed_provenance_failures(run_dir, read_json(run_dir / "checkpoint_evaluations.json")))
        deployed = read_json(run_dir / "deployed" / "manifest.json") or {"checkpoints": [{}]}
        values["deployed_final_checkpoint_sha256"] = deployed["checkpoints"][-1].get("checkpoint_sha256")
        values["deployed_final_checkpoint_path"] = deployed["checkpoints"][-1].get("checkpoint_path")
    return failures + gate_failures, values


def write_record(plan: dict, plan_sha: str, root: Path, run: dict, attempt: int, failures: list[str], values: dict, started: float, source_sha: str, model_id: str) -> dict:
    dirname = e1.attempt_dirname(run, attempt)
    record = {
        "format": "autopareto_phase5a_run_record_v1",
        "synthetic": False,
        "plan_sha256": plan_sha,
        "stage": "P5A",
        "run_id": run["run_id"],
        "attempt": attempt,
        "run_dir": str(root / "runs" / dirname),
        "order": run["order"],
        "recipe": run["recipe"],
        "role": run["role"],
        "phase1_model_id": model_id,
        "source_sha256": source_sha,
        "cell": run["cell"],
        "contract": "fixed_tokens",
        "token_budget": plan["token_budget"],
        "seed": run["seed"],
        "status": "completed" if not failures else "failed",
        "integrity": {"passed": not failures, "failures": failures},
        "metrics": values,
        "attempt_wall_seconds": time.time() - started,
        "finished_utc": utc_now(),
    }
    write_json_new(root / "runs" / dirname / "phase5a_run_record.json", record)
    return record


def run_with_retry(plan: dict, plan_sha: str, root: Path, run: dict, events: Path) -> dict | None:
    e1_plan, _ = e1.load_plan()
    for attempt in (1, 2):
        run_dir = root / "runs" / e1.attempt_dirname(run, attempt)
        if run_dir.exists():
            record = read_json(run_dir / "phase5a_run_record.json")
            if record and record["integrity"]["passed"]:
                append_jsonl(events, {"utc": utc_now(), "event": "resume_skip_completed", "run_id": run["run_id"], "attempt": attempt})
                return record
            continue  # an interrupted or failed attempt counts toward the single retry
        e1.assert_gpu_idle()
        started = time.time()
        append_jsonl(events, {"utc": utc_now(), "event": "attempt_started", "run_id": run["run_id"], "attempt": attempt})
        if run["role"] == "drift_control":
            e1_run = {k: run[k] for k in ("run_id", "order", "recipe", "cell", "seed")} | {"stage": "P5A"}
            failures, values = e1.execute_token_attempt(e1_plan, root, e1_run, attempt)
            recipe = e1_plan["recipes"][run["recipe"]]
            source_sha, model_id = recipe["source_sha256"], recipe["phase1_model_id"]
        else:
            failures, values = execute_opus_attempt(plan, root, run, attempt)
            recipe = plan["recipes"][run["recipe"]]
            source_sha, model_id = recipe["source_sha256"], recipe["phase1_model_id"]
        record = write_record(plan, plan_sha, root, run, attempt, failures, values, started, source_sha, model_id)
        append_jsonl(events, {"utc": utc_now(), "event": "attempt_finished", "run_id": run["run_id"], "attempt": attempt, "status": record["status"], "failures": failures})
        print(json.dumps({"run_id": run["run_id"], "attempt": attempt, "status": record["status"], "failures": failures[:5]}, sort_keys=True), flush=True)
        if record["integrity"]["passed"]:
            return record
    append_jsonl(events, {"utc": utc_now(), "event": "run_reported_failed", "run_id": run["run_id"]})
    return None


# ---------------------------------------------------------------------------
# Downstream suite (frozen evaluator, new database)
# ---------------------------------------------------------------------------

DOWNSTREAM_SCHEMA = """CREATE TABLE experiments (
    id INTEGER PRIMARY KEY, campaign_id TEXT, experiment_number INTEGER,
    method TEXT, training_seed INTEGER, status TEXT, description TEXT,
    git_commit TEXT, code_diff TEXT, val_bpb REAL,
    inference_tokens_per_second REAL, peak_inference_cuda_memory_mb REAL,
    training_tokens_per_second REAL, total_training_tokens INTEGER,
    peak_training_cuda_memory_mb REAL, parameter_count INTEGER,
    training_seconds REAL, inference_seconds REAL, total_runtime_seconds REAL,
    checkpoint_path TEXT, sample_path TEXT, log_path TEXT,
    crash_or_failure_reason TEXT, protected_manifest_sha256 TEXT,
    started_at TEXT, finished_at TEXT, phase TEXT, valid_result INTEGER,
    exclude_from_analysis INTEGER, UNIQUE(campaign_id, experiment_number))"""


def downstream_models(plan: dict, root: Path) -> list[dict]:
    e1_plan, _ = e1.load_plan()
    models = []
    for run in runs_in_order(plan):
        if run["role"] == "drift_control":
            continue
        record = None
        for attempt in (1, 2):
            record = read_json(root / "runs" / e1.attempt_dirname(run, attempt) / "phase5a_run_record.json")
            if record and record["integrity"]["passed"]:
                break
            record = None
        if record is None:
            continue
        recipe = plan["recipes"][run["recipe"]]
        models.append({
            "method": f"{recipe['phase1_model_id']}_tokens201M_seed{run['seed']}",
            "seed": run["seed"],
            "candidate": str(ROOT / recipe["pair_a_source_path"]),
            "checkpoint": record["metrics"]["deployed_final_checkpoint_path"],
            "checkpoint_sha256": record["metrics"]["deployed_final_checkpoint_sha256"],
            "note": f"Phase 5A matched tokens 201.3M (Phase-1 harness, deployed checkpoint), {run['recipe']}",
        })
    e1_root = (ROOT / e1_plan["output_root"]).resolve() / "E1" / "runs"
    for row in plan["comparator_runs"]:
        recipe = e1_plan["recipes"][row["recipe"]]
        models.append({
            "method": f"{recipe['phase1_model_id']}_tokens201M_seed{row['seed']}",
            "seed": row["seed"],
            "candidate": str(ROOT / recipe["source_path"]),
            "checkpoint": str(e1_root / row["run_id"] / "final.pt"),
            "checkpoint_sha256": row["checkpoint_sha256"],
            "note": f"E1 matched tokens 201.3M ({row['run_id']}), {row['recipe']}",
        })
    return models


def execute_downstream(plan: dict, plan_sha: str, root: Path) -> int:
    spec = plan["downstream"]
    for rel, expected in (("downstream_evaluator.py", spec["evaluator_sha256"]), ("evaluation/downstream_suite.json", spec["suite_sha256"])):
        if sha256(ROOT / rel) != expected:
            print(json.dumps({"status": "refused", "reason": f"{rel} hash differs"}))
            return 2
    out = root / "downstream"
    db_path = out / "phase5a_downstream.db"
    if db_path.exists():
        print(json.dumps({"status": "refused", "reason": f"{db_path} exists"}))
        return 2
    models = downstream_models(plan, root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "runs").mkdir(exist_ok=True)
    from autopareto_db import migrate

    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    db.execute(DOWNSTREAM_SCHEMA)
    db.execute("CREATE TABLE campaign_state (campaign_id TEXT PRIMARY KEY, method TEXT, status TEXT, attempt_limit INTEGER, next_experiment_number INTEGER)")
    migrate(db)
    for index, model in enumerate(models, start=1):
        db.execute(
            """INSERT INTO experiments (id, campaign_id, experiment_number, method, training_seed, status,
               description, git_commit, code_diff, checkpoint_path, phase, training_status, inference_status)
               VALUES (?, 'phase5a-downstream-20261108', ?, ?, ?, 'completed', ?, 'phase5a', '', ?, 'phase5a', 'completed', 'compatible')""",
            (index, index, model["method"], model["seed"], model["note"] + "; registered only to drive the frozen evaluator", model["checkpoint"]),
        )
    db.commit()
    db.close()
    events = out / "downstream_events.jsonl"
    results = []
    for index, model in enumerate(models, start=1):
        if sha256(Path(model["checkpoint"])) != model["checkpoint_sha256"]:
            append_jsonl(events, {"utc": utc_now(), "event": "checkpoint_hash_mismatch", "method": model["method"]})
            results.append({"method": model["method"], "status": "hash_mismatch"})
            continue
        command = phase1.command_prefix() + [
            "downstream_evaluator.py", "--experiment-id", str(index), "--candidate", model["candidate"],
            "--checkpoint", model["checkpoint"], "--db", str(db_path), "--output-dir", str(out / "runs"),
            "--display-name", model["method"], "--selection-role", f"phase5a: {model['note']}",
        ]
        append_jsonl(events, {"utc": utc_now(), "event": "downstream_started", "id": index, "method": model["method"]})
        with (out / f"run_{index:02d}.log").open("w") as stream:
            done = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        append_jsonl(events, {"utc": utc_now(), "event": "downstream_finished", "id": index, "method": model["method"], "exit": done.returncode})
        results.append({"id": index, "method": model["method"], "exit": done.returncode, **model})
        print(json.dumps({"downstream": model["method"], "exit": done.returncode}), flush=True)
    write_json_new(out / "downstream_models.json", {"plan_sha256": plan_sha, "models": results, "recorded_utc": utc_now()})
    return 0 if all(row.get("exit") == 0 for row in results) else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("train", "downstream"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--expect-plan-sha256")
    args = parser.parse_args()
    plan, plan_sha = load_plan()
    if args.expect_plan_sha256 and args.expect_plan_sha256 != plan_sha:
        print(json.dumps({"status": "refused", "reason": "plan sha256 mismatch", "plan_sha256": plan_sha}))
        return 2
    rows = validate(plan, plan_sha)
    failures = [row for row in rows if row["status"] != "ok"]
    if not args.execute:
        print(json.dumps({"status": "validated" if not failures else "FAILED", "plan_sha256": plan_sha, "checks": rows}, indent=2, sort_keys=True))
        return 0 if not failures else 1
    if failures:
        print(json.dumps({"status": "refused", "failures": failures}, indent=2, sort_keys=True))
        return 2
    root = stage_root(plan)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "gpu.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"status": "refused", "reason": f"GPU lock held: {lock_path}"}))
            return 2
        if args.stage == "downstream":
            return execute_downstream(plan, plan_sha, root)
        e1_plan, e1_sha = e1.load_plan()
        problem = e1.uv_lock_problem(e1_plan)
        if problem:
            print(json.dumps({"status": "refused", "reason": problem}))
            return 2
        phase1.validate_persistent_runtime()
        env_failures, env = environment_checks(e1_plan, e1_sha)
        if env_failures:
            print(json.dumps({"status": "refused", "environment_failures": env_failures}, indent=2, sort_keys=True))
            return 2
        events = root / "stage_events.jsonl"
        write_json_new(root / f"stage_manifest-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json", {
            "format": "autopareto_phase5a_stage_manifest_v1",
            "plan_sha256": plan_sha,
            "e1_plan_sha256": e1_sha,
            "phase5a_checks": rows,
            "e1_environment": env,
            "started_utc": utc_now(),
        })
        append_jsonl(events, {"utc": utc_now(), "event": "stage_started", "plan_sha256": plan_sha})
        records = {}
        for run in runs_in_order(plan):
            records[run["run_id"]] = run_with_retry(plan, plan_sha, root, run, events)
        append_jsonl(events, {"utc": utc_now(), "event": "stage_finished"})
    summary = {
        "format": "autopareto_phase5a_execution_summary_v1",
        "plan_sha256": plan_sha,
        "completed": sorted(k for k, v in records.items() if v),
        "reported_failed": sorted(k for k, v in records.items() if not v),
        "recorded_utc": utc_now(),
    }
    write_json_new(root / f"phase5a_execution_summary-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not summary["reported_failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
