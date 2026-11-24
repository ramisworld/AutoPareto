#!/usr/bin/env python3
"""Phase 6 executor: unified benchmark of the 13 frozen candidates.

    python scripts/run_phase6.py                                  # validate only (default)
    python scripts/run_phase6.py --stage train --execute          # P1 matched-token runs + drift controls
    python scripts/run_phase6.py --stage fixed --execute          # P2 300 s runs (historical accounting)
    python scripts/run_phase6.py --stage downstream --execute     # frozen 4-task evaluator on new checkpoints
    python scripts/run_phase6.py --stage inference --block P0a --execute

Pre-registration: ``evaluation/PHASE6_PREREG.md``; plan: ``evaluation/phase6_unified_plan.json``.
Thin by design, like Phase 5A:

* Token cells call the E1 executor's ``execute_token_attempt`` unchanged, with a per-run
  E1-shaped plan view whose only differences are the recipe and that recipe's own batch.
  The protected Phase-1 launcher, source transform, recorders, evaluator and the E1 gates
  (``gate_token_run``) are therefore byte for byte those of E1.
* The Opus drift control calls Phase 5A's ``execute_opus_attempt`` unchanged.
* Fixed-time cells call E1's ``execute_fixed_time_attempt`` and ``gate_fixed_time_run``
  unchanged (historical launcher, 300 s of synchronised step time after step 10).
* Inference runs ``scripts/phase6_unified_inference.py`` in Env B, one process per run,
  serially, after an ``nvidia-smi`` idle check.

Nothing outside ``results/camera_ready_2026/phase6`` is written. Every attempt is kept;
one retry per run.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PLAN_PATH = ROOT / "evaluation/phase6_unified_plan.json"
PLAN_FORMAT = "autopareto_phase6_unified_plan_v1"
ENV_B_PYTHON = "/opt/autopareto_runtime/bin/python"
REGISTERED_CACHE = "/var/cache/autopareto/pair_a_registered_cache"
SETUP_MARKER = "# Setup: tokenizer, model, optimizer, dataloader"
# Inference reads checkpoints from local overlay disk, not the network volume: the volume stalls
# intermittently, and cold-start timings should not depend on it. Files are keyed by SHA-256.
LOCAL_STAGE = Path("/root/phase6_stage")


def load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


e1 = load_module("phase6_e1_executor", ROOT / "scripts/run_camera_ready_e0_e1.py")
p5a = load_module("phase6_p5a_executor", ROOT / "scripts/run_phase5a_opus_fixed_tokens.py")
phase1 = e1.phase1
sha256, read_json, write_json_new, append_jsonl, utc_now = e1.sha256, e1.read_json, e1.write_json_new, e1.append_jsonl, e1.utc_now


def load_plan() -> tuple[dict, str]:
    raw = PLAN_PATH.read_bytes()
    plan = json.loads(raw)
    if plan.get("format") != PLAN_FORMAT:
        raise RuntimeError(f"unexpected plan format {plan.get('format')}")
    return plan, hashlib.sha256(raw).hexdigest()


def stage_root(plan: dict) -> Path:
    root = (ROOT / plan["output_root"]).resolve()
    for forbidden in (ROOT / "results/phase1", ROOT / "results/camera_ready_2026/amendment1",
                      ROOT / "results/camera_ready_2026/phase5"):
        forbidden = forbidden.resolve()
        if root == forbidden or forbidden in root.parents or root in forbidden.parents:
            raise RuntimeError(f"output root overlaps protected evidence: {root}")
    return root


# ---------------------------------------------------------------------------
# Plan views
# ---------------------------------------------------------------------------

def e1_view(plan: dict, recipe_key: str) -> dict:
    """E1 plan with exactly one recipe (this one) and that recipe's batch."""
    e1_plan, _ = e1.load_plan()
    recipe = plan["recipes"][recipe_key]
    view = copy.deepcopy(e1_plan)
    view["recipes"] = {recipe_key: {
        "source_path": recipe["source_path"],
        "source_sha256": recipe["source_sha256"],
        "phase1_model_id": recipe["phase1_model_id"],
    }}
    view["batch"] = dict(view["batch"], TOTAL_BATCH_SIZE=recipe["TOTAL_BATCH_SIZE"],
                         DEVICE_BATCH_SIZE=recipe["DEVICE_BATCH_SIZE"])
    view["implementation_sha256"] = {**e1_plan["implementation_sha256"]}
    return view


def run_dir_of(root: Path, run: dict, attempt: int) -> Path:
    return root / "runs" / e1.attempt_dirname(run, attempt)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def definitions(text: str) -> str:
    return text.split(SETUP_MARKER, 1)[0]


def validate(plan: dict) -> list[dict]:
    rows: list[dict] = []

    def add(name: str, ok: bool, detail) -> None:
        rows.append({"check": name, "status": "ok" if ok else "FAIL", "detail": detail})

    for name, spec in plan["comparator_plans"].items():
        add(f"comparator_plan_{name}", sha256(ROOT / spec["path"]) == spec["sha256"], spec["path"])
    bad = {rel: sha256(ROOT / rel) if (ROOT / rel).is_file() else None
           for rel, expected in plan["implementation_sha256"].items()
           if not (ROOT / rel).is_file() or sha256(ROOT / rel) != expected}
    add("implementation_sha256", not bad, bad or f"{len(plan['implementation_sha256'])} files match")
    problems = []
    for key, recipe in plan["recipes"].items():
        path = ROOT / recipe["source_path"]
        if not path.is_file() or sha256(path) != recipe["source_sha256"]:
            problems.append(f"{key}: source missing or hash differs")
            continue
        if e1.read_total_batch_size(path.read_text()) != recipe["TOTAL_BATCH_SIZE"]:
            problems.append(f"{key}: TOTAL_BATCH_SIZE differs from plan")
        if "pair_a_source_path" in recipe and sha256(ROOT / recipe["pair_a_source_path"]) != recipe["pair_a_source_sha256"]:
            problems.append(f"{key}: Pair-A source hash differs")
    add("recipe_sources", not problems, problems or f"{len(plan['recipes'])} sources match")
    missing = []
    for row in plan["reused_runs"]:
        run_dir = ROOT / row["run_dir"]
        record = read_json(run_dir / "camera_ready_run_record.json") or read_json(run_dir / "phase5a_run_record.json")
        if not record or not record.get("integrity", {}).get("passed") or record.get("seed") != row["seed"]:
            missing.append(row["run_dir"])
    add("reused_run_records", not missing, missing or f"{len(plan['reused_runs'])} reused runs passed their gates")
    return rows


# ---------------------------------------------------------------------------
# P1 / P2 training
# ---------------------------------------------------------------------------

def execute_train_attempt(plan: dict, root: Path, run: dict, attempt: int) -> tuple[list[str], dict]:
    recipe = plan["recipes"][run["recipe"]]
    if "phase5a_recipe" in recipe:
        p5_plan, _ = p5a.load_plan()
        p5_run = {"run_id": run["run_id"], "order": run["order"], "recipe": recipe["phase5a_recipe"],
                  "cell": run["cell"], "seed": run["seed"], "role": run["role"]}
        return p5a.execute_opus_attempt(p5_plan, root, p5_run, attempt)
    view = e1_view(plan, run["recipe"])
    e1_run = {k: run[k] for k in ("run_id", "order", "recipe", "cell", "seed")} | {"stage": "P6"}
    return e1.execute_token_attempt(view, root, e1_run, attempt)


def execute_fixed_attempt(plan: dict, root: Path, run: dict, attempt: int, launcher: bytes) -> tuple[list[str], dict]:
    view = e1_view(plan, run["recipe"])
    e1_run = {k: run[k] for k in ("run_id", "order", "recipe", "cell", "seed")} | {"stage": "P6"}
    return e1.execute_fixed_time_attempt(view, root, e1_run, attempt, launcher)


def write_record(plan_sha: str, root: Path, run: dict, attempt: int, failures: list[str], values: dict,
                 started: float, recipe: dict) -> dict:
    record = {
        "format": "autopareto_phase6_run_record_v1",
        "plan_sha256": plan_sha,
        "stage": run["stage"],
        "run_id": run["run_id"],
        "attempt": attempt,
        "run_dir": str(run_dir_of(root, run, attempt)),
        "order": run["order"],
        "recipe": run["recipe"],
        "display": recipe["display"],
        "role": run["role"],
        "phase1_model_id": recipe["phase1_model_id"],
        "source_sha256": recipe["source_sha256"],
        "TOTAL_BATCH_SIZE": recipe["TOTAL_BATCH_SIZE"],
        "cell": run["cell"],
        "seed": run["seed"],
        "status": "completed" if not failures else "failed",
        "integrity": {"passed": not failures, "failures": failures},
        "metrics": values,
        "attempt_wall_seconds": time.time() - started,
        "gpu": e1.nvidia_smi(["--query-gpu=name,uuid,driver_version"]) if hasattr(e1, "nvidia_smi") else None,
        "finished_utc": utc_now(),
    }
    write_json_new(run_dir_of(root, run, attempt) / "phase6_run_record.json", record)
    return record


def run_with_retry(plan: dict, plan_sha: str, root: Path, run: dict, events: Path, launcher: bytes | None) -> dict | None:
    recipe = plan["recipes"][run["recipe"]]
    for attempt in (1, 2):
        run_dir = run_dir_of(root, run, attempt)
        if run_dir.exists():
            record = read_json(run_dir / "phase6_run_record.json")
            if record and record["integrity"]["passed"]:
                append_jsonl(events, {"utc": utc_now(), "event": "resume_skip_completed", "run_id": run["run_id"], "attempt": attempt})
                return record
            continue
        e1.assert_gpu_idle()
        started = time.time()
        append_jsonl(events, {"utc": utc_now(), "event": "attempt_started", "run_id": run["run_id"], "attempt": attempt})
        try:
            if run["stage"] == "train":
                failures, values = execute_train_attempt(plan, root, run, attempt)
            else:
                failures, values = execute_fixed_attempt(plan, root, run, attempt, launcher)
        except Exception as exc:  # recorded as a failed attempt, never hidden
            failures, values = [f"executor exception: {type(exc).__name__}: {exc}"], {}
            run_dir.mkdir(parents=True, exist_ok=True)
        record = write_record(plan_sha, root, run, attempt, failures, values, started, recipe)
        append_jsonl(events, {"utc": utc_now(), "event": "attempt_finished", "run_id": run["run_id"], "attempt": attempt,
                              "status": record["status"], "failures": failures[:8]})
        print(json.dumps({"run_id": run["run_id"], "attempt": attempt, "status": record["status"],
                          "heldout": values.get("heldout_bpb"), "tok_s": values.get("steady_state_tokens_per_second"),
                          "failures": failures[:3], "wall_s": round(time.time() - started)}), flush=True)
        if record["integrity"]["passed"]:
            return record
    append_jsonl(events, {"utc": utc_now(), "event": "run_reported_failed", "run_id": run["run_id"]})
    return None


# ---------------------------------------------------------------------------
# Frozen downstream evaluator (P1)
# ---------------------------------------------------------------------------

def execute_downstream(plan: dict, plan_sha: str, root: Path) -> int:
    spec = plan["frozen_downstream"]
    for rel, expected in (("downstream_evaluator.py", spec["evaluator_sha256"]), ("evaluation/downstream_suite.json", spec["suite_sha256"])):
        if sha256(ROOT / rel) != expected:
            print(json.dumps({"status": "refused", "reason": f"{rel} hash differs"}))
            return 2
    out = root / "downstream_frozen"
    out.mkdir(parents=True, exist_ok=True)
    (out / "runs").mkdir(exist_ok=True)
    db_path = out / "phase6_downstream.db"
    models = []
    for run in sorted(plan["runs"], key=lambda r: r["order"]):
        if run["role"] != "new":
            continue
        record = None
        for attempt in (1, 2):
            candidate = read_json(run_dir_of(root, run, attempt) / "phase6_run_record.json")
            if candidate and candidate["integrity"]["passed"]:
                record = candidate
                break
        if record is None:
            continue
        recipe = plan["recipes"][run["recipe"]]
        models.append({
            "method": f"{recipe['phase1_model_id']}_tokens201M_seed{run['seed']}",
            "recipe": run["recipe"], "seed": run["seed"],
            "candidate": str(ROOT / recipe["source_path"]),
            "checkpoint": str(Path(record["run_dir"]) / "final.pt"),
            "checkpoint_sha256": record["metrics"]["checkpoint_sha256"],
        })
    from autopareto_db import migrate

    fresh = not db_path.exists()
    db = sqlite3.connect(db_path)
    if fresh:
        db.execute(p5a.DOWNSTREAM_SCHEMA)
        db.execute("CREATE TABLE campaign_state (campaign_id TEXT PRIMARY KEY, method TEXT, status TEXT, attempt_limit INTEGER, next_experiment_number INTEGER)")
        migrate(db)
    events = out / "downstream_events.jsonl"
    results = []
    for model in models:
        row = db.execute("SELECT id FROM experiments WHERE method = ?", (model["method"],)).fetchone()
        if row is None:
            index = (db.execute("SELECT COALESCE(MAX(id), 0) FROM experiments").fetchone()[0] or 0) + 1
            db.execute(
                """INSERT INTO experiments (id, campaign_id, experiment_number, method, training_seed, status,
                   description, git_commit, code_diff, checkpoint_path, phase, training_status, inference_status)
                   VALUES (?, 'phase6-downstream-20261116', ?, ?, ?, 'completed', ?, 'phase6', '', ?, 'phase6', 'completed', 'compatible')""",
                (index, index, model["method"], model["seed"], "Phase 6 matched tokens 201.3M; registered only to drive the frozen evaluator", model["checkpoint"]))
            db.commit()
        else:
            index = row[0]
        result_dir = out / "runs" / f"{index:04d}"
        if (result_dir / "results.json").is_file():
            results.append({"id": index, **model, "exit": 0, "resumed": True})
            continue
        if sha256(Path(model["checkpoint"])) != model["checkpoint_sha256"]:
            append_jsonl(events, {"utc": utc_now(), "event": "checkpoint_hash_mismatch", "method": model["method"]})
            results.append({"id": index, **model, "exit": "hash_mismatch"})
            continue
        e1.assert_gpu_idle()
        command = phase1.command_prefix() + [
            "downstream_evaluator.py", "--experiment-id", str(index), "--candidate", model["candidate"],
            "--checkpoint", model["checkpoint"], "--db", str(db_path), "--output-dir", str(out / "runs"),
            "--display-name", model["method"], "--selection-role", "phase6 matched tokens 201.3M",
        ]
        append_jsonl(events, {"utc": utc_now(), "event": "downstream_started", "id": index, "method": model["method"]})
        with (out / f"run_{index:02d}.log").open("w") as stream:
            done = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        append_jsonl(events, {"utc": utc_now(), "event": "downstream_finished", "id": index, "method": model["method"], "exit": done.returncode})
        results.append({"id": index, **model, "exit": done.returncode})
        print(json.dumps({"downstream": model["method"], "exit": done.returncode}), flush=True)
    db.close()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    write_json_new(out / f"downstream_models-{stamp}.json", {"plan_sha256": plan_sha, "models": results, "recorded_utc": utc_now()})
    return 0 if all(row.get("exit") == 0 for row in results) else 1


# ---------------------------------------------------------------------------
# Inference (P0, controls, anchors)
# ---------------------------------------------------------------------------

def reused_checkpoint(plan: dict, recipe_key: str, seed: int) -> tuple[Path, str] | None:
    """(path, recorded SHA-256) of a reused checkpoint; the hash is verified on the staged copy."""
    for row in plan["reused_runs"]:
        if row["recipe"] == recipe_key and row["seed"] == seed and row["cell"] == "tokens_201326592":
            run_dir = ROOT / row["run_dir"]
            p5 = read_json(run_dir / "phase5a_run_record.json")
            if p5:
                return Path(p5["metrics"]["deployed_final_checkpoint_path"]), p5["metrics"]["deployed_final_checkpoint_sha256"]
            record = read_json(run_dir / "camera_ready_run_record.json")
            return run_dir / "final.pt", record["metrics"]["checkpoint_sha256"]
    return None


def new_checkpoint(plan: dict, root: Path, recipe_key: str, seed: int) -> tuple[Path, str] | None:
    for run in plan["runs"]:
        if run["recipe"] == recipe_key and run["seed"] == seed and run["role"] == "new":
            for attempt in (1, 2):
                record = read_json(run_dir_of(root, run, attempt) / "phase6_run_record.json")
                if record and record["integrity"]["passed"]:
                    return Path(record["run_dir"]) / "final.pt", record["metrics"]["checkpoint_sha256"]
    return None


def stage_local(path: Path, expected: str) -> Path:
    """Return a local copy of a checkpoint whose SHA-256 is verified to equal (or, for the bracketing
    baseline, start with) ``expected``. The network copy is read only when no verified local copy exists."""
    LOCAL_STAGE.mkdir(parents=True, exist_ok=True)
    for staged in sorted(LOCAL_STAGE.glob(f"{expected}*.pt")):
        if sha256(staged).startswith(expected):
            return staged
    partial = LOCAL_STAGE / f"{expected}.partial"
    shutil.copyfile(path, partial)
    digest = sha256(partial)
    if not digest.startswith(expected):
        partial.unlink()
        raise RuntimeError(f"checkpoint hash differs: {path}")
    staged = LOCAL_STAGE / f"{digest}.pt"
    partial.rename(staged)
    return staged


def inference_context(plan: dict, root: Path, recipe_key: str, mode: str) -> Path:
    recipe = plan["recipes"][recipe_key]
    name = recipe_key + ("-eager_reference" if mode == "native_eager_reference" else "")
    context = root / "inference" / "contexts" / name
    source = ROOT / recipe.get("pair_a_source_path", recipe["source_path"])
    text = source.read_text()
    if mode == "native_eager_reference":
        ablation = load_module("phase6_p5b_ablation", ROOT / "scripts/phase5b_inference_ablation.py")
        settings = {"cuda_graph": False, "compile_regions": False, "sdpa_decode_attention": True, "ring_kv_cache": False}
        if ablation.FACTORY not in text:
            raise RuntimeError(f"{recipe_key}: adapter factory not found for the eager reference")
        text = text.replace(ablation.FACTORY, ablation.BLOCK.format(variant="P6_eager_reference_all_off", settings=settings), 1)
    prepare = ROOT / recipe["inference_context_prepare"]
    if context.exists():
        if (context / "train.py").read_text() != text or sha256(context / "prepare.py") != sha256(prepare):
            raise RuntimeError(f"existing context {context} differs from the plan")
        return context
    context.mkdir(parents=True)
    (context / "train.py").write_text(text)
    shutil.copy2(prepare, context / "prepare.py")
    write_json_new(context / "context_manifest.json", {
        "recipe": recipe_key, "mode": mode, "source": str(source.relative_to(ROOT)), "source_sha256": sha256(source),
        "train_sha256": sha256(context / "train.py"), "prepare_sha256": sha256(context / "prepare.py"),
        "derived_definitions_equal_source": definitions(text) == definitions((ROOT / recipe["source_path"]).read_text()),
    })
    return context


def execute_inference(plan: dict, plan_sha: str, root: Path, block: str, labels: set[str] | None) -> int:
    spec = plan["inference"]
    base_ctx = (ROOT / spec["bracketing_baseline"]["context_train"]).parent
    try:
        base_ck = stage_local(Path(spec["bracketing_baseline"]["checkpoint"]),
                              spec["bracketing_baseline"]["checkpoint_sha256_prefix"])
    except (OSError, RuntimeError) as error:
        print(json.dumps({"status": "refused", "reason": f"bracketing baseline checkpoint: {error}"}))
        return 2
    out = root / "inference" / "runs"
    out.mkdir(parents=True, exist_ok=True)
    events = root / "inference" / "inference_events.jsonl"
    env = dict(os.environ)
    for key in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        env.pop(key, None)
    env.update({"AUTOPARETO_REGISTERED_CACHE": REGISTERED_CACHE, "HF_HOME": f"{REGISTERED_CACHE}/huggingface",
                "HF_HUB_CACHE": f"{REGISTERED_CACHE}/huggingface/hub", "HF_HUB_OFFLINE": "1"})
    failed = 0
    for run in sorted(spec["runs"], key=lambda r: r["order"]):
        if run["block"] != block or (labels and run["label"] not in labels):
            continue
        found = reused_checkpoint(plan, run["recipe"], run["seed"]) or new_checkpoint(plan, root, run["recipe"], run["seed"])
        if found is None:
            append_jsonl(events, {"utc": utc_now(), "event": "checkpoint_unavailable", "label": run["label"]})
            print(json.dumps({"label": run["label"], "status": "checkpoint_unavailable"}), flush=True)
            failed += 1
            continue
        source_checkpoint, expected_sha = found
        checkpoint = stage_local(source_checkpoint, expected_sha)
        mode = "native" if run["mode"] == "native_eager_reference" else run["mode"]
        context = inference_context(plan, root, run["recipe"], run["mode"])
        done = False
        for attempt in (1, 2):
            name = run["label"] if attempt == 1 else f"{run['label']}-attempt2"
            output = out / f"{name}.json"
            if output.exists():
                status = (read_json(output) or {}).get("status")
                if status in ("completed", "correctness_only", "incorrect_not_timed", "incompatible_adapter"):
                    done = True
                    break
                continue
            e1.assert_gpu_idle()
            command = [ENV_B_PYTHON, str(ROOT / "scripts/phase6_unified_inference.py"), "--train", str(context / "train.py"),
                       "--checkpoint", str(checkpoint), "--mode", mode, "--label", name, "--output", str(output),
                       "--bracketed-baseline-context", str(base_ctx), "--bracketed-baseline-checkpoint", str(base_ck)]
            if run.get("correctness_only"):
                command.append("--correctness-only")
            append_jsonl(events, {"utc": utc_now(), "event": "inference_started", "label": name, "plan_sha256": plan_sha,
                                  "checkpoint": str(checkpoint), "checkpoint_source": str(source_checkpoint),
                                  "load": Path("/proc/loadavg").read_text().split()[:3]})
            with (out / f"{name}.log").open("w") as stream:
                proc = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            status = (read_json(output) or {}).get("status") if output.exists() else None
            append_jsonl(events, {"utc": utc_now(), "event": "inference_finished", "label": name, "exit": proc.returncode, "status": status})
            tail = (out / f"{name}.log").read_text(errors="replace").strip().splitlines()[-1:] or [""]
            print(tail[0][:400], flush=True)
            if status in ("completed", "correctness_only", "incorrect_not_timed", "incompatible_adapter"):
                done = True
                break
        failed += 0 if done else 1
    return 0 if not failed else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("train", "fixed", "downstream", "inference"))
    parser.add_argument("--block", choices=("P0a", "P0b", "control", "anchor"))
    parser.add_argument("--labels", nargs="*")
    parser.add_argument("--run-ids", nargs="*")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--expect-plan-sha256")
    args = parser.parse_args()
    plan, plan_sha = load_plan()
    if args.expect_plan_sha256 and args.expect_plan_sha256 != plan_sha:
        print(json.dumps({"status": "refused", "reason": "plan sha256 mismatch", "plan_sha256": plan_sha}))
        return 2
    rows = validate(plan)
    failures = [row for row in rows if row["status"] != "ok"]
    if not args.execute:
        print(json.dumps({"status": "validated" if not failures else "FAILED", "plan_sha256": plan_sha, "checks": rows}, indent=1))
        return 0 if not failures else 1
    if failures:
        print(json.dumps({"status": "refused", "failures": failures}, indent=1))
        return 2
    root = stage_root(plan)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "gpu.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"status": "refused", "reason": "GPU lock held"}))
            return 2
        if args.stage == "inference":
            if not args.block:
                parser.error("--block is required for inference")
            return execute_inference(plan, plan_sha, root, args.block, set(args.labels) if args.labels else None)
        if args.stage == "downstream":
            return execute_downstream(plan, plan_sha, root)
        e1_plan, e1_sha = e1.load_plan()
        problem = e1.uv_lock_problem(e1_plan)
        if problem:
            print(json.dumps({"status": "refused", "reason": problem}))
            return 2
        phase1.validate_persistent_runtime()
        env_failures, env = p5a.environment_checks(e1_plan, e1_sha)
        if env_failures:
            print(json.dumps({"status": "refused", "environment_failures": env_failures}, indent=1))
            return 2
        launcher = None
        if args.stage == "fixed":
            launcher, _ = e1.resolve_fixed_time_launcher(e1_plan, None)
            if launcher is None:
                print(json.dumps({"status": "refused", "reason": "historical launcher unavailable"}))
                return 2
        runs = plan["runs"] if args.stage == "train" else plan["fixed_time_runs"]
        if args.run_ids:
            runs = [run for run in runs if run["run_id"] in set(args.run_ids)]
        events = root / f"{args.stage}_events.jsonl"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        write_json_new(root / f"{args.stage}_stage_manifest-{stamp}.json", {
            "format": "autopareto_phase6_stage_manifest_v1", "plan_sha256": plan_sha, "e1_plan_sha256": e1_sha,
            "checks": rows, "e1_environment": env, "started_utc": utc_now(), "run_ids": [r["run_id"] for r in runs],
        })
        records = {}
        for run in sorted(runs, key=lambda r: r["order"]):
            records[run["run_id"]] = run_with_retry(plan, plan_sha, root, run, events, launcher)
    summary = {
        "format": "autopareto_phase6_execution_summary_v1", "stage": args.stage, "plan_sha256": plan_sha,
        "completed": sorted(k for k, v in records.items() if v),
        "reported_failed": sorted(k for k, v in records.items() if not v),
        "recorded_utc": utc_now(),
    }
    write_json_new(root / f"{args.stage}_execution_summary-{stamp}.json", summary)
    print(json.dumps(summary, indent=1))
    return 0 if not summary["reported_failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
