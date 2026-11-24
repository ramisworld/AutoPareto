#!/usr/bin/env python3
"""Phase 6 P2 under amendment A1 (``evaluation/PHASE6_AMENDMENT_A1.md``).

    uv run --locked python scripts/run_phase6_p2.py                                   # validate only
    uv run --locked python scripts/run_phase6_p2.py --stage pilot --execute ...       # non-study A1 pilot
    uv run --locked python scripts/run_phase6_p2.py --stage fixed --execute ...       # 34 fixed-time runs
    uv run --locked python scripts/run_phase6_p2.py --stage lmeval --execute ...      # cross-check, then 39

Fixed-time attempts call the parent runner's ``execute_fixed_attempt`` (E1's executor and gates,
unchanged) with a local stage root; the attempt directory is then copied to the network volume with
every file's SHA-256 verified. Added gates: timing anomaly (every attempt) and E1 comparability
(AR-ref drift control). lm-eval runs ``scripts/phase6_lmeval.py`` in Env C, one process per checkpoint.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
A1_PLAN_PATH = ROOT / "evaluation/phase6_amendment_a1_plan.json"
A1_FORMAT = "autopareto_phase6_amendment_a1_plan_v1"
STEP_RE = re.compile(r"step (\d+) \([^)]*\) \| loss: [^|]*\| lrm: [^|]*\| dt: (\d+)ms")

spec = importlib.util.spec_from_file_location("phase6_parent_runner", ROOT / "scripts/run_phase6.py")
rp6 = importlib.util.module_from_spec(spec)
sys.modules["phase6_parent_runner"] = rp6
spec.loader.exec_module(rp6)
e1 = rp6.e1
sha256, read_json, write_json_new, append_jsonl, utc_now = rp6.sha256, rp6.read_json, rp6.write_json_new, rp6.append_jsonl, rp6.utc_now


def load_a1() -> tuple[dict, str]:
    raw = A1_PLAN_PATH.read_bytes()
    plan = json.loads(raw)
    if plan.get("format") != A1_FORMAT:
        raise RuntimeError(f"unexpected A1 plan format {plan.get('format')}")
    return plan, hashlib.sha256(raw).hexdigest()


def freeze_sha(python: str) -> str:
    out = subprocess.run(["uv", "pip", "freeze", "--python", python], capture_output=True, text=True, check=True).stdout
    return hashlib.sha256(out.encode()).hexdigest()


def tree_manifest(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}


def validate(a1: dict, stage: str | None) -> list[dict]:
    rows = []

    def add(name: str, ok: bool, detail) -> None:
        rows.append({"check": name, "status": "ok" if ok else "FAIL", "detail": detail})

    parent, parent_sha = rp6.load_plan()
    add("parent_plan_sha256", parent_sha == a1["parent_plan_sha256"], parent_sha)
    parent_rows = rp6.validate(parent)
    add("parent_plan_checks", all(r["status"] == "ok" for r in parent_rows),
        [r for r in parent_rows if r["status"] != "ok"] or f"{len(parent_rows)} ok")
    bad = {rel: sha256(ROOT / rel) for rel, exp in a1["implementation_sha256"].items()
           if not (ROOT / rel).is_file() or sha256(ROOT / rel) != exp}
    add("a1_implementation_sha256", not bad, bad or f"{len(a1['implementation_sha256'])} files match")
    local = a1["local_storage"]
    if stage in ("pilot", "fixed"):
        env = local["env_a"]
        add("uv_project_environment", os.environ.get("UV_PROJECT_ENVIRONMENT") == env["path"], os.environ.get("UV_PROJECT_ENVIRONMENT"))
        add("uv_project", os.environ.get("UV_PROJECT") == str(ROOT), os.environ.get("UV_PROJECT"))
        add("runner_in_local_env", Path(sys.prefix).resolve() == Path(env["path"]).resolve(), sys.prefix)
        local_freeze = freeze_sha(f"{env['path']}/bin/python")
        add("env_a_freeze_equal", local_freeze == env["freeze_sha256"], local_freeze)
        data = local["data"]
        runtime = Path.home() / ".cache/autoresearch"
        add("data_symlink", runtime.resolve() == Path(data["path"]).resolve(), str(runtime.resolve()))
        add("persistent_cache_env", os.environ.get("AUTOPARETO_PERSISTENT_CACHE") == data["path"], os.environ.get("AUTOPARETO_PERSISTENT_CACHE"))
        got = tree_manifest(Path(data["path"])) if Path(data["path"]).is_dir() else {}
        wrong = sorted(rel for rel, digest in data["files"].items() if got.get(rel) != digest)
        add("data_bytes_match_e1_manifest", not wrong, wrong or f"{len(data['files'])} manifest files match ({len(got)} files present)")
    if stage == "lmeval":
        lm = local["lmeval"]
        add("env_c_freeze", sha256(Path(lm["freeze_file"])) == lm["freeze_sha256"] if Path(lm["freeze_file"]).is_file() else False, lm["freeze_file"])
        got = tree_manifest(Path(lm["dataset_cache"])) if Path(lm["dataset_cache"]).is_dir() else {}
        got = {k: v for k, v in got.items() if not k.endswith(".lock")}
        add("lmeval_dataset_cache", hashlib.sha256(json.dumps(got, sort_keys=True).encode()).hexdigest() == lm["dataset_cache_manifest_sha256"],
            f"{len(got)} files")
    return rows


# ---------------------------------------------------------------------------
# Fixed time under A1
# ---------------------------------------------------------------------------

def timing_anomaly(training_log: Path, gate: dict) -> dict:
    text = training_log.read_text(errors="replace") if training_log.is_file() else ""
    steps = {}
    for step, dt in STEP_RE.findall(text):
        steps[int(step)] = int(dt) / 1000.0
    times = [t for s, t in sorted(steps.items()) if s > gate["first_step_excluded"]]
    if len(times) < gate["min_parsed_steps"]:
        return {"parsed_steps": len(times), "median_seconds": None, "slow_steps": None, "excess_seconds": None,
                "passed": False, "reason": f"only {len(times)} steps parsed"}
    median = statistics.median(times)
    slow = [t for t in times if t > gate["slow_factor"] * median]
    excess = sum(t - median for t in slow)
    return {"parsed_steps": len(times), "median_seconds": median, "slow_steps": len(slow), "excess_seconds": excess,
            "max_step_seconds": max(times), "passed": excess <= gate["max_excess_seconds"],
            "reason": None if excess <= gate["max_excess_seconds"] else f"{excess:.1f} s of slow steps > {gate['max_excess_seconds']} s"}


def copy_verified(local_dir: Path, volume_dir: Path) -> dict:
    """Copy one attempt directory to the volume. compile_cache/ becomes one tar; every file is verified."""
    if volume_dir.exists():
        raise RuntimeError(f"refusing to overwrite {volume_dir}")
    staging = volume_dir.with_name(volume_dir.name + ".a1_partial")
    if staging.exists():
        raise RuntimeError(f"stale partial copy {staging}; inspect before resuming")
    staging.mkdir(parents=True)
    files = {}
    for path in sorted(local_dir.rglob("*")):
        rel = path.relative_to(local_dir)
        if rel.parts[0] == "compile_cache" or not path.is_file():
            continue
        target = staging / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        files[str(rel)] = sha256(path)
    cache = local_dir / "compile_cache"
    tar_info = None
    if cache.is_dir():
        local_tar = local_dir.parent / f"{local_dir.name}.compile_cache.tar"
        if not local_tar.exists():
            with tarfile.open(local_tar, "w") as archive:
                archive.add(cache, arcname="compile_cache")
        shutil.copy2(local_tar, staging / "compile_cache.tar")
        files["compile_cache.tar"] = sha256(local_tar)
        tar_info = {"local_tar": str(local_tar), "members": sum(1 for p in cache.rglob("*") if p.is_file())}
    mismatched = [rel for rel, digest in files.items() if sha256(staging / rel) != digest]
    if mismatched:
        raise RuntimeError(f"copy verification failed for {mismatched[:5]}")
    manifest = {"format": "autopareto_phase6_a1_copy_manifest_v1", "local_dir": str(local_dir),
                "volume_dir": str(volume_dir), "files": files, "compile_cache": tar_info,
                "verified_files": len(files), "copied_utc": utc_now()}
    write_json_new(staging / "a1_copy_manifest.json", manifest)
    staging.rename(volume_dir)
    return {"verified_files": len(files), "manifest_sha256": sha256(volume_dir / "a1_copy_manifest.json")}


def comparability(a1: dict, values: dict) -> dict:
    gate = a1["e1_comparability_gate"]
    tokens = values.get("tokens")
    held = values.get("heldout_bpb")
    t_lo, t_hi = gate["tokens_bounds"]
    h_lo, h_hi = gate["heldout_bounds"]
    ok = isinstance(tokens, int) and t_lo <= tokens <= t_hi and isinstance(held, float) and h_lo <= held <= h_hi
    return {"tokens": tokens, "tokens_bounds": [t_lo, t_hi], "heldout_bpb": held, "heldout_bounds": [h_lo, h_hi], "passed": ok}


def run_fixed(parent: dict, a1: dict, a1_sha: str, volume_root: Path, local_root: Path, run: dict,
              launcher: bytes, events: Path, pilot: bool) -> tuple[dict | None, bool]:
    """Returns (passing record or None, stop_stage)."""
    recipe = parent["recipes"][run["recipe"]]
    used = set()
    for attempt in (1, 2):
        vdir = rp6.run_dir_of(volume_root, run, attempt)
        if vdir.exists():
            record = read_json(vdir / "phase6_run_record.json")
            if record and record["integrity"]["passed"]:
                append_jsonl(events, {"utc": utc_now(), "event": "resume_skip_completed", "run_id": run["run_id"], "attempt": attempt})
                return record, False
            used.add(attempt)
    for attempt in (1, 2):
        if attempt in used:
            continue
        ldir = rp6.run_dir_of(local_root, run, attempt)
        vdir = rp6.run_dir_of(volume_root, run, attempt)
        started = time.time()
        if ldir.exists():  # interrupted before its copy: keep it as a failed attempt
            failures, values = ["interrupted attempt: local directory without a volume record"], {}
        else:
            e1.assert_gpu_idle()
            append_jsonl(events, {"utc": utc_now(), "event": "attempt_started", "run_id": run["run_id"], "attempt": attempt,
                                  "load": Path("/proc/loadavg").read_text().split()[:3]})
            try:
                failures, values = rp6.execute_fixed_attempt(parent, local_root, run, attempt, launcher)
            except Exception as exc:  # recorded as a failed attempt, never hidden
                failures, values = [f"executor exception: {type(exc).__name__}: {exc}"], {}
                ldir.mkdir(parents=True, exist_ok=True)
        anomaly = timing_anomaly(ldir / "training.log", a1["timing_anomaly_gate"])
        values["timing_anomaly"] = anomaly
        if not failures and not anomaly["passed"]:
            failures.append(f"timing-anomaly gate: {anomaly['reason']}")
        drift = None
        if run["role"] == "drift_control" and not failures:
            drift = comparability(a1, values)
            values["e1_comparability"] = drift
        copy = copy_verified(ldir, vdir)
        record = {
            "format": "autopareto_phase6_run_record_v1", "plan_sha256": a1["parent_plan_sha256"],
            "amendment": {"name": "A1", "plan_sha256": a1_sha}, "pilot": pilot,
            "stage": run["stage"], "run_id": run["run_id"], "attempt": attempt, "run_dir": str(vdir),
            "executed_in": str(ldir), "copy": copy, "order": run["order"], "recipe": run["recipe"],
            "display": recipe["display"], "role": run["role"], "phase1_model_id": recipe["phase1_model_id"],
            "source_sha256": recipe["source_sha256"], "TOTAL_BATCH_SIZE": recipe["TOTAL_BATCH_SIZE"],
            "cell": run["cell"], "seed": run["seed"],
            "status": "completed" if not failures else "failed",
            "integrity": {"passed": not failures, "failures": failures}, "metrics": values,
            "attempt_wall_seconds": time.time() - started,
            "gpu": e1.nvidia_smi(["--query-gpu=name,uuid,driver_version"]) if hasattr(e1, "nvidia_smi") else None,
            "finished_utc": utc_now(),
        }
        write_json_new(vdir / "phase6_run_record.json", record)
        append_jsonl(events, {"utc": utc_now(), "event": "attempt_finished", "run_id": run["run_id"], "attempt": attempt,
                              "status": record["status"], "failures": failures[:8], "timing_anomaly_s": anomaly.get("excess_seconds"),
                              "e1_comparability": drift})
        print(json.dumps({"run_id": run["run_id"], "attempt": attempt, "status": record["status"],
                          "tokens": values.get("tokens"), "heldout": values.get("heldout_bpb"),
                          "anomaly_s": anomaly.get("excess_seconds"), "e1_gate": (drift or {}).get("passed"),
                          "failures": failures[:3], "wall_s": round(time.time() - started)}), flush=True)
        if not failures:
            if drift is not None and not drift["passed"]:
                append_jsonl(events, {"utc": utc_now(), "event": "e1_comparability_failed_stage_stopped", "run_id": run["run_id"]})
                return record, True
            return record, False
    append_jsonl(events, {"utc": utc_now(), "event": "run_reported_failed", "run_id": run["run_id"]})
    return None, False


# ---------------------------------------------------------------------------
# lm-eval
# ---------------------------------------------------------------------------

A2_PLAN_PATH = ROOT / "evaluation/phase6_amendment_a2_plan.json"


def a2_checkpoint(volume_root: Path, recipe_key: str, seed: int) -> tuple[Path, str] | None:
    """Amendment A2: the one extra attempt of a P1 run whose two attempts failed on a volume error.
    Returned only if its record passed and names the A2 plan file as it is now."""
    if not A2_PLAN_PATH.is_file():
        return None
    a2 = json.loads(A2_PLAN_PATH.read_bytes())
    run = a2["run"]
    if (run["recipe"], run["seed"]) != (recipe_key, seed):
        return None
    record = read_json(volume_root / "runs" / run["attempt_dirname"] / "phase6_run_record.json")
    if not (record and record["integrity"]["passed"] and record.get("amendment", {}).get("name") == "A2"
            and record["amendment"]["plan_sha256"] == sha256(A2_PLAN_PATH)):
        return None
    return Path(record["run_dir"]) / "final.pt", record["metrics"]["checkpoint_sha256"]


def lmeval_models(parent: dict, a1: dict, volume_root: Path) -> list[dict]:
    reused = read_json(ROOT / a1["lmeval"]["reused_candidates_file"])
    by_checkpoint = {Path(m["checkpoint"]).resolve(): m for m in reused["models"]}
    models = []
    for recipe_key in parent["candidates_in_order"]:
        for seed in parent["seeds"]:
            found = rp6.reused_checkpoint(parent, recipe_key, seed)
            if found:
                frozen = by_checkpoint.get(Path(found[0]).resolve())
                if frozen is None:
                    raise RuntimeError(f"no frozen-evaluator candidate for {recipe_key} seed {seed}")
                candidate = Path(frozen["candidate"])
                frozen_result = ROOT / a1["lmeval"]["reused_results_dir"] / f"{int(frozen['id']):04d}" / "results.json"
            else:
                found = rp6.new_checkpoint(parent, volume_root, recipe_key, seed) or a2_checkpoint(volume_root, recipe_key, seed)
                if found is None:
                    raise RuntimeError(f"no P1 checkpoint for {recipe_key} seed {seed}")
                candidate = ROOT / parent["recipes"][recipe_key]["source_path"]
                frozen_result = None
            models.append({"label": f"{recipe_key}-seed{seed}", "recipe": recipe_key, "seed": seed,
                           "checkpoint": str(found[0]), "checkpoint_sha256": found[1],
                           "candidate": str(candidate), "frozen_result": str(frozen_result) if frozen_result else None})
    return models


def frozen_value(result_file: Path, task: str, metric: str) -> float:
    for row in read_json(result_file)["task_results"]:
        if row["task_name"] == task and row["metric_name"] == metric:
            return float(row["metric_value"])
    raise KeyError(f"{task}/{metric} not in {result_file}")


def lmeval_value(payload: dict, task: str) -> float:
    results = payload["results"]
    if task == "blimp":
        subs = [v["acc,none"] for k, v in results.items() if k.startswith("blimp_") and "acc,none" in v]
        if len(subs) != 67:
            raise RuntimeError(f"expected 67 BLiMP sub-tasks, found {len(subs)}")
        return sum(subs) / len(subs)
    return float(results[task]["acc,none"])


def run_lmeval_one(a1: dict, model: dict, tasks: list[str], output: Path, events: Path) -> int:
    if output.exists():
        return 0
    lm = a1["local_storage"]["lmeval"]
    checkpoint = rp6.stage_local(Path(model["checkpoint"]), model["checkpoint_sha256"])
    env = dict(os.environ)
    for key in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF", "UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV"):
        env.pop(key, None)
    env.update({"HF_HOME": f"{lm['dataset_cache']}/hf", "HF_DATASETS_CACHE": f"{lm['dataset_cache']}/hf/datasets",
                "HF_HUB_CACHE": f"{lm['dataset_cache']}/hf/hub", "HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1"})
    command = [lm["python"], str(ROOT / "scripts/phase6_lmeval.py"), "--candidate", model["candidate"],
               "--checkpoint", str(checkpoint), "--output", str(output), "--tasks", ",".join(tasks),
               "--label", model["label"]]
    e1.assert_gpu_idle()
    append_jsonl(events, {"utc": utc_now(), "event": "lmeval_started", "label": model["label"], "tasks": tasks,
                          "checkpoint": str(checkpoint), "checkpoint_source": model["checkpoint"]})
    with output.with_suffix(".log").open("w") as stream:
        done = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    append_jsonl(events, {"utc": utc_now(), "event": "lmeval_finished", "label": model["label"], "exit": done.returncode})
    print(json.dumps({"lmeval": model["label"], "tasks": len(tasks), "exit": done.returncode}), flush=True)
    return done.returncode


def execute_lmeval(parent: dict, a1: dict, a1_sha: str, volume_root: Path) -> int:
    spec = a1["lmeval"]
    out = volume_root / "lmeval"
    (out / "crosscheck").mkdir(parents=True, exist_ok=True)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    events = out / "lmeval_events.jsonl"
    models = lmeval_models(parent, a1, volume_root)
    by_label = {m["label"]: m for m in models}
    rows, ok = [], True
    for label in spec["crosscheck_labels"]:
        model = by_label[label]
        output = out / "crosscheck" / f"{label}.json"
        if run_lmeval_one(a1, model, spec["crosscheck_tasks"], output, events) != 0:
            print(json.dumps({"status": "stopped", "reason": f"cross-check run failed for {label}"}))
            return 1
        payload = read_json(output)
        for task, (frozen_task, frozen_metric) in spec["crosscheck_metrics"].items():
            ours = lmeval_value(payload, task)
            theirs = frozen_value(Path(model["frozen_result"]), frozen_task, frozen_metric)
            diff = 100 * (ours - theirs)
            passed = abs(diff) <= spec["crosscheck_tolerance_points"]
            ok &= passed
            rows.append({"label": label, "task": task, "lmeval": ours, "frozen": theirs, "diff_points": diff, "passed": passed})
    report = out / "crosscheck_report.json"
    if not report.exists():
        write_json_new(report, {"format": "autopareto_phase6_lmeval_crosscheck_v1", "a1_plan_sha256": a1_sha,
                                "tolerance_points": spec["crosscheck_tolerance_points"], "rows": rows,
                                "passed": ok, "recorded_utc": utc_now()})
    print(json.dumps({"crosscheck_passed": ok, "max_abs_diff_points": max(abs(r["diff_points"]) for r in rows)}), flush=True)
    if not ok:
        print(json.dumps({"status": "stopped", "reason": "lm-eval cross-check outside tolerance", "rows": rows}, indent=1))
        return 1
    failed = 0
    for model in models:
        failed += run_lmeval_one(a1, model, spec["tasks"], out / "runs" / f"{model['label']}.json", events) != 0
    write_json_new(out / f"lmeval_models-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json",
                   {"a1_plan_sha256": a1_sha, "models": models, "failed": failed, "recorded_utc": utc_now()})
    return 0 if not failed else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("pilot", "fixed", "lmeval"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--expect-plan-sha256")
    args = parser.parse_args()
    a1, a1_sha = load_a1()
    if args.expect_plan_sha256 and args.expect_plan_sha256 != a1_sha:
        print(json.dumps({"status": "refused", "reason": "A1 plan sha256 mismatch", "a1_plan_sha256": a1_sha}))
        return 2
    rows = validate(a1, args.stage)
    failures = [r for r in rows if r["status"] != "ok"]
    if not args.execute:
        print(json.dumps({"status": "validated" if not failures else "FAILED", "a1_plan_sha256": a1_sha, "checks": rows}, indent=1))
        return 0 if not failures else 1
    if failures:
        print(json.dumps({"status": "refused", "failures": failures}, indent=1))
        return 2
    parent, _ = rp6.load_plan()
    volume_root = rp6.stage_root(parent)
    with (volume_root / "gpu.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"status": "refused", "reason": "GPU lock held"}))
            return 2
        if args.stage == "lmeval":
            return execute_lmeval(parent, a1, a1_sha, volume_root)
        e1_plan, e1_sha = e1.load_plan()
        problem = e1.uv_lock_problem(e1_plan)
        if problem:
            print(json.dumps({"status": "refused", "reason": problem}))
            return 2
        rp6.phase1.validate_persistent_runtime()
        env_failures, env = rp6.p5a.environment_checks(e1_plan, e1_sha)
        if env_failures:
            print(json.dumps({"status": "refused", "environment_failures": env_failures}, indent=1))
            return 2
        launcher, _ = e1.resolve_fixed_time_launcher(e1_plan, None)
        if launcher is None:
            print(json.dumps({"status": "refused", "reason": "historical launcher unavailable"}))
            return 2
        by_id = {run["run_id"]: run for run in parent["fixed_time_runs"]}
        if args.stage == "pilot":
            pilot = a1["pilot"]
            runs = [dict(by_id[pilot["run_id"]], run_id=pilot["pilot_run_id"])]
            v_root, l_root = ROOT / pilot["volume_root"], Path(pilot["local_root"])
        else:
            runs = [by_id[run_id] for run_id in a1["fixed_time_order"]]
            v_root, l_root = volume_root, Path(a1["local_storage"]["run_root"])
        v_root.mkdir(parents=True, exist_ok=True)
        events = v_root / f"{args.stage}_events.jsonl"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        write_json_new(v_root / f"{args.stage}_stage_manifest-{stamp}.json", {
            "format": "autopareto_phase6_a1_stage_manifest_v1", "a1_plan_sha256": a1_sha,
            "parent_plan_sha256": a1["parent_plan_sha256"], "e1_plan_sha256": e1_sha, "checks": rows,
            "e1_environment": env, "started_utc": utc_now(), "run_ids": [r["run_id"] for r in runs]})
        records, stopped = {}, False
        for run in runs:
            records[run["run_id"]], stop = run_fixed(parent, a1, a1_sha, v_root, l_root, run, launcher, events, args.stage == "pilot")
            if stop:
                stopped = True
                break
    summary = {"format": "autopareto_phase6_a1_execution_summary_v1", "stage": args.stage, "a1_plan_sha256": a1_sha,
               "completed": sorted(k for k, v in records.items() if v), "reported_failed": sorted(k for k, v in records.items() if not v),
               "stopped_by_e1_comparability_gate": stopped, "recorded_utc": utc_now()}
    write_json_new(v_root / f"{args.stage}_execution_summary-{stamp}.json", summary)
    print(json.dumps(summary, indent=1))
    return 0 if not summary["reported_failed"] and not stopped else 1


if __name__ == "__main__":
    raise SystemExit(main())
