#!/usr/bin/env python3
"""Phase 6 amendment A2 (``evaluation/PHASE6_AMENDMENT_A2.md``): one extra attempt of P1 MOTPE seed 8.

    uv run --locked python scripts/run_phase6_a2.py                                         # validate only
    uv run --locked python scripts/run_phase6_a2.py --stage all --execute --expect-plan-sha256 <A2 sha>

Stages, in order (``all`` runs them and stops at the first that fails):

* ``train``: attempt 3 through the parent runner's ``execute_train_attempt`` (E1's token executor and
  gates, unchanged), in a local stage root under A1's storage rules. The attempt directory is then
  copied to the volume with every file's SHA-256 verified (A1's ``copy_verified``).
* ``downstream``: the parent runner's ``execute_downstream`` (frozen evaluator) for this one run.
* ``inference``: the parent runner's ``execute_inference`` for this run's P0b label.

The parent functions are called unchanged. For ``downstream`` and ``inference``, only the lookup of
this one run's record is pointed at attempt 3, for the duration of the call.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
A2_FORMAT = "autopareto_phase6_amendment_a2_plan_v1"

spec = importlib.util.spec_from_file_location("phase6_p2_runner", ROOT / "scripts/run_phase6_p2.py")
p2 = importlib.util.module_from_spec(spec)
sys.modules["phase6_p2_runner"] = p2
spec.loader.exec_module(p2)
rp6, e1 = p2.rp6, p2.e1
sha256, read_json, write_json_new, append_jsonl, utc_now = p2.sha256, p2.read_json, p2.write_json_new, p2.append_jsonl, p2.utc_now


def load_a2() -> tuple[dict, str]:
    raw = p2.A2_PLAN_PATH.read_bytes()
    plan = json.loads(raw)
    if plan.get("format") != A2_FORMAT:
        raise RuntimeError(f"unexpected A2 plan format {plan.get('format')}")
    return plan, hashlib.sha256(raw).hexdigest()


def parent_run(parent: dict, a2: dict) -> dict:
    return next(run for run in parent["runs"] if run["run_id"] == a2["run"]["run_id"])


def validate(a2: dict) -> list[dict]:
    a1, a1_sha = p2.load_a1()
    rows = p2.validate(a1, "fixed")  # parent plan and checks, A1 hashes, local Env A, local data bytes

    def add(name: str, ok: bool, detail) -> None:
        rows.append({"check": name, "status": "ok" if ok else "FAIL", "detail": detail})

    add("a1_plan_sha256", a1_sha == a2["a1_plan_sha256"], a1_sha)
    bad = {rel: sha256(ROOT / rel) if (ROOT / rel).is_file() else None for rel, exp in a2["implementation_sha256"].items()
           if not (ROOT / rel).is_file() or sha256(ROOT / rel) != exp}
    add("a2_implementation_sha256", not bad, bad or f"{len(a2['implementation_sha256'])} files match")
    parent, _ = rp6.load_plan()
    root = rp6.stage_root(parent)
    run = parent_run(parent, a2)
    want = a2["run"]
    add("run_matches_parent_plan", (run["recipe"], run["seed"], run["role"]) == (want["recipe"], want["seed"], "new")
        and e1.attempt_dirname(run, want["attempt"]) == want["attempt_dirname"], want)
    for row in a2["failed_attempts"]:
        path = root / row["record"]
        record = read_json(path) if path.is_file() else None
        ok = bool(record) and sha256(path) == row["sha256"] and record["attempt"] == row["attempt"] and not record["integrity"]["passed"]
        add(f"failed_attempt_{row['attempt']}_on_record", ok, str(path))
    return rows


@contextlib.contextmanager
def record_lookup_at(attempt: int):
    """Point the parent runner's run-directory lookup at ``attempt`` for the duration of one call."""
    original = rp6.run_dir_of
    rp6.run_dir_of = lambda root, run, _attempt: original(root, run, attempt)
    try:
        yield
    finally:
        rp6.run_dir_of = original


@contextlib.contextmanager
def checkpoint_lookup_with_a2(a2: dict):
    original = rp6.new_checkpoint

    def lookup(plan: dict, root: Path, recipe_key: str, seed: int):
        if (recipe_key, seed) == (a2["run"]["recipe"], a2["run"]["seed"]):
            return p2.a2_checkpoint(root, recipe_key, seed)
        return original(plan, root, recipe_key, seed)

    rp6.new_checkpoint = lookup
    try:
        yield
    finally:
        rp6.new_checkpoint = original


def stage_train(parent: dict, a1: dict, a1_sha: str, a2: dict, a2_sha: str, root: Path, events: Path) -> int:
    run = parent_run(parent, a2)
    attempt = a2["run"]["attempt"]
    recipe = parent["recipes"][run["recipe"]]
    local_root = Path(a2["local_root"])
    vdir, ldir = rp6.run_dir_of(root, run, attempt), rp6.run_dir_of(local_root, run, attempt)
    if vdir.exists():
        record = read_json(vdir / "phase6_run_record.json")
        if record and record["integrity"]["passed"]:
            print(json.dumps({"stage": "train", "status": "resume_skip_completed", "run_dir": str(vdir)}), flush=True)
            return 0
        print(json.dumps({"stage": "train", "status": "refused", "reason": f"{vdir} exists without a passing record; A2 allows one attempt"}))
        return 1
    if ldir.exists():
        print(json.dumps({"stage": "train", "status": "refused", "reason": f"{ldir} exists (interrupted attempt); A2 allows one attempt"}))
        return 1
    e1_plan, e1_sha = e1.load_plan()
    problem = e1.uv_lock_problem(e1_plan)
    if problem:
        print(json.dumps({"stage": "train", "status": "refused", "reason": problem}))
        return 1
    rp6.phase1.validate_persistent_runtime()
    env_failures, env = rp6.p5a.environment_checks(e1_plan, e1_sha)
    if env_failures:
        print(json.dumps({"stage": "train", "status": "refused", "environment_failures": env_failures}, indent=1))
        return 1
    e1.assert_gpu_idle()
    started = time.time()
    append_jsonl(events, {"utc": utc_now(), "event": "attempt_started", "run_id": run["run_id"], "attempt": attempt,
                          "load": Path("/proc/loadavg").read_text().split()[:3]})
    try:
        failures, values = rp6.execute_train_attempt(parent, local_root, run, attempt)
    except Exception as exc:  # recorded as a failed attempt, never hidden
        failures, values = [f"executor exception: {type(exc).__name__}: {exc}"], {}
        ldir.mkdir(parents=True, exist_ok=True)
    # Reported, not a gate: P1 matched-token runs have no timing gate.
    values["timing_anomaly"] = p2.timing_anomaly(ldir / "training.log", a1["timing_anomaly_gate"])
    copy_info = p2.copy_verified(ldir, vdir)
    record = {
        "format": "autopareto_phase6_run_record_v1", "plan_sha256": a2["parent_plan_sha256"],
        "amendment": {"name": "A2", "plan_sha256": a2_sha, "storage": "A1", "a1_plan_sha256": a1_sha},
        "stage": run["stage"], "run_id": run["run_id"], "attempt": attempt, "run_dir": str(vdir),
        "executed_in": str(ldir), "copy": copy_info, "order": run["order"], "recipe": run["recipe"],
        "display": recipe["display"], "role": run["role"], "phase1_model_id": recipe["phase1_model_id"],
        "source_sha256": recipe["source_sha256"], "TOTAL_BATCH_SIZE": recipe["TOTAL_BATCH_SIZE"],
        "cell": run["cell"], "seed": run["seed"],
        "status": "completed" if not failures else "failed",
        "integrity": {"passed": not failures, "failures": failures}, "metrics": values,
        "e1_environment": env, "attempt_wall_seconds": time.time() - started,
        "gpu": e1.nvidia_smi(["--query-gpu=name,uuid,driver_version"]) if hasattr(e1, "nvidia_smi") else None,
        "finished_utc": utc_now(),
    }
    write_json_new(vdir / "phase6_run_record.json", record)
    append_jsonl(events, {"utc": utc_now(), "event": "attempt_finished", "run_id": run["run_id"], "attempt": attempt,
                          "status": record["status"], "failures": failures[:8],
                          "timing_anomaly_s": values["timing_anomaly"].get("excess_seconds")})
    print(json.dumps({"run_id": run["run_id"], "attempt": attempt, "status": record["status"],
                      "heldout": values.get("heldout_bpb"), "tok_s": values.get("steady_state_tokens_per_second"),
                      "anomaly_s": values["timing_anomaly"].get("excess_seconds"), "failures": failures[:3],
                      "wall_s": round(time.time() - started)}), flush=True)
    return 0 if not failures else 1


def stage_downstream(parent: dict, parent_sha: str, a2: dict, root: Path) -> int:
    if p2.a2_checkpoint(root, a2["run"]["recipe"], a2["run"]["seed"]) is None:
        print(json.dumps({"stage": "downstream", "status": "refused", "reason": "no passing A2 record"}))
        return 1
    view = copy.deepcopy(parent)
    view["runs"] = [parent_run(parent, a2)]
    with record_lookup_at(a2["run"]["attempt"]):
        return rp6.execute_downstream(view, parent_sha, root)


def stage_inference(parent: dict, parent_sha: str, a2: dict, root: Path) -> int:
    if p2.a2_checkpoint(root, a2["run"]["recipe"], a2["run"]["seed"]) is None:
        print(json.dumps({"stage": "inference", "status": "refused", "reason": "no passing A2 record"}))
        return 1
    with checkpoint_lookup_with_a2(a2):
        return rp6.execute_inference(parent, parent_sha, root, "P0b", set(a2["inference_labels"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("train", "downstream", "inference", "all"), default="all")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--expect-plan-sha256")
    args = parser.parse_args()
    a2, a2_sha = load_a2()
    if args.expect_plan_sha256 and args.expect_plan_sha256 != a2_sha:
        print(json.dumps({"status": "refused", "reason": "A2 plan sha256 mismatch", "a2_plan_sha256": a2_sha}))
        return 2
    rows = validate(a2)
    failures = [row for row in rows if row["status"] != "ok"]
    if not args.execute:
        print(json.dumps({"status": "validated" if not failures else "FAILED", "a2_plan_sha256": a2_sha, "checks": rows}, indent=1))
        return 0 if not failures else 1
    if failures:
        print(json.dumps({"status": "refused", "failures": failures}, indent=1))
        return 2
    parent, parent_sha = rp6.load_plan()
    a1, a1_sha = p2.load_a1()
    root = rp6.stage_root(parent)
    events = root / "a2_events.jsonl"
    stages = ["train", "downstream", "inference"] if args.stage == "all" else [args.stage]
    results = {}
    rc = 0
    with (root / "gpu.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"status": "refused", "reason": "GPU lock held"}))
            return 2
        for stage in stages:
            append_jsonl(events, {"utc": utc_now(), "event": "stage_started", "stage": stage, "a2_plan_sha256": a2_sha})
            if stage == "train":
                rc = stage_train(parent, a1, a1_sha, a2, a2_sha, root, events)
            elif stage == "downstream":
                rc = stage_downstream(parent, parent_sha, a2, root)
            else:
                rc = stage_inference(parent, parent_sha, a2, root)
            results[stage] = rc
            append_jsonl(events, {"utc": utc_now(), "event": "stage_finished", "stage": stage, "exit": rc})
            if rc:
                break
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary = {"format": "autopareto_phase6_a2_execution_summary_v1", "a2_plan_sha256": a2_sha, "a1_plan_sha256": a1_sha,
               "parent_plan_sha256": parent_sha, "stages": results, "recorded_utc": utc_now()}
    write_json_new(root / f"a2_execution_summary-{stamp}.json", summary)
    print(json.dumps(summary, indent=1))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
