#!/usr/bin/env python3
"""Phase 6 amendment A4 (``evaluation/PHASE6_AMENDMENT_A4.md``): P2 lm-eval with a gate that leaves out BLiMP.

    uv run --locked python scripts/run_phase6_a4.py                                     # validate only
    uv run --locked python scripts/run_phase6_a4.py --execute --expect-plan-sha256 <A4 sha>

Calls the A1 runner's ``execute_lmeval`` unchanged, with a copy of the A1 plan whose
``crosscheck_metrics`` omit BLiMP. The original cross-check report is never rewritten; A4 writes its
own gate report beside it.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
A4_PLAN_PATH = ROOT / "evaluation/phase6_amendment_a4_plan.json"
A4_FORMAT = "autopareto_phase6_amendment_a4_plan_v1"

spec = importlib.util.spec_from_file_location("phase6_p2_runner", ROOT / "scripts/run_phase6_p2.py")
p2 = importlib.util.module_from_spec(spec)
sys.modules["phase6_p2_runner"] = p2
spec.loader.exec_module(p2)
rp6 = p2.rp6
sha256, read_json, write_json_new, append_jsonl, utc_now = p2.sha256, p2.read_json, p2.write_json_new, p2.append_jsonl, p2.utc_now


def load_a4() -> tuple[dict, str]:
    raw = A4_PLAN_PATH.read_bytes()
    plan = json.loads(raw)
    if plan.get("format") != A4_FORMAT:
        raise RuntimeError(f"unexpected A4 plan format {plan.get('format')}")
    return plan, hashlib.sha256(raw).hexdigest()


def gated_a1(a1: dict, a4: dict) -> dict:
    out = copy.deepcopy(a1)
    metrics = out["lmeval"]["crosscheck_metrics"]
    for task in a4["gate"]["excluded"]:
        metrics.pop(task)
    if sorted(metrics) != sorted(a4["gate"]["compared"]):
        raise RuntimeError(f"gate tasks {sorted(metrics)} differ from the A4 plan")
    return out


def validate(a4: dict) -> list[dict]:
    rows = []

    def add(name: str, ok: bool, detail) -> None:
        rows.append({"check": name, "status": "ok" if ok else "FAIL", "detail": detail})

    a1, a1_sha = p2.load_a1()
    add("a1_plan_sha256", a1_sha == a4["a1_plan_sha256"], a1_sha)
    a1_rows = p2.validate(a1, "lmeval")
    add("a1_lmeval_checks", all(r["status"] == "ok" for r in a1_rows),
        [r for r in a1_rows if r["status"] != "ok"] or f"{len(a1_rows)} ok")
    bad = {rel: sha256(ROOT / rel) if (ROOT / rel).is_file() else None for rel, exp in a4["implementation_sha256"].items()
           if not (ROOT / rel).is_file() or sha256(ROOT / rel) != exp}
    add("a4_implementation_sha256", not bad, bad or f"{len(a4['implementation_sha256'])} files match")
    parent, _ = rp6.load_plan()
    report = rp6.stage_root(parent) / "lmeval/crosscheck_report.json"
    add("original_crosscheck_report_unchanged", report.is_file() and sha256(report) == a4["original_crosscheck_report"]["sha256"],
        str(report))
    try:
        gated_a1(a1, a4)
        add("gate_tasks", True, a4["gate"])
    except (KeyError, RuntimeError) as error:
        add("gate_tasks", False, str(error))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--expect-plan-sha256")
    args = parser.parse_args()
    a4, a4_sha = load_a4()
    if args.expect_plan_sha256 and args.expect_plan_sha256 != a4_sha:
        print(json.dumps({"status": "refused", "reason": "A4 plan sha256 mismatch", "a4_plan_sha256": a4_sha}))
        return 2
    rows = validate(a4)
    failures = [row for row in rows if row["status"] != "ok"]
    if not args.execute:
        print(json.dumps({"status": "validated" if not failures else "FAILED", "a4_plan_sha256": a4_sha, "checks": rows}, indent=1))
        return 0 if not failures else 1
    if failures:
        print(json.dumps({"status": "refused", "failures": failures}, indent=1))
        return 2
    a1, a1_sha = p2.load_a1()
    parent, _ = rp6.load_plan()
    volume_root = rp6.stage_root(parent)
    out = volume_root / "lmeval"
    events = out / "a4_events.jsonl"
    with (volume_root / "gpu.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"status": "refused", "reason": "GPU lock held"}))
            return 2
        append_jsonl(events, {"utc": utc_now(), "event": "a4_lmeval_started", "a4_plan_sha256": a4_sha, "gate": a4["gate"]})
        rc = p2.execute_lmeval(parent, gated_a1(a1, a4), a1_sha, volume_root)
        append_jsonl(events, {"utc": utc_now(), "event": "a4_lmeval_finished", "exit": rc})
        gate_report = out / "crosscheck_report_a4.json"
        if not gate_report.exists():
            rows = []
            by_label = {m["label"]: m for m in p2.lmeval_models(parent, a1, volume_root)}
            for label in a1["lmeval"]["crosscheck_labels"]:
                payload = read_json(out / "crosscheck" / f"{label}.json")
                model = by_label[label]
                for task, (frozen_task, frozen_metric) in a1["lmeval"]["crosscheck_metrics"].items():
                    ours = p2.lmeval_value(payload, task)
                    theirs = p2.frozen_value(Path(model["frozen_result"]), frozen_task, frozen_metric)
                    diff = 100 * (ours - theirs)
                    rows.append({"label": label, "task": task, "lmeval": ours, "frozen": theirs, "diff_points": diff,
                                 "in_gate": task in a4["gate"]["compared"],
                                 "passed": abs(diff) <= a1["lmeval"]["crosscheck_tolerance_points"]})
            write_json_new(gate_report, {"format": "autopareto_phase6_lmeval_crosscheck_a4_v1", "a4_plan_sha256": a4_sha,
                                         "a1_plan_sha256": a1_sha, "gate": a4["gate"], "rows": rows,
                                         "gate_passed": all(r["passed"] for r in rows if r["in_gate"]),
                                         "blimp_explanation": "evaluation/PHASE6_AMENDMENT_A4.md section 2",
                                         "recorded_utc": utc_now()})
    print(json.dumps({"status": "finished", "exit": rc, "a4_plan_sha256": a4_sha}))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
