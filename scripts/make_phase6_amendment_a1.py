#!/usr/bin/env python3
"""Write evaluation/phase6_amendment_a1_plan.json (Phase 6 amendment A1). Run on the pod, after the
local Env A copy and the local data copy exist; refuses to overwrite unless --force."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation/phase6_amendment_a1_plan.json"
PARENT = ROOT / "evaluation/phase6_unified_plan.json"
PARENT_SHA = "dc1e605e82c5f598bbf91ab17f63d1db30a27511f5d264142c9b31eb6fe61096"
E1_PLAN = ROOT / "evaluation/camera_ready_e0_e1_plan_amendment1.json"
E1_RUNS = ROOT / "results/camera_ready_2026/amendment1/E1/runs"
VOLUME_ENV_PYTHON = str(ROOT / ".venv/bin/python")
LOCAL_ENV = "/opt/autopareto_envA"
LOCAL_DATA = "/root/autoresearch_cache_local"
LMEVAL_CACHE = "/root/phase6_lmeval_cache"
ENV_C_FREEZE = "/root/envC_freeze.txt"
IMPLEMENTATION = ["evaluation/PHASE6_AMENDMENT_A1.md", "scripts/run_phase6_p2.py", "scripts/phase6_lmeval.py",
                  "scripts/make_phase6_amendment_a1.py"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze_sha(python: str) -> str:
    out = subprocess.run(["uv", "pip", "freeze", "--python", python], capture_output=True, text=True, check=True).stdout
    return hashlib.sha256(out.encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if OUT.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {OUT}")
    if sha256(PARENT) != PARENT_SHA:
        raise SystemExit("parent plan hash differs")
    parent = json.loads(PARENT.read_text())
    e1 = json.loads(E1_PLAN.read_text())
    volume_freeze = freeze_sha(VOLUME_ENV_PYTHON)
    local_freeze = freeze_sha(f"{LOCAL_ENV}/bin/python")
    if volume_freeze != local_freeze:
        raise SystemExit(f"local Env A differs from the volume Env A ({local_freeze} != {volume_freeze})")
    cache = {str(p.relative_to(LMEVAL_CACHE)): sha256(p) for p in sorted(Path(LMEVAL_CACHE).rglob("*"))
             if p.is_file() and not p.name.endswith(".lock")}
    e1_runs = {}
    for seed in (6, 7, 8):
        record = json.loads((E1_RUNS / f"e1-arref-time300s-seed{seed}/camera_ready_run_record.json").read_text())
        e1_runs[seed] = {"tokens": record["metrics"]["tokens"], "heldout_bpb": record["metrics"]["heldout_bpb"],
                         "record_sha256": sha256(E1_RUNS / f"e1-arref-time300s-seed{seed}/camera_ready_run_record.json")}
    tokens = [v["tokens"] for v in e1_runs.values()]
    held = [v["heldout_bpb"] for v in e1_runs.values()]
    fixed = sorted(parent["fixed_time_runs"], key=lambda r: r["order"])
    drift = [r["run_id"] for r in fixed if r["role"] == "drift_control"]
    plan = {
        "format": "autopareto_phase6_amendment_a1_plan_v1",
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "document": "evaluation/PHASE6_AMENDMENT_A1.md",
        "parent_plan_sha256": PARENT_SHA,
        "implementation_sha256": {rel: sha256(ROOT / rel) for rel in IMPLEMENTATION},
        "local_storage": {
            "run_root": "/root/phase6_local",
            "env_a": {"path": LOCAL_ENV, "freeze_sha256": volume_freeze, "volume_env_python": VOLUME_ENV_PYTHON},
            "data": {"path": LOCAL_DATA, "source": "/workspace/.cache/autoresearch",
                     "files": {row["path"]: row["sha256"] for row in e1["data_identity_gate"]["files"]},
                     "manifest_source": str(E1_PLAN.relative_to(ROOT))},
            "lmeval": {"python": "/opt/autopareto_lmeval/bin/python", "freeze_file": ENV_C_FREEZE,
                       "freeze_sha256": sha256(Path(ENV_C_FREEZE)), "dataset_cache": LMEVAL_CACHE,
                       "dataset_cache_files": len(cache),
                       "dataset_cache_manifest_sha256": hashlib.sha256(json.dumps(cache, sort_keys=True).encode()).hexdigest(),
                       "lm_eval_version": "0.4.13"},
        },
        "timing_anomaly_gate": {"first_step_excluded": 10, "slow_factor": 2.0, "max_excess_seconds": 3.0, "min_parsed_steps": 100},
        "e1_comparability_gate": {"e1_arref_time300s": e1_runs,
                                  "tokens_bounds": [int(0.99 * min(tokens)), int(1.01 * max(tokens))],
                                  "heldout_bounds": [min(held) - 0.003, max(held) + 0.003]},
        "fixed_time_order": drift + [r["run_id"] for r in fixed if r["role"] != "drift_control"],
        "pilot": {"run_id": "p6-opus_a27-time300s-seed6", "pilot_run_id": "a1pilot-opus_a27-time300s-seed6",
                  "volume_root": "results/camera_ready_2026/phase6_pilot/a1", "local_root": "/root/phase6_local_pilot"},
        "lmeval": {
            "tasks": ["arc_challenge", "hellaswag", "winogrande", "openbookqa", "sciq", "boolq", "wikitext"],
            "crosscheck_labels": ["arref-seed6", "gpt_a22-seed6", "opus_a23-seed6"],
            "crosscheck_tasks": ["arc_easy", "piqa", "lambada_openai", "blimp"],
            "crosscheck_metrics": {"arc_easy": ["arc_easy", "accuracy"], "piqa": ["piqa", "accuracy"],
                                   "lambada_openai": ["lambada_openai", "exact_match"], "blimp": ["blimp", "macro_accuracy"]},
            "crosscheck_tolerance_points": 0.5,
            "reused_candidates_file": "results/camera_ready_2026/phase5/A/downstream/downstream_models.json",
            "reused_results_dir": "results/camera_ready_2026/phase5/A/downstream/runs",
            "frozen_for_all_39": ["arc_easy", "piqa", "lambada_openai", "blimp", "heldout_climbmix_bpb"],
        },
    }
    OUT.write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"written": str(OUT), "sha256": sha256(OUT), "fixed_runs": len(plan["fixed_time_order"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
