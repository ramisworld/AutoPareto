#!/usr/bin/env python3
"""Write evaluation/phase6_unified_plan.json (Phase 6 pre-registration, machine-readable).

Deterministic: every hash is computed from the committed files; run orders come
from a fixed shuffle seed. Refuses to overwrite an existing plan unless --force
(a plan already used for study runs must be amended, not rewritten).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation/phase6_unified_plan.json"
TOKEN_BUDGET = 201326592
SEEDS = [6, 7, 8]
ORDER_SEED = 20261116


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def batch_of(source: Path) -> int:
    match = re.search(r"^TOTAL_BATCH_SIZE\s*=\s*([^#\n]+)", source.read_text(), re.M)
    return int(eval(match.group(1).strip(), {"__builtins__": {}}))  # e.g. "2**17"


HISTORICAL = {
    "baseline": ("Baseline", "baseline", "results/candidates/baseline-a40-seed0/0002-train.py", "baseline",
                 "registered starting model (50.33M parameters)"),
    "arref": ("AR-ref", "autoresearch", "results/candidates/autoresearch-diagnostic5-seed42-20260724/0005-train.py",
              "autoresearch_compatible", "AutoResearch (GPT-5.6) reference recipe; E1 recipe AR-ref"),
    "rs": ("Random Search", "classical", "results/candidates/random-search-seed42-20260805/0010-train.py",
           "random_search_quality", "lowest valid compatible val BPB of random-search-seed42-20260805 (trial 10, exp 247)"),
    "tpe": ("TPE", "classical", "results/candidates/tpe-seed42-20260805/0021-train.py", "tpe_quality",
            "lowest valid compatible val BPB of tpe-seed42-20260805 (trial 21, exp 183); Phase 1 panel recipe"),
    "motpe": ("MOTPE", "classical", "results/candidates/motpe-seed42-20260805/0022-train.py", "motpe_quality",
              "lowest valid compatible val BPB of motpe-seed42-20260805 (trial 22, exp 234); Phase 1 panel recipe"),
    "nsgaii": ("NSGA-II", "classical", "results/candidates/nsgaii-seed42-20260805/0013-train.py", "nsgaii_quality",
               "lowest valid compatible val BPB of nsgaii-seed42-20260805 (trial 13, exp 275)"),
    "cmaes": ("CMA-ES", "classical", "results/candidates/cmaes-seed42-20260805/0015-train.py", "cmaes_quality",
              "lowest valid compatible val BPB of cmaes-seed42-20260805 (trial 15, exp 202)"),
    "gpt_a22": ("GPT A22", "gpt56_autopareto", "results/candidates/autopareto-seed0-20260731/0022-train.py",
                "autopareto_292", "GPT-5.6 AutoPareto original campaign seed 0, attempt 22 (AP292, AP-Q); E1 recipe AP-Q"),
    "gpt_a24": ("GPT A24", "gpt56_autopareto", "results/candidates/autopareto-seed0-20260731/0024-train.py",
                "autopareto_294", "GPT-5.6 AutoPareto original campaign seed 0, attempt 24 (AP294, AP-S)"),
    "gpt_a25": ("GPT A25", "gpt56_autopareto", "results/candidates/autopareto-seed0-20260731/0025-train.py",
                "autopareto_299", "GPT-5.6 AutoPareto original campaign seed 0, attempt 25 (AP299, AP-F)"),
}
OPUS = {"opus_a23": ("Opus A23", "OPUS-A23", "23"), "opus_a24": ("Opus A24", "OPUS-A24", "24"),
        "opus_a27": ("Opus A27", "OPUS-A27", "27")}
NEW_TOKEN_RECIPES = ["baseline", "rs", "tpe", "motpe", "nsgaii", "cmaes", "gpt_a24", "gpt_a25"]
NEW_FIXED_RECIPES = NEW_TOKEN_RECIPES + ["opus_a23", "opus_a24", "opus_a27"]
ALL = ["baseline", "arref", "rs", "tpe", "motpe", "nsgaii", "cmaes", "gpt_a22", "gpt_a24", "gpt_a25",
       "opus_a23", "opus_a24", "opus_a27"]
IMPLEMENTATION = [
    "scripts/run_phase6.py",
    "scripts/phase6_unified_inference.py",
    "scripts/make_phase6_plan.py",
    "scripts/run_camera_ready_e0_e1.py",
    "scripts/run_phase1_equal_tokens.py",
    "scripts/run_phase5a_opus_fixed_tokens.py",
    "scripts/phase5a_export_deployed_checkpoints.py",
    "scripts/evaluate_phase1_checkpoints.py",
    "scripts/phase5b_inference_ablation.py",
    "study/contracts_2026/inference_benchmark.py",
    "study/contracts_2026/benchmark_policy.py",
    "protected/inference_adapter.py",
    "protected/phase1_train_launcher.py",
    "protected/phase1_source_transform.py",
    "protected/phase1_runtime.py",
    "downstream_evaluator.py",
    "evaluation/downstream_suite.json",
    "prepare.py",
    "study/contracts_2026/historical_reference/baseline_source/prepare.py",
    "evaluation/PHASE6_PREREG.md",
]


def rotated_orders(keys: list[str]) -> list[list[str]]:
    shuffled = list(keys)
    random.Random(ORDER_SEED).shuffle(shuffled)
    return [shuffled[i:] + shuffled[:i] for i in range(len(SEEDS))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if OUT.exists() and not args.force:
        raise SystemExit(f"{OUT} exists; amend it instead of rewriting")
    e1_plan = json.loads((ROOT / "evaluation/camera_ready_e0_e1_plan_amendment1.json").read_text())
    p5_raw = (ROOT / "evaluation/phase5a_opus_fixed_tokens_plan.json").read_bytes()
    p5 = json.loads(p5_raw)
    recipes = {}
    for key, (display, family, rel, model_id, provenance) in HISTORICAL.items():
        source = ROOT / rel
        recipes[key] = {
            "display": display, "family": family, "source_path": rel, "source_sha256": sha(source),
            "phase1_model_id": model_id, "TOTAL_BATCH_SIZE": batch_of(source), "DEVICE_BATCH_SIZE": 64,
            "inference_context_prepare": "prepare.py", "provenance": provenance,
        }
    for key, (display, p5_key, attempt) in OPUS.items():
        recipe = p5["recipes"][p5_key]
        recipes[key] = {
            "display": display, "family": "opus55_autopareto", "phase5a_recipe": p5_key,
            "source_path": recipe["source_path"], "source_sha256": recipe["source_sha256"],
            "pair_a_source_path": recipe["pair_a_source_path"], "pair_a_source_sha256": recipe["pair_a_source_sha256"],
            "phase1_model_id": recipe["phase1_model_id"], "TOTAL_BATCH_SIZE": p5["TOTAL_BATCH_SIZE"], "DEVICE_BATCH_SIZE": 64,
            "inference_context_prepare": "study/contracts_2026/historical_reference/baseline_source/prepare.py",
            "provenance": f"Claude Opus 5.5 AutoPareto (Pair-A contract T) attempt {attempt}",
        }
    for key in ("arref", "gpt_a22"):
        e1_key = "AR-ref" if key == "arref" else "AP-Q"
        assert e1_plan["recipes"][e1_key]["source_sha256"] == recipes[key]["source_sha256"], key
    reused = []
    for key, e1_name in (("arref", "arref"), ("gpt_a22", "apq")):
        for seed in SEEDS:
            reused.append({"recipe": key, "seed": seed, "cell": "tokens_201326592", "origin": "E1 amendment 1",
                           "run_dir": f"results/camera_ready_2026/amendment1/E1/runs/e1-{e1_name}-tokens201326592-seed{seed}"})
            reused.append({"recipe": key, "seed": seed, "cell": "time_300s", "origin": "E1 amendment 1",
                           "run_dir": f"results/camera_ready_2026/amendment1/E1/runs/e1-{e1_name}-time300s-seed{seed}"})
    for key, (_, p5_key, attempt) in OPUS.items():
        for seed in SEEDS:
            reused.append({"recipe": key, "seed": seed, "cell": "tokens_201326592", "origin": "Phase 5A",
                           "run_dir": f"results/camera_ready_2026/phase5/A/runs/p5a-opus{attempt}-tokens201326592-seed{seed}"})
    runs = []
    order = 0
    for seed, keys in zip(SEEDS, rotated_orders(NEW_TOKEN_RECIPES)):
        for key in keys:
            order += 1
            runs.append({"order": order, "stage": "train", "run_id": f"p6-{key}-tokens201326592-seed{seed}",
                         "recipe": key, "cell": "tokens_201326592", "seed": seed, "role": "new"})
    for key in ("arref", "gpt_a22", "opus_a23"):
        order += 1
        runs.append({"order": order, "stage": "train", "run_id": f"p6-{key}-tokens201326592-seed6-drift",
                     "recipe": key, "cell": "tokens_201326592", "seed": 6, "role": "drift_control"})
    order = 0
    fixed = []
    for seed, keys in zip(SEEDS, rotated_orders(NEW_FIXED_RECIPES)):
        for key in keys:
            order += 1
            fixed.append({"order": order, "stage": "fixed", "run_id": f"p6-{key}-time300s-seed{seed}",
                          "recipe": key, "cell": "time_300s", "seed": seed, "role": "new"})
    order += 1
    fixed.append({"order": order, "stage": "fixed", "run_id": "p6-arref-time300s-seed6-drift",
                  "recipe": "arref", "cell": "time_300s", "seed": 6, "role": "drift_control"})
    inference = []
    order = 0
    for seed, keys in zip(SEEDS, rotated_orders(ALL)):
        for key in keys:
            order += 1
            inference.append({"order": order, "mode": "native", "recipe": key, "seed": seed,
                              "label": f"{key}-seed{seed}-native",
                              "block": "P0a" if key in ("arref", "gpt_a22", "opus_a23", "opus_a24", "opus_a27") else "P0b"})
    for key in rotated_orders(ALL)[0]:
        order += 1
        inference.append({"order": order, "mode": "uncached", "recipe": key, "seed": 6, "label": f"{key}-seed6-uncached",
                          "block": "control"})
    for key in ("opus_a23", "opus_a24", "opus_a27"):
        order += 1
        inference.append({"order": order, "mode": "native_eager_reference", "recipe": key, "seed": 6,
                          "label": f"{key}-seed6-eager", "block": "control"})
        order += 1
        inference.append({"order": order, "mode": "canonical", "recipe": key, "seed": 6,
                          "label": f"{key}-seed6-canonical", "block": "control", "correctness_only": True})
    for key in ("arref", "gpt_a22", "opus_a23"):
        order += 1
        inference.append({"order": order, "mode": "native", "recipe": key, "seed": 6, "label": f"{key}-seed6-native-anchor",
                          "block": "anchor"})
    plan = {
        "format": "autopareto_phase6_unified_plan_v1",
        "prepared_on": "2026-11-16",
        "prose": "evaluation/PHASE6_PREREG.md",
        "output_root": "results/camera_ready_2026/phase6",
        "token_budget": TOKEN_BUDGET,
        "seeds": SEEDS,
        "order_shuffle_seed": ORDER_SEED,
        "comparator_plans": {
            "e1": {"path": "evaluation/camera_ready_e0_e1_plan_amendment1.json",
                   "sha256": sha(ROOT / "evaluation/camera_ready_e0_e1_plan_amendment1.json")},
            "phase5a": {"path": "evaluation/phase5a_opus_fixed_tokens_plan.json", "sha256": hashlib.sha256(p5_raw).hexdigest()},
        },
        "recipes": recipes,
        "candidates_in_order": ALL,
        "reused_runs": reused,
        "runs": runs,
        "fixed_time_runs": fixed,
        "opus_fixed_time": ("Phase 5A mapped source (the Pair-A loop driver restored to the panel baseline's time driver) "
                            "in the unchanged E1 fixed-time cell: identical accounting. If the frozen evaluator cannot "
                            "load a converted checkpoint (risk: A27 int8 tables), the cell is scored on a deployed "
                            "re-export under a documented amendment."),
        "inference": {
            "protocol": "autopareto_search_inference_bracketed_v2",
            "environment": "Env B /opt/autopareto_runtime (torch 2.8.0+cu128, triton 3.4.0)",
            "bracketing_baseline": {
                "context_train": "study/contracts_2026/historical_reference/baseline_source/train.py",
                "context_train_sha256": sha(ROOT / "study/contracts_2026/historical_reference/baseline_source/train.py"),
                "context_prepare_sha256": sha(ROOT / "study/contracts_2026/historical_reference/baseline_source/prepare.py"),
                "checkpoint": "/workspace/inputs/pair_a_baseline_T/result.checkpoint.pt",
                "checkpoint_sha256_prefix": "f7ca17c6",
            },
            "near_tie_logits": 0.25,
            "runs": inference,
        },
        "frozen_downstream": {"evaluator_sha256": sha(ROOT / "downstream_evaluator.py"),
                              "suite_sha256": sha(ROOT / "evaluation/downstream_suite.json")},
        "implementation_sha256": {rel: sha(ROOT / rel) for rel in IMPLEMENTATION if (ROOT / rel).is_file()},
    }
    OUT.write_text(json.dumps(plan, indent=1) + "\n")
    print(OUT, hashlib.sha256(OUT.read_bytes()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
