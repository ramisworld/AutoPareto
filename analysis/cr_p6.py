#!/usr/bin/env python3
"""Phase 6 unified 13-model dataset, statistics and quality-versus-decode Pareto frontier.

Reads only recorded artefacts (Phase 6 records, reused E1 and Phase 5A records, the frozen downstream
results, the 1,024/128 inference records and lm-eval outputs). A value that was not recorded stays
None; nothing is estimated or filled in.

    python3 analysis/cr_p6.py            # writes results/camera_ready_2026/phase6/analysis/
"""

from __future__ import annotations

import csv
import json
import math
import sqlite3
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results/camera_ready_2026"
P6 = R / "phase6"
OUT = P6 / "analysis"
PLAN = json.loads((ROOT / "evaluation/phase6_unified_plan.json").read_text())
SEEDS = PLAN["seeds"]
ORDER = PLAN["candidates_in_order"]
DISPLAY = {k: v["display"] for k, v in PLAN["recipes"].items()}
FAMILY = {k: v["family"] for k, v in PLAN["recipes"].items()}
LMEVAL_TASKS = ["arc_challenge", "hellaswag", "winogrande", "openbookqa", "sciq", "boolq"]
FROZEN = {"arc_easy": ("arc_easy", "accuracy"), "piqa": ("piqa", "accuracy"),
          "lambada": ("lambada_openai", "exact_match"), "blimp": ("blimp", "macro_accuracy"),
          "heldout_frozen": ("heldout_climbmix_bpb", "bits_per_byte")}


def rj(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def reused(recipe: str, seed: int, cell: str) -> dict | None:
    for run in PLAN["reused_runs"]:
        if (run["recipe"], run["seed"], run["cell"]) == (recipe, seed, cell):
            d = ROOT / run["run_dir"]
            for name in ("camera_ready_run_record.json", "phase5a_run_record.json"):
                if (d / name).is_file():
                    return {"record": rj(d / name), "source": run["origin"], "run_dir": run["run_dir"]}
            raise FileNotFoundError(d)
    return None


def tokens_cell(recipe: str, seed: int) -> dict:
    hit = reused(recipe, seed, "tokens_201326592")
    if hit:
        return hit
    run_id = f"p6-{recipe}-tokens201326592-seed{seed}"
    d = P6 / "runs" / (run_id + ("-attempt3" if (recipe, seed) == ("motpe", 8) else ""))
    rec = rj(d / "phase6_run_record.json")
    assert rec["integrity"]["passed"], d
    return {"record": rec, "source": "Phase 6 A2" if (recipe, seed) == ("motpe", 8) else "Phase 6 P1",
            "run_dir": str(d.relative_to(ROOT))}


def fixed_cell(recipe: str, seed: int) -> dict | None:
    hit = reused(recipe, seed, "time_300s")
    if hit:
        return hit
    d = P6 / "runs" / f"p6-{recipe}-time300s-seed{seed}"
    if not (d / "phase6_run_record.json").is_file():
        return None
    rec = rj(d / "phase6_run_record.json")
    assert rec["integrity"]["passed"], d
    return {"record": rec, "source": "Phase 6 P2", "run_dir": str(d.relative_to(ROOT))}


def frozen_downstream() -> dict[str, dict]:
    """checkpoint sha256 -> {metric: value}, from Phase 5A (reused) and Phase 6 (new) frozen runs."""
    by_sha = {}
    sources = [(R / "phase5/A/downstream/runs", None), (P6 / "downstream_frozen/runs", None)]
    for runs_dir, _ in sources:
        for f in sorted(runs_dir.glob("*/results.json")):
            res = rj(f)
            vals = {}
            for key, (task, metric) in FROZEN.items():
                hits = [t["metric_value"] for t in res["task_results"] if t["task_name"] == task and t["metric_name"] == metric]
                vals[key] = float(hits[0]) if hits else None
            by_sha[res["checkpoint_sha256"]] = vals | {"downstream_results": str(f.relative_to(ROOT))}
    return by_sha


def inference(recipe: str, seed: int) -> dict | None:
    f = P6 / "inference/runs" / f"{recipe}-seed{seed}-native.json"
    if not f.is_file():
        return None
    x = rj(f)
    s = x["summary"]
    fwc = x["full_workload_correctness"]
    return {"status": x["status"], "hard_mismatches": fwc.get("hard_mismatches"), "correct": fwc.get("passed"),
            "decode_tok_s": s["decode_tokens_per_second"], "decode_tok_s_ci95": s["decode_tokens_per_second_ci95"],
            "speedup_vs_bracket": s["speedup_vs_bracketing_baseline"], "speedup_ci95": s["speedup_ci95"],
            "bracket_tok_s": s["bracketing_baseline_tokens_per_second"],
            "e2e_tok_s": s["end_to_end_tokens_per_second_median"], "e2e_ms": s["end_to_end_milliseconds_median"],
            "prefill_ms": s["prefill_milliseconds_median"], "peak_allocated_mib": s["peak_allocated_mib"],
            "weight_mib": s["weight_mib"], "kv_cache_mib": s["kv_cache_mib"],
            "stability": s.get("normalized_inference_status"), "record": str(f.relative_to(ROOT))}


def lmeval(recipe: str, seed: int) -> dict | None:
    f = P6 / "lmeval/runs" / f"{recipe}-seed{seed}.json"
    if not f.is_file():
        return None
    res = rj(f)["results"]
    out = {t: float(res[t]["acc,none"]) for t in LMEVAL_TASKS if t in res}
    if "wikitext" in res:
        out["wikitext_word_ppl"] = float(res["wikitext"]["word_perplexity,none"])
    return out | {"lmeval_record": str(f.relative_to(ROOT))}


def build() -> list[dict]:
    ds = frozen_downstream()
    rows = []
    for recipe in ORDER:
        for seed in SEEDS:
            tok = tokens_cell(recipe, seed)
            m = tok["record"]["metrics"]
            ck = m.get("deployed_final_checkpoint_sha256") or m["checkpoint_sha256"]
            down = ds.get(ck) or ds.get(m["checkpoint_sha256"])
            fx = fixed_cell(recipe, seed)
            fm = fx["record"]["metrics"] if fx else {}
            inf = inference(recipe, seed)
            rows.append({
                "recipe": recipe, "display": DISPLAY[recipe], "family": FAMILY[recipe], "seed": seed,
                "params": m.get("parameter_count"),
                "tokens_source": tok["source"], "tokens_run_dir": tok["run_dir"],
                "heldout_bpb": m["heldout_bpb"], "heldout_ci95": m.get("heldout_ci95"),
                "train_tok_s": m.get("steady_state_tokens_per_second"),
                "peak_train_mb": m.get("peak_training_memory_mb"),
                "fixed_source": fx["source"] if fx else None, "fixed_run_dir": fx["run_dir"] if fx else None,
                "fixed_heldout_bpb": fm.get("heldout_bpb"), "fixed_tokens": fm.get("tokens"),
                "fixed_train_tok_s": fm.get("steady_state_tokens_per_second"),
                "fixed_timing_anomaly_s": (fm.get("timing_anomaly") or {}).get("excess_seconds"),
                **{f"frozen_{k}": (down or {}).get(k) for k in FROZEN},
                "frozen_results": (down or {}).get("downstream_results"),
                **{f"inf_{k}": v for k, v in (inf or {}).items()},
                **{f"lm_{k}": v for k, v in (lmeval(recipe, seed) or {}).items()},
            })
    return rows


def summarise(rows: list[dict]) -> list[dict]:
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float)) and k not in ("seed",)]
    keys += sorted({k for r in rows for k in r if k.startswith("lm_") and isinstance(r[k], float)} - set(keys))
    out = []
    for recipe in ORDER:
        group = [r for r in rows if r["recipe"] == recipe]
        row = {"recipe": recipe, "display": DISPLAY[recipe], "family": FAMILY[recipe], "n_seeds": len(group)}
        for k in keys:
            vals = [r.get(k) for r in group if isinstance(r.get(k), (int, float))]
            row[k + "_mean"] = statistics.mean(vals) if vals else None
            row[k + "_sd"] = statistics.stdev(vals) if len(vals) > 1 else None
            row[k + "_n"] = len(vals)
        out.append(row)
    return out


def pareto(summary: list[dict]) -> list[str]:
    """Non-dominated recipes: lower mean held-out BPB (matched tokens) and higher mean native decode tok/s."""
    pts = [(s["recipe"], s["heldout_bpb_mean"], s["inf_decode_tok_s_mean"]) for s in summary
           if s["heldout_bpb_mean"] is not None and s["inf_decode_tok_s_mean"] is not None]
    front = []
    for name, q, v in pts:
        dominated = any((q2 <= q and v2 >= v) and (q2 < q or v2 > v) for _, q2, v2 in pts)
        if not dominated:
            front.append(name)
    return front


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = build()
    summary = summarise(rows)
    front = pareto(summary)
    missing = [(r["recipe"], r["seed"], k) for r in rows for k in ("heldout_bpb", "fixed_heldout_bpb", "frozen_arc_easy",
               "frozen_blimp", "inf_decode_tok_s") if r.get(k) is None]
    per_seed = {}
    for key in ("heldout_bpb", "fixed_heldout_bpb"):
        per_seed[key] = {}
        for seed in SEEDS:
            pts = [(r["recipe"], r[key], r["inf_decode_tok_s"]) for r in rows if r["seed"] == seed]
            per_seed[key][str(seed)] = [n for n, q, v in pts
                                        if not any(q2 <= q and v2 >= v and (q2 < q or v2 > v) for _, q2, v2 in pts)]
    stability = {"unstable_runs": [(r["recipe"], r["seed"]) for r in rows if r["inf_stability"] != "stable"],
                 "bracket_tok_s": [min(r["inf_bracket_tok_s"] for r in rows), max(r["inf_bracket_tok_s"] for r in rows)]}
    dev = []
    for recipe in ORDER:
        g = [r for r in rows if r["recipe"] == recipe]
        stab = [r["inf_decode_tok_s"] for r in g if r["inf_stability"] == "stable"]
        dev += [abs(r["inf_decode_tok_s"] / statistics.mean(stab) - 1) for r in g if r["inf_stability"] != "stable" and stab]
    stability["max_rel_dev_unstable_vs_stable"] = max(dev) if dev else None
    stability["max_seed_cv"] = max(statistics.stdev([r["inf_decode_tok_s"] for r in rows if r["recipe"] == x]) /
                                   statistics.mean([r["inf_decode_tok_s"] for r in rows if r["recipe"] == x]) for x in ORDER)
    correctness = {"runs": len(rows), "passed": sum(1 for r in rows if r["inf_correct"]),
                   "hard_mismatches": sum(r["inf_hard_mismatches"] for r in rows)}
    for r in rows:
        x = json.loads((ROOT / r["inf_record"]).read_text())["full_workload_correctness"]
        correctness["near_ties"] = correctness.get("near_ties", 0) + x["near_tie_disagreements"]
        correctness["max_abs_logit_error"] = max(correctness.get("max_abs_logit_error", 0), x["max_abs_logit_error"])
    fixed_anomaly = max(r["fixed_timing_anomaly_s"] or 0 for r in rows if r["fixed_source"] == "Phase 6 P2")
    payload = {"format": "autopareto_phase6_unified_v1", "pareto_per_seed": per_seed, "inference_stability": stability,
               "inference_correctness": correctness, "fixed_max_timing_excess_s": fixed_anomaly, "plan_sha256": "dc1e605e82c5f598bbf91ab17f63d1db30a27511f5d264142c9b31eb6fe61096",
               "rows": rows, "summary": summary, "pareto_quality_vs_decode": front, "missing": missing}
    (OUT / "unified_13.json").write_text(json.dumps(payload, indent=1, default=str) + "\n")
    with (OUT / "unified_13_rows.csv").open("w", newline="") as f:
        cols = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 999)
        w = csv.DictWriter(f, lineterminator="\n", fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({"rows": len(rows), "pareto": front, "missing": missing[:20], "n_missing": len(missing)}))
    for s in summary:
        print(f"{s['display']:14s} bpb={s['heldout_bpb_mean']:.4f}±{s['heldout_bpb_sd'] or 0:.4f} "
              f"fixed={s['fixed_heldout_bpb_mean'] if s['fixed_heldout_bpb_mean'] is None else round(s['fixed_heldout_bpb_mean'],4)} "
              f"decode={s['inf_decode_tok_s_mean']:.0f} x{s['inf_speedup_vs_bracket_mean']:.2f} e2e={s['inf_e2e_tok_s_mean']:.0f} "
              f"arcE={s['frozen_arc_easy_mean']:.3f} blimp={s['frozen_blimp_mean']:.3f} params={s['params_mean']/1e6:.1f}M")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
