#!/usr/bin/env python3
"""Phase 5 analysis, implementing exactly the pre-registered plan (evaluation/PHASE5_PREREG.md).

    python3 scripts/analyze_phase5.py [--evidence-root DIR] [--out DIR] [--no-figures]

5A inputs (read-only):
- results/camera_ready_2026/phase5/A/runs/*/phase5a_run_record.json;
- the downstream bundles under phase5/A/downstream;
- E1 run records (amendment1/E1/runs) and Phase 2's August-panel summary (phase2_verification.json).

5B inputs: results/camera_ready_2026/phase5/B/runs/*.json and kernel_census/*.json. The 39 raw benchmark JSONs
are committed as B/runs_json.tar.xz (members runs/*.json, byte-identical to the pod files; see B/SHA256SUMS);
the analysis reads the directory when the files are present and the archive otherwise.

Outputs: <out>/phase5_analysis.json, PHASE5_TABLES.md, phase5_tables.tex and figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
T95 = 4.302652729911275  # t(0.975, df = 2)
BUDGET = 201_326_592
TOKEN_CELLS = (88_604_672, 157_286_400, 201_326_592)
OPUS = ("OPUS-A23", "OPUS-A24", "OPUS-A27")
LABEL = {"AR-ref": "AutoResearch (AR-ref)", "AP-Q": "AP292 (AP-Q)", "OPUS-A23": "Opus 23", "OPUS-A24": "Opus 24", "OPUS-A27": "Opus 27"}
DS_KEYS = (
    ("heldout_climbmix_bpb.bits_per_byte", "Held-out BPB (suite)", -1),
    ("lambada_openai.exact_match", "LAMBADA", 1),
    ("blimp.macro_accuracy", "BLiMP", 1),
    ("piqa.accuracy", "PIQA", 1),
    ("arc_easy.accuracy", "ARC-E", 1),
)
VARIANTS = ("V0_as_submitted", "V1_no_cuda_graph", "V2_no_fusion", "V3_sdpa_attention", "V4_eager_unfused", "V5_linear_kv_cache")
VLABEL = {"V0_as_submitted": "V0 as submitted", "V1_no_cuda_graph": "V1 no CUDA graph", "V2_no_fusion": "V2 no fusion",
          "V3_sdpa_attention": "V3 SDPA attention", "V4_eager_unfused": "V4 eager, unfused", "V5_linear_kv_cache": "V5 linear K/V"}


def load(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(values: list[float]) -> dict:
    n = len(values)
    mean = statistics.fmean(values) if values else None
    sd = statistics.stdev(values) if n > 1 else None
    half = T95 * sd / math.sqrt(n) if (sd is not None and n == 3) else None
    return {"n": n, "values": values, "mean": mean, "sd": sd,
            "ci95": None if half is None else [mean - half, mean + half]}


def t2_two_sided_p(t: float) -> float:
    # Student t with df = 2 has the closed-form CDF 0.5 + t / (2 sqrt(2 + t^2)).
    return 2 * (0.5 - abs(t) / (2 * math.sqrt(2 + t * t)))


def paired(values_a: dict[int, float], values_b: dict[int, float]) -> dict:
    seeds = sorted(set(values_a) & set(values_b))
    diffs = [values_a[s] - values_b[s] for s in seeds]
    out = summary(diffs)
    out["seeds"] = seeds
    if out["sd"]:
        t = out["mean"] / (out["sd"] / math.sqrt(len(diffs)))
        out["t"] = t
        out["p_two_sided"] = t2_two_sided_p(t) if len(diffs) == 3 else None
    out["verdict"] = (
        "confirmed difference" if out["ci95"] and (out["ci95"][0] > 0 or out["ci95"][1] < 0) else "inconclusive"
    )
    return out


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered)
    adjusted, running = {}, 0.0
    for i, (key, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[key] = running
    return adjusted


# ---------------------------------------------------------------------------
# 5A
# ---------------------------------------------------------------------------

def passed_records(runs_dir: Path, name: str) -> dict[str, dict]:
    out = {}
    for path in sorted(runs_dir.glob(f"*/{name}")):
        record = load(path)
        if record.get("integrity", {}).get("passed") and not record.get("synthetic"):
            out[record["run_id"]] = record
    return out


def exchange_rate(b: float, curve: list[tuple[int, float]]) -> dict:
    """Token multiple k = BUDGET / T*, where AR-ref reaches BPB b at T* (piecewise linear in ln tokens)."""
    pts = sorted(curve)
    xs = [math.log(t) for t, _ in pts]
    ys = [y for _, y in pts]
    if b > ys[0]:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        t_star = math.exp(xs[0] + (b - ys[0]) / slope)
        return {"bound": "lower", "k_lower_bound": BUDGET / pts[0][0], "k_linear_extrapolation": BUDGET / t_star,
                "t_star_extrapolated": t_star}
    if b < ys[-1]:
        return {"bound": "upper", "k_upper_bound": 1.0}
    for i in range(len(pts) - 1):
        if ys[i] >= b >= ys[i + 1]:
            frac = (ys[i] - b) / (ys[i] - ys[i + 1]) if ys[i] != ys[i + 1] else 0.0
            t_star = math.exp(xs[i] + frac * (xs[i + 1] - xs[i]))
            return {"bound": None, "k": BUDGET / t_star, "t_star": t_star}
    return {"bound": "non-monotone curve"}


def analyze_5a(evidence: Path) -> dict:
    plan = load(evidence / "evaluation/phase5a_opus_fixed_tokens_plan.json")
    p5 = evidence / plan["output_root"]
    records = passed_records(p5 / "runs", "phase5a_run_record.json")
    e1 = passed_records(evidence / "results/camera_ready_2026/amendment1/E1/runs", "camera_ready_run_record.json")
    result: dict = {"plan_sha256": sha(evidence / "evaluation/phase5a_opus_fixed_tokens_plan.json"),
                    "runs_passed": sorted(records), "runs_expected": [r["run_id"] for r in plan["runs"]]}
    result["runs_missing_or_failed"] = sorted(set(result["runs_expected"]) - set(records))

    per = {k: {"heldout": {}, "validation": {}, "tps": {}, "peak_mb": {}, "milestones": {}, "params": None} for k in ("AR-ref", "AP-Q", *OPUS)}
    for rec in e1.values():
        if rec["token_budget"] == BUDGET and rec["recipe"] in ("AR-ref", "AP-Q"):
            m, s, r = rec["metrics"], rec["seed"], rec["recipe"]
            per[r]["heldout"][s] = m["heldout_bpb"]; per[r]["validation"][s] = m["validation_bpb"]
            per[r]["tps"][s] = m["steady_state_tokens_per_second"]; per[r]["peak_mb"][s] = m["peak_training_memory_mb"]
            per[r]["milestones"][s] = {int(k): v for k, v in m["milestone_validation_bpb"].items()}
            per[r]["params"] = m.get("parameter_count")
    control = None
    for rec in records.values():
        m, s, r = rec["metrics"], rec["seed"], rec["recipe"]
        if rec["role"] == "drift_control":
            control = rec
            continue
        per[r]["heldout"][s] = m["heldout_bpb"]; per[r]["validation"][s] = m["validation_bpb"]
        per[r]["tps"][s] = m["steady_state_tokens_per_second"]; per[r]["peak_mb"][s] = m["peak_training_memory_mb"]
        per[r]["milestones"][s] = {int(k): v for k, v in m["milestone_validation_bpb"].items()}
        per[r]["params"] = m.get("parameter_count")
        per[r].setdefault("in_process_validation", {})[s] = m["validation_bpb_in_process"]
        per[r].setdefault("updates", {})[s] = m["updates"]
        per[r].setdefault("tokens", {})[s] = m["tokens"]

    # Drift control.
    drift = None
    if control is not None:
        ref = next(r for r in e1.values() if r["run_id"] == "e1-arref-tokens201326592-seed6")["metrics"]
        cm = control["metrics"]
        drift = {
            "heldout": [cm["heldout_bpb"], ref["heldout_bpb"], cm["heldout_bpb"] - ref["heldout_bpb"]],
            "validation": [cm["validation_bpb"], ref["validation_bpb"], cm["validation_bpb"] - ref["validation_bpb"]],
            "tokens_equal": cm["tokens"] == ref["tokens"], "updates_equal": cm["updates"] == ref["updates"],
            "tps": [cm["steady_state_tokens_per_second"], ref["steady_state_tokens_per_second"]],
            "checkpoint_identical": cm["checkpoint_sha256"] == ref["checkpoint_sha256"],
        }
        drift["session_confounded"] = (abs(drift["heldout"][2]) > 0.005 or abs(drift["validation"][2]) > 0.005
                                       or not drift["tokens_equal"] or not drift["updates_equal"])
    result["drift_control"] = drift
    # Descriptive reference: AR-ref's own E1 fixed-time cell (300 s, ~88.0M tokens), seeds 6-8.
    t300 = {r["seed"]: r["metrics"]["heldout_bpb"] for r in e1.values() if r["recipe"] == "AR-ref" and r.get("cell") == "time_300s"}
    result["arref_300s_heldout"] = summary([t300[s] for s in sorted(t300)])
    result["arref_300s_heldout"]["by_seed"] = {str(s): t300[s] for s in sorted(t300)}

    result["recipes"] = {}
    for key, d in per.items():
        result["recipes"][key] = {
            "heldout": summary([d["heldout"][s] for s in sorted(d["heldout"])]),
            "validation": summary([d["validation"][s] for s in sorted(d["validation"])]),
            "training_tokens_per_second": summary([d["tps"][s] for s in sorted(d["tps"])]),
            "peak_training_memory_mb": summary([d["peak_mb"][s] for s in sorted(d["peak_mb"])]),
            "parameter_count": d["params"],
            "seeds": sorted(d["heldout"]),
            "by_seed": {str(s): {"heldout": d["heldout"].get(s), "validation": d["validation"].get(s), "tps": d["tps"].get(s),
                                 "in_process_validation": (d.get("in_process_validation") or {}).get(s),
                                 "tokens": (d.get("tokens") or {}).get(s), "updates": (d.get("updates") or {}).get(s)}
                        for s in sorted(d["heldout"])},
            "milestones_mean": {str(t): statistics.fmean(d["milestones"][s][t] for s in d["milestones"])
                                for t in plan["milestone_tokens"]} if d["milestones"] else {},
        }

    # Primary and secondary contrasts.
    contrasts = {}
    for r in OPUS:
        for comp in ("AR-ref", "AP-Q"):
            for metric in ("heldout", "validation"):
                contrasts[f"{r} - {comp} [{metric}]"] = paired(per[r][metric], per[comp][metric])
    for metric in ("heldout", "validation"):  # reference contrast (E1 replication, not part of the Phase 5 family)
        contrasts[f"AP-Q - AR-ref [{metric}]"] = paired(per["AP-Q"][metric], per["AR-ref"][metric])
    primary_keys = [f"{r} - AR-ref [heldout]" for r in OPUS]
    pv = {k: contrasts[k].get("p_two_sided") for k in primary_keys if contrasts[k].get("p_two_sided") is not None}
    for k, adj in holm(pv).items():
        contrasts[k]["p_holm_primary_family"] = adj
    result["contrasts"] = contrasts
    result["primary_keys"] = primary_keys
    if drift and drift["session_confounded"]:
        result["drift_label"] = "session-confounded: contrasts are descriptive only"

    # Throughput and the exploratory exchange rate.
    curves = {}
    for rec in e1.values():
        if rec["recipe"] == "AR-ref" and rec["token_budget"] in TOKEN_CELLS:
            curves.setdefault(rec["seed"], []).append((rec["token_budget"], rec["metrics"]["heldout_bpb"]))
    tradeoff = {}
    for r in ("AP-Q", *OPUS):
        rho = {s: per[r]["tps"][s] / per["AR-ref"]["tps"][s] for s in per[r]["tps"] if s in per["AR-ref"]["tps"]}
        ks = {s: exchange_rate(per[r]["heldout"][s], curves[s]) for s in per[r]["heldout"] if s in curves and len(curves[s]) == 3}
        k_values = [v["k"] for v in ks.values() if v.get("bound") is None]
        tradeoff[r] = {"rho_by_seed": rho, "rho": summary([rho[s] for s in sorted(rho)]),
                       "k_by_seed": ks, "k": summary(k_values) if len(k_values) == len(ks) and k_values else None,
                       "k_bounds": sorted({v.get("bound") for v in ks.values() if v.get("bound")})}
    result["throughput_tradeoff"] = tradeoff

    # Downstream (descriptive).
    ds_dir = p5 / "downstream"
    downstream = {}
    if (ds_dir / "downstream_models.json").is_file():
        models = {row["id"]: row for row in load(ds_dir / "downstream_models.json")["models"] if "id" in row}
        for bundle_path in sorted((ds_dir / "runs").glob("*/results.json")):
            bundle = load(bundle_path)
            model = models.get(bundle["experiment_id"])
            if model is None or model.get("exit") != 0:
                continue
            metrics = {f"{row['task_name']}.{row['metric_name']}": row["metric_value"] for row in bundle["task_results"]}
            recipe = recipe_of_method(model["method"])
            downstream.setdefault(recipe, {})[int(model["seed"])] = {k: metrics.get(k) for k, _, _ in DS_KEYS}
    ds_result = {}
    for recipe, seeds in downstream.items():
        ds_result[recipe] = {k: summary([seeds[s][k] for s in sorted(seeds) if seeds[s][k] is not None]) for k, _, _ in DS_KEYS}
    ds_contrasts = {}
    for r in OPUS:
        for comp in ("AR-ref", "AP-Q"):
            if r in downstream and comp in downstream:
                for k, _, _ in DS_KEYS:
                    ds_contrasts[f"{r} - {comp} [{k}]"] = paired(
                        {s: v[k] for s, v in downstream[r].items()}, {s: v[k] for s, v in downstream[comp].items()})
    result["downstream"] = {"by_recipe": ds_result, "paired": ds_contrasts}

    # Descriptive references: the August panel (seeds 3-5) and Opus's Pair-A fixed-time discovery runs (seed 3).
    p2 = load(ROOT / "results/camera_ready_2026/phase2/phase2_verification.json")  # generated on the Mac, read from the repository
    ref = {}
    for name, key in (("AR-ref", "autoresearch_compatible_tokens201M"), ("AP-Q", "autopareto_292_tokens201M")):
        ss = p2["downstream"][key].get("seed_summary", {})
        ref[name] = {k: ss.get(k, {}).get("mean") for k, _, _ in DS_KEYS}
    for a in (23, 24, 27):
        single = p2["downstream"][f"opus55_t_attempt{a}"]["single"]["metrics"]
        ref[f"OPUS-A{a} Pair-A 300 s seed 3"] = {k: (single.get(k) or {}).get("value") for k, _, _ in DS_KEYS}
    result["descriptive_references"] = ref
    return result


def recipe_of_method(method: str) -> str:
    if method.startswith("opus55_t_attempt"):
        return f"OPUS-A{method.split('opus55_t_attempt')[1][:2]}"
    if method.startswith("autopareto_292"):
        return "AP-Q"
    if method.startswith("autoresearch_compatible"):
        return "AR-ref"
    return method


# ---------------------------------------------------------------------------
# 5B
# ---------------------------------------------------------------------------

def benchmark_run_jsons(out_root: Path):
    """Yield (path, parsed JSON or None) for the 39 benchmark runs, from B/runs or else B/runs_json.tar.xz."""
    files = sorted((out_root / "runs").glob("*.json"))
    if files:
        for path in files:
            try:
                yield path, load(path)
            except json.JSONDecodeError:
                yield path, None
        return
    import tarfile
    with tarfile.open(out_root / "runs_json.tar.xz", "r:xz") as tar:
        for member in sorted(tar.getmembers(), key=lambda m: m.name):
            if not (member.isfile() and member.name.startswith("runs/") and member.name.endswith(".json")):
                continue
            path = out_root / member.name
            try:
                yield path, json.loads(tar.extractfile(member).read())
            except json.JSONDecodeError:
                yield path, None


def analyze_5b(evidence: Path) -> dict:
    out_root = evidence / "results/camera_ready_2026/phase5/B"
    runs = {}
    for path, d in benchmark_run_jsons(out_root):
        stem = path.stem.split("_", 1)[1]  # strip the order prefix
        if d is None:
            continue
        reps = d.get("inference_repetitions") or []
        corr = d.get("correctness") or {}
        model, variant, repeat = parse_id(stem)
        sample = path.with_suffix("").with_suffix(".sample.txt")
        runs[stem] = {
            "model": model, "variant": variant, "repeat": repeat,
            "decode_tps": d.get("cached_decode_tokens_per_second_median"),
            "decode_cv": d.get("cached_decode_coefficient_of_variation"),
            "decode_drift": d.get("decode_block_relative_drift"),
            "stability": d.get("inference_status"),
            "prefill_ms": d.get("prefill_milliseconds"),
            "e2e_tps": statistics.median(r["end_to_end_tokens_per_second"] for r in reps) if reps else None,
            "peak_mib": (d.get("peak_allocated_inference_memory_bytes") or 0) / 2**20,
            "kv_mib": (d.get("kv_cache_bytes") or 0) / 2**20,
            "correct": corr.get("passed"), "verification_mode": corr.get("verification_mode"),
            "sample_sha256": sha(sample) if sample.is_file() else None,
        }
    cells: dict = {}
    for row in runs.values():
        cells.setdefault((row["model"], row["variant"]), []).append(row)
    table = {}
    for (model, variant), rows in sorted(cells.items()):
        def mean_of(key):
            vals = [r[key] for r in rows if r[key] is not None]
            return statistics.fmean(vals) if vals else None
        table[f"{model}/{variant}"] = {
            "model": model, "variant": variant, "n": len(rows),
            **{key: mean_of(key) for key in ("decode_tps", "prefill_ms", "e2e_tps", "peak_mib", "kv_mib")},
            "decode_tps_values": [r["decode_tps"] for r in rows], "prefill_ms_values": [r["prefill_ms"] for r in rows],
            "e2e_tps_values": [r["e2e_tps"] for r in rows],
            "stability": [r["stability"] for r in rows], "correct": [r["correct"] for r in rows],
            "verification_mode": sorted({str(r["verification_mode"]) for r in rows}),
        }
    factors = {}
    base = table.get("pair_a_baseline/canonical")
    for model in ("opus_attempt23", "opus_attempt24", "opus_attempt27"):
        v = {var: table.get(f"{model}/{var}") for var in VARIANTS}
        if not v["V0_as_submitted"]:
            continue
        f = {}
        def ratio(a, b, key):
            return (a[key] / b[key]) if (a and b and a.get(key) and b.get(key)) else None
        for key in ("decode_tps", "e2e_tps"):
            f[key] = {
                "graph (V0/V1)": ratio(v["V0_as_submitted"], v["V1_no_cuda_graph"], key),
                "fusion (V0/V2)": ratio(v["V0_as_submitted"], v["V2_no_fusion"], key),
                "graph+fusion (V0/V4)": ratio(v["V0_as_submitted"], v["V4_eager_unfused"], key),
                "attention kernel (V0/V3)": ratio(v["V0_as_submitted"], v["V3_sdpa_attention"], key),
                "ring buffer (V0/V5)": ratio(v["V0_as_submitted"], v["V5_linear_kv_cache"], key),
                "architecture, eager (V4/baseline)": ratio(v["V4_eager_unfused"], base, key),
                "total (V0/baseline)": ratio(v["V0_as_submitted"], base, key),
            }
            g, fu, j = f[key]["graph (V0/V1)"], f[key]["fusion (V0/V2)"], f[key]["graph+fusion (V0/V4)"]
            f[key]["interaction (joint / (graph x fusion))"] = j / (g * fu) if (g and fu and j) else None
        f["prefill_ms"] = {var: ratio(v[var], v["V0_as_submitted"], "prefill_ms") for var in VARIANTS if v[var]}
        f["prefill_ms"]["V0/baseline"] = ratio(v["V0_as_submitted"], base, "prefill_ms")
        f["kv_mib"] = {"V5/V0": ratio(v["V5_linear_kv_cache"], v["V0_as_submitted"], "kv_mib")}
        f["peak_mib"] = {"V5/V0": ratio(v["V5_linear_kv_cache"], v["V0_as_submitted"], "peak_mib")}
        best = max((var for var in VARIANTS if v[var] and v[var]["e2e_tps"]), key=lambda var: v[var]["e2e_tps"], default=None)
        f["best_end_to_end_variant"] = best
        samples = {var: sorted({r["sample_sha256"] for r in runs.values() if r["model"] == model and r["variant"] == var}) for var in VARIANTS}
        f["greedy_samples_identical_across_variants"] = len({s for vals in samples.values() for s in vals}) == 1
        factors[model] = f
    census = {}
    for path in sorted((out_root / "kernel_census").glob("*.json")):
        d = load(path)
        census[path.stem] = {k: d.get(k) for k in ("gpu_kernels_per_token", "host_kernel_launch_calls_per_token",
                                                   "host_graph_launch_calls_per_token", "profiler_host_self_time_us_per_token")}
    return {"runs": runs, "cells": table, "factors": factors, "kernel_census": census,
            "unstable_runs": sorted(k for k, r in runs.items() if r["stability"] not in ("stable", None)),
            "correctness_failures": sorted(k for k, r in runs.items() if r["correct"] is False)}


def parse_id(stem: str) -> tuple[str, str, int]:
    if stem.startswith("baseline-r"):
        return "pair_a_baseline", "canonical", int(stem[-1])
    head, rep = stem.rsplit("-r", 1)
    attempt, variant = head.split("-", 1)
    return f"opus_attempt{attempt[1:]}", variant, int(rep)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def f4(x, d=4):
    return "—" if x is None else f"{x:.{d}f}"


def ci(s, d=4):
    if not s or s.get("ci95") is None:
        return "—"
    return f"[{s['ci95'][0]:+.{d}f}, {s['ci95'][1]:+.{d}f}]"


def render_md(a: dict, b: dict | None) -> str:
    L = ["# Phase 5 tables (generated by scripts/analyze_phase5.py; do not edit)", ""]
    if a:
        L += ["## 5A: matched tokens (201,326,592 tokens, 1,536 updates, seeds 6–8)", "",
              f"Runs passed: {len(a['runs_passed'])}/{len(a['runs_expected'])}. Missing/failed: {a['runs_missing_or_failed'] or 'none'}.", ""]
        d = a.get("drift_control")
        if d:
            L += [f"Drift control (AR-ref seed 6): held-out {d['heldout'][0]:.4f} vs E1 {d['heldout'][1]:.4f} (Δ {d['heldout'][2]:+.4f}); "
                  f"validation Δ {d['validation'][2]:+.4f}; tokens/updates equal: {d['tokens_equal'] and d['updates_equal']}; "
                  f"checkpoint bit-identical: {d['checkpoint_identical']}; session-confounded: {d['session_confounded']}.", ""]
        L += ["| Recipe | Params | Held-out BPB mean ± sd [95% CI] | Validation BPB | Train tok/s (k) | ρ vs AR-ref | k (exchange rate) |",
              "|---|---|---|---|---|---|---|"]
        for r in ("AR-ref", "AP-Q", *OPUS):
            s = a["recipes"][r]
            if not s["heldout"]["values"]:
                continue
            h, v, t = s["heldout"], s["validation"], s["training_tokens_per_second"]
            tr = a["throughput_tradeoff"].get(r, {})
            rho = tr.get("rho", {}).get("mean")
            k = tr.get("k")
            ktxt = "—" if r == "AR-ref" else (f"{k['mean']:.2f}" if k and k.get("mean") else ", ".join(tr.get("k_bounds") or []) or "—")
            if r != "AR-ref" and not (k and k.get("mean")) and tr.get("k_by_seed"):
                lo = [x.get("k_lower_bound") for x in tr["k_by_seed"].values() if x.get("k_lower_bound")]
                ex = [x.get("k_linear_extrapolation") for x in tr["k_by_seed"].values() if x.get("k_linear_extrapolation")]
                if lo:
                    ktxt = f"> {min(lo):.2f} (extrap. {statistics.fmean(ex):.2f})"
            L.append(f"| {LABEL[r]} | {(s['parameter_count'] or 0)/1e6:.1f}M | {h['mean']:.4f} ± {f4(h['sd'])} {ci(h).replace('+','')} | "
                     f"{v['mean']:.4f} ± {f4(v['sd'])} | {t['mean']/1e3:.0f} | {'1' if r == 'AR-ref' else f4(rho, 2)} | {ktxt} |")
        L += ["", "| Contrast (paired by seed) | Mean Δ | 95% CI | per seed | p (Holm, primary) | verdict |", "|---|---|---|---|---|---|"]
        for key, c in a["contrasts"].items():
            if c["n"] == 0:
                continue
            per = ", ".join(f"{x:+.4f}" for x in c["values"])
            holm_p = c.get("p_holm_primary_family")
            L.append(f"| {key}{' **(primary)**' if key in a['primary_keys'] else ''} | {c['mean']:+.4f} | {ci(c)} | {per} | "
                     f"{'—' if holm_p is None else f'{holm_p:.3f}'} | {c['verdict']} |")
        ds = a["downstream"]["by_recipe"]
        if ds:
            L += ["", "Downstream (frozen suite; mean ± sd over seeds 6–8, descriptive):", "",
                  "| Recipe | " + " | ".join(lbl for _, lbl, _ in DS_KEYS) + " |", "|---" * (len(DS_KEYS) + 1) + "|"]
            for r in ("AR-ref", "AP-Q", *OPUS):
                if r in ds:
                    L.append(f"| {LABEL[r]} | " + " | ".join(f"{ds[r][k]['mean']:.3f} ± {f4(ds[r][k]['sd'], 3)}" if ds[r][k]["values"] else "—" for k, _, _ in DS_KEYS) + " |")
            L += ["", "| Paired downstream contrast | " + " | ".join(lbl for _, lbl, _ in DS_KEYS) + " |", "|---" * (len(DS_KEYS) + 1) + "|"]
            for r in OPUS:
                for comp in ("AR-ref", "AP-Q"):
                    cells = [a["downstream"]["paired"].get(f"{r} - {comp} [{k}]") for k, _, _ in DS_KEYS]
                    if all(cells):
                        L.append(f"| {LABEL[r]} − {LABEL[comp]} | " + " | ".join(f"{c['mean']:+.3f} {ci(c, 3)}" for c in cells) + " |")
    if b and b.get("cells"):
        L += ["", "## 5B: inference ablations (existing Pair-A checkpoints; protected benchmark, candidate-only)", "",
              "| Model / variant | Decode tok/s (runs) | Prefill ms | End-to-end tok/s | Peak MiB | K/V MiB | Stable | Correct (mode) |",
              "|---|---|---|---|---|---|---|---|"]
        for key, c in b["cells"].items():
            L.append(f"| {c['model'].replace('opus_attempt', 'Opus ')} / {VLABEL.get(c['variant'], c['variant'])} | {c['decode_tps']:.0f} "
                     f"({', '.join(f'{x:.0f}' for x in c['decode_tps_values'])}) | {c['prefill_ms']:.1f} | {c['e2e_tps']:.0f} | "
                     f"{c['peak_mib']:.0f} | {c['kv_mib']:.2f} | {'/'.join(str(s) for s in c['stability'])} | "
                     f"{'/'.join(str(x) for x in c['correct'])} ({', '.join(c['verification_mode'])}) |")
        L += ["", "Multiplicative factors (decode tok/s; end-to-end tok/s in parentheses):", ""]
        for model, f in b["factors"].items():
            L.append(f"- **{model.replace('opus_attempt', 'Opus ')}**: " + "; ".join(
                f"{name} {f4(f['decode_tps'][name], 2)}× ({f4(f['e2e_tps'].get(name), 2)}×)" for name in f["decode_tps"]) +
                f"; best end-to-end: {f['best_end_to_end_variant']}; greedy samples identical across variants: {f['greedy_samples_identical_across_variants']}.")
        if b["kernel_census"]:
            L += ["", "| Kernel census (attempt 23) | GPU kernels / token | host launches / token | graph launches / token | host µs / token |", "|---|---|---|---|---|"]
            for k, c in b["kernel_census"].items():
                L.append(f"| {k} | {f4(c['gpu_kernels_per_token'], 1)} | {f4(c['host_kernel_launch_calls_per_token'], 1)} | "
                         f"{f4(c['host_graph_launch_calls_per_token'], 1)} | {f4(c['profiler_host_self_time_us_per_token'], 0)} |")
        L += ["", f"Unstable runs (frozen rule, reported not re-run): {b['unstable_runs'] or 'none'}. Correctness failures: {b['correctness_failures'] or 'none'}."]
    return "\n".join(L) + "\n"


def render_tex(a: dict, b: dict | None) -> str:
    """Paper-ready booktabs tables (numbers only from the analysis JSON)."""
    L = ["% Generated by scripts/analyze_phase5.py from results/camera_ready_2026/phase5/analysis/phase5_analysis.json; do not edit."]
    if a:
        def d(c):
            return "---" if not c or c.get("mean") is None else f"${c['mean']:+.3f}$ [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]"
        L += [r"\begin{tabular}{@{}lrrcccc@{}}", r"\toprule",
              r"Recipe & Params & Train tok/s & Held-out BPB & $\Delta$ vs AR-ref & $\Delta$ vs AP-Q & BLiMP \\",
              r"\midrule"]
        ds = a["downstream"]["by_recipe"]
        for r in ("AR-ref", "AP-Q", *OPUS):
            s_ = a["recipes"][r]
            if not s_["heldout"]["values"]:
                continue
            h = s_["heldout"]
            vs_ar = a["contrasts"].get(f"{r} - AR-ref [heldout]")
            vs_ap = a["contrasts"].get(f"{r} - AP-Q [heldout]")
            bl = ds.get(r, {}).get("blimp.macro_accuracy", {})
            bl_text = "---" if not bl.get("values") else f"{bl['mean']:.3f}"
            L.append(f"{LABEL[r]} & {s_['parameter_count']/1e6:.1f}M & {s_['training_tokens_per_second']['mean']/1e3:.0f}k & "
                     f"${h['mean']:.4f} \\pm {h['sd']:.4f}$ & {d(vs_ar)} & {d(vs_ap)} & "
                     f"{bl_text} \\\\")
        L += [r"\bottomrule", r"\end{tabular}", ""]
    if b and b.get("factors"):
        L += [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
              r"Opus 23 adapter variant & Decode tok/s & Prefill ms & End-to-end tok/s & Peak MiB & Decode vs V0 \\", r"\midrule"]
        cells = b["cells"]
        v0 = cells.get("opus_attempt23/V0_as_submitted")
        for var in VARIANTS:
            c = cells.get(f"opus_attempt23/{var}")
            if not c:
                continue
            L.append(f"{VLABEL[var]} & {c['decode_tps']:.0f} & {c['prefill_ms']:.1f} & {c['e2e_tps']:.0f} & {c['peak_mib']:.0f} & "
                     f"{c['decode_tps']/v0['decode_tps']:.2f}$\\times$ \\\\")
        base = cells.get("pair_a_baseline/canonical")
        if base:
            L.append(r"\midrule")
            L.append(f"Pair-A baseline (8 layers, canonical eager) & {base['decode_tps']:.0f} & {base['prefill_ms']:.1f} & "
                     f"{base['e2e_tps']:.0f} & {base['peak_mib']:.0f} & {base['decode_tps']/v0['decode_tps']:.3f}$\\times$ \\\\")
        L += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

COLOR = {"AR-ref": "#eb6834", "AP-Q": "#2a78d6", "OPUS-A23": "#6250d6", "OPUS-A24": "#6250d6", "OPUS-A27": "#6250d6"}
MARK = {"AR-ref": "o", "AP-Q": "s", "OPUS-A23": "D", "OPUS-A24": "^", "OPUS-A27": "v"}


def figures(a: dict, b: dict | None, out: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    ink, muted = "#222222", "#6b6b6b"
    made = []
    if a:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.6, 2.1), gridspec_kw={"width_ratios": [1.1, 1]})
        order = [r for r in ("AR-ref", "AP-Q", *OPUS) if a["recipes"][r]["heldout"]["values"]]
        for i, r in enumerate(order):
            h = a["recipes"][r]["heldout"]
            ax1.scatter([i] * len(h["values"]), h["values"], s=12, color=COLOR[r], marker=MARK[r], alpha=0.55, linewidths=0, zorder=2)
            if h["ci95"]:
                ax1.errorbar(i + 0.22, h["mean"], yerr=[[h["mean"] - h["ci95"][0]], [h["ci95"][1] - h["mean"]]], fmt=MARK[r],
                             color=COLOR[r], ms=4, capsize=2, lw=0.9, zorder=3)
        ref300 = a.get("arref_300s_heldout") or {}
        if ref300.get("mean"):
            for ax in (ax1, ax2):
                ax.axhline(ref300["mean"], color=COLOR["AR-ref"], lw=0.8, ls=(0, (3, 2)), zorder=1)
            ax1.text(-0.3, ref300["mean"] - 0.0009, "AR-ref, 300 s (88M tok.)", fontsize=6, color=ink, ha="left", va="top")
        ax1.set_xticks(range(len(order)), [LABEL[r].replace(" (", "\n(") for r in order], fontsize=6.5)
        ax1.set_ylabel("Held-out BPB at 201.3M tokens (↓)")
        ax1.set_title("(a) Matched tokens, seeds 6–8", loc="left", fontsize=7.5, color=ink)
        ax1.grid(axis="y", lw=0.4, color="#e3e3e3")
        for r in order:
            s = a["recipes"][r]
            xs = [s["by_seed"][k]["tps"] / 1e3 for k in s["by_seed"]]
            ys = [s["by_seed"][k]["heldout"] for k in s["by_seed"]]
            ax2.scatter(xs, ys, s=12, color=COLOR[r], marker=MARK[r], alpha=0.55, linewidths=0)
            ax2.scatter([statistics.fmean(xs)], [statistics.fmean(ys)], s=26, color=COLOR[r], marker=MARK[r], edgecolors="white", linewidths=0.6, zorder=3)
            if r in ("AR-ref", "AP-Q", "OPUS-A24"):
                text = "Opus 23 / 24 / 27" if r == "OPUS-A24" else LABEL[r].split(" (")[0]
                ax2.annotate(text, (statistics.fmean(xs), statistics.fmean(ys)), xytext=(-10, 3) if r == "OPUS-A24" else (4, 5),
                             textcoords="offset points", fontsize=6.3, color=ink, ha="right" if r == "OPUS-A24" else "left",
                             va="bottom" if r == "OPUS-A24" else "baseline")
        ax2.set_xlabel("Training throughput (k tok/s)")
        ax2.set_title("(b) Quality vs training throughput", loc="left", fontsize=7.5, color=ink)
        ax2.grid(lw=0.4, color="#e3e3e3")
        fig.tight_layout(pad=0.4, w_pad=1.2)
        for ext in ("pdf", "png"):
            fig.savefig(out / f"phase5a_matched_tokens.{ext}", dpi=300)
            made.append(str(out / f"phase5a_matched_tokens.{ext}"))
        plt.close(fig)
    if b and b.get("cells"):
        fig, axes = plt.subplots(1, 3, figsize=(5.6, 2.2), sharey=True)
        base = b["cells"].get("pair_a_baseline/canonical")
        models = ("opus_attempt23", "opus_attempt24", "opus_attempt27")
        shades = {"opus_attempt23": "#6250d6", "opus_attempt24": "#8f83e3", "opus_attempt27": "#b9b1ef"}
        metrics = (("decode_tps", "Decode tok/s (log)", True), ("prefill_ms", "Prefill ms (log)", True), ("e2e_tps", "End-to-end tok/s (log)", True))
        for ax, (key, title, logx) in zip(axes, metrics):
            for j, model in enumerate(models):
                vals = [b["cells"].get(f"{model}/{v}", {}).get(key) for v in VARIANTS]
                ys = [i + (j - 1) * 0.22 for i in range(len(VARIANTS))]
                pts = [(v, y) for v, y in zip(vals, ys) if v]
                ax.scatter([v for v, _ in pts], [y for _, y in pts], s=16, color=shades[model], marker=("D", "^", "v")[j],
                           edgecolors="white", linewidths=0.4, zorder=3, label=model.replace("opus_attempt", "Opus "))
            if base and base.get(key):
                ax.axvline(base[key], color=muted, lw=0.8, ls="--")
                ax.text(base[key], -0.55, " baseline", fontsize=6, color=muted, va="center")
            if logx:
                from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
                ax.set_xscale("log")
                ticks = {"decode_tps": [100, 300, 1000, 3000], "prefill_ms": [3, 10, 30, 100], "e2e_tps": [150, 300, 600]}[key]
                ax.xaxis.set_major_locator(FixedLocator(ticks))
                ax.set_xticklabels([f"{t:g}" for t in ticks])
                ax.xaxis.set_minor_locator(NullLocator())
                ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_title(title, loc="left", fontsize=7.2, color=ink)
            ax.grid(axis="x", lw=0.4, color="#e3e3e3", which="both")
        axes[0].set_yticks(range(len(VARIANTS)), [VLABEL[v] for v in VARIANTS], fontsize=6.5)
        axes[0].invert_yaxis()
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=6.3, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.6, -0.01))
        fig.tight_layout(pad=0.4, w_pad=0.8, rect=(0, 0.07, 1, 1))
        for ext in ("pdf", "png"):
            fig.savefig(out / f"phase5b_inference_ablations.{ext}", dpi=300)
            made.append(str(out / f"phase5b_inference_ablations.{ext}"))
        plt.close(fig)
    return made


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "results/camera_ready_2026/phase5/analysis")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--skip-5b", action="store_true")
    args = parser.parse_args()
    a = analyze_5a(args.evidence_root)
    b5 = args.evidence_root / "results/camera_ready_2026/phase5/B"
    b = None if args.skip_5b or not ((b5 / "runs").is_dir() or (b5 / "runs_json.tar.xz").is_file()) else analyze_5b(args.evidence_root)
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {"format": "autopareto_phase5_analysis_v1", "phase5a": a, "phase5b": b}
    (args.out / "phase5_analysis.json").write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n")
    (args.out / "PHASE5_TABLES.md").write_text(render_md(a, b))
    (args.out / "phase5_tables.tex").write_text(render_tex(a, b))
    if not args.no_figures:
        figures(a, b, args.out)
    print((args.out / "PHASE5_TABLES.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
