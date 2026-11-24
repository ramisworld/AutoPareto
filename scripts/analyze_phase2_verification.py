#!/usr/bin/env python3
"""Phase 2 (camera-ready, 2026-11-07): build the paper-ready comparison from raw evidence only. Reads; never measures.

Inputs (all hash-recorded in the output):
  * Opus campaign records (protected BPB, tokens, discovery-time inference) for attempts 23, 24, 27;
  * Phase 2A: protected BPB re-evaluation and three bracketed inference repeats per winner, on pod4;
  * Phase 2B: the frozen downstream suite (evaluator 696f9c46, suite 3dce2c73) on six checkpoints, on pod4;
  * August 2026 201M-token panel raw bundles (same evaluator and suite) for AP292, AutoResearch and the Phase-1
    baseline, seeds 3-5, reused unchanged;
  * Phase 2C: held-out loss by token position, and the BLiMP paradigm -> phenomenon map.

usage: python3 -B scripts/analyze_phase2_verification.py --pod-copy DIR --winner-dir DIR --august-bundles DIR \
           --campaign-results DIR --out-dir DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

T95_N3 = 4.302652729911275  # two-sided 95% t quantile, 2 degrees of freedom
AUGUST_RUNS = {  # downstream_run_id -> (model, seed); from downstream_200m.db (snapshot 2026-09-01)
    14: ("autopareto_292", 3), 15: ("autopareto_292", 4), 16: ("autopareto_292", 5),
    23: ("autoresearch_compatible", 3), 24: ("autoresearch_compatible", 4), 25: ("autoresearch_compatible", 5),
    26: ("phase1_baseline", 3), 27: ("phase1_baseline", 4), 28: ("phase1_baseline", 5),
}
WINNERS = {23: "046", 24: "047", 27: "052"}
P2_DOWNSTREAM = {1: "opus55_t_attempt23", 2: "opus55_t_attempt24", 3: "opus55_t_attempt27", 4: "pair_a_baseline_t",
                 5: "autoresearch_compatible_time300s", 6: "autopareto_292_tokens201M_repro"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(correct: float, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z = 1.959963984540054
    p = correct / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def rep_median(result: dict, key: str):
    """Median of a field over the evaluator's timed repetitions (each one prompt: 1,024-token prefill + 128 tokens)."""
    vals = [r[key] for r in result.get("inference_repetitions") or [] if r.get(key) is not None]
    return statistics.median(vals) if vals else None


def seed_summary(values: list[float]) -> dict:
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else None
    half = T95_N3 * sd / math.sqrt(len(values)) if sd is not None and len(values) == 3 else None
    return {"mean": mean, "sd": sd, "n": len(values), "ci95": None if half is None else [mean - half, mean + half],
            "values": values}


def task_metrics(bundle: dict) -> dict:
    out = {}
    for row in bundle["task_results"]:
        key = f"{row['task_name']}.{row['metric_name']}"
        item = {"value": row["metric_value"], "se": row.get("metric_standard_error"), "n": row.get("sample_count"),
                "ci": [row.get("confidence_low"), row.get("confidence_high")]}
        raw = row.get("raw") or {}
        if "correct" in raw and row.get("sample_count"):
            item["correct"] = raw["correct"]
            item["ci"] = list(wilson(raw["correct"], int(row["sample_count"])))
        if row["task_name"] == "blimp" and row["metric_name"] == "micro_accuracy":
            item["configurations"] = raw["configurations"]
            correct = sum(c["correct"] for c in raw["configurations"].values())
            count = sum(c["count"] for c in raw["configurations"].values())
            item["ci"] = list(wilson(correct, count))
        out[key] = item
    return out


def blimp_phenomena(configs: dict, paradigm_map: dict) -> dict:
    pooled: dict[str, list[int]] = {}
    for name, c in configs.items():
        term = paradigm_map[name]["linguistics_term"]
        pooled.setdefault(term, [0, 0])
        pooled[term][0] += c["correct"]
        pooled[term][1] += c["count"]
    return {term: {"accuracy": k / n, "correct": k, "count": n, "wilson95": list(wilson(k, n))}
            for term, (k, n) in sorted(pooled.items())}


def bootstrap_bucket_diffs(a: dict, b: dict, reps: int = 1000, seed: int = 20261107) -> list[dict]:
    """Paired (same segments) bootstrap of BPB(a) - BPB(b) per position bucket, and of the late-minus-early change."""
    rng = random.Random(seed)
    nseg = len(a["per_segment_nats"][0])
    nb = len(a["buckets"])
    ln2 = math.log(2)

    def bpb(model, idx, bucket):
        n = sum(model["per_segment_nats"][bucket][i] for i in idx)
        d = sum(model["per_segment_bytes"][bucket][i] for i in idx)
        return n / (ln2 * d)

    full = list(range(nseg))
    point = [bpb(a, full, k) - bpb(b, full, k) for k in range(nb)]
    samples = [[] for _ in range(nb)]
    late_minus_early = []
    for _ in range(reps):
        idx = [rng.randrange(nseg) for _ in range(nseg)]
        diffs = [bpb(a, idx, k) - bpb(b, idx, k) for k in range(nb)]
        for k in range(nb):
            samples[k].append(diffs[k])
        late_minus_early.append(diffs[nb - 1] - diffs[1])
    q = lambda xs: [sorted(xs)[int(0.025 * len(xs))], sorted(xs)[int(0.975 * len(xs)) - 1]]
    rows = [{"bucket": a["buckets"][k], "diff": point[k], "ci95": q(samples[k])} for k in range(nb)]
    rows.append({"bucket": "late[1024,2048) minus early[128,256) of the difference", "diff": point[nb - 1] - point[1],
                 "ci95": q(late_minus_early)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod-copy", type=Path, required=True, help="Mac copy of pod4 /workspace/.../phase2")
    ap.add_argument("--winner-dir", type=Path, required=True, help="Mac copy of /var/lib/autopareto/phase2_winner_verification_*")
    ap.add_argument("--august-bundles", type=Path, required=True)
    ap.add_argument("--campaign-results", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--baseline-probe-dir", type=Path, default=None,
                    help="pod4 inference-only probe repeats (baseline recipe as candidate): same-pod end-to-end reference")
    args = ap.parse_args()
    inputs: dict[str, str] = {}

    def load(path: Path) -> dict:
        inputs[str(path)] = sha(path)
        return json.loads(path.read_text())

    report: dict = {"format": "autopareto_phase2_verification_v1", "winners": {}, "downstream": {}, "blimp": {},
                    "context_position": {}, "inputs": inputs}

    # ---- 2A: winners
    for attempt, pos in WINNERS.items():
        rec = load(args.campaign_results / f"position-{pos}-pair-a-fixed-time-seed3.json")
        ir = rec["inference_result"]
        disc = {"protected_bpb": rec["protected_validation_bpb"], "decode_tok_s": ir["cached_decode_tokens_per_second_median"],
                "speedup": ir["normalized_decode_throughput"], "status": ir["normalized_inference_status"],
                "e2e_tok_s": ir["candidate_only_memory_measurement"]["end_to_end_tokens_per_second"],
                "prefill_ms": ir.get("prefill_milliseconds"),
                "peak_alloc_mb": ir["candidate_only_memory_measurement"]["peak_allocated_cuda_memory_mb"],
                "tokens": rec["total_tokens"], "updates": rec["optimizer_updates"],
                "training_seconds": rec["protected_measured_training_seconds"]}
        wd = args.winner_dir / f"attempt-{attempt}"
        bpb = load(wd / "protected_bpb.json")
        repeats = []
        for i in (1, 2, 3):
            p = wd / f"repeat_{i}.json"
            if not p.exists():
                repeats.append({"repeat": i, "missing": True})
                continue
            d = load(p)
            c = d.get("candidate_only_memory_measurement") or {}
            corr = d.get("correctness") or {}
            repeats.append({"repeat": i, "status": d.get("status"), "normalized_status": d.get("normalized_inference_status"),
                            "normalized_drift": d.get("normalized_relative_drift"),
                            "decode_tok_s": d.get("cached_decode_tokens_per_second_median"),
                            "block_tok_s": d.get("decode_block_velocities"),
                            "speedup": d.get("normalized_decode_throughput"),
                            "speedup_ci95": d.get("normalized_decode_throughput_ci_95pct"),
                            "baseline_tok_s": d.get("absolute_baseline_tok_s"),
                            "e2e_tok_s": c.get("end_to_end_tokens_per_second"), "prefill_ms": d.get("prefill_milliseconds"),
                            "e2e_rep_median_tok_s": rep_median(d, "end_to_end_tokens_per_second"),
                            "timed_repetitions": len(d.get("inference_repetitions") or []),
                            "first_token_ms": d.get("first_token_milliseconds"),
                            "peak_alloc_mb": c.get("peak_allocated_cuda_memory_mb"),
                            "kv_cache_bytes": d.get("kv_cache_bytes"),
                            "correctness": {k: v for k, v in corr.items() if not isinstance(v, (dict, list))}})
        ok = [r for r in repeats if not r.get("missing") and r.get("decode_tok_s")]
        med = lambda key: statistics.median([r[key] for r in ok if r.get(key) is not None]) if ok else None
        report["winners"][attempt] = {
            "discovery": disc, "protected_bpb_remeasured": bpb["protected_validation_bpb"],
            "protected_bpb_delta": bpb["protected_validation_bpb"] - disc["protected_bpb"], "repeats": repeats,
            "summary": {"decode_tok_s_median": med("decode_tok_s"), "speedup_median": med("speedup"),
                        "baseline_tok_s_median": med("baseline_tok_s"), "e2e_tok_s_median": med("e2e_tok_s"),
                        "e2e_rep_median_tok_s_median": med("e2e_rep_median_tok_s"),
                        "normalized_drift": [r.get("normalized_drift") for r in ok],
                        "prefill_ms_median": med("prefill_ms"), "peak_alloc_mb_median": med("peak_alloc_mb"),
                        "stable_repeats": sum(r.get("normalized_status") == "stable" for r in ok),
                        "repeats": len(ok)}}
    base_bpb = args.winner_dir / "pair_a_baseline_t" / "protected_bpb.json"
    if base_bpb.exists():
        report["pair_a_baseline_t_protected_bpb_remeasured"] = load(base_bpb)["protected_validation_bpb"]

    # ---- 2B: downstream
    paradigm_map = load(args.pod_copy / "context_position" / "blimp_map.json")["paradigms"]
    for run_id, (model, seed) in AUGUST_RUNS.items():
        b = load(args.august_bundles / f"{run_id:04d}" / "results.json")
        report["downstream"].setdefault(f"{model}_tokens201M", {})[str(seed)] = {
            "source": f"August panel run {run_id}", "checkpoint_sha256": b["checkpoint_sha256"], "metrics": task_metrics(b)}
    for run_dir in sorted((args.pod_copy / "downstream" / "runs").iterdir()):
        b = load(run_dir / "results.json")
        name = P2_DOWNSTREAM[b["experiment_id"]]
        report["downstream"].setdefault(name, {})["single"] = {
            "source": f"Phase 2 run {b['downstream_run_id']}", "checkpoint_sha256": b["checkpoint_sha256"],
            "evaluator_sha256": b["evaluator_sha256"], "suite_sha256": b["suite_sha256"], "metrics": task_metrics(b),
            "generations": [{k: g[k] for k in ("prompt_id", "generation_seed", "generated_text")} for g in b["generations"]]}
    # seed summaries for the 201M panel
    for name, runs in report["downstream"].items():
        seeds = {k: v for k, v in runs.items() if k.isdigit()}
        if len(seeds) >= 2:
            keys = next(iter(seeds.values()))["metrics"].keys()
            runs["seed_summary"] = {key: seed_summary([seeds[s]["metrics"][key]["value"] for s in sorted(seeds)]) for key in keys}
    # BLiMP phenomena
    for name, runs in report["downstream"].items():
        for label, run in runs.items():
            if not isinstance(run, dict) or "metrics" not in run:
                continue
            report["blimp"].setdefault(name, {})[str(label)] = blimp_phenomena(
                run["metrics"]["blimp.micro_accuracy"]["configurations"], paradigm_map)

    # ---- 2C: context position
    cp = {}
    for path in sorted((args.pod_copy / "context_position").glob("*.json")):
        if path.name == "blimp_map.json":
            continue
        d = load(path)
        cp[d["model"]] = d
        report["context_position"][d["model"]] = {"overall_bpb": d["overall_bpb"], "buckets": d["buckets"],
                                                  "bucket_bpb": d["bucket_bpb"], "segments_sha256": d["segments_sha256"]}
    seg_hashes = {d["segments_sha256"] for d in cp.values()}
    report["context_position_same_segments"] = len(seg_hashes) == 1
    comparisons = []
    for opus in ("opus55_t_attempt23", "opus55_t_attempt24", "opus55_t_attempt27"):
        for ref in ("pair_a_baseline_t", "autoresearch_compatible_time300s_seed0", "autopareto_292_tokens201M_seed3",
                    "autoresearch_compatible_tokens201M_seed3"):
            if opus in cp and ref in cp:
                comparisons.append({"a": opus, "b": ref, "rows": bootstrap_bucket_diffs(cp[opus], cp[ref])})
    report["context_position_paired"] = comparisons

    # ---- baseline-recipe end-to-end reference on the same pod (Phase 1 inference-only probe, doc 21 section 8)
    if args.baseline_probe_dir is not None:
        probe = []
        for i in (1, 2, 3):
            d = load(args.baseline_probe_dir / f"repeat_{i}.json")
            c = d.get("candidate_only_memory_measurement") or {}
            probe.append({"decode_tok_s": d.get("cached_decode_tokens_per_second_median"),
                          "e2e_tok_s": c.get("end_to_end_tokens_per_second"), "prefill_ms": d.get("prefill_milliseconds"),
                          "e2e_rep_median_tok_s": rep_median(d, "end_to_end_tokens_per_second"),
                          "peak_alloc_mb": c.get("peak_allocated_cuda_memory_mb"),
                          "normalized_status": d.get("normalized_inference_status"),
                          "normalized_drift": d.get("normalized_relative_drift")})
        report["baseline_recipe_probe"] = {"repeats": probe, "median": {k: statistics.median(p[k] for p in probe)
                                                                        for k in ("decode_tok_s", "e2e_tok_s", "e2e_rep_median_tok_s", "prefill_ms", "peak_alloc_mb")}}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "phase2_verification.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    (args.out_dir / "PHASE2_TABLES.md").write_text(render(report))
    md, tex = headline(report)
    (args.out_dir / "PHASE2_HEADLINE.md").write_text(md)
    (args.out_dir / "phase2_headline_table.tex").write_text(tex)
    print("wrote", args.out_dir / "phase2_verification.json", "PHASE2_TABLES.md, PHASE2_HEADLINE.md, phase2_headline_table.tex")
    return 0


def render(r: dict) -> str:
    f = lambda x, d=4: "—" if x is None else f"{x:.{d}f}"
    out = ["# Phase 2 tables (generated by scripts/analyze_phase2_verification.py; do not edit by hand)", ""]
    # Table 1
    w = r["winners"]
    probe = (r.get("baseline_recipe_probe") or {}).get("median") or {}
    probe_reps = (r.get("baseline_recipe_probe") or {}).get("repeats") or []
    out += ["## Table 1. Opus winners re-verified on pod4 (frozen Pair-A harness; fixed time 300 s; seed 3)", "",
            "| | Attempt 23 | Attempt 24 | Attempt 27 | Baseline recipe |", "|---|---|---|---|---|"]
    def row(label, fn, base=""):
        out.append(f"| {label} | " + " | ".join(fn(w[a]) for a in (23, 24, 27)) + f" | {base} |")
    row("Training tokens in 300 s (updates)", lambda x: f"{x['discovery']['tokens']/1e6:.1f}M ({x['discovery']['updates']})", "84.9M (162)")
    row("Protected BPB: campaign → re-measured", lambda x: f"{f(x['discovery']['protected_bpb'],5)} → {f(x['protected_bpb_remeasured'],5)}",
        f"1.20110 → {f(r.get('pair_a_baseline_t_protected_bpb_remeasured'),5)}")
    row("Decode tok/s, median of 3 repeats (campaign)", lambda x: f"{f(x['summary']['decode_tok_s_median'],0)} ({f(x['discovery']['decode_tok_s'],0)})",
        f"{f(probe.get('decode_tok_s'),0)}")
    row("Decode speedup vs paired baseline (campaign)", lambda x: f"{f(x['summary']['speedup_median'],1)}× ({f(x['discovery']['speedup'],1)}×)", "1×")
    row("Repeats stable under the frozen 2% drift rule", lambda x: f"{x['summary']['stable_repeats']}/{x['summary']['repeats']}", f"{sum(p['normalized_status'] == 'stable' for p in probe_reps)}/{len(probe_reps)} (probe)" if probe_reps else "—")
    row("End-to-end tok/s incl. 1,024-token prefill (campaign)", lambda x: f"{f(x['summary']['e2e_tok_s_median'],0)} ({f(x['discovery']['e2e_tok_s'],0)})",
        f"{f(probe.get('e2e_tok_s'),0)}")
    row("End-to-end tok/s, median over all timed repetitions", lambda x: f(x['summary']['e2e_rep_median_tok_s_median'], 0),
        f(probe.get('e2e_rep_median_tok_s'), 0))
    row("End-to-end speedup vs baseline recipe (frozen field; all-repetition median)",
        lambda x: (f"{x['summary']['e2e_tok_s_median']/probe['e2e_tok_s']:.1f}× ; "
                   f"{x['summary']['e2e_rep_median_tok_s_median']/probe['e2e_rep_median_tok_s']:.1f}×") if probe.get("e2e_tok_s") else "—", "1×")
    row("Prefill of 1,024 tokens, ms (median)", lambda x: f(x['summary']['prefill_ms_median'], 1), f(probe.get('prefill_ms'), 1))
    row("Normalized drift per repeat (rule: ≤ 0.02)", lambda x: ", ".join(f"{v:.3f}" for v in x['summary']['normalized_drift']), ", ".join(f"{p['normalized_drift']:.3f}" for p in probe_reps) or "—")
    row("Peak allocated inference memory, MiB", lambda x: f(x['summary']['peak_alloc_mb_median'], 1), f(probe.get('peak_alloc_mb'), 1))
    row("Correctness (all repeats)", lambda x: ", ".join(sorted({f"{rr['correctness'].get('passed')}/{rr['correctness'].get('verification_mode')}" for rr in x['repeats'] if not rr.get('missing')})), "True/bf16_fast_path")
    out.append("")
    # Table 2
    d = r["downstream"]
    cols = [("heldout_climbmix_bpb.bits_per_byte", "Held-out BPB ↓", 4), ("lambada_openai.exact_match", "LAMBADA acc ↑", 3),
            ("lambada_openai.perplexity", "LAMBADA ppl ↓", 2), ("blimp.macro_accuracy", "BLiMP ↑", 3),
            ("piqa.accuracy", "PIQA ↑", 3), ("arc_easy.accuracy", "ARC-E ↑", 3), ("arc_easy.length_normalized_accuracy", "ARC-E norm ↑", 3)]
    out += ["## Table 2. Frozen downstream suite (evaluator 696f9c46, suite 3dce2c73)", "",
            "| Model | Budget | Seeds | " + " | ".join(c[1] for c in cols) + " |", "|---" * (3 + len(cols)) + "|"]
    single = [("opus55_t_attempt23", "Opus 23", "fixed 300 s (197M tok)"), ("opus55_t_attempt24", "Opus 24", "fixed 300 s (198M tok)"),
              ("opus55_t_attempt27", "Opus 27", "fixed 300 s (198M tok)"), ("pair_a_baseline_t", "Pair-A baseline", "fixed 300 s (85M tok)"),
              ("autoresearch_compatible_time300s", "AutoResearch", "fixed 300 s (88M tok)")]
    for key, label, budget in single:
        m = (d.get(key) or {}).get("single", {}).get("metrics")
        if m:
            seed = "0" if "autoresearch" in key else "3"
            out.append(f"| {label} | {budget} | {seed} | " + " | ".join(f(m[c]["value"], p) for c, _, p in cols) + " |")
    for key, label in (("autopareto_292_tokens201M", "AP292"), ("autoresearch_compatible_tokens201M", "AutoResearch"),
                       ("phase1_baseline_tokens201M", "Phase-1 baseline")):
        s = (d.get(key) or {}).get("seed_summary")
        if s:
            out.append(f"| {label} | matched 201M tok | 3–5 | " + " | ".join(f"{f(s[c]['mean'], p)} ± {f(s[c]['sd'], p)}" for c, _, p in cols) + " |")
    rep = (d.get("autopareto_292_tokens201M_repro") or {}).get("single", {}).get("metrics")
    orig = (d.get("autopareto_292_tokens201M") or {}).get("3", {}).get("metrics")
    if rep and orig:
        out.append(f"| AP292 seed 3, re-evaluated on pod4 (Aug value) | matched 201M tok | 3 | " +
                   " | ".join(f"{f(rep[c]['value'], p)} ({f(orig[c]['value'], p)})" for c, _, p in cols) + " |")
    out.append("")
    # Table 3: BLiMP phenomena
    b = r["blimp"]
    order = [("opus55_t_attempt23", "single", "Opus 23"), ("opus55_t_attempt24", "single", "Opus 24"), ("opus55_t_attempt27", "single", "Opus 27"),
             ("pair_a_baseline_t", "single", "Pair-A base 300 s"), ("autoresearch_compatible_time300s", "single", "AR 300 s")]
    seeds = [("autopareto_292_tokens201M", "AP292 201M"), ("autoresearch_compatible_tokens201M", "AR 201M"), ("phase1_baseline_tokens201M", "Base 201M")]
    terms = sorted(next(iter(b["opus55_t_attempt23"].values())).keys()) if "opus55_t_attempt23" in b else []
    out += ["## Table 3. BLiMP accuracy by phenomenon (single runs: Wilson 95% CI; 201M panel: mean of seeds 3–5 ± t-based 95% CI half-width)", "",
            "| Phenomenon | n | " + " | ".join(o[2] for o in order) + " | " + " | ".join(s[1] for s in seeds) + " |",
            "|---" * (2 + len(order) + len(seeds)) + "|"]
    for t in terms:
        n = b["opus55_t_attempt23"]["single"][t]["count"]
        cells = []
        for key, lab, _ in order:
            x = (b.get(key) or {}).get(lab, {}).get(t)
            cells.append("—" if not x else f"{x['accuracy']:.3f} [{x['wilson95'][0]:.3f}, {x['wilson95'][1]:.3f}]")
        for key, _ in seeds:
            vals = [b[key][str(s)][t]["accuracy"] for s in (3, 4, 5) if str(s) in b.get(key, {})]
            if len(vals) == 3:
                ss = seed_summary(vals)
                cells.append(f"{ss['mean']:.3f} ± {ss['ci95'][1]-ss['mean']:.3f}")
            else:
                cells.append("—")
        out.append(f"| {t} | {n} | " + " | ".join(cells) + " |")
    out.append("")
    # Table 4: context position
    cp = r["context_position"]
    if cp:
        buckets = next(iter(cp.values()))["buckets"]
        out += ["## Table 4. Held-out BPB by target position in the 2,048-token window (exploratory; same 2,560 segments for every model)", "",
                "| Model | " + " | ".join(f"[{lo},{hi})" for lo, hi in buckets) + " | late − early ([1024,2048) − [128,256)) |",
                "|---" * (2 + len(buckets)) + "|"]
        for name, x in cp.items():
            bb = x["bucket_bpb"]
            out.append(f"| {name} | " + " | ".join(f(v, 4) for v in bb) + f" | {bb[-1]-bb[1]:+.4f} |")
        out += ["", "Paired differences (A − B, bootstrap over segments, 1,000 reps):", "",
                "| A | B | " + " | ".join(f"[{lo},{hi})" for lo, hi in buckets) + " | change late vs early |", "|---" * (3 + len(buckets)) + "|"]
        for comp in r.get("context_position_paired", []):
            out.append(f"| {comp['a']} | {comp['b']} | " + " | ".join(f"{row['diff']:+.4f} [{row['ci95'][0]:+.4f}, {row['ci95'][1]:+.4f}]" for row in comp["rows"]) + " |")
    return "\n".join(out) + "\n"


def headline(r: dict) -> tuple[str, str]:
    """One paper-ready comparison table: fixed-time block (single runs) and matched-token block (seeds 3-5)."""
    d, w = r["downstream"], r["winners"]
    probe = r.get("baseline_recipe_probe") or {}
    pm, preps = probe.get("median") or {}, probe.get("repeats") or []
    tasks = [("heldout_climbmix_bpb.bits_per_byte", 3), ("lambada_openai.exact_match", 3), ("blimp.macro_accuracy", 3),
             ("piqa.accuracy", 3), ("arc_easy.accuracy", 3)]
    head = ["Model", "Budget", "Pair-A val BPB", "Held-out BPB", "LAMBADA", "BLiMP", "PIQA", "ARC-E",
            "Decode speedup (stable)", "E2E speedup", "Peak MiB"]
    rows: list[tuple[str, list[str]]] = []
    for a, key in ((23, "opus55_t_attempt23"), (24, "opus55_t_attempt24"), (27, "opus55_t_attempt27")):
        x = w[a]; m = d[key]["single"]["metrics"]; sm = x["summary"]
        rows.append(("fixed", [f"Opus 5.5 attempt {a}", f"300 s ({x['discovery']['tokens']/1e6:.0f}M tok)",
                     f"{x['protected_bpb_remeasured']:.4f}"] + [f"{m[k]['value']:.{p}f}" for k, p in tasks] +
                     [f"{sm['speedup_median']:.1f}× ({sm['stable_repeats']}/{sm['repeats']})",
                      f"{sm['e2e_tok_s_median']/pm['e2e_tok_s']:.1f}×" if pm.get("e2e_tok_s") else "—",
                      f"{sm['peak_alloc_mb_median']:.0f}"]))
    m = d["pair_a_baseline_t"]["single"]["metrics"]
    rows.append(("fixed", ["Pair-A baseline", "300 s (85M tok)", f"{r.get('pair_a_baseline_t_protected_bpb_remeasured', float('nan')):.4f}"] +
                 [f"{m[k]['value']:.{p}f}" for k, p in tasks] +
                 [f"1× ({sum(q['normalized_status'] == 'stable' for q in preps)}/{len(preps)})", "1×", f"{pm.get('peak_alloc_mb', float('nan')):.0f}"]))
    m = d["autoresearch_compatible_time300s"]["single"]["metrics"]
    rows.append(("fixed", ["AutoResearch", "300 s (88M tok)", "—"] + [f"{m[k]['value']:.{p}f}" for k, p in tasks] + ["—", "—", "—"]))
    for key, label in (("autopareto_292_tokens201M", "AutoPareto AP292"), ("autoresearch_compatible_tokens201M", "AutoResearch"),
                       ("phase1_baseline_tokens201M", "Phase-1 baseline")):
        sm = d[key]["seed_summary"]
        rows.append(("matched", [label, "201M tok, seeds 3–5", "—"] +
                     [f"{sm[k]['mean']:.{p}f} ± {sm[k]['sd']:.{p}f}" for k, p in tasks] + ["—", "—", "—"]))
    md = ["# Phase 2 headline comparison (generated; do not edit by hand)", "",
          "| " + " | ".join(head) + " |", "|---" * len(head) + "|"]
    for block, cells in rows:
        md.append("| " + " | ".join(cells) + " |")
    md += ["", "Fixed-time rows: one training run each (Opus and Pair-A baseline: Pair-A harness, seed 3; AutoResearch: "
           "Phase-1 harness, seed 0). Matched-token rows: mean ± sd over seeds 3–5 (August panel, reused). "
           "Pair-A val BPB is the search objective (protected evaluator), re-measured on pod4. Held-out BPB and the four "
           "tasks use the frozen downstream suite (evaluator 696f9c46, suite 3dce2c73). Single-run 95% intervals: held-out "
           "BPB ±0.007, LAMBADA ±0.012, BLiMP ±0.003 (micro), PIQA ±0.023, ARC-E ±0.019. Speedups are decode-only "
           "(bracketed, vs the paired Pair-A baseline, median of 3 repeats; stable = normalized drift ≤ 0.02) and "
           "end-to-end (1,024-token prefill + 128 generated tokens, vs the baseline recipe measured as a candidate on the "
           "same pod). Peak = peak allocated CUDA memory during inference.", ""]
    esc = lambda t: t.replace("±", r"$\pm$").replace("×", r"$\times$").replace("–", "--").replace("—", "---").replace("%", r"\%")
    tex = [r"% Generated by scripts/analyze_phase2_verification.py (Phase 2, 2026-11-07); do not edit by hand.",
           r"\begin{table}[t]", r"\centering\small\setlength{\tabcolsep}{3pt}",
           r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{ll" + "r" * (len(head) - 2) + "}", r"\toprule",
           " & ".join(esc(h) for h in head) + r" \\", r"\midrule",
           r"\multicolumn{" + str(len(head)) + r"}{l}{\emph{Fixed training time (300 s, A40)}} \\"]
    current = "fixed"
    for block, cells in rows:
        if block != current:
            tex += [r"\midrule", r"\multicolumn{" + str(len(head)) + r"}{l}{\emph{Matched training tokens (201.3M)}} \\"]
            current = block
        tex.append(" & ".join(esc(c) for c in cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}}",
            r"\caption{Opus 5.5 Contract-T winners re-verified on a fresh A40, with the frozen downstream suite. "
            r"Fixed-time rows are single runs; matched-token rows are mean $\pm$ sd over three seeds. Decode speedup is "
            r"against the paired baseline (stable repeats out of 3 under the frozen 2\% drift rule); end-to-end includes a "
            r"1,024-token prefill.}", r"\label{tab:phase2-verification}", r"\end{table}", ""]
    return "\n".join(md) + "\n", "\n".join(tex)


if __name__ == "__main__":
    raise SystemExit(main())
