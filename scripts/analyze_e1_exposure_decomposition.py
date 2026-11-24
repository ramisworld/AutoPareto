#!/usr/bin/env python3
"""EXPLORATORY (not pre-registered) E1 analysis: token exposure versus architecture.

    python3 scripts/analyze_e1_exposure_decomposition.py \
        --runs-root results/camera_ready_2026/amendment1/E1 \
        --out-dir results/camera_ready_2026/amendment1/E1/analysis_exploratory

The pre-registered E1 analysis (scripts/analyze_camera_ready_e1.py, unchanged) answers
"how does Delta = BPB(AP-Q) - BPB(AR-ref) change with tokens, and where does it flip".
This script adds the reviewers' framing on the same records, and is labelled exploratory
everywhere it reports:

1. Per seed s, each recipe's token-response curve f_r,s(T) is the piecewise-linear
   interpolation of its three fixed-token BPBs in ln(tokens) (88.6M, 157.3M, 201.3M),
   extended linearly from the end segment when T falls slightly outside that range.
2. The 300 s gap G_s = BPB(AP-Q, 300 s) - BPB(AR-ref, 300 s) is split, with T_A and T_R
   the tokens each recipe actually processed in its 300 s run:
     exposure      E_s = f_AR(T_A) - f_AR(T_R)   (AR-ref given AP-Q's 300 s tokens)
     architecture  A_s = f_AP(T_A) - f_AR(T_A)   (AP-Q minus AR-ref at equal tokens)
     schedule      R_s = G_s - (E_s + A_s)       (fixed-time path vs token-cell path)
   E + A + R = G exactly. The other ordering (exposure measured on AP-Q's curve,
   architecture at AR-ref's exposure) is reported too; the two differ by the interaction.
3. Sample efficiency: k_s = f_AP^-1(f_AR(T_R)) / T_R, the token multiple AP-Q needs to
   reach AR-ref's quality at AR-ref's 300 s exposure; compared with the steady-state
   throughput ratio rho_s of the paired 300 s runs. AP-Q wins at fixed time iff rho > k
   (up to the schedule term).

Intervals: mean +/- t(0.975, n-1) sd/sqrt(n) over seeds, as in the pre-registered plan.
Run selection, metric extraction and the t quantile are imported from the pre-registered
script so both analyses read identical inputs. Paired fixed-token deltas are cross-checked
against the pre-registered e1_analysis.json when it exists.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
PREREG_PATH = ROOT / "scripts/analyze_camera_ready_e1.py"
TOKEN_CELLS = ["tokens_088604672", "tokens_157286400", "tokens_201326592"]
FIXED = "time_300s"
AP, AR = "AP-Q", "AR-ref"
METRICS = [("heldout_bpb", "held-out BPB"), ("validation_bpb", "validation BPB")]
BLUE, ORANGE = "#2a78d6", "#eb6834"          # paper colours: AP-Q, AR-ref (analysis/cr_figures.py)
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#9a9893", "#e6e5e1"


def load_prereg():
    spec = importlib.util.spec_from_file_location("prereg_e1", PREREG_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def interp(xs: list[float], ys: list[float], x: float) -> tuple[float, bool]:
    """Piecewise-linear y(x); linear extension from the end segment outside [xs[0], xs[-1]]."""
    if x <= xs[0]:
        i = 0
    elif x >= xs[-1]:
        i = len(xs) - 2
    else:
        i = max(j for j in range(len(xs) - 1) if xs[j] <= x)
    y = ys[i] + (ys[i + 1] - ys[i]) * (x - xs[i]) / (xs[i + 1] - xs[i])
    return y, not (xs[0] <= x <= xs[-1])


def inverse(xs: list[float], ys: list[float], y: float) -> tuple[float | None, str]:
    """x at which the (decreasing) curve reaches y; 'in_range', 'extrapolated' or 'not_monotone'."""
    if any(ys[i + 1] >= ys[i] for i in range(len(ys) - 1)):
        return None, "not_monotone"
    if ys[0] >= y >= ys[-1]:
        i = max(j for j in range(len(ys) - 1) if ys[j] >= y)
        return xs[i] + (xs[i + 1] - xs[i]) * (y - ys[i]) / (ys[i + 1] - ys[i]), "in_range"
    i = 0 if y > ys[0] else len(xs) - 2
    return xs[i] + (xs[i + 1] - xs[i]) * (y - ys[i]) / (ys[i + 1] - ys[i]), "extrapolated"


def run(runs_root: Path, out_dir: Path, figure: bool, allow_synthetic: bool = False) -> dict:
    pre = load_prereg()
    plan = json.loads(pre.PLAN_PATH.read_text())
    records, synthetic = pre.load_records(runs_root, allow_synthetic=allow_synthetic)
    if synthetic and (ROOT / "results").resolve() in (out_dir.resolve(), *out_dir.resolve().parents):
        raise RuntimeError("synthetic output must not be written under results/")
    chosen, excluded = pre.select(records, plan)
    seeds = list(plan["seeds"]["E1"])
    budgets = [plan["cells"][c]["token_budget"] for c in TOKEN_CELLS]
    lx = [math.log(b) for b in budgets]

    def summ(values):
        return pre.summarize(values, 0.95)

    def val(recipe, cell, seed, name):
        rec = chosen.get((recipe, cell, seed))
        return pre.metric(rec, name) if rec else None

    out: dict = {
        "format": "autopareto_e1_exposure_decomposition_v1",
        "status": "EXPLORATORY - not pre-registered; written after the plan, before E1 results were seen",
        "plan_sha256": __import__("hashlib").sha256(pre.PLAN_PATH.read_bytes()).hexdigest(),
        "inputs": {"runs_root": str(runs_root), "n_records": len(records), "selected": len(chosen),
                   "excluded_or_missing": excluded},
        "seeds": seeds,
        "synthetic": synthetic,
    }

    # Descriptive per recipe and cell.
    desc = {}
    for recipe in (AR, AP):
        for cell in TOKEN_CELLS + [FIXED]:
            row = {}
            for name in ("heldout_bpb", "validation_bpb", "validation_bpb_in_process", "tokens", "updates",
                         "steady_state_tokens_per_second", "peak_training_memory_mb"):
                vals = [val(recipe, cell, s, name) for s in seeds]
                vals = [v for v in vals if v is not None]
                row[name] = summ(vals) if vals else None
            desc[f"{recipe}|{cell}"] = row
    out["descriptive"] = desc

    # Matched-token deltas (same quantity as the pre-registered analysis, recomputed for the cross-check).
    matched = {}
    for name, _ in METRICS:
        matched[name] = {}
        for cell in TOKEN_CELLS + [FIXED]:
            d = {s: val(AP, cell, s, name) - val(AR, cell, s, name) for s in seeds
                 if val(AP, cell, s, name) is not None and val(AR, cell, s, name) is not None}
            matched[name][cell] = {"per_seed": d, **summ(list(d.values()))}
    out["paired_delta"] = matched

    # Decomposition of the 300 s gap.
    decomp = {}
    for name, _ in METRICS:
        rows = {}
        for s in seeds:
            need = [val(r, c, s, name) for r in (AP, AR) for c in TOKEN_CELLS + [FIXED]]
            ta, tr = val(AP, FIXED, s, "tokens"), val(AR, FIXED, s, "tokens")
            if any(v is None for v in need) or ta is None or tr is None:
                continue
            fa = [val(AP, c, s, name) for c in TOKEN_CELLS]
            fr = [val(AR, c, s, name) for c in TOKEN_CELLS]
            g = val(AP, FIXED, s, name) - val(AR, FIXED, s, name)
            fr_ta, x1 = interp(lx, fr, math.log(ta))
            fr_tr, x2 = interp(lx, fr, math.log(tr))
            fa_ta, x3 = interp(lx, fa, math.log(ta))
            fa_tr, x4 = interp(lx, fa, math.log(tr))
            e, a = fr_ta - fr_tr, fa_ta - fr_ta
            e2, a2 = fa_ta - fa_tr, fa_tr - fr_tr
            target = fr_tr
            t_need, how = inverse(lx, fa, target)
            k = math.exp(t_need) / tr if t_need is not None else None
            rho = val(AP, FIXED, s, "steady_state_tokens_per_second") / val(AR, FIXED, s, "steady_state_tokens_per_second")
            rows[s] = {
                "tokens_AP_300s": ta, "tokens_AR_300s": tr, "token_ratio_300s": ta / tr,
                "gap_300s": g,
                "exposure_on_AR_curve": e, "architecture_at_AP_exposure": a, "schedule_residual": g - (e + a),
                "exposure_on_AP_curve": e2, "architecture_at_AR_exposure": a2, "schedule_residual_alt": g - (e2 + a2),
                "interaction": a - a2,
                "token_schedule_prediction": e + a,
                "AP_tokens_to_match_AR_300s_quality": math.exp(t_need) if t_need is not None else None,
                "sample_efficiency_multiple_k": k, "k_inversion": how,
                "throughput_ratio_rho": rho,
                "extrapolated": {"AR_at_T_A": x1, "AR_at_T_R": x2, "AP_at_T_A": x3, "AP_at_T_R": x4},
            }
        keys = ["gap_300s", "exposure_on_AR_curve", "architecture_at_AP_exposure", "schedule_residual",
                "exposure_on_AP_curve", "architecture_at_AR_exposure", "schedule_residual_alt", "interaction",
                "token_schedule_prediction", "token_ratio_300s", "sample_efficiency_multiple_k", "throughput_ratio_rho"]
        decomp[name] = {"per_seed": rows,
                        "summary": {k: summ([r[k] for r in rows.values() if r[k] is not None]) for k in keys}}
    out["decomposition_300s"] = decomp

    # Seed consistency: within-recipe SD per cell, and per-seed curve shape.
    consistency = {}
    for name, _ in METRICS:
        sds = {f"{r}|{c}": (desc[f"{r}|{c}"][name] or {}).get("sd") for r in (AR, AP) for c in TOKEN_CELLS + [FIXED]}
        shapes = {}
        for s in seeds:
            d = [matched[name][c]["per_seed"].get(s) for c in TOKEN_CELLS]
            shapes[s] = None if None in d else ("increasing" if d[0] < d[1] < d[2] else "not_strictly_increasing")
        consistency[name] = {"within_recipe_sd": sds, "max_within_recipe_sd": max(v for v in sds.values() if v is not None),
                             "per_seed_delta_curve": shapes}
    out["seed_consistency"] = consistency

    # Cross-check against the pre-registered output, if present.
    prereg_json = runs_root / "analysis" / "e1_analysis.json"
    check = {"prereg_analysis": str(prereg_json), "present": prereg_json.exists()}
    if prereg_json.exists():
        pr = json.loads(prereg_json.read_text())
        diffs = []
        result = pr.get("result", pr)
        for name, _ in METRICS:
            for cell in TOKEN_CELLS + [FIXED]:
                theirs = result["metrics"][name]["per_cell"][cell]
                ours = matched[name][cell]
                for s, v in zip(theirs["seeds"], theirs["values"]):
                    diffs.append(abs(v - ours["per_seed"][s]))
        check["max_abs_difference"] = max(diffs) if diffs else None
        check["passed"] = bool(diffs) and max(diffs) < 1e-12
    out["cross_check"] = check

    # Replication of the Phase-1 panel's 201.3M contrast (seeds 3-5) on fresh seeds 6-8.
    panel_path = ROOT / "paper/camera_ready/data/cr_stats.json"
    rep = {"source": str(panel_path.relative_to(ROOT)), "present": panel_path.exists()}
    if panel_path.exists():
        cr = json.loads(panel_path.read_text())
        for name, key in (("heldout_bpb", "matched_heldout"), ("validation_bpb", "matched_val")):
            panel = cr[key]["diffs"]
            e1 = list(matched[name]["tokens_201326592"]["per_seed"].values())
            rep[name] = {"panel_seeds_3_5": summ(panel), "e1_seeds_6_8": summ(e1),
                         "welch_e1_minus_panel": pre.welch(e1, panel, 0.95)}
    out["replication_vs_phase1_panel"] = rep

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "e1_exposure_decomposition.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    md, tex = render(out, budgets)
    (out_dir / "E1_EXPOSURE_DECOMPOSITION.md").write_text(md)
    (out_dir / "e1_tables.tex").write_text(tex)
    if figure:
        make_figure(out, chosen, seeds, budgets, out_dir, synthetic)
    return out


def f4(x, d=4, sign=True):
    if x is None:
        return "n/a"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def ci(s, d=4):
    if not s or s.get("mean") is None:
        return "n/a"
    lo, hi = s["ci"]
    if lo is None:
        return f"{s['mean']:+.{d}f}"
    return f"{s['mean']:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def msd(s, d=4):
    if not s or s.get("mean") is None:
        return "n/a"
    return f"{s['mean']:.{d}f} ± {s['sd']:.{d}f}" if s.get("sd") is not None else f"{s['mean']:.{d}f}"


def render(out: dict, budgets: list[int]) -> tuple[str, str]:
    labels = {"tokens_088604672": "88.6M tokens (676 updates)", "tokens_157286400": "157.3M tokens (1,200 updates)",
              "tokens_201326592": "201.3M tokens (1,536 updates)", FIXED: "300 s fixed time"}
    desc, pdelta = out["descriptive"], out["paired_delta"]
    L = (["# SYNTHETIC TEST DATA - NOT EXPERIMENTAL RESULTS", ""] if out.get("synthetic") else []) + [
         "# E1: token exposure versus architecture (EXPLORATORY decomposition)", "",
         "Not pre-registered. The pre-registered verdicts are in `../analysis/e1_analysis.md`.",
         f"Plan sha256 `{out['plan_sha256'][:12]}…`; {out['inputs']['selected']} of 24 runs selected; "
         f"excluded or missing: {out['inputs']['excluded_or_missing'] or 'none'}.",
         f"Cross-check of paired deltas against the pre-registered analysis: {out['cross_check']}.", "",
         "## Table A. BPB by recipe and budget (mean ± SD over seeds 6–8) and paired Δ = AP-Q − AR-ref (mean [95% t-CI])", "",
         "| Budget | Tokens AR-ref / AP-Q | AR-ref held-out | AP-Q held-out | Δ held-out | Δ validation | Seeds with Δ<0 |",
         "|---|---|---|---|---|---|---|"]
    for cell in TOKEN_CELLS + [FIXED]:
        ar, ap = desc[f"{AR}|{cell}"], desc[f"{AP}|{cell}"]
        tok = f"{ar['tokens']['mean'] / 1e6:.1f}M / {ap['tokens']['mean'] / 1e6:.1f}M" if ar["tokens"] and ap["tokens"] else "n/a"
        dh, dv = pdelta["heldout_bpb"][cell], pdelta["validation_bpb"][cell]
        neg = sum(1 for v in dh["per_seed"].values() if v < 0)
        L.append(f"| {labels[cell]} | {tok} | {msd(ar['heldout_bpb'])} | {msd(ap['heldout_bpb'])} | {ci(dh)} | {ci(dv)} | {neg}/{dh['n']} |")
    for name, label in METRICS:
        dd = out["decomposition_300s"][name]
        sm = dd["summary"]
        L += ["", f"## Table B ({label}). Splitting the 300 s gap into exposure, architecture and schedule", "",
              "Negative = favours AP-Q. Components add up exactly to the observed gap.", "",
              "| Component | Mean [95% t-CI] | Per seed (6, 7, 8) |", "|---|---|---|"]
        rows = [("Observed 300 s gap G", "gap_300s"),
                ("Exposure: AR-ref moved from its 300 s tokens to AP-Q's", "exposure_on_AR_curve"),
                ("Architecture: AP-Q − AR-ref at AP-Q's 300 s tokens", "architecture_at_AP_exposure"),
                ("Schedule residual (fixed-time vs token-cell path)", "schedule_residual"),
                ("Alt. exposure: AP-Q's own gain from its extra tokens", "exposure_on_AP_curve"),
                ("Alt. architecture: at AR-ref's 300 s tokens", "architecture_at_AR_exposure"),
                ("Interaction (architecture gap change with exposure)", "interaction")]
        for lab, key in rows:
            per = ", ".join(f4(dd["per_seed"][s][key]) for s in sorted(dd["per_seed"]))
            L.append(f"| {lab} | {ci(sm[key])} | {per} |")
        L += ["", "| Ratio | Mean [95% t-CI] | Per seed |", "|---|---|---|"]
        for lab, key in [("300 s token ratio T_A/T_R", "token_ratio_300s"),
                         ("Steady-state throughput ratio ρ (300 s runs)", "throughput_ratio_rho"),
                         ("Token multiple k AP-Q needs to match AR-ref's 300 s quality", "sample_efficiency_multiple_k")]:
            per = ", ".join(f4(dd["per_seed"][s][key], 3, False) for s in sorted(dd["per_seed"]))
            s = sm[key]
            txt = f"{s['mean']:.3f} [{s['ci'][0]:.3f}, {s['ci'][1]:.3f}]" if s and s.get("ci") and s["ci"][0] is not None else "n/a"
            L.append(f"| {lab} | {txt} | {per} |")
        extra = {s: [k for k, v in r["extrapolated"].items() if v] for s, r in dd["per_seed"].items()}
        L.append("")
        L.append(f"Extrapolated curve points (outside 88.6M–201.3M): {extra}. k inversion: "
                 f"{ {s: r['k_inversion'] for s, r in dd['per_seed'].items()} }.")
    sc = out["seed_consistency"]["heldout_bpb"]
    L += ["", "## Seed consistency (held-out)", "",
          f"- Largest within-recipe SD in any cell: {sc['max_within_recipe_sd']:.4f}.",
          f"- Per-seed Δ curve shape over 88.6M → 157.3M → 201.3M: {sc['per_seed_delta_curve']}.", ""]
    rep = out.get("replication_vs_phase1_panel", {})
    if rep.get("present"):
        L += ["## Replication of the Phase-1 panel at 201.3M tokens (seeds 3–5 → fresh seeds 6–8)", "",
              "| Metric | Panel seeds 3–5 | E1 seeds 6–8 | Welch E1 − panel |", "|---|---|---|---|"]
        for name, label in METRICS:
            r = rep[name]
            w = r["welch_e1_minus_panel"]
            L.append(f"| {label} | {ci(r['panel_seeds_3_5'])} | {ci(r['e1_seeds_6_8'])} | "
                     f"{w['difference']:+.4f} [{w['ci'][0]:+.4f}, {w['ci'][1]:+.4f}] |")
        L.append("")
    md = "\n".join(L) + "\n"

    def tx(s, d=3):
        if not s or s.get("mean") is None:
            return "n/a"
        lo, hi = s["ci"]
        return f"${s['mean']:+.{d}f}$ \\scriptsize[{lo:+.{d}f}, {hi:+.{d}f}]" if lo is not None else f"${s['mean']:+.{d}f}$"

    def tm(s, d=3):
        return f"{s['mean']:.{d}f}$\\pm${s['sd']:.{d}f}" if s and s.get("sd") is not None else "n/a"

    T = ["% Generated by scripts/analyze_e1_exposure_decomposition.py (exploratory decomposition; Table A rows are the pre-registered E1 quantities)",
         "\\begin{table}[t]\\centering\\small",
         "\\caption{E1, seeds 6--8: held-out BPB at matched tokens and at fixed time. $\\Delta$ = AP-Q $-$ AR-ref, mean [95\\% $t$-CI], $n=3$; negative favours AP-Q.}",
         "\\label{tab:e1}",
         "\\begin{tabular}{lrcccc}", "\\toprule",
         "Budget & Tokens (AR-ref / AP-Q) & AR-ref & AP-Q & $\\Delta$ held-out & $\\Delta<0$ \\\\", "\\midrule"]
    tl = {"tokens_088604672": "88.6M tokens", "tokens_157286400": "157.3M tokens", "tokens_201326592": "201.3M tokens", FIXED: "300 s"}
    for cell in TOKEN_CELLS + [FIXED]:
        ar, ap = desc[f"{AR}|{cell}"], desc[f"{AP}|{cell}"]
        dh = pdelta["heldout_bpb"][cell]
        tok = f"{ar['tokens']['mean'] / 1e6:.1f}M / {ap['tokens']['mean'] / 1e6:.1f}M" if ar["tokens"] and ap["tokens"] else "n/a"
        if cell == FIXED:
            T.append("\\midrule")
        T.append(f"{tl[cell]} & {tok} & {tm(ar['heldout_bpb'])} & {tm(ap['heldout_bpb'])} & {tx(dh)} & "
                 f"{sum(1 for v in dh['per_seed'].values() if v < 0)}/{dh['n']} \\\\")
    T += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    dd = out["decomposition_300s"]["heldout_bpb"]["summary"]
    T += ["\\begin{table}[t]\\centering\\small",
          "\\caption{Exploratory split of the 300 s held-out gap (seeds 6--8). Exposure and architecture are read off each seed's fixed-token curves at the tokens each recipe processed in 300 s; the residual is the schedule difference between the fixed-time and fixed-token paths. Components sum to the gap.}",
          "\\label{tab:e1-split}", "\\begin{tabular}{lc}", "\\toprule", "Component & held-out BPB, mean [95\\% $t$-CI] \\\\", "\\midrule"]
    for lab, key in [("Observed gap at 300 s", "gap_300s"), ("Exposure (AP-Q's extra tokens)", "exposure_on_AR_curve"),
                     ("Architecture at equal tokens", "architecture_at_AP_exposure"), ("Schedule residual", "schedule_residual")]:
        T.append(f"{lab} & {tx(dd[key])} \\\\")
    rho, k = dd["throughput_ratio_rho"], dd["sample_efficiency_multiple_k"]
    if rho and rho.get("mean") is not None and k and k.get("mean") is not None:
        T.append("\\midrule")
        T.append(f"Throughput ratio $\\rho$ (AP-Q/AR-ref) & {rho['mean']:.2f}$\\times$ \\scriptsize[{rho['ci'][0]:.2f}, {rho['ci'][1]:.2f}] \\\\")
        T.append(f"Tokens AP-Q needs for AR-ref's 300 s quality, $k$ & {k['mean']:.2f}$\\times$ \\scriptsize[{k['ci'][0]:.2f}, {k['ci'][1]:.2f}] \\\\")
    T += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return md, "\n".join(T)


def make_figure(out: dict, chosen: dict, seeds: list[int], budgets: list[int], out_dir: Path, synthetic: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "pdf.fonttype": 42})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.6, 2.7), gridspec_kw={"width_ratios": [1.35, 1.0]})
    name = "heldout_bpb"
    desc = out["descriptive"]
    for recipe, color in ((AR, ORANGE), (AP, BLUE)):
        means = [desc[f"{recipe}|{c}"][name]["mean"] for c in TOKEN_CELLS]
        ax.plot(budgets, means, color=color, lw=2.0, zorder=2)
        for b, c in zip(budgets, TOKEN_CELLS):
            s = desc[f"{recipe}|{c}"][name]
            if s.get("ci") and s["ci"][0] is not None:
                ax.plot([b, b], s["ci"], color=color, lw=1.0, zorder=2)
            for j, seed in enumerate(seeds):
                rec = chosen.get((recipe, c, seed))
                if rec:
                    ax.scatter([b * math.exp(0.035 + 0.025 * j)], [rec["metrics"][name]], s=14, facecolor="white", edgecolor=color, lw=0.8, zorder=3)
        for seed in seeds:
            rec = chosen.get((recipe, FIXED, seed))
            if rec:
                ax.scatter([rec["metrics"]["tokens"]], [rec["metrics"][name]], s=34, marker="D", color=color,
                           edgecolor="white", lw=0.8, zorder=4)
        ax.annotate(recipe, (budgets[-1], means[-1]), xytext=(26, 0), textcoords="offset points",
                    va="center", fontsize=7, color=INK)
    ax.set_xscale("log")
    ax.set_xticks(budgets)
    ax.set_xticklabels([f"{b / 1e6:.1f}M" for b in budgets])
    ax.minorticks_off()
    ax.set_xlim(budgets[0] / 1.15, budgets[-1] * 1.35)
    ax.set_xlabel("training tokens (log scale)")
    ax.set_ylabel("held-out BPB (lower is better)")
    ax.grid(color=GRID, lw=0.5)
    ax.set_title("(a) Token response, seeds 6–8", loc="left", fontsize=8, color=INK)
    handles = [Line2D([], [], color=INK2, lw=2.0), Line2D([], [], ls="", marker="o", markerfacecolor="white", markeredgecolor=INK2, markersize=4),
               Line2D([], [], ls="", marker="D", color=INK2, markersize=4.5)]
    ax.legend(handles, ["fixed tokens: mean, 95% CI", "one seed", "300 s run (at its tokens)"], frameon=False,
              fontsize=6.2, loc="upper right")

    dd = out["decomposition_300s"][name]
    rows = [("observed 300 s gap", "gap_300s"), ("exposure", "exposure_on_AR_curve"),
            ("architecture,\nequal tokens", "architecture_at_AP_exposure"), ("schedule\nresidual", "schedule_residual")]
    for i, (lab, key) in enumerate(rows):
        s = dd["summary"][key]
        y = len(rows) - 1 - i
        if s.get("mean") is None:
            continue
        bx.barh([y], [s["mean"]], height=0.5, color="#c9c8c3" if i else INK2, zorder=2)
        if s["ci"][0] is not None:
            bx.plot(s["ci"], [y, y], color=INK, lw=1.0, zorder=3)
        for v in s["values"]:
            bx.scatter([v], [y], s=10, facecolor="white", edgecolor=INK, lw=0.6, zorder=4)
    bx.axvline(0, color=INK2, lw=0.7)
    bx.set_yticks(range(len(rows)))
    bx.set_yticklabels([r[0] for r in rows][::-1], fontsize=6.6)
    bx.set_xlabel("Δ held-out BPB (AP-Q − AR-ref)")
    bx.grid(axis="x", color=GRID, lw=0.5)
    bx.set_title("(b) Splitting the 300 s gap", loc="left", fontsize=8, color=INK)
    lo, hi = bx.get_xlim()
    bx.text(lo, -0.85, "← favours AP-Q", fontsize=6, color=INK2, ha="left", va="top")
    bx.text(hi, -0.85, "favours AR-ref →", fontsize=6, color=INK2, ha="right", va="top")
    bx.set_ylim(-1.1, len(rows) - 0.5)
    fig.text(0.995, 0.005, "panel (b): exploratory decomposition", fontsize=5.5, color=MUTED, ha="right", va="bottom")
    if synthetic:
        fig.suptitle("SYNTHETIC TEST DATA - NOT EXPERIMENTAL RESULTS", color="#e34948", fontsize=9, fontweight="bold")
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(out_dir / f"e1_exposure_vs_architecture.{suffix}", dpi=220)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", type=Path, default=ROOT / "results/camera_ready_2026/amendment1/E1")
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--allow-synthetic", action="store_true", help="testing only; output must be outside results/")
    a = p.parse_args()
    out = run(a.runs_root, a.out_dir or a.runs_root / "analysis_exploratory", figure=not a.no_figure, allow_synthetic=a.allow_synthetic)
    sm = out["decomposition_300s"]["heldout_bpb"]["summary"]
    print(json.dumps({k: (sm[k]["mean"], sm[k]["ci"]) for k in sm if sm[k] and sm[k].get("mean") is not None}, indent=1))
    print("cross_check", out["cross_check"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
