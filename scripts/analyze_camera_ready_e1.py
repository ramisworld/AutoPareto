#!/usr/bin/env python3
"""Pre-registered E1 analysis: AP-Q minus AR-ref as training tokens grow.

    python scripts/analyze_camera_ready_e1.py --runs-root results/camera_ready_2026/amendment1/E1
    python scripts/analyze_camera_ready_e1.py --self-test          # synthetic data only

Inputs are the per-run ``camera_ready_run_record.json`` files written by
``scripts/run_camera_ready_e0_e1.py``.  Only runs whose integrity gates passed
are used.  Every number reported here follows ``analysis_plan`` in
``evaluation/camera_ready_e0_e1_plan_amendment1.json``:

* paired Delta = BPB(AP-Q) - BPB(AR-ref) per budget and seed, mean with a
  t-based 95% interval (n = 3, df = 2);
* crossover = linear interpolation of the mean Delta in log tokens, with an
  exact seed bootstrap (all n^n resamples of seeds, each seed carrying its whole
  Delta curve);
* the 300 s cell compared with the earlier seeds 0-2 result (-0.0067);
* schedule checks (fixed time vs the token cell of similar exposure);
* a figure (PDF and PNG) of Delta against log tokens with per-seed points.

The statistics use only the standard library; matplotlib is needed only for the figure.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
# Amendment 1 (2026-11-04) is the active plan; its E1 section is unchanged from the original.
PLAN_PATH = ROOT / "evaluation/camera_ready_e0_e1_plan_amendment1.json"
RECORD_NAME = "camera_ready_run_record.json"
RECORD_FORMAT = "autopareto_camera_ready_run_record_v1"
SEED_STYLE = [("#2a78d6", "o"), ("#eb6834", "s"), ("#1baf7a", "^")]  # validated categorical slots 1-3
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


# ---------------------------------------------------------------------------
# Student t (standard library only)
# ---------------------------------------------------------------------------

def _betacf(a: float, b: float, x: float) -> float:
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c if abs(1.0 + aa / c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c if abs(1.0 + aa / c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(log_front) * _betacf(a, b, x) / a
    return 1.0 - math.exp(log_front) * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    tail = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t > 0 else tail


def t_quantile(p: float, df: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    if p < 0.5:
        return -t_quantile(1.0 - p, df)
    low, high = 0.0, 1.0
    while t_cdf(high, df) < p:
        high *= 2.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        if t_cdf(mid, df) < p:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def summarize(values: list[float], confidence: float) -> dict:
    n = len(values)
    out: dict = {"n": n, "values": list(values)}
    if n == 0:
        return {**out, "mean": None, "sd": None, "se": None, "df": None, "t_critical": None, "ci": [None, None]}
    mean = statistics.fmean(values)
    if n < 2:
        return {**out, "mean": mean, "sd": None, "se": None, "df": None, "t_critical": None, "ci": [None, None]}
    sd = statistics.stdev(values)
    se = sd / math.sqrt(n)
    tcrit = t_quantile(0.5 + confidence / 2.0, n - 1)
    return {**out, "mean": mean, "sd": sd, "se": se, "df": n - 1, "t_critical": tcrit,
            "ci": [mean - tcrit * se, mean + tcrit * se], "same_sign_seeds": _same_sign(values)}


def _same_sign(values: list[float]) -> str:
    if all(v > 0 for v in values):
        return "all_positive"
    if all(v < 0 for v in values):
        return "all_negative"
    return "mixed"


def welch(a: list[float], b: list[float], confidence: float) -> dict:
    """CI for mean(a) - mean(b) with Welch-Satterthwaite degrees of freedom."""
    if len(a) < 2 or len(b) < 2:
        return {"difference": None, "ci": [None, None], "df": None}
    va, vb = statistics.variance(a) / len(a), statistics.variance(b) / len(b)
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    diff = statistics.fmean(a) - statistics.fmean(b)
    tcrit = t_quantile(0.5 + confidence / 2.0, df)
    return {"difference": diff, "se": se, "df": df, "t_critical": tcrit, "ci": [diff - tcrit * se, diff + tcrit * se]}


def percentile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------

def crossings(xs: list[float], ys: list[float]) -> list[dict]:
    """Zero crossings of the piecewise-linear curve (x = log tokens)."""
    found = []
    for i in range(len(xs) - 1):
        y0, y1 = ys[i], ys[i + 1]
        if y0 == 0.0:
            found.append({"x": xs[i], "direction": "touch"})
        elif y0 * y1 < 0.0:
            x = xs[i] + (0.0 - y0) * (xs[i + 1] - xs[i]) / (y1 - y0)
            found.append({"x": x, "direction": "upward" if y1 > y0 else "downward"})
    if ys and ys[-1] == 0.0:
        found.append({"x": xs[-1], "direction": "touch"})
    for row in found:
        row["tokens"] = math.exp(row["x"])
    return found


def classify_curve(ys: list[float]) -> str:
    if all(y > 0 for y in ys):
        return "all_positive"
    if all(y < 0 for y in ys):
        return "all_negative"
    return "crosses_or_touches_zero"


def crossover_bootstrap(curves: dict[str, list[float]], xs: list[float], spec: dict) -> dict:
    seeds = sorted(curves)
    n = len(seeds)
    if n == 0:
        return {"n_seeds": 0}
    if n <= spec["exact_enumeration_max_seeds"]:
        resamples = list(itertools.product(range(n), repeat=n))
        method = f"exact enumeration of all {n}^{n} = {len(resamples)} equally likely seed resamples"
    else:
        rng = random.Random(spec["random_seed"])
        resamples = [tuple(rng.randrange(n) for _ in range(n)) for _ in range(spec["random_resamples"])]
        method = f"{len(resamples)} random seed resamples (seed {spec['random_seed']})"
    first_tokens, shapes = [], {"all_positive": 0, "all_negative": 0, "crosses_or_touches_zero": 0}
    for sample in resamples:
        mean_curve = [statistics.fmean(curves[seeds[i]][k] for i in sample) for k in range(len(xs))]
        shapes[classify_curve(mean_curve)] += 1
        found = crossings(xs, mean_curve)
        if found:
            first_tokens.append(found[0]["tokens"])
    first_tokens.sort()
    total = len(resamples)
    return {
        "n_seeds": n,
        "seeds": seeds,
        "method": method,
        "resamples": total,
        "fraction_with_crossover_in_range": len(first_tokens) / total,
        "fraction_all_positive": shapes["all_positive"] / total,
        "fraction_all_negative": shapes["all_negative"] / total,
        "crossover_tokens_percentiles_conditional_on_crossing": {
            "p2.5": percentile(first_tokens, 0.025),
            "p50": percentile(first_tokens, 0.5),
            "p97.5": percentile(first_tokens, 0.975),
        },
        "note": "percentiles are over resamples whose mean curve crosses zero inside the tested range; with n = 3 the bootstrap distribution is coarse (at most 10 distinct seed multisets).",
    }


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

def load_records(runs_root: Path, allow_synthetic: bool) -> tuple[list[dict], bool]:
    records = []
    for path in sorted((runs_root / "runs").glob(f"*/{RECORD_NAME}")):
        payload = json.loads(path.read_text())
        if payload.get("format") != RECORD_FORMAT:
            raise RuntimeError(f"unexpected record format: {path}")
        records.append(payload)
    synthetic = [bool(r.get("synthetic")) for r in records]
    if any(synthetic) and not all(synthetic):
        raise RuntimeError("refusing to mix synthetic and real run records")
    if any(synthetic) and not allow_synthetic:
        raise RuntimeError("synthetic records found; pass --allow-synthetic (self-test only)")
    return records, bool(records) and all(synthetic)


def select(records: list[dict], plan: dict) -> tuple[dict, list[dict]]:
    chosen: dict = {}
    excluded = []
    for record in records:
        if record.get("stage") != "E1":
            continue
        key = (record["recipe"], record["cell"], int(record["seed"]))
        if not record.get("integrity", {}).get("passed"):
            excluded.append({"run_id": record["run_id"], "attempt": record.get("attempt"), "reason": record.get("integrity", {}).get("failures")})
            continue
        if key in chosen:
            raise RuntimeError(f"two passing attempts for {key}")
        chosen[key] = record
    design = {(r["recipe"], r["cell"], int(r["seed"])) for r in plan["E1"]["runs"]}
    unknown = set(chosen) - design
    if unknown:
        raise RuntimeError(f"records outside the pre-registered design: {sorted(unknown)}")
    missing = sorted(design - set(chosen))
    return chosen, [{"missing_or_failed": list(key)} for key in missing] + excluded


def metric(record: dict, name: str) -> float | None:
    value = record.get("metrics", {}).get(name)
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def paired(chosen: dict, cell: str, seeds: list[int], name: str, treatment: str, control: str) -> dict[int, float]:
    out = {}
    for seed in seeds:
        a, b = chosen.get((treatment, cell, seed)), chosen.get((control, cell, seed))
        if a and b and metric(a, name) is not None and metric(b, name) is not None:
            out[seed] = metric(a, name) - metric(b, name)
    return out


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def categorize(summary: dict, margin: float) -> str:
    low, high = summary["ci"]
    mean = summary["mean"]
    if mean is None or low is None:
        return "insufficient_data"
    if high < 0:
        return "AP-Q_better"
    if low > 0:
        return "AR-ref_better"
    if abs(mean) < margin:
        return "approximately_zero"
    return "inconclusive"


def analyze(plan: dict, chosen: dict) -> dict:
    spec = plan["analysis_plan"]
    confidence = spec["confidence_level"]
    treatment, control = spec["treatment_recipe"], spec["control_recipe"]
    seeds = list(plan["seeds"]["E1"])
    token_cells = spec["token_cells_in_order"]
    fixed_cell = spec["fixed_time_cell"]
    budgets = [plan["cells"][c]["token_budget"] for c in token_cells]
    xs = [math.log(b) for b in budgets]
    margin = spec["decision_rules"]["approximately_zero_margin_bpb"]
    result: dict = {"metrics": {}, "token_budgets": budgets, "seeds": seeds}

    for name in [spec["primary_metric"], *spec["secondary_metrics"]]:
        per_cell, deltas = {}, {}
        for cell in token_cells + [fixed_cell]:
            values = paired(chosen, cell, seeds, name, treatment, control)
            deltas[cell] = values
            per_cell[cell] = {**summarize([values[s] for s in sorted(values)], confidence), "seeds": sorted(values)}
        means = [per_cell[c]["mean"] for c in token_cells]
        curve_ok = all(m is not None for m in means)
        found = crossings(xs, means) if curve_ok else []
        complete = {str(s): [deltas[c][s] for c in token_cells] for s in seeds if all(s in deltas[c] for c in token_cells)}
        contrast = {s: deltas[token_cells[-1]][s] - deltas[token_cells[0]][s] for s in seeds if s in deltas[token_cells[-1]] and s in deltas[token_cells[0]]}
        contrast_summary = summarize([contrast[s] for s in sorted(contrast)], confidence)
        increasing = curve_ok and all(means[i] < means[i + 1] for i in range(len(means) - 1))
        if contrast_summary["ci"][0] is not None and contrast_summary["ci"][0] > 0 and increasing:
            h_a = "supported"
        elif contrast_summary["ci"][1] is not None and contrast_summary["ci"][1] < 0:
            h_a = "contradicted"
        else:
            h_a = "inconclusive"
        first = per_cell[token_cells[0]]
        last = per_cell[token_cells[-1]]
        cat_first, cat_last = categorize(first, margin), categorize(last, margin)
        h_b = {
            "AP-Q_better": "supported",
            "approximately_zero": "not_rejected_delta_approximately_zero",
            "inconclusive": "not_rejected" if (first["mean"] or 0) <= 0 else "inconclusive_mean_positive",
            "AR-ref_better": "contradicted",
        }.get(cat_first, "insufficient_data")
        if cat_first == "AP-Q_better" and cat_last == "AR-ref_better":
            outcome = "crossover"
            text = "AP-Q is better below about {:.1f}M training tokens and worse above (mean-curve crossover).".format(found[0]["tokens"] / 1e6) if found else "crossover categories met but no interpolated crossing"
        elif cat_first in {"approximately_zero", "AR-ref_better"}:
            outcome = "exposure_explains_fixed_time_advantage"
            text = "At equal tokens AP-Q is not better even at 88.6M; the fixed-time advantage is explained by extra exposure."
        else:
            outcome = "unresolved"
            text = "Neither pre-registered outcome pattern is met; report the per-budget intervals."
        result["metrics"][name] = {
            "per_cell": per_cell,
            "mean_curve": {"log_tokens": xs, "mean_delta": means},
            "crossover": {"crossings": found, "shape": classify_curve(means) if curve_ok else "incomplete"},
            "crossover_bootstrap": crossover_bootstrap(complete, xs, spec["bootstrap"]),
            "H_E1a": {"verdict": h_a, "means_strictly_increasing": increasing, "contrast_largest_minus_smallest_budget": contrast_summary},
            "H_E1b": {"verdict": h_b, "category_at_smallest_budget": cat_first, "category_at_largest_budget": cat_last},
            "outcome_table": {"outcome": outcome, "paper_text_template": text},
        }

    # Fixed-time cell vs the earlier seeds 0-2 result.
    history = spec["historical_fixed_time"]
    hist_values = [history["per_seed_delta"][k] for k in sorted(history["per_seed_delta"])]
    current = paired(chosen, fixed_cell, seeds, history["metric"], treatment, control)
    current_summary = summarize([current[s] for s in sorted(current)], confidence)
    comparison = welch([current[s] for s in sorted(current)], hist_values, confidence)
    low, high = current_summary["ci"]
    result["fixed_time_vs_history"] = {
        "metric": history["metric"],
        "historical_seeds": sorted(int(k) for k in history["per_seed_delta"]),
        "historical_summary": summarize(hist_values, confidence),
        "historical_reported_mean": history["reported_mean"],
        "e1_summary": {**current_summary, "seeds": sorted(current)},
        "e1_ci_contains_historical_mean": None if low is None else low <= history["reported_mean"] <= high,
        "welch_e1_minus_history": comparison,
        "verdict": None if comparison["ci"][0] is None else ("consistent" if comparison["ci"][0] <= 0 <= comparison["ci"][1] else "differs"),
        "tokens_per_recipe": {
            recipe: {str(s): chosen[(recipe, fixed_cell, s)]["metrics"].get("tokens") for s in seeds if (recipe, fixed_cell, s) in chosen}
            for recipe in (treatment, control)
        },
    }

    # Schedule checks: fixed-time vs the token cell of similar exposure, same recipe and seed.
    schedule = []
    for check in spec["schedule_checks"]:
        row = {**check, "metrics": {}}
        for name in [spec["primary_metric"], *spec["secondary_metrics"]]:
            diffs = {}
            for seed in seeds:
                a = chosen.get((check["recipe"], fixed_cell, seed))
                b = chosen.get((check["recipe"], check["token_cell"], seed))
                if a and b and metric(a, name) is not None and metric(b, name) is not None:
                    diffs[seed] = metric(a, name) - metric(b, name)
            summary = summarize([diffs[s] for s in sorted(diffs)], confidence)
            detected = summary["ci"][0] is not None and not (summary["ci"][0] <= 0 <= summary["ci"][1]) and abs(summary["mean"]) > margin
            row["metrics"][name] = {**summary, "seeds": sorted(diffs), "schedule_effect_detected": detected}
        row["fixed_time_tokens"] = {str(s): chosen[(check["recipe"], fixed_cell, s)]["metrics"].get("tokens") for s in seeds if (check["recipe"], fixed_cell, s) in chosen}
        row["token_cell_tokens"] = plan["cells"][check["token_cell"]]["token_budget"]
        schedule.append(row)
    result["schedule_checks"] = schedule

    # Descriptive throughput / memory per recipe and cell.
    descriptive = {}
    for recipe in (treatment, control):
        for cell in token_cells + [fixed_cell]:
            rows = [chosen[(recipe, cell, s)]["metrics"] for s in seeds if (recipe, cell, s) in chosen]
            def mean_of(key):
                values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
                return statistics.fmean(values) if values else None
            descriptive[f"{recipe}|{cell}"] = {
                "n": len(rows),
                "tokens_mean": mean_of("tokens"),
                "updates_mean": mean_of("updates"),
                "steady_state_seconds_mean": mean_of("steady_state_seconds"),
                "steady_state_tokens_per_second_mean": mean_of("steady_state_tokens_per_second"),
                "peak_training_memory_mb_mean": mean_of("peak_training_memory_mb"),
            }
    result["descriptive"] = descriptive
    return result


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(plan: dict, chosen: dict, result: dict, output_dir: Path, synthetic: bool) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec = plan["analysis_plan"]
    seeds = result["seeds"]
    token_cells = spec["token_cells_in_order"]
    fixed_cell = spec["fixed_time_cell"]
    budgets = result["token_budgets"]
    treatment, control = spec["treatment_recipe"], spec["control_recipe"]
    rows = [(spec["primary_metric"], "held-out BPB"), (spec["secondary_metrics"][0], "validation BPB")]
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.0), gridspec_kw={"width_ratios": [3.2, 1.0]}, sharey="row")
    # Mean and CI sit exactly on each budget; seed points sit just to the right so none is hidden.
    log_offset = {s: 0.03 + 0.025 * i for i, s in enumerate(seeds)}
    side_offset = {s: 0.24 + 0.08 * i for i, s in enumerate(seeds)}
    point = {"s": 22, "edgecolors": "white", "linewidths": 0.6, "zorder": 3}

    def summary_bar(axis, x0, summary):
        if summary["mean"] is not None:
            axis.plot([x0 - 0.16, x0 + 0.16], [summary["mean"]] * 2, color=INK, linewidth=1.6, zorder=2)
            if summary["ci"][0] is not None:
                axis.plot([x0, x0], summary["ci"], color=INK, linewidth=1.0, zorder=2)

    for r, (name, label) in enumerate(rows):
        block = result["metrics"][name]
        ax, side = axes[r][0], axes[r][1]
        for ax_ in (ax, side):
            ax_.axhline(0.0, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
            ax_.grid(axis="y", color=GRID, linewidth=0.5)
            ax_.spines[["top", "right"]].set_visible(False)
        for i, seed in enumerate(seeds):
            color, marker = SEED_STYLE[i % len(SEED_STYLE)]
            pts = [(b * math.exp(log_offset[seed]), block["per_cell"][c]["values"][block["per_cell"][c]["seeds"].index(seed)])
                   for c, b in zip(token_cells, budgets) if seed in block["per_cell"][c]["seeds"]]
            if pts:
                ax.scatter(*zip(*pts), color=color, marker=marker, label=f"seed {seed}", **point)
            fixed = block["per_cell"][fixed_cell]
            if seed in fixed["seeds"] and name != spec["secondary_metrics"][0]:
                side.scatter([side_offset[seed]], [fixed["values"][fixed["seeds"].index(seed)]], color=color, marker=marker, **point)
        means = [block["per_cell"][c]["mean"] for c in token_cells]
        cis = [block["per_cell"][c]["ci"] for c in token_cells]
        valid = [(b, m, ci) for b, m, ci in zip(budgets, means, cis) if m is not None]
        if valid:
            ax.plot([v[0] for v in valid], [v[1] for v in valid], color=INK, linewidth=1.6, zorder=2, label="mean, 95% t-CI")
            for b, m, ci in valid:
                if ci[0] is not None:
                    ax.plot([b, b], ci, color=INK, linewidth=1.0, zorder=2)
        crossing = block["crossover"]["crossings"]
        if crossing:
            ax.axvline(crossing[0]["tokens"], color=MUTED, linewidth=0.8, linestyle=":")
            ax.annotate(f"crossover ~{crossing[0]['tokens'] / 1e6:.0f}M", (crossing[0]["tokens"], 1.0), xycoords=("data", "axes fraction"),
                        xytext=(3, -10), textcoords="offset points", color=MUTED, fontsize=7)
        ax.set_xscale("log")
        ax.set_xticks(budgets)
        ax.set_xticklabels([f"{b / 1e6:.1f}M" for b in budgets])
        ax.minorticks_off()
        ax.set_xlim(budgets[0] / 1.18, budgets[-1] * 1.18)
        ax.set_ylabel(f"Δ {label}\n(AP-Q − AR-ref)")
        ax.set_title(f"Fixed tokens: Δ {label}", loc="left", fontsize=8.5, color=INK)
        if r == 1:
            ax.set_xlabel("training tokens (log scale), batch 131,072")
        # Fixed-time side panel.
        if name == spec["secondary_metrics"][0]:
            fx = result["fixed_time_vs_history"]
            for j, value in enumerate(fx["e1_summary"]["values"]):
                seed = fx["e1_summary"]["seeds"][j]
                color, marker = SEED_STYLE[seeds.index(seed) % len(SEED_STYLE)]
                side.scatter([side_offset[seed]], [value], color=color, marker=marker, **point)
            for j, value in enumerate(fx["historical_summary"]["values"]):
                side.scatter([1.0 + 0.24 + 0.08 * j], [value], color=MUTED, marker="D", **{**point, "s": 18})
            summary_bar(side, 0.0, fx["e1_summary"])
            summary_bar(side, 1.0, fx["historical_summary"])
            side.set_xticks([0.0, 1.0])
            side.set_xticklabels(["seeds\n6-8", "seeds 0-2\n(history)"], fontsize=7)
            side.set_xlim(-0.35, 1.6)
            side.set_title("300 s: Δ val (in-process)", loc="left", fontsize=8.5, color=INK)
        else:
            summary_bar(side, 0.0, block["per_cell"][fixed_cell])
            side.set_xticks([0.0])
            side.set_xticklabels(["seeds\n6-8"], fontsize=7)
            side.set_xlim(-0.35, 0.6)
            side.set_title(f"300 s: Δ {label.split()[0]}", loc="left", fontsize=8.5, color=INK)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
    if synthetic:
        fig.suptitle("SYNTHETIC TEST DATA - NOT EXPERIMENTAL RESULTS", color="#e34948", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95 if synthetic else 1))
    paths = []
    for suffix in ("pdf", "png"):
        path = output_dir / f"e1_delta_vs_tokens.{suffix}"
        fig.savefig(path, dpi=200)
        paths.append(str(path))
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def fmt(value, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:+.{digits}f}"


def markdown(plan: dict, result: dict, synthetic: bool) -> str:
    spec = plan["analysis_plan"]
    lines = []
    if synthetic:
        lines += ["**SYNTHETIC TEST DATA - NOT EXPERIMENTAL RESULTS**", ""]
    lines += ["# E1 analysis (pre-registered)", "", f"Delta = BPB({spec['treatment_recipe']}) - BPB({spec['control_recipe']}); mean with 95% t-interval over seeds {result['seeds']}.", ""]
    for name, block in result["metrics"].items():
        lines += [f"## {name}", "", "| cell | n | mean Delta | 95% CI | per seed |", "|---|---|---|---|---|"]
        for cell, row in block["per_cell"].items():
            lines.append(f"| {cell} | {row['n']} | {fmt(row['mean'])} | [{fmt(row['ci'][0])}, {fmt(row['ci'][1])}] | {', '.join(fmt(v) for v in row['values'])} |")
        cross = block["crossover"]["crossings"]
        boot = block["crossover_bootstrap"]
        lines += ["", f"- Crossover (mean curve, log-token interpolation): " + (", ".join(f"{c['tokens'] / 1e6:.1f}M ({c['direction']})" for c in cross) or f"none in range ({block['crossover']['shape']})"),
                  f"- Seed bootstrap: {boot.get('method')}; P(crossover in range) = {boot.get('fraction_with_crossover_in_range')}",
                  f"- H-E1a: {block['H_E1a']['verdict']}; H-E1b: {block['H_E1b']['verdict']}; outcome: {block['outcome_table']['outcome']}", ""]
    fx = result["fixed_time_vs_history"]
    lines += ["## 300 s cell vs seeds 0-2", "",
              f"- E1 mean {fmt(fx['e1_summary']['mean'])} [{fmt(fx['e1_summary']['ci'][0])}, {fmt(fx['e1_summary']['ci'][1])}] vs historical {fmt(fx['historical_reported_mean'])}; Welch difference {fmt(fx['welch_e1_minus_history']['difference'])} [{fmt(fx['welch_e1_minus_history']['ci'][0])}, {fmt(fx['welch_e1_minus_history']['ci'][1])}] -> {fx['verdict']}", ""]
    lines += ["## Schedule checks", ""]
    for row in result["schedule_checks"]:
        primary = row["metrics"][spec["primary_metric"]]
        lines.append(f"- {row['recipe']}: {row['fixed_time_cell']} - {row['token_cell']}: {fmt(primary['mean'])} [{fmt(primary['ci'][0])}, {fmt(primary['ci'][1])}]; effect detected: {primary['schedule_effect_detected']}; fixed-time tokens {row['fixed_time_tokens']}")
    return "\n".join(lines) + "\n"


def run_analysis(runs_root: Path, plan: dict, output_dir: Path, *, allow_synthetic: bool, figure: bool, overwrite: bool) -> dict:
    records, synthetic = load_records(runs_root, allow_synthetic)
    if not records:
        raise RuntimeError(f"no run records under {runs_root / 'runs'}")
    if synthetic and (ROOT / "results").resolve() in (output_dir.resolve(), *output_dir.resolve().parents):
        raise RuntimeError("synthetic analysis output must not be written under results/")
    chosen, exclusions = select(records, plan)
    result = analyze(plan, chosen)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [output_dir / "e1_analysis.json", output_dir / "e1_analysis.md", output_dir / "e1_delta_vs_tokens.pdf", output_dir / "e1_delta_vs_tokens.png"]
    if not overwrite and any(path.exists() for path in targets):
        raise RuntimeError(f"analysis outputs already exist in {output_dir}; pass --overwrite")
    figure_paths: list[str] = []
    figure_status = "skipped (--no-figure)"
    if figure:
        try:
            figure_paths = make_figure(plan, chosen, result, output_dir, synthetic)
            figure_status = "written"
        except ImportError as exc:
            figure_status = f"skipped: {exc}"
    payload = {
        "format": "autopareto_camera_ready_e1_analysis_v1",
        "synthetic": synthetic,
        "label": "SYNTHETIC TEST DATA - NOT EXPERIMENTAL RESULTS" if synthetic else "E1 results",
        "runs_root": str(runs_root),
        "plan_sha256": __import__("hashlib").sha256(PLAN_PATH.read_bytes()).hexdigest(),
        "runs_used": len(chosen),
        "exclusions_and_missing": exclusions,
        "figure": {"status": figure_status, "paths": figure_paths},
        **result,
    }
    (output_dir / "e1_analysis.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (output_dir / "e1_analysis.md").write_text(markdown(plan, result, synthetic))
    return payload


# ---------------------------------------------------------------------------
# Synthetic self-test (never touches results/)
# ---------------------------------------------------------------------------

def fabricate(plan: dict, directory: Path, *, delta: dict[str, float], noise: list[float], drop: set | None = None) -> None:
    """Write clearly labelled synthetic run records with known paired Deltas."""
    base = {"tokens_088604672": 1.150, "tokens_157286400": 1.118, "tokens_201326592": 1.101, "time_300s": 1.152}
    seed_offset = {6: 0.003, 7: -0.001, 8: 0.0}
    for run in plan["E1"]["runs"]:
        key = (run["recipe"], run["cell"], run["seed"])
        if drop and key in drop:
            continue
        cell = plan["cells"][run["cell"]]
        index = plan["seeds"]["E1"].index(run["seed"])
        control = base[run["cell"]] + seed_offset[run["seed"]]
        value = control + (delta[run["cell"]] + noise[index] if run["recipe"] == "AP-Q" else 0.0)
        tokens = cell.get("token_budget") or (157_286_400 if run["recipe"] == "AP-Q" else 88_473_600)
        record = {
            "format": RECORD_FORMAT, "synthetic": True, "label": "SYNTHETIC TEST DATA - NOT EXPERIMENTAL RESULTS",
            "stage": "E1", "run_id": run["run_id"], "attempt": 1, "recipe": run["recipe"], "cell": run["cell"],
            "seed": run["seed"], "status": "completed", "integrity": {"passed": True, "failures": []},
            "metrics": {
                "heldout_bpb": value + 0.06, "validation_bpb": value, "validation_bpb_in_process": value - 1e-5,
                "tokens": tokens, "updates": tokens // 131072, "steady_state_seconds": tokens / 4e5,
                "steady_state_tokens_per_second": 4e5, "peak_training_memory_mb": 12000.0,
            },
        }
        path = directory / "runs" / run["run_id"] / RECORD_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record))


def self_test(plan: dict, keep: Path | None) -> int:
    def close(a, b, tol):
        assert a is not None and abs(a - b) <= tol, (a, b)

    # Student-t quantiles against published values.
    for df, expected in ((1, 12.706204736), (2, 4.302652730), (4, 2.776445105), (10, 2.228138852), (30, 2.042272456)):
        close(t_quantile(0.975, df), expected, 1e-6)
    close(t_cdf(0.0, 3), 0.5, 1e-12)

    workdir = Path(tempfile.mkdtemp(prefix="SYNTHETIC-e1-selftest-")) if keep is None else keep
    if (ROOT / "results").resolve() in (workdir.resolve(), *workdir.resolve().parents):
        raise RuntimeError("self-test directory must be outside results/")
    try:
        delta = {"tokens_088604672": -0.010, "tokens_157286400": 0.015, "tokens_201326592": 0.040, "time_300s": -0.007}
        noise = [-0.002, 0.0, 0.002]
        full = workdir / "case_full"
        fabricate(plan, full, delta=delta, noise=noise)
        out = run_analysis(full, plan, full / "analysis", allow_synthetic=True, figure=True, overwrite=False)
        held = out["metrics"]["heldout_bpb"]
        t1, t2 = 88_604_672, 157_286_400
        expected_cross = math.exp(math.log(t1) + 0.4 * (math.log(t2) - math.log(t1)))
        close(held["crossover"]["crossings"][0]["tokens"], expected_cross, 1.0)
        assert held["crossover"]["crossings"][0]["direction"] == "upward"
        first = held["per_cell"]["tokens_088604672"]
        close(first["mean"], -0.010, 1e-12)
        close(first["ci"][1] - first["mean"], 4.302652730 * 0.002 / math.sqrt(3), 1e-9)
        assert held["H_E1a"]["verdict"] == "supported", held["H_E1a"]
        assert held["H_E1b"]["verdict"] == "supported", held["H_E1b"]
        assert held["outcome_table"]["outcome"] == "crossover"
        boot = held["crossover_bootstrap"]
        assert boot["resamples"] == 27 and boot["fraction_with_crossover_in_range"] == 1.0, boot
        fx = out["fixed_time_vs_history"]
        assert fx["verdict"] == "consistent", fx
        assert out["synthetic"] is True
        assert out["figure"]["status"] == "written" or out["figure"]["status"].startswith("skipped"), out["figure"]
        assert "SYNTHETIC" in (full / "analysis" / "e1_analysis.md").read_text()

        # Exposure outcome and a failed run (n = 2 at one budget).
        exposure = workdir / "case_exposure_missing"
        fabricate(plan, exposure, delta={**delta, "tokens_088604672": 0.012}, noise=noise,
                  drop={("AR-ref", "tokens_157286400", 8)})
        out2 = run_analysis(exposure, plan, exposure / "analysis", allow_synthetic=True, figure=False, overwrite=False)
        held2 = out2["metrics"]["heldout_bpb"]
        assert held2["per_cell"]["tokens_157286400"]["n"] == 2
        assert held2["crossover_bootstrap"]["n_seeds"] == 2
        assert held2["H_E1b"]["verdict"] == "contradicted"
        assert held2["outcome_table"]["outcome"] == "exposure_explains_fixed_time_advantage"
        assert any("missing_or_failed" in row for row in out2["exclusions_and_missing"])

        # Guards: synthetic records are refused without the flag.
        try:
            run_analysis(full, plan, full / "analysis-guard", allow_synthetic=False, figure=False, overwrite=False)
        except RuntimeError:
            pass
        else:
            raise AssertionError("synthetic guard did not trigger")
        print(json.dumps({"self_test": "passed", "synthetic_workdir": str(workdir), "figure": out["figure"]}, indent=2))
        return 0
    finally:
        if keep is None:
            shutil.rmtree(workdir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", type=Path, default=ROOT / "results/camera_ready_2026/amendment1/E1")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-synthetic", action="store_true", help="accept synthetic records (testing only)")
    parser.add_argument("--self-test", action="store_true", help="fabricate synthetic inputs in a temp dir and check known answers")
    parser.add_argument("--keep-self-test-dir", type=Path, help="write the synthetic self-test here (outside results/) and keep it")
    args = parser.parse_args()
    plan = json.loads(PLAN_PATH.read_text())
    if args.self_test:
        return self_test(plan, args.keep_self_test_dir)
    output_dir = args.output_dir or args.runs_root / "analysis"
    payload = run_analysis(args.runs_root, plan, output_dir, allow_synthetic=args.allow_synthetic, figure=not args.no_figure, overwrite=args.overwrite)
    print(json.dumps({
        "status": "ok", "synthetic": payload["synthetic"], "runs_used": payload["runs_used"], "output_dir": str(output_dir),
        "figure": payload["figure"], "primary": {k: v["outcome_table"]["outcome"] for k, v in payload["metrics"].items()},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
