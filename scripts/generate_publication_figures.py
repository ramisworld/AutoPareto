#!/usr/bin/env python3
"""Generate publication figures from the frozen Phase 1 artifacts.

This script is deliberately CPU-only. It reads the checked-in Phase 1 summary CSV/
JSON and small aggregate downstream summaries; it never loads a checkpoint or invokes
an evaluator. The figures are descriptive and keep each metric in its own panel.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / "plots"
PHASE1_JSON = ROOT / "results" / "phase1" / "phase1_equal_token_results.json"
PHASE1_CSV = ROOT / "results" / "phase1" / "phase1_equal_token_results.csv"
DOWNSTREAM_SUMMARY = ROOT / "results" / "phase1" / "public_downstream_metrics.json"
DISCOVERY_BPB = ROOT / "results" / "phase1" / "discovery_heldout_bpb.json"

MODEL_ORDER = [
    "baseline",
    "autoresearch_compatible",
    "tpe_quality",
    "motpe_quality",
    "autopareto_292",
    "autopareto_299",
    "autopareto_294",
]
LABELS = {
    "baseline": "Baseline",
    "autoresearch_compatible": "AutoResearch",
    "tpe_quality": "TPE",
    "motpe_quality": "MOTPE",
    "autopareto_292": "AP292",
    "autopareto_299": "AP299",
    "autopareto_294": "AP294",
}
COLORS = {
    "baseline": "#5b6675",
    "autoresearch_compatible": "#386cb0",
    "tpe_quality": "#6b7280",
    "motpe_quality": "#8b5e34",
    "autopareto_292": "#c2410c",
    "autopareto_299": "#d97706",
    "autopareto_294": "#eab308",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    PLOTS.mkdir(exist_ok=True)
    fig.savefig(PLOTS / f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(PLOTS / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_phase1() -> tuple[dict[str, dict], list[dict]]:
    summary = json.loads(PHASE1_JSON.read_text())
    models = {item["model_id"]: item for item in summary["models"]}
    with PHASE1_CSV.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 21
    assert all(int(row["tokens"]) == 201_326_592 for row in rows)
    return models, rows


def load_downstream() -> dict[tuple[str, str], dict[str, tuple[float, float]]]:
    payload = json.loads(DOWNSTREAM_SUMMARY.read_text())
    assert payload["campaign"] == "phase1-200m-downstream"
    assert payload["token_budget"] == 201_326_592
    output: dict[tuple[str, str], dict[str, tuple[float, float]]] = {}
    for method, model in payload["models"].items():
        for task, metrics in model["metrics"].items():
            output[(method, task)] = {
                metric: (
                    float(values["mean"]),
                    float(values["sample_standard_deviation"]),
                )
                for metric, values in metrics.items()
            }
    # Every 200M recipe has the complete downstream bundle used by the figures.
    assert all(
        sum(len(metrics) for (method, task), metrics in output.items() if method == recipe) >= 6
        for recipe in MODEL_ORDER
    )
    return output


def load_parameter_counts() -> dict[str, int]:
    payload = json.loads(DOWNSTREAM_SUMMARY.read_text())
    counts = {
        method: int(model["parameter_count"])
        for method, model in payload["models"].items()
    }
    assert set(counts) == set(MODEL_ORDER)
    return counts


def method_stats(phase1: dict[str, dict], parameter_counts: dict[str, int], model: str) -> dict[str, tuple[float, float]]:
    item = phase1[model]

    def pair(key: str) -> tuple[float, float]:
        value = item[key]
        return float(value["mean"]), float(value["sample_standard_deviation"])

    return {
        "params_m": (parameter_counts[model] / 1e6, 0.0),
        "heldout_bpb": pair("heldout_bpb"),
        "decode": pair("stabilized_decode_tokens_per_second"),
        "wall_s": pair("total_scientific_wall_seconds"),
    }


def pareto_frontier(phase1: dict[str, dict], parameter_counts: dict[str, int]) -> None:
    fig, ax = plt.subplots(figsize=(8.7, 5.4))
    for model in MODEL_ORDER:
        stats = method_stats(phase1, parameter_counts, model)
        x, xerr = stats["decode"]
        y, yerr = stats["heldout_bpb"]
        emphasis = model in {"autopareto_292", "autopareto_299", "autopareto_294"}
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            ms=10 if model == "autopareto_292" else 7,
            lw=1.2,
            capsize=3,
            color=COLORS[model],
            markeredgecolor="#111827" if emphasis else "white",
            markeredgewidth=1.5 if model == "autopareto_292" else 0.8,
            label=LABELS[model],
            zorder=4 if emphasis else 3,
        )
        dx, dy = {
            "baseline": (-8, 0.0016),
            "autoresearch_compatible": (-18, -0.0032),
            "tpe_quality": (-10, 0.0028),
            "motpe_quality": (-25, -0.0030),
            "autopareto_292": (7, -0.0027),
            "autopareto_299": (7, 0.0027),
            "autopareto_294": (7, -0.0030),
        }[model]
        ax.annotate(LABELS[model], (x, y), xytext=(dx, dy), textcoords="offset points")
    ax.invert_yaxis()
    ax.set_xlabel("Stabilized cached decode throughput (tokens/s), mean ± sample SD")
    ax.set_ylabel("Held-out ClimbMix BPB, mean ± sample SD (lower is better)")
    fig.suptitle("201.3M-token verification: quality–decode frontier", y=0.98, fontsize=14)
    fig.text(
        0.01,
        0.93,
        "Three seeds (3, 4, 5); A40; each point is a recipe mean, not a single run",
        fontsize=9,
        color="#4b5563",
    )
    ax.legend(loc="lower left", ncol=2)
    ax.set_xlim(175, 535)
    ax.set_ylim(1.205, 1.085)
    fig.text(
        0.01,
        -0.015,
        "AP292 is the balanced AutoPareto candidate; AP299/AP294 move right toward efficiency while giving up held-out BPB.",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.90])
    save(fig, "main_pareto_frontier")


def point_panel(ax: plt.Axes, labels: list[str], values: list[float], errors: list[float], title: str, ylabel: str, fmt: str = "{:.2f}") -> None:
    x = np.arange(len(labels))
    colors = ["#386cb0", "#c2410c", "#64748b", "#8b5e34", "#eab308"]
    for i, (value, error) in enumerate(zip(values, errors)):
        ax.errorbar(i, value, yerr=error, fmt="o", ms=8, capsize=4, color=colors[i], lw=1.4)
        ax.text(i, value + error + (max(values) - min(values) or 1) * 0.04, fmt.format(value), ha="center", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    low = min(0, min(values) - max(errors) * 1.4)
    high = max(values) + max(errors) * 2.0 + (max(values) - min(values) or 1) * 0.08
    ax.set_ylim(low, high)


def ap292_vs_autoresearch(phase1: dict[str, dict], parameter_counts: dict[str, int], downstream: dict[tuple[str, str], dict[str, tuple[float, float]]]) -> None:
    models = ["autoresearch_compatible", "autopareto_292"]
    labels = ["AutoResearch", "AP292"]
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.5))
    panels = [
        ("params_m", "Parameters (millions)", "Parameters", "{:.1f}"),
        ("wall_s", "Fixed-token scientific wall time (s)", "Seconds", "{:.1f}"),
        ("decode", "Cached decode throughput (tokens/s)", "Tokens/s", "{:.1f}"),
        ("heldout_bpb", "Held-out BPB (lower is better)", "Bits/byte", "{:.3f}"),
    ]
    for ax, (key, title, ylabel, fmt) in zip(axes[0], panels):
        pairs = [method_stats(phase1, parameter_counts, model)[key] for model in models]
        point_panel(ax, labels, [p[0] for p in pairs], [p[1] for p in pairs], title, ylabel, fmt)

    task_panels = [
        ("arc_easy", "accuracy", "ARC-Easy accuracy (higher is better)"),
        ("piqa", "accuracy", "PIQA accuracy (higher is better)"),
        ("lambada_openai", "exact_match", "LAMBADA exact match (higher is better)"),
        ("blimp", "micro_accuracy", "BLiMP micro accuracy (higher is better)"),
    ]
    for ax, (task, metric, title) in zip(axes[1], task_panels):
        pairs = [downstream[(model, task)][metric] for model in models]
        point_panel(ax, labels, [100 * p[0] for p in pairs], [100 * p[1] for p in pairs], title, "Percent", "{:.1f}%")
        ax.set_ylim(0, max(100 * (p[0] + p[1]) for p in pairs) * 1.25)
    fig.suptitle("AutoResearch versus AutoPareto 292 at 201.3M training tokens", y=1.01, fontsize=14)
    fig.text(0.01, -0.015, "Each panel has its own units and scale; points are means ± sample SD across seeds 3–5.", fontsize=9)
    fig.tight_layout()
    save(fig, "ap292_vs_autoresearch")


def downstream_comparison(downstream: dict[tuple[str, str], dict[str, tuple[float, float]]]) -> None:
    models = ["autoresearch_compatible", "autopareto_292", "tpe_quality", "motpe_quality"]
    labels = ["AutoResearch", "AP292", "TPE", "MOTPE"]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.5))
    panels = [
        ("arc_easy", "accuracy", "ARC-Easy accuracy (higher is better)"),
        ("piqa", "accuracy", "PIQA accuracy (higher is better)"),
        ("lambada_openai", "exact_match", "LAMBADA exact match (higher is better)"),
        ("blimp", "micro_accuracy", "BLiMP micro accuracy (higher is better)"),
        ("lambada_openai", "perplexity", "LAMBADA perplexity (lower is better)"),
        ("heldout_climbmix_bpb", "bits_per_byte", "Held-out BPB (lower is better)"),
    ]
    for ax, (task, metric, title) in zip(axes.flat, panels):
        pairs = [downstream[(model, task)][metric] for model in models]
        scale = 100 if metric in {"accuracy", "exact_match", "micro_accuracy"} else 1
        point_panel(ax, labels, [scale * p[0] for p in pairs], [scale * p[1] for p in pairs], title, "Percent" if scale == 100 else "Value", "{:.1f}%" if scale == 100 else "{:.2f}")
        if scale == 100:
            ax.set_ylim(0, max(scale * (p[0] + p[1]) for p in pairs) * 1.25)
    fig.suptitle("201.3M-token verification: downstream comparison", y=1.01, fontsize=14)
    fig.text(0.01, -0.015, "Means ± sample SD across seeds 3–5. LAMBADA perplexity and BPB are separate lower-is-better panels.", fontsize=9)
    fig.tight_layout()
    save(fig, "downstream_comparison")


def load_discovery_bpb() -> dict[str, float]:
    payload = json.loads(DISCOVERY_BPB.read_text())
    assert payload["campaign"] == "five-minute discovery"
    return {model: float(value["heldout_bpb"]) for model, value in payload["models"].items()}


def discovery_to_verification(phase1: dict[str, dict]) -> None:
    old = load_discovery_bpb()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(MODEL_ORDER))
    for i, model in enumerate(MODEL_ORDER):
        item = phase1[model]["heldout_bpb"]
        color = COLORS[model]
        ax.plot([i - 0.16, i + 0.16], [old[model], item["mean"]], color="#cbd5e1", lw=1.5, zorder=1)
        ax.scatter(i - 0.16, old[model], s=48, color="#64748b", edgecolor="white", zorder=3)
        ax.errorbar(i + 0.16, item["mean"], yerr=item["sample_standard_deviation"], fmt="o", ms=8, capsize=3, color=color, zorder=4)
    ax.set_xticks(x, [LABELS[m] for m in MODEL_ORDER])
    ax.set_ylabel("Held-out ClimbMix BPB (lower is better)")
    fig.suptitle("Discovery-to-verification context: held-out BPB", y=0.98, fontsize=14)
    fig.text(0.01, 0.93, "Left: historical five-minute selected representative; right: 201.3M-token mean ± SD", fontsize=9, color="#4b5563")
    ax.set_ylim(1.05, 1.32)
    ax.legend([plt.Line2D([0], [0], marker="o", color="#64748b", lw=0), plt.Line2D([0], [0], marker="o", color="#c2410c", lw=0)], ["5-minute discovery", "201.3M verification"], loc="upper right")
    fig.text(0.01, -0.015, "This is contextual, not a paired causal estimate: the campaigns use different seeds and the evaluator SHAs differ.", fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 0.90])
    save(fig, "discovery_to_verification")


def main() -> None:
    configure_style()
    phase1, _ = load_phase1()
    downstream = load_downstream()
    parameter_counts = load_parameter_counts()
    pareto_frontier(phase1, parameter_counts)
    ap292_vs_autoresearch(phase1, parameter_counts, downstream)
    downstream_comparison(downstream)
    discovery_to_verification(phase1)
    print(f"wrote publication figures to {PLOTS}")


if __name__ == "__main__":
    main()
