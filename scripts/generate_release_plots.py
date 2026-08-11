#!/usr/bin/env python3
"""Regenerate the released plots from the reviewed public CSV exports."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PLOTS = ROOT / "plots"


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: str) -> float:
    return float(value)


def save(figure: plt.Figure, name: str) -> None:
    PLOTS.mkdir(exist_ok=True)
    figure.tight_layout()
    figure.savefig(PLOTS / f"{name}.svg", bbox_inches="tight")
    figure.savefig(PLOTS / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def discovery_frontier(experiments: list[dict[str, str]]) -> None:
    seed0 = [row for row in experiments if row["campaign_id"] == "autopareto-seed0-20260731" and row["status"] == "completed" and row["inference_status"] == "compatible"]
    baseline = next(row for row in experiments if row["campaign_id"] == "baseline-a40-cached-fresh-20260722" and row["training_seed"] == "0")
    reference = next(row for row in experiments if row["campaign_id"] == "autoresearch-seed42-20260723" and row["experiment_number"] == "48")

    figure, axis = plt.subplots(figsize=(10, 6.2))
    axis.scatter(
        [number(row["val_bpb"]) for row in seed0],
        [number(row["cached_decode_tokens_per_second"]) for row in seed0],
        s=[45 + number(row["allocated_inference_memory_mib"]) * 0.55 for row in seed0],
        c="#4f8cff", alpha=0.62, edgecolors="#1d4ed8", linewidths=0.8,
        label="Seed-0 discovery candidates",
    )
    axis.scatter(number(baseline["val_bpb"]), number(baseline["cached_decode_tokens_per_second"]), marker="s", s=70, c="#0f172a", label="Same-seed baseline")
    axis.scatter(number(reference["val_bpb"]), number(reference["cached_decode_tokens_per_second"]), marker="D", s=58, c="#f97316", label="Historical AutoResearch reference")
    for row in seed0:
        attempt = int(row["experiment_number"])
        if attempt in {8, 12, 22, 24, 25}:
            axis.annotate(f"A{attempt}", (number(row["val_bpb"]), number(row["cached_decode_tokens_per_second"])), xytext=(7, 7), textcoords="offset points", fontsize=9)
    axis.set_title("Discovery frontier (seed 0)")
    axis.set_xlabel("Validation BPB (lower is better)")
    axis.set_ylabel("Cached decode throughput (tokens/s, higher is better)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, loc="lower right")
    figure.text(0.125, 0.01, "bfloat16, one NVIDIA A40, 256 input + 256 generated tokens, batch 1, 10 measured repetitions. Discovery evidence only.", fontsize=8)
    save(figure, "discovery_frontier")


def ablation_plot() -> None:
    labels = ["Baseline", "Batch-only", "Full candidate\n(exp. 68)", "Non-batch\nbundle"]
    values = [1.2118846990, 1.1039678381, 1.1014253274, 1.2188020040]
    figure, axis = plt.subplots(figsize=(9.5, 5.4))
    bars = axis.bar(labels, values, color=["#94a3b8", "#2563eb", "#2563eb", "#dc2626"])
    axis.set_ylim(1.08, 1.24)
    axis.set_ylabel("Mean validation BPB (lower is better)")
    axis.set_title("Five-minute validation-BPB ablation")
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.002, f"{value:.4f}", ha="center", fontsize=9)
    figure.text(0.125, 0.01, "The batch-only condition explains nearly all observed validation-BPB reduction; the non-batch bundle does not establish an independent reduction.", fontsize=8)
    save(figure, "ablation_validation_bpb")


def deployment_plot(measurements: list[dict[str, str]]) -> None:
    selected = [row for row in measurements if row["experiment_id"] == "4" and row["is_warmup"] == "0"]
    figure, axis = plt.subplots(figsize=(9.5, 5.6))
    colors = {"1": "#2563eb", "2": "#16a34a", "4": "#f97316", "8": "#7c3aed"}
    for batch in ("1", "2", "4", "8"):
        grouped: dict[int, list[float]] = {}
        for row in selected:
            if row["batch_size"] == batch and row["decode_tokens_per_second"]:
                grouped.setdefault(int(row["context_tokens"]), []).append(number(row["decode_tokens_per_second"]))
        contexts = sorted(grouped)
        axis.plot(contexts, [median(grouped[context]) for context in contexts], marker="o", color=colors[batch], label=f"batch {batch}")
    axis.set_title("Deployment workload characterization")
    axis.set_xlabel("Input context tokens")
    axis.set_ylabel("Cached decode throughput (tokens/s)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.text(0.125, 0.01, "Completed baseline measurement, bfloat16 cached decode, 128 generated tokens, five repetitions per workload.", fontsize=8)
    save(figure, "deployment_workload")


def main() -> int:
    experiments = read_csv("experiment_records.csv")
    deployment = read_csv("deployment_measurements.csv")
    discovery_frontier(experiments)
    ablation_plot()
    deployment_plot(deployment)
    print(f"Wrote plots to {PLOTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

