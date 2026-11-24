#!/usr/bin/env python3
"""Read-only extraction of Phase 1 (201M-token) per-run evidence.

Reads only JSON/log files under the extracted snapshot. Writes nothing into the repo.
Output: JSON to stdout (redirect into the scratch folder).
"""
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path("/Users/Rami/Documents/dev/autopareto")
P1 = ROOT / "results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1"

STEP_RE = re.compile(r"step (\d{5}) \(([\d.]+)%\) \| loss: ([\d.]+) \| lrm: ([\d.]+) \| dt: (\d+)ms \| tok/sec: ([\d,]+) \| mfu: ([\d.]+)% \| epoch: (\d+)")


def parse_log(path: Path):
    text = path.read_text(errors="replace").replace("\r", "\n")
    flops = None
    m = re.search(r"Estimated FLOPs per token: ([0-9.e+]+)", text)
    if m:
        flops = float(m.group(1))
    cfg = None
    m = re.search(r"Model config: (\{.*\})", text)
    if m:
        cfg = m.group(1)
    steps = {}
    max_epoch = 0
    for mm in STEP_RE.finditer(text):
        s = int(mm.group(1))
        steps[s] = {
            "pct": float(mm.group(2)),
            "loss_ema": float(mm.group(3)),
            "lrm": float(mm.group(4)),
            "dt_ms": int(mm.group(5)),
            "mfu_h100_pct": float(mm.group(7)),
            "epoch": int(mm.group(8)),
        }
        max_epoch = max(max_epoch, int(mm.group(8)))
    tail = {}
    for key in ("val_bpb", "training_seconds", "total_seconds", "peak_vram_mb", "mfu_percent", "total_tokens_M", "num_steps", "num_params_M", "depth"):
        m = re.search(rf"^{key}:\s+([0-9.]+)", text, re.M)
        if m:
            tail[key] = float(m.group(1))
    return flops, cfg, steps, max_epoch, tail


def main():
    out = []
    for run_dir in sorted((P1 / "runs").glob("phase1-*")):
        tm = json.loads((run_dir / "training_metrics.json").read_text())
        ce = json.loads((run_dir / "checkpoint_evaluations.json").read_text())
        flops, cfg, steps, max_epoch, tail = parse_log(run_dir / "training.log")
        step_rows = [json.loads(line) for line in (run_dir / "step_timing.jsonl").read_text().splitlines() if line.strip()]
        step0 = step_rows[0]["step_wall_seconds"]
        steady = [r["step_wall_seconds"] for r in step_rows[11:]]
        milestones = []
        for c in ce["checkpoints"]:
            st = c["optimizer_step"]
            # loss printed at zero-based step index st-1 (the step that produced this checkpoint)
            log_row = steps.get(st - 1, {})
            milestones.append({
                "token_count": c["token_count"],
                "optimizer_step": st,
                "validation_bpb": c["validation_bpb"],
                "train_loss_ema_at_step": log_row.get("loss_ema"),
                "lrm_at_step": log_row.get("lrm"),
                "cumulative_steady_state_seconds": c["cumulative_steady_state_seconds"],
            })
        out.append({
            "run": run_dir.name,
            "model_id": run_dir.name.split("-")[1],
            "seed": tm["training_seed"],
            "parameter_count": tm["parameter_count"],
            "tokens_per_optimizer_step": tm["tokens_per_optimizer_step"],
            "optimizer_steps": tm["optimizer_steps"],
            "flops_per_token_code_estimate": flops,
            "model_config_logged": cfg,
            "steady_state_tps": tm["steady_state_training_tokens_per_second"],
            "steady_state_equivalent_seconds": tm["steady_state_equivalent_seconds"],
            "total_scientific_step_wall_seconds": tm["total_scientific_step_wall_seconds"],
            "end_to_end_runtime_seconds": tm.get("end_to_end_runtime_seconds"),
            "step0_seconds_compile_included": step0,
            "median_steady_step_seconds": statistics.median(steady),
            "peak_training_cuda_memory_mb": tm["peak_training_cuda_memory_mb"],
            "val_bpb_final_trainscript": tm["val_bpb"],
            "heldout_bpb": ce["heldout_bpb"]["metric_value"],
            "heldout_ci": [ce["heldout_bpb"]["confidence_low"], ce["heldout_bpb"]["confidence_high"]],
            "heldout_dataset_config": ce["heldout_bpb"]["dataset_config"],
            "heldout_tokens": ce["heldout_bpb"]["raw"]["evaluated_tokens"],
            "max_epoch_logged": max_epoch,
            "milestones": milestones,
            "log_tail": tail,
        })
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
