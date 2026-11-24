#!/usr/bin/env python3
"""Read-only tabulation of Pair-A history.jsonl attempts (both arms)."""
import json
import re
import sys
from pathlib import Path

RUN = Path("/Users/Rami/Documents/dev/autopareto/runs/pair-a-official-20260914-new-a40-continuation-20260916-r2")

KEYS = ("DEPTH", "ASPECT_RATIO", "HEAD_DIM", "WINDOW_PATTERN", "TOTAL_BATCH_SIZE", "DEVICE_BATCH_SIZE",
        "MATRIX_LR", "EMBEDDING_LR", "UNEMBEDDING_LR", "SCALAR_LR", "WEIGHT_DECAY", "WARMUP_RATIO",
        "WARMDOWN_RATIO", "FINAL_LR_FRAC")


def patch_constants(patch: str):
    """Return {name: (before, after)} for constant assignment lines changed in a unified diff."""
    before, after = {}, {}
    for line in patch.splitlines():
        m = re.match(r"^([+-])([A-Z_]+)\s*=\s*([^#\n]+)", line)
        if m and m.group(2) in KEYS:
            (before if m.group(1) == "-" else after)[m.group(2)] = m.group(3).strip()
    return {k: (before.get(k), after.get(k)) for k in sorted(set(before) | set(after))}


def main():
    out = []
    for arm in ("pair-a-fixed-time-seed3", "pair-a-fixed-tokens-seed3"):
        for i, line in enumerate((RUN / arm / "history.jsonl").read_text().splitlines(), start=1):
            if not line.strip():
                continue
            r = json.loads(line)
            res = r.get("result") or {}
            pd = r.get("pareto_decision") or res.get("pareto_decision") or {}
            out.append({
                "arm": arm,
                "line": i,
                "global_position": r.get("global_position"),
                "campaign_attempt": r.get("campaign_attempt"),
                "contract": r.get("contract"),
                "status": r.get("status"),
                "change_category": r.get("change_category"),
                "intended_change": (r.get("intended_change") or "")[:400],
                "parent": r.get("parent_experiment_id"),
                "decision": pd.get("decision"),
                "quality_status": pd.get("quality_status"),
                "num_params": res.get("num_params"),
                "total_tokens": res.get("total_tokens"),
                "total_batch_size": res.get("total_batch_size"),
                "device_batch_size": res.get("device_batch_size"),
                "accumulation_steps": res.get("accumulation_steps"),
                "optimizer_updates": res.get("optimizer_updates"),
                "tps": res.get("training_throughput_tokens_per_second"),
                "flops_estimate": res.get("training_flops_estimate"),
                "measured_training_seconds": res.get("measured_training_seconds"),
                "val_bpb": res.get("validation_bpb"),
                "normalized_decode": (res.get("inference_result") or {}).get("normalized_decode_throughput"),
                "constants_changed": patch_constants(r.get("candidate_patch") or ""),
                "experiment_id": res.get("experiment_id"),
            })
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
