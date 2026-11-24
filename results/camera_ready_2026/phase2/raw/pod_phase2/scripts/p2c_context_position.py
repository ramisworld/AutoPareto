#!/usr/bin/env python3
"""Phase 2C (2026-11-07, exploratory, post hoc): does sliding-window attention cost language quality at long range?

For each checkpoint, score the frozen suite's exact held-out ClimbMix segments (shard_06541, 2,560 x 2,048 tokens,
built by downstream_evaluator.heldout_segments) with the frozen ModelScorer, and split the byte-weighted loss by the
target token's position in the 2,048-token window. Sliding windows of 256 tokens can only matter for positions
>= 256, so a long-range deficit shows as a smaller loss drop from early to late positions than in full-context models.
Per-segment, per-bucket nats and bytes are saved, so paired bootstrap intervals can be computed off-pod.

Also records the BLiMP paradigm -> (linguistics_term, field) map from the suite's pinned dataset revision.
Usage (repo root, Env A), one process per model (the evaluator warns against reloading candidates in-process):
  uv run --locked python p2c_context_position.py OUT_DIR MODEL_NAME | blimp_map
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/autopareto-camera-ready")
sys.path.insert(0, str(ROOT))
import downstream_evaluator as de  # noqa: E402

BUCKETS = [(0, 128), (128, 256), (256, 512), (512, 1024), (1024, 2048)]
C = Path("/var/lib/autopareto/pair_a_claude_opus55_t_seed3_continuation_20261105c/attempts")
IN = ROOT / "results/camera_ready_2026/phase2/inputs"
E0 = ROOT / "results/camera_ready_2026/amendment1/E0/runs/e0-arref-time300s-seed0"
AP = ROOT / "results/candidates/autopareto-seed0-20260731/0022-train.py"
AR = ROOT / "results/candidates/autoresearch-diagnostic5-seed42-20260724/0005-train.py"
MODELS = [
    ("opus55_t_attempt23", IN / "sources/opus_attempt23.train.py", next(C.glob("claude-opus55-position-046-*/position-046-*.checkpoint.pt"))),
    ("opus55_t_attempt24", IN / "sources/opus_attempt24.train.py", next(C.glob("claude-opus55-position-047-*/position-047-*.checkpoint.pt"))),
    ("opus55_t_attempt27", IN / "sources/opus_attempt27.train.py", next(C.glob("claude-opus55-position-052-*/position-052-*.checkpoint.pt"))),
    ("pair_a_baseline_t", ROOT / "study/contracts_2026/historical_reference/baseline_source/train.py",
     Path("/var/lib/autopareto/pair_a_claude_opus55_baseline_attempt2/nonstudy_tests/t_smoke_artifacts/result.checkpoint.pt")),
    ("autoresearch_compatible_time300s_seed0", E0 / "isolate/train.py", E0 / "final.pt"),
    ("autopareto_292_tokens201M_seed3", AP, IN / "phase1_checkpoints/phase1-autopareto_292-seed3/final.pt"),
    ("autopareto_292_tokens201M_seed4", AP, IN / "phase1_checkpoints/phase1-autopareto_292-seed4/final.pt"),
    ("autopareto_292_tokens201M_seed5", AP, IN / "phase1_checkpoints/phase1-autopareto_292-seed5/final.pt"),
    ("autoresearch_compatible_tokens201M_seed3", AR, IN / "phase1_checkpoints/phase1-autoresearch_compatible-seed3/final.pt"),
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    out_dir, which = Path(sys.argv[1]), sys.argv[2]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{which}.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    suite = de.load_suite()
    if which == "blimp_map":
        import datasets
        task = next(t for t in suite["tasks"] if t["name"] == "blimp")
        mapping = {}
        for config in sorted(datasets.get_dataset_config_names(task["dataset"], revision=task["dataset_revision"])):
            row = datasets.load_dataset(task["dataset"], config, revision=task["dataset_revision"], split=task["split"])[0]
            mapping[config] = {"linguistics_term": row.get("linguistics_term"), "field": row.get("field"), "UID": row.get("UID")}
        out.write_text(json.dumps({"dataset_revision": task["dataset_revision"], "paradigms": mapping}, indent=1) + "\n")
        print("BLiMP map:", len(mapping), "paradigms")
        return 0
    name, candidate, checkpoint = next(m for m in MODELS if m[0] == which)
    spec = suite["heldout_bpb"]
    shard = de.download_heldout_shard(spec)
    started = time.time()
    scorer = de.ModelScorer(candidate=candidate, checkpoint=checkpoint,
                            max_length=int(suite["runtime"]["max_sequence_tokens"]),
                            batch_size=int(suite["runtime"]["scoring_batch_size"]))
    segments = de.heldout_segments(scorer, shard, token_count=int(spec["evaluated_tokens"]),
                                   sequence_length=scorer.max_length)
    torch = scorer.torch
    from prepare import get_token_bytes
    token_bytes = get_token_bytes(device=scorer.device)
    nats = [[0.0] * len(segments) for _ in BUCKETS]
    nbytes = [[0.0] * len(segments) for _ in BUCKETS]
    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    with torch.inference_mode(), autocast:
        for offset in range(0, len(segments), scorer.batch_size):
            chunk = segments[offset:offset + scorer.batch_size]
            tensor = torch.tensor(chunk, dtype=torch.long, device=scorer.device)
            inputs, targets = de.contiguous_causal_batch(tensor)
            losses = scorer.model(inputs, targets, reduction="none").reshape(targets.shape).float()
            sizes = token_bytes[targets].float()
            masked = losses * (sizes > 0)
            for b, (lo, hi) in enumerate(BUCKETS):
                bn = masked[:, lo:hi].sum(dim=1).tolist()
                bb = sizes[:, lo:hi].sum(dim=1).tolist()
                for row in range(len(chunk)):
                    nats[b][offset + row] = bn[row]
                    nbytes[b][offset + row] = bb[row]
    total_bpb = sum(map(sum, nats)) / (math.log(2) * sum(map(sum, nbytes)))
    segment_hash = hashlib.sha256(json.dumps(segments).encode()).hexdigest()
    report = {"format": "autopareto_phase2_context_position_v1", "model": name, "buckets": BUCKETS,
              "suite_sha256": sha(de.DEFAULT_SUITE), "evaluator_sha256": sha(Path(de.__file__)),
              "shard_sha256": sha(shard), "segments": len(segments), "segments_sha256": segment_hash,
              "candidate": str(candidate), "candidate_sha256": sha(candidate), "checkpoint": str(checkpoint),
              "checkpoint_sha256": sha(checkpoint), "overall_bpb": total_bpb,
              "bucket_bpb": [sum(nats[b]) / (math.log(2) * sum(nbytes[b])) for b in range(len(BUCKETS))],
              "per_segment_nats": nats, "per_segment_bytes": nbytes, "seconds": time.time() - started}
    out.write_text(json.dumps(report) + "\n")
    print(name, f"overall {total_bpb:.5f}", [round(x, 4) for x in report["bucket_bpb"]], f"{report['seconds']:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
