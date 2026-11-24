#!/usr/bin/env python3
"""DERIVED: decompose Pair-A fixed-time parent->child BPB deltas into a token-exposure part and a residual.

Assumption (stated, not measured within Pair-A): at fixed model, val BPB changes by s * ln(D_child/D_parent),
with s taken from this repo's discovery->201M same-recipe comparisons (131K-batch recipes: s ~ -0.062..-0.082).
Single-seed deltas; 5-min seed SD is ~0.002-0.008 BPB (see report). Read-only.
"""
import json, math, sys
rows = json.load(open(sys.argv[1]))
T = {r["experiment_id"]: r for r in rows if r["arm"].endswith("time-seed3") and r["experiment_id"]}
base = {"val_bpb": 1.1998857791422985, "total_tokens": None, "num_params": 50332176}
hist = [json.loads(l) for l in open(sys.argv[2])]
for h in hist:
    res = h.get("result") or {}
    pid = h.get("parent_experiment_id")
    parent = T.get(pid)
    if parent is None:
        continue  # first attempt (parent = registered baseline; baseline token count not in this file)
    d_bpb = res["validation_bpb"] - parent["val_bpb"]
    dl = math.log(res["total_tokens"] / parent["total_tokens"])
    out = [h["campaign_attempt"], h["change_category"][:44], parent["num_params"], res["num_params"],
           parent["total_tokens"], res["total_tokens"], round(d_bpb, 4), round(dl, 3)]
    for s in (-0.062, -0.07, -0.082):
        out.append(round(d_bpb - s * dl, 4))
    print(*out, sep=" | ")
