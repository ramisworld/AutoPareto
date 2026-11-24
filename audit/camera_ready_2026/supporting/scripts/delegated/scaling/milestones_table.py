#!/usr/bin/env python3
"""DERIVED: milestone means/SD per recipe, LR multiplier at milestone, gaps vs AR, and 5-min vs 201M comparisons."""
import json, statistics, sys, sqlite3, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import model_math as mm
p1 = json.load(open(HERE / "phase1_runs.json"))
names = {"autoresearch": "autoresearch_compatible", "tpe": "tpe_quality", "motpe": "motpe_quality"}
by = {}
for r in p1:
    by.setdefault(names.get(r["model_id"], r["model_id"]), []).append(r)
order = ["baseline", "autoresearch_compatible", "tpe_quality", "motpe_quality", "autopareto_292", "autopareto_299", "autopareto_294"]
ms = {}
print("recipe | milestone tokens | mean val BPB | SD | lrm | train-loss EMA mean")
for k in order:
    rs = by[k]
    for i in range(4):
        vals = [r["milestones"][i]["validation_bpb"] for r in rs]
        lrm = sorted(set(r["milestones"][i]["lrm_at_step"] for r in rs))
        tl = statistics.mean(r["milestones"][i]["train_loss_ema_at_step"] for r in rs)
        tok = rs[0]["milestones"][i]["token_count"]
        ms[(k, i)] = statistics.mean(vals)
        print(f"{k} | {tok:,} | {statistics.mean(vals):.4f} | {statistics.stdev(vals):.4f} | {lrm} | {tl:.4f}")
print("\nGap (recipe - AR) val BPB by milestone (25/50/75/100%):")
for k in order:
    print(k, [round(ms[(k, i)] - ms[("autoresearch_compatible", i)], 4) for i in range(4)])
print("\nRank order by milestone:")
for i in range(4):
    print(i, [k for k in sorted(order, key=lambda k: ms[(k, i)])])
# heldout means and val-heldout offset
print("\nfinal val vs heldout means:")
for k in order:
    v = statistics.mean(r["val_bpb_final_trainscript"] for r in by[k]); h = statistics.mean(r["heldout_bpb"] for r in by[k])
    print(k, round(v, 5), round(h, 5), "offset", round(h - v, 4))
# 5-minute comparisons (val BPB from results.db, read-only immutable)
con = sqlite3.connect("file:/Users/Rami/Documents/dev/autopareto/results/results.db?mode=ro&immutable=1", uri=True)
def vals(ids):
    q = "select id,val_bpb,total_training_tokens,training_seed,valid_result from experiments where id in (%s)" % ",".join(map(str, ids))
    return con.execute(q).fetchall()
five = {
    "baseline": [3, 4, 5, 6, 7],
    "autoresearch_compatible": [68, 75, 77],
    "tpe_quality": [183],
    "motpe_quality": [234, 300],
    "autopareto_292": [291, 292, 293],
    "autopareto_299": [297, 298, 299],
    "autopareto_294": [294, 295, 296],
}
print("\n5-min (complete schedule) vs 201M (complete schedule), val BPB, code-FLOPs:")
for k in order:
    rows = vals(five[k])
    v5 = statistics.mean(r[1] for r in rows); d5 = statistics.mean(r[2] for r in rows)
    sd5 = statistics.stdev([r[1] for r in rows]) if len(rows) > 1 else float("nan")
    d, a, hd, kr, w = mm.RECIPES[k]; fpt = mm.flops_per_token(mm.config(d, a, hd, kr, w))
    v201 = ms[(k, 3)]
    c5, c201 = fpt * d5, fpt * 201326592
    slope = (v201 - v5) / math.log(c201 / c5)
    print(f"{k} | 5min n={len(rows)} seeds={[r[3] for r in rows]} D5={d5/1e6:.1f}M val={v5:.4f} sd={sd5:.4f} C5={c5:.3e} | 201M val={v201:.4f} C201={c201:.3e} | tokens x{201326592/d5:.3f} dBPB={v201-v5:+.4f} slope/efold={slope:+.4f}")
