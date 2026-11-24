#!/usr/bin/env python3
"""DERIVED cost/feasibility tables for the scaling audit (pure arithmetic, read-only inputs).

Inputs (all produced by sibling scripts in this folder from repo records):
  phase1_runs.json   -> measured steady-state tokens/s per recipe (201M-token panel)
Constants (cited in the report):
  A40 rate $0.44/h (PROJECT_STATE.md L960); A40 memory 46,068 MiB (a40_environment.json);
  memory fit M_MiB = 2194 + 5.0145*d*w at DEVICE_BATCH_SIZE=64 (MHA, HEAD_DIM=128; n=25 runs);
  code-FLOP throughput 70e12/s central (observed 55-80e12 across 127 runs).
"""
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import model_math as mm  # noqa: E402

RATE = 0.44
MEM_LIMIT = 46068
MEM_TARGET = 40000
A_MEM, B_MEM = 2194.0, 5.0145
TFLOPS_CENTRAL, TFLOPS_LO, TFLOPS_HI = 70e12, 60e12, 80e12
OVERHEAD_S = 300  # compile (~90-160 s) + final eval/save (~75-112 s), from phase1 end_to_end - sci_wall + step0
D1 = 201_326_592
C_REF = 2.1e16  # ~300 A40-seconds of code-FLOPs (discovery budget); see report section C
TOK_PER_SHARD = 62e6  # INFERRED (~4.0 bytes/token, ~250 MB text/shard); lower bound 28.7M observed

names = {"autoresearch": "autoresearch_compatible", "tpe": "tpe_quality", "motpe": "motpe_quality"}
p1 = json.load(open(HERE / "phase1_runs.json"))
tps = {}
for r in p1:
    k = names.get(r["model_id"], r["model_id"])
    tps.setdefault(k, []).append(r["steady_state_tps"])
tps = {k: statistics.mean(v) for k, v in tps.items()}


def mem_mib(d, w, n_params, dbs):
    return 16 * n_params / 2**20 + (dbs / 64) * (A_MEM + B_MEM * d * w)


def pick_dbs(d, w, n_params, tbs):
    for dbs in (64, 32, 16, 8, 4):
        if tbs % (dbs * 2048) == 0 and mem_mib(d, w, n_params, dbs) <= MEM_TARGET:
            return dbs, mem_mib(d, w, n_params, dbs)
    return None, None


def table_i():
    print("== (i) fixed model, 4x/16x/64x of 201,326,592 tokens; measured phase1 tok/s; $0.44/h")
    tot = {4: 0.0, 16: 0.0, 64: 0.0}
    for k in mm.RECIPES:
        row = [k, round(tps[k])]
        for m in (4, 16, 64):
            D = m * D1
            s = D / tps[k] + OVERHEAD_S
            h = s / 3600
            tot[m] += 3 * h
            row += [f"{m}x: D={D/1e9:.3f}B t={h:.2f}h ${h*RATE:.2f} shards~{math.ceil(D/TOK_PER_SHARD)}"]
        print(" | ".join(str(x) for x in row))
    for m in tot:
        print(f"  all 7 recipes x 3 seeds at {m}x: {tot[m]:.1f} GPU-h  ${tot[m]*RATE:.0f}")


FAMILIES = {
    # family: (aspect, head_dim, kv_ratio, window, recipe whose tbs/consts apply, tbs)
    "baseline(AR64,SSSL,TBS2^19)": (64, 128, None, "SSSL", 524288),
    "autoresearch(AR64,SSSS,TBS2^17)": (64, 128, None, "SSSS", 131072),
    "tpe(AR80,SSSS,TBS2^17)": (80, 128, 1, "SSSS", 131072),
    "motpe(AR48,L,TBS2^17)": (48, 128, 1, "L", 131072),
    "ap292/ap294(AR128,SSSL,TBS2^17)": (128, 128, None, "SSSL", 131072),
    "ap299(AR160,SSSL,TBS2^17)": (160, 128, None, "SSSL", 131072),
}


def best_depth(aspect, hd, kr, win, C, ratio=20.0):
    best = None
    for d in range(2, 41):
        c = mm.config(d, aspect, hd, kr, win)
        p = mm.params(c)
        fpt = mm.flops_per_token(c)
        D = C / fpt
        r = D / p["non_embedding"]
        score = abs(math.log(r / ratio))
        if best is None or score < best[0]:
            best = (score, d, c, p, fpt, D, r)
    return best


def table_ii():
    print("== (ii) compute ladder C = k * C_REF (C_REF=2.1e16 code-FLOPs ~ 300 A40-s); depth-dial promotion; D/N_flop ~ 20")
    grand = 0.0
    for fam, (a, hd, kr, win, tbs) in FAMILIES.items():
        for k in (1, 4, 16, 64):
            C = k * C_REF
            _, d, c, p, fpt, D, r = best_depth(a, hd, kr, win, C)
            steps = D / tbs
            dbs, mem = pick_dbs(d, c["width"], p["total"], tbs)
            mem64 = mem_mib(d, c["width"], p["total"], 64)
            h_c = C / TFLOPS_CENTRAL / 3600
            h_lo = C / TFLOPS_HI / 3600
            h_hi = C / TFLOPS_LO / 3600
            if k > 1:
                grand += 3 * (h_c + OVERHEAD_S / 3600)
            print(f"{fam:34s} {k:>2}x C={C:.2e} d={d:2d} w={c['width']:4d} effAR={c['eff_aspect']:.0f} "
                  f"N_total={p['total']/1e6:6.1f}M N_flop={p['non_embedding']/1e6:6.1f}M fpt={fpt:.3e} "
                  f"D={D/1e9:5.2f}B D/Nflop={r:4.1f} steps={steps:,.0f} mem@64={mem64/1024:5.1f}GiB "
                  f"dbs={dbs} mem={mem/1024 if mem else float('nan'):4.1f}GiB "
                  f"t={h_c:5.2f}h [{h_lo:.2f}-{h_hi:.2f}] ${h_c*RATE:.2f} shards~{math.ceil(D/TOK_PER_SHARD)}")
    print(f"  6 families x 3 seeds x (4x,16x,64x): {grand:.0f} GPU-h ${grand*RATE:.0f}")


def table_ii_width_dial():
    print("== (ii-b) AP292 width-dial (DEPTH fixed at 4, ASPECT_RATIO raised in steps of 32)")
    for k in (4, 16, 64):
        C = k * C_REF
        best = None
        for a in range(128, 1025, 32):
            c = mm.config(4, a, 128, None, "SSSL")
            p = mm.params(c)
            fpt = mm.flops_per_token(c)
            D = C / fpt
            r = D / p["non_embedding"]
            s = abs(math.log(r / 20))
            if best is None or s < best[0]:
                best = (s, a, c, p, fpt, D, r)
        _, a, c, p, fpt, D, r = best
        dbs, mem = pick_dbs(4, c["width"], p["total"], 131072)
        print(f"  {k:>2}x d=4 AR={a} w={c['width']} N_total={p['total']/1e6:.1f}M N_flop={p['non_embedding']/1e6:.1f}M "
              f"D={D/1e9:.2f}B D/Nflop={r:.1f} dbs={dbs} mem={mem/1024:.1f}GiB t={C/TFLOPS_CENTRAL/3600:.2f}h")


if __name__ == "__main__":
    table_i()
    table_ii()
    table_ii_width_dial()
