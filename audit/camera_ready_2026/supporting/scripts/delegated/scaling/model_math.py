#!/usr/bin/env python3
"""Pure-arithmetic replica of the frozen train.py sizing/FLOP formulas (no torch, no repo imports).

Mirrors results/candidates/baseline-a40-seed0/0002-train.py:
  build_model_config (L469-477), has_ve (L47-49), MLP 4x (L102-103),
  _compute_window_sizes (L195-206), estimate_flops (L208-222), num_scaling_params (L224-234),
  plus the KV_HEAD_RATIO rule of tpe-seed42-20260805/0021-train.py (L476-480).
"""
from __future__ import annotations

VOCAB = 8192
SEQ = 2048


def config(depth, aspect, head_dim=128, kv_ratio=None, window="SSSL", mlp_mult=4):
    base = depth * aspect
    width = ((base + head_dim - 1) // head_dim) * head_dim
    heads = width // head_dim
    if kv_ratio is None:
        kv = heads
    else:
        target = max(1, round(heads / kv_ratio))
        kv = max(c for c in range(1, heads + 1) if heads % c == 0 and c <= target)
    return dict(depth=depth, aspect=aspect, width=width, heads=heads, kv=kv, head_dim=head_dim,
                window=window, mlp_mult=mlp_mult, eff_aspect=width / depth)


def has_ve(i, n):
    return i % 2 == (n - 1) % 2


def windows(cfg):
    pat = cfg["window"].upper()
    out = []
    for i in range(cfg["depth"]):
        out.append(SEQ if pat[i % len(pat)] == "L" else SEQ // 2)
    out[-1] = SEQ
    return out


def params(cfg):
    d, w, hd, kvh, n = cfg["depth"], cfg["width"], cfg["head_dim"], cfg["kv"], cfg["depth"]
    kv_dim = kvh * hd
    m = cfg["mlp_mult"]
    wte = VOCAB * w
    lm_head = w * VOCAB
    n_ve = sum(1 for i in range(n) if has_ve(i, n))
    ve = n_ve * VOCAB * kv_dim
    per_layer = w * w + w * kv_dim * 2 + w * w + 2 * w * (m * w)
    gates = n_ve * 32 * kvh
    matrices = d * per_layer + gates
    scalars = 2 * d
    total = wte + lm_head + ve + matrices + scalars
    return dict(wte=wte, lm_head=lm_head, ve=ve, matrices=matrices, scalars=scalars, total=total,
                non_embedding=total - wte - ve - scalars)


def flops_per_token(cfg):
    p = params(cfg)
    attn = sum(12 * cfg["heads"] * cfg["head_dim"] * min(win, SEQ) for win in windows(cfg))
    return 6 * p["non_embedding"] + attn


RECIPES = {
    # name: (depth, aspect, head_dim, kv_ratio, window)
    "baseline": (8, 64, 128, None, "SSSL"),
    "autoresearch_compatible": (8, 64, 128, None, "SSSS"),
    "tpe_quality": (6, 80, 128, 1, "SSSS"),
    "motpe_quality": (6, 48, 128, 1, "L"),
    "autopareto_292": (4, 128, 128, None, "SSSL"),
    "autopareto_299": (3, 160, 128, None, "SSSL"),
    "autopareto_294": (3, 128, 128, None, "SSSL"),
}

if __name__ == "__main__":
    import json
    rows = {}
    for name, (d, a, hd, kr, win) in RECIPES.items():
        c = config(d, a, hd, kr, win)
        rows[name] = dict(cfg=c, params=params(c), flops_per_token=flops_per_token(c))
    print(json.dumps(rows, indent=1))
