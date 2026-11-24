#!/usr/bin/env python3
"""Camera-ready revision 4 (2026-11-14): reader-first figures and tables.

Reads only paper/camera_ready/data/cr_v2.json (written by cr_v2.py from committed analysis outputs),
the read-only Pair-A attempt records (parents and verbatim hypotheses) and the historical lineage
records already used by cr_figures.py.  Launches nothing; no GPU; no network.

Writes
  paper/figures/cr_opus_lineage.{pdf,png}   Opus decision lineage (recorded parent links only)
  paper/figures/cr_inference.{pdf,png}      where the decode speed-up comes from, and what is timed
  paper/figures/cr_e1_main.{pdf,png}        E1 token response and exploratory 300 s split
  paper/generated/camera_ready/table_results.tex    compact main-results table (held-out BPB)
  paper/generated/camera_ready/table_apq_path.tex   AP-Q path, one row per step (replaces cr_decision_graph)
  paper/generated/camera_ready/table_funnel_v3.tex  selection funnel, chronological
  paper/generated/camera_ready/cr_macros_v3.tex     numbers quoted in the text
"""
from __future__ import annotations

import json
import os
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / 'paper/camera_ready/.mplcache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from cr_common import CR_DATA, GEN_CR, read_json
from cr_v2 import (AQUA, BLUE, CELLS, GRAY, GRID, IA_MODELS, INK, INK2, MUTED, OPUS_ROOT, ORANGE, VIOLET,
                   _ancestry, _pair_a_parents, draw_e1, draw_split, save, sgn, style)

LILAC = '#b9b0ee'      # inference-only edits (training byte-identical to the parent)
PHASE = ('#f4f2fb', '#f7f7f5', '#f4f2fb')


def hyp_records():
    out = {}
    for line in (OPUS_ROOT / 'history.jsonl').read_text().splitlines():
        d = json.loads(line)
        out[d['campaign_attempt']] = d
    return out


def verbatim(H, a, text):
    blob = ' '.join(' '.join(str(H[a].get(k) or '') for k in ('hypothesis', 'intended_change')).split())
    assert ' '.join(text.split()) in blob, f'A{a}: quote not found verbatim: {text!r}'
    return text


# ----------------------------------------------------------------------------- Opus lineage
INFERENCE_PREFIX = 'inference'


def fig_opus_lineage(D):
    """Figure 1.  Top: validation BPB of every Opus attempt with its recorded parent link.
    Bottom: decode-only (scored) and end-to-end (recorded) speed-up on the path to the best recipe."""
    style(1.0)
    H = hyp_records()
    par = _pair_a_parents(OPUS_ROOT)
    rows = {r['attempt']: r for r in D['agents']['opus']['rows']}
    base = D['agents']['baseline']
    valid = {a: r for a, r in rows.items() if r['valid']}
    best = min(valid.values(), key=lambda r: r['val_bpb'])['attempt']
    path = _ancestry(par, best)
    assert path == [1, 3, 4, 5, 7, 8, 10, 11, 13, 14, 17, 18, 19, 20, 21, 22, 23], path
    frontier = [int(a) for a in ('23', '24', '27')]
    assert D['agents']['opus']['frontier_extended'] == frontier
    failed = [a for a, r in rows.items() if not r['valid']]
    assert failed == [2, 9], failed
    # every drawn edge is a recorded parent link
    edges = [(par[a] if par[a] is not None else 0, a) for a in sorted(rows)]
    bpb = lambda a: base['val_bpb'] if a == 0 else rows[a]['val_bpb']
    inf_only = {a for a, r in rows.items() if (r['category'] or '').startswith(INFERENCE_PREFIX)}
    # an inference-only edit leaves training byte-identical, so its BPB equals its parent's up to run noise
    for a in inf_only:
        assert abs(rows[a]['val_bpb'] - bpb(par[a])) < 0.0015, (a, rows[a]['val_bpb'], bpb(par[a]))

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(5.5, 2.7), sharex=True,
                                 gridspec_kw={'height_ratios': [1.85, 1.0], 'hspace': 0.08})
    for a_ in (ax, bx):
        for (x0, x1), c in zip(((0.5, 7.5), (7.5, 22.5), (22.5, 27.5)), PHASE):
            a_.axvspan(x0, x1, color=c, lw=0, zorder=0)
        a_.axvline(25.5, color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
        a_.grid(axis='y', color=GRID, lw=0.5, zorder=0)
    fs = 6.3
    ytop = 1.209
    for x, t in ((4.0, 'A  Batch and depth'), (15.0, 'B  Mostly inference engineering'), (25.0, 'C  Model edits')):
        ax.text(x, 1.004, t, ha='center', va='bottom', fontsize=6.4, color=INK, fontweight='bold',
                transform=ax.get_xaxis_transform())
    ax.text(25.62, 1.1885, 'planned\n25 attempts\n| extension', fontsize=5.3, color=INK2, va='center', ha='left')

    # edges: recorded parent -> child
    for p, a in edges:
        if a in failed:
            continue
        on = a in path
        ax.plot([p, a], [bpb(p), bpb(a)], color=VIOLET if on else MUTED, lw=1.3 if on else 0.7,
                ls='-' if on else (0, (2.2, 1.4)), zorder=2, solid_capstyle='round')
    # nodes
    ax.scatter([0], [base['val_bpb']], s=26, color=INK, zorder=5)
    for a, r in valid.items():
        sq = a in inf_only
        on = a in path or a in frontier
        face = (LILAC if sq else VIOLET) if on else 'white'
        ax.scatter([a], [r['val_bpb']], s=22 if not sq else 18, marker='s' if sq else 'o', facecolor=face,
                   edgecolor=VIOLET, lw=0.8, zorder=4)
        if a in frontier:
            ax.scatter([a], [r['val_bpb']], s=70, marker='o', facecolor='none', edgecolor=INK, lw=0.9, zorder=5)
    for a in failed:
        ax.scatter([a], [1.0815], marker='x', s=16, color=GRAY, lw=1.0, zorder=4)

    def call(text, xy, xyt, ha='left', col=INK, va='center'):
        ax.annotate(text, xy, xytext=xyt, textcoords='data', fontsize=fs, ha=ha, va=va, color=col, linespacing=1.1,
                    arrowprops={'arrowstyle': '-', 'lw': 0.5, 'color': INK2, 'shrinkA': 1.5, 'shrinkB': 2.5}, zorder=6)

    r = rows
    ax.text(0.45, 1.2035, f"B  baseline: 8 layers, {r[1]['params']/1e6:.0f}M parameters", fontsize=fs, va='center', color=INK)
    call(f"A1  batch halved: {r[1]['updates']} updates instead of 163", (1, r[1]['val_bpb']), (1.9, 1.1915))
    call(f"A3–A5  depth 8→6→4→3: {r[1]['params']/1e6:.0f}M→{r[5]['params']/1e6:.0f}M parameters,\n"
         f"so {r[1]['tokens']/1e6:.0f}M→{r[5]['tokens']/1e6:.0f}M tokens fit in 300 s", (4, r[4]['val_bpb']), (8.3, 1.174))
    call(f"A6  depth 2: worse (+{r[6]['val_bpb'] - r[5]['val_bpb']:.3f})", (6, r[6]['val_bpb']), (8.3, 1.156))
    call(f"A7  batch halved again: {r[7]['updates']:,} updates", (7, r[7]['val_bpb']), (8.3, 1.1415))
    call(f"A12  a third halving costs tokens\n({r[12]['tokens']/1e6:.0f}M vs {r[11]['tokens']/1e6:.0f}M): worse",
         (12, r[12]['val_bpb']), (12.8, 1.127))
    call(f"A25  attention gate:\n{r[25]['tokens']/1e6:.0f}M tokens, worse",
         (25, r[25]['val_bpb']), (24.6, 1.153), ha='right')
    gain = r[7]['val_bpb'] - r[best]['val_bpb']
    nlate = sum(1 for a in valid if a > 7)
    ax.annotate('', xy=(27.4, 1.0912), xytext=(7.6, 1.0912),
                arrowprops={'arrowstyle': '|-|', 'lw': 0.6, 'color': INK2, 'mutation_scale': 2.2})
    ax.text(17.5, 1.0893, f'after A7: {nlate} more valid attempts improve BPB by only {gain:.4f} in total',
            fontsize=5.9, color=INK2, ha='center', va='top')
    call("ringed: frontier A23, A24, A27", (24, r[24]['val_bpb']), (22.9, 1.1135), ha='right')
    ax.set_ylim(1.078, ytop)
    ax.set_ylabel('Validation BPB\n(lower is better)')
    hand = [Line2D([], [], ls='none', marker='o', ms=4.2, mfc=VIOLET, mec=VIOLET, label='edit changes training'),
            Line2D([], [], ls='none', marker='s', ms=3.8, mfc=LILAC, mec=VIOLET, label='inference edit (agent label)'),
            Line2D([], [], ls='none', marker='o', ms=4.2, mfc='white', mec=VIOLET, label='branch off the best path'),
            Line2D([], [], color=VIOLET, lw=1.3, label='recorded parent path to A23'),
            Line2D([], [], ls='none', marker='x', ms=3.6, mec=GRAY, mew=1.0, label='failed attempt (counted)')]
    ax.legend(handles=hand, loc='lower center', bbox_to_anchor=(0.5, 1.075), ncol=5, frameon=False, fontsize=5.8,
              handletextpad=0.25, borderaxespad=0.0, columnspacing=0.9, handlelength=1.3)

    # bottom: speed on the path (and frontier branch points)
    pts = [a for a in path] + [a for a in frontier if a not in path]
    unc = {a for a, rr in rows.items() if rr['quality_status'] == 'quality_valid_speed_uncertain'}
    xs = [0] + path
    dec = [1.0] + [rows[a]['decode_x'] for a in path]
    e2e = [1.0] + [rows[a]['e2e_x'] for a in path]
    bx.plot(xs, dec, color=VIOLET, lw=1.3, zorder=3)
    bx.plot(xs, e2e, color=VIOLET, lw=1.0, ls=(0, (3, 1.6)), zorder=3)
    for a in pts:
        al = 0.45 if a in unc else 1.0
        bx.scatter([a], [rows[a]['decode_x']], s=14, color=VIOLET, zorder=4, alpha=al)
        bx.scatter([a], [rows[a]['e2e_x']], s=14, facecolor='white', edgecolor=VIOLET, lw=0.8, zorder=4, alpha=al)
    for a in frontier:
        if a not in path:
            p = par[a]
            bx.plot([p, a], [rows[p]['decode_x'], rows[a]['decode_x']], color=MUTED, lw=0.6, ls=(0, (2.2, 1.4)), zorder=2)
            bx.plot([p, a], [rows[p]['e2e_x'], rows[a]['e2e_x']], color=MUTED, lw=0.6, ls=(0, (2.2, 1.4)), zorder=2)
    bx.scatter([0], [1.0], s=20, color=INK, zorder=5)
    mech = [(10, 'CUDA graph\nreplay', 8.0, 17), (14, 'compiled,\nfused kernels', 12.2, 36), (18, 'ring K/V\ncache', 16.0, 55),
            (19, 'custom\nattention', 19.6, 55), (22, 'fuse remaining\nkernels', 23.6, 55)]
    for a, t, tx, ty in mech:
        bx.annotate(t, (a, rows[a]['decode_x']), xytext=(tx, ty), textcoords='data',
                    fontsize=5.7, ha='center', va='bottom', color=INK, linespacing=1.0,
                    arrowprops={'arrowstyle': '-', 'lw': 0.5, 'color': INK2, 'shrinkA': 0.5, 'shrinkB': 2.5})
    bx.text(27.6, rows[27]['decode_x'] * 0.72, 'decode only\n(scored)', fontsize=5.9, va='top', ha='right', color=INK, linespacing=1.0)
    bx.text(27.6, rows[27]['e2e_x'] * 0.72, 'end to end (recorded, not scored)', fontsize=5.9, va='top', ha='right', color=INK)
    bx.set_yscale('log')
    bx.set_yticks([1, 2, 5, 10, 30])
    bx.set_yticklabels(['1×', '2×', '5×', '10×', '30×'])
    bx.minorticks_off()
    bx.set_ylim(0.8, 190)
    bx.set_ylabel('Speed-up over\nbaseline (log)')
    bx.set_xlim(-0.6, 27.8)
    bx.set_xticks([0, 1, 3, 5, 7, 10, 14, 17, 20, 23, 25, 27])
    bx.set_xticklabels(['B', '1', '3', '5', '7', '10', '14', '17', '20', '23', '25', '27'])
    bx.set_xlabel('Attempt (Claude Opus 5.5, Pair-A, seed 3, 300 s contract)')
    fig.align_ylabels([ax, bx])
    save(fig, 'cr_opus_lineage')
    return {'path': path, 'late_valid': nlate, 'late_gain': gain, 'inference_only': sorted(inf_only),
            'speed_uncertain': sorted(unc)}


# ----------------------------------------------------------------------------- inference mechanism
VARIANTS = [('V0_as_submitted', 'As submitted (all mechanisms)'),
            ('V5_linear_kv_cache', 'Linear K/V cache instead of ring'),
            ('V3_sdpa_attention', 'Standard attention kernel (SDPA)'),
            ('V2_no_fusion', 'No kernel fusion'),
            ('V1_no_cuda_graph', 'No CUDA graph'),
            ('V4_eager_unfused', 'Neither graph nor fusion')]


def fig_inference(D):
    style(1.0)
    ia = D['phase5']['ablation']
    sp, cells, b = ia['speedup'], ia['cells'], ia['baseline']
    med = lambda v, k: statistics.median(sp[m][v][k] for m in IA_MODELS)
    rng = lambda v, k: (min(sp[m][v][k] for m in IA_MODELS), max(sp[m][v][k] for m in IA_MODELS))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.5, 1.3), gridspec_kw={'width_ratios': [1.1, 1.0], 'wspace': 0.42})
    n = len(VARIANTS) + 1
    labels = []
    for i, (v, lab) in enumerate(VARIANTS + [('baseline', 'Registered baseline (8 layers)')]):
        y = n - 1 - i
        labels.append((y, lab))
        vals = {k: (1.0 if v == 'baseline' else med(v, k)) for k in ('decode', 'e2e')}
        tops = {k: (1.0 if v == 'baseline' else max(vals[k], rng(v, k)[1])) for k in ('decode', 'e2e')}
        # when decode and end-to-end agree within 15% one label serves both bars (no stacked, touching labels)
        joint = vals['decode'] / vals['e2e'] < 1.15
        for k, dy, face in (('decode', 0.17, VIOLET), ('e2e', -0.17, 'white')):
            val = vals[k]
            if v != 'baseline':
                ax.barh(y + dy, val - 1, left=1, height=0.32, color=face, edgecolor=VIOLET, lw=0.7, zorder=3)
                lo, hi = rng(v, k)
                if hi / lo > 1.03:
                    ax.plot([lo, hi], [y + dy, y + dy], color=INK, lw=0.6, zorder=4)
            if not joint:
                txt = f'{val:.0f}×' if (k == 'decode' and val >= 10) else f'{val:.1f}×'
                ax.text(tops[k] * 1.09, y + dy, txt, va='center', ha='left', fontsize=5.8, color=INK)
        if joint:
            ax.text(max(tops.values()) * 1.09, y, f"{vals['decode']:.1f}× both", va='center', ha='left', fontsize=5.8, color=INK)
    ax.axvline(1, color=INK2, lw=0.6, zorder=2)
    ax.set_xscale('log')
    ax.set_xlim(0.9, 75)
    ax.set_xticks([1, 2, 5, 10, 30])
    ax.set_xticklabels(['1×', '2×', '5×', '10×', '30×'])
    ax.minorticks_off()
    ax.set_yticks([y for y, _ in labels])
    ax.set_yticklabels([l for _, l in labels], fontsize=6.1)
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', color=GRID, lw=0.5, zorder=0)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel('Speed-up over the baseline (log scale)')
    ax.set_title('(a) Remove one mechanism at a time', loc='left', fontsize=7.0)
    h = [Rectangle((0, 0), 1, 1, facecolor=VIOLET, edgecolor=VIOLET, label='decode only (scored)'),
         Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=VIOLET, label='end to end')]
    ax.legend(handles=h, loc='lower right', frameon=False, fontsize=5.8, handlelength=1.2, borderaxespad=0.2)

    # (b) time per generation: prefill (untimed by the score) + 127 cached decode steps (scored)
    def times(v):
        if v == 'baseline':
            return b['prefill_ms'], 127 / b['decode_tps'] * 1000, b['e2e_tps']
        pre = statistics.median(cells[f'{m}/{v}']['prefill_ms'] for m in IA_MODELS)
        dec = statistics.median(127 / cells[f'{m}/{v}']['decode_tps'] * 1000 for m in IA_MODELS)
        e2e = statistics.median(cells[f'{m}/{v}']['e2e_tps'] for m in IA_MODELS)
        return pre, dec, e2e
    # the recorded end-to-end rate matches (within 1.5%) 128 tokens / (prefill + 127 decode steps)
    for m in IA_MODELS:
        for v, _ in VARIANTS:
            c = cells[f'{m}/{v}']
            derived = 128 / (c['prefill_ms'] / 1000 + 127 / c['decode_tps'])
            assert abs(derived / c["e2e_tps"] - 1) < 0.015, (m, v, derived, c['e2e_tps'])
    rowsb = [('baseline', 'Baseline'), ('V1_no_cuda_graph', 'Opus,\nno graph'), ('V0_as_submitted', 'Opus,\nas submitted')]
    T = {v: times(v) for v, _ in rowsb}
    for i, (v, lab) in enumerate(rowsb):
        y = len(rowsb) - 1 - i
        pre, dec, _ = T[v]
        bx.barh(y, pre, left=0, height=0.5, color='#d9d7d1', edgecolor=INK2, lw=0.5, hatch='////', zorder=3)
        bx.barh(y, dec, left=pre, height=0.5, color=VIOLET, edgecolor=VIOLET, lw=0.5, zorder=3)
        lab_ = f'{pre + dec:.0f} ms = {pre:.0f} prefill + {dec:.0f} decode' if v == 'V0_as_submitted' else f'{pre + dec:.0f} ms'
        if v == 'V1_no_cuda_graph':
            bx.text((pre + dec) / 2, y, lab_, va='center', ha='center', fontsize=5.9, color='white')
        else:
            bx.text(pre + dec + 12, y, lab_, va='center', ha='left', fontsize=6.0, color=INK)
    preV, decV, _ = T['V0_as_submitted']
    preB, decB, _ = T['baseline']
    bx.text(preB + decB * 0.5, 2.0, f'decode: {decB:.0f} ms', ha='center', va='center', fontsize=5.9, color='white')
    dec_x = decB / decV
    e2e_x = (preB + decB) / (preV + decV)
    # the summary names its comparison so it cannot be read as describing the no-graph row it sits beside
    bx.text(905, 1.33, 'as submitted vs baseline:', ha='right', va='center', fontsize=5.8, color=INK2)
    bx.text(905, 0.88, f'decode: {dec_x:.0f}\u00d7 faster\ngeneration: {e2e_x:.1f}\u00d7 faster', ha='right', va='center',
            fontsize=6.1, color=INK, fontweight='bold', linespacing=1.2)
    bx.legend(handles=[Rectangle((0, 0), 1, 1, facecolor='#d9d7d1', edgecolor=INK2, hatch='////', lw=0.5, label='prefill (not scored)'),
                       Rectangle((0, 0), 1, 1, facecolor=VIOLET, edgecolor=VIOLET, label='decode (scored)')],
              loc='lower right', bbox_to_anchor=(1.0, -0.02), frameon=False, fontsize=5.6, handlelength=1.2, ncol=2,
              columnspacing=0.8, handletextpad=0.3, borderaxespad=0.0)
    bx.set_yticks(range(len(rowsb)))
    bx.set_yticklabels([l for _, l in rowsb][::-1], fontsize=6.1)
    bx.tick_params(axis='y', length=0)
    bx.spines['left'].set_visible(False)
    bx.set_xlim(0, 910)
    bx.set_ylim(-0.95, 2.45)
    bx.grid(axis='x', color=GRID, lw=0.5, zorder=0)
    bx.set_xlabel('Milliseconds for one generation')
    bx.set_title('(b) Where the time goes', loc='left', fontsize=7.0)
    save(fig, 'cr_inference')
    return {'pre_base_ms': preB, 'dec_base_ms': decB, 'pre_opus_ms': preV, 'dec_opus_ms': decV,
            'dec_x': dec_x, 'e2e_x': e2e_x, 'nograph_pre_ms': T['V1_no_cuda_graph'][0],
            'nograph_dec_ms': T['V1_no_cuda_graph'][1], 'nofusion_pre_ms': times('V2_no_fusion')[0]}


# ----------------------------------------------------------------------------- E1
def fig_e1_main(D):
    style(1.0)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.5, 1.6), gridspec_kw={'width_ratios': [1.15, 1.0], 'wspace': 0.55})
    draw_e1(ax, D, '(a) Quality vs training tokens (E1)')
    ax.set_xticklabels(['88.6M', '157M', '201M'])
    ax.set_xlabel('Training tokens (log scale); diamonds: 300 s runs')
    from matplotlib.ticker import FormatStrFormatter, MultipleLocator
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_ylabel('Held-out BPB (lower is better)')
    draw_split(bx, D, '(b) The 300 s gap, split (exploratory)')
    bx.set_yticklabels(['Observed\n300 s gap', 'Schedule\nresidual', 'Smaller model\nat equal tokens',
                        'AP-Q\'s extra tokens'])
    bx.set_xlim(-0.112, 0.088)
    bx.set_xlabel('Contribution to AP-Q − AR-ref (BPB)')
    save(fig, 'cr_e1_main')


# ----------------------------------------------------------------------------- tables
def table_results(D):
    """Compact main-results table: held-out BPB difference recipe minus AR-ref, paired by seed."""
    pc = D['e1']['per_cell']['heldout_bpb']
    mt = D['phase5']['matched_tokens']['recipes']
    cur = D['e1']['curves']
    tok = lambda rec, c: statistics.mean(cur[f'{rec}|{c}']['tokens']) / 1e6
    fmt = lambda x: sgn(x, 3)
    row = lambda lab, budget, toks, m, lo, hi, vals: (rf"{lab} & {budget} & {toks} & {fmt(m)} [{fmt(lo)}, {fmt(hi)}] & "
                                                     rf"{sum(v < 0 for v in vals)}/3 \\")
    lines = ['% Generated by analysis/cr_v3.py from cr_v2.json (E1 e1_analysis.json; Phase 5 phase5_analysis.json). Do not edit by hand.',
             r'\begin{tabular}{@{}llccc@{}}', r'\toprule',
             r'Recipe & Budget & Tokens (M) & $\Delta$ held-out BPB [95\% CI] & Better \\', r'\midrule']
    c = pc['time_300s']
    lines.append(row('AP-Q', '300\\,s', f"{tok('AR-ref', 'time_300s'):.0f} / {tok('AP-Q', 'time_300s'):.0f}", c['mean'], *c['ci'], c['values']))
    for cell, lab in (('tokens_088604672', '88.6M tok.'), ('tokens_157286400', '157.3M tok.'), ('tokens_201326592', '201.3M tok.')):
        c = pc[cell]
        lines.append(row('AP-Q', lab, 'equal', c['mean'], *c['ci'], c['values']))
    lines.append(r'\midrule')
    for k, lab in (('OPUS-A23', 'Opus A23'), ('OPUS-A24', 'Opus A24'), ('OPUS-A27', 'Opus A27')):
        x = mt[k]['vs_ar']['heldout']
        lines.append(row(lab, '201.3M tok.', 'equal', x['mean'], *x['ci95'], x['values']))
    lines += [r'\bottomrule', r'\end{tabular}']
    (GEN_CR / 'table_results.tex').write_text('\n'.join(lines) + '\n')


def fig_e1_curve(D):
    style(1.0)
    fig, ax = plt.subplots(figsize=(2.45, 1.5))
    draw_e1(ax, D, '')
    ax.set_title('')
    ax.set_xticklabels(['88.6M', '157M', '201M'])
    ax.set_xlabel('Training tokens (log scale)')
    from matplotlib.ticker import FormatStrFormatter, MultipleLocator
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_ylabel('Held-out BPB (lower is better)')
    for txt in list(ax.texts):
        if txt.get_text() == '300 s':
            txt.remove()
    cur = D['e1']['curves']
    for rec, dx, dy in (('AR-ref', 7, 2), ('AP-Q', -6, 7)):
        tx = statistics.mean(t / 1e6 for t in cur[f'{rec}|time_300s']['tokens'])
        ty = statistics.mean(cur[f'{rec}|time_300s']['heldout'])
        ax.annotate('300 s', (tx, ty), xytext=(dx, dy), textcoords='offset points', ha='left' if dx > 0 else 'right',
                    fontsize=plt.rcParams['font.size'] * 0.9, color=INK2)
    save(fig, 'cr_e1_curve')


def tex_escape(t):
    rep = {'\\': r'\textbackslash{}', '_': r'\_', '%': r'\%', '&': r'\&', '#': r'\#', '$': r'\$', '{': r'\{', '}': r'\}',
           '^': r'\^{}', '~': r'\~{}'}
    t = ''.join(rep.get(ch, ch) for ch in t)
    return t.replace('->', r'$\to$')


def table_apq_path():
    """The AP-Q path, one row per recorded step: edit (derived from the frozen sources), measured outcome,
    archive decision and the opening of the recorded hypothesis (verbatim)."""
    ev = read_json(CR_DATA / 'cr_evidence.json')
    labels = read_json(CR_DATA / 'lineage_edit_labels.json')
    lin = {r['id']: r for r in ev['seed0_lineage']}
    lab = {x['exp']: x for x in labels['labels']}
    PATH = [8, 113, 116, 118, 119, 120, 130, 131, 132, 134]
    out = ['% Generated by analysis/cr_v3.py from cr_evidence.json (seed-0 lineage) and lineage_edit_labels.json. Do not edit by hand.',
           r'\begin{tabular}{@{}l>{\raggedright\arraybackslash}p{2.35cm}rrr>{\raggedright\arraybackslash}p{6.0cm}@{}}', r'\toprule',
           r'Step & Edit (from the frozen source) & Val.\ BPB & $\Delta$ & Tokens (M) & Recorded hypothesis (opening words, verbatim) \\', r'\midrule']
    for eid in PATH:
        r = lin[eid]
        if eid == 8:
            out.append(rf"B & registered baseline: depth 8, width 512, batch $2^{{19}}$ & {r['val_bpb']:.4f} & & "
                       rf"{r['total_training_tokens'] / 1e6:.0f} & \\")
            continue
        p = lin[r['parent_experiment_id']]
        words = ' '.join(r['hypothesis'].split()).split(' ')
        quote = tex_escape(' '.join(words[:22])) + (r' \ldots' if len(words) > 22 else '')
        assert '–' not in quote and '—' not in quote
        tag = r' \textbf{(AP-Q)}' if eid == 134 else ''
        out.append(rf"A{r['experiment_number']}{tag} & {lab[eid]['label']} & {r['val_bpb']:.4f} & {sgn(r['val_bpb'] - p['val_bpb'], 4)} & "
                   rf"{r['total_training_tokens'] / 1e6:.0f} & ``{quote}'' \\")
    out += [r'\bottomrule', r'\end{tabular}']
    (GEN_CR / 'table_apq_path.tex').write_text('\n'.join(out) + '\n')


def table_funnel_v3():
    """Chronological selection funnel.  Every fact is asserted against the cr_tables.py output."""
    src = (GEN_CR / 'table_funnel.tex').read_text()
    for fact in ('75 (seeds 0, 1, 42)', 'rule fixed Jul 27', '11: exp 103, 108, 109, 124, 134, 136, 137, 147, 152, 158, 162',
                 'Global four-role selector (Aug 10, after discovery)', '4, all seed 0: exp 124, 134, 136, 137',
                 '201M-token panel (set Aug 21--25; run Aug 25--27)', '3: exp 134, 136, 137 (exp 124 dropped; no rule recorded)',
                 '68 (65 incompatible)', 'best BPB: 183, 234'):
        assert fact in src, fact
    rows = [
        (r'1. Discovery (Jul 22--Aug 7)', 'agents', r'75 AutoPareto attempts on seeds 0, 1 and 42 (plus AutoResearch, TPE, MOTPE)'),
        (r'2. Representatives (rule fixed Jul 27, \emph{before} discovery)', 'pre-specified', r'11 AutoPareto recipes, including exp 109 (seed 42)'),
        (r'3. Four roles (Aug 10, \emph{after} discovery)', 'post hoc', r'4 seed-0 recipes: exp 124, 134 (AP-Q), 136, 137; then seeds 0--2 re-runs'),
        (r'4. 201M-token panel (Aug 21--25, after step-3 re-runs)', 'post hoc', r'3 of the 4 (exp 124 dropped, no rule recorded) + AR-ref, TPE, MOTPE, baseline'),
        (r'5. E1 and Opus retraining (Nov 4 and Nov 8)', 'pre-registered', r'AP-Q, AR-ref; Opus A23, A24, A27 (its whole frontier)'),
    ]
    out = ['% Generated by analysis/cr_v3.py; facts asserted against table_funnel.tex (cr_tables.py). Do not edit by hand.',
           r'\begin{tabular}{@{}>{\raggedright\arraybackslash}p{4.3cm}l>{\raggedright\arraybackslash}p{6.4cm}@{}}', r'\toprule',
           r'Stage (2026) & Selection & What was kept \\', r'\midrule']
    out += [rf'{a} & {b} & {c} \\' for a, b, c in rows]
    out += [r'\bottomrule', r'\end{tabular}']
    (GEN_CR / 'table_funnel_v3.tex').write_text('\n'.join(out) + '\n')


# ----------------------------------------------------------------------------- revision 5 (2026-11-14)
def badge(ax, x, y, n, **kw):
    ax.text(x, y, str(n), ha='center', va='center', fontsize=5.6, color='white', fontweight='bold', zorder=8,
            bbox={'boxstyle': 'circle,pad=0.18', 'facecolor': INK, 'edgecolor': 'none'}, **kw)


def fig_opus_milestones(D):
    """Main-text Figure 2 (revision 5): the six milestones of Opus's search, drawn only from recorded parent links.
    The complete lineage (every attempt and branch) stays in the appendix figure cr_opus_lineage."""
    style(1.0)
    par = _pair_a_parents(OPUS_ROOT)
    rows = {r['attempt']: r for r in D['agents']['opus']['rows']}
    base = D['agents']['baseline']
    valid = {a: r for a, r in rows.items() if r['valid']}
    best = min(valid.values(), key=lambda r: r['val_bpb'])['attempt']
    path = _ancestry(par, best)
    assert path == [1, 3, 4, 5, 7, 8, 10, 11, 13, 14, 17, 18, 19, 20, 21, 22, 23], path
    assert par[1] is None and par[6] == 5 and par[7] == 5, 'A6 and A7 must both branch from A5 in the records'
    frontier = [23, 24, 27]
    assert D['agents']['opus']['frontier_extended'] == frontier
    assert par[24] == 23 and par[27] == 26 and par[26] == 23
    failed = sorted(a for a, r in rows.items() if not r['valid'])
    assert failed == [2, 9], failed
    bpb = lambda a: base['val_bpb'] if a == 0 else rows[a]['val_bpb']
    # milestone facts, asserted from the records
    assert rows[1]['category'] == 'optimization_batch_size' and rows[7]['category'] == 'optimization_batch_size'
    assert all(rows[a]['category'] == 'architecture_depth' for a in (3, 4, 5, 6))
    assert rows[10]['category'] == 'inference_adapter_decode_speed' and rows[22]['category'] == 'inference_decode_kernel_fusion'
    assert rows[6]['val_bpb'] > rows[5]['val_bpb'] and rows[7]['params'] == rows[5]['params']

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(5.5, 1.94), sharex=True,
                                 gridspec_kw={'height_ratios': [1.75, 1.0], 'hspace': 0.1})
    for a_ in (ax, bx):
        a_.grid(axis='y', color=GRID, lw=0.5, zorder=0)
    fs = 6.4
    # top: validation BPB. Gray: other valid attempts (no edges; full lineage in the appendix).
    others = [a for a in valid if a not in path and a not in frontier and a != 6]
    ax.scatter(others, [rows[a]['val_bpb'] for a in others], s=11, facecolor='white', edgecolor=MUTED, lw=0.7, zorder=3)
    for p, a in [(par[a] if par[a] is not None else 0, a) for a in path]:
        ax.plot([p, a], [bpb(p), bpb(a)], color=VIOLET, lw=1.4, zorder=4, solid_capstyle='round')
    for a in (6, 24, 26, 27):          # recorded side branches that matter for the milestones
        p = par[a]
        ax.plot([p, a], [bpb(p), bpb(a)], color=VIOLET, lw=0.8, ls=(0, (2.2, 1.4)), zorder=3)
    ax.scatter([0], [base['val_bpb']], s=24, color=INK, zorder=6)
    ax.scatter(path, [rows[a]['val_bpb'] for a in path], s=15, color=VIOLET, zorder=5, edgecolor='white', lw=0.4)
    ax.scatter([6, 26], [rows[6]['val_bpb'], rows[26]['val_bpb']], s=15, facecolor='white', edgecolor=VIOLET, lw=0.9, zorder=5)
    for a in frontier:
        ax.scatter([a], [rows[a]['val_bpb']], s=58, facecolor='none', edgecolor=INK, lw=0.9, zorder=6)
        if a != 23:
            ax.scatter([a], [rows[a]['val_bpb']], s=15, color=VIOLET, zorder=5, edgecolor='white', lw=0.4)

    def lab(n, x, y, text, xy=None, ha='left'):
        dx = 0.62 if ha == 'left' else -0.62
        badge(ax, x, y, n)
        ax.text(x + dx, y, text, ha=ha, va='center', fontsize=fs, color=INK, zorder=7)
        if xy is not None:
            ax.annotate('', xy, xytext=(x, y), arrowprops={'arrowstyle': '-', 'lw': 0.5, 'color': INK2,
                                                            'shrinkA': 3.2, 'shrinkB': 2.2}, zorder=2)

    r = rows
    lab(1, 2.0, 1.1905, f"Halve the batch (A1): {r[1]['updates']} updates instead of 163", (1, r[1]['val_bpb']))
    lab(2, 6.0, 1.1655, f"Cut depth 8 → 3 layers (A3–A5): {r[1]['tokens']/1e6:.0f}M → {r[5]['tokens']/1e6:.0f}M "
                        f"tokens in 300 s", (4.15, r[4]['val_bpb']))
    lab(3, 7.6, 1.1435, f"2 layers (A6): worse by {r[6]['val_bpb'] - r[5]['val_bpb']:.3f}, abandoned", (6, r[6]['val_bpb']))
    lab(4, 8.2, 1.1225, "Back to 3 layers, batch halved again (A7)", (7, r[7]['val_bpb']))
    lab(6, 27.4, 1.1125, 'Frontier (A23, A24, A27)', (25.6, 1.1008), ha='right')
    gain = r[7]['val_bpb'] - r[best]['val_bpb']
    nlate = sum(1 for a in valid if a > 7)
    ax.text(13.0, 1.0878, f'after A7: {nlate} more valid attempts gain only {gain:.4f} BPB in total',
            fontsize=5.9, color=INK2, ha='center', va='center')
    ax.set_ylim(1.082, 1.207)
    ax.set_ylabel('Validation BPB\n(lower = better)')
    hand = [Line2D([], [], color=VIOLET, lw=1.4, marker='o', ms=3.4, mec='white', mew=0.4, label='path to the best recipe (recorded parents)'),
            Line2D([], [], color=VIOLET, lw=0.8, ls=(0, (2.2, 1.4)), marker='o', ms=3.4, mfc='white', mec=VIOLET, label='side branch'),
            Line2D([], [], ls='none', marker='o', ms=3.2, mfc='white', mec=MUTED, label='other valid attempts'),
            Line2D([], [], ls='none', marker='o', ms=6.2, mfc='none', mec=INK, label='final frontier')]
    ax.legend(handles=hand, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=5.9,
              handletextpad=0.35, borderaxespad=0.15, columnspacing=1.1, handlelength=1.9)

    # bottom: scored decode-only speed-up along the path
    xs = [0] + path
    dec = [1.0] + [rows[a]['decode_x'] for a in path]
    bx.plot(xs, dec, color=VIOLET, lw=1.4, zorder=3)
    bx.scatter(path, dec[1:], s=13, color=VIOLET, zorder=4, edgecolor='white', lw=0.4)
    bx.scatter([0], [1.0], s=20, color=INK, zorder=5)
    badge(bx, 1.2, 15.0, 5)
    bx.text(1.82, 15.0, 'CUDA graph replay (A10)', ha='left', va='center', fontsize=fs, color=INK)
    bx.annotate('', (10, rows[10]['decode_x']), xytext=(9.3, 13.5),
                arrowprops={'arrowstyle': '-', 'lw': 0.5, 'color': INK2, 'shrinkA': 0.5, 'shrinkB': 2.2})
    bx.text(12.4, 62, 'kernel fusion (A14, A22)', ha='left', va='center', fontsize=fs, color=INK)
    for a, xt in ((14, 13.4), (22, 19.6)):
        bx.annotate('', (a, rows[a]['decode_x']), xytext=(xt, 52),
                    arrowprops={'arrowstyle': '-', 'lw': 0.5, 'color': INK2, 'shrinkA': 0.5, 'shrinkB': 2.2})
    bx.text(23.6, rows[23]['decode_x'], f"{rows[23]['decode_x']:.0f}×", ha='left', va='center', fontsize=fs,
            color=INK, fontweight='bold')
    bx.text(27.7, 2.2, 'decode only (scored); a full generation is about 5× faster (Figure 4)', ha='right',
            va='center', fontsize=5.8, color=INK2)
    bx.set_yscale('log')
    bx.set_yticks([1, 3, 10, 30])
    bx.set_yticklabels(['1×', '3×', '10×', '30×'])
    bx.minorticks_off()
    bx.set_ylim(0.75, 110)
    bx.set_ylabel('Decode\nspeed-up')
    bx.set_xlim(-0.6, 27.8)
    bx.set_xticks([0, 1, 3, 5, 7, 10, 14, 17, 20, 23, 27])
    bx.set_xticklabels(['B', '1', '3', '5', '7', '10', '14', '17', '20', '23', '27'])
    bx.set_xlabel('Attempt (Claude Opus 5.5, seed 3, 300 s per attempt; B = starting model)')
    fig.align_ylabels([ax, bx])
    save(fig, 'cr_opus_milestones')
    return {'late_valid': nlate, 'late_gain': gain}



def fig_loop_compact():
    """Main-text Figure 1 (revision 5): the AutoPareto loop in one row, drawn at print width so that every
    label prints at 5.9 pt or larger.  Hand-authored content (same steps as generate_figures.research_loop)."""
    import textwrap
    from matplotlib.patches import FancyBboxPatch
    style(1.0)
    FIGW, FIGH = 5.5, 1.06
    fig = plt.figure(figsize=(FIGW, FIGH))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set(xlim=(0, FIGW), ylim=(0, FIGH)); ax.axis('off')
    blue, green, orange, ink, body = '#dbe7f0', '#dee9e0', '#f6e3c6', '#25333f', '#3d4952'
    TS, BS = 6.7, 5.9

    def box(x, w, title, text, col, y=0.33, h=0.58):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=.05', facecolor=col,
                                    edgecolor=ink, lw=.7, zorder=2))
        ax.text(x + w / 2, y + h - .05, title, ha='center', va='top', fontsize=TS, fontweight='bold', color=ink)
        chars = max(8, int((w - .06) * 72 / (0.56 * BS)))
        ax.text(x + w / 2, y + h - .175, '\n'.join(textwrap.wrap(text, chars)), ha='center', va='top', fontsize=BS,
                color=body, linespacing=1.12)
        return x, w, y, h

    def arrow(a, b):
        ax.annotate('', xy=b, xytext=a, arrowprops={'arrowstyle': '-|>', 'mutation_scale': 7, 'lw': .9,
                                                    'color': '#5c6a75', 'shrinkA': 1, 'shrinkB': 1}, zorder=3)
    B = [box(0.03, 0.86, 'Human', 'objectives, 300 s budget, seed, one editable file', blue),
         box(1.03, 1.10, 'Coding agent', 'reads every record, picks a parent, states a hypothesis, edits train.py', green)]
    x0, w, g = 2.36, 0.70, 0.06
    steps = [('Validate', 'only train.py changed, hashes match'), ('Train', '300 s on one A40, or N* tokens'),
             ('Evaluate', 'BPB, decode speed, memory, correctness'), ('Record', 'hypothesis, patch and metrics, failures too')]
    for i, (t, b) in enumerate(steps):
        B.append(box(x0 + i * (w + g), w, t, b, orange))
    for a, b in zip(B, B[1:]):
        arrow((a[0] + a[1], a[2] + a[3] / 2), (b[0], b[2] + b[3] / 2))
    ax.add_patch(FancyBboxPatch((x0 - .07, .25), 4 * w + 3 * g + .14, .80, boxstyle='round,pad=0,rounding_size=.04',
                                facecolor='none', edgecolor='#8b98a4', lw=.8, linestyle=(0, (3, 2)), zorder=0))
    ax.text(x0 + (4 * w + 3 * g) / 2, 0.99, 'PROTECTED CODE, HASH-CHECKED EVERY ATTEMPT', ha='center', va='center',
            fontsize=5.6, fontweight='bold', color='#53616e')
    last, agent = B[-1], B[1]
    ys = 0.16
    ax.plot([last[0] + last[1] / 2, last[0] + last[1] / 2, agent[0] + agent[1] / 2], [last[2], ys, ys], color='#5c6a75', lw=.9, zorder=1)
    arrow((agent[0] + agent[1] / 2, ys), (agent[0] + agent[1] / 2, agent[2]))
    ax.text(3.35, ys - .035, 'the next attempt sees every earlier record and may branch from any eligible parent',
            ha='center', va='top', fontsize=5.6, color=body)
    save(fig, 'research_loop_compact')

SEED_MARK = {6: 'o', 7: 's', 8: '^'}


def fig_e1_verify(D):
    """Main-text Figure 3 (revision 5): (a) E1 learning curves, equal time vs equal tokens; (b) paired
    differences against AR-ref, one marker shape per seed (the same seeds 6, 7, 8 in every column)."""
    style(1.0)
    pc = D['e1']['per_cell']['heldout_bpb']
    cur = D['e1']['curves']
    mt = D['phase5']['matched_tokens']['recipes']
    for c in CELLS:
        assert pc[c]['seeds'] == [6, 7, 8] and pc[c]['n'] == 3
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.5, 1.6), gridspec_kw={'width_ratios': [1.0, 1.3], 'wspace': 0.44})

    # (a) learning curves
    for rec, col in (('AR-ref', ORANGE), ('AP-Q', BLUE)):
        xs = [statistics.mean(cur[f'{rec}|{c}']['tokens']) / 1e6 for c in CELLS[:3]]
        ys = [statistics.mean(cur[f'{rec}|{c}']['heldout']) for c in CELLS[:3]]
        ax.plot(xs, ys, color=col, lw=1.5, zorder=2)
        ax.scatter(xs, ys, s=15, color=col, zorder=4, edgecolors='white', linewidths=0.6)
        tx = statistics.mean(cur[f'{rec}|time_300s']['tokens']) / 1e6
        ty = statistics.mean(cur[f'{rec}|time_300s']['heldout'])
        ax.scatter([tx], [ty], marker='D', s=30, color=col, zorder=5, edgecolors=INK, linewidths=0.7)
        ax.annotate(rec, (xs[-1], ys[-1]), xytext=(4, 0), textcoords='offset points', va='center', color=INK, fontsize=6.4)
    tAR = statistics.mean(cur['AR-ref|time_300s']['tokens']) / 1e6
    tAP = statistics.mean(cur['AP-Q|time_300s']['tokens']) / 1e6
    yAR = statistics.mean(cur['AR-ref|time_300s']['heldout'])
    yAP = statistics.mean(cur['AP-Q|time_300s']['heldout'])
    ax.plot([tAR, tAP], [yAR, yAP], color=INK2, lw=0.7, ls=(0, (1.5, 1.5)), zorder=1)
    ax.text((tAR * tAP) ** 0.5, max(yAR, yAP) + 0.0035, '300 s each', ha='center', va='bottom', fontsize=5.9, color=INK)
    ax.set_xscale('log')
    ax.set_xticks([88.6, 157.3, 201.3])
    ax.set_xticklabels(['88.6M', '157M', '201M'])
    ax.minorticks_off()
    ax.set_xlim(80, 262)
    from matplotlib.ticker import FormatStrFormatter, MultipleLocator
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_ylim(1.088, 1.205)
    ax.legend(handles=[Line2D([], [], color=INK2, lw=1.2, marker='o', ms=3.2, mec='white', label='equal tokens'),
                       Line2D([], [], ls='none', marker='D', ms=3.6, mfc=INK2, mec=INK, mew=0.6, label='equal time (300 s)')],
              loc='lower left', frameon=False, fontsize=5.7, handlelength=1.5, handletextpad=0.35, borderaxespad=0.2)
    ax.set_xlabel('Training tokens (log scale)')
    ax.set_ylabel('Held-out BPB (lower is better)')
    ax.grid(axis='y', color=GRID, lw=0.5)
    ax.set_title('(a) Held-out BPB, seed means', loc='left', fontsize=6.6)

    # (b) paired differences, recipe minus AR-ref
    cols = [('time_300s', '300 s'), ('tokens_088604672', '88.6M'), ('tokens_157286400', '157M'), ('tokens_201326592', '201M')]
    for i, (c, _) in enumerate(cols):
        st = pc[c]
        bx.plot([i, i], st['ci'], color=BLUE, lw=1.1, zorder=3)
        bx.scatter([i], [st['mean']], s=26, color=BLUE, zorder=5, edgecolors='white', linewidths=0.6)
        for s, v in zip(st['seeds'], st['values']):
            bx.scatter([i + 0.2], [v], s=10, marker=SEED_MARK[s], facecolor='white', edgecolor=BLUE, lw=0.7, zorder=4)
    ox = 4.35
    for j, k in enumerate(('OPUS-A23', 'OPUS-A24', 'OPUS-A27')):
        x = mt[k]['vs_ar']['heldout']
        xx = ox + 0.32 * j
        bx.plot([xx, xx], x['ci95'], color=VIOLET, lw=1.1, zorder=3)
        bx.scatter([xx], [x['mean']], s=22, color=VIOLET, zorder=5, edgecolors='white', linewidths=0.6)
    bx.axhline(0, color=INK2, lw=0.7, zorder=2)
    bx.text(5.3, 0.003, 'no difference', fontsize=5.6, color=INK2, va='bottom', ha='right')
    bx.text(0.12, pc['time_300s']['ci'][1] + 0.004, 'not\nresolved', ha='center', va='bottom', fontsize=5.9, color=INK, linespacing=1.0)
    bx.text(2.1, 0.066, 'AP-Q worse for all 3 seeds\nat every token budget', ha='center', va='bottom', fontsize=5.9, color=INK,
            linespacing=1.05)
    bx.text(ox + 0.32, 0.075, 'Opus\nworse still', ha='center', va='bottom', fontsize=5.9, color=INK, linespacing=1.0)
    bx.set_xticks([0, 1, 2, 3, ox + 0.32])
    bx.set_xticklabels(['300 s', '88.6M', '157M', '201M', '201M'])
    bx.set_xlim(-0.55, 5.35)
    bx.set_ylim(-0.03, 0.094)
    bx.axvline(0.6, color=GRID, lw=0.8, zorder=1)
    bx.axvline(3.68, color=GRID, lw=0.8, zorder=1)
    for x, t in ((0.0, 'equal time'), (2.0, 'AP-Q, equal tokens'), (ox + 0.32, 'Opus A23/24/27')):
        bx.text(x, -0.155, t, transform=bx.get_xaxis_transform(), ha='center', va='top', fontsize=5.9, color=INK2)
    bx.set_ylabel('Recipe − AR-ref (BPB)\nabove 0: recipe worse')
    from matplotlib.ticker import MultipleLocator as _ML
    bx.yaxis.set_major_locator(_ML(0.02))
    bx.grid(axis='y', color=GRID, lw=0.5)
    hand = [Line2D([], [], ls='none', marker=SEED_MARK[s], ms=3.3, mfc='white', mec=BLUE, mew=0.7, label=f'seed {s}')
            for s in (6, 7, 8)] + [Line2D([], [], color=INK2, lw=1.0, marker='o', ms=3.4, mfc=INK2, mec='white', label='mean, 95% CI')]
    bx.legend(handles=hand, loc='lower right', ncol=2, frameon=False, fontsize=5.6,
              handletextpad=0.2, columnspacing=0.7, borderaxespad=0.15, handlelength=1.2)
    bx.set_title('(b) Difference from AR-ref, paired by seed', loc='left', fontsize=6.6)
    save(fig, 'cr_e1_verify')


# ----------------------------------------------------------------------------- Pair-A contract pilot (archival)
PILOT_LAUNCHES = [   # (label, final continuation directory holding the complete history of the launch)
    ('Sep 5', 'runs/pair-a'),   # the audit registry's first launch lineage (namespace pair-a)
    ('Sep 6 (fresh)', 'runs/pair-a-official-20260906-fresh'),
    ('Sep 6 (warm-up v2)', 'runs/pair-a-official-20260906-fixed-warmup-v2-continuation-20260907-new-a40-r2'),
    ('Sep 9', 'runs/pair-a-official-20260909-new-a40'),
    ('Sep 14', 'runs/pair-a-official-20260914-new-a40-continuation-20260916-r2'),
]
PILOT_QUOTE = (2, 'pair-a-fixed-tokens-seed3', 1, 'by increasing model capacity')


def _pilot_arm(root, arm):
    p = root / arm / 'history.jsonl'
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        res = d.get('result') or {}
        src = root / arm / 'sources' / f"{res.get('experiment_id')}.train.py"
        cfg = {}
        if src.exists():
            t = src.read_text()
            for k in ('DEPTH', 'TOTAL_BATCH_SIZE'):
                m = re.search(rf'^{k}\s*=\s*([^#\n]+)', t, re.M)
                if m:
                    cfg[k] = m.group(1).strip()
        step = (res.get('candidate_reported') or {}).get('step')
        # valid: completed and trained (the Sep 6 restart defect checkpointed an untrained model: step 0, BPB 3.216)
        ok = d.get('status') == 'completed' and res.get('protected_validation_bpb') is not None and (step or 0) > 0
        if ok:
            assert res['protected_validation_bpb'] < 1.3, (root.name, arm, d['campaign_attempt'])
        out.append({'attempt': d['campaign_attempt'], 'valid': ok, 'bpb': res.get('protected_validation_bpb'),
                    'params': res.get('num_params'), 'tokens': res.get('total_tokens'), 'depth': cfg.get('DEPTH'),
                    'batch': cfg.get('TOTAL_BATCH_SIZE'), 'hyp': d.get('hypothesis') or ''})
    return out


def table_contract_pilot():
    """Archival summary of every Codex launch of the Pair-A contract pilot, both arms.  Read-only."""
    from cr_common import EVIDENCE
    lines = ['% Generated by analysis/cr_v3.py from the Pair-A launch histories (read-only). Do not edit by hand.',
             r'\begin{tabular}{@{}llrlllr@{}}', r'\toprule',
             r'Launch & Arm & Valid / all & Batch sizes & Depths & Parameters (M), valid & Best: params \\', r'\midrule']
    S = {}
    for i, (lab, rel) in enumerate(PILOT_LAUNCHES):
        root = EVIDENCE / rel
        for arm, an in (('pair-a-fixed-time-seed3', 'time'), ('pair-a-fixed-tokens-seed3', 'tokens')):
            rs = _pilot_arm(root, arm)
            v = [r for r in rs if r['valid']]
            S[(i, an)] = v
            if not rs:
                lines.append(rf'{lab} & {an} & 0 / 0 & & & & \\')
                continue
            bs = sorted({r['batch'] for r in v if r['batch']}, key=lambda b: -int(b.split('**')[1]))
            ds = sorted({int(r['depth']) for r in v if r['depth']})
            if v:
                pm = [r['params'] / 1e6 for r in v]
                b = min(v, key=lambda r: r['bpb'])
                rng = f'{min(pm):.1f}--{max(pm):.1f}' if max(pm) - min(pm) > 0.05 else f'{pm[0]:.1f}'
                bt = rf"{b['params'] / 1e6:.1f}"
            else:
                rng, bt = '', ''
            bl = ', '.join(b.replace('2**', '$2^{') + '}$' for b in bs)
            dl = (f'{ds[0]}--{ds[-1]}' if len(ds) > 1 else str(ds[0])) if ds else ''
            lines.append(rf"{lab if an == 'time' else ''} & {an} & {len(v)} / {len(rs)} & {bl} & {dl} & {rng} & {bt} \\")
        if i < len(PILOT_LAUNCHES) - 1:
            lines.append(r'\addlinespace[1pt]')
    lines += [r'\bottomrule', r'\end{tabular}']
    (GEN_CR / 'table_contract_pilot.tex').write_text('\n'.join(lines) + '\n')
    # the facts quoted in the text, asserted from the records
    best = lambda i, an: min(S[(i, an)], key=lambda r: r['bpb'])['params'] / 1e6
    pairs = [(best(i, 'time'), best(i, 'tokens')) for i in (2, 3, 4)]
    assert all(n > t for t, n in pairs), pairs
    assert abs(best(1, 'time') - best(1, 'tokens')) < 1e-9          # Sep 6 fresh: neither arm edited the architecture
    halved = {an: [i for i in range(5) if any(r['batch'] not in (None, '2**19') for r in S[(i, an)])] for an in ('time', 'tokens')}
    assert halved['time'] == [1, 2, 4] and halved['tokens'] == [0, 1, 2, 3], halved
    grow = [r for i in range(5) for r in S[(i, 'tokens')] if r['params'] > 50.4e6]
    assert {int(r['depth']) for r in grow} == {10, 12}
    assert not [r for i in range(5) for r in S[(i, 'time')] if r['params'] > 51e6]
    i, arm, a, q = PILOT_QUOTE
    hyp = [r for r in _pilot_arm(EVIDENCE / PILOT_LAUNCHES[i][1], arm) if r['attempt'] == a][0]['hyp']
    assert q in hyp, q
    assert not S[(0, 'time')] and len(S[(0, 'tokens')]) == 1
    return {'pairs': pairs, 'ratio_lo': min(n / t for t, n in pairs), 'ratio_hi': max(n / t for t, n in pairs),
            'valid': {k: len(v) for k, v in S.items()}}


def macros(D, L, I, P):
    rows = {r['attempt']: r for r in D['agents']['opus']['rows']}
    pp = P['pairs']
    mt = D['phase5']['matched_tokens']['recipes']
    vs = [mt[k]['vs_apq']['heldout']['mean'] for k in ('OPUS-A23', 'OPUS-A24', 'OPUS-A27')]
    AP = (min(vs), max(vs))
    v2 = (GEN_CR / 'cr_macros_v2.tex').read_text()
    assert f'\\newcommand{{\\CRMTDAPLo}}{{+{AP[0]:.3f}}}' in v2 and f'\\newcommand{{\\CRMTDAPHi}}{{+{AP[1]:.3f}}}' in v2, AP
    M = {
        'CRFivePilotTA': f'{pp[0][0]:.1f}', 'CRFivePilotNA': f'{pp[0][1]:.1f}',
        'CRFivePilotTB': f'{pp[1][0]:.1f}', 'CRFivePilotNB': f'{pp[1][1]:.1f}',
        'CRFivePilotTC': f'{pp[2][0]:.1f}', 'CRFivePilotNC': f'{pp[2][1]:.1f}',
        'CRFivePilotRatioLo': f"{P['ratio_lo']:.1f}", 'CRFivePilotRatioHi': f"{P['ratio_hi']:.1f}",
        'CRFiveMTDAPLoAbs': f'{AP[0]:.3f}', 'CRFiveMTDAPHiAbs': f'{AP[1]:.3f}',
        'CRThreeLateValid': str(L['late_valid']), 'CRThreeLateGain': f"{L['late_gain']:.4f}",
        'CRThreePreBase': f"{I['pre_base_ms']:.0f}", 'CRThreeDecBase': f"{I['dec_base_ms']:.0f}",
        'CRThreePreOpus': f"{I['pre_opus_ms']:.0f}", 'CRThreeDecOpus': f"{I['dec_opus_ms']:.0f}",
        'CRThreeTotBase': f"{I['pre_base_ms'] + I['dec_base_ms']:.0f}", 'CRThreeTotOpus': f"{I['pre_opus_ms'] + I['dec_opus_ms']:.0f}",
        'CRThreeDecX': f"{I['dec_x']:.0f}", 'CRThreeEtoEX': f"{I['e2e_x']:.1f}",
        'CRThreeCaptureMs': f"{I['pre_opus_ms'] - I['nograph_pre_ms']:.0f}",
        'CRThreePreNoFusion': f"{I['nofusion_pre_ms']:.0f}",
        'CRThreeOneUpd': f"{rows[1]['updates']}", 'CRThreeSevenUpd': f"{rows[7]['updates']:,}".replace(',', '{,}'),
        'CRThreeSixWorse': f"{rows[6]['val_bpb'] - rows[5]['val_bpb']:.3f}",
        'CRThreeTwelveTok': f"{rows[12]['tokens'] / 1e6:.0f}", 'CRThreeElevenTok': f"{rows[11]['tokens'] / 1e6:.0f}",
        'CRThreeTwentyFiveTok': f"{rows[25]['tokens'] / 1e6:.0f}",
        'CRThreeOneTok': f"{rows[1]['tokens'] / 1e6:.0f}", 'CRThreeFiveTok': f"{rows[5]['tokens'] / 1e6:.0f}",
        'CRThreeFiveParams': f"{rows[5]['params'] / 1e6:.1f}",
        'CRThreeInfOnly': str(len(L['inference_only'])),
    }
    text = '% Generated by analysis/cr_v3.py. Do not edit by hand.\n' + ''.join(
        f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in sorted(M.items()))
    (GEN_CR / 'cr_macros_v3.tex').write_text(text)
    return M


PART1_OPUS_QUOTES = {
    1: 'The model is therefore optimizer-update-limited, not token-limited.',
    7: 'the quality prediction was falsified',
    10: 'the only lever that can raise speed without a quality cost',
    11: 'Prefill time rose from 6.4 ms to 81 ms',
}
PART1_OTHER_QUOTES = ('roughly four times as many optimizer updates within the fixed training window',
                      'the smaller model processes more tokens in the fixed training time',
                      'higher token throughput within Contract T')


def check_part1_quotes():
    """Every agent excerpt quoted in Part 1 must appear verbatim in its attempt record (Opus) or in the
    verbatim-checked reasoning table (Codex, AutoPareto)."""
    H = hyp_records()
    for a, q in PART1_OPUS_QUOTES.items():
        verbatim(H, a, q)
    reasoning = ' '.join((GEN_CR / 'table_reasoning.tex').read_text().split())
    for q in PART1_OTHER_QUOTES:
        assert q in reasoning, q
    prog = (OPUS_ROOT.parents[2] / 'study/contracts_2026/programs/program_fixed_time.md').read_text()
    assert 'diagnostic prefill/TTFT' in prog


def main():
    check_part1_quotes()
    D = read_json(CR_DATA / 'cr_v2.json')
    L = fig_opus_lineage(D)
    L5 = fig_opus_milestones(D)
    fig_loop_compact()
    assert L5['late_valid'] == L['late_valid'] and L5['late_gain'] == L['late_gain']
    I = fig_inference(D)
    fig_e1_main(D)
    fig_e1_verify(D)
    table_results(D)
    table_apq_path()
    table_funnel_v3()
    P = table_contract_pilot()
    M = macros(D, L, I, P)
    print('cr_v3:', json.dumps({k: M[k] for k in ('CRThreeDecX', 'CRThreeEtoEX', 'CRThreeLateGain', 'CRThreeLateValid')}))


if __name__ == '__main__':
    main()
