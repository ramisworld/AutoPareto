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
    mech = [(10, 'CUDA-graph\nreplay', 8.0, 17), (14, 'compiled,\nfused kernels', 12.2, 36), (18, 'ring K/V\ncache', 16.0, 55),
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
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.5, 1.62), gridspec_kw={'width_ratios': [1.1, 1.0], 'wspace': 0.42})
    n = len(VARIANTS) + 1
    labels = []
    for i, (v, lab) in enumerate(VARIANTS + [('baseline', 'Registered baseline (8 layers)')]):
        y = n - 1 - i
        labels.append((y, lab))
        for k, dy, face in (('decode', 0.17, VIOLET), ('e2e', -0.17, 'white')):
            val = 1.0 if v == 'baseline' else med(v, k)
            if v != 'baseline':
                ax.barh(y + dy, val - 1, left=1, height=0.32, color=face, edgecolor=VIOLET, lw=0.7, zorder=3)
                lo, hi = rng(v, k)
                if hi / lo > 1.03:
                    ax.plot([lo, hi], [y + dy, y + dy], color=INK, lw=0.6, zorder=4)
            txt = f'{val:.0f}×' if (k == 'decode' and val >= 10) else f'{val:.1f}×'
            xt = max(val, rng(v, k)[1] if v != 'baseline' else 1.0)
            ax.text(xt * 1.09, y + dy, txt, va='center', ha='left', fontsize=5.8, color=INK)
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
        bx.text(pre + dec + 12, y, f'{pre + dec:.0f} ms', va='center', ha='left', fontsize=6.0, color=INK)
    preV, decV, _ = T['V0_as_submitted']
    preB, decB, _ = T['baseline']
    bx.text(preB + decB * 0.5, 2.0, f'decode (scored): {decB:.0f} ms', ha='center', va='center', fontsize=5.9, color='white')
    bx.annotate(f'prefill + graph capture (not scored):\n{preV:.0f} ms, against {preB:.0f} ms for the baseline',
                (preV * 0.55, 0.2), xytext=(250, 0.42), textcoords='data', fontsize=5.8, ha='left', va='center', color=INK,
                arrowprops={'arrowstyle': '-', 'lw': 0.5, 'color': INK2, 'shrinkA': 1, 'shrinkB': 1})
    bx.annotate(f'decode (scored): {decV:.0f} ms', (preV + decV * 0.5, -0.2), xytext=(250, -0.5),
                textcoords='data', fontsize=5.8, ha='left', va='center', color=INK,
                arrowprops={'arrowstyle': '-', 'lw': 0.5, 'color': INK2, 'shrinkA': 1, 'shrinkB': 1})
    dec_x = decB / decV
    e2e_x = (preB + decB) / (preV + decV)
    bx.text(905, 1.5, f'decode only: {dec_x:.0f}× faster\nend to end: {e2e_x:.1f}× faster', ha='right', va='center',
            fontsize=6.3, color=INK, fontweight='bold', linespacing=1.15)
    bx.set_yticks(range(len(rowsb)))
    bx.set_yticklabels([l for _, l in rowsb][::-1], fontsize=6.1)
    bx.tick_params(axis='y', length=0)
    bx.spines['left'].set_visible(False)
    bx.set_xlim(0, 910)
    bx.set_ylim(-0.85, 2.45)
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


def macros(D, L, I):
    rows = {r['attempt']: r for r in D['agents']['opus']['rows']}
    M = {
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
    I = fig_inference(D)
    fig_e1_main(D)
    fig_e1_curve(D)
    table_results(D)
    table_apq_path()
    table_funnel_v3()
    M = macros(D, L, I)
    print('cr_v3:', json.dumps({k: M[k] for k in ('CRThreeDecX', 'CRThreeEtoEX', 'CRThreeLateGain', 'CRThreeLateValid')}))


if __name__ == '__main__':
    main()
