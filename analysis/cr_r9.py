#!/usr/bin/env python3
"""Camera-ready revision 9 (2026-11-15): final visual and scientific presentation.

One candidate comparison (Figure 1), three research histories (Figure 2), matched-token verification with one
downstream table (Figure 3, Table 1), the Opus decode progression with its ablations (Figure 4), and the appendix
scatterplot, complete histories, consolidated E1 figure and per-seed downstream table.

Candidate names (glossary): GPT A22 = AP-Q = exp 134 (AP292 in the submitted version); GPT A24 = AP-S = exp 136
(AP294); GPT A25 = AP-F = exp 137 (AP299); AR-ref = exp 68; Opus A23, A24, A27 = Opus attempts.

Decode protocol rule (author's instruction): every decode and memory value in a Part-1 figure comes from the
benchmark of 1,024 input and 128 generated tokens.  Only the registered baseline and Opus's recipes were measured
on it; nothing is estimated for the others.  The original 256-token benchmark appears only in the appendix
scatterplot (its own panel) and the panel table.

Reads only committed analysis outputs (paper/camera_ready/data/*.json), the read-only Phase 5 analysis, the
experiment-database snapshot (AutoResearch lineage), the extended-experiment attempt records (history.jsonl) and
the raw Opus re-measurement records (via cr_v3.p2_raw_decode).  Launches nothing; no GPU; no network.
"""
from __future__ import annotations

import json
import os
import re
import statistics
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / 'paper/camera_ready/.mplcache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

from cr_common import CR_DATA, GEN_CR, read_json, write_json
from cr_v2 import (AQUA, BLUE, CELLS, GRAY, GRID, INK, INK2, MUTED, OPUS_ROOT, ORANGE, P5, VIOLET, draw_split,
                   save, style)
from cr_v3 import IA_MODELS, SEED_MARK, p2_raw_decode, tex_escape
from cr_r8 import _need, _path, draw_track, track_apq, track_ar, track_extended, tracks_data

COL = {'base': GRAY, 'ar': ORANGE, 'cls': AQUA, 'gpt': BLUE, 'opus': VIOLET}
LIGHT = {'ar': '#f6c2a9', 'gpt': '#a9c9ef', 'opus': '#c3bcf0', 'cls': '#a7e1cb', 'base': '#cfcecb'}
PANEL = [('baseline', 'Baseline', 'base'), ('autoresearch_compatible', 'AR-ref', 'ar'), ('tpe_quality', 'TPE', 'cls'),
         ('motpe_quality', 'MOTPE', 'cls'), ('autopareto_292', 'GPT A22', 'gpt'), ('autopareto_294', 'GPT A24', 'gpt'),
         ('autopareto_299', 'GPT A25', 'gpt')]
OPUS = [('23', 'Opus A23'), ('24', 'Opus A24'), ('27', 'Opus A27')]
MINUS = '−'


def load():
    D = read_json(CR_DATA / 'cr_v2.json')
    S = read_json(CR_DATA / 'cr_stats.json')
    R = p2_raw_decode(D)
    return D, S, R


# ============================================================================ Figure 1: one candidate comparison
def compare_data(D, S, R):
    """One row per candidate.  Quality: held-out BPB after exactly 201,326,592 training tokens (panel seeds 3-5 for
    the original recipes, retraining seeds 6-8 for Opus's).  Training throughput from the same runs (steady-state
    tokens/s).  Decode and memory: benchmark of 1,024 input and 128 generated tokens only (registered baseline and
    Opus's recipes); no value is drawn for a recipe that was never run on it."""
    t1 = S['table1']
    m = D['phase5']['matched_tokens']['recipes']
    rows = []
    for key, name, kind in PANEL:
        x = t1[key]
        rows.append({'name': name, 'kind': kind, 'params_M': x['params'] / 1e6, 'seeds': '3-5',
                     'heldout': x['heldout_bpb']['mean'], 'heldout_seeds': x['heldout_bpb']['values'],
                     'train_tok_s': x['tokens'] / x['steady_s']['mean'], 'decode': None, 'mem': None,
                     'source': 'cr_stats.json table1 (201.3M-token panel, seeds 3-5)'})
    base_vals = [v for a in R for v in R[a]['baseline_values']]
    rows[0].update({'decode': statistics.median(base_vals), 'decode_lo': min(base_vals), 'decode_hi': max(base_vals),
                    'mem': D['agents']['baseline']['peak_mib'],
                    'source_decode': 'baseline windows bracketing each Opus recipe in the fresh-A40 re-measurement '
                                     '(raw repeat records); memory: registered baseline, same benchmark'})
    for a, name in OPUS:
        x = m[f'OPUS-A{a}']
        rows.append({'name': name, 'kind': 'opus', 'params_M': x['params'] / 1e6, 'seeds': '6-8',
                     'heldout': x['heldout']['mean'], 'heldout_seeds': x['heldout']['values'],
                     'train_tok_s': x['tok_s'], 'decode': R[a]['decode_tok_s'], 'speedup': R[a]['decode_x'],
                     'mem': D['agents']['opus']['winners'][a]['peak_mib'],
                     'source': 'cr_v2.json phase5.matched_tokens (201.3M tokens, seeds 6-8); decode and memory: raw '
                               'repeat records of the fresh-A40 re-measurement (median of 3 repeats)'})
    by = {r['name']: r for r in rows}
    ar, g22 = by['AR-ref'], by['GPT A22']
    _need(abs(ar['train_tok_s'] - 289195) < 30 and abs(g22['train_tok_s'] - 522900) < 60, 'panel throughput')
    _need([round(by[n]['decode']) for n in ('Opus A23', 'Opus A24', 'Opus A27')] == [4570, 4578, 4592], 'Opus rates')
    _need(round(by['Opus A24']['mem']) == 97 and round(by['Baseline']['mem']) == 337, 'memory')
    return {'rows': rows, 'train_ratio_g22_ar': g22['train_tok_s'] / ar['train_tok_s'],
            'opus_train_ratio_ar': [by[n]['train_tok_s'] / ar['train_tok_s'] for n in ('Opus A23', 'Opus A24', 'Opus A27')],
            'e18_g22_heldout': m['AP-Q']['heldout']['mean'], 'e18_ar_heldout': m['AR-ref']['heldout']['mean']}


def fig_compare(D, S, R):
    C = compare_data(D, S, R)
    rows = C['rows']
    style(1.0)
    # vertical layout: a small gap between approaches
    gaps = {'AR-ref': 0.35, 'TPE': 0.35, 'GPT A22': 0.35, 'Opus A23': 0.35}
    ys, y = [], 0.0
    for r in rows:
        y += gaps.get(r['name'], 0.0)
        ys.append(y)
        y += 1.0
    ymax = ys[-1]
    fig = plt.figure(figsize=(5.5, 2.28))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 0.92, 1.28, 0.74], wspace=0.12, left=0.19, right=0.988,
                          top=0.755, bottom=0.115)
    axs = [fig.add_subplot(gs[0, i]) for i in range(4)]
    fs, fsn = 5.9, 5.4
    for ax in axs:
        ax.set_ylim(ymax + 0.6, -0.6)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='x', length=2, pad=1.5, labelsize=5.6)
        ax.grid(axis='x', color=GRID, lw=0.45, zorder=0)
        ax.set_axisbelow(True)
        for i, yy in enumerate(ys):
            if i % 2 == 0:
                ax.axhspan(yy - 0.5, yy + 0.5, color='#f6f6f4', zorder=0, lw=0)

    def lab(ax, x, yy, txt, extra=None, ecol=INK, bold_extra=False):
        t = ax.annotate(txt, (x, yy), xytext=(2.2, 0), textcoords='offset points', ha='left', va='center',
                        fontsize=fs, color=INK, zorder=6)
        if extra:
            fig.canvas.draw()
            bb = t.get_window_extent(fig.canvas.get_renderer())
            x1 = ax.transData.inverted().transform((bb.x1, 0))[0]
            ax.annotate(extra, (x1, yy), xytext=(2.0, 0), textcoords='offset points', ha='left', va='center',
                        fontsize=fs if bold_extra else fsn, color=ecol, zorder=6,
                        fontweight='bold' if bold_extra else 'normal')

    a0 = axs[0]
    x0f = a0.get_position().x0
    for r, yy in zip(rows, ys):
        yf = fig.transFigure.inverted().transform(a0.transData.transform((0, yy)))[1]
        fig.add_artist(Rectangle((0.012, yf - 0.024), 0.0045, 0.048, transform=fig.transFigure,
                                 color=COL[r['kind']], lw=0))
        fig.text(0.024, yf, r['name'], ha='left', va='center', fontsize=6.4,
                 color=INK, fontweight='bold' if r['name'] in ('GPT A22', 'Opus A23') else 'normal')
        fig.text(x0f - 0.006, yf, f"{r['params_M']:.1f}M", ha='right', va='center', fontsize=5.8, color=INK2)
    # (1) held-out BPB after 201.3M tokens
    ax = axs[0]
    ax.set_xlim(1.088, 1.222)
    ax.set_xticks([1.10, 1.14, 1.18])
    for r, yy in zip(rows, ys):
        mk = 'D' if r['seeds'] == '6-8' else 'o'
        for v in r['heldout_seeds']:
            ax.scatter([v], [yy], s=7, marker=mk, facecolors='none', edgecolors=COL[r['kind']], linewidths=0.5, zorder=3)
        ax.scatter([r['heldout']], [yy], s=17 if mk == 'o' else 14, marker=mk, color=COL[r['kind']], zorder=4,
                   edgecolors='white', linewidths=0.4)
        lab(ax, max([r['heldout']] + r['heldout_seeds']), yy, f"{r['heldout']:.4f}")
    # (2) training throughput
    ax = axs[1]
    ax.set_xlim(0, 1340)
    ax.set_xticks([0, 400, 800, 1200])
    for r, yy in zip(rows, ys):
        k = r['train_tok_s'] / 1e3
        ax.barh(yy, k, height=0.6, color=COL[r['kind']], zorder=3, lw=0)
        extra = f"{C['train_ratio_g22_ar']:.2f}× AR-ref" if r['name'] == 'GPT A22' else None
        lab(ax, k, yy, f'{k:.0f}', extra, BLUE)
    # (3) decode throughput, 1,024 in / 128 out; (4) peak inference memory, same benchmark
    ax, mx = axs[2], axs[3]
    ax.set_xlim(0, 7700)
    ax.set_xticks([0, 2500, 5000])
    ax.set_xticklabels(['0', '2,500', '5,000'])
    mx.set_xlim(0, 455)
    mx.set_xticks([0, 150, 300])
    for r, yy in zip(rows, ys):
        if r['decode'] is None:
            continue
        ax.barh(yy, r['decode'], height=0.6, color=COL[r['kind']], zorder=3, lw=0)
        if 'decode_lo' in r:
            ax.plot([r['decode_lo'], r['decode_hi']], [yy, yy], color=INK, lw=0.8, zorder=4, solid_capstyle='butt')
            lab(ax, r['decode_hi'] + 60, yy, f"{r['decode_lo']:.0f}–{r['decode_hi']:.0f}", 'baseline runs', MUTED)
        else:
            lab(ax, r['decode'], yy, f"{r['decode']:,.0f}", f"{r['speedup']:.1f}×", VIOLET, bold_extra=True)
        mx.barh(yy, r['mem'], height=0.6, color=COL[r['kind']], zorder=3, lw=0)
        lab(mx, r['mem'], yy, f"{r['mem']:.0f}")
    # the recipes never run on this benchmark: one note, no per-cell placeholders
    blank = [yy for r, yy in zip(rows, ys) if r['decode'] is None]
    y0, y1 = min(blank), max(blank)
    for a_ in (ax, mx):
        a_.add_patch(Rectangle((0, y0 - 0.42), 1, y1 - y0 + 0.84, transform=a_.get_yaxis_transform(),
                               facecolor='white', edgecolor='none', zorder=1))
    bb2, bb3 = ax.get_position(), mx.get_position()
    xc = (bb2.x0 + bb3.x1) / 2
    yc = ax.transData.transform((0, (y0 + y1) / 2))[1]
    yc = fig.transFigure.inverted().transform((0, yc))[1]
    fig.text(xc, yc, 'Not run on this benchmark.\nTheir decode speed and memory on the\n'
             'original 256-token benchmark are in the appendix.',
             ha='center', va='center', fontsize=5.6, color=MUTED, style='italic', linespacing=1.25)
    # column headers
    heads = [('Held-out BPB ↓', 'after 201.3M\ntraining tokens'),
             ('Training throughput ↑', 'thousand\ntokens/s'),
             ('Decode throughput ↑', 'tokens/s, batch 1; 1,024-token\nprompt, 128 generated tokens'),
             ('Peak memory ↓', 'MiB during inference,\nsame benchmark')]
    for a_, (h1, h2) in zip(axs, heads):
        b = a_.get_position()
        fig.text((b.x0 + b.x1) / 2, 0.985, h1, ha='center', va='top', fontsize=6.5, color=INK, fontweight='bold')
        fig.text((b.x0 + b.x1) / 2, 0.915, h2, ha='center', va='top', fontsize=5.4, color=INK2, linespacing=1.1)
    b0 = axs[0].get_position()
    fig.text(0.012, 0.985, 'Candidate', ha='left', va='top', fontsize=6.5, color=INK, fontweight='bold')
    fig.text(b0.x0 - 0.006, 0.915, 'parameters', ha='right', va='top', fontsize=5.4, color=INK2)
    hand = [Line2D([], [], ls='none', marker='o', ms=3.4, mfc=INK2, mec='white', label='mean, seeds 3–5'),
            Line2D([], [], ls='none', marker='D', ms=3.0, mfc=INK2, mec='white', label='mean, seeds 6–8'),
            Line2D([], [], ls='none', marker='o', ms=2.6, mfc='none', mec=INK2, mew=0.6, label='single seeds')]
    fig.legend(handles=hand, loc='lower left', bbox_to_anchor=(0.005, -0.01), ncol=3, frameon=False, fontsize=5.6,
               handletextpad=0.3, columnspacing=1.0, handlelength=1.0)
    fig.text(0.988, 0.012, 'Opus speed-up: × the registered baseline, measured between baseline runs',
             ha='right', va='bottom', fontsize=5.6, color=INK2)
    save(fig, 'cr_compare')
    write_json(CR_DATA / 'cr_compare.json', C)
    return C


# ============================================================================ Appendix: quality vs decode
def fig_scatter(D, S, R):
    """Held-out BPB (201.3M tokens) against decode throughput.  The two decode benchmarks get separate panels and
    x-axes; the y-axis (held-out BPB after 201.3M tokens, same shard and evaluator) is shared, with the seed group
    named in each panel.  Marker area: peak inference memory on that panel's benchmark (same scale in both)."""
    t1 = S['table1']
    C = compare_data(D, S, R)
    by = {r['name']: r for r in C['rows']}
    style(1.0)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.5, 2.25), sharey=True,
                                 gridspec_kw={'width_ratios': [1.0, 1.0], 'wspace': 0.08})
    area = lambda mib: mib * 0.42
    off = {'Baseline': (-2, -11, 'center'), 'AR-ref': (8, 0, 'left'), 'TPE': (8, -4, 'left'),
           'MOTPE': (0, 9, 'center'), 'GPT A22': (7, -8, 'left'), 'GPT A24': (-8, 0, 'right'),
           'GPT A25': (0, -9, 'center')}
    for key, name, kind in PANEL:
        x = t1[key]
        dx, hy = x['decode_tok_s'], x['heldout_bpb']
        for v, h in zip(dx['values'], hy['values']):
            ax.scatter([v], [h], s=5, facecolors='none', edgecolors=COL[kind], linewidths=0.5, zorder=3)
        ax.scatter([dx['mean']], [hy['mean']], s=area(x['peak_alloc_mib']['mean']), color=COL[kind], alpha=0.85,
                   edgecolors='white', linewidths=0.6, zorder=4)
        ox, oy, ha = off[name]
        ax.annotate(f"{name}  {x['peak_alloc_mib']['mean']:.0f} MiB", (dx['mean'], hy['mean']), xytext=(ox, oy),
                    textcoords='offset points', ha=ha, va='center', fontsize=5.9, color=INK,
                    fontweight='bold' if name in ('AR-ref', 'GPT A22') else 'normal')
    ax.set_xlim(130, 600)
    ax.set_xticks([200, 300, 400, 500])
    ax.set_xlabel('Decode tokens/s (higher is better)')
    ax.set_ylabel('Held-out BPB after 201.3M tokens\n(lower is better)')
    ax.set_title('(a) Original benchmark: 256-token prompt,\n256 generated. BPB: panel seeds 3–5',
                 loc='left', fontsize=6.4)
    ax.grid(color=GRID, lw=0.45)
    # (b) extended benchmark, log x: the registered baseline's bracketing runs and Opus's recipes
    base = by['Baseline']
    bx.axvspan(base['decode_lo'], base['decode_hi'], color='#e6e5e1', zorder=1, lw=0)
    bx.text((base['decode_lo'] * base['decode_hi']) ** 0.5, 1.196, 'baseline\nruns', ha='center', va='top',
            fontsize=5.6, color=INK2, linespacing=1.0)
    for name, col, ls in (('AR-ref', ORANGE, (0, (4, 2))), ('GPT A22', BLUE, (0, (1.5, 1.5)))):
        h = C['e18_ar_heldout'] if name == 'AR-ref' else C['e18_g22_heldout']
        bx.axhline(h, color=col, lw=0.8, ls=ls, zorder=1)
        bx.text(118, h + 0.0009, f'{name}, seeds 6–8: {h:.4f} (decode not run on this benchmark)', fontsize=5.5,
                color=INK2, va='bottom', ha='left')
    pos = {'Opus A24': (-14, 11, 'right'), 'Opus A23': (-16, 0, 'right'), 'Opus A27': (-14, -9, 'right')}
    for name in ('Opus A23', 'Opus A24', 'Opus A27'):
        r = by[name]
        bx.scatter([r['decode']], [r['heldout']], s=area(r['mem']), color=VIOLET, alpha=0.55, edgecolors=INK,
                   linewidths=0.5, zorder=4)
        ox, oy, ha = pos[name]
        bx.annotate(f"{name}  {r['mem']:.0f} MiB, {r['speedup']:.1f}×", (r['decode'], r['heldout']), xytext=(ox, oy),
                    textcoords='offset points', ha=ha, va='center', fontsize=5.9, color=INK,
                    fontweight='bold' if name == 'Opus A23' else 'normal',
                    arrowprops={'arrowstyle': '-', 'lw': 0.45, 'color': INK2, 'shrinkA': 0, 'shrinkB': 3})
    bx.set_xscale('log')
    bx.set_xlim(100, 9000)
    bx.set_xticks([150, 500, 1500, 4500])
    bx.set_xticklabels(['150', '500', '1,500', '4,500'])
    bx.minorticks_off()
    bx.set_xlabel('Decode tokens/s (log scale)')
    bx.set_title('(b) Extended benchmark: 1,024-token prompt,\n128 generated. BPB: retraining seeds 6–8',
                 loc='left', fontsize=6.4)
    bx.grid(color=GRID, lw=0.45)
    ax.set_ylim(1.088, 1.202)
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    hand = [plt.scatter([], [], s=area(m), color=GRAY, alpha=0.6, edgecolors='white', label=f'{m} MiB')
            for m in (100, 200, 300)]
    ax.legend(handles=hand, loc='lower right', frameon=False, fontsize=5.6, title='marker area: memory',
              title_fontsize=5.6, handletextpad=0.3, labelspacing=0.9, borderaxespad=0.3)
    save(fig, 'cr_scatter')


# ============================================================================ Figure 2: three research histories
def _q(nodes, a, frag):
    """A quoted fragment must occur verbatim in that attempt's record."""
    _need(frag in ' '.join(nodes[a]['text'].split()), (a, frag))
    return '“' + frag + '”'


def lineage3_data():
    """Rows of the three main-text histories.  Every node is a real attempt; every drawn edge is its recorded parent
    link; every quote is asserted verbatim against the attempt record; every number comes from the record."""
    sgn = lambda v, nd=3: f'{v:+.{nd}f}'.replace('-', MINUS)
    # ---------------- AutoResearch (GPT-5.6), quality only, seed 42: 50 attempts + 5-attempt follow-up (D1-D5)
    ar, ar_b = track_ar()
    _need(_path(ar, 55) == [4, 5, 6, 9, 13, 15, 16, 19, 21, 48, 51, 52, 55], 'AR path')
    g = [a for a in range(22, 48)]
    _need(all(ar[a]['parent'] == 21 for a in g) and ar[48]['parent'] == 21, 'AR #22-#48 children of #21')
    gfail = sum(1 for a in g if ar[a]['state'] == 'failed')
    _need(gfail == 22 and 'Halving total batch size again to 65,536' in ar[41]['text'] and ar[41]['state'] == 'failed', 'AR #41')
    ar_fail = sum(1 for a in ar if a <= 50 and ar[a]['state'] == 'failed')
    _need(ar_fail == 31, ar_fail)
    AR = {'key': 'ar', 'nodes': ar, 'base_bpb': ar_b['val_bpb'],
          'head': 'AutoResearch, GPT-5.6', 'sub': 'BPB only; seed 42; 50 + 5 attempts',
          'rows': [
              {'a': 0, 'label': 'start: 8 layers, 50.3M', 'label2': f"val {ar_b['val_bpb']:.3f}"},
              {'a': 4, 'label': '#4 short attention windows',
               'quote': _q(ar, 4, 'converting redundant attention compute into more training tokens')},
              {'a': 5, 'label': '#5 batch halved', 'label2': f"val {ar[5]['val_bpb']:.3f}", 'quote': _q(ar, 5, 'twice as many optimizer updates within the fixed training time')},
              {'a': 6, 'label': '#6 halved again', 'label2': f"val {ar[6]['val_bpb']:.3f}", 'hi': True},
              {'a': 9, 'label': '#9 longer LR warm-down', 'label2': '#13 embedding LR', 'also': [13]},
              {'a': 15, 'label': '#15, #16 output-head LR', 'also': [16]},
              {'a': 19, 'label': '#19 scalar LR', 'label2': ['#21 weight decay', f"val {ar[21]['val_bpb']:.4f}"], 'also': [21]},
              {'a': 'g', 'group': g, 'parent': 21, 'lane': 1,
               'label': f'#22 to #47: 26 tries from #21, incl. batch halved again (#41); {gfail} failed'},
              {'a': 48, 'label': '#48 six Newton-Schulz steps'},
              {'a': 51, 'label': 'D1, D2 final LR floor (failed a cache check later found faulty)', 'also': [52]},
              {'a': 55, 'label': 'D5 LR floor = AR-ref', 'label2': f"val {ar[55]['val_bpb']:.4f}",
               'final': True},
          ],
          'note': f'{ar_fail} of the 50 attempts failed, most at a cache check later found faulty; model size never changed'}
    # ---------------- AutoPareto (GPT-5.6), seed 0, 25 attempts
    ap, ap_b = track_apq()
    _need(_path(ap, 22) == [1, 4, 6, 7, 8, 18, 19, 20, 22], 'GPT path')
    _need([ap[a]['parent'] for a in (9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 23, 24, 25)] ==
          [8, 9, 7, 7, 12, 13, 13, 13, 8, 20, 22, 22, 24], 'GPT side parents')
    _need(ap[17]['state'] == 'failed' and ap[14]['state'] == 'dominated' and ap[16]['state'] == 'dominated', 'GPT states')
    _need(not any(re.search(r'build_inference_adapter|CUDA graph|cuda_graph|torch.compile', n['text']) for n in ap.values()),
          'GPT: no inference-code hypothesis')
    ap_dec = ap[22]['decode'] / ap_b['inference_tokens_per_second']
    GP = {'key': 'gpt', 'nodes': ap, 'base_bpb': ap_b['val_bpb'],
          'head': 'AutoPareto, GPT-5.6', 'sub': 'BPB, speed, memory; seed 0; 25 attempts',
          'rows': [
              {'a': 0, 'label': 'start: 8 layers, 50.3M',
               'label2': f"val {ap_b['val_bpb']:.3f}; {ap_b['total_training_tokens'] / 1e6:.0f}M tokens"},
              {'a': 1, 'label': f"A1 batch ¼: val {ap[1]['val_bpb']:.3f}",
               'quote': _q(ap, 1, 'roughly four times as many optimizer updates'), 'hi': True},
              {'a': 4, 'label': 'A4 depth 8 to 7', 'quote': _q(ap, 4, 'materially improving cached decode speed')},
              {'a': 6, 'label': 'A6 depth 6, width 384'},
              {'a': 7, 'label': f"A7 depth 5: val {ap[7]['val_bpb']:.4f}", 'label2': f"{ap[7]['tokens'] / 1e6:.0f}M tokens"},
              {'a': 8, 'label': f"A8 depth 4, width 256: {sgn(ap[8]['val_bpb'] - ap[7]['val_bpb'])}",
               'quote': _q(ap, 8, 'the smaller model processes more tokens'), 'bad': True},
              {'a': 9, 'label': 'A9, A10 width 320, 384', 'also': [10], 'lane': 1},
              {'a': 'g', 'group': [11, 12, 13, 14, 15, 16], 'parent': 7, 'lane': 2,
               'label': 'A11 to A16: learning rates'},
              {'a': 17, 'label': 'A17 depth 3', 'label2': 'infrastructure crash', 'lane': 1},
              {'a': 18, 'label': f"A18 width 512: val {ap[18]['val_bpb']:.4f}",
               'quote': _q(ap, 18, 'recover enough validation quality'), 'hi': True},
              {'a': 19, 'label': 'A19 matrix LR', 'label2': 'A20 embedding LR', 'also': [20]},
              {'a': 21, 'label': 'A21 embedding LR', 'lane': 1},
              {'a': 22, 'label': 'A22 head LR = GPT A22',
               'label2': f"val {ap[22]['val_bpb']:.4f}; decode {ap_dec:.1f}×", 'final': True},
              {'a': 23, 'label': 'A23 two KV heads', 'lane': 1},
              {'a': 24, 'label': 'A24 depth 3 = GPT A24 (17.9M)', 'lane': 1, 'ring': True},
              {'a': 25, 'label': 'A25 width 512 = GPT A25 (26.2M)', 'lane': 1, 'ring': True},
          ],
          'note': 'no attempt wrote its own inference code (0 of 25; 0 of 75 across seeds 0, 1, 42)'}
    # ---------------- AutoPareto (Claude Opus 5.5), seed 3, 27 attempts
    op = track_extended(OPUS_ROOT, 'Opus')
    R23 = read_json(CR_DATA / 'cr_v2.json')['agents']['opus']['winners']['23']['decode_x']
    OB = read_json(CR_DATA / 'cr_v2.json')['agents']['baseline']
    OR = {r['attempt']: r for r in read_json(CR_DATA / 'cr_v2.json')['agents']['opus']['rows']}
    _need(abs(OR[5]['val_bpb'] - op[5]['val_bpb']) < 1e-12, 'Opus rows agree')
    _need(round(R23, 1) == 31.8, R23)
    _need(_path(op, 23) == [1, 3, 4, 5, 7, 8, 10, 11, 13, 14, 17, 18, 19, 20, 21, 22, 23], 'Opus path')
    _need([op[a]['parent'] for a in (2, 6, 9, 12, 15, 16, 24, 25, 26, 27)] == [1, 5, 8, 11, 14, 14, 23, 23, 23, 26], 'Opus side')
    x = lambda a: f"{op[a]['speedup']:.1f}×"
    _need([round(op[a]['speedup'], 1) for a in (4, 10, 14, 18, 19, 22)] == [1.9, 8.6, 16.8, 24.7, 29.6, 32.0], 'Opus speeds')
    _need(round(op[7]['peak_mib'] - op[8]['peak_mib']) == 52, 'A8 memory')
    OP = {'key': 'opus', 'nodes': op, 'squares': True,
          'head': 'AutoPareto, Claude Opus 5.5', 'sub': 'BPB, speed, memory; seed 3; 27 attempts',
          'bands': [(1, 7, 'shrinks model and batch'), (8, 22, 'rewrites the decode code'), (23, 27, 'frontier')],
          'rows': [
              {'a': 0, 'label': 'start: 8 layers, 50.3M', 'label2': f"val {OB['val_bpb']:.3f}; {OB['tokens'] / 1e6:.0f}M tokens"},
              {'a': 1, 'label': 'A1 batch halved', 'quote': _q(op, 1, 'optimizer-update-limited, not token-limited')},
              {'a': 2, 'label': 'A2 depth 6: run failed', 'lane': 1},
              {'a': 3, 'label': f'A3, A4 depth 6, then 4: decode {x(4)}', 'also': [4], 'hi': True},
              {'a': 5, 'label': f"A5 depth 3: {x(5)}; {OR[5]['tokens'] / 1e6:.0f}M tokens"},
              {'a': 6, 'label': f"A6 depth 2: {x(6)}, val {sgn(op[6]['val_bpb'] - op[5]['val_bpb'])}", 'lane': 1, 'bad': True},
              {'a': 7, 'label': f"A7 from A5, batch halved: val {op[7]['val_bpb']:.4f}",
               'quote': _q(op, 7, 'the quality prediction was falsified')},
              {'a': 8, 'label': f"A8 own inference code: {MINUS}{op[7]['peak_mib'] - op[8]['peak_mib']:.0f} MiB",
               'quote': _q(op, 8, 'CUDA-graph decode will be the next isolated speed change')},
              {'a': 9, 'label': 'A9 graph: accounting check failed', 'lane': 1},
              {'a': 10, 'label': f'A10 CUDA graph replay: {x(10)}', 'hi': True,
               'quote': _q(op, 10, 'the only lever that can raise speed without a quality cost')},
              {'a': 11, 'label': 'A11 graph memory'},
              {'a': 12, 'label': 'A12 batch halved again: worse', 'lane': 1},
              {'a': 13, 'label': f'A13 bf16 weights: {x(13)}'},
              {'a': 14, 'label': f'A14 compiled kernel fusion: {x(14)}', 'hi': True},
              {'a': 15, 'label': 'A15, A16 LR, KV heads: worse', 'also': [16], 'lane': 1},
              {'a': 17, 'label': 'A17 shorter windows'},
              {'a': 18, 'label': f'A18 ring K/V cache: {x(18)}', 'hi': True},
              {'a': 19, 'label': f'A19 custom decode attention: {x(19)}', 'hi': True},
              {'a': 20, 'label': 'A20, A21 QK gain; prefill logits', 'also': [21]},
              {'a': 22, 'label': f'A22 fused decode step: {x(22)}', 'hi': True},
              {'a': 23, 'label': 'A23 token shift = Opus A23',
               'label2': f"val {op[23]['val_bpb']:.4f}; {R23:.1f}× re-measured", 'final': True},
              {'a': 24, 'label': f"A24 shared table = Opus A24: {op[24]['peak_mib']:.0f} MiB", 'lane': 1, 'ring': True},
              {'a': 25, 'label': 'A25 output gate: worse', 'lane': 1},
              {'a': 26, 'label': 'A26 prefill attention kernel', 'lane': 2},
              {'a': 27, 'label': f"A27 int8 tables = Opus A27: {op[27]['peak_mib']:.0f} MiB", 'lane': 2, 'ring': True},
          ],
          'note': None}
    return [AR, GP, OP]


def _fit_wrap(ax, text, fs, x0, style_='italic'):
    """Wrap text so every line fits between axes-fraction x0 and the right edge of the axes."""
    fig = ax.figure
    rnd = fig.canvas.get_renderer()
    avail = ax.get_window_extent(rnd).width * (0.985 - x0)
    for w in range(70, 12, -1):
        lines = textwrap.wrap(text, w, break_on_hyphens=False)
        ok = True
        for ln in lines:
            t = ax.text(0, 0, ln, fontsize=fs, style=style_)
            ok = ok and t.get_window_extent(rnd).width <= avail
            t.remove()
            if not ok:
                break
        if ok:
            return lines
    raise AssertionError(text)


LP = 0.9           # line pitch inside a row (y units)
LX = [0.06, 0.098, 0.136]   # lane x (axes fraction)
XL = 0.176                  # label start


def prepare_history(ax, T, fs=5.5):
    """Wrap every label and quote to the column width; return the number of text lines and rows."""
    nl = 0
    for r in T['rows']:
        lines = _fit_wrap(ax, r['label'], fs, XL, 'normal')
        r['_nbold'] = len(lines)
        for l2 in ([r['label2']] if isinstance(r.get('label2'), str) else r.get('label2', [])):
            lines += _fit_wrap(ax, l2, fs, XL, 'normal')
        r['_nlab'] = len(lines)
        r['_nq'] = 0
        if r.get('quote'):
            ql = _fit_wrap(ax, r['quote'], fs - 0.35, XL)
            lines += ql
            r['_nq'] = len(ql)
        if r.get('group'):
            lines.append('')            # glyph strip
        r['_lines'] = lines
        nl += len(lines)
    T['_nlines'], T['_nrows'] = nl, len(T['rows'])
    # the column note is laid out inside the column's own height budget (revision 13: it no longer overlaps the legend)
    T['_nnote'] = len(_fit_wrap(ax, T['note'], fs - 0.4, XL)) if T.get('note') else 0
    return nl, len(T['rows'])


def draw_history(ax, T, gap, ytot, fs=5.5):
    """One research history as a parent-linked tree, read top to bottom in attempt order.  Lane 0 is the recorded
    path to the final representative; lanes 1-2 are branches."""
    col, lcol = COL[T['key']], LIGHT[T['key']]
    nodes = T['nodes']
    y, Y = 0.0, {}
    for r in T['rows']:
        r['_y'] = y
        Y[r['a']] = y
        for k in r.get('also', []) + r.get('group', []):
            Y[k] = y
        y += len(r['_lines']) * LP + gap
    ax.set_xlim(0, 1)
    ax.set_ylim(ytot, -2.6)
    ax.axis('off')
    lane = {}
    for r in T['rows']:
        lane[r['a']] = r.get('lane', 0)
        for k in r.get('also', []) + r.get('group', []):
            lane[k] = r.get('lane', 0)
    # phase bands, labelled in the left gutter
    for a0, a1, t in T.get('bands', []):
        r1 = next(rr for rr in T['rows'] if rr['a'] == a1)
        y0, y1 = Y[a0] - 0.5 * gap - 0.35, r1['_y'] + len(r1['_lines']) * LP - 0.35 + 0.5 * gap
        ax.add_patch(Rectangle((0.0, y0), 1.0, y1 - y0, color='#f1effb' if 'decode' in t else '#f6f6f3', lw=0, zorder=0))
        ax.text(0.016, (y0 + y1) / 2, t, ha='center', va='center', fontsize=fs - 0.3, color=INK2, style='italic',
                rotation=90, linespacing=1.0, zorder=1)
    # edges: recorded parent -> child (both drawn)
    for r in T['rows']:
        if r['a'] == 0:
            continue
        par = r['parent'] if r.get('group') else nodes[r['a']]['parent']
        _need(par in Y, f"parent of {r['a']} is drawn")
        for k in r.get('also', []):
            _need(nodes[k]['parent'] in Y, f'parent of {k} is drawn')
        x0, y0 = LX[lane.get(par, 0)], Y[par]
        x1, y1 = LX[lane[r['a']]], r['_y']
        on = lane[r['a']] == 0 and lane.get(par, 0) == 0
        if x1 == x0:
            xs, ys_ = [x0, x1], [y0, y1]
        else:
            xs, ys_ = [x0, x0, x1], [y0, y1 - 0.5, y1]
        ax.plot(xs, ys_, color=col if on else '#bdbcb8', lw=1.5 if on else 0.6, zorder=2, solid_capstyle='round',
                solid_joinstyle='round')
    # nodes and text
    for r in T['rows']:
        a, yy, xx = r['a'], r['_y'], LX[lane[r['a']]]
        if a == 0:
            ax.scatter([xx], [yy], s=20, color=GRAY, zorder=4, edgecolors='white', linewidths=0.5)
        elif r.get('group'):
            ax.scatter([xx], [yy], s=10, facecolors='white', edgecolors='#8f8e8a', linewidths=0.6, zorder=4)
        else:
            st = nodes[a]['state']
            sq = T.get('squares') and nodes[a]['category'].startswith('inference')
            if st == 'failed':
                ax.scatter([xx], [yy], marker='x', s=13, color='#8f8e8a' if lane[a] else col, linewidths=0.9, zorder=4)
            elif lane[a] == 0:
                ax.scatter([xx], [yy], s=15 if sq else 19, marker='s' if sq else 'o', color=col, zorder=4,
                           edgecolors='white', linewidths=0.45)
            else:
                ax.scatter([xx], [yy], s=11 if sq else 13, marker='s' if sq else 'o',
                           facecolors=lcol if r.get('ring') else 'white',
                           edgecolors=col if r.get('ring') else '#8f8e8a', linewidths=0.7, zorder=4)
            for j, k in enumerate(r.get('also', [])):
                stk = nodes[k]['state']
                kx = xx + 0.026 * (j + 1)
                if stk == 'failed':
                    ax.scatter([kx], [yy], marker='x', s=10, color=col if lane[a] == 0 else '#8f8e8a', linewidths=0.7, zorder=4)
                else:
                    ax.scatter([kx], [yy], marker='o', s=10, facecolors=col if lane[a] == 0 else 'white',
                               edgecolors=col if lane[a] == 0 else '#8f8e8a', linewidths=0.6, zorder=4)
        if r.get('final') or r.get('ring'):
            ax.scatter([xx], [yy], s=52, facecolors='none', edgecolors=col, linewidths=0.9, zorder=5)
        for i, line in enumerate(r['_lines']):
            if not line:
                continue
            is_q = i >= r['_nlab']
            ax.text(XL, yy + i * LP, line, ha='left', va='center', fontsize=fs - (0.35 if is_q else 0),
                    color=(INK2 if is_q else (INK if lane[a] == 0 else INK2)), style='italic' if is_q else 'normal',
                    fontweight='bold' if (r.get('final') and i < r['_nbold']) else 'normal', zorder=6)
        if r.get('group'):
            gy = yy + (len(r['_lines']) - 1) * LP
            for j, k in enumerate(r['group']):
                gx = XL + 0.006 + j * 0.0215
                if nodes[k]['state'] == 'failed':
                    ax.scatter([gx], [gy], marker='x', s=6, color='#8f8e8a', linewidths=0.6, zorder=4)
                else:
                    ax.scatter([gx], [gy], marker='o', s=5, facecolors='white', edgecolors='#8f8e8a', linewidths=0.5, zorder=4)
    ax.text(0.0, -2.45, T['head'], ha='left', va='top', fontsize=HEAD3, color=col, fontweight='bold')
    ax.text(0.0, -1.45, T['sub'], ha='left', va='top', fontsize=SUB3, color=INK2)
    if T.get('note'):
        for i, ln in enumerate(_fit_wrap(ax, T['note'], fs - 0.4, XL)):
            ax.text(XL, y + 0.1 + i * LP, ln, ha='left', va='top', fontsize=fs - 0.4, color=MUTED, style='italic')
    return y


FS3 = 6.2      # revision 13: labels 5.65 -> 6.2 pt, quotes 5.3 -> 5.85 pt (Figure 2 readability)
HEAD3, SUB3, G03, H3 = 7.2, 5.7, 0.2, 4.28


def fig_lineage3():
    T = lineage3_data()
    style(1.0)
    fig = plt.figure(figsize=(5.5, H3))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.82, 0.98, 1.26], wspace=0.035, left=0.004, right=0.996, top=0.985,
                          bottom=0.05)
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
    fig.canvas.draw()
    for ax, t in zip(axs, T):
        prepare_history(ax, t, fs=FS3)
    G0 = G03
    need = lambda t: (t['_nlines'] + t['_nnote']) * LP + (0.4 if t['_nnote'] else 0)
    tot = max(need(t) + t['_nrows'] * G0 for t in T)
    for ax, t in zip(axs, T):
        gap = (tot - need(t)) / t['_nrows']
        draw_history(ax, t, gap, tot + 0.3, fs=FS3)
    handles = [Line2D([], [], color=INK2, lw=1.5, marker='o', ms=3.4, mec='white', label='recorded path to the final model'),
               Line2D([], [], color='#b9b8b4', lw=0.6, marker='o', ms=3.0, mfc='white', mec='#8f8e8a', label='branch'),
               Line2D([], [], ls='none', marker='x', ms=3.4, mec='#8f8e8a', label='failed attempt'),
               Line2D([], [], ls='none', marker='s', ms=3.2, mfc=VIOLET, mec='white', label='own inference code'),
               Line2D([], [], ls='none', marker='o', ms=6.0, mfc='none', mec=INK2, label='final representative or frontier')]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=6.1, frameon=False, handlelength=1.6,
               bbox_to_anchor=(0.5, -0.005), columnspacing=1.2, handletextpad=0.4)
    # every label must stay inside its own column
    fig.canvas.draw()
    rnd = fig.canvas.get_renderer()
    bad = []
    for i, ax in enumerate(axs):
        right = axs[i + 1].get_window_extent(rnd).x0 if i < 2 else fig.bbox.x1
        for tx in ax.texts:
            if tx.get_rotation():
                continue
            bb = tx.get_window_extent(rnd)
            if bb.x1 > right + 0.5:
                bad.append(f'{tx.get_text()!r} ({bb.x1:.1f} > {right:.1f})')
    save(fig, 'cr_lineage3')
    _need(not bad, 'labels leave their column: ' + '; '.join(bad))
    out = [{'head': t['head'], 'rows': [{k: v for k, v in r.items() if not k.startswith('_')} for r in t['rows']]}
           for t in T]
    write_json(CR_DATA / 'cr_lineage3.json', out)
    return T


# ============================================================================ Table 1 and per-seed appendix table
BENCH = [('heldout', 'heldout_climbmix_bpb/bits_per_byte', 'heldout_climbmix_bpb.bits_per_byte'),
         ('arc_easy', 'arc_easy/accuracy', 'arc_easy.accuracy'), ('piqa', 'piqa/accuracy', 'piqa.accuracy'),
         ('lambada', 'lambada_openai/exact_match', 'lambada_openai.exact_match'),
         ('blimp', 'blimp/macro_accuracy', 'blimp.macro_accuracy')]


def bench_data(D, S):
    """Every candidate with matched-token downstream results, per seed.  Panel (seeds 3-5): cr_evidence.json
    panel_downstream, cross-checked against cr_stats.json.  Retraining (seeds 6-8): Phase 5 analysis, seed order
    checked against each recipe's by-seed held-out record."""
    E = read_json(CR_DATA / 'cr_evidence.json')
    by = {}
    for r in E['panel_downstream']:
        by.setdefault((r['model'], f"{r['task']}/{r['metric']}"), {})[r['seed']] = r['value']
    groups = {'3-5': [], '6-8': []}
    for key, name, kind in PANEL:
        row = {'name': name, 'kind': kind, 'params': S['table1'][key]['params'] / 1e6, 'seeds': [3, 4, 5]}
        for k, pk, _ in BENCH:
            vals = [by[(key, pk)][s_] for s_ in (3, 4, 5)]
            _need(abs(statistics.mean(vals) - S['downstream_means'][key][pk]['mean']) < 1e-12, (key, pk))
            row[k] = vals
        _need(max(abs(a - b) for a, b in zip(row['heldout'], S['table1'][key]['heldout_bpb']['values'])) < 1e-12, key)
        groups['3-5'].append(row)
    P = read_json(P5)['phase5a']
    br = P['downstream']['by_recipe']
    for k_, name, kind in (('AR-ref', 'AR-ref', 'ar'), ('AP-Q', 'GPT A22', 'gpt'), ('OPUS-A23', 'Opus A23', 'opus'),
                           ('OPUS-A24', 'Opus A24', 'opus'), ('OPUS-A27', 'Opus A27', 'opus')):
        rec = P['recipes'][k_]
        _need(rec['seeds'] == [6, 7, 8], k_)
        row = {'name': name, 'kind': kind, 'params': rec['parameter_count'] / 1e6, 'seeds': [6, 7, 8]}
        for k, _, qk in BENCH:
            row[k] = br[k_][qk]['values']
            _need(abs(statistics.mean(row[k]) - br[k_][qk]['mean']) < 1e-12, (k_, qk))
        _need(max(abs(row['heldout'][i] - rec['by_seed'][str(s_)]['heldout']) for i, s_ in enumerate((6, 7, 8))) < 1e-12,
              f'{k_}: downstream values are in seed order 6, 7, 8')
        groups['6-8'].append(row)
    acc_sd = [statistics.stdev(r[k]) for g in groups.values() for r in g for k in ('arc_easy', 'piqa', 'lambada', 'blimp')]
    return {'groups': groups, 'max_acc_sd_pts': 100 * max(acc_sd), 'median_acc_sd_pts': 100 * statistics.median(acc_sd)}


def _fmt(k, v):
    return f'{v:.4f}' if k == 'heldout' else f'{100 * v:.1f}'


def table_bench_main(B):
    """Main-text Table 1: one comparison table, two seed groups never mixed; bold = best displayed mean per metric
    within a seed group (ties at the displayed precision share it)."""
    out = ['% Generated by analysis/cr_r9.py (bench_data). Do not edit by hand.',
           r'\begin{tabular}{@{}lrrrrrr@{}}', r'\toprule',
           r' & Params & Held-out & ARC-Easy & PIQA & LAMBADA & BLiMP \\',
           r'Candidate & (M) & BPB\,$\downarrow$ & $\uparrow$ & $\uparrow$ & $\uparrow$ & $\uparrow$ \\']
    best = {}
    for gname, title in (('3-5', r'Seeds 3--5 (201.3M-token panel)'), ('6-8', r'Seeds 6--8 (E1 and Opus retraining)')):
        rows = B['groups'][gname]
        out += [r'\midrule', r'\multicolumn{7}{@{}l}{\emph{' + title + r'}} \\']
        for k, _, _ in BENCH:
            disp = [_fmt(k, statistics.mean(r[k])) for r in rows]
            bv = min(disp, key=float) if k == 'heldout' else max(disp, key=float)
            best[(gname, k)] = [r['name'] for r, d in zip(rows, disp) if d == bv]
        for r in rows:
            cells = []
            for k, _, _ in BENCH:
                d = _fmt(k, statistics.mean(r[k]))
                cells.append(r'\textbf{' + d + '}' if r['name'] in best[(gname, k)] else d)
            out.append(f"{r['name']} & {r['params']:.1f} & " + ' & '.join(cells) + r' \\')
    out += [r'\bottomrule', r'\end{tabular}', '']
    (GEN_CR / 'table_bench_main.tex').write_text('\n'.join(out))
    return best


def table_bench_seeds(B):
    """Appendix: every per-seed value behind Table 1."""
    out = ['% Generated by analysis/cr_r9.py (bench_data). Do not edit by hand.',
           r'\begin{tabular}{@{}llccccc@{}}', r'\toprule',
           r'Candidate & Seeds & Held-out BPB & ARC-Easy & PIQA & LAMBADA & BLiMP \\']
    for gname in ('3-5', '6-8'):
        out.append(r'\midrule')
        for r in B['groups'][gname]:
            cells = [' / '.join(_fmt(k, v) for v in r[k]) for k, _, _ in BENCH]
            out.append(f"{r['name']} & {r['seeds'][0]}, {r['seeds'][1]}, {r['seeds'][2]} & " + ' & '.join(cells) + r' \\')
    out += [r'\bottomrule', r'\end{tabular}', '']
    (GEN_CR / 'table_bench_seeds.tex').write_text('\n'.join(out))


# ============================================================================ Figure 3: equal tokens, compact
def fig_e1_compact(D):
    """Held-out BPB against training tokens on seeds 6-8: E1's equal-token budgets and 300 s runs for AR-ref and
    GPT A22, and Opus's three recipes at 201.3M tokens (seed means; bars: range over seeds)."""
    style(1.0)
    cur = D['e1']['curves']
    mt = D['phase5']['matched_tokens']['recipes']
    fig, ax = plt.subplots(figsize=(2.2, 1.78))
    fig.subplots_adjust(left=0.215, right=0.97, top=0.97, bottom=0.2)
    for rec, name, col in (('AR-ref', 'AR-ref', ORANGE), ('AP-Q', 'GPT A22', BLUE)):
        xs = [statistics.mean(cur[f'{rec}|{c}']['tokens']) / 1e6 for c in CELLS[:3]]
        ys = [statistics.mean(cur[f'{rec}|{c}']['heldout']) for c in CELLS[:3]]
        for c in CELLS[:3]:
            v = cur[f'{rec}|{c}']['heldout']
            ax.plot([statistics.mean(cur[f'{rec}|{c}']['tokens']) / 1e6] * 2, [min(v), max(v)], color=col, lw=0.6, zorder=2)
        ax.plot(xs, ys, color=col, lw=1.5, zorder=3)
        ax.scatter(xs, ys, s=13, color=col, zorder=4, edgecolors='white', linewidths=0.5)
        tx = statistics.mean(cur[f'{rec}|time_300s']['tokens']) / 1e6
        ty = statistics.mean(cur[f'{rec}|time_300s']['heldout'])
        ax.scatter([tx], [ty], marker='D', s=24, color=col, zorder=5, edgecolors=INK, linewidths=0.6)
        ax.annotate(name, (xs[-1], ys[-1]), xytext=(4, -1 if rec == 'AP-Q' else 0), textcoords='offset points',
                    va='center', ha='left', color=INK, fontsize=6.0, fontweight='bold')
    ops = [mt[f'OPUS-A{a}']['heldout']['mean'] for a in ('23', '24', '27')]
    ax.plot([201.3, 201.3], [min(ops), max(ops)], color=VIOLET, lw=2.6, solid_capstyle='round', zorder=4)
    ax.annotate('Opus A23,\nA24, A27', (201.3, statistics.mean(ops)), xytext=(4, 0), textcoords='offset points',
                va='center', ha='left', color=INK, fontsize=6.0, fontweight='bold', linespacing=1.0)
    tAR = statistics.mean(cur['AR-ref|time_300s']['tokens']) / 1e6
    tAP = statistics.mean(cur['AP-Q|time_300s']['tokens']) / 1e6
    yAR = statistics.mean(cur['AR-ref|time_300s']['heldout'])
    ax.annotate('300 s each', ((tAR * tAP) ** 0.5, yAR + 0.004), ha='center', va='bottom', fontsize=5.7, color=INK2)
    ax.set_xscale('log')
    ax.set_xticks([88.6, 157.3, 201.3])
    ax.set_xticklabels(['88.6M', '157M', '201M'])
    ax.minorticks_off()
    ax.set_xlim(80, 300)
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_ylim(1.088, 1.205)
    ax.set_xlabel('Training tokens (log scale)', fontsize=6.2)
    ax.set_ylabel('Held-out BPB (lower is better)', fontsize=6.2)
    ax.tick_params(labelsize=5.8)
    ax.grid(axis='y', color=GRID, lw=0.45)
    ax.legend(handles=[Line2D([], [], color=INK2, lw=1.2, marker='o', ms=3, mec='white', label='equal tokens'),
                       Line2D([], [], ls='none', marker='D', ms=3.4, mfc=INK2, mec=INK, mew=0.5, label='equal time (300 s)')],
              loc='upper right', frameon=False, fontsize=5.6, handlelength=1.4, handletextpad=0.35, borderaxespad=0.1)
    save(fig, 'cr_e1_compact')


# ============================================================================ Figure 4: how Opus reached ~32x
STAGES = [(0, 'Starting model (8 layers)'), (4, 'A4 four layers'), (5, 'A5 three layers'), (10, 'A10 CUDA graph replay'),
          (14, 'A14 compiled kernel fusion'), (18, 'A18 ring K/V cache'), (19, 'A19 custom decode attention'),
          (22, 'A22 fused decode step'), (23, 'A23 final recipe')]
ABL = [('V0_as_submitted', 'As submitted'), ('V5_linear_kv_cache', 'Plain K/V cache'),
       ('V3_sdpa_attention', 'Standard attention'), ('V2_no_fusion', 'No fusion'),
       ('V1_no_cuda_graph', 'No CUDA graph'), ('V4_eager_unfused', 'No graph or fusion')]


def fig_inference2(D, R):
    style(1.0)
    rows = {r['attempt']: r for r in D['agents']['opus']['rows']}
    unc = {a for a, r in rows.items() if r['quality_status'] == 'quality_valid_speed_uncertain'}
    ia = D['phase5']['ablation']
    sp, cells, b = ia['speedup'], ia['cells'], ia['baseline']
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(5.0, 1.38),
                                     gridspec_kw={'width_ratios': [0.9, 0.8, 0.8], 'wspace': 1.62})
    fs = 6.2
    # (a) the recorded path, cumulative: each attempt keeps every earlier change
    n = len(STAGES)
    for i, (a, lab) in enumerate(STAGES):
        y = n - 1 - i
        v = 1.0 if a == 0 else rows[a]['decode_x']
        inf = a >= 10
        c = VIOLET if inf else '#b3aaf0'
        ax.barh(y, v - 1, left=1, height=0.62, facecolor=c, edgecolor='white' if a in unc else c,
                hatch='//////' if a in unc else None, lw=0, zorder=3)
        ax.text(v * 1.07, y, f'{v:.1f}×', va='center', ha='left', fontsize=fs, color=INK)
    a23 = R['23']['decode_x']
    ax.barh(-1, a23 - 1, left=1, height=0.62, facecolor='white', edgecolor=VIOLET, lw=0.9, zorder=3)
    ax.text(a23 * 1.07, -1, f'{a23:.1f}×', va='center', ha='left', fontsize=fs, color=INK, fontweight='bold')
    ax.set_yticks([n - 1 - i for i in range(n)] + [-1])
    ax.set_yticklabels([l for _, l in STAGES] + ['A23 re-measured, fresh A40'], fontsize=fs)
    ax.set_ylim(-1.6, n - 0.45)
    _need([round(rows[a]['decode_x'], 1) for a, _ in STAGES[1:]] == [1.9, 2.5, 8.6, 16.8, 24.7, 29.6, 32.0, 30.8], 'stages')
    # (b) the ablation session: remove one mechanism from the submitted recipes
    m = len(ABL)
    for i, (v, lab) in enumerate(ABL):
        y = m - 1 - i
        vals = [sp[k][v]['decode'] for k in IA_MODELS]
        med = statistics.median(vals)
        bx.barh(y, med - 1, left=1, height=0.62, color=VIOLET if v == 'V0_as_submitted' else '#8f86e6', lw=0, zorder=3)
        bx.text(med * 1.07, y, f'{med:.1f}×', va='center', ha='left', fontsize=fs, color=INK,
                fontweight='bold' if v == 'V0_as_submitted' else 'normal')
    bx.set_yticks([m - 1 - i for i in range(m)])
    bx.set_yticklabels([l for _, l in ABL], fontsize=fs)
    bx.set_ylim(-0.6, m - 0.4)
    for a_ in (ax, bx):
        a_.axvline(1, color=INK2, lw=0.6, zorder=2)
        a_.set_xscale('log')
        a_.set_xlim(0.9, 110)
        a_.set_xticks([1, 3, 10, 30])
        a_.set_xticklabels(['1', '3', '10', '30'], fontsize=fs)
        a_.minorticks_off()
        a_.tick_params(axis='y', length=0, pad=2)
        a_.spines['left'].set_visible(False)
        a_.grid(axis='x', color=GRID, lw=0.45, zorder=0)
    ax.set_title('(a) Opus’s path', loc='left', fontsize=6.6)
    bx.set_title('(b) Ablations: remove one part', loc='left', fontsize=6.6)
    ax.set_xlabel('speed-up × (search records)', fontsize=fs)
    bx.set_xlabel('speed-up × (ablation session)', fontsize=fs)
    # (c) one whole generation (revision 11): drawn from the 13-recipe benchmark, the paper's primary source
    # (results/camera_ready_2026/phase6/analysis/unified_13.json, seed means; Opus = mean of A23, A24, A27). The
    # ablation session's own whole-generation ratio is still computed and checked below, and returned for the record.
    def times(v):
        if v == 'baseline':
            return b['prefill_ms'], 127 / b['decode_tps'] * 1000
        pre = statistics.median(cells[f'{k}/{v}']['prefill_ms'] for k in IA_MODELS)
        dec = statistics.median(127 / cells[f'{k}/{v}']['decode_tps'] * 1000 for k in IA_MODELS)
        return pre, dec
    for k in IA_MODELS:
        for v, _ in ABL:
            c_ = cells[f'{k}/{v}']
            derived = 128 / (c_['prefill_ms'] / 1000 + 127 / c_['decode_tps'])
            _need(abs(derived / c_['e2e_tps'] - 1) < 0.015, (k, v))
    rb = [('baseline', 'Baseline'), ('V0_as_submitted', 'Opus')]
    TA = {v: times(v) for v, _ in rb}
    e2e_ablation = sum(TA['baseline']) / sum(TA['V0_as_submitted'])
    U = json.loads((Path(__file__).resolve().parents[1] / 'results/camera_ready_2026/phase6/analysis/unified_13.json').read_text())
    SU = {s['recipe']: s for s in U['summary']}
    ops = [SU[r] for r in ('opus_a23', 'opus_a24', 'opus_a27')]
    T = {'baseline': (SU['baseline']['inf_prefill_ms_mean'], SU['baseline']['inf_e2e_ms_mean'] - SU['baseline']['inf_prefill_ms_mean']),
         'V0_as_submitted': (statistics.mean(o['inf_prefill_ms_mean'] for o in ops),
                             statistics.mean(o['inf_e2e_ms_mean'] - o['inf_prefill_ms_mean'] for o in ops))}
    for i, (v, lab) in enumerate(rb):
        y = 1 - i
        pre, dec = T[v]
        cx.barh(y, pre, height=0.5, color='#d9d7d1', edgecolor=INK2, lw=0.5, hatch='////', zorder=3)
        cx.barh(y, dec, left=pre, height=0.5, color=VIOLET if v != 'baseline' else GRAY, lw=0, zorder=3)
        cx.text(pre + dec + 18, y, f'{pre + dec:.0f} ms', va='center', ha='left', fontsize=fs, color=INK)
    preB, decB = T['baseline']
    preV, decV = T['V0_as_submitted']
    cx.text(preV + decV + 18, 0 - 0.42, f'prefill {preV:.0f} + decode {decV:.0f}', va='center', ha='left',
            fontsize=fs - 0.4, color=INK2)
    e2e_x = (preB + decB) / (preV + decV)
    _need(round(e2e_x, 1) in {round(o['inf_e2e_tok_s_mean'] / SU['baseline']['inf_e2e_tok_s_mean'], 1) for o in ops},
          'panel (c) matches the 13-recipe end-to-end ratio')
    cx.set_yticks([1, 0])
    cx.set_yticklabels([l for _, l in rb], fontsize=fs)
    cx.tick_params(axis='y', length=0, pad=2)
    cx.spines['left'].set_visible(False)
    cx.set_xlim(0, 1150)
    cx.set_xticks([0, 400, 800])
    cx.tick_params(axis='x', labelsize=fs)
    cx.set_ylim(-0.75, 2.05)
    cx.grid(axis='x', color=GRID, lw=0.45, zorder=0)
    cx.set_xlabel('ms (13-recipe test)', fontsize=fs)
    cx.set_title(f'(c) Whole generation: {e2e_x:.1f}×', loc='left', fontsize=6.6)
    cx.legend(handles=[Rectangle((0, 0), 1, 1, facecolor='#d9d7d1', edgecolor=INK2, hatch='////', lw=0.5, label='prefill, not scored'),
                       Rectangle((0, 0), 1, 1, facecolor=VIOLET, label='decode, scored')],
              loc='upper left', bbox_to_anchor=(0.0, 1.02), frameon=False, fontsize=5.8, handlelength=1.1,
              handletextpad=0.3, borderaxespad=0.0, labelspacing=0.25, ncol=1)
    save(fig, 'cr_inference2')
    return {'e2e_x': e2e_ablation, 'e2e_x_unified': e2e_x, 'dec_x': TA['baseline'][1] / TA['V0_as_submitted'][1],
            'pre_base_ms': TA['baseline'][0], 'pre_opus_ms': TA['V0_as_submitted'][0]}


# ============================================================================ Appendix: every attempt of every search
def fig_histories():
    """Every attempt of the four agent searches (one row per attempt, recorded parent links), drawn with the
    revision-8 decision-log renderer at full-page height; labels carry the revision-9 candidate names."""
    T = tracks_data()
    t_ar, t_ap, t_cx, t_op = T
    ap = t_ap['nodes']
    t_ap['labels'][22] = f"A22 head LR: GPT A22\nval {ap[22]['val_bpb']:.4f}"
    t_ap['labels'][24] = 'A24 depth 3: GPT A24'
    t_ap['labels'][25] = 'A25 width 512: GPT A25'
    t_ar['head'] = ('AutoResearch, GPT-5.6', 'quality only; seed 42;\n50 + 5 attempts')
    t_ap['head'] = ('AutoPareto, GPT-5.6', 'seed 0; 25 attempts')
    t_cx['head'] = ('AutoPareto, GPT-5.6 (later)', 'Opus’s harness; seed 3;\n22 attempts')
    t_op['head'] = ('AutoPareto, Opus 5.5', 'seed 3; 27 attempts')
    op = t_op['nodes']
    t_op['labels'][23] = f"A23 token shift: Opus A23"
    t_op['labels'][24] = f"A24 shared table: Opus A24"
    t_op['labels'][27] = f"A27 int8 tables: Opus A27"
    for t in T:
        for k, v in list(t['labels'].items()):
            _need(not re.search(r'AP-[QSF]', v), v)
    style(1.0)
    fig = plt.figure(figsize=(5.5, 7.55))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.9, 1.0, 0.98, 1.1], wspace=0.07, left=0.03, right=0.995,
                          top=0.975, bottom=0.04)
    axs = [fig.add_subplot(gs[0, i]) for i in range(4)]
    for ax, t in zip(axs, (t_ar, t_ap, t_cx, t_op)):
        draw_track(ax, t, fs=5.75)
        ax.set_ylim(28.2, -3.0)
    for y in (1, 5, 10, 15, 20, 25):
        axs[1].text(-0.005, y, str(y), ha='right', va='center', fontsize=5.4, color=MUTED)
    handles = [Line2D([], [], color=INK2, lw=1.5, marker='o', ms=3.2, label='recorded parent path'),
               Line2D([], [], color='none', marker='o', ms=2.8, mfc='white', mec='#8f8e8a', label='other attempt'),
               Line2D([], [], color='none', marker='x', ms=3.2, mec=INK2, label='failed run'),
               Line2D([], [], color='none', marker='s', ms=3.0, mfc=VIOLET, mec='white', label='own inference code'),
               Line2D([], [], color='none', marker='o', ms=5.5, mfc='none', mec=VIOLET, label='final frontier')]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=6.0, frameon=False, handlelength=1.4,
               bbox_to_anchor=(0.5, -0.003), columnspacing=1.3)
    fig.canvas.draw()
    rnd = fig.canvas.get_renderer()
    for i, ax in enumerate(axs):
        right = axs[i + 1].get_window_extent(rnd).x0 if i < 3 else fig.bbox.x1
        for tx in ax.texts:
            if not tx.get_rotation():
                _need(tx.get_window_extent(rnd).x1 <= right + 1.0, f'label leaves its column: {tx.get_text()!r}')
    save(fig, 'cr_histories')


# ============================================================================ Appendix: E1 in one figure
def fig_e1_full(D):
    style(1.0)
    pc = D['e1']['per_cell']['heldout_bpb']
    cur = D['e1']['curves']
    mt = D['phase5']['matched_tokens']['recipes']
    for c in CELLS:
        _need(pc[c]['seeds'] == [6, 7, 8] and pc[c]['n'] == 3, c)
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(5.5, 2.1),
                                     gridspec_kw={'width_ratios': [0.95, 1.35, 0.95], 'wspace': 0.95})
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.2, top=0.88)
    for rec, name, col in (('AR-ref', 'AR-ref', ORANGE), ('AP-Q', 'GPT A22', BLUE)):
        xs = [statistics.mean(cur[f'{rec}|{c}']['tokens']) / 1e6 for c in CELLS[:3]]
        ys = [statistics.mean(cur[f'{rec}|{c}']['heldout']) for c in CELLS[:3]]
        for c in CELLS[:3]:
            ax.scatter([t / 1e6 for t in cur[f'{rec}|{c}']['tokens']], cur[f'{rec}|{c}']['heldout'], s=6,
                       facecolors='none', edgecolors=col, linewidths=0.5, zorder=3)
        ax.plot(xs, ys, color=col, lw=1.4, zorder=2)
        ax.scatter(xs, ys, s=12, color=col, zorder=4, edgecolors='white', linewidths=0.5)
        tx = statistics.mean(cur[f'{rec}|time_300s']['tokens']) / 1e6
        ty = statistics.mean(cur[f'{rec}|time_300s']['heldout'])
        ax.scatter([tx], [ty], marker='D', s=22, color=col, zorder=5, edgecolors=INK, linewidths=0.6)
        ax.annotate(name, (xs[-1], ys[-1]), xytext=(7, -5 if rec == 'AP-Q' else 0), textcoords='offset points',
                    va='center', color=INK, fontsize=5.6)
    ax.set_xscale('log')
    ax.set_xticks([88.6, 201.3])
    ax.set_xticklabels(['88.6M', '201M'])
    ax.minorticks_off()
    ax.set_xlim(80, 290)
    ax.tick_params(axis='x', labelsize=5.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xlabel('Training tokens (log)\ndiamonds: 300 s', fontsize=5.8)
    ax.set_ylabel('Held-out BPB')
    ax.grid(axis='y', color=GRID, lw=0.5)
    ax.set_title('(a) Seeds (hollow) and means', loc='left', fontsize=6.4)
    cols = [('time_300s', '300 s'), ('tokens_088604672', '88.6M'), ('tokens_157286400', '157M'), ('tokens_201326592', '201M')]
    for i, (c, _) in enumerate(cols):
        st = pc[c]
        bx.plot([i, i], st['ci'], color=BLUE, lw=1.1, zorder=3)
        bx.scatter([i], [st['mean']], s=20, color=BLUE, zorder=5, edgecolors='white', linewidths=0.5)
        for s_, v in zip(st['seeds'], st['values']):
            bx.scatter([i + 0.22], [v], s=8, marker=SEED_MARK[s_], facecolor='white', edgecolor=BLUE, lw=0.6, zorder=4)
    ox = 4.35
    for j, k in enumerate(('OPUS-A23', 'OPUS-A24', 'OPUS-A27')):
        x_ = mt[k]['vs_ar']['heldout']
        xx = ox + 0.3 * j
        bx.plot([xx, xx], x_['ci95'], color=VIOLET, lw=1.1, zorder=3)
        bx.scatter([xx], [x_['mean']], s=18, color=VIOLET, zorder=5, edgecolors='white', linewidths=0.5)
    bx.axhline(0, color=INK2, lw=0.7, zorder=2)
    bx.set_xticks([0, 1, 2, 3, ox + 0.3])
    bx.set_xticklabels(['300 s', '88.6', '157', '201', '201'], fontsize=5.2)
    bx.set_xlabel('Budget (tokens in M)', fontsize=5.8)
    bx.set_xlim(-0.55, 5.25)
    bx.set_ylim(-0.03, 0.115)
    bx.axvline(0.6, color=GRID, lw=0.8, zorder=1)
    bx.axvline(3.68, color=GRID, lw=0.8, zorder=1)
    for x_, t in ((0.0, 'equal\ntime'), (2.0, 'GPT A22,\nequal tokens'), (ox + 0.3, 'Opus,\n201M')):
        bx.text(x_, 0.99, t, transform=bx.get_xaxis_transform(), ha='center', va='top', fontsize=5.3, color=INK2,
                linespacing=1.0)
    bx.set_ylabel('Difference from AR-ref (BPB)', fontsize=6.0, labelpad=2)
    bx.tick_params(axis='y', labelsize=5.4)
    bx.yaxis.set_major_locator(MultipleLocator(0.02))
    bx.grid(axis='y', color=GRID, lw=0.5)
    hand = [Line2D([], [], ls='none', marker=SEED_MARK[s_], ms=3.0, mfc='white', mec=BLUE, mew=0.6, label=f'seed {s_}')
            for s_ in (6, 7, 8)]
    bx.legend(handles=hand, loc='lower right', ncol=3, frameon=False, fontsize=5.4, handletextpad=0.1,
              columnspacing=0.5, borderaxespad=0.1, handlelength=1.0)
    bx.set_title('(b) Paired by seed; 95% CIs', loc='left', fontsize=6.4)
    draw_split(cx, D, title='(c) 300 s gap, exploratory')
    cx.set_xlim(-0.115, 0.08)
    cx.set_xlabel('GPT A22 − AR-ref (BPB)')
    cx.set_title('(c) 300 s gap, exploratory', loc='left', fontsize=6.4)
    cx.tick_params(axis='y', labelsize=5.4)
    cx.tick_params(axis='x', labelsize=5.4)
    for t in cx.texts:
        t.set_fontsize(5.6)
    save(fig, 'cr_e1_full')


# ============================================================================ macros quoted in the revision-9 text
def macros_r9(C, B, best, I):
    ev = read_json(CR_DATA / 'cr_evidence.json')
    lin = {r['id']: r for r in ev['seed0_lineage']}
    g22_dec_x = lin[134]['inference_tokens_per_second'] / lin[8]['inference_tokens_per_second']
    _need(round(g22_dec_x, 1) == 1.9, g22_dec_x)
    by = {r['name']: r for r in C['rows']}
    mean = lambda g, n, k: 100 * statistics.mean(next(r for r in B['groups'][g] if r['name'] == n)[k])
    _need(best[('3-5', 'arc_easy')] == ['GPT A22'] and best[('6-8', 'arc_easy')] == ['GPT A22', 'Opus A27'], best)
    _need(best[('6-8', 'piqa')] == ['Opus A23'] and best[('6-8', 'lambada')] == ['Opus A27'], best)
    _need(best[('3-5', 'lambada')] == ['MOTPE', 'GPT A22'] and best[('3-5', 'blimp')] == ['AR-ref'] == best[('6-8', 'blimp')], best)
    _need(best[('3-5', 'heldout')] == ['AR-ref'] == best[('6-8', 'heldout')], best)
    M = {
        'CRNineTrainRatio': f"{C['train_ratio_g22_ar']:.2f}",
        'CRNineOpTrainRatioLo': f"{min(C['opus_train_ratio_ar']):.2f}",
        'CRNineOpTrainRatioHi': f"{max(C['opus_train_ratio_ar']):.2f}",
        'CRNineGptDecX': f'{g22_dec_x:.1f}',
        'CRNineOpMemRed': f"{100 * (1 - by['Opus A23']['mem'] / by['Baseline']['mem']):.0f}",
        'CRNineBaseMem': f"{by['Baseline']['mem']:.0f}",
        'CRNineOpMemTwentyThree': f"{by['Opus A23']['mem']:.0f}",
        'CRNineOpMemTwentySeven': f"{by['Opus A27']['mem']:.0f}",
        'CRNineArcGptPanel': f"{mean('3-5', 'GPT A22', 'arc_easy'):.1f}",
        'CRNineArcARPanel': f"{mean('3-5', 'AR-ref', 'arc_easy'):.1f}",
        'CRNineLambGptPanel': f"{mean('3-5', 'GPT A22', 'lambada'):.1f}",
        'CRNineLambARPanel': f"{mean('3-5', 'AR-ref', 'lambada'):.1f}",
        'CRNineArcSix': f"{mean('6-8', 'GPT A22', 'arc_easy'):.1f}",
        'CRNineArcARSix': f"{mean('6-8', 'AR-ref', 'arc_easy'):.1f}",
        'CRNinePiqaOpus': f"{mean('6-8', 'Opus A23', 'piqa'):.1f}",
        'CRNinePiqaARSix': f"{mean('6-8', 'AR-ref', 'piqa'):.1f}",
        'CRNineLambOpus': f"{mean('6-8', 'Opus A27', 'lambada'):.1f}",
        'CRNineLambARSix': f"{mean('6-8', 'AR-ref', 'lambada'):.1f}",
        'CRNineBenchSDMax': f"{B['max_acc_sd_pts']:.1f}",
        'CRNineBenchSDMed': f"{B['median_acc_sd_pts']:.1f}",
        'CRNineEtoEX': f"{I['e2e_x']:.1f}",
        'CRNineARHeldSix': f"{C['e18_ar_heldout']:.4f}",
        'CRNineGptHeldSix': f"{C['e18_g22_heldout']:.4f}",
    }
    _need(M['CRNineArcSix'] == f"{mean('6-8', 'Opus A27', 'arc_easy'):.1f}", 'ARC-Easy tie')
    lines = ['% Generated by analysis/cr_r9.py. Do not edit by hand.'] + [f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in M.items()]
    (GEN_CR / 'cr_macros_r9.tex').write_text('\n'.join(lines) + '\n')
    return M


# ============================================================================ every agent quote in Part 1 is verbatim
def check_part1_quotes():
    """Every ``...'' quotation in Part 1 must occur verbatim (whitespace-normalised) in an agent's attempt record:
    AutoResearch and GPT-5.6 seed 0 (database snapshot), the extended GPT-5.6 and Opus arms (history.jsonl)."""
    from cr_common import ROOT
    from cr_v2 import CODEX_ROOT
    tex = (ROOT / 'paper/main.tex').read_text()
    part1 = tex[tex.index('\\section{Autonomous research result}'):tex.index('\\label{partoneend}')]
    quotes = [' '.join(q.split()) for q in re.findall(r"``(.+?)''", part1, re.S)]
    corpus = []
    for nodes in (track_ar()[0], track_apq()[0], track_extended(OPUS_ROOT, 'Opus'), track_extended(CODEX_ROOT, 'GPT')):
        corpus += [' '.join(n['text'].split()) for n in nodes.values()]
    # full Opus records (hypothesis, intended change, patch) for quotes outside the hypothesis field
    for line in (OPUS_ROOT / 'history.jsonl').read_text().splitlines():
        d = json.loads(line)
        corpus.append(' '.join('\n'.join(str(d.get(k) or '') for k in ('hypothesis', 'intended_change', 'candidate_patch')).split()))
    blob = '\n'.join(corpus)
    missing = [q for q in quotes if q not in blob]
    _need(quotes and not missing, f'Part 1 quotes not found verbatim: {missing}')
    return quotes


def main():
    D, S, R = load()
    C = fig_compare(D, S, R)
    fig_scatter(D, S, R)
    fig_lineage3()
    B = bench_data(D, S)
    best = table_bench_main(B)
    table_bench_seeds(B)
    fig_e1_compact(D)
    I = fig_inference2(D, R)
    fig_histories()
    fig_e1_full(D)
    M = macros_r9(C, B, best, I)
    print('cr_r9: Part 1 quotes verified:', len(check_part1_quotes()))
    print(json.dumps(M))
    print('cr_r9:', json.dumps({'train_ratio': round(C['train_ratio_g22_ar'], 3)}))


if __name__ == '__main__':
    main()
