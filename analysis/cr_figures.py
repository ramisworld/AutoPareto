#!/usr/bin/env python3
"""Camera-ready figures, drawn only from extracted primary records.

  (a) cr_pareto          held-out BPB vs cached decode; per-seed points and seed means; memory as marker area
  (b) cr_reversal        AP-Q minus AR-ref per seed: fixed time (seeds 0-2) vs matched 201M tokens (seeds 3-5)
                         EXISTING records only -- this is not the planned E1 token-response crossover
  (c) cr_decision_graph  research-decision record for the AP-Q lineage: recorded parent, edit (derived from
                         the frozen source), verbatim hypothesis (truncated) and measured outcome

Inputs: paper/camera_ready/data/{cr_evidence,cr_stats,lineage_edit_labels}.json (see cr_evidence.py,
lineage_labels.py).  Outputs: paper/figures/cr_*.{pdf,png} and poster versions in
paper/camera_ready/poster_figures/.  Palette: validated reference slots 1-3 (blue, orange, aqua) plus neutral
gray; every mark is direct-labelled because aqua is below 3:1 contrast on white.
"""
from __future__ import annotations

import os
import re
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / 'paper/camera_ready/.mplcache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from cr_common import CR_DATA, FIG, NAMES, PANEL_ORDER, POSTER, read_json

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
GRAY, INK, INK2, MUTED, GRID = '#8a8985', '#0b0b0b', '#52514e', '#9a9893', '#e6e5e1'
COLOR = {'autopareto_292': BLUE, 'autoresearch_compatible': ORANGE, 'autopareto_299': AQUA,
         'autopareto_294': AQUA, 'baseline': GRAY, 'tpe_quality': GRAY, 'motpe_quality': GRAY}


def style(scale=1.0):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5 * scale, 'axes.labelsize': 7.5 * scale,
                         'axes.titlesize': 8 * scale, 'xtick.labelsize': 6.8 * scale, 'ytick.labelsize': 6.8 * scale,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': INK2,
                         'axes.linewidth': 0.6, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.labelcolor': INK,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})


def save(fig, name, poster=False):
    meta = {'Creator': 'AutoPareto camera-ready analysis (analysis/cr_figures.py)', 'Author': ''}
    if poster:
        POSTER.mkdir(parents=True, exist_ok=True)
        fig.savefig(POSTER / f'{name}.pdf', bbox_inches='tight', metadata=meta)
        fig.savefig(POSTER / f'{name}.png', dpi=300, bbox_inches='tight')
    else:
        fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight', metadata=meta)
        fig.savefig(FIG / f'{name}.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------------------ (a) Pareto

def fig_pareto(ev, S, scale=1.0, size=(3.05, 2.15), poster=False):
    style(scale)
    fig, ax = plt.subplots(figsize=size, layout='constrained')
    runs = ev['panel_runs']
    t1 = S['table1']
    area = lambda mib: (mib / 250.0) * 150 * scale ** 2
    for m in PANEL_ORDER:
        c = COLOR[m]
        rs = [r for r in runs if r['model'] == m]
        ax.scatter([r['decode_tok_s'] for r in rs], [r['heldout_bpb'] for r in rs], s=9 * scale ** 2,
                   facecolor='white', edgecolor=c, lw=0.7, zorder=3)
        mx, my = t1[m]['decode_tok_s']['mean'], t1[m]['heldout_bpb']['mean']
        ax.scatter([mx], [my], s=area(t1[m]['peak_alloc_mib']['mean']), color=c, alpha=0.85,
                   edgecolor='white', lw=1.0, zorder=4)
    # direct labels (text in ink, never series color)
    offs = {'baseline': (0, -12, 'center'), 'autoresearch_compatible': (9, 7, 'left'), 'tpe_quality': (8, -3, 'left'),
            'motpe_quality': (6, 10, 'center'), 'autopareto_292': (2, -12, 'center'), 'autopareto_299': (-9, 3, 'right'),
            'autopareto_294': (-9, -4, 'right')}
    for m in PANEL_ORDER:
        dx, dy, ha = offs[m]
        mib = t1[m]['peak_alloc_mib']['mean']
        bold = m in ('autopareto_292', 'autoresearch_compatible')
        ax.annotate(f'{NAMES[m]}  {mib:.0f} MiB', (t1[m]['decode_tok_s']['mean'], t1[m]['heldout_bpb']['mean']),
                    xytext=(dx * scale, dy * scale), textcoords='offset points', ha=ha, va='center',
                    fontsize=6.4 * scale, color=INK if bold else INK2, fontweight='bold' if bold else 'normal')
    ax.set_xlabel('Cached-decode tokens/s (higher is better)')
    ax.set_ylabel('Held-out BPB (lower is better)')
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_xlim(115, 560)
    ax.set_ylim(1.088, 1.205)
    # size key
    keys = [100, 175, 250]
    handles = [Line2D([], [], ls='', marker='o', markersize=(area(k)) ** 0.5, markerfacecolor='#c9c8c3',
                      markeredgecolor='white') for k in keys]
    handles.append(Line2D([], [], ls='', marker='o', markersize=3 * scale, markerfacecolor='white', markeredgecolor=INK2))
    ax.legend(handles, [f'{k} MiB' for k in keys] + ['one seed'], loc='upper left', frameon=False, fontsize=5.8 * scale,
              title='marker area: peak inference memory', title_fontsize=5.8 * scale, ncol=2, handletextpad=0.3,
              columnspacing=0.8, borderaxespad=0.3, labelspacing=0.9)
    if poster:
        ax.set_title('Matched exposure (201M tokens, seeds 3-5): AP-Q trades quality for 1.99x decode', loc='left',
                     fontsize=8.5 * scale)
    save(fig, 'cr_pareto', poster)


# ------------------------------------------------------------------ (b) reversal

def fig_reversal(ev, S, scale=1.0, size=(2.75, 2.15), poster=False):
    style(scale)
    fig, ax = plt.subplots(figsize=size, layout='constrained')
    cols = [
        ('archive\nseeds\n0 vs 42', [S['discovery_apq_minus_ar']], None, 'unmatched'),
        ('re-run\nseeds 0-2\nval', S['fixed_time_apq_minus_ar']['diffs'], S['fixed_time_apq_minus_ar'], 'paired'),
        ('201M tok.\nseeds 3-5\nval', S['matched_val']['diffs'], S['matched_val'], 'paired'),
        ('201M tok.\nseeds 3-5\nheld-out', S['matched_heldout']['diffs'], S['matched_heldout'], 'paired'),
    ]
    ax.axhline(0, color=INK2, lw=0.7, zorder=1)
    ax.axvspan(-0.5, 1.5, color='#f2f1ed', zorder=0, lw=0)
    for i, (lab, diffs, st, kind) in enumerate(cols):
        jit = [-0.12, 0.0, 0.12][:len(diffs)] if len(diffs) > 1 else [0.0]
        if st is not None:
            lo, hi = st['ci95']
            ax.plot([i, i], [lo, hi], color=INK, lw=1.0, zorder=2, solid_capstyle='butt')
            ax.plot([i - 0.2, i + 0.2], [st['mean'], st['mean']], color=INK, lw=1.6, zorder=3)
        ax.scatter([i + j for j in jit], diffs, s=16 * scale ** 2,
                   facecolor=(BLUE if kind == 'paired' else 'white'), edgecolor=(BLUE if kind == 'paired' else GRAY),
                   lw=0.9, zorder=4)
    ax.text(0.5, 0.0605, 'fixed time (300 s)', fontsize=6.0 * scale, color=INK, va='top', ha='center', fontweight='bold')
    ax.text(2.5, 0.0605, 'fixed tokens (201M)', fontsize=6.0 * scale, color=INK, va='top', ha='center', fontweight='bold')
    ax.text(1.62, 0.0045, '$\\uparrow$ AR-ref better', fontsize=5.8 * scale, color=INK2, ha='left', va='bottom')
    ax.text(1.62, -0.0045, '$\\downarrow$ AP-Q better', fontsize=5.8 * scale, color=INK2, ha='left', va='top')
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[0] for c in cols], fontsize=5.9 * scale)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.0165, 0.0625)
    ax.set_ylabel('AP-Q minus AR-ref, BPB')
    ax.grid(axis='y', color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], ls='', marker='o', markerfacecolor=BLUE, markeredgecolor=BLUE, markersize=4 * scale),
               Line2D([], [], color=INK, lw=1.6), Line2D([], [], color=INK, lw=1.0)]
    ax.legend(handles, ['one seed', 'mean', '95% CI (t, n=3)'], loc='upper left', bbox_to_anchor=(0.0, 0.90),
              frameon=False, fontsize=5.6 * scale, handlelength=1.2)
    if poster:
        ax.set_title('Same two recipes, opposite ranking (existing records; not the E1 sweep)', loc='left',
                     fontsize=8.5 * scale)
    save(fig, 'cr_reversal', poster)


# ------------------------------------------------------------------ (c) decision graph

PATH = [8, 113, 116, 118, 119, 120, 130, 131, 132, 134]


def _first_sentence(h, limit):
    h = ' '.join(h.split())
    if len(h) <= limit:
        return h
    cut = h[:limit].rsplit(' ', 1)[0]
    return cut + ' ...'


def fig_decision(ev, labels, S, scale=1.0, size=(6.9, 9.4), poster=False, hyp_chars=330, wrap=98):
    style(scale)
    lin = {r['id']: r for r in ev['seed0_lineage']}
    lab = {x['exp']: x for x in labels['labels']}
    children = {}
    for r in ev['seed0_lineage']:
        if r['parent_experiment_id'] is not None:
            children.setdefault(r['parent_experiment_id'], []).append(r['id'])
    fig, ax = plt.subplots(figsize=size)
    ax.set_xlim(0, 100)
    n = len(PATH)
    ax.set_ylim(-0.55, n - 0.45)
    ax.invert_yaxis()
    ax.axis('off')
    pooled = S['pooled_fixed_time_sd']
    for k, eid in enumerate(PATH):
        r = lin[eid]
        y = k
        on_noise = eid in (131, 132, 134)
        face = '#eef4fc' if eid != 134 else '#d8e7fa'
        edge = BLUE
        ax.add_patch(FancyBboxPatch((1, y - 0.46), 73, 0.92, boxstyle='square,pad=0.0',
                                    facecolor=face if eid != 8 else '#f2f1ed', edgecolor=edge if eid != 8 else GRAY,
                                    lw=0.9 if eid == 134 else 0.6, zorder=2))
        if eid == 8:
            head = 'Registered baseline (exp 8)   depth 8, width 512, batch $2^{19}$'
            meta = f"val BPB {r['val_bpb']:.4f}   decode {r['inference_tokens_per_second']:.0f} tok/s   memory {r['peak_inference_cuda_memory_mb']:.0f} MiB   (seed 0, 300 s)"
            hyp = None
        else:
            p = lin[r['parent_experiment_id']]
            d = r['val_bpb'] - p['val_bpb']
            tag = '   [AP-Q]' if eid == 134 else ''
            head = f"A{r['experiment_number']:02d} (exp {eid}) from A{p['experiment_number']:02d}: {lab[eid]['label']}{tag}" if p['id'] != 8 \
                else f"A{r['experiment_number']:02d} (exp {eid}) from baseline: {lab[eid]['label']}"
            meta = (f"val BPB {r['val_bpb']:.4f} ($\\Delta$ {d:+.4f})   decode {r['inference_tokens_per_second']:.0f} tok/s   "
                    f"memory {r['peak_inference_cuda_memory_mb']:.0f} MiB   decision: {r['decision']}")
            hyp = _first_sentence(r['hypothesis'], hyp_chars)
        ax.text(2.2, y - 0.40, head, fontsize=6.6 * scale, fontweight='bold', color=INK, va='top', zorder=3)
        ax.text(2.2, y - 0.21, meta, fontsize=5.9 * scale, color=INK2, va='top', zorder=3)
        if hyp:
            wrapped = textwrap.fill('"' + hyp + '"', wrap)
            ax.text(2.2, y - 0.04, wrapped, fontsize=5.5 * scale, color=INK2, va='top', style='italic', zorder=3,
                    linespacing=1.15)
        # recorded intent vs built model (width rounding)
        if eid != 8:
            m = re.search(r'width (?:from )?(\d+) to (\d+)', r['intended_change'] or '')
            built = lab[eid]['config']['n_embd']
            pid = r['parent_experiment_id']
            pbuilt = labels['baseline_config']['n_embd'] if pid == labels['baseline_exp'] else lab[pid]['config']['n_embd']
            if m and (int(m.group(1)), int(m.group(2))) != (pbuilt, built):
                ax.text(75.5, y + 0.14, f'Recorded intended change: width {m.group(1)}$\\to${m.group(2)};\n'
                        f'built (width rounding): {pbuilt}$\\to${built}.', fontsize=5.3 * scale, color='#b2412a',
                        ha='left', va='top', zorder=3)
        # siblings explored from the same parent (off-path), with outcomes
        if k + 1 < n:
            nxt = PATH[k + 1]
            sib = [c for c in children.get(eid, []) if c != nxt]
            if sib:
                lines = []
                for c in sorted(sib):
                    rc = lin[c]
                    val = f"{rc['val_bpb']:.4f}" if rc['val_bpb'] is not None else 'crash (missing tokenizer)'
                    desc = lab[c]['label'].replace('$\\to$', '->').replace('$', '').replace('\\to', '->')
                    nsub = len(children.get(c, []))
                    sub = f' (+{nsub} descendants)' if nsub else ''
                    lines.append(f"A{rc['experiment_number']:02d} {desc}: {val}{sub}")
                ax.text(75.5, y - 0.40, 'Also branched from this node:\n' + '\n'.join(lines), fontsize=5.2 * scale,
                        color=MUTED, va='top', zorder=3, linespacing=1.15)
            ax.annotate('', xy=(10, k + 1 - 0.46), xytext=(10, k + 0.46),
                        arrowprops={'arrowstyle': '-|>', 'color': BLUE, 'lw': 1.0, 'mutation_scale': 7 * scale}, zorder=1)
    # noise bracket over A19-A22
    y0, y1 = PATH.index(131) - 0.46, PATH.index(134) + 0.46
    ax.plot([74.6, 74.6], [y0, y1], color=ORANGE, lw=1.2)
    ax.text(75.5, (y0 + y1) / 2 + 0.55, f'Each step after A18 moves val BPB\nby at most {S["lineage_steps_after_A18"]["A19->A20"]:.4f}, below the pooled\n'
                                       f'fixed-time seed SD ({pooled:.4f})', fontsize=5.4 * scale, color='#b2412a', va='center')
    if poster:  # the paper version carries this text in its LaTeX caption
      ax.set_title('Research-decision record for the AP-Q lineage (seed-0 campaign, 300 s contract). Italic text is the agent\'s\n'
                   'recorded hypothesis, quoted verbatim; edit labels are derived from the frozen sources; outcomes are measured values.',
                   fontsize=6.6 * scale, loc='left', color=INK2)
    save(fig, 'cr_decision_graph', poster)


def main():
    ev = read_json(CR_DATA / 'cr_evidence.json')
    S = read_json(CR_DATA / 'cr_stats.json')
    labels = read_json(CR_DATA / 'lineage_edit_labels.json')
    FIG.mkdir(parents=True, exist_ok=True)
    fig_pareto(ev, S)
    fig_reversal(ev, S)
    fig_decision(ev, labels, S)
    fig_pareto(ev, S, scale=1.9, size=(7.2, 5.4), poster=True)
    fig_reversal(ev, S, scale=1.9, size=(6.6, 5.4), poster=True)
    fig_decision(ev, labels, S, scale=1.6, size=(13.5, 15.5), poster=True, hyp_chars=330, wrap=108)
    print('wrote cr_pareto, cr_reversal, cr_decision_graph (paper + poster)')


if __name__ == '__main__':
    main()
