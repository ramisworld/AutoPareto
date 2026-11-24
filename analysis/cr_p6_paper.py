#!/usr/bin/env python3
"""Camera-ready revision 10: Phase 6 unified 13-model comparison for the paper.

Reads results/camera_ready_2026/phase6/analysis/unified_13.json (written by analysis/cr_p6.py from recorded
artefacts only) and writes:
  paper/figures/cr_compare13.{pdf,png}        Figure 1: one row per recipe (revision-9 layout), all 13 recipes
  paper/figures/cr_unified.{pdf,png}          Part 1 (revision 11): quality against decode speed, both contracts;
                                              rings mark runs outside the 2% drift rule; stability_check() asserts
                                              that dropping them leaves the frontier of the seed means unchanged
  paper/generated/camera_ready/table_unified.tex       appendix: one row per recipe, seeds 6-8 means
  paper/generated/camera_ready/table_unified_lmeval.tex   lm-eval tasks (appendix), when recorded
  paper/generated/camera_ready/cr_macros_p6.tex
  paper/camera_ready/data/cr_p6.json
Every decode value comes from the one Phase 6 benchmark (1,024 input, 128 generated tokens, same A40, same
session protocol) for all 13 recipes. Launches nothing; no GPU; no network.
"""
from __future__ import annotations

import json
import math
import os
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

from cr_common import CR_DATA, GEN_CR, paired, write_json
from cr_v2 import AQUA, BLUE, GRAY, GRID, INK, INK2, MUTED, ORANGE, VIOLET, save, style

ROOT = Path(__file__).resolve().parents[1]
UNI = ROOT / 'results/camera_ready_2026/phase6/analysis/unified_13.json'
COL = {'baseline': GRAY, 'autoresearch': ORANGE, 'classical': AQUA, 'gpt': BLUE, 'opus': VIOLET}
MARK = {'baseline': 's', 'autoresearch': 'D', 'classical': 'o', 'gpt': '^', 'opus': 'v'}
MINUS = '−'


def group(recipe, family):
    if family in ('baseline', 'autoresearch', 'classical'):
        return family
    return 'gpt' if recipe.startswith('gpt') else 'opus'


def load():
    U = json.loads(UNI.read_text())
    S = {s['recipe']: s for s in U['summary']}
    rows = U['rows']
    return U, S, rows


def pareto(points):
    """points: [(name, bpb, speed)] -> non-dominated names (lower bpb, higher speed)."""
    out = []
    for n, q, v in points:
        if not any(q2 <= q and v2 >= v and (q2 < q or v2 > v) for _, q2, v2 in points):
            out.append(n)
    return out


def place_column(items, lo, hi, gap):
    """items: [(name, y)] -> {name: text_y}, sorted by y, at least `gap` apart, kept inside [lo, hi]."""
    items = sorted(items, key=lambda t: t[1])
    ys = []
    for _, y in items:
        ys.append(max(y, ys[-1] + gap) if ys else y)
    over = ys[-1] - hi if ys else 0
    if over > 0:
        ys = [y - over for y in ys]
        for i in range(len(ys) - 2, -1, -1):
            ys[i] = min(ys[i], ys[i + 1] - gap)
    return {n: y for (n, _), y in zip(items, ys)}


def panel(ax, S, rows, key, title, ylim, col_x, unstable=frozenset()):
    """One contract's BPB-decode projection. Seed points of inference runs outside the benchmark's 2% drift rule
    (`unstable`, (recipe, seed) pairs) are drawn as rings: reported as measured, never dropped."""
    pts, off = [], {}
    lo, hi = ylim
    lbox = dict(boxstyle='square,pad=0.06', fc='white', ec='none', alpha=0.88)
    for r, s in S.items():
        g = group(r, s['family'])
        q, v = s[f'{key}_mean'], s['inf_decode_tok_s_mean']
        pts.append((r, q, v))
        if q > hi:  # off scale: marker on the edge, value in the label
            off[r] = q
            ax.scatter([v], [hi - 0.0016], s=22, marker=MARK[g], color=COL[g], edgecolor='white', lw=0.5, zorder=4,
                       clip_on=False)
            continue
        seeds = [(x[key], x['inf_decode_tok_s'], (r, x['seed']) in unstable) for x in rows if x['recipe'] == r]
        ok = [(a, b) for a, b, u in seeds if not u]
        bad = [(a, b) for a, b, u in seeds if u]
        ax.scatter([b for _, b in ok], [a for a, _ in ok], s=6, color=COL[g], alpha=0.45, lw=0, zorder=2)
        ax.scatter([b for _, b in bad], [a for a, _ in bad], s=7, facecolors='none', edgecolors=COL[g], alpha=0.75,
                   lw=0.5, zorder=2)
        ax.scatter([v], [q], s=24, marker=MARK[g], color=COL[g], edgecolor='white', lw=0.5, zorder=4)
    front = pareto(pts)
    fp = sorted([p for p in pts if p[0] in front], key=lambda p: p[2])
    xs, ys = [], []
    for i, (_, q, v) in enumerate(fp):
        if i:
            xs.append(v)
            ys.append(ys[-1])
        xs.append(v)
        ys.append(q)
    ax.plot(xs, ys, color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=3)
    gap = (hi - lo) * 0.062
    column = [(r, min(q, hi - 0.0016)) for r, q, v in pts if not r.startswith('opus')]
    ty = place_column(column, lo + gap * 0.4, hi - gap * 0.75, gap)
    for r, q, v in pts:
        if r.startswith('opus'):
            continue
        y = min(q, hi - 0.0016)
        txt = S[r]['display'] + (f' ({q:.3f}, off scale)' if r in off else '')
        ax.annotate(txt, (v, y), xytext=(col_x, ty[r]), textcoords='data', ha='left', va='center', fontsize=6.0,
                    color=INK, zorder=6, bbox=lbox,
                    arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.35, shrinkA=0.5, shrinkB=2.5))
    oq = max(q for r, q, _ in pts if r.startswith('opus'))
    ov = statistics.mean(v for r, _, v in pts if r.startswith('opus'))
    ax.annotate('Opus A23,\nA24, A27', (ov, oq), xytext=(0, -6), textcoords='offset points', ha='center', va='top',
                fontsize=6.0, color=INK, zorder=6)
    ax.set_xscale('log')
    ax.set_xlim(115, 7600)
    ax.set_xticks([125, 250, 500, 1000, 2000, 4000])
    ax.set_xticklabels(['125', '250', '500', '1k', '2k', '4k'])
    ax.minorticks_off()
    ax.set_ylim(hi, lo)
    ax.grid(color=GRID, lw=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel('Decode speed, tokens/s (log scale)', fontsize=6.2, labelpad=1.5)
    ax.set_title(title, fontsize=6.8, loc='left', color=INK)
    ax.tick_params(labelsize=6.0, length=2, pad=1.5)
    return front


def fig_unified(S, rows, unstable=frozenset()):
    """Part 1 figure (revision 11): the two-objective projection (held-out BPB against decode speed) of the 13-recipe
    test under both training contracts. Memory, the third objective, is in Figure 1 and the appendix table."""
    style(1.0)
    fig, axs = plt.subplots(1, 2, figsize=(5.5, 1.70), gridspec_kw={'wspace': 0.26})
    f_tok = panel(axs[0], S, rows, 'heldout_bpb', '(a) Equal training tokens (201.3M)', (1.0905, 1.192), 540, unstable)
    f_fix = panel(axs[1], S, rows, 'fixed_heldout_bpb', '(b) Equal training time (300 s)', (1.1485, 1.192), 540,
                  unstable)
    axs[0].set_ylabel('Held-out BPB (lower is better)', fontsize=6.3)
    handles = [Line2D([], [], marker=MARK[g], color=COL[g], ls='', ms=4.2, mec='white', mew=0.5, label=l)
               for g, l in [('baseline', 'Baseline'), ('autoresearch', 'AutoResearch'), ('classical', 'Classical search'),
                            ('gpt', 'GPT-5.6 AutoPareto'), ('opus', 'Opus 5.5 AutoPareto')]]
    handles.append(Line2D([], [], color=INK2, lw=0.7, ls=(0, (3, 2)), label='frontier of seed means'))
    fig.legend(handles=handles, loc='upper center', ncol=6, fontsize=5.8, frameon=False, bbox_to_anchor=(0.5, 1.085),
               handletextpad=0.25, columnspacing=0.6, handlelength=1.6)
    save(fig, 'cr_unified')
    return f_tok, f_fix


def stability_check(U, S, rows):
    """Post hoc sensitivity check of the inference runs that failed the benchmark's 2% drift rule (reported as
    measured, as pre-registered). Recomputes every recipe's mean decode rate from its stable runs only and the
    Pareto frontier of the seed means under both contracts; then repeats the frontier with the scored speed-up
    (each run normalised by its own bracketing baseline). The paper says the stable-only frontier is unchanged and
    that normalising changes only which Opus recipes are on it, so both are asserted here."""
    uns = {tuple(x) for x in U['inference_stability']['unstable_runs']}
    R = list(S)
    stable = {}
    for r in R:
        v = [x['inf_decode_tok_s'] for x in rows if x['recipe'] == r and (r, x['seed']) not in uns]
        if not v:
            raise SystemExit(f'stability_check: {r} has no stable inference run')
        stable[r] = statistics.mean(v)
    shift = {r: stable[r] / S[r]['inf_decode_tok_s_mean'] - 1 for r in R}
    out = {'unstable_runs': sorted(uns), 'stable_only_decode_mean': stable, 'relative_shift': shift}
    for key in ('heldout_bpb', 'fixed_heldout_bpb'):
        allf = set(pareto([(r, S[r][key + '_mean'], S[r]['inf_decode_tok_s_mean']) for r in R]))
        stf = set(pareto([(r, S[r][key + '_mean'], stable[r]) for r in R]))
        spf = set(pareto([(r, S[r][key + '_mean'], S[r]['inf_speedup_vs_bracket_mean']) for r in R]))
        if stf != allf:
            raise SystemExit(f'stability_check: stable-only frontier differs under {key}: {sorted(stf)} vs {sorted(allf)}')
        nonop = lambda s: {r for r in s if not r.startswith('opus')}
        if nonop(spf) != nonop(allf) or not any(r.startswith('opus') for r in spf):
            raise SystemExit(f'stability_check: speed-up frontier differs beyond the Opus recipes under {key}')
        out[key] = {'frontier_all': sorted(allf), 'frontier_stable_only': sorted(stf), 'frontier_speedup': sorted(spf)}
    op = [S[r]['inf_decode_tok_s_mean'] for r in R if r.startswith('opus')]
    out['opus_decode_spread'] = max(op) / min(op) - 1
    seed6 = sum(1 for _, sd in uns if sd == 6)
    mac = [f"\\newcommand{{\\CRSixStableShiftMax}}{{{100 * max(abs(x) for x in shift.values()):.1f}}}",
           f"\\newcommand{{\\CRSixUnstableSeedSix}}{{{seed6}}}",
           f"\\newcommand{{\\CRSixOpDecSpread}}{{{100 * out['opus_decode_spread']:.1f}}}"]
    ev, mac2 = inference_evidence(U, S, uns)
    out['run_evidence'] = ev
    return out, mac + mac2


def inference_evidence(U, S, uns):
    """Read every native inference record (raw protocol fields, nothing re-timed) and summarise where the 2% flag
    came from and what the full-workload correctness check found.

    The registered rule (study/contracts_2026/benchmark_policy.py): a run is unstable when the interquartile range of
    its eight candidate/baseline window ratios exceeds 2% of their median (or their CV exceeds 5%). The same rule is
    applied by the protocol to each arm's own windows (candidate, baseline before, baseline after) as diagnostics.
    Correctness (scripts/phase6_unified_inference.py): every generated token of 8 prompts x 128 positions against the
    recipe's own uncached model.forward, teacher-forced; a disagreement is a near-tie when the reference prefers its
    own argmax by at most 0.25 logits, otherwise a hard mismatch."""
    infdir = ROOT / 'results/camera_ready_2026/phase6/inference/runs'
    runs = []
    for r in U['rows']:
        d = json.loads((infdir / f"{r['recipe']}-seed{r['seed']}-native.json").read_text())
        p = d['registered_protocol_result']
        b = p['baseline_inference_result']
        c = d['full_workload_correctness']
        runs.append({'recipe': r['recipe'], 'seed': r['seed'], 'start': d['started_utc'],
                     'flagged': p['normalized_inference_status'] == 'unstable',
                     'reason': p.get('normalized_stability_failure_reason'),
                     'ratio_iqr': p['normalized_relative_drift'], 'cand_iqr': p['decode_block_relative_drift'],
                     'base_before_iqr': b['baseline_before_absolute_diagnostics']['decode_block_relative_drift'],
                     'base_after_iqr': b['baseline_after_absolute_diagnostics']['decode_block_relative_drift'],
                     'timed_equal': d['timed_path_outputs']['all_equal'], 'checked': c['tokens_checked'],
                     'agree': c['token_agreement'], 'near': c['near_tie_disagreements'], 'hard': c['hard_mismatches'],
                     'max_deficit': max(x['max_reference_deficit'] for x in c['per_prompt'])})
    if {(x['recipe'], x['seed']) for x in runs if x['flagged']} != set(uns):
        raise SystemExit('inference_evidence: flags in the records differ from unified_13.json')
    fl = [x for x in runs if x['flagged']]
    lim = 0.02
    base_drift = sum(1 for x in fl if max(x['base_before_iqr'], x['base_after_iqr']) > lim)
    cand_drift = sum(1 for x in fl if x['cand_iqr'] > lim)
    cancelled = sum(1 for x in runs if not x['flagged'] and x['cand_iqr'] > lim)
    cv_flag = sum(1 for x in fl if 'CV' in (x['reason'] or ''))
    drift_flag = sum(1 for x in fl if x['ratio_iqr'] > lim)
    cv_only = [x for x in fl if x['ratio_iqr'] <= lim]
    if drift_flag + len(cv_only) != len(fl) or any('CV' not in (x['reason'] or '') for x in cv_only):
        raise SystemExit('inference_evidence: flag reasons do not reconcile')
    opus_iqr = max(x['cand_iqr'] for x in runs if x['recipe'].startswith('opus'))
    ref_iqr = max(x['cand_iqr'] for x in runs if not x['recipe'].startswith('opus'))
    # time clustering: the first block of the evening session of 16 November (seed 6 of the ten recipes not reused
    # from E1 or the Opus retraining), against every other run
    ev = sorted([x for x in runs if x['start'] >= '2026-11-16T20'], key=lambda x: x['start'])
    blk = [x for x in ev if x['seed'] == 6]
    rest = [x for x in runs if x not in blk]
    t0, t1 = blk[0]['start'][11:16], blk[-1]['start'][11:16]
    chk, agr = sum(x['checked'] for x in runs), sum(x['agree'] for x in runs)
    near, hard = sum(x['near'] for x in runs), sum(x['hard'] for x in runs)
    if chk - agr != near + hard or hard != 0 or not all(x['timed_equal'] for x in runs):
        raise SystemExit('inference_evidence: correctness totals do not reconcile')
    opus = [S[r] for r in ('opus_a23', 'opus_a24', 'opus_a27')]
    dec_vs_recipe = statistics.mean(o['inf_decode_tok_s_mean'] for o in opus) / S['baseline']['inf_decode_tok_s_mean']
    m = lambda k, v: f'\\newcommand{{\\CRSix{k}}}{{{v}}}'
    names = lambda xs: ', '.join(f"{S[x['recipe']]['display']} seed {x['seed']}" for x in xs)
    mac = [m('FlagBaseDrift', base_drift), m('FlagCandDrift', cand_drift), m('FlagCV', cv_flag),
           m('FlagDrift', drift_flag), m('FlagCVOnly', len(cv_only)), m('FlagCVOnlyName', names(cv_only)),
           m('Cancelled', cancelled), m('OpusWinIQR', f'{100 * opus_iqr:.2f}'), m('RefWinIQR', f'{100 * ref_iqr:.1f}'),
           m('BlockFlags', sum(x['flagged'] for x in blk)), m('BlockRuns', len(blk)), m('BlockStart', t0),
           m('BlockEnd', t1), m('RestFlags', sum(x['flagged'] for x in rest)), m('RestRuns', len(rest)),
           m('TokensChecked', f'{chk:,}'), m('TokensAgree', f'{agr:,}'), m('AgreePct', f'{100 * agr / chk:.1f}'),
           m('NearTieMaxDeficit', f"{max(x['max_deficit'] for x in runs):.2f}"),
           m('TimedEqual', sum(x['timed_equal'] for x in runs)), m('DecXVsBaseRecipe', f'{dec_vs_recipe:.1f}')]
    return {'runs': runs, 'block': [t0, t1]}, mac


ORDER = ['baseline', 'arref', 'rs', 'tpe', 'motpe', 'nsgaii', 'cmaes', 'gpt_a22', 'gpt_a24', 'gpt_a25',
         'opus_a23', 'opus_a24', 'opus_a27']
STAR = '★'


def every_seed_front(U):
    """{contract key: recipes on the BPB-decode Pareto frontier in every seed (6, 7 and 8)}."""
    ps = U['pareto_per_seed']
    return {k: set.intersection(*[set(v) for v in ps[k].values()]) for k in ps}


def fig_compare13(U, S, rows):
    """Figure 1 (revision 10b): one row per recipe, the revision-9 layout filled from the 13-recipe test only.

    Every value comes from unified_13.json: seeds 6-8 on one A40. Held-out BPB after 201.3M tokens and after 300 s
    (both on one shared axis); training throughput from the 300 s runs (all passed the A1 timing-anomaly gate, so the
    one storage-stalled matched-token run is not used); decode speed and peak allocated memory from the one inference
    benchmark (1,024 input, 128 generated tokens, batch 1). Speed-ups (Opus only, as in revision 9) are over the
    bracketing registered-baseline runs. A star marks recipes on the BPB-decode Pareto frontier in every seed under
    that contract. Nothing is estimated or filled in."""
    front = every_seed_front(U)
    seeds = {r: sorted((x for x in rows if x['recipe'] == r), key=lambda x: x['seed']) for r in ORDER}
    _ok = all(len(seeds[r]) == 3 for r in ORDER) and all(
        S[r][k] is not None for r in ORDER for k in ('heldout_bpb_mean', 'fixed_heldout_bpb_mean', 'fixed_train_tok_s_mean',
                                                     'inf_decode_tok_s_mean', 'inf_peak_allocated_mib_mean'))
    if not _ok:
        raise SystemExit('fig_compare13: a recorded value is missing')
    style(1.0)
    gaps = {'arref': 0.35, 'rs': 0.35, 'gpt_a22': 0.35, 'opus_a23': 0.35}
    ys, y = [], 0.0
    for r in ORDER:
        y += gaps.get(r, 0.0)
        ys.append(y)
        y += 1.0
    ymax = ys[-1]
    fig = plt.figure(figsize=(5.5, 2.52))
    gs = fig.add_gridspec(1, 5, width_ratios=[0.78, 0.78, 0.80, 1.30, 0.72], wspace=0.12, left=0.21, right=0.988,
                          top=0.80, bottom=0.095)
    axs = [fig.add_subplot(gs[0, i]) for i in range(5)]
    fs, fsn = 6.0, 5.7
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
            ax.annotate(extra, (x1, yy), xytext=(1.2, 0), textcoords='offset points', ha='left', va='center',
                        fontsize=fs if bold_extra else fsn, color=ecol, zorder=6,
                        fontweight='bold' if bold_extra else 'normal')

    a0 = axs[0]
    x0f = a0.get_position().x0
    for r, yy in zip(ORDER, ys):
        g = group(r, S[r]['family'])
        yf = fig.transFigure.inverted().transform(a0.transData.transform((0, yy)))[1]
        fig.add_artist(Rectangle((0.012, yf - 0.019), 0.0045, 0.038, transform=fig.transFigure, color=COL[g], lw=0))
        fig.text(0.024, yf, S[r]['display'], ha='left', va='center', fontsize=6.3, color=INK)
        fig.text(x0f - 0.006, yf, f"{S[r]['params_mean'] / 1e6:.1f}M", ha='right', va='center', fontsize=5.8, color=INK2)
    # (1, 2) held-out BPB after 201.3M tokens and after 300 s, one shared axis
    out = []
    for ax, key in ((axs[0], 'heldout_bpb'), (axs[1], 'fixed_heldout_bpb')):
        ax.set_xlim(1.088, 1.312)
        ax.set_xticks([1.10, 1.15, 1.20, 1.25])
        ax.set_xticklabels(['1.10', '', '1.20', ''])
        for r, yy in zip(ORDER, ys):
            g = group(r, S[r]['family'])
            vals = [x[key] for x in seeds[r]]
            for v in vals:
                ax.scatter([v], [yy], s=7, marker='o', facecolors='none', edgecolors=COL[g], linewidths=0.5, zorder=3)
            m = S[r][f'{key}_mean']
            ax.scatter([m], [yy], s=17, marker='o', color=COL[g], zorder=4, edgecolors='white', linewidths=0.4)
            lab(ax, max([m] + vals), yy, f'{m:.3f}', STAR if r in front[key] else None, INK)
            out.append({'recipe': r, 'contract': key, 'mean': m, 'seeds': vals, 'every_seed_front': r in front[key]})
    # (3) training throughput, from the 300 s runs
    ax = axs[2]
    ax.set_xlim(0, 1000)
    ax.set_xticks([0, 400, 800])
    for r, yy in zip(ORDER, ys):
        k = S[r]['fixed_train_tok_s_mean'] / 1e3
        ax.barh(yy, k, height=0.6, color=COL[group(r, S[r]['family'])], zorder=3, lw=0)
        lab(ax, k, yy, f'{k:.0f}')
    # (4) decode throughput and (5) peak inference memory, one benchmark for all 13
    ax, mx = axs[3], axs[4]
    ax.set_xlim(0, 8300)
    ax.set_xticks([0, 2500, 5000])
    ax.set_xticklabels(['0', '2,500', '5,000'])
    mx.set_xlim(0, 455)
    mx.set_xticks([0, 150, 300])
    for r, yy in zip(ORDER, ys):
        g = group(r, S[r]['family'])
        d, mem = S[r]['inf_decode_tok_s_mean'], S[r]['inf_peak_allocated_mib_mean']
        ax.barh(yy, d, height=0.6, color=COL[g], zorder=3, lw=0)
        if g == 'opus':
            lab(ax, d, yy, f'{d:,.0f}', f"{S[r]['inf_speedup_vs_bracket_mean']:.1f}×", VIOLET, bold_extra=True)
        else:
            lab(ax, d, yy, f'{d:,.0f}')
        mx.barh(yy, mem, height=0.6, color=COL[g], zorder=3, lw=0)
        lab(mx, mem, yy, f'{mem:.0f}')
    # column headers
    b0, b1 = axs[0].get_position(), axs[1].get_position()
    fig.text((b0.x0 + b1.x1) / 2, 0.985, 'Held-out BPB ↓', ha='center', va='top', fontsize=6.5, color=INK, fontweight='bold')
    for b, sub in ((b0, 'after 201.3M\ntraining tokens'), (b1, 'after 300 s\nof training')):
        fig.text((b.x0 + b.x1) / 2, 0.925, sub, ha='center', va='top', fontsize=5.8, color=INK2, linespacing=1.05)
    heads = [(axs[2], 'Training speed ↑', 'thousand tokens/s,\n300 s runs'),
             (axs[3], 'Decode throughput ↑', 'tokens/s, batch 1; 1,024-token\nprompt, 128 generated'),
             (axs[4], 'Peak memory ↓', 'MiB, inference,\nsame benchmark')]
    for a_, h1, h2 in heads:
        b = a_.get_position()
        fig.text((b.x0 + b.x1) / 2, 0.985, h1, ha='center', va='top', fontsize=6.5, color=INK, fontweight='bold')
        fig.text((b.x0 + b.x1) / 2, 0.925, h2, ha='center', va='top', fontsize=5.8, color=INK2, linespacing=1.05)
    fig.text(0.012, 0.985, 'Recipe', ha='left', va='top', fontsize=6.5, color=INK, fontweight='bold')
    fig.text(b0.x0 - 0.006, 0.925, 'parameters', ha='right', va='top', fontsize=5.8, color=INK2)
    hand = [Line2D([], [], ls='none', marker='o', ms=3.4, mfc=INK2, mec='white', label='mean, seeds 6–8'),
            Line2D([], [], ls='none', marker='o', ms=2.6, mfc='none', mec=INK2, mew=0.6, label='single seeds')]
    leg = fig.legend(handles=hand, loc='lower left', bbox_to_anchor=(0.005, -0.012), ncol=2, frameon=False,
                     fontsize=5.9, handletextpad=0.3, columnspacing=1.0, handlelength=1.0)
    fig.canvas.draw()
    lx = fig.transFigure.inverted().transform((leg.get_window_extent(fig.canvas.get_renderer()).x1, 0))[0]
    fig.text(lx + 0.012, 0.012, f'{STAR} on the frontier in every seed', ha='left', va='bottom',
             fontsize=5.9, color=INK2)
    fig.text(0.988, 0.012, 'Opus ×: speed-up over the registered baseline',
             ha='right', va='bottom', fontsize=5.9, color=INK2)
    save(fig, 'cr_compare13')
    write_json(CR_DATA / 'cr_compare13.json', {
        'source': str(UNI.relative_to(ROOT)), 'order': ORDER,
        'every_seed_front': {k: sorted(v) for k, v in front.items()}, 'bpb': out,
        'train_ktok_s_300s': {r: S[r]['fixed_train_tok_s_mean'] / 1e3 for r in ORDER},
        'decode_tok_s': {r: S[r]['inf_decode_tok_s_mean'] for r in ORDER},
        'speedup_vs_bracket': {r: S[r]['inf_speedup_vs_bracket_mean'] for r in ORDER},
        'peak_allocated_mib': {r: S[r]['inf_peak_allocated_mib_mean'] for r in ORDER}})
    return front


def fmt(x, nd):
    return '--' if x is None else f'{x:.{nd}f}'.replace('-', MINUS)


def table_unified(S, f_tok, f_fix, front):
    order = ['baseline', 'arref', 'rs', 'tpe', 'motpe', 'nsgaii', 'cmaes', 'gpt_a22', 'gpt_a24', 'gpt_a25',
             'opus_a23', 'opus_a24', 'opus_a27']
    best = {}
    for k, low in [('heldout_bpb', True), ('fixed_heldout_bpb', True), ('inf_decode_tok_s', False),
                   ('inf_e2e_tok_s', False), ('inf_peak_allocated_mib', True), ('frozen_arc_easy', False),
                   ('frozen_piqa', False), ('frozen_lambada', False), ('frozen_blimp', False)]:
        nd = {'heldout_bpb': 3, 'fixed_heldout_bpb': 3, 'inf_decode_tok_s': 0, 'inf_e2e_tok_s': 0,
              'inf_peak_allocated_mib': 0}.get(k, 1)
        sc = 100 if k.startswith('frozen') else 1
        vals = {r: round(S[r][f'{k}_mean'] * sc, nd) for r in order}
        target = (min if low else max)(vals.values())
        best[k] = {r for r, v in vals.items() if v == target}
    lines = [r'\begin{tabular}{@{}lrrrrrrrrrr@{}}', r'\toprule',
             r'& & \multicolumn{2}{c}{Held-out BPB} & \multicolumn{3}{c}{Inference (1,024 in, 128 out)} & '
             r'\multicolumn{4}{c}{Zero-shot accuracy (\%)} \\',
             r'\cmidrule(lr){3-4}\cmidrule(lr){5-7}\cmidrule(l){8-11}',
             r'Recipe & Params & 201.3M tok. & 300\,s & Decode & End to end & MiB & ARC-E & PIQA & LAMB. & BLiMP \\',
             r'\midrule']
    groups = {'arref': True, 'rs': True, 'gpt_a22': True, 'opus_a23': True}
    for r in order:
        s = S[r]
        if r in groups:
            lines.append(r'\addlinespace[1.5pt]')

        def cell(k, nd, scale=1.0, sd=False):
            v = s[f'{k}_mean']
            t = '--' if v is None else f'{v * scale:.{nd}f}'
            if sd and s.get(f'{k}_sd') is not None:
                t += r'{\tiny$\pm$' + f'{s[k + "_sd"] * scale:.{nd}f}' + '}'
            t = r'\textbf{' + t + '}' if r in best.get(k, ()) else t
            return t + (r'$^\star$' if r in front.get(k, ()) else '')
        name = s['display']
        lines.append(' & '.join([
            name, f"{s['params_mean'] / 1e6:.1f}M", cell('heldout_bpb', 3, sd=True), cell('fixed_heldout_bpb', 3, sd=True),
            f"{s['inf_decode_tok_s_mean']:,.0f}" if r not in best['inf_decode_tok_s'] else r'\textbf{' + f"{s['inf_decode_tok_s_mean']:,.0f}" + '}',
            cell('inf_e2e_tok_s', 0), cell('inf_peak_allocated_mib', 0),
            cell('frozen_arc_easy', 1, 100), cell('frozen_piqa', 1, 100), cell('frozen_lambada', 1, 100),
            cell('frozen_blimp', 1, 100)]) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (GEN_CR / 'table_unified.tex').write_text('\n'.join(lines) + '\n')


def table_lmeval(S):
    order = ['baseline', 'arref', 'rs', 'tpe', 'motpe', 'nsgaii', 'cmaes', 'gpt_a22', 'gpt_a24', 'gpt_a25',
             'opus_a23', 'opus_a24', 'opus_a27']
    tasks = [('lm_arc_challenge', 'ARC-C'), ('lm_hellaswag', 'HellaSwag'), ('lm_winogrande', 'WinoGr.'),
             ('lm_openbookqa', 'OBQA'), ('lm_sciq', 'SciQ'), ('lm_boolq', 'BoolQ')]
    if not all(S[r].get(f'{k}_mean') is not None and S[r].get(f'{k}_n') == 3 for r in order for k, _ in tasks):
        return False
    lines = [r'\begin{tabular}{@{}l' + 'r' * (len(tasks) + 1) + r'@{}}', r'\toprule',
             'Recipe & ' + ' & '.join(n for _, n in tasks) + r' & WikiText ppl \\', r'\midrule']
    for r in order:
        s = S[r]
        # bold every value equal to the best at the printed precision (ties are all bold)
        best = {k: max(round(100 * S[x][k + '_mean'], 1) for x in order) for k, _ in tasks}
        bestppl = min(round(S[x]['lm_wikitext_word_ppl_mean'], 1) for x in order)
        bold = lambda t, on: r'\textbf{' + t + '}' if on else t
        vals = [bold(f"{100 * s[k + '_mean']:.1f}", round(100 * s[k + '_mean'], 1) == best[k]) for k, _ in tasks] + \
               [bold(f"{s['lm_wikitext_word_ppl_mean']:.1f}", round(s['lm_wikitext_word_ppl_mean'], 1) == bestppl)]
        lines.append(s['display'] + ' & ' + ' & '.join(vals) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (GEN_CR / 'table_unified_lmeval.tex').write_text('\n'.join(lines) + '\n')
    return True


def paired_vs_ar(rows):
    """recipe minus AR-ref, paired by seed (6-8), 95% t intervals, for both training contracts."""
    out = {}
    for key in ('heldout_bpb', 'fixed_heldout_bpb'):
        ar = {r['seed']: r[key] for r in rows if r['recipe'] == 'arref'}
        for rec in sorted({r['recipe'] for r in rows} - {'arref'}):
            d = [r[key] - ar[r['seed']] for r in sorted(rows, key=lambda x: x['seed']) if r['recipe'] == rec]
            out.setdefault(rec, {})[key] = paired(d)
    return out


def macros(S, f_tok, f_fix, have_lm, P):
    def m(name, val):
        return f'\\newcommand{{\\CRSix{name}}}{{{val}}}'
    base, opus = S['baseline'], [S[r] for r in ('opus_a23', 'opus_a24', 'opus_a27')]
    dec = [o['inf_decode_tok_s_mean'] for o in opus]
    e2e = [o['inf_e2e_tok_s_mean'] for o in opus]
    pre = [o['inf_prefill_ms_mean'] for o in opus]
    sp = [o['inf_speedup_vs_bracket_mean'] for o in opus]
    e2e_x = [x / base['inf_e2e_tok_s_mean'] for x in e2e]
    out = [
        m('DecodeLo', f'{min(dec):,.0f}'), m('DecodeHi', f'{max(dec):,.0f}'),
        m('SpeedLo', f'{min(sp):.1f}'), m('SpeedHi', f'{max(sp):.1f}'),
        m('EtoELo', f'{min(e2e):.0f}'), m('EtoEHi', f'{max(e2e):.0f}'),
        m('EtoEXLo', f'{min(e2e_x):.1f}'), m('EtoEXHi', f'{max(e2e_x):.1f}'),
        m('PrefillLo', f'{min(pre):.0f}'), m('PrefillHi', f'{max(pre):.0f}'),
        m('BasePrefill', f"{base['inf_prefill_ms_mean']:.1f}"), m('BaseDecode', f"{base['inf_decode_tok_s_mean']:.0f}"),
        m('BaseEtoE', f"{base['inf_e2e_tok_s_mean']:.0f}"), m('BaseEtoEms', f"{base['inf_e2e_ms_mean']:.0f}"),
        m('OpusEtoEms', f"{statistics.mean(o['inf_e2e_ms_mean'] for o in opus):.0f}"),
        m('GptBestDecode', f"{max(S[r]['inf_decode_tok_s_mean'] for r in ('gpt_a22', 'gpt_a24', 'gpt_a25')):.0f}"),
        m('ARHeld', f"{S['arref']['heldout_bpb_mean']:.3f}"), m('ARFixed', f"{S['arref']['fixed_heldout_bpb_mean']:.3f}"),
        m('FrontTokens', ', '.join(S[r]['display'] for r in sorted(f_tok, key=lambda r: S[r]['inf_decode_tok_s_mean']))),
        m('FrontFixed', ', '.join(S[r]['display'] for r in sorted(f_fix, key=lambda r: S[r]['inf_decode_tok_s_mean']))),
        m('Models', '13'), m('Runs', '39'),
        m('GptDecRatio', f"{S['gpt_a22']['inf_decode_tok_s_mean'] / S['arref']['inf_decode_tok_s_mean']:.1f}"),
        m('GptMemRed', f"{100 * (1 - S['gpt_a22']['inf_peak_allocated_mib_mean'] / S['arref']['inf_peak_allocated_mib_mean']):.0f}"),
        m('GptMem', f"{S['gpt_a22']['inf_peak_allocated_mib_mean']:.0f}"), m('ARMem', f"{S['arref']['inf_peak_allocated_mib_mean']:.0f}"),
        m('OpMemLo', f"{min(o['inf_peak_allocated_mib_mean'] for o in opus):.0f}"),
        m('OpMemHi', f"{max(o['inf_peak_allocated_mib_mean'] for o in opus):.0f}"),
        m('OpDecDms', f"{statistics.mean(o['inf_e2e_ms_mean'] - o['inf_prefill_ms_mean'] for o in opus):.0f}"),
        m('BaseDecDms', f"{base['inf_e2e_ms_mean'] - base['inf_prefill_ms_mean']:.0f}"),
    ]
    def d(rec, key, nd=3):
        x = P[rec][key]
        lo, hi = x['ci95']
        return f"{x['mean']:+.{nd}f}".replace('-', '$-$'), f"[{lo:+.{nd}f}, {hi:+.{nd}f}]".replace('-', '$-$')
    for rec, tag in [('gpt_a22', 'Gpt'), ('opus_a23', 'OpTwThree'), ('opus_a24', 'OpTwFour'), ('opus_a27', 'OpTwSeven'),
                     ('tpe', 'Tpe'), ('nsgaii', 'Nsga'), ('rs', 'Rs')]:
        for key, kt in [('heldout_bpb', 'Tok'), ('fixed_heldout_bpb', 'Time')]:
            mean, ci = d(rec, key)
            out += [m(f'{tag}{kt}D', mean), m(f'{tag}{kt}CI', ci)]
    tokd = {r: P[r]['heldout_bpb']['mean'] for r in P}
    worse = [r for r in P if P[r]['heldout_bpb']['ci95'][0] > 0]
    out += [m('TokWorseCount', str(len(worse))), m('TokDLo', f"{min(tokd.values()):.3f}"), m('TokDHi', f"{max(tokd.values()):.3f}"),
            m('TimeResolved', str(sum(1 for r in P if P[r]['fixed_heldout_bpb']['ci95'][1] < 0 or P[r]['fixed_heldout_bpb']['ci95'][0] > 0)))]
    (GEN_CR / 'cr_macros_p6.tex').write_text('\n'.join(out) + '\n')
    return out


def table_paired(S, P):
    """Appendix: recipe minus AR-ref, paired by seed, 95% t intervals (2 df), both contracts."""
    order = ['baseline', 'rs', 'tpe', 'motpe', 'nsgaii', 'cmaes', 'gpt_a22', 'gpt_a24', 'gpt_a25',
             'opus_a23', 'opus_a24', 'opus_a27']
    def cell(x):
        lo, hi = x['ci95']
        t = f"{x['mean']:+.3f} [{lo:+.3f}, {hi:+.3f}]".replace('-', '$-$')
        return r'\textbf{' + t + '}' if (lo > 0 or hi < 0) else t
    lines = [r'\begin{tabular}{@{}lrrrr@{}}', r'\toprule',
             r'Recipe & 201.3M tokens & Seeds worse & 300\,s & Seeds worse \\', r'\midrule']
    for r in order:
        a, b = P[r]['heldout_bpb'], P[r]['fixed_heldout_bpb']
        lines.append(f"{S[r]['display']} & {cell(a)} & {sum(d > 0 for d in a['diffs'])} of 3 & {cell(b)} & "
                     f"{sum(d > 0 for d in b['diffs'])} of 3 " + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (GEN_CR / 'table_unified_paired.tex').write_text('\n'.join(lines) + '\n')


def macros_extra(U, S):
    def m(name, val):
        return f'\\newcommand{{\\CRSix{name}}}{{{val}}}'
    def names(rs):
        n = [S[r]['display'] for r in sorted(rs, key=lambda r: S[r]['inf_decode_tok_s_mean'])]
        return n[0] if len(n) == 1 else ', '.join(n[:-1]) + ' and ' + n[-1]
    ps = U['pareto_per_seed']
    every = {k: set.intersection(*[set(v) for v in ps[k].values()]) for k in ps}
    two = {k: {r for r in S if sum(r in v for v in ps[k].values()) == 2} for k in ps}
    st, co = U['inference_stability'], U['inference_correctness']
    out = [m('FrontEveryTok', names(every['heldout_bpb'])), m('FrontTwoTok', names(two['heldout_bpb'])),
           m('FrontEveryTime', names(every['fixed_heldout_bpb'])),
           m('UnstableRuns', str(len(st['unstable_runs']))), m('UnstableMaxDev', f"{100 * st['max_rel_dev_unstable_vs_stable']:.0f}"),
           m('SeedCVMax', f"{100 * st['max_seed_cv']:.0f}"),
           m('BracketLo', f"{st['bracket_tok_s'][0]:.0f}"), m('BracketHi', f"{st['bracket_tok_s'][1]:.0f}"),
           m('CorrectRuns', str(co['passed'])), m('HardMismatches', str(co['hard_mismatches'])),
           m('NearTies', str(co['near_ties'])), m('MaxLogitErr', f"{co['max_abs_logit_error']:.2f}"),
           m('FixedMaxExcess', f"{U['fixed_max_timing_excess_s']:.2f}"),
           m('GptTrainRatio', f"{S['gpt_a22']['fixed_train_tok_s_mean'] / S['arref']['fixed_train_tok_s_mean']:.2f}"),
           m('OpTrainRatio', f"{statistics.mean(S[r]['fixed_train_tok_s_mean'] for r in ('opus_a23', 'opus_a24', 'opus_a27')) / S['arref']['fixed_train_tok_s_mean']:.1f}")]
    for k, tag in [('frozen_arc_easy', 'Arc'), ('frozen_piqa', 'Piqa'), ('frozen_lambada', 'Lamb'), ('frozen_blimp', 'Blimp')]:
        best = max(S, key=lambda r: S[r][k + '_mean'])
        out += [m(f'{tag}Best', S[best]['display']), m(f'{tag}BestVal', f"{100 * S[best][k + '_mean']:.1f}"),
                m(f'{tag}AR', f"{100 * S['arref'][k + '_mean']:.1f}")]
    # single-task leads (revision 12): margin of each leader over the runner-up, and the ARC-Easy leader over AR-ref
    marg = []
    for k in ('frozen_arc_easy', 'frozen_piqa', 'frozen_lambada'):
        v = sorted((S[r][k + '_mean'] for r in S), reverse=True)
        marg.append(100 * (v[0] - v[1]))
    arc_best = max(S, key=lambda r: S[r]['frozen_arc_easy_mean'])
    out += [m('LeadRunnerMax', f'{max(marg):.1f}'),
            m('ArcBestOverAR', f"{100 * (S[arc_best]['frozen_arc_easy_mean'] - S['arref']['frozen_arc_easy_mean']):.1f}")]
    def spearman(x, y):
        def rk(v):
            o = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
            for pos, i in enumerate(o):
                r[i] = pos
            return r
        a, b = rk(x), rk(y); n = len(x)
        return 1 - 6 * sum((p - q) ** 2 for p, q in zip(a, b)) / (n * (n * n - 1))
    if all(S[r].get('lm_wikitext_word_ppl_mean') is not None for r in S):
        R = list(S)
        bpb = [S[r]['heldout_bpb_mean'] for r in R]
        rho = {k: spearman(bpb, [S[r][k + '_mean'] for r in R]) for k in
               ('lm_wikitext_word_ppl', 'frozen_blimp', 'lm_sciq', 'lm_hellaswag', 'lm_arc_challenge', 'lm_winogrande',
                'lm_openbookqa', 'lm_boolq')}
        near = max(abs(rho[k]) for k in ('lm_hellaswag', 'lm_arc_challenge', 'lm_winogrande', 'lm_openbookqa', 'lm_boolq'))
        out += [m('RhoWiki', f"{rho['lm_wikitext_word_ppl']:.2f}"), m('RhoBlimp', f"{rho['frozen_blimp']:.2f}"),   # used inside math only (Table 9 caption): plain '-' is a math minus
                m('RhoSciq', f"{rho['lm_sciq']:.2f}".replace('-', '$-$')), m('RhoNearMax', f"{near:.2f}"),
                m('LmSDMax', f"{max(100 * S[r][k + '_sd'] for r in S for k in ('lm_sciq', 'lm_boolq', 'lm_hellaswag', 'lm_arc_challenge', 'lm_winogrande', 'lm_openbookqa')):.1f}")]
    infdir = ROOT / 'results/camera_ready_2026/phase6/inference/runs'
    sp, dc, pre = [], [], []
    for r in ('arref', 'gpt_a22', 'opus_a23'):
        x = json.loads((infdir / f'{r}-seed6-native.json').read_text())['summary']
        y = json.loads((infdir / f'{r}-seed6-native-anchor.json').read_text())['summary']
        sp.append(abs(y['speedup_vs_bracketing_baseline'] / x['speedup_vs_bracketing_baseline'] - 1))
        dc.append(abs(y['decode_tokens_per_second'] / x['decode_tokens_per_second'] - 1))
        if r == 'opus_a23':
            pre = [x['prefill_milliseconds_median'], y['prefill_milliseconds_median']]
    out += [m('AnchorSpeedMax', f"{100 * max(sp):.1f}"), m('AnchorDecodeMax', f"{100 * max(dc):.0f}"),
            m('AnchorPreA', f"{max(pre):.0f}"), m('AnchorPreB', f"{min(pre):.0f}")]
    out.append(m('AccSDMax', f"{max(100 * S[r][k + '_sd'] for r in S for k in ('frozen_arc_easy', 'frozen_piqa', 'frozen_lambada', 'frozen_blimp')):.1f}"))
    with (GEN_CR / 'cr_macros_p6.tex').open('a') as f:
        f.write('\n'.join(out) + '\n')
    return out


def main():
    U, S, rows = load()
    front = fig_compare13(U, S, rows)
    f_tok, f_fix = fig_unified(S, rows, frozenset(tuple(x) for x in U['inference_stability']['unstable_runs']))
    stab, stab_mac = stability_check(U, S, rows)
    table_unified(S, f_tok, f_fix, front)
    have_lm = table_lmeval(S)
    P = paired_vs_ar(rows)
    out = macros(S, f_tok, f_fix, have_lm, P)
    out += macros_extra(U, S)
    # recipes on the frontier of the seed means but not on the frontier in every seed (appendix scatter caption)
    def names(rs):
        n = [S[r]['display'] for r in sorted(rs, key=lambda r: S[r]['inf_decode_tok_s_mean'])]
        return n[0] if len(n) == 1 else ', '.join(n[:-1]) + ' and ' + n[-1]
    extra = [f"\\newcommand{{\\CRSixFrontMeanOnlyTok}}{{{names(set(f_tok) - front['heldout_bpb'])}}}",
             f"\\newcommand{{\\CRSixFrontMeanOnlyTime}}{{{names(set(f_fix) - front['fixed_heldout_bpb'])}}}"] + stab_mac
    # post hoc multiplicity check (revision 12): Bonferroni over the 12 comparisons per contract, t with 2 df
    # (closed form for df = 2: t_p = q * sqrt(2 / (1 - q^2)), q = 2p - 1). Reported next to the per-comparison CIs.
    def t2(pp):
        q = 2 * pp - 1
        return q * math.sqrt(2 / (1 - q * q))
    tb = t2(1 - 0.025 / 12)
    bonf = {}
    for key in ('heldout_bpb', 'fixed_heldout_bpb'):
        ar = {x['seed']: x[key] for x in rows if x['recipe'] == 'arref'}
        keep, lost = [], []
        for r in ORDER:
            if r == 'arref':
                continue
            dd = [x[key] - ar[x['seed']] for x in rows if x['recipe'] == r]
            mm, se = statistics.mean(dd), statistics.stdev(dd) / math.sqrt(len(dd))
            unadj = P[r][key]['ci95'][0] > 0
            (keep if mm - tb * se > 0 else lost if unadj else []).append(r)
        bonf[key] = {'t_crit': tb, 'resolved': keep, 'lost': lost}
    nm = lambda rs: ', '.join(S[r]['display'] for r in rs[:-1]) + ' and ' + S[rs[-1]]['display'] if len(rs) > 1 else S[rs[0]]['display']
    extra += [f"\\newcommand{{\\CRSixBonfTok}}{{{len(bonf['heldout_bpb']['resolved'])}}}",
              f"\\newcommand{{\\CRSixBonfTokLost}}{{{nm(bonf['heldout_bpb']['lost'])}}}",
              f"\\newcommand{{\\CRSixBonfTime}}{{{len(bonf['fixed_heldout_bpb']['resolved'])}}}",
              f"\\newcommand{{\\CRSixBonfTimeKept}}{{{nm(bonf['fixed_heldout_bpb']['resolved']).replace('Baseline', 'the baseline')}}}",
              f"\\newcommand{{\\CRSixBonfT}}{{{tb:.1f}}}"]
    # recipes resolved as worse than AR-ref at 300 s (per-comparison 95% t interval above zero), for Part 1
    worse = [r for r in ORDER if r != 'arref' and P[r]['fixed_heldout_bpb']['ci95'][0] > 0]
    wn = [S[r]['display'] if r != 'baseline' else 'the baseline' for r in worse]
    if len(worse) != sum(1 for r in P if P[r]['fixed_heldout_bpb']['ci95'][0] > 0):
        raise SystemExit('worse-at-300s list incomplete')
    extra.append(f"\\newcommand{{\\CRSixTimeWorseNames}}{{{', '.join(wn[:-1]) + ' and ' + wn[-1]}}}")
    with (GEN_CR / 'cr_macros_p6.tex').open('a') as f:
        f.write('\n'.join(extra) + '\n')
    out += extra
    table_paired(S, P)
    write_json(CR_DATA / 'cr_p6.json', {'front_tokens': f_tok, 'front_fixed': f_fix, 'every_seed_front': {k: sorted(v) for k, v in front.items()},
                                        'lmeval_table': have_lm, 'paired_vs_arref': P, 'stability_check': stab, 'bonferroni': bonf,
                                        'source': str(UNI.relative_to(ROOT))})
    print(json.dumps({'front_tokens': f_tok, 'front_fixed': f_fix, 'lmeval_table': have_lm}))
    print('\n'.join(out))


if __name__ == '__main__':
    main()
