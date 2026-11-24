#!/usr/bin/env python3
"""Camera-ready revision 8 (2026-11-15): results overview, three-agent research comparison and the compact
verification table for Part 1.

Reads only committed analysis outputs (paper/camera_ready/data/cr_v2.json, cr_stats.json, cr_evidence.json,
lineage_edit_labels.json), the read-only experiment-database snapshot (AutoResearch lineage), the read-only
extended-experiment attempt records (history.jsonl of the GPT-5.6 and Opus arms) and the raw Opus
re-measurement records (via cr_v3.p2_raw_decode, SHA256SUMS-checked).  Launches nothing; no GPU; no network.

Writes
  paper/figures/cr_overview.{pdf,png}        Figure 1: every verified representative under its own protocol
  paper/figures/cr_tracks.{pdf,png}          Figure 2: how the agents searched (recorded parent links only)
  paper/generated/camera_ready/table_verify_main.tex   Table 1: matched-token quality and downstream tasks
  paper/generated/camera_ready/table_milestones.tex    appendix: milestone hypotheses (verbatim) and outcomes
  paper/generated/camera_ready/cr_macros_r8.tex        numbers quoted in the revised Part 1
  paper/camera_ready/data/cr_overview.json, cr_tracks.json   every plotted value with its source
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

from cr_common import CR_DATA, GEN_CR, connect_ro, read_json, write_json
from cr_v2 import (AQUA, BLUE, CODEX_ROOT, GRAY, GRID, INK, INK2, MUTED, OPUS_ROOT, ORANGE, VIOLET, save, style)
from cr_v3 import p2_raw_decode, tex_escape

# One colour per approach, used in every revision-8 figure.
COL = {'base': GRAY, 'ar': ORANGE, 'cls': AQUA, 'gpt': BLUE, 'opus': VIOLET}
LIGHT = {'ar': '#f6c2a9', 'gpt': '#a9c9ef', 'opus': '#c3bcf0', 'cls': '#a7e1cb', 'base': '#cfcecb'}


# ============================================================================ Figure 1: results overview
def overview_data(D, S, R):
    """Every value plotted in Figure 1, with its source.  Two blocks, each internally consistent:
    original experiments (201.3M-token panel, seeds 3-5, decode benchmark 256 in / 256 out) and
    extended experiments (201.3M-token retraining on seeds 6-8; decode benchmark 1,024 in / 128 out,
    fresh-A40 re-measurement read from the raw repeat records)."""
    t1 = S['table1']
    panel_src = 'cr_stats.json table1: 201.3M-token panel, means over seeds 3-5'
    top = []
    for key, name, kind in (('baseline', 'Baseline', 'base'), ('autoresearch_compatible', 'AR-ref', 'ar'),
                            ('tpe_quality', 'TPE', 'cls'), ('motpe_quality', 'MOTPE', 'cls'),
                            ('autopareto_292', 'AP-Q', 'gpt')):
        x = t1[key]
        top.append({'name': name, 'kind': kind, 'params_M': x['params'] / 1e6,
                    'heldout': x['heldout_bpb']['mean'], 'heldout_seeds': x['heldout_bpb']['values'],
                    'train_tok_s': x['tokens'] / x['steady_s']['mean'],
                    'decode': x['decode_tok_s']['mean'], 'mem': x['peak_alloc_mib']['mean'], 'source': panel_src})
    m = D['phase5']['matched_tokens']['recipes']
    ret_src = 'cr_v2.json phase5.matched_tokens: 201.3M-token retraining, seeds 6-8 (AR-ref, AP-Q: E1 runs)'
    pb = read_json(CR_DATA / 'cr_comparison.json')['panel_b']['Registered baseline']
    base_lo = min(min(R[a]['baseline_values']) for a in R)
    base_hi = max(max(R[a]['baseline_values']) for a in R)
    bot = [{'name': 'Baseline', 'kind': 'base', 'params_M': 50.332176, 'heldout': None, 'train_tok_s': None,
            'decode': pb['decode_tok_s'], 'decode_lo': base_lo, 'decode_hi': base_hi, 'mem': pb['peak_mib'],
            'source': 'registered baseline: calibration-run decode and memory; bracketing-baseline range from the '
                      'raw re-measurement repeats; not retrained at 201.3M tokens'}]
    for key, name, kind in (('AR-ref', 'AR-ref', 'ar'), ('AP-Q', 'AP-Q', 'gpt')):
        x = m[key]
        bot.append({'name': name, 'kind': kind, 'params_M': x['params'] / 1e6, 'heldout': x['heldout']['mean'],
                    'heldout_seeds': x['heldout']['values'], 'train_tok_s': x['tok_s'], 'decode': None, 'mem': None,
                    'source': ret_src + '; decode benchmark of the extended experiments not run on this recipe'})
    for a in ('23', '24', '27'):
        x = m[f'OPUS-A{a}']
        bot.append({'name': f'Opus A{a}', 'kind': 'opus', 'params_M': x['params'] / 1e6,
                    'heldout': x['heldout']['mean'], 'heldout_seeds': x['heldout']['values'],
                    'train_tok_s': x['tok_s'], 'decode': R[a]['decode_tok_s'], 'speedup': R[a]['decode_x'],
                    'mem': D['agents']['opus']['winners'][a]['peak_mib'],
                    'source': ret_src + '; decode and memory: raw repeat records of the fresh-A40 re-measurement '
                              '(median of 3 repeats, SHA256SUMS-checked)'})
    # facts the text quotes
    ar, apq = top[1], top[4]
    assert abs(ar['train_tok_s'] - 289195) < 30 and abs(apq['train_tok_s'] - 522900) < 60, (ar, apq)
    return {'top': top, 'bottom': bot,
            'train_ratio_apq_ar': apq['train_tok_s'] / ar['train_tok_s'],
            'decode_ratio_apq_ar': apq['decode'] / ar['decode'],
            'mem_reduction_apq_ar_pct': 100 * (1 - apq['mem'] / ar['mem'])}


def fig_overview(D, S, R):
    """Figure 1.  Rows: verified representatives.  Columns: held-out BPB at 201.3M tokens, training throughput,
    decode throughput, inference memory.  Each block has its own decode and memory axes (different benchmarks);
    nothing is drawn where a recipe was not measured."""
    O = overview_data(D, S, R)
    style(1.0)
    top, bot = O['top'], O['bottom']
    nt, nb = len(top), len(bot)
    fig = plt.figure(figsize=(5.5, 2.06))
    gs = fig.add_gridspec(2, 4, height_ratios=[nt + 0.2, nb + 0.2], hspace=0.66, wspace=0.15,
                          left=0.18, right=0.985, top=0.745, bottom=0.135)
    fs, fsl, fsn = 5.8, 6.2, 5.3
    cols = [('Held-out BPB', 'after 201.3M training tokens', 'lower is better'),
            ('Training speed', 'thousand tokens/s', 'higher is better'),
            ('Decode speed', 'generated tokens/s, batch 1', 'higher is better'),
            ('Inference memory', 'peak allocated MiB', 'lower is better')]
    xlim_q = (1.088, 1.192)
    xlim_t = (0, 1200)
    axes = {}

    def value(ax, x, i, txt, extra=None, ecol=INK):
        """Value label just right of the mark; an optional comparison in the approach colour after it."""
        an = ax.annotate(txt, (x, i), xytext=(2.2, 0), textcoords='offset points', ha='left', va='center',
                         fontsize=fs, color=INK, zorder=6)
        if extra:
            ax.annotate(extra, (x, i), xytext=(2.2 + 3.15 * len(txt.replace('$', '').replace('\\times', 'x')) + 2.5, 0),
                        textcoords='offset points', ha='left', va='center', fontsize=fsn, color=ecol, zorder=6)
        return an

    def missing(ax, i, txt):
        ax.text(0.03, i, txt, transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize=fsn,
                color=MUTED, style='italic')

    for bi, (rows, tag) in enumerate(((top, 'top'), (bot, 'bot'))):
        n = len(rows)
        for ci in range(4):
            ax = fig.add_subplot(gs[bi, ci])
            axes[(tag, ci)] = ax
            ax.set_ylim(n - 0.45, -0.55)
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='x', length=2, pad=1.5, labelsize=5.5)
            ax.grid(axis='x', color=GRID, lw=0.45, zorder=0)
            ax.set_axisbelow(True)
            for i in range(n):
                if i % 2 == 0:
                    ax.axhspan(i - 0.5, i + 0.5, color='#f7f7f5', zorder=0, lw=0)
        # row labels: approach colour, name, size
        a0 = axes[(tag, 0)]
        for i, r in enumerate(rows):
            a0.add_patch(plt.Rectangle((-0.84, i - 0.32), 0.022, 0.64, transform=a0.get_yaxis_transform(),
                                       color=COL[r['kind']], clip_on=False, lw=0))
            a0.text(-0.805, i, r['name'], transform=a0.get_yaxis_transform(), ha='left', va='center', fontsize=fsl,
                    color=INK, fontweight='bold' if r['name'] in ('AR-ref', 'AP-Q', 'Opus A23') else 'normal')
            a0.text(-0.025, i, f"{r['params_M']:.1f}M", transform=a0.get_yaxis_transform(), ha='right',
                    va='center', fontsize=5.5, color=MUTED)
        # (1) held-out BPB: dots, hollow seeds, filled mean
        ax = axes[(tag, 0)]
        ax.set_xlim(*xlim_q)
        ax.set_xticks([1.10, 1.14, 1.18])
        for i, r in enumerate(rows):
            if r['heldout'] is None:
                missing(ax, i, 'not retrained')
                continue
            for v in r.get('heldout_seeds', []):
                ax.scatter([v], [i], s=7, facecolors='none', edgecolors=COL[r['kind']], linewidths=0.5, zorder=3)
            ax.scatter([r['heldout']], [i], s=16, color=COL[r['kind']], zorder=4, edgecolors='white', linewidths=0.4)
            value(ax, max([r['heldout']] + list(r.get('heldout_seeds', []))), i, f"{r['heldout']:.4f}")
        # (2) training speed
        ax = axes[(tag, 1)]
        ax.set_xlim(*xlim_t)
        ax.set_xticks([0, 500, 1000])
        for i, r in enumerate(rows):
            if r['train_tok_s'] is None:
                missing(ax, i, 'not retrained')
                continue
            k = r['train_tok_s'] / 1e3
            ax.barh(i, k, height=0.58, color=COL[r['kind']], zorder=3, lw=0)
            extra = f"{O['train_ratio_apq_ar']:.2f}$\\times$ AR-ref" if (tag == 'top' and r['name'] == 'AP-Q') else None
            value(ax, k, i, f'{k:.0f}', extra, BLUE)
        # (3) decode speed and (4) memory: own axis per block, because the benchmarks differ
        ax, mx = axes[(tag, 2)], axes[(tag, 3)]
        if tag == 'top':
            ax.set_xlim(0, 900)
            ax.set_xticks([0, 400, 800])
            mx.set_xlim(0, 430)
            mx.set_xticks([0, 200, 400])
        else:
            ax.set_xlim(0, 9600)
            ax.set_xticks([0, 4000, 8000])
            ax.set_xticklabels(['0', '4,000', '8,000'])
            mx.set_xlim(0, 430)
            mx.set_xticks([0, 200, 400])
        for i, r in enumerate(rows):
            if r['decode'] is None:
                missing(ax, i, 'not run on this benchmark')
                missing(mx, i, 'not measured')
                continue
            ax.barh(i, r['decode'], height=0.58, color=COL[r['kind']], zorder=3, lw=0)
            if 'decode_lo' in r:
                ax.plot([r['decode_lo'], r['decode_hi']], [i, i], color=INK2, lw=0.8, zorder=4, solid_capstyle='butt')
                value(ax, r['decode_hi'], i, f"{r['decode_lo']:.0f} to {r['decode_hi']:.0f}", 'bracketing runs', MUTED)
            elif 'speedup' in r:
                value(ax, r['decode'], i, f"{r['decode']:,.0f}", f"{r['speedup']:.1f}$\\times$", VIOLET)
            elif tag == 'top' and r['name'] == 'AP-Q':
                value(ax, r['decode'], i, f"{r['decode']:.0f}", f"{O['decode_ratio_apq_ar']:.2f}$\\times$ AR-ref", BLUE)
            else:
                value(ax, r['decode'], i, f"{r['decode']:.0f}")
            mx.barh(i, r['mem'], height=0.58, color=COL[r['kind']], zorder=3, lw=0)
            if tag == 'top' and r['name'] == 'AP-Q':
                value(mx, r['mem'], i, f"{r['mem']:.0f}", f"$-${O['mem_reduction_apq_ar_pct']:.0f}% vs AR-ref", BLUE)
            elif tag == 'bot' and r['name'] == 'Baseline':
                value(mx, r['mem'], i, f"{r['mem']:.0f}")
            else:
                value(mx, r['mem'], i, f"{r['mem']:.0f}")
    # column titles once at the top; block headers above each block
    for ci, (t1_, t2_, t3_) in enumerate(cols):
        b = axes[('top', ci)].get_position()
        xc = (b.x0 + b.x1) / 2
        H = fig.get_figheight() * 72
        fig.text(xc, 1.0, t1_, ha='center', va='top', fontsize=6.5, color=INK, fontweight='bold')
        fig.text(xc, 1.0 - 8.6 / H, f'{t2_}\n{t3_}', ha='center', va='top', fontsize=5.4, color=MUTED, linespacing=1.1)
    hdr = {'top': 'Original experiments (Jul to Aug): panel seeds 3 to 5; decode benchmark 256 tokens in, 256 out',
           'bot': 'Extended experiments (Sep to Nov): retraining seeds 6 to 8; decode benchmark 1,024 in, 128 out'}
    for tag in ('top', 'bot'):
        b = axes[(tag, 0)].get_position()
        fig.text(0.008, b.y1 + 0.013, hdr[tag], ha='left', va='bottom', fontsize=6.2, color=INK2,
                 fontweight='bold')
    fig.text(0.008, 0.004, 'Bars and filled dots: means over three seeds (hollow dots: single seeds). Decode and '
             'memory have a separate axis per block because the two benchmarks differ.', fontsize=5.3,
             color=MUTED, ha='left', va='bottom')
    save(fig, 'cr_overview')
    write_json(CR_DATA / 'cr_overview.json', O)
    return O


# ============================================================================ Figure 2: how the agents searched
def _hist(root):
    return {json.loads(l)['campaign_attempt']: json.loads(l) for l in (root / 'history.jsonl').read_text().splitlines()}


def _parent_attempt(pid):
    if not pid:
        return None
    m = re.search(r'attempt-(\d+)$', pid)
    return int(m.group(1)) if m else 0


def _txt(rec):
    return ' '.join(' '.join(str(rec.get(k) or '') for k in ('hypothesis', 'intended_change')).split())


def sg(v, nd=3):
    """Signed number with a true minus sign (U+2212) for figure text."""
    s = f'{v:+.{nd}f}'
    return s.replace('-', '\u2212')


def _need(cond, msg):
    if not cond:
        raise AssertionError(msg)


def track_extended(root, name):
    """Nodes of one extended-experiment arm, read from its history.jsonl."""
    H = _hist(root)
    nodes = {}
    for a, d in sorted(H.items()):
        r = d['result']
        ir = r.get('inference_result') or {}
        st = d.get('status')
        q = r.get('quality_status')
        state = ('failed' if st != 'completed' else 'dominated' if q == 'dominated_valid' else
                 'uncertain' if q == 'quality_valid_speed_uncertain' else 'valid')
        nodes[a] = {'x': a, 'parent': _parent_attempt(d.get('parent_experiment_id')), 'state': state,
                    'category': d.get('change_category') or '', 'val_bpb': r.get('validation_bpb'),
                    'speedup': ir.get('normalized_decode_throughput'), 'abs_tok_s': ir.get('absolute_candidate_tok_s'),
                    'peak_mib': (ir.get('peak_allocated_inference_memory_bytes') or 0) / 2 ** 20 or None,
                    'params': r.get('num_params'), 'total_batch': r.get('total_batch_size'),
                    'train_tok_s': r.get('training_throughput_tokens_per_second'), 'text': _txt(d),
                    'source': f'{root.relative_to(root.parents[2])}/history.jsonl attempt {a}'}
    return nodes


def track_apq():
    ev = read_json(CR_DATA / 'cr_evidence.json')
    lin = {r['id']: r for r in ev['seed0_lineage']}
    num = {i: r['experiment_number'] for i, r in lin.items() if i != 8}
    nodes = {}
    for i, r in lin.items():
        if i == 8:
            continue
        p = r['parent_experiment_id']
        nodes[num[i]] = {'x': num[i], 'parent': 0 if p == 8 else num[p], 'id': i,
                         'state': 'failed' if r['status'] != 'completed' else
                         ('dominated' if r['decision'] == 'dominated' else 'valid'),
                         'category': r['change_category'] or '', 'val_bpb': r['val_bpb'],
                         'decode': r['inference_tokens_per_second'], 'params': r['parameter_count'],
                         'tokens': r['total_training_tokens'], 'text': ' '.join((r['hypothesis'] or '').split()),
                         'source': f'cr_evidence.json seed0_lineage id {i} (exp {i})'}
    return nodes, lin[8]


def track_ar():
    con = connect_ro()
    rows = con.execute("select id, campaign_id, experiment_number, status, val_bpb, decision, parent_experiment_id, "
                       "hypothesis, parameter_count from experiments where campaign_id in "
                       "('autoresearch-seed42-20260723','autoresearch-diagnostic5-seed42-20260724') order by id").fetchall()
    base = con.execute("select id, val_bpb from experiments where id = 13").fetchone()
    pos = {}
    for r in rows:
        pos[r['id']] = r['experiment_number'] + (50 if r['campaign_id'].startswith('autoresearch-diag') else 0)
    nodes = {}
    for r in rows:
        p = r['parent_experiment_id']
        nodes[pos[r['id']]] = {'x': pos[r['id']], 'parent': 0 if p == 13 else (pos[p] if p in pos else None),
                               'id': r['id'], 'state': 'failed' if r['status'] in ('crash', 'interrupted') else 'valid',
                               'status': r['status'], 'decision': r['decision'], 'val_bpb': r['val_bpb'],
                               'params': r['parameter_count'], 'text': ' '.join((r['hypothesis'] or '').split()),
                               'source': f"downstream_200m.db experiments id {r['id']} ({r['campaign_id']} "
                                         f"#{r['experiment_number']})"}
    return nodes, dict(base)


def _path(nodes, end):
    out, a = [], end
    while a not in (None, 0):
        out.append(a)
        a = nodes[a]['parent']
    return out[::-1]


def tracks_data():
    """Every node of the four tracks, the recorded parent of each, and the milestone labels.  Each label is
    checked against the record it describes."""
    # ---------------- AutoResearch (quality only), seed 42: 50-attempt campaign + 5-attempt follow-up
    ar, ar_base = track_ar()
    ar_path = _path(ar, 55)
    _need(ar_path == [4, 5, 6, 9, 13, 15, 16, 19, 21, 48, 51, 52, 55], ar_path)
    _need(all(ar[a]['params'] == 50332176 for a in ar_path), 'AR-ref path keeps the 50.3M model')
    _need('more training tokens' in ar[4]['text'] and 'Halving the total batch to 262144' in ar[5]['text'], 'AR 4/5')
    _need('Halving the retained winner batch from 262K to 131K' in ar[6]['text'], 'AR 6')
    _need('converting redundant attention compute into more training tokens' in ar[4]['text'], 'AR #4 quote')
    _need(ar[55]['id'] == 68 and ar[52]['id'] == 65, 'AR-ref is exp 68, parent exp 65')
    _need(ar[51]['status'] == 'crash' and ar[52]['status'] == 'crash', 'D1, D2 failed the old cache check')
    ar_fail = sum(1 for a in ar if a <= 50 and ar[a]['state'] == 'failed')
    T_ar = {'key': 'ar', 'nodes': ar, 'path': ar_path, 'yscale': 0.5, 'n': 55,
            'head': ('AutoResearch (GPT-5.6)', 'quality only; seed 42;\n50 + 5 attempts'),
            'labels': {48: '#48 Newton-Schulz steps'},
            'groups': [(4, 6, f"#4 short windows;\n#5, #6 batch ½ twice:\nval {ar[4]['val_bpb']:.3f} to {ar[6]['val_bpb']:.3f}"),
                       (9, 21, "#9 to #21: six optimizer\nand schedule tweaks"),
                       (51, 55, f"follow-up D1 to D5:\nfinal LR floor; D5 is\nAR-ref, val {ar[55]['val_bpb']:.4f}")],
            'notes': [(17.5, f"{ar_fail} failed runs (crosses),\nmost at a cache check\nlater found faulty")],
            'ticks': [(1, '1'), (10, '10'), (20, '20'), (30, '30'), (40, '40'), (50, '50'), (51, 'D1'), (55, 'D5')],
            'ar_fail': ar_fail}
    # ---------------- AutoPareto GPT-5.6, original, seed 0 (AP-Q lineage)
    ap, ap_base = track_apq()
    ap_path = _path(ap, 22)
    _need(ap_path == [1, 4, 6, 7, 8, 18, 19, 20, 22], ap_path)
    _need(ap[1]['text'].startswith('Reducing total batch size from 2^19 to 2^17'), 'AP A1')
    _need(ap[8]['val_bpb'] - ap[7]['val_bpb'] > 0.06 and ap[18]['parent'] == 8, 'AP A8 worse, A18 from A8')
    _need(ap[24]['parent'] == 22 and ap[25]['parent'] == 24 and ap[17]['state'] == 'failed', 'AP-S/AP-F, A17')
    lab = {x['exp']: x['label'] for x in read_json(CR_DATA / 'lineage_edit_labels.json')['labels']}
    for a_, frag in ((1, 'batch $2^{19}\\to2^{17}$'), (4, 'depth 8$\\to$7'), (6, 'width 512$\\to$384'),
                     (7, 'depth 6$\\to$5'), (8, 'width 384$\\to$256'), (18, 'width 256$\\to$512'),
                     (19, 'matrix LR'), (20, 'embed LR'), (22, 'head LR'), (24, 'depth 4$\\to$3'),
                     (25, 'width 384$\\to$512')):
        _need(frag in lab[ap[a_]['id']], (a_, frag, lab[ap[a_]['id']]))
    T_ap = {'key': 'gpt', 'nodes': ap, 'path': ap_path, 'n': 25,
            'head': ('AutoPareto (GPT-5.6)', 'original harness; seed 0;\n25 attempts'),
            'labels': {1: f"A1 batch ¼: val {sg(ap[1]['val_bpb'] - ap_base['val_bpb'])}",
                       4: 'A4 depth 7', 6: 'A6 depth 6, width 384', 7: 'A7 depth 5',
                       8: f"A8 depth 4, width 256:\nval {sg(ap[8]['val_bpb'] - ap[7]['val_bpb'])} (worse)",
                       17: 'A17 crash (infrastructure)',
                       18: 'A18 width 512 again',
                       19: 'A19 matrix LR', 20: 'A20 embedding LR',
                       22: f"A22 head LR: AP-Q\nval {ap[22]['val_bpb']:.4f}, {ap[22]['decode']:.0f} tok/s",
                       24: 'A24 depth 3: AP-S', 25: 'A25 width 512: AP-F'},
            'groups': [], 'bold': (22,)}
    # ---------------- AutoPareto GPT-5.6, extended harness, seed 3
    cx = track_extended(CODEX_ROOT, 'GPT-5.6')
    cx_path = _path(cx, 22)
    _need(cx_path == [1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 21, 22], cx_path)
    _need(not any(n['category'].startswith('inference') for n in cx.values()), 'GPT-5.6 made no inference edit')
    _need(not any(re.search(r'adapter|graph|compile', n['text'], re.I) for n in cx.values()), 'no adapter text')
    _need(cx[1]['speedup'] < 1 and 'grouped-query' in cx[1]['text'], 'cx A1')
    _need(cx[12]['val_bpb'] - cx[11]['val_bpb'] < -0.06 and cx[12]['total_batch'] == 131072, 'cx A12')
    _need(cx[13]['params'] < 16e6 and cx[14]['params'] > 24e6, 'cx A13/A14')
    best_cx = min((n for n in cx.values() if n['val_bpb']), key=lambda n: n['val_bpb'])
    _need(best_cx['x'] == 14, 'cx best val is A14')
    T_cx = {'key': 'gpt', 'nodes': cx, 'path': cx_path, 'n': 22, 'light_after': 14,
            'head': ('AutoPareto (GPT-5.6)', 'extended harness; seed 3;\n22 attempts'),
            'labels': {1: f"A1 2 KV heads: {cx[1]['speedup']:.2f}$\\times$", 2: 'A2 batch ½', 3: 'A3 depth 7',
                       4: 'A4 MLP 3$\\times$',
                       12: f"A12 batch ½: val {sg(cx[12]['val_bpb'] - cx[11]['val_bpb'])}",
                       13: 'A13 depth 6, width 384',
                       14: f"A14 width 512 again:\nval {cx[14]['val_bpb']:.4f}, {cx[14]['speedup']:.2f}$\\times$",
                       17: 'A17 MLP 2$\\times$',
                       18: f"A18 depth 5: {cx[18]['speedup']:.2f}$\\times$", 22: 'A22 head dim 96'},
            'groups': [(5, 11, 'value embeddings, windows')],
            'end_note': 'stopped by a sandbox failure;\nno inference-code edit', 'bold': (14,)}
    # ---------------- AutoPareto Claude Opus 5.5, extended harness, seed 3
    op = track_extended(OPUS_ROOT, 'Opus')
    _need([a for a, n in op.items() if n['state'] == 'failed'] == [2, 9], 'Opus failures')
    _need(op[2]['parent'] == 1 and op[9]['parent'] == 8 and op[6]['parent'] == 5 and op[7]['parent'] == 5, 'Opus parents')
    _need(op[24]['parent'] == 23 and op[26]['parent'] == 23 and op[27]['parent'] == 26 and op[25]['parent'] == 23, 'tail')
    _need(op[1]['total_batch'] == 262144 and op[7]['total_batch'] == 131072, 'Opus batch halvings')
    _need([round(op[a]['params'] / 1e6, 1) for a in (3, 4, 5, 6)] == [39.8, 29.4, 26.2, 18.9], 'Opus depth cuts')
    _need('Batch-1 cached decode in attempt 8 is launch-bound' in op[10]['text'], 'Opus A10')
    _need('CUDA-graph decode will be the next isolated speed change' in op[8]['text'], 'Opus A8 plan')
    _need(op[8]['category'].startswith('inference') and op[14]['category'] == 'inference_adapter_decode_speed', 'A8/A14')
    _need(op[18]['category'] == 'inference_kv_cache' and op[19]['category'] == 'inference_decode_attention_kernel', '18/19')
    _need(op[22]['category'] == 'inference_decode_kernel_fusion', 'A22')
    _need('QK_GAIN = 1.35' in op[20]['text'] and 'short_window = long_window // 8' in op[17]['text'], 'A17/A20')
    _need('MATRIX_LR = 0.04 * 2**0.5' in op[15]['text'] and 'N_KV_HEAD = 2' in op[16]['text'], 'A15/A16')
    _need('PrecisionLinear' in op[13]['text'] and '_get_capture_stream' in op[11]['text'], 'A11/A13')
    _need(op[12]['total_batch'] == 65536 and op[12]['val_bpb'] > op[11]['val_bpb'], 'A12')
    _need(op[25]['state'] == 'dominated' and op[25]['val_bpb'] > op[23]['val_bpb'], 'A25')
    _need('flash_attn_func' in op[26]['text'] and 'last' in op[21]['text'].lower(), 'A21/A26')
    _need(cx[22]['text'].find('HEAD_DIM from 128 to 96') >= 0 and cx[17]['text'].find('2 * config.n_embd') >= 0, 'cx A17/A22')
    _need('MLP dimensions from 4 * config.n_embd to 3 * config.n_embd' in cx[4]['text'] and 'DEPTH from 8 to 7' in cx[3]['text'], 'cx A3/A4')
    _need('ASPECT_RATIO from 64 to 80' in cx[14]['text'] and 'DEPTH from 6 to 5' in cx[18]['text'], 'cx A14/A18')
    _need('TOTAL_BATCH_SIZE from 2**19 to 2**18' in cx[2]['text'], 'cx A2')
    _need('preserving the architecture-dependent decode speed' in cx[12]['text'], 'cx A12 quote')
    _need('If this candidate is valid, CUDA-graph decode will be the next isolated speed change.' in op[8]['text'], 'A8 q')
    _need('Batch-1 cached decode in attempt 8 is launch-bound' in op[10]['text'], 'A10 quote')
    _need('token-shift' in op[23]['text'] and 'int8' in op[27]['text'] and 'ONE shared table' in op[24]['text'], 'tail')
    op_path = _path(op, 23)
    _need(op_path == [1, 3, 4, 5, 7, 8, 10, 11, 13, 14, 17, 18, 19, 20, 21, 22, 23], op_path)
    sx = lambda a: f"{op[a]['speedup']:.1f}$\\times$"
    T_op = {'key': 'opus', 'nodes': op, 'path': op_path + [26, 27], 'n': 27, 'frontier': [23, 24, 27],
            'squares': True, 'head': ('AutoPareto (Claude Opus 5.5)', 'extended harness; seed 3;\n27 attempts'),
            'labels': {1: 'A1 batch ½', 2: 'A2 depth 6: run failed', 3: 'A3 depth 6', 4: 'A4 depth 4', 5: 'A5 depth 3',
                       6: f"A6 depth 2: val {sg(op[6]['val_bpb'] - op[5]['val_bpb'])}",
                       7: 'A7 from A5: batch ½',
                       8: 'A8 own inference adapter', 9: 'A9 graph: check failed',
                       10: f'A10 CUDA graph replay: {sx(10)}', 11: 'A11 graph memory',
                       12: 'A12 batch ½ again: worse', 13: 'A13 bf16 weights',
                       14: f'A14 compiled fusion: {sx(14)}', 15: 'A15 higher matrix LR: worse',
                       16: 'A16 2 KV heads: worse', 17: 'A17 short windows',
                       18: f'A18 ring K/V cache: {sx(18)}', 19: f'A19 attention kernel: {sx(19)}',
                       20: f"A20 QK gain: val {op[20]['val_bpb']:.4f}", 21: 'A21 last-row logits',
                       22: f'A22 fused decode step: {sx(22)}',
                       23: f"A23 token shift: val {op[23]['val_bpb']:.4f}",
                       24: f"A24 shared table: {op[24]['peak_mib']:.0f} MiB", 25: 'A25 output gate: worse',
                       26: 'A26 FA3 prefill', 27: f"A27 int8 tables: {op[27]['peak_mib']:.0f} MiB"},
            'groups': [], 'bold': (23, 24, 27),
            'bands': [(1, 7, 'training'), (8, 22, 'inference code'), (23, 27, 'frontier')]}
    return [T_ar, T_ap, T_cx, T_op]


def draw_track(ax, T, fs=5.1):
    """One research process as a vertical decision log: one row per attempt (top to bottom), the recorded parent
    path on the main line, other attempts slightly left.  An edge is drawn only for a recorded parent link that
    passes no other drawn node."""
    col, lcol = COL[T['key']], LIGHT[T['key']]
    nodes, path = T['nodes'], set(T['path'])
    ys = T.get('yscale', 1.0)
    light_after = T.get('light_after')
    XP, XO, XL = 0.07, 0.025, 0.115            # path line, other attempts, label start (axes fraction)
    Y = lambda a: 0 if a == 0 else nodes[a]['x'] * ys
    X = lambda a: XP if (a == 0 or a in path) else XO
    ax.set_xlim(0, 1)
    ax.set_ylim(28.2, -3.55)
    ax.axis('off')
    for a0, a1, lab in T.get('bands', []):
        ax.add_patch(plt.Rectangle((0.0, a0 - 0.5), 1.0, a1 - a0 + 1, color='#f3f1fb' if 'inference' in lab else '#f6f6f3',
                                   zorder=0, lw=0))
        ax.text(0.99, (a0 + a1) / 2, lab, ha='right', va='center', rotation=90, fontsize=4.9, color=MUTED,
                style='italic', zorder=1)
    occ = {a: Y(a) for a in nodes if X(a) == XO}

    def clear(a, b):
        lo, hi = sorted((Y(a), Y(b)))
        return not any(lo < y < hi for y in occ.values())
    for a, n in nodes.items():
        p = n['parent']
        if p is None:
            continue
        on = a in path
        if not on and not (p == 0 or p in path) and not clear(p, a):
            continue
        if not on and T.get('key') == 'ar':
            continue
        c = (lcol if (light_after and a > light_after) else col) if on else '#bdbcb8'
        ax.plot([X(p), X(a)], [Y(p), Y(a)], color=c, lw=1.5 if on else 0.55, zorder=2, solid_capstyle='round',
                ls=(0, (2, 1.4)) if (n['state'] == 'failed' and not on) else '-')
    ax.scatter([XP], [0], s=18, color=GRAY, zorder=4, edgecolors='white', linewidths=0.5)
    ax.text(XL, 0, 'B  starting model, 50.3M', ha='left', va='center', fontsize=fs, color=INK2)
    for a, n in nodes.items():
        on = a in path
        sq = T.get('squares') and n['category'].startswith('inference')
        fc = lcol if (light_after and a > light_after) else col
        if n['state'] == 'failed':
            ax.scatter([X(a)], [Y(a)], marker='x', s=11 if ys == 1 else 7, color=(fc if on else '#8f8e8a'),
                       linewidths=0.8, zorder=4)
        elif on:
            ax.scatter([X(a)], [Y(a)], s=(14 if sq else 17) * (1 if ys == 1 else 0.6), marker='s' if sq else 'o',
                       color=fc, zorder=4, edgecolors='white', linewidths=0.45)
        else:
            ax.scatter([X(a)], [Y(a)], s=7 if ys == 1 else 4, marker='s' if sq else 'o', facecolors='white',
                       edgecolors='#9f9e9a', linewidths=0.5, zorder=3)
    for a in T.get('frontier', []):
        ax.scatter([X(a)], [Y(a)], s=46, facecolors='none', edgecolors=col, linewidths=0.85, zorder=5)
    for a, text in T['labels'].items():
        on = a in path
        ml = '\n' in text
        ax.text(XL, Y(a) - (0.55 if ml else 0), text, ha='left', va='top' if ml else 'center', fontsize=fs, zorder=6,
                linespacing=1.05,
                color=INK if on else INK2, fontweight='bold' if a in T.get('bold', ()) else 'normal')
    for a0, a1, text in T.get('groups', []):
        y0, y1 = Y(a0), Y(a1)
        xb = XL - 0.015
        ax.plot([xb, xb + 0.012, xb + 0.012, xb], [y0, y0, y1, y1], color='#a5a4a0', lw=0.5, zorder=1)
        ax.text(XL + 0.012, (y0 + y1) / 2, text, ha='left', va='center', fontsize=fs, color=INK, zorder=6,
                linespacing=1.05)
    for y, text in T.get('notes', []):
        ax.text(XL, y, text, ha='left', va='center', fontsize=fs - 0.2, color=MUTED, style='italic', zorder=6,
                linespacing=1.05)
    if T.get('end_note'):
        y = Y(max(nodes)) + 1.3
        ax.text(XL, y, T['end_note'], ha='left', va='top', fontsize=fs - 0.2, color=MUTED, style='italic',
                linespacing=1.1, wrap=False)
    for y, lab in T.get('ticks', []):
        ax.text(-0.005, y * ys, lab, ha='right', va='center', fontsize=4.6, color=MUTED)
    h1, h2 = T['head']
    ax.text(0.0, -2.75, h1, ha='left', va='bottom', fontsize=5.9, color=col, fontweight='bold')
    ax.text(0.0, -2.6, h2, ha='left', va='top', fontsize=5.2, color=MUTED, linespacing=1.05)


def fig_tracks():
    T = tracks_data()
    style(1.0)
    fig = plt.figure(figsize=(5.5, 2.57))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.9, 1.0, 0.98, 1.1], wspace=0.07, left=0.03, right=0.995,
                          top=0.915, bottom=0.07)
    axs = [fig.add_subplot(gs[0, i]) for i in range(4)]
    for ax, t in zip(axs, T):
        draw_track(ax, t)
    # shared attempt numbers for the three 22-27-attempt campaigns
    for y in (1, 5, 10, 15, 20, 25):
        axs[1].text(-0.005, y, str(y), ha='right', va='center', fontsize=4.6, color=MUTED)
    b0, b1, b2, b3 = (a.get_position() for a in axs)
    fig.text(b0.x0, 0.985, 'Original experiments (Jul to Aug)', ha='left', va='top', fontsize=6.4, color=INK,
             fontweight='bold')
    fig.text(b2.x0, 0.985, 'Extended experiments (Sep to Nov): same program, start and benchmark', ha='left',
             va='top', fontsize=6.4, color=INK, fontweight='bold')
    fig.add_artist(Line2D([b2.x0 - 0.008, b2.x0 - 0.008], [0.07, 0.975], color='#d5d4d0', lw=0.6))
    handles = [Line2D([], [], color=INK2, lw=1.5, marker='o', ms=3.2, label='recorded parent path'),
               Line2D([], [], color='none', marker='o', ms=2.8, mfc='white', mec='#8f8e8a', label='other attempt'),
               Line2D([], [], color='none', marker='x', ms=3.2, mec=INK2, label='failed run'),
               Line2D([], [], color='none', marker='s', ms=3.0, mfc=VIOLET, mec='white', label='inference-code edit'),
               Line2D([], [], color='none', marker='o', ms=5.5, mfc='none', mec=VIOLET, label='final frontier')]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=5.3, frameon=False, handlelength=1.4,
               bbox_to_anchor=(0.5, -0.01), columnspacing=1.3)
    # every label must stay inside its own column
    fig.canvas.draw()
    rnd = fig.canvas.get_renderer()
    for i, ax in enumerate(axs):
        right = axs[i + 1].get_window_extent(rnd).x0 if i < 3 else fig.bbox.x1
        for tx in ax.texts:
            bb = tx.get_window_extent(rnd)
            _need(bb.x1 <= right + 1.0, f'label leaves its column: {tx.get_text()!r}')
    save(fig, 'cr_tracks')
    out = []
    for t in T:
        out.append({'head': t['head'], 'path': t['path'], 'labels': t['labels'], 'groups': t.get('groups', []),
                    'nodes': {str(a): {k: v for k, v in n.items() if k != 'text'} for a, n in t['nodes'].items()}})
    write_json(CR_DATA / 'cr_tracks.json', out)
    return T


# ============================================================================ Table 1: did the discoveries hold up?
def table_verify_main(D):
    """Main-text Table 1: held-out BPB and the four downstream tasks at 201.3M matched training tokens.  Two seed
    groups, never mixed: the submitted panel (seeds 3-5; AR-ref and AP-Q shown, all seven in the appendix) and the
    E1 / Opus retraining (seeds 6-8).  Bold marks the best displayed mean within a group (ties at the shown precision
    share it).  In the seeds 3-5 group the bold cells are also the best of all seven panel recipes."""
    from cr_v2 import BENCH, bench
    B = bench(D)
    keys = [k for k, _, _ in BENCH]
    lower = {'heldout'}
    fmt = lambda k, v: f'{v:.4f}' if k == 'heldout' else f'{100 * v:.1f}'

    def best(rows, k):
        vals = [float(fmt(k, r[k][0])) for r in rows]
        b = min(vals) if k in lower else max(vals)
        return {r['name'] for r, v in zip(rows, vals) if v == b}
    panel_all = B['panel']
    panel_show = [r for r in panel_all if r['name'] in ('AR-ref', 'AP-Q')]
    for k in keys:   # bold among the two shown equals bold among all seven panel recipes
        shown = best(panel_show, k)
        allb = best(panel_all, k)
        _need(shown <= allb, (k, shown, allb))
    p5 = B['phase5']
    out = ['% Generated by analysis/cr_r8.py from cr_v2.bench() (panel: cr_evidence.json panel_downstream; '
           'seeds 6-8: phase5_analysis.json). Do not edit by hand.',
           r'\begin{tabular}{@{}lrrrrrr@{}}', r'\toprule',
           r' & Params & Held-out & ARC-Easy & PIQA & LAMBADA & BLiMP \\',
           r'Recipe & (M) & BPB $\downarrow$ & acc.\ $\uparrow$ & acc.\ $\uparrow$ & exact $\uparrow$ & acc.\ $\uparrow$ \\',
           r'\midrule']
    for title, rows in ((r'E1 and Opus retraining, seeds 6--8', p5),):
        out.append(r'\multicolumn{7}{@{}l}{\emph{' + title + r'}} \\')
        bb = {k: best(rows, k) for k in keys}
        for r in rows:
            cells = []
            for k in keys:
                s = fmt(k, r[k][0])
                cells.append(rf'\textbf{{{s}}}' if r['name'] in bb[k] else s)
            out.append(r'\quad ' + f"{r['name']} & {r['params']:.1f} & " + ' & '.join(cells) + r' \\')
    out += [r'\bottomrule', r'\end{tabular}']
    (GEN_CR / 'table_verify_main.tex').write_text('\n'.join(out) + '\n')
    full = ['% Generated by analysis/cr_r8.py from cr_v2.bench(). Bold: best mean within its seed group. Do not edit by hand.',
            r'\begin{tabular}{lrrrrrr}', r'\toprule',
            r' & Params & Held-out & ARC-Easy & PIQA & LAMBADA & BLiMP \\',
            r'Recipe & (M) & BPB $\downarrow$ & acc.\,$\uparrow$ & acc.\,$\uparrow$ & exact\,$\uparrow$ & acc.\,$\uparrow$ \\',
            r'\midrule']
    for gi, (title, rows) in enumerate(((r'Panel (seeds 3--5)', panel_all), (r'Opus retraining (E1 seeds 6--8)', p5))):
        if gi:
            full.append(r'\midrule')
        full.append(r'\multicolumn{7}{l}{\emph{' + title + r'}} \\')
        bb = {k: best(rows, k) for k in keys}
        for r in rows:
            cells = [(rf'\textbf{{{fmt(k, r[k][0])}}}' if r['name'] in bb[k] else fmt(k, r[k][0])) for k in keys]
            full.append(r'\quad ' + f"{r['name']} & {r['params']:.1f} & " + ' & '.join(cells) + r' \\')
    full += [r'\bottomrule', r'\end{tabular}']
    (GEN_CR / 'table_bench_bold.tex').write_text('\n'.join(full) + '\n')
    pw = {k: sorted(best(panel_all, k)) for k in keys}
    _need(pw['heldout'] == ['AR-ref'] and pw['blimp'] == ['AR-ref'] and pw['piqa'] == ['AR-ref'], pw)
    _need(pw['arc_easy'] == ['AP-Q'] and pw['lambada'] == ['AP-Q', 'MOTPE'], pw)
    win = {k: sorted(best(p5, k)) for k in keys}
    return {'winners_seeds678': win, 'max_acc_sd_pts': B['max_acc_sd_pts'], 'median_acc_sd_pts': B['median_acc_sd_pts']}


# ============================================================================ macros
def macros_r8(O, T, V, R):
    T_ar, T_ap, T_cx, T_op = T
    cx, op, ap = T_cx['nodes'], T_op['nodes'], T_ap['nodes']
    bot = {r['name']: r for r in O['bottom']}
    top = {r['name']: r for r in O['top']}
    m = {
        'CREightTrainRatio': f"{O['train_ratio_apq_ar']:.2f}",
        'CREightDecodeRatio': f"{O['decode_ratio_apq_ar']:.2f}",
        'CREightMemRed': f"{O['mem_reduction_apq_ar_pct']:.0f}",
        'CREightARTrain': f"{top['AR-ref']['train_tok_s'] / 1e3:.0f}",
        'CREightAPTrain': f"{top['AP-Q']['train_tok_s'] / 1e3:.0f}",
        'CREightOpTrainLo': f"{min(bot[n]['train_tok_s'] for n in ('Opus A23', 'Opus A24', 'Opus A27')) / 1e3:.0f}",
        'CREightOpTrainHi': f"{max(bot[n]['train_tok_s'] for n in ('Opus A23', 'Opus A24', 'Opus A27')) / 1e3:.0f}",
        'CREightARDecode': f"{top['AR-ref']['decode']:.0f}",
        'CREightAPDecode': f"{top['AP-Q']['decode']:.0f}",
        'CREightCxBest': f"{cx[14]['val_bpb']:.4f}",
        'CREightCxBestX': f"{cx[14]['speedup']:.2f}",
        'CREightCxMaxX': f"{max(n['speedup'] for n in cx.values() if n['speedup']):.2f}",
        'CREightCxFirstX': f"{cx[1]['speedup']:.2f}",
        'CREightCxBatchGain': f"{cx[11]['val_bpb'] - cx[12]['val_bpb']:.3f}",
        'CREightOpSevenVal': f"{op[7]['val_bpb']:.4f}",
        'CREightOpSixWorse': f"{op[6]['val_bpb'] - op[5]['val_bpb']:.3f}",
        'CREightOpTenX': f"{op[10]['speedup']:.1f}",
        'CREightOpFourteenX': f"{op[14]['speedup']:.1f}",
        'CREightOpEighteenX': f"{op[18]['speedup']:.1f}",
        'CREightOpNineteenX': f"{op[19]['speedup']:.1f}",
        'CREightOpTwentyTwoX': f"{op[22]['speedup']:.1f}",
        'CREightOpSevenX': f"{op[7]['speedup']:.2f}",
        'CREightOpEightMemDrop': f"{op[7]['peak_mib'] - op[8]['peak_mib']:.0f}",
        'CREightAPOneGain': f"{T_ap['nodes'][1]['val_bpb'] - 1.2137349928434058:.3f}".replace('-', ''),
        'CREightAPEightWorse': f"{ap[8]['val_bpb'] - ap[7]['val_bpb']:.3f}",
        'CREightAPQDisc': f"{ap[22]['decode']:.0f}",
        'CREightARFail': str(T_ar['ar_fail']),
        'CREightAPBaseDisc': f"{read_json(CR_DATA / 'cr_evidence.json')['seed0_lineage'][0]['inference_tokens_per_second']:.0f}",
        'CREightARFour': f"{T_ar['nodes'][4]['val_bpb']:.3f}",
        'CREightARSix': f"{T_ar['nodes'][6]['val_bpb']:.3f}",
        'CREightOpTwentySevenX': f"{R['27']['decode_x']:.1f}",
        'CREightOpTwentyFourParams': f"{op[24]['params'] / 1e6:.1f}",
        'CREightBenchSDMax': f"{V['max_acc_sd_pts']:.1f}",
        'CREightBenchSDMed': f"{V['median_acc_sd_pts']:.1f}",
    }
    # facts the text states in words, checked here
    _need(V['winners_seeds678']['heldout'] == ['AR-ref'] and V['winners_seeds678']['blimp'] == ['AR-ref'], V)
    _need(V['winners_seeds678']['arc_easy'] == ['AP-Q', 'Opus A27'], V)
    _need(V['winners_seeds678']['piqa'] == ['Opus A23'] and V['winners_seeds678']['lambada'] == ['Opus A27'], V)
    s0 = read_json(CR_DATA / 'cr_evidence.json')['seed0_lineage'][0]
    _need(s0['id'] == 8 and abs(1.2137349928434058 - s0['val_bpb']) < 1e-12, 'seed-0 baseline')
    lines = ['% Generated by analysis/cr_r8.py. Do not edit by hand.']
    lines += [rf'\newcommand{{\{k}}}{{{v}}}' for k, v in m.items()]
    (GEN_CR / 'cr_macros_r8.tex').write_text('\n'.join(lines) + '\n')
    return m


def main():
    D = read_json(CR_DATA / 'cr_v2.json')
    S = read_json(CR_DATA / 'cr_stats.json')
    R = p2_raw_decode(D)
    O = fig_overview(D, S, R)
    T = fig_tracks()
    V = table_verify_main(D)
    A = table_access()
    M = macros_r8(O, T, V, R)
    _need(A['op']['own'] == 19, A['op']['own'])
    print('cr_r8:', json.dumps({k: M[k] for k in ('CREightTrainRatio', 'CREightDecodeRatio', 'CREightMemRed',
                                                  'CREightCxBest', 'CREightOpTenX')}))



# ============================================================================ appendix: who could edit inference code
def table_access():
    """Appendix table: the inference-editing surface and conditions of every AutoPareto search, read from the
    candidate sources and attempt records.  An 'own adapter' is a candidate train.py defining
    build_inference_adapter (the documented hook, present in every AutoPareto campaign)."""
    from cr_common import EVIDENCE
    q = sorted((EVIDENCE / 'queue').glob('autopareto-seed*/*-train.py'))
    _need(len(q) == 75, len(q))
    orig_adapter = sum('build_inference_adapter' in f.read_text() for f in q)
    from cr_common import git_show, sha256_bytes
    _need(sha256_bytes(git_show('5ea452c:train.py')).startswith('428e22f1'), 'original shared baseline')
    _need(sha256_bytes(git_show('65120a5:train.py')).startswith('eb267bef'), 'seed-42 start')
    for ref, h in (('c200ccb', 'af42d041'), ('8548308', '7d6817f0')):
        prog = git_show(f'{ref}:programs/program_autopareto.md')
        _need(sha256_bytes(prog).startswith(h), ref)
        _need('a candidate cache adapter that satisfies the protected interface' in ' '.join(prog.decode().split())
              and b'build_inference_adapter(model)' in prog, ref)
    ext = (OPUS_ROOT.parents[2] / 'study/contracts_2026/programs/program_fixed_time.md').read_bytes()
    norm = lambda bts: ' '.join(bts.decode().split())
    _need(sha256_bytes(ext).startswith('8acffb40') and
          'an inference adapter compatible with the protected interface' in norm(ext), 'extended program')
    rows = []
    for name, root in (('GPT-5.6, extended', CODEX_ROOT), ('Claude Opus 5.5, extended', OPUS_ROOT)):
        H = _hist(root)
        srcs = sorted((root / 'sources').glob('pair-a-fixed-time-seed3-attempt-*.train.py'))
        own = sum('def build_inference_adapter' in f.read_text() for f in srcs)
        first = min((int(re.search(r'attempt-(\d+)', f.name).group(1)) for f in srcs
                     if 'def build_inference_adapter' in f.read_text()), default=None)
        recs = list(H.values())
        prov = [r.get('scientist', {}).get('process_diagnostics', {}) for r in recs]
        tmo = sorted({p.get('timeout_seconds') for p in prov if p.get('timeout_seconds')})
        ev = []
        for r in recs:
            e = ((r['result'].get('inference_result') or {}).get('evaluator_sha256') or '')[:8]
            if e and e not in ev:
                ev.append(e)
        rows.append({'name': name, 'model': recs[0]['agent_model_identifier'], 'reasoning': recs[0]['reasoning_setting'],
                     'attempts': len(recs), 'valid': sum(r['status'] == 'completed' for r in recs),
                     'own': own, 'first': first, 'timeouts': tmo, 'evaluators': ev,
                     'program': recs[0]['program_sha256'][:8], 'start': recs[0]['starting_source_sha256'][:8],
                     'writable': recs[0]['scientist']['writable_files']})
    cx, op = rows
    _need(cx['program'] == op['program'] and cx['start'] == op['start'] and cx['writable'] == op['writable'] == ['train.py'],
          'same program, start and writable file')
    _need(orig_adapter == 0 and cx['own'] == 0 and op['own'] > 0 and op['first'] == 8, (orig_adapter, cx, op))
    tt = lambda r: ' then '.join(f'{x:.0f}\\,s' for x in r['timeouts'])
    op_eval = r'} then \texttt{'.join(op['evaluators'])      # roman "then" between the two evaluator hashes
    out = ['% Generated by analysis/cr_r8.py from queue/autopareto-seed*/ and the extended attempt records. Do not edit by hand.',
           r'\begin{tabular}{@{}>{\raggedright\arraybackslash}p{3.0cm}>{\raggedright\arraybackslash}p{3.15cm}'
           r'>{\raggedright\arraybackslash}p{3.15cm}>{\raggedright\arraybackslash}p{3.15cm}@{}}', r'\toprule',
           r' & GPT-5.6, original (3 campaigns) & GPT-5.6, extended & Opus 5.5, extended \\', r'\midrule',
           r'Writable file & \texttt{train.py} & \texttt{train.py} & \texttt{train.py} \\',
           r'Own-inference hook & documented & documented & documented \\',
           rf"Program; start (SHA-256) & \texttt{{af42d041}} then \texttt{{7d6817f0}}; \texttt{{428e22f1}}, \texttt{{eb267bef}} & \texttt{{{cx['program']}}}; \texttt{{{cx['start']}}} & \texttt{{{op['program']}}}; \texttt{{{op['start']}}} \\",
           r'Decode benchmark & 256 in, 256 out, 10 repetitions & 1{,}024 in, 128 out, bracketed & 1{,}024 in, 128 out, bracketed \\',
           rf"Model, reasoning & \texttt{{gpt-5.6-sol}}, high & \texttt{{{cx['model']}}}, {cx['reasoning']} & \texttt{{{op['model']}}}, {op['reasoning']} \\",
           r'Execution & agent app, instruction-based isolation & container, shell inside & host, file tools only \\',
           rf"Call timeout & none recorded & {tt(cx)} & {tt(op)} (from A10) \\",
           rf"Evaluator (SHA-256) & original runner & \texttt{{{', '.join(cx['evaluators'])}}} & \texttt{{{op_eval}}} (A8~on) \\",
           rf"Attempts (completed) & 75 (74) & {cx['attempts']} ({cx['valid']}) & {op['attempts']} ({op['valid']}) \\",
           rf"Attempts defining their own inference code & {orig_adapter} of 75 & {cx['own']} of {cx['attempts']} & {op['own']} of {op['valid']} valid (first: A{op['first']}) \\",
           r'\bottomrule', r'\end{tabular}']
    (GEN_CR / 'table_access.tex').write_text('\n'.join(out) + '\n')
    return {'orig_adapter': orig_adapter, 'cx': cx, 'op': op}


if __name__ == '__main__':
    main()
