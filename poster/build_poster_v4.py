#!/usr/bin/env python3
"""AutoPareto poster, version 4: A1 portrait (594 x 841 mm), visual-first redesign.

    /path/to/paper/.venv/bin/python poster/build_poster_v4.py

Writes poster/out_v4/autopareto_poster_v4_A1.{pdf,svg} and PNG previews. Versions 1 to 3 are unchanged.
Every number comes from poster_data.load(), which reads the paper's frozen records and asserts each printed value
against the paper. Diagrams are original vector drawings; the CUDA execution lanes are drawn to time scale per decoded
token (1/decode speed-up from the paper's ablation medians), with schematic kernel blocks.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.mplcache'))
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

import poster_data

OUT = HERE / 'out_v4'
W, H = 594.0, 841.0                 # A1 portrait, mm
PT = 72 / 25.4
MX, GUT = 20.0, 18.0
COLW = (W - 2 * MX - 2 * GUT) / 3
COLX = [MX + i * (COLW + GUT) for i in range(3)]
BODY_TOP, BODY_BOT = 93.0, 581.0

NAVY, INK, INK2, MUTED, RULE = '#1c2541', '#16181d', '#454b57', '#7d8491', '#c9cdd3'
LAV, LAV_EDGE = '#eeecfa', '#c3bcf0'
COL = {'baseline': '#8a8985', 'autoresearch': '#eb6834', 'classical': '#1baf7a', 'gpt': '#2a78d6', 'opus': '#6250d6'}
MARK = {'baseline': 's', 'autoresearch': 'D', 'classical': 'o', 'gpt': '^', 'opus': 'v'}
GEMV, TINY, PREFILL = '#3d4f6b', '#aab2c0', '#d8d4cc'
VIOLET = COL['opus']
BODY, CAP, TAB, TICK, FLAB = 19.0, 15.0, 16.0, 14.0, 15.5


def load_fonts():
    from fontTools.ttLib import TTCollection
    want = {'Regular': 'reg', 'Medium': 'med', 'Demi Bold': 'demi', 'Bold': 'bold', 'Italic': 'it',
            'Medium Italic': 'medit'}
    tmp = Path(tempfile.mkdtemp(prefix='poster_fonts_'))
    fp = {}
    for f in TTCollection('/System/Library/Fonts/Avenir Next.ttc').fonts:
        sub = f['name'].getDebugName(4).replace('Avenir Next', '').strip() or 'Regular'
        if sub in want:
            p = tmp / f"AvenirNext-{sub.replace(' ', '')}.ttf"
            f.save(str(p))
            font_manager.fontManager.addfont(str(p))
            fp[want[sub]] = FontProperties(fname=str(p))
    if set(want.values()) - set(fp):
        raise SystemExit('fonts missing')
    return fp


FP = load_fonts()
plt.rcParams.update({'pdf.fonttype': 42, 'svg.fonttype': 'none', 'font.family': 'Avenir Next',
                     'mathtext.fontset': 'cm', 'axes.unicode_minus': True, 'hatch.linewidth': 0.7})
fig = plt.figure(figsize=(W / 25.4, H / 25.4), dpi=72)
cv = fig.add_axes([0, 0, 1, 1])
cv.set_xlim(0, W)
cv.set_ylim(H, 0)
cv.axis('off')
REN = fig.canvas.get_renderer()


# ---------------------------------------------------------------- primitives (page mm, y down)
def prop(style, size):
    p = FP[style].copy()
    p.set_size(size)
    p.set_math_fontfamily('cm')
    return p


def ismath(s):
    return s.count('$') >= 2


def tw(s, size, style='reg'):
    w, _, _ = REN.get_text_width_height_descent(s, prop(style, size), ismath=ismath(s))
    return w / fig.dpi * 25.4


def th(s, size, style='reg'):
    _, h, d = REN.get_text_width_height_descent(s, prop(style, size), ismath=ismath(s))
    return h / fig.dpi * 25.4, d / fig.dpi * 25.4


def text(x, y, s, size, style='reg', color=INK, ha='left', va='baseline', ax=None, **kw):
    return (ax or cv).text(x, y, s, fontproperties=prop(style, size), color=color, ha=ha, va=va, **kw)


def mm(size):
    return size / PT


def rect(x, y, w, h, fc, ec='none', lw=0.0, z=1, **kw):
    p = Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=z, **kw)
    cv.add_patch(p)
    return p


def box(x, y, w, h, fc, ec='none', lw=0.0, r=2.0, z=0):
    cv.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw,
                                zorder=z))


def hline(x0, x1, y, color=RULE, lw=0.8, z=2):
    cv.plot([x0, x1], [y, y], color=color, lw=lw, zorder=z, solid_capstyle='butt')


def arrow(x0, y0, x1, y1, color=INK2, lw=1.3, ms=11, rad=0.0, z=5):
    cv.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=ms, color=color, lw=lw,
                                 connectionstyle=f'arc3,rad={rad}', zorder=z, shrinkA=0, shrinkB=0))


def para(x, y, w, runs, size=BODY, style='reg', color=INK, lead=1.30, justify=True, align='left', draw=True):
    """Justified rich paragraph; runs are str or (text, style, colour); style 'math' keeps a $...$ token whole.
    A run that starts with punctuation is glued to the previous one. y is the top; returns the bottom."""
    if isinstance(runs, str):
        runs = [(runs, style, color)]
    toks, prev_space = [], True
    for r in runs:
        t, st, co = (r + (color,))[:3] if len(r) == 2 else r
        st = st or style
        if st == 'math':
            toks.append([t, 'reg', co, not prev_space])
            prev_space = False
            continue
        parts = t.split(' ')
        for i, p in enumerate(parts):
            if p == '' and i > 0:
                continue
            if p == '':
                continue
            glue = i == 0 and not prev_space and t[:1] in ').,;:'
            toks.append([p, st, co, glue])
        prev_space = t.endswith(' ')
    sp = tw('a a', size) - tw('aa', size)
    lines, cur, curw = [], [], 0.0
    for tk in toks:
        ww = tw(tk[0], size, tk[1])
        gap = sp if cur and not tk[3] else 0.0
        if cur and curw + gap + ww > w:
            lines.append(cur)
            cur, curw = [tk], ww
        else:
            cur.append(tk)
            curw += gap + ww
    if cur:
        lines.append(cur)
    yy = y + mm(size) * 0.76
    for li, ln in enumerate(lines):
        widths = [tw(t[0], size, t[1]) for t in ln]
        gaps = [0.0] + [0.0 if t[3] else sp for t in ln[1:]]
        natural = sum(widths) + sum(gaps)
        n_g = sum(1 for g in gaps[1:] if g > 0)
        extra = (w - natural) / n_g if (justify and li < len(lines) - 1 and n_g) else 0.0
        if extra > 1.4 * sp:
            extra = 0.0
        if align == 'center':
            xx = x + (w - natural) / 2
        else:
            xx = x
        if draw:
            for t, wd, g in zip(ln, widths, gaps):
                xx += g + (extra if g > 0 else 0.0)
                text(xx, yy, t[0], size, t[1], t[2])
                xx += wd
        yy += mm(size) * lead
    return yy - mm(size) * lead + mm(size) * 0.30


def display(cx, y, s, size=19.0, color=INK):
    h, d = th(s, size)
    text(cx, y + h - d, s, size, 'reg', color, ha='center')
    return y + h + 0.5


def section_title(x, y, w, title):
    text(x + w / 2, y + mm(25) * 0.76, title, 25, 'demi', INK, ha='center')
    hline(x, x + w, y + mm(25) + 2.2, color='#9aa0a8', lw=1.0)
    return y + mm(25) + 6.5


def caption(x, y, w, label, body):
    return para(x, y, w, [(label + ' ', 'demi', VIOLET), (body, 'reg', INK2)], CAP, lead=1.25)


def callout(x, y, w, runs, size=18.0, pad=5.0, align='center'):
    bottom = para(x + pad, y + pad, w - 2 * pad, runs, size, lead=1.25, justify=False, align=align, draw=False)
    box(x, y, w, bottom - y + pad - 1.0, LAV, ec=LAV_EDGE, lw=1.0, r=2.0)
    para(x + pad, y + pad, w - 2 * pad, runs, size, lead=1.25, justify=False, align=align)
    return bottom + pad - 1.0


def ax_at(x, y, w, h):
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])


def style_ax(ax, size=TICK):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#9aa0a8')
        ax.spines[s].set_linewidth(0.9)
    ax.tick_params(labelsize=size, length=3.5, width=0.9, color='#9aa0a8', labelcolor=INK2, pad=3)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontproperties(prop('reg', size))
    ax.set_facecolor('none')


def table(x, y, w, cols, rows, size=TAB, shade=()):
    """Booktabs-style table. cols: [(header, width_fraction, align)]; rows: lists of strings. Returns the bottom."""
    xs, acc = [], x
    for _, frac, _ in cols:
        xs.append(acc)
        acc += frac * w
    rh = mm(size) * 1.55
    hline(x, x + w, y, color=INK, lw=1.3)
    yy = y + 1.6
    hh = max(len(h.split('\n')) for h, _, _ in cols)
    for (hd, frac, al), cx in zip(cols, xs):
        for k, ln in enumerate(hd.split('\n')):
            tx = cx + (frac * w - 1.5 if al == 'r' else (frac * w / 2 if al == 'c' else 0.5))
            text(tx, yy + mm(size) * (0.85 + k * 1.15), ln, size - 0.5, 'demi', INK,
                 ha={'r': 'right', 'c': 'center', 'l': 'left'}[al])
    yy += mm(size) * (0.85 + (hh - 1) * 1.15) + 2.4
    hline(x, x + w, yy, color=INK, lw=0.7)
    yy += 1.0
    for i, row in enumerate(rows):
        if i in shade:
            rect(x, yy, w, rh, LAV, z=0)
        for (hd, frac, al), cx, cell in zip(cols, xs, row):
            st = 'demi' if isinstance(cell, tuple) else 'reg'
            cell = cell[0] if isinstance(cell, tuple) else cell
            tx = cx + (frac * w - 1.5 if al == 'r' else (frac * w / 2 if al == 'c' else 0.5))
            text(tx, yy + rh * 0.70, cell, size, st, INK, ha={'r': 'right', 'c': 'center', 'l': 'left'}[al])
        yy += rh
    hline(x, x + w, yy + 0.6, color=INK, lw=1.3)
    return yy + 1.5


P, M = poster_data.load()
R = {r['key']: r for r in P['recipes']}
Q = P['quotes']
L = P['lat']
A = P['abl']
C = P['census']
X0, X1 = MX, W - MX
CW = X1 - X0
CPU_FILL, GPU_FILL, LAUNCH, KERN, SMALLK = '#f3f0e8', '#edf1f6', '#2b2f36', '#3d4f6b', '#9aa6b8'
OPUS_IDS = ('opus_a23', 'opus_a24', 'opus_a27')
FAM = [('baseline', 'Baseline'), ('autoresearch', 'AutoResearch'), ('classical', 'Classical (5)'),
       ('gpt', 'GPT-5.6'), ('opus', 'Claude Opus 5.5')]
X_ = r'$\times$'


def sec_head(x, y, w, title, takeaway=None, tsize=30, ksize=20):
    text(x, y + mm(tsize) * 0.76, title, tsize, 'demi', INK)
    hline(x, x + w, y + mm(tsize) + 2.2, color='#aab0b8', lw=1.0)
    yy = y + mm(tsize) + 6.0
    if takeaway:
        yy = para(x, yy, w, takeaway, ksize, 'reg', INK2, lead=1.22, justify=False)
    return yy


def fit_size(s, size, style, maxw):
    while tw(s, size, style) > maxw and size > 10:
        size -= 0.5
    return size


# ================================================================ header, footer
def header():
    rect(0, 0, W, 86, NAVY, z=0)
    l1, l2 = 'AutoPareto: Auditing Fixed-Time Autonomous', 'Transformer Search with Matched-Exposure Verification'
    fs = min(fit_size(l1, 46, 'bold', CW - 20), fit_size(l2, 46, 'bold', CW - 20))
    text(W / 2, 30, l1, fs, 'bold', 'white', ha='center')
    text(W / 2, 48.5, l2, fs, 'bold', 'white', ha='center')
    text(W / 2, 65, 'Rami Alobaidy', 23, 'med', '#e7eaf0', ha='center')
    text(W / 2, 77, 'NeurIPS 2026  ·  Workshop for Autonomous Machine Learning Research (AutoMLR)  ·  Sydney', 16.5,
         'reg', '#b7bfcc', ha='center')


def footer():
    rect(0, H - 19, W, 19, NAVY, z=0)
    yb = H - 8.2
    text(X0, yb, 'rameonix@gmail.com', 14, 'reg', '#dfe3ea')
    text(W / 2, yb, 'NeurIPS 2026  ·  AutoMLR Workshop', 14, 'reg', '#dfe3ea', ha='center')
    text(X1, yb, 'Records and code: release pending', 14, 'reg', '#dfe3ea', ha='right')


# ================================================================ hero: question and the headline result
def hero(y):
    qw, h = 232.0, 100.0
    box(X0, y, qw, h, LAV, ec=LAV_EDGE, lw=1.2, r=2.5)
    text(X0 + 9, y + 13, 'Question', 17, 'demi', VIOLET)
    yy = para(X0 + 9, y + 17, qw - 18, [('What does an AI research agent discover when it rewrites a Transformer’s '
                                        'training code under a 300 s budget, scored on quality, decode speed and '
                                        'memory at once?', 'reg', INK)], 23, lead=1.22, justify=False)
    text(X0 + 9, yy + 11, 'Answer', 17, 'demi', VIOLET)
    para(X0 + 9, yy + 15, qw - 18, [('Claude Opus 5.5 shrank the model, then built its own inference engine: ',
                                     'demi', INK),
                                    ('decoding 32× faster (verified), a whole generation 4.6× faster.', 'demi', VIOLET)],
         22, lead=1.22, justify=False)
    # right: one generation, to scale
    rx = X0 + qw + 16
    rw = X1 - rx
    text(rx, y + 7.5, 'One generation: 1,024-token prompt, 128 new tokens', 17, 'demi', INK)
    bx0 = rx + 27
    sc = (rw - 27 - 26) / 900.0
    rows = [('Baseline', L['base_pre'], L['base_dec'], COL['baseline'], y + 13),
            ('Opus', L['op_pre'], L['op_dec'], VIOLET, y + 28)]
    for name, pre, dec, col, ry in rows:
        text(rx, ry + 8.0, name, 17, 'demi', INK)
        rect(bx0, ry, pre * sc, 11, PREFILL, ec='#9b968c', lw=0.7, z=3, hatch='///')
        rect(bx0 + pre * sc, ry, dec * sc, 11, col, z=3)
        text(bx0 + (pre + dec) * sc + 2.5, ry + 8.2, f'{pre + dec:.0f} ms', 17, 'demi', INK)
    text(bx0 + (L['base_pre'] + L['base_dec'] / 2) * sc, y + 13 + 7.9, f'decode {L["base_dec"]:.0f} ms', 15, 'demi',
         'white', ha='center')
    text(bx0 + L['op_pre'] * sc / 2, y + 28 + 7.9, f'prefill {L["op_pre"]:.0f} ms', 14.5, 'demi', INK, ha='center')
    yk = y + 44
    rect(bx0, yk, 6, 4.2, PREFILL, ec='#9b968c', lw=0.7, z=3, hatch='///')
    text(bx0 + 8, yk + 3.7, 'prefill: not timed by the score', 14, 'reg', INK2)
    rect(bx0 + 84, yk, 3, 4.2, COL['baseline'], z=3)
    rect(bx0 + 87, yk, 3, 4.2, VIOLET, z=3)
    text(bx0 + 92, yk + 3.7, 'decode steps: scored', 14, 'reg', INK2)
    # the two numbers
    ys = yk + 10
    half = (rw - 6) / 2
    for i, (num, col, l1, l2) in enumerate((('32×', VIOLET, 'faster decode steps',
                                             f'scored: {M["CRSixSpeedLo"]} to {M["CRSixSpeedHi"]}×'),
                                            ('4.6×', INK, 'faster generation',
                                             f'{L["base_tot"]:.0f} to {L["op_tot"]:.0f} ms'))):
        bxl = rx + i * (half + 6)
        bh_ = y + h - ys
        box(bxl, ys, half, bh_, '#f5f6f8', ec='#dde0e5', lw=0.8, r=2.5)
        cy = ys + bh_ / 2
        nw = tw(num, 50, 'bold')
        tx_w = max(tw(l1, 18.5, 'demi'), tw(l2, 16.5))
        sx = bxl + (half - nw - 4 - tx_w) / 2
        text(sx, cy + 6.3, num, 50, 'bold', col)
        text(sx + nw + 4, cy - 1.0, l1, 18.5, 'demi', INK)
        text(sx + nw + 4, cy + 7.0, l2, 16.5, 'reg', INK2)
    return y + h


# ================================================================ 1: how Opus found 32x
def discovery(y):
    yy = sec_head(X0, y, CW, 'How Opus found 32×', [
        ('Like GPT-5.6, Opus first shrank the model. Then it diagnosed launch-bound decoding and built an inference '
         'engine.', 'reg', INK2)])
    ch = 128.0
    ax = ax_at(X0 + 19, yy + 12, 380, ch)
    bx = ax_at(X0 + 424, yy + 12, CW - 424, ch)
    text(X0 + 19, yy + 7.5, 'Opus search records (one timing per attempt)', 15, 'demi', INK2)
    text(X0 + 424, yy + 7.5, 'Verified: 13-recipe test', 15, 'demi', INK2)
    rows = P['opus_rows']
    lo, hi = 0.85, 85
    phases = [(0.5, 7.5, 'shrink the model'), (7.5, 22.5, 'build an inference engine'), (22.5, 27.5, 'frontier')]
    for a0, a1, lab in phases:
        ax.axvspan(a0, a1, color='#f1effc' if a0 == 7.5 else '#f6f7f8', zorder=0, lw=0)
        ax.text((a0 + a1) / 2, 0.965, lab, transform=ax.get_xaxis_transform(), ha='center', va='top',
                fontproperties=prop('demi', 15), color=INK2)
    by = {r['a']: r for r in rows}
    par = P['opus_parent']
    path, cur = [], 23
    while cur:
        path.append(cur)
        cur = par.get(cur)
    path = sorted(path)
    ax.plot(path, [by[k]['dec'] for k in path], color=VIOLET, lw=1.8, alpha=0.6, zorder=2)
    for r in rows:
        unc = 'uncertain' in r['status']
        ax.scatter([r['a']], [r['dec']], s=42, color='white' if unc else VIOLET, edgecolor=VIOLET, lw=1.3, zorder=3)
    marks = [(4, 'fewer layers', (14, -18), 'left'), (10, 'CUDA graph', (-14, 16), 'right'),
             (14, 'kernel fusion', (-14, 16), 'right'), (18, 'ring K/V cache', (16, -22), 'left'),
             (19, 'custom attention', (-30, 22), 'right'), (22, 'fused decode step', (10, 22), 'left')]
    for at, lab, off, ha in marks:
        v = by[at]['dec']
        ax.scatter([at], [v], s=120, color='white' if 'uncertain' in by[at]['status'] else VIOLET, edgecolor=VIOLET,
                   lw=2.0, zorder=4)
        ax.annotate(f'A{at} {lab}  {v:.1f}×', (at, v), xytext=off, textcoords='offset points', ha=ha, va='center',
                    fontproperties=prop('demi', 15), color=INK, zorder=6,
                    arrowprops=dict(arrowstyle='-', color='#9aa0a8', lw=0.9, shrinkA=2, shrinkB=5))
    ax.scatter([0.345], [0.12], s=46, color=VIOLET, edgecolor=VIOLET, lw=1.3, transform=ax.transAxes, zorder=6)
    ax.text(0.365, 0.12, 'speed valid', transform=ax.transAxes, va='center', fontproperties=prop('reg', 13.5),
            color=INK2)
    ax.scatter([0.345], [0.05], s=46, color='white', edgecolor=VIOLET, lw=1.3, transform=ax.transAxes, zorder=6)
    ax.text(0.365, 0.05, 'speed-uncertain record; line: recorded parent path', transform=ax.transAxes, va='center',
            fontproperties=prop('reg', 13.5), color=INK2)
    ax.set_xlim(0.5, 27.5)
    ax.set_yscale('log')
    ax.set_ylim(lo, hi)
    ax.set_yticks([1, 2, 5, 10, 20, 32])
    ax.set_yticklabels(['1×', '2×', '5×', '10×', '20×', '32×'])
    ax.minorticks_off()
    ax.set_xticks([1, 5, 10, 15, 20, 25, 27])
    ax.set_xlabel('attempt', fontproperties=prop('reg', TICK), color=INK2, labelpad=2)
    ax.set_ylabel('decode speed-up', fontproperties=prop('reg', TICK), color=INK2, labelpad=4)
    ax.grid(axis='y', color='#e6e8ec', lw=0.8, zorder=0)
    style_ax(ax)
    # verified strip: same y scale
    groups = [(0, ['baseline', 'arref', 'rs', 'tpe', 'motpe', 'nsgaii', 'cmaes'], 'others'),
              (1, ['gpt_a22', 'gpt_a24', 'gpt_a25'], 'GPT-5.6'), (2, list(OPUS_IDS), 'Opus')]
    for gx, keys, lab in groups:
        for j, k in enumerate(keys):
            r = R[k]
            jit = (j - (len(keys) - 1) / 2) * 0.09
            bx.scatter([gx + jit], [r['speedup']], s=70 if r['group'] == 'opus' else 52, marker=MARK[r['group']],
                       color=COL[r['group']], edgecolor='white', lw=0.8, zorder=4)
    sp = lambda ks: (min(R[k]['speedup'] for k in ks), max(R[k]['speedup'] for k in ks))
    o_lo, o_hi = sp(OPUS_IDS)
    g_lo, g_hi = sp(['gpt_a22', 'gpt_a24', 'gpt_a25'])
    bx.annotate(f'{o_lo:.1f} to {o_hi:.1f}×', (2, o_hi), xytext=(0, -20), textcoords='offset points', ha='center',
                fontproperties=prop('bold', 15), color=VIOLET)
    bx.annotate(f'{g_lo:.1f} to {g_hi:.1f}×\nno inference\ncode', (1, g_hi), xytext=(0, 9), textcoords='offset points',
                ha='center', va='bottom', fontproperties=prop('demi', 13.5), color=COL['gpt'], linespacing=1.05)
    bx.set_xlim(-0.6, 2.6)
    bx.set_yscale('log')
    bx.set_ylim(lo, hi)
    bx.set_yticks([1, 2, 5, 10, 20, 32])
    bx.set_yticklabels([])
    bx.minorticks_off()
    bx.set_xticks([0, 1, 2])
    bx.set_xticklabels(['other\nrecipes', 'GPT-5.6', 'Opus'])
    bx.grid(axis='y', color='#e6e8ec', lw=0.8, zorder=0)
    style_ax(bx)
    return yy + 12 + ch + 12


# ================================================================ 2: the CUDA execution diagram
def lanes_row(x0, y, Lrow, kind):
    """One decoded token on CPU and GPU lanes. Lrow: drawn length (time to scale)."""
    cpu_y, gpu_y, lh_ = y, y + 16.5, 10.0
    text(x0 - 2.5, cpu_y + 7.0, 'CPU', 13, 'demi', MUTED, ha='right')
    text(x0 - 2.5, gpu_y + 7.0, 'GPU', 13, 'demi', MUTED, ha='right')
    rect(x0, cpu_y, Lrow, lh_, CPU_FILL, z=1)
    rect(x0, gpu_y, Lrow, lh_, GPU_FILL, z=1)
    if kind == 'eager':
        k = 13
        step = Lrow / k
        for i in range(k):
            lx = x0 + i * step + 0.6
            lw_ = step * 0.66
            rect(lx, cpu_y + 1.5, lw_, lh_ - 3, LAUNCH, z=3)
            kx = lx + lw_ + 0.9
            kw = step * 0.17
            rect(kx, gpu_y + 1.5, kw, lh_ - 3, KERN if i % 4 == 0 else SMALLK, z=3)
            arrow(lx + lw_ * 0.8, cpu_y + lh_ - 0.6, kx + kw * 0.5, gpu_y + 0.9, color='#8a909a', lw=0.8, ms=6)
            # idle span on the GPU until the next kernel
            nx = x0 + (i + 1) * step + 0.6 + step * 0.66 + 0.9
            if i < k - 1:
                rect(kx + kw + 0.4, gpu_y + 3.5, nx - kx - kw - 0.8, lh_ - 7, '#d3d8df', z=2)
    else:
        fused = kind == 'graph+fusion'
        rect(x0 + 0.6, cpu_y + 1.5, 7.0, lh_ - 3, VIOLET, z=3)
        arrow(x0 + 4.0, cpu_y + lh_ - 0.6, x0 + 9.0, gpu_y + 0.9, color=VIOLET, lw=1.0, ms=7)
        n = 8 if fused else 26
        gx0 = x0 + 8.6
        span = Lrow - 9.2
        wk = span / n
        for i in range(n):
            c = VIOLET if fused else (KERN if i % 4 == 0 else SMALLK)
            rect(gx0 + i * wk, gpu_y + 1.5, wk - 0.35, lh_ - 3, c, z=3)
    return gpu_y + lh_


def cuda(y):
    yy = sec_head(X0, y, CW, 'Inside one decode step', [
        ('At batch one, decoding is dominated by the CPU launching hundreds of tiny GPU kernels. ', 'reg', INK2),
        ('CUDA graph replay removes the launches; kernel fusion merges the kernels.', 'demi', INK)])
    lab_w = 66.0
    x0 = X0 + lab_w + 14
    Lmax = 284.0
    rows = [('Eager decoding', 'every kernel launched by the CPU', 'eager', 'a23-V4_eager_unfused', 'V4_eager_unfused'),
            ('CUDA graph replay', 'the step replayed by one graph call', 'graph', 'a23-V2_no_fusion', 'V2_no_fusion'),
            ('Graph replay + fusion', 'Opus’s final engine', 'graph+fusion', 'a23-V0_as_submitted', 'V0_as_submitted')]
    ry = yy + 10
    ends = []
    for i, (t1, t2, kind, ck, ak) in enumerate(rows):
        Lrow = Lmax * A['V4_eager_unfused'] / A[ak]
        text(X0, ry + 7.5, t1, 18, 'demi', VIOLET if i == 2 else INK)
        text(X0, ry + 14.5, t2, 14, 'reg', INK2)
        text(X0, ry + 21.5, f"{C[ck]['launches']:.0f} launches · {C[ck]['kernels']:.0f} kernels", 14, 'reg', MUTED)
        lanes_row(x0, ry, Lrow, kind)
        ex = x0 + Lrow + 3
        text(ex, ry + 21.0, f'{A[ak]:.1f}×', 21, 'bold', VIOLET if i == 2 else INK)
        ends.append((ry, Lrow))
        ry += 42.0
    # eager annotation
    text(x0 + 2, ends[0][0] + 32.0, 'GPU idle between kernels, waiting for each launch', 14, 'it', INK2)
    # insets in the time saved
    gy, gL = ends[1]
    ix = x0 + gL + 32
    text(ix, gy + 7.0, 'Record per prompt, replay per token', 16.5, 'demi', INK)
    para(ix, gy + 9.5, x0 + Lmax - ix + 10, [('Opus captures the decode step as a CUDA graph during each prompt’s '
                                             'prefill; every new token is then one graph call.', 'reg', INK2)], 14,
         lead=1.18, justify=False)
    fy, fL = ends[2]
    fx = x0 + fL + 32
    text(fx, fy + 7.0, 'Fuse the small kernels', 16.5, 'demi', INK)
    bx_ = fx
    for j, lab in enumerate(['norm', 'rotary', 'gate', 'add']):
        rect(bx_ + j * 15.5, fy + 11.0, 14.5, 8.0, SMALLK, z=3)
        text(bx_ + j * 15.5 + 7.25, fy + 16.4, lab, 12.5, 'demi', 'white', ha='center')
    arrow(bx_ + 4 * 15.5 + 1.5, fy + 15.0, bx_ + 4 * 15.5 + 10.5, fy + 15.0, color=INK2, lw=1.4, ms=10)
    rect(bx_ + 4 * 15.5 + 12, fy + 11.0, 30, 8.0, VIOLET, z=3)
    text(bx_ + 4 * 15.5 + 27, fy + 16.4, 'fused', 12.5, 'demi', 'white', ha='center')
    text(fx, fy + 25.5, f'compiled regions (TorchInductor): {C["a23-V4_eager_unfused"]["kernels"]:.0f} to '
                        f'{C["a23-V0_as_submitted"]["kernels"]:.0f} kernels per token', 13.5, 'it', INK2)
    # time axis
    ta = ry - 42.0 + 16.5 + 10.0 + 9.5
    arrow(x0, ta, x0 + Lmax, ta, color=MUTED, lw=1.0, ms=9)
    text(x0 + Lmax, ta + 5.2, 'time per decoded token, to scale (speed-up vs the starting model)', 13, 'reg', MUTED,
         ha='right')
    # right: measured interaction and the diagnosis
    px = x0 + Lmax + 26
    pw = X1 - px
    text(px, yy + 7.5, 'Measured on the final recipes', 15, 'demi', INK2)
    gap_x, gap_y = 15.0, 13.0
    cw_, ch_ = (pw - 10 - gap_x) / 2, 26.0
    gx = [px + 10, px + 10 + cw_ + gap_x]
    gyy = [yy + 22, yy + 22 + ch_ + gap_y]
    text(gx[0] + cw_ / 2, yy + 17.5, 'no graph', 13, 'demi', MUTED, ha='center')
    text(gx[1] + cw_ / 2, yy + 17.5, 'graph', 13, 'demi', VIOLET, ha='center')
    text(px + 4, gyy[0] + ch_ / 2, 'unfused', 13, 'demi', MUTED, ha='center', va='center', rotation=90)
    text(px + 4, gyy[1] + ch_ / 2, 'fused', 13, 'demi', VIOLET, ha='center', va='center', rotation=90)
    cells = {(0, 0): 'V4_eager_unfused', (0, 1): 'V2_no_fusion', (1, 0): 'V1_no_cuda_graph', (1, 1): 'V0_as_submitted'}
    for (j, i), ak in cells.items():
        best = (j, i) == (1, 1)
        box(gx[i], gyy[j], cw_, ch_, LAV if best else '#f5f6f8', ec=VIOLET if best else '#dde0e5',
            lw=1.6 if best else 0.8, r=2)
        text(gx[i] + cw_ / 2, gyy[j] + 15.5, f'{A[ak]:.1f}×', 22, 'bold', VIOLET if best else INK, ha='center')
    # gains along the arrows (medians of per-recipe ratios)
    for j in (0, 1):
        ym = gyy[j] + ch_ / 2 + 2.2
        arrow(gx[0] + cw_ + 1.2, ym, gx[1] - 1.2, ym, color=INK2 if j == 0 else VIOLET, lw=1.3, ms=9)
    for i in (0, 1):
        xm = gx[i] + cw_ / 2
        arrow(xm, gyy[0] + ch_ + 0.8, xm, gyy[1] - 0.8, color=INK2 if i == 0 else VIOLET, lw=1.3, ms=9)
    mx_ = gx[0] + cw_ + gap_x / 2
    g0 = A['V2_no_fusion'] / A['V4_eager_unfused']
    g1 = A['V0_as_submitted'] / A['V1_no_cuda_graph']
    f0 = A['V1_no_cuda_graph'] / A['V4_eager_unfused']
    f1 = A['V0_as_submitted'] / A['V2_no_fusion']
    text(mx_, gyy[0] + ch_ / 2 - 2.0, f'×{g0:.1f}', 13.5, 'demi', INK2, ha='center')
    text(mx_, gyy[1] + ch_ / 2 - 2.0, f'×{g1:.1f}', 13.5, 'demi', VIOLET, ha='center')
    text(gx[0] + cw_ / 2 + 2.5, gyy[0] + ch_ + gap_y / 2 + 2.0, f'×{f0:.2f}', 13.5, 'demi', INK2)
    text(gx[1] + cw_ / 2 + 2.5, gyy[0] + ch_ + gap_y / 2 + 2.0, f'×{f1:.2f}', 13.5, 'demi', VIOLET)
    para(px, gyy[1] + ch_ + 4, pw, [(f'Fusion gains ×{f0:.2f} without the graph and ×{f1:.2f} with it: the two '
                                     'interact. Medians over Opus A23, A24, A27 (arrows: ratios of the cells); counts: '
                                     'profiler census of A23. Starting model: 1.0×, '
                                     f'{C["baseline"]["launches"]:.0f} launches, {C["baseline"]["kernels"]:.0f} kernels.',
                                     'reg', INK2)], 13.5, lead=1.18, justify=False)
    qy = gyy[1] + ch_ + 31
    qb = para(px + 5, qy + 10.5, pw - 10, [(Q['op10a'], 'it', INK)], 16.5, lead=1.18, justify=False, draw=False)
    box(px, qy, pw, qb - qy + 4.5, LAV, ec=LAV_EDGE, lw=1.0, r=2)
    text(px + 5, qy + 7.8, 'Why: Opus’s hypothesis before A10', 14.5, 'demi', VIOLET)
    para(px + 5, qy + 10.5, pw - 10, [(Q['op10a'], 'it', INK)], 16.5, lead=1.18, justify=False)
    return ta + 9


# ================================================================ 3a: Pareto, 3b: the contract
def pareto(x, y, w):
    yy = sec_head(x, y, w, 'Verified: a frontier, not a leaderboard', [
        ('13 recipes, means of seeds 6 to 8, one benchmark. No recipe leads every objective.', 'reg', INK2)])
    lx = x
    for g, lab in FAM:
        cv.scatter([lx + 2], [yy + 5.5], s=60, marker=MARK[g], color=COL[g], edgecolor='white', lw=0.7, zorder=5)
        text(lx + 5.5, yy + 7.2, lab, 14, 'reg', INK2)
        lx += tw(lab, 14) + 14
    pw = (w - 30) / 2
    ph = 130.0
    for j, (key, title, ylim) in enumerate((('tok', 'Equal tokens (201.3M)', (1.0905, 1.1935)),
                                            ('fix', 'Equal time (300 s)', (1.1485, 1.1935)))):
        axx = x + 16 + j * (pw + 14)
        text(axx, yy + 18.5, title, 17, 'demi', INK)
        ax = ax_at(axx, yy + 22, pw, ph)
        lo, hi = ylim
        pts = [(r['key'], r[key], r['dec']) for r in P['recipes']]
        fr = poster_data.pareto(pts)
        fp = sorted([p for p in pts if p[0] in fr], key=lambda p: p[2])
        # boundary of the set dominated in these two objectives: a point at speed x is dominated when its BPB is no
        # better than the best frontier recipe at least as fast as x
        xs, ys = [112.0], [fp[0][1]]
        for i, (_, q, v) in enumerate(fp):
            xs.append(v)
            ys.append(q)
            if i + 1 < len(fp):
                xs.append(v)
                ys.append(fp[i + 1][1])
        xs.append(fp[-1][2])
        ys.append(hi)
        ax.fill_between(xs, ys, [hi] * len(xs), color='#edeff2', zorder=0, lw=0)
        ax.plot(xs[1:-1], ys[1:-1], color=INK2, lw=1.4, ls=(0, (4, 2)), zorder=4)
        for r in P['recipes']:
            q, v, g = r[key], r['dec'], r['group']
            ax.scatter([v], [min(q, hi - 0.002)], s=110 if g == 'opus' else 70, marker=MARK[g], color=COL[g],
                       edgecolor='white', lw=0.9, zorder=5, clip_on=False)
        ax.set_xscale('log')
        ax.set_xlim(112, 9000)
        ax.set_xticks([125, 250, 500, 1000, 2000, 4000])
        ax.set_xticklabels(['125', '250', '500', '1k', '2k', '4k'])
        ax.minorticks_off()
        ax.set_ylim(hi, lo)
        ax.set_yticks([1.10, 1.12, 1.14, 1.16, 1.18] if key == 'tok' else [1.15, 1.16, 1.17, 1.18, 1.19])
        ax.set_yticklabels([f'{t:.2f}' for t in ax.get_yticks()])
        ax.grid(color='#e6e8ec', lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        style_ax(ax)
        ax.set_xlabel('decode tokens/s', fontproperties=prop('reg', TICK), color=INK2, labelpad=2)
        if j == 0:
            ax.set_ylabel('held-out BPB (better ↑)'.replace('↑', 'up'), fontproperties=prop('reg', TICK), color=INK2,
                          labelpad=4)
        ov = sum(R[k]['dec'] for k in OPUS_IDS) / 3
        oq = max(R[k][key] for k in OPUS_IDS)
        ax.annotate('Opus', (ov, oq), xytext=(0, -11), textcoords='offset points', ha='center', va='top',
                    fontproperties=prop('demi', 15), color=VIOLET)
        ax.annotate('AR-ref', (R['arref']['dec'], R['arref'][key]), xytext=(8, 4) if key == 'tok' else (2, 13),
                    textcoords='offset points', ha='left' if key == 'tok' else 'center', va='center',
                    fontproperties=prop('demi', 14.5), color=COL['autoresearch'])
        ax.annotate('GPT A22', (R['gpt_a22']['dec'], R['gpt_a22'][key]), xytext=(7, -10 if key == 'tok' else 9),
                    textcoords='offset points', ha='left', va='center', fontproperties=prop('demi', 14.5),
                    color=COL['gpt'])
        ax.text(0.04 if key == 'tok' else 0.60, 0.16 if key == 'tok' else 0.32, 'dominated in\nquality and speed',
                transform=ax.transAxes, ha='left', fontproperties=prop('it', 13.5), color=MUTED, linespacing=1.05)
        if key == 'fix':
            ax.annotate(f'Baseline {R["baseline"]["fix"]:.3f}, off scale', (R['baseline']['dec'], hi - 0.002),
                        xytext=(8, 0), textcoords='offset points', ha='left', va='center',
                        fontproperties=prop('reg', 13), color=MUTED)
    # memory: the third objective, as a strip
    my = yy + 22 + ph + 19
    text(x + 16, my, 'Third objective: peak inference memory (MiB)', 15.5, 'demi', INK)
    sx0, sx1 = x + 16, x + w - 6
    msc = lambda m: sx0 + (sx1 - sx0) * m / 360.0
    ly = my + 17
    hline(sx0, sx1, ly, color='#aab0b8', lw=1.0)
    for t in (0, 100, 200, 300):
        cv.plot([msc(t), msc(t)], [ly, ly + 1.6], color='#aab0b8', lw=1.0)
        text(msc(t), ly + 6.2, str(t), 13, 'reg', INK2, ha='center')
    for r in P['recipes']:
        cv.scatter([msc(r['mem'])], [ly - 2.6], s=80 if r['group'] == 'opus' else 60, marker=MARK[r['group']],
                   color=COL[r['group']], edgecolor='white', lw=0.7, zorder=5)
    om = [R[k]['mem'] for k in OPUS_IDS]
    text(msc(sum(om) / 3), ly - 7.5, f'Opus {min(om):.0f} to {max(om):.0f}', 14, 'demi', VIOLET, ha='center')
    text(msc(R['gpt_a22']['mem']), ly - 7.5, f'GPT A22 {R["gpt_a22"]["mem"]:.0f}', 14, 'demi', COL['gpt'],
         ha='center')
    text(msc(R['arref']['mem']), ly - 7.5, f'AR-ref {R["arref"]["mem"]:.0f}', 14, 'demi', COL['autoresearch'],
         ha='center')
    return ly + 8


def contract(x, y, w):
    yy = sec_head(x, y, w, 'The contract decides', [
        ('A 300 s budget rewards speed with more data. Equal tokens reveal learning per token.', 'reg', INK2)])
    ax = ax_at(x + 17, yy + 8, w - 20, 166)
    lo, hi = 1.090, 1.176
    keys_hi = {'arref': COL['autoresearch'], 'gpt_a22': COL['gpt'], 'opus_a27': VIOLET}
    for r in P['recipes']:
        if r['key'] in keys_hi or r['fix'] > hi or r['tok'] > hi:
            continue
        ax.annotate('', (201.3, r['tok']), (r['ftok'] / 1e6, r['fix']),
                    arrowprops=dict(arrowstyle='-|>', color='#c3c8cf', lw=0.9, mutation_scale=8, shrinkA=2, shrinkB=2))
    for k, col in keys_hi.items():
        r = R[k]
        x300 = r['ftok'] / 1e6
        ax.annotate('', (201.3, r['tok']), (x300, r['fix']),
                    arrowprops=dict(arrowstyle='-|>', color=col, lw=2.2, mutation_scale=14, shrinkA=4, shrinkB=5))
        ax.scatter([x300], [r['fix']], s=110, marker='D', color='white', edgecolor=col, lw=2.2, zorder=5)
        ax.scatter([201.3], [r['tok']], s=110, color=col, edgecolor='white', lw=1.0, zorder=5)
    ax.axvline(201.3, color=INK2, lw=1.0, ls=(0, (3, 2)), zorder=1)
    ax.set_xlim(70, 222)
    ax.set_ylim(hi, lo)
    ax.set_xticks([88, 158, 201.3])
    ax.set_xticklabels(['88M', '158M', '201.3M'])
    ax.set_yticks([1.10, 1.12, 1.14, 1.16])
    ax.set_yticklabels(['1.10', '1.12', '1.14', '1.16'])
    ax.grid(axis='y', color='#e6e8ec', lw=0.7, zorder=0)
    style_ax(ax)
    ax.set_xlabel('training tokens', fontproperties=prop('reg', TICK), color=INK2, labelpad=2)
    ax.set_ylabel('held-out BPB', fontproperties=prop('reg', TICK), color=INK2, labelpad=4)
    ax.text(R['arref']['ftok'] / 1e6, R['arref']['fix'] + 0.0045, 'AR-ref', ha='center', va='top',
            fontproperties=prop('demi', 14.5), color=COL['autoresearch'])
    ax.text(R['gpt_a22']['ftok'] / 1e6 - 3, R['gpt_a22']['fix'] - 0.0035, 'GPT A22', ha='right', va='bottom',
            fontproperties=prop('demi', 14.5), color=COL['gpt'])
    ax.text(R['opus_a27']['ftok'] / 1e6 - 3, R['opus_a27']['fix'] + 0.0048, 'Opus A27', ha='right', va='top',
            fontproperties=prop('demi', 14.5), color=VIOLET)
    ax.text(205.0, 1.0915, 'equal\ntokens', ha='left', va='top', fontproperties=prop('demi', 13.5), color=INK2,
            linespacing=1.0)
    ax.text(0.04, 0.80, 'At 300 s the leaders are not\nseparated, but saw very\ndifferent amounts of data.',
            transform=ax.transAxes, ha='left', va='top', fontproperties=prop('it', 13.5), color=INK2, linespacing=1.1)
    ax.text(196, 1.0960, 'AR-ref learns\nmost per token', ha='right', va='center', fontproperties=prop('it', 13.5),
            color=INK2, linespacing=1.05)
    # legend glyphs
    ax.scatter([0.05], [0.955], s=80, marker='D', color='white', edgecolor=INK2, lw=1.8, transform=ax.transAxes,
               clip_on=False, zorder=6)
    ax.text(0.09, 0.955, 'equal time (300 s)', transform=ax.transAxes, va='center', fontproperties=prop('reg', 13.5),
            color=INK2)
    ax.scatter([0.05], [0.895], s=70, color=INK2, transform=ax.transAxes, clip_on=False, zorder=6)
    ax.text(0.09, 0.895, 'equal tokens (201.3M)', transform=ax.transAxes, va='center',
            fontproperties=prop('reg', 13.5), color=INK2)
    return yy + 8 + 166 + 12


def notes(y):
    return para(X0, y, CW, [
        ('Limits. ', 'demi', INK),
        ('Three seeds; each compared search ran once, under different harnesses, so this is no ranking of models; the '
         '32× compares inference engines as well as models and measures speed, not quality. ', 'reg', INK2),
        ('AI use. ', 'demi', INK),
        ('Research agents GPT-5.6 and Claude Opus 5.5; Claude (Opus 5.5, Claude Code) ran the camera-ready experiments '
         'under approved plans and drafted this poster; authorship is human. ', 'reg', INK2),
        ('References. ', 'demi', INK),
        ('Karpathy, autoresearch (2026); Ferreira et al., arXiv:2603.24647 (2026).', 'reg', INK2)], 13.5, lead=1.22)


def build():
    header()
    footer()
    y = hero(96)
    y = discovery(y + 12)
    y = cuda(y + 6)
    lw_ = 330.0
    y1 = pareto(X0, y + 6, lw_)
    y2 = contract(X0 + lw_ + 18, y + 6, CW - lw_ - 18)
    yn = notes(max(y1, y2) + 6)
    print(f'sections end: discovery/cuda -> {y:.1f}; pareto {y1:.1f}; contract {y2:.1f}; notes {yn:.1f}; '
          f'footer at {H - 19:.1f}')
    OUT.mkdir(exist_ok=True)
    meta = {'Title': 'AutoPareto: NeurIPS 2026 AutoMLR poster (A1 portrait, version 4)', 'Author': 'Rami Alobaidy',
            'Subject': 'Auditing Fixed-Time Autonomous Transformer Search with Matched-Exposure Verification'}
    fig.savefig(OUT / 'autopareto_poster_v4_A1.pdf', metadata=meta)
    fig.savefig(OUT / 'autopareto_poster_v4_A1.svg')
    fig.savefig(OUT / 'autopareto_poster_v4_preview.png', dpi=150)
    fig.savefig(OUT / 'autopareto_poster_v4_preview_small.png', dpi=40)
    print('wrote', OUT)


if __name__ == '__main__':
    build()
