#!/usr/bin/env python3
"""AutoPareto poster, version 5: A1 landscape (841 x 594 mm), three columns in the style of a conference poster,
with the visual-first content of version 4.

    /path/to/paper/.venv/bin/python poster/build_poster_v5.py

Writes poster/out_v5/autopareto_poster_v5_A1_landscape.{pdf,svg} and PNG previews. Versions 1 to 4 are unchanged.
Every number comes from poster_data.load(), which reads the paper's frozen records and asserts each printed value
against the paper. The CUDA execution lanes are drawn to time scale per decoded token (1/decode speed-up from the
paper's ablation medians); their blocks are schematic.
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

OUT = HERE / 'out_v5'
W, H = 841.0, 594.0                 # A1 landscape, mm
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
CPU_FILL, GPU_FILL, LAUNCH, KERN, SMALLK = '#f3f0e8', '#edf1f6', '#2b2f36', '#3d4f6b', '#9aa6b8'
OPUS_IDS = ('opus_a23', 'opus_a24', 'opus_a27')
FAM = [('baseline', 'Baseline'), ('autoresearch', 'AutoResearch'), ('classical', 'Classical (5)'),
       ('gpt', 'GPT-5.6'), ('opus', 'Claude Opus 5.5')]
HEAD_H, FOOT_H = 64.0, 16.0
TOP = HEAD_H + 9.0
FIGN = iter(range(1, 20))


def fit_size(s, size, style, maxw):
    while tw(s, size, style) > maxw and size > 10:
        size -= 0.5
    return size


def stitle(x, y, w, title, size=27):
    text(x + w / 2, y + mm(size) * 0.76, title, size, 'demi', INK, ha='center')
    hline(x, x + w, y + mm(size) + 2.4, color='#8f96a1', lw=1.0)
    return y + mm(size) + 7.0


def finding(x, y, w, runs, size=18.5):
    return para(x, y, w, runs, size, 'reg', INK2, lead=1.22, justify=False)


def figcap(x, y, w, body):
    return para(x, y, w, [(f'Figure {next(FIGN)}. ', 'demi', VIOLET), (body, 'reg', INK2)], 15, lead=1.22,
                justify=False)


# ================================================================ header, footer
def header():
    rect(0, 0, W, HEAD_H, NAVY, z=0)
    t = 'AutoPareto: Auditing Fixed-Time Autonomous Transformer Search with Matched-Exposure Verification'
    fs = fit_size(t, 46, 'bold', W - 60)
    text(W / 2, 27, t, fs, 'bold', 'white', ha='center')
    text(W / 2, 43, 'Rami Alobaidy', 23, 'med', '#e7eaf0', ha='center')
    text(W / 2, 55, 'NeurIPS 2026  ·  Workshop for Autonomous Machine Learning Research (AutoMLR)  ·  Sydney', 16.5,
         'reg', '#b7bfcc', ha='center')


def footer():
    rect(0, H - FOOT_H, W, FOOT_H, NAVY, z=0)
    yb = H - 7.4
    text(MX, yb, 'rameonix@gmail.com', 14, 'reg', '#dfe3ea')
    text(W / 2, yb, 'NeurIPS 2026  ·  AutoMLR Workshop', 14, 'reg', '#dfe3ea', ha='center')
    text(W - MX, yb, 'Records and code: release pending', 14, 'reg', '#dfe3ea', ha='right')


# ================================================================ column 1: question, headline, the contract
def latency_fig(x, y, w):
    text(x, y + 6.5, 'One generation: 1,024-token prompt, 128 new tokens', 17.5, 'demi', INK)
    bx0 = x + 29
    sc = (w - 29 - 27) / 900.0
    rows = [('Baseline', L['base_pre'], L['base_dec'], COL['baseline'], y + 12),
            ('Opus', L['op_pre'], L['op_dec'], VIOLET, y + 30)]
    for name, pre, dec, col, ry in rows:
        text(x, ry + 9.6, name, 17.5, 'demi', INK)
        rect(bx0, ry, pre * sc, 14, PREFILL, ec='#9b968c', lw=0.7, z=3, hatch='///')
        rect(bx0 + pre * sc, ry, dec * sc, 14, col, z=3)
        text(bx0 + (pre + dec) * sc + 2.0, ry + 9.8, f'{pre + dec:.0f} ms', 17.5, 'demi', INK)
    text(bx0 + (L['base_pre'] + L['base_dec'] / 2) * sc, y + 12 + 9.4, f'decode {L["base_dec"]:.0f} ms', 15.5, 'demi',
         'white', ha='center')
    text(bx0 + L['op_pre'] * sc / 2, y + 30 + 9.4, f'prefill {L["op_pre"]:.0f} ms', 15, 'demi', INK, ha='center')
    yk = y + 49
    rect(bx0, yk, 6, 4.2, PREFILL, ec='#9b968c', lw=0.7, z=3, hatch='///')
    text(bx0 + 8, yk + 3.7, 'prefill: recorded, not scored', 13.5, 'reg', INK2)
    rect(bx0 + 82, yk, 3, 4.2, COL['baseline'], z=3)
    rect(bx0 + 85, yk, 3, 4.2, VIOLET, z=3)
    text(bx0 + 90, yk + 3.7, 'decode steps: scored', 13.5, 'reg', INK2)
    ys = yk + 10
    half = (w - 6) / 2
    bh_ = 38.0
    for i, (num, col, l1, l2) in enumerate((('32×', VIOLET, 'faster decode steps',
                                             f'scored: {M["CRSixSpeedLo"]} to {M["CRSixSpeedHi"]}×'),
                                            ('4.6×', INK, 'faster generation',
                                             f'{L["base_tot"]:.0f} to {L["op_tot"]:.0f} ms'))):
        bxl = x + i * (half + 6)
        box(bxl, ys, half, bh_, '#f5f6f8', ec='#dde0e5', lw=0.8, r=2.5)
        cy = ys + bh_ / 2
        nw = tw(num, 52, 'bold')
        tx_w = max(tw(l1, 17.5, 'demi'), tw(l2, 15.5))
        sx = bxl + (half - nw - 4 - tx_w) / 2
        text(sx, cy + 6.6, num, 52, 'bold', col)
        text(sx + nw + 4, cy - 1.0, l1, 17.5, 'demi', INK)
        text(sx + nw + 4, cy + 7.0, l2, 15.5, 'reg', INK2)
    return ys + bh_


def contract_fig(x, y, w, h):
    ax = ax_at(x + 17, y, w - 19, h)
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
    ax.text(206.5, 1.0915, 'equal\ntokens', ha='left', va='top', fontproperties=prop('demi', 13.5), color=INK2,
            linespacing=1.0)
    ax.text(196, 1.0960, 'AR-ref learns\nmost per token', ha='right', va='center', fontproperties=prop('it', 14),
            color=INK2, linespacing=1.05)
    ax.text(0.04, 0.78, 'At 300 s the leaders are not\nseparated, but saw very\ndifferent amounts of data.',
            transform=ax.transAxes, ha='left', va='top', fontproperties=prop('it', 14), color=INK2, linespacing=1.1)
    ax.scatter([0.05], [0.955], s=80, marker='D', color='white', edgecolor=INK2, lw=1.8, transform=ax.transAxes,
               clip_on=False, zorder=6)
    ax.text(0.09, 0.955, 'equal time (300 s)', transform=ax.transAxes, va='center', fontproperties=prop('reg', 14),
            color=INK2)
    ax.scatter([0.05], [0.89], s=70, color=INK2, transform=ax.transAxes, clip_on=False, zorder=6)
    ax.text(0.09, 0.89, 'equal tokens (201.3M)', transform=ax.transAxes, va='center', fontproperties=prop('reg', 14),
            color=INK2)
    return y + h + 14.5


def column1(x, w):
    y = stitle(x, TOP, w, 'Autonomous research under a budget')
    y = callout(x, y + 1, w, [('An AI agent rewrites a Transformer’s training code, trains each version for 300 s on '
                               'one GPU, and is scored on ', 'reg', INK),
                              ('quality, decode speed and memory', 'demi', INK),
                              ('. What does it discover, and does that survive a fair test?', 'reg', INK)],
                size=21, pad=7.0)
    y = latency_fig(x, y + 10, w)
    y = figcap(x, y + 3, w, 'Claude Opus 5.5’s recipes decode 32× faster than the starting model, but a whole '
                            'generation is only 4.6× faster: the prefill, where Opus records its CUDA graph, was '
                            'recorded but not scored (13-recipe test, means).')
    y = stitle(x, y + 11, w, 'The contract decides')
    y = finding(x, y + 0.5, w, [('A 300 s budget rewards speed with more data; ', 'reg', INK2),
                                ('equal tokens reveal learning per token.', 'demi', INK)])
    y = contract_fig(x, y + 6, w, 236)
    n_grey = sum(1 for r in P['recipes'] if r['key'] not in ('arref', 'gpt_a22', 'opus_a27')
                 and r['fix'] <= 1.176 and r['tok'] <= 1.176)
    y = figcap(x, y + 1, w, 'Each arrow runs from a recipe’s 300 s run (tokens it trained on, its BPB) to its '
                            f'201.3M-token run. Grey: {n_grey} other recipes (Baseline and GPT A24 lie off scale).')
    return y


# ================================================================ column 2: how Opus found 32x, inside one decode step
def discovery_fig(x, y, w, h):
    sw = 56.0
    aw = w - 16 - sw - 10
    ax = ax_at(x + 16, y + 7, aw, h)
    bx = ax_at(x + 16 + aw + 10, y + 7, sw, h)
    text(x + 16, y + 3.5, 'Opus search records', 14.5, 'demi', INK2)
    text(x + 16 + aw + 10, y + 3.5, 'Verified', 14.5, 'demi', INK2)
    rows = P['opus_rows']
    by = {r['a']: r for r in rows}
    lo, hi = 0.85, 95
    phases = [(0.5, 7.5, 'shrink the\nmodel'), (7.5, 22.5, 'build an inference engine'), (22.5, 27.5, 'frontier')]
    for a0, a1, lab in phases:
        ax.axvspan(a0, a1, color='#f1effc' if a0 == 7.5 else '#f6f7f8', zorder=0, lw=0)
        ax.text((a0 + a1) / 2, 0.975, lab, transform=ax.get_xaxis_transform(), ha='center', va='top',
                fontproperties=prop('demi', 13.5), color=INK2, linespacing=1.0)
    par = P['opus_parent']
    path, cur = [], 23
    while cur:
        path.append(cur)
        cur = par.get(cur)
    path = sorted(path)
    ax.plot(path, [by[k]['dec'] for k in path], color=VIOLET, lw=1.8, alpha=0.6, zorder=2)
    for r in rows:
        unc = 'uncertain' in r['status']
        ax.scatter([r['a']], [r['dec']], s=34, color='white' if unc else VIOLET, edgecolor=VIOLET, lw=1.2, zorder=3)
    marks = [(4, 'fewer layers', (12, -16), 'left'), (10, 'CUDA graph', (-10, 15), 'right'),
             (14, 'kernel fusion', (-10, 15), 'right'), (18, 'ring K/V cache', (12, -20), 'left'),
             (19, 'custom attention', (-26, 20), 'right'), (22, 'fused decode step', (-4, 33), 'center')]
    for at, lab, off, ha in marks:
        v = by[at]['dec']
        ax.scatter([at], [v], s=95, color='white' if 'uncertain' in by[at]['status'] else VIOLET, edgecolor=VIOLET,
                   lw=1.8, zorder=4)
        ax.annotate(f'A{at} {lab}  {v:.1f}×', (at, v), xytext=off, textcoords='offset points', ha=ha, va='center',
                    fontproperties=prop('demi', 13.5), color=INK, zorder=6,
                    arrowprops=dict(arrowstyle='-', color='#9aa0a8', lw=0.8, shrinkA=2, shrinkB=4))
    ax.scatter([0.55], [0.12], s=40, color=VIOLET, edgecolor=VIOLET, lw=1.2, transform=ax.transAxes, zorder=6)
    ax.text(0.57, 0.12, 'speed valid', transform=ax.transAxes, va='center', fontproperties=prop('reg', 13),
            color=INK2)
    ax.scatter([0.55], [0.05], s=40, color='white', edgecolor=VIOLET, lw=1.2, transform=ax.transAxes, zorder=6)
    ax.text(0.57, 0.05, 'speed-uncertain; line: parent path', transform=ax.transAxes, va='center',
            fontproperties=prop('reg', 13), color=INK2)
    ax.set_xlim(0.5, 27.5)
    ax.set_yscale('log')
    ax.set_ylim(lo, hi)
    ax.set_yticks([1, 2, 5, 10, 20, 32])
    ax.set_yticklabels(['1×', '2×', '5×', '10×', '20×', '32×'])
    ax.minorticks_off()
    ax.set_xticks([1, 5, 10, 15, 20, 25, 27])
    ax.set_xlabel('attempt', fontproperties=prop('reg', TICK), color=INK2, labelpad=2)
    ax.set_ylabel('decode speed-up', fontproperties=prop('reg', TICK), color=INK2, labelpad=3)
    ax.grid(axis='y', color='#e6e8ec', lw=0.8, zorder=0)
    style_ax(ax)
    groups = [(0, ['baseline', 'arref', 'rs', 'tpe', 'motpe', 'nsgaii', 'cmaes']),
              (1, ['gpt_a22', 'gpt_a24', 'gpt_a25']), (2, list(OPUS_IDS))]
    for gx, keys in groups:
        for j, k in enumerate(keys):
            r = R[k]
            jit = (j - (len(keys) - 1) / 2) * 0.10
            bx.scatter([gx + jit], [r['speedup']], s=60 if r['group'] == 'opus' else 44, marker=MARK[r['group']],
                       color=COL[r['group']], edgecolor='white', lw=0.7, zorder=4)
    sp = lambda ks: (min(R[k]['speedup'] for k in ks), max(R[k]['speedup'] for k in ks))
    o_lo, o_hi = sp(OPUS_IDS)
    g_lo, g_hi = sp(['gpt_a22', 'gpt_a24', 'gpt_a25'])
    bx.annotate(f'{o_lo:.1f} to\n{o_hi:.1f}×', (2, o_hi), xytext=(-2, -14), textcoords='offset points', ha='right',
                va='top', fontproperties=prop('bold', 13.5), color=VIOLET, linespacing=1.0)
    bx.annotate(f'{g_lo:.1f} to\n{g_hi:.1f}×', (1, g_hi), xytext=(0, 8), textcoords='offset points', ha='center',
                va='bottom', fontproperties=prop('demi', 12.5), color=COL['gpt'], linespacing=1.0)
    bx.set_xlim(-0.6, 2.6)
    bx.set_yscale('log')
    bx.set_ylim(lo, hi)
    bx.set_yticks([1, 2, 5, 10, 20, 32])
    bx.set_yticklabels([])
    bx.minorticks_off()
    bx.set_xticks([0, 1, 2])
    bx.set_xticklabels(['others', 'GPT', 'Opus'])
    bx.grid(axis='y', color='#e6e8ec', lw=0.8, zorder=0)
    style_ax(bx, 12.5)
    return y + 7 + h + 14.5


def lanes_row(x0, y, Lrow, kind):
    cpu_y, gpu_y, lh_ = y, y + 17.0, 11.0
    text(x0 - 2.0, cpu_y + 6.7, 'CPU', 12.5, 'demi', MUTED, ha='right')
    text(x0 - 2.0, gpu_y + 6.7, 'GPU', 12.5, 'demi', MUTED, ha='right')
    rect(x0, cpu_y, Lrow, lh_, CPU_FILL, z=1)
    rect(x0, gpu_y, Lrow, lh_, GPU_FILL, z=1)
    if kind == 'eager':
        k = 11
        step = Lrow / k
        for i in range(k):
            lx = x0 + i * step + 0.5
            lw_ = step * 0.64
            rect(lx, cpu_y + 1.5, lw_, lh_ - 3, LAUNCH, z=3)
            kx = lx + lw_ + 0.8
            kw = step * 0.17
            rect(kx, gpu_y + 1.5, kw, lh_ - 3, KERN if i % 4 == 0 else SMALLK, z=3)
            arrow(lx + lw_ * 0.8, cpu_y + lh_ - 0.5, kx + kw * 0.5, gpu_y + 0.8, color='#8a909a', lw=0.8, ms=6)
            nx = x0 + (i + 1) * step + 0.5 + step * 0.64 + 0.8
            if i < k - 1:
                rect(kx + kw + 0.4, gpu_y + 3.4, nx - kx - kw - 0.8, lh_ - 6.8, '#d3d8df', z=2)
    else:
        fused = kind == 'graph+fusion'
        rect(x0 + 0.5, cpu_y + 1.5, 6.0, lh_ - 3, VIOLET, z=3)
        arrow(x0 + 3.5, cpu_y + lh_ - 0.5, x0 + 8.0, gpu_y + 0.8, color=VIOLET, lw=1.0, ms=7)
        n = 7 if fused else 22
        gx0 = x0 + 7.5
        wk = (Lrow - 8.0) / n
        for i in range(n):
            c = VIOLET if fused else (KERN if i % 4 == 0 else SMALLK)
            rect(gx0 + i * wk, gpu_y + 1.5, wk - 0.3, lh_ - 3, c, z=3)
    return gpu_y + lh_


def cuda_fig(x, y, w):
    lab_w = 63.0
    x0 = x + lab_w + 11
    Lmax = w - lab_w - 11 - 15
    rows = [('Eager decoding', 'CPU launches every kernel', 'eager', 'a23-V4_eager_unfused', 'V4_eager_unfused'),
            ('CUDA graph replay', 'one graph call per token', 'graph', 'a23-V2_no_fusion', 'V2_no_fusion'),
            ('Graph + fusion', 'Opus’s final engine', 'graph+fusion', 'a23-V0_as_submitted', 'V0_as_submitted')]
    ry = y
    ends = []
    for i, (t1, t2, kind, ck, ak) in enumerate(rows):
        Lrow = Lmax * A['V4_eager_unfused'] / A[ak]
        text(x, ry + 7.5, t1, 17, 'demi', VIOLET if i == 2 else INK)
        text(x, ry + 14.2, t2, 14, 'reg', INK2)
        text(x, ry + 20.9, f"{C[ck]['launches']:.0f} launches · {C[ck]['kernels']:.0f} kernels", 13.5, 'reg', MUTED)
        lanes_row(x0, ry, Lrow, kind)
        text(x0 + Lrow + 2.5, ry + 21.5, f'{A[ak]:.1f}×', 20, 'bold', VIOLET if i == 2 else INK)
        ends.append((ry, Lrow))
        ry += 45.0
    text(x0 + 2, ends[0][0] + 33.0, 'GPU idle between kernels, waiting for each launch', 13.5, 'it', INK2)
    gy, gL = ends[1]
    ix = x0 + gL + 26
    text(ix, gy + 7.5, 'Record per prompt, replay per token', 16, 'demi', INK)
    para(ix, gy + 10.0, x0 + Lmax - ix + 12, [('The decode step is captured as a CUDA graph in each prompt’s prefill; '
                                             'every new token is one graph call.', 'reg', INK2)], 13.5, lead=1.18,
         justify=False)
    fy, fL = ends[2]
    fx = x0 + fL + 26
    text(fx, fy + 7.5, 'Fuse the small kernels', 16, 'demi', INK)
    for j, lab in enumerate(['norm', 'rotary', 'gate', 'add']):
        rect(fx + j * 14.5, fy + 11.5, 13.5, 8.0, SMALLK, z=3)
        text(fx + j * 14.5 + 6.75, fy + 16.9, lab, 12.5, 'demi', 'white', ha='center')
    arrow(fx + 4 * 14.5 + 1.0, fy + 15.5, fx + 4 * 14.5 + 9.0, fy + 15.5, color=INK2, lw=1.3, ms=9)
    rect(fx + 4 * 14.5 + 10.5, fy + 11.5, 24, 8.0, VIOLET, z=3)
    text(fx + 4 * 14.5 + 22.5, fy + 16.9, 'fused', 12.5, 'demi', 'white', ha='center')
    text(fx, fy + 26.0, f'TorchInductor regions: {C["a23-V4_eager_unfused"]["kernels"]:.0f} to '
                        f'{C["a23-V0_as_submitted"]["kernels"]:.0f} kernels per token', 13, 'it', INK2)
    ta = ry - 45.0 + 17.0 + 11.0 + 8.0
    arrow(x0, ta, x0 + Lmax, ta, color=MUTED, lw=1.0, ms=9)
    text(x0 + Lmax, ta + 5.0, 'time per decoded token, to scale (speed-up vs the starting model)', 12.5, 'reg', MUTED,
         ha='right')
    # measured 2x2 and the hypothesis, side by side
    yy = ta + 12
    gwid = w * 0.52
    gap_x, gap_y = 13.0, 11.0
    cw_, ch_ = (gwid - 9 - gap_x) / 2, 25.0
    gx = [x + 9, x + 9 + cw_ + gap_x]
    gyy = [yy + 8, yy + 8 + ch_ + gap_y]
    text(gx[0] + cw_ / 2, yy + 5, 'no graph', 13, 'demi', MUTED, ha='center')
    text(gx[1] + cw_ / 2, yy + 5, 'graph', 13, 'demi', VIOLET, ha='center')
    text(x + 3.5, gyy[0] + ch_ / 2, 'unfused', 12.5, 'demi', MUTED, ha='center', va='center', rotation=90)
    text(x + 3.5, gyy[1] + ch_ / 2, 'fused', 12.5, 'demi', VIOLET, ha='center', va='center', rotation=90)
    cells = {(0, 0): 'V4_eager_unfused', (0, 1): 'V2_no_fusion', (1, 0): 'V1_no_cuda_graph', (1, 1): 'V0_as_submitted'}
    for (j, i), ak in cells.items():
        best = (j, i) == (1, 1)
        box(gx[i], gyy[j], cw_, ch_, LAV if best else '#f5f6f8', ec=VIOLET if best else '#dde0e5',
            lw=1.6 if best else 0.8, r=2)
        text(gx[i] + cw_ / 2, gyy[j] + 16.0, f'{A[ak]:.1f}×', 23, 'bold', VIOLET if best else INK, ha='center')
    g0 = A['V2_no_fusion'] / A['V4_eager_unfused']
    g1 = A['V0_as_submitted'] / A['V1_no_cuda_graph']
    f0 = A['V1_no_cuda_graph'] / A['V4_eager_unfused']
    f1 = A['V0_as_submitted'] / A['V2_no_fusion']
    for j in (0, 1):
        ym = gyy[j] + ch_ / 2 + 2.0
        arrow(gx[0] + cw_ + 1.0, ym, gx[1] - 1.0, ym, color=INK2 if j == 0 else VIOLET, lw=1.2, ms=8)
    for i in (0, 1):
        xm = gx[i] + cw_ / 2
        arrow(xm, gyy[0] + ch_ + 0.8, xm, gyy[1] - 0.8, color=INK2 if i == 0 else VIOLET, lw=1.2, ms=8)
    mx_ = gx[0] + cw_ + gap_x / 2
    text(mx_, gyy[0] + ch_ / 2 - 1.8, f'×{g0:.1f}', 13, 'demi', INK2, ha='center')
    text(mx_, gyy[1] + ch_ / 2 - 1.8, f'×{g1:.1f}', 13, 'demi', VIOLET, ha='center')
    text(gx[0] + cw_ / 2 + 2.2, gyy[0] + ch_ + gap_y / 2 + 2.0, f'×{f0:.2f}', 13, 'demi', INK2)
    text(gx[1] + cw_ / 2 + 2.2, gyy[0] + ch_ + gap_y / 2 + 2.0, f'×{f1:.2f}', 13, 'demi', VIOLET)
    qx = x + gwid + 8
    qw = w - gwid - 8
    qb = para(qx + 5, yy + 15, qw - 10, [(Q['op10a'], 'it', INK)], 16.5, lead=1.2, justify=False, draw=False)
    box(qx, yy + 4, qw, qb - yy - 4 + 5, LAV, ec=LAV_EDGE, lw=1.0, r=2)
    text(qx + 5, yy + 11.5, 'Opus’s hypothesis before A10', 14, 'demi', VIOLET)
    para(qx + 5, yy + 15, qw - 10, [(Q['op10a'], 'it', INK)], 16.5, lead=1.2, justify=False)
    para(qx, qb + 6, qw, [(f'Fusion gains ×{f0:.2f} without the graph, ×{f1:.2f} with it: the two interact.',
                           'reg', INK2)], 13.5, lead=1.2, justify=False)
    return gyy[1] + ch_ + 4


def column2(x, w):
    y = stitle(x, TOP, w, 'How Opus found 32×')
    y = finding(x, y + 0.5, w, [('Like GPT-5.6, Opus first shrank the model. ', 'reg', INK2),
                                ('Then it diagnosed launch-bound decoding and built an inference engine.', 'demi',
                                 INK)])
    y = discovery_fig(x, y + 3, w, 150)
    y = figcap(x, y, w, 'Decode speed-up of each Opus attempt in its search records, and of every recipe re-timed in '
                        'the 13-recipe test. GPT-5.6 wrote no inference code in any campaign.')
    y = stitle(x, y + 9, w, 'Inside one decode step')
    y = finding(x, y + 0.5, w, [('At batch one, the CPU launching tiny kernels dominates. ', 'reg', INK2),
                                ('Graph replay removes the launches; fusion merges the kernels.', 'demi', INK)])
    y = cuda_fig(x, y + 6, w)
    y = figcap(x, y + 3, w, 'Rows to scale: time per token = 1/decode speed-up (ablation medians of Opus A23, A24, '
                            'A27); blocks schematic. Counts: profiler census of A23. Grid arrows: ratios of the cells.')
    return y


# ================================================================ column 3: verified frontier, take-home
def pareto_fig(x, y, w):
    lx = x + 3
    for g, lab in FAM:
        cv.scatter([lx + 2], [y + 3.5], s=55, marker=MARK[g], color=COL[g], edgecolor='white', lw=0.7, zorder=5)
        text(lx + 5.2, y + 5.1, lab, 13.5, 'reg', INK2)
        lx += tw(lab, 13.5) + 11.5
    yy = y + 10
    ph = 138.0
    for j, (key, title, ylim) in enumerate((('tok', 'Equal tokens (201.3M)', (1.0905, 1.1935)),
                                            ('fix', 'Equal time (300 s)', (1.1485, 1.1935)))):
        text(x + 16, yy + 5.5, title, 16.5, 'demi', INK)
        ax = ax_at(x + 16, yy + 8.5, w - 18, ph)
        lo, hi = ylim
        pts = [(r['key'], r[key], r['dec']) for r in P['recipes']]
        fr = poster_data.pareto(pts)
        fp = sorted([p for p in pts if p[0] in fr], key=lambda p: p[2])
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
            ax.scatter([v], [min(q, hi - 0.002)], s=105 if g == 'opus' else 66, marker=MARK[g], color=COL[g],
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
        ax.set_ylabel('held-out BPB', fontproperties=prop('reg', TICK), color=INK2, labelpad=4)
        if j == 1:
            ax.set_xlabel('decode tokens/s (log)', fontproperties=prop('reg', TICK), color=INK2, labelpad=2)
        else:
            ax.tick_params(labelbottom=True)
        ov = sum(R[k]['dec'] for k in OPUS_IDS) / 3
        oq = max(R[k][key] for k in OPUS_IDS)
        ax.annotate('Opus', (ov, oq), xytext=(-12, 0), textcoords='offset points', ha='right', va='center',
                    fontproperties=prop('demi', 15), color=VIOLET)
        ax.annotate('AR-ref', (R['arref']['dec'], R['arref'][key]), xytext=(8, 4) if key == 'tok' else (2, 13),
                    textcoords='offset points', ha='left' if key == 'tok' else 'center', va='center',
                    fontproperties=prop('demi', 14.5), color=COL['autoresearch'])
        ax.annotate('GPT A22', (R['gpt_a22']['dec'], R['gpt_a22'][key]), xytext=(7, -10 if key == 'tok' else 9),
                    textcoords='offset points', ha='left', va='center', fontproperties=prop('demi', 14.5),
                    color=COL['gpt'])
        ax.text(0.97, 0.07, 'dominated in quality and speed', transform=ax.transAxes, ha='right',
                fontproperties=prop('it', 13.5), color=MUTED)
        if key == 'fix':
            ax.annotate(f'Baseline {R["baseline"]["fix"]:.3f}, off scale', (R['baseline']['dec'], hi - 0.002),
                        xytext=(8, 0), textcoords='offset points', ha='left', va='center',
                        fontproperties=prop('reg', 13), color=MUTED)
        yy += 8.5 + ph + (12 if j == 0 else 15)
    # memory strip
    my = yy + 2
    text(x + 16, my, 'Third objective: peak inference memory (MiB)', 15, 'demi', INK)
    sx0, sx1 = x + 16, x + w - 6
    msc = lambda m: sx0 + (sx1 - sx0) * m / 360.0
    ly = my + 16
    hline(sx0, sx1, ly, color='#aab0b8', lw=1.0)
    for t in (0, 100, 200, 300):
        cv.plot([msc(t), msc(t)], [ly, ly + 1.6], color='#aab0b8', lw=1.0)
        text(msc(t), ly + 6.0, str(t), 13, 'reg', INK2, ha='center')
    for r in P['recipes']:
        cv.scatter([msc(r['mem'])], [ly - 2.6], s=75 if r['group'] == 'opus' else 55, marker=MARK[r['group']],
                   color=COL[r['group']], edgecolor='white', lw=0.7, zorder=5)
    om = [R[k]['mem'] for k in OPUS_IDS]
    text(msc(sum(om) / 3), ly - 7.5, f'Opus {min(om):.0f} to {max(om):.0f}', 14, 'demi', VIOLET, ha='center')
    text(msc(R['gpt_a22']['mem']), ly - 7.5, f'GPT A22 {R["gpt_a22"]["mem"]:.0f}', 14, 'demi', COL['gpt'],
         ha='center')
    text(msc(R['arref']['mem']) - 4, ly - 7.5, f'AR-ref {R["arref"]["mem"]:.0f}', 14, 'demi', COL['autoresearch'],
         ha='center')
    return ly + 9


def takehome(x, y, w):
    items = [('Budget. ', 'Under 300 s every agent found the same lever, smaller batches and models.'),
             ('Discovery. ', 'Opus alone built an inference engine: decoding 31.4 to 32.1× faster in 96 to 110 MiB, '
                             'with the same greedy tokens up to BF16 near-ties.'),
             ('Contract. ', 'Equal tokens favour AR-ref; at 300 s the leaders are not separated; the scored 32× is '
                            '4.6× for a whole generation.')]
    pad = 5.5
    yy = y + pad + mm(17.5) + 2.0
    for k, (h_, b_) in enumerate(items):
        yy = para(x + pad + 7, yy, w - 2 * pad - 7, [(h_, 'demi', INK), (b_, 'reg', INK)], 17, lead=1.22,
                  justify=False, draw=False) + 2.2
    box(x, y, w, yy - y + pad - 2.2, LAV, ec=LAV_EDGE, lw=1.0, r=2)
    text(x + pad, y + pad + mm(17.5) * 0.76, 'Take-home', 17.5, 'demi', INK)
    yy = y + pad + mm(17.5) + 2.0
    for k, (h_, b_) in enumerate(items):
        text(x + pad, yy + mm(17) * 0.76, f'{k + 1}.', 17, 'demi', VIOLET)
        yy = para(x + pad + 7, yy, w - 2 * pad - 7, [(h_, 'demi', INK), (b_, 'reg', INK)], 17, lead=1.22,
                  justify=False) + 2.2
    return yy + pad - 2.2


def column3(x, w):
    y = stitle(x, TOP, w, 'Verified: a frontier, not a leaderboard')
    y = finding(x, y + 0.5, w, [('13 recipes, seed means, one benchmark. ', 'reg', INK2),
                                ('No recipe leads every objective.', 'demi', INK)])
    y = pareto_fig(x, y + 4, w)
    y = figcap(x, y + 1, w, 'Held-out BPB against decode speed, means of seeds 6 to 8. Dashed: frontier of the means; '
                            'shaded: dominated in these two objectives.')
    y = takehome(x, y + 7, w)
    y = para(x, y + 5, w, [
        ('Limits. ', 'demi', INK),
        ('Three seeds; each compared search ran once, under different harnesses, so this is no ranking of models; the '
         '32× compares inference engines as well as models and measures speed, not quality. ', 'reg', INK2),
        ('AI use. ', 'demi', INK),
        ('Claude (Opus 5.5, Claude Code) ran the camera-ready experiments under approved plans and drafted this poster; '
         'authorship is human. ', 'reg', INK2),
        ('Refs. ', 'demi', INK),
        ('Karpathy, autoresearch (2026); Ferreira et al., arXiv:2603.24647 (2026).', 'reg', INK2)], 13, lead=1.22)
    return y


def build():
    header()
    footer()
    ends = [column1(COLX[0], COLW), column2(COLX[1], COLW), column3(COLX[2], COLW)]
    print('column ends (mm):', [round(e, 1) for e in ends], 'footer at', H - FOOT_H)
    OUT.mkdir(exist_ok=True)
    meta = {'Title': 'AutoPareto: NeurIPS 2026 AutoMLR poster (A1 landscape, version 5)', 'Author': 'Rami Alobaidy',
            'Subject': 'Auditing Fixed-Time Autonomous Transformer Search with Matched-Exposure Verification'}
    fig.savefig(OUT / 'autopareto_poster_v5_A1_landscape.pdf', metadata=meta)
    fig.savefig(OUT / 'autopareto_poster_v5_A1_landscape.svg')
    fig.savefig(OUT / 'autopareto_poster_v5_preview.png', dpi=150)
    fig.savefig(OUT / 'autopareto_poster_v5_preview_small.png', dpi=36)
    print('wrote', OUT)


if __name__ == '__main__':
    build()
