#!/usr/bin/env python3
"""AutoPareto poster, version 6: A1 portrait (594 x 841 mm), organised around five questions.

    /path/to/paper/.venv/bin/python poster/build_poster_v6.py

Writes poster/out_v6/autopareto_poster_v6_A1.{pdf,svg}, a 150-dpi PNG, a 2-3 m viewing preview and enlarged crops of
each figure. Versions 1 to 5 are unchanged. Every number comes from poster_data.load(), which reads the paper's frozen
records and asserts each printed value against the paper. The CPU/GPU lanes are drawn to time scale per decoded token
(1/decode speed-up, ablation-session medians); their blocks are schematic.
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

OUT = HERE / 'out_v6'
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
OPUS_IDS = ('opus_a23', 'opus_a24', 'opus_a27')
ARCH = '#3f7f8f'                      # Opus architecture changes (vs violet: inference engine)
GPTC, ARC, CLC, BASEC = COL['gpt'], COL['autoresearch'], COL['classical'], COL['baseline']
LAUNCH, KERN, SMALLK, LANE = '#2b2f36', '#3d4f6b', '#9aa6b8', '#eef1f5'
HEAD_H, FOOT_H = 84.0, 15.0
SEC, LEDE, LAB, TK, NOTE = 32, 20, 16.5, 15, 14


def fit_size(s, size, style, maxw):
    while tw(s, size, style) > maxw and size > 10:
        size -= 0.5
    return size


def sec_head(x, y, w, num, title, lede=None):
    r = 6.0
    cv.add_patch(Circle((x + r, y + r + 0.5), r, fc=VIOLET, ec='none', zorder=3))
    text(x + r, y + r + 0.65, str(num), 20, 'bold', 'white', ha='center', va='center', zorder=4)
    text(x + 2 * r + 4, y + r + 4.4, title, SEC, 'demi', INK)
    hline(x, x + w, y + 2 * r + 4.5, color='#b9bec6', lw=1.0)
    yy = y + 2 * r + 8.0
    if lede:
        yy = para(x, yy, w, lede, LEDE, 'reg', INK2, lead=1.2, justify=False)
    return yy


def note(x, y, w, s, size=NOTE, color=MUTED, style='reg'):
    return para(x, y, w, [(s, style, color)], size, lead=1.2, justify=False)


def panel_title(x, y, s, size=17.5, color=INK):
    text(x, y + mm(size) * 0.76, s, size, 'demi', color)
    return y + mm(size) + 2.0


# ================================================================ header and footer
def header():
    rect(0, 0, W, HEAD_H, NAVY, z=0)
    rect(0, HEAD_H - 2.2, W, 2.2, VIOLET, z=1)
    text(X0, 35, 'AutoPareto', 104, 'bold', 'white')
    hl = 'An AI research agent shrank the model and rewrote the inference engine: ~32× faster decoding, 4.6× faster generation'
    text(X0, 50.5, hl, fit_size(hl, 25, 'demi', CW), 'demi', '#cfc8ff')
    ft = 'Auditing Fixed-Time Autonomous Transformer Search with Matched-Exposure Verification'
    text(X0, 62.5, ft, 18.5, 'med', '#e7eaf0')
    text(X0, 74.0, 'Rami Alobaidy   ·   NeurIPS 2026 Workshop for Autonomous Machine Learning Research (AutoMLR), Sydney',
         15.5, 'reg', '#b7bfcc')


def footer():
    rect(0, H - FOOT_H, W, FOOT_H, NAVY, z=0)
    yb = H - 6.2
    text(X0, yb, 'rameonix@gmail.com', 13.5, 'reg', '#dfe3ea')
    text(W / 2, yb, 'NeurIPS 2026  ·  AutoMLR Workshop', 13.5, 'reg', '#dfe3ea', ha='center')
    text(X1, yb, 'Records and code: release pending', 13.5, 'reg', '#dfe3ea', ha='right')


# ================================================================ 1: the question and the headline
def flow(x, y, w):
    bw, gap, bh = (w - 2 * 9) / 3, 9.0, 21.0
    items = [('Agent', 'GPT-5.6 or Claude Opus 5.5'), ('edits train.py', 'with a recorded hypothesis'),
             ('trains 300 s', 'protected harness, one A40')]
    for i, (t1, t2) in enumerate(items):
        bx = x + i * (bw + gap)
        box(bx, y, bw, bh, LAV if i == 0 else '#f3f4f6', ec=LAV_EDGE if i == 0 else '#d8dbe0', lw=1.0, r=2)
        text(bx + bw / 2, y + 8.6, t1, 17, 'demi', INK, ha='center')
        text(bx + bw / 2, y + 15.6, t2, 13.5, 'reg', INK2, ha='center')
        if i < 2:
            arrow(bx + bw + 1.0, y + bh / 2, bx + bw + gap - 1.0, y + bh / 2, color='#8f96a3', lw=1.5, ms=12)
    yo = y + bh + 9
    tx_ = x + 2 * (bw + gap) + bw / 2
    arrow(tx_, y + bh + 1.0, tx_, yo - 1.0, color='#8f96a3', lw=1.5, ms=12)
    text(tx_ - 2.5, y + bh + 6.2, 'scored on', 13.5, 'it', INK2, ha='right')
    objs = [('Quality', 'bits per byte, unseen text', 'lower is better'), ('Decode speed', 'tokens/s, batch 1',
                                                                         'higher is better'),
            ('Memory', 'peak inference MiB', 'lower is better')]
    ow = (w - 2 * 5) / 3
    for i, (t1, t2, t3) in enumerate(objs):
        bx = x + i * (ow + 5)
        box(bx, yo, ow, 25, 'white', ec=VIOLET, lw=1.3, r=2)
        text(bx + ow / 2, yo + 8.0, t1, 17, 'demi', VIOLET, ha='center')
        text(bx + ow / 2, yo + 14.6, t2, 13.5, 'reg', INK2, ha='center')
        text(bx + ow / 2, yo + 20.6, t3, 13.5, 'it', MUTED, ha='center')
    return yo + 25


def headline(x, y, w):
    ix, iw = x + 7, w - 14
    yy = panel_title(ix, y + 6, 'One complete generation: 1,024-token prompt, 128 new tokens', 16.5)
    bx0 = ix + 30
    sc = (iw - 30 - 24) / 900.0
    rows = [('Starting\nmodel', L['base_pre'], L['base_dec'], BASEC, yy + 2),
            ('Opus', L['op_pre'], L['op_dec'], VIOLET, yy + 19)]
    for name, pre, dec, col, ry in rows:
        for k, ln in enumerate(name.split('\n')):
            text(ix, ry + 7.0 + k * 6.2 - (3.1 if '\n' in name else 0), ln, 16, 'demi', INK)
        rect(bx0, ry, pre * sc, 13, PREFILL, ec='#9b968c', lw=0.7, z=3, hatch='///')
        rect(bx0 + pre * sc, ry, dec * sc, 13, col, z=3)
        text(bx0 + (pre + dec) * sc + 2.0, ry + 9.0, f'{pre + dec:.0f} ms', 17, 'bold', INK)
    text(bx0 + (L['base_pre'] + L['base_dec'] / 2) * sc, yy + 2 + 8.8, f'decode {L["base_dec"]:.0f} ms', 15.5, 'demi',
         'white', ha='center')
    text(bx0 + L['op_pre'] * sc / 2, yy + 19 + 8.8, f'prefill {L["op_pre"]:.0f} ms', 15, 'demi', INK, ha='center')
    ox = bx0 + (L['op_pre'] + L['op_dec']) * sc
    text(ox + 26, yy + 19 + 8.8, f'decode {L["op_dec"]:.0f} ms', 15, 'demi', VIOLET)
    yk = yy + 37
    rect(bx0, yk, 6, 4.4, PREFILL, ec='#9b968c', lw=0.7, z=3, hatch='///')
    text(bx0 + 8, yk + 3.8, 'prefill: Opus captures its CUDA graph here; recorded, not scored', 13.5, 'reg', INK2)
    # the two numbers
    ys = yk + 9
    half = (iw - 5) / 2
    for i, (num, col, l1, l2) in enumerate((('~32×', VIOLET, 'faster decoding', 'the scored metric'),
                                            ('4.6×', INK, 'faster complete', 'generation'))):
        bxl = ix + i * (half + 5)
        box(bxl, ys, half, 27, 'white', ec='#dcdfe4', lw=0.8, r=2)
        nw = tw(num, 46, 'bold')
        tx_w = max(tw(l1, 16.5, 'demi'), tw(l2, 16.5, 'demi'))
        sx = bxl + (half - nw - 3.5 - tx_w) / 2
        text(sx, ys + 19.8, num, 46, 'bold', col)
        text(sx + nw + 3.5, ys + 12.0, l1, 16.5, 'demi', INK)
        text(sx + nw + 3.5, ys + 19.0, l2, 16.5, 'demi', INK if i else MUTED)
    # quality at 300 s, on one axis
    yq = ys + 30
    text(ix, yq + 4.5, 'Quality at 300 s (held-out BPB, lower is better)', 15, 'demi', INK)
    ax_y = yq + 14
    qx0, qx1 = ix + 6, ix + iw - 6
    lo, hi = 1.145, 1.250
    qx = lambda v: qx0 + (qx1 - qx0) * (v - lo) / (hi - lo)
    hline(qx0, qx1, ax_y, color='#9aa0a8', lw=1.0)
    for t in (1.15, 1.20, 1.25):
        cv.plot([qx(t), qx(t)], [ax_y, ax_y + 1.4], color='#9aa0a8', lw=1.0)
        text(qx(t), ax_y + 5.6, f'{t:.2f}', 13.5, 'reg', INK2, ha='center')
    b, a, o = R['baseline']['fix'], R['arref']['fix'], R['opus_a27']['fix']
    cv.scatter([qx(b)], [ax_y], s=90, marker='s', color=BASEC, zorder=5)
    cv.scatter([qx(a)], [ax_y], s=90, marker='D', color=ARC, zorder=5)
    cv.scatter([qx(o)], [ax_y], s=110, marker='v', color=VIOLET, zorder=6)
    text(qx(b), ax_y - 3.0, f'starting model {b:.3f}', 14, 'demi', INK2, ha='center')
    text(qx(a) - 1.8, ax_y - 3.0, f'AR-ref {a:.3f}', 14, 'demi', ARC, ha='right')
    text(qx(o) + 1.8, ax_y - 3.0, f'Opus A27 {o:.3f}: not resolved from AR-ref', 14, 'demi', VIOLET, ha='left')
    text(ix + iw, ax_y + 12.0, '13-recipe test, seeds 6 to 8; Opus = mean of A23, A24, A27', 13, 'it', MUTED,
         ha='right')
    bot = ax_y + 17
    box(x, y, w, bot - y, '#f6f5fd', ec=LAV_EDGE, lw=1.0, r=2.5)
    return bot


def section1(y):
    lw_ = 262.0
    yy = sec_head(X0, y, lw_, 1, 'What does AutoPareto test?',
                  [('An AI agent repeatedly rewrites a Transformer’s training program, trains each version for 300 s, '
                    'and is scored on three objectives at once.', 'reg', INK2)])
    yf = flow(X0, yy + 4, lw_)
    yb = note(X0, yf + 5, lw_, 'Compared against AutoResearch, an agent scored on quality only, and five classical '
                               'optimizers.', 15, INK2)
    yh = headline(X0 + lw_ + 14, y + 1, CW - lw_ - 14)
    return max(yb, yh)


# ================================================================ 2: what the agents discovered
def lineage(x, y, w, h):
    yy = panel_title(x, y, 'Opus’s search, attempt by attempt')
    note(x, yy - 0.5, w, 'historical search records: Opus’s own campaign, one timing per attempt', 14, MUTED, 'it')
    ax = ax_at(x + 17, yy + 6.5, w - 19, h)
    rows = P['opus_rows']
    by = {r['a']: r for r in rows}
    lo, hi = 0.85, 125
    for a0, a1, lab, col in ((0.5, 7.5, 'smaller model, batch', ARCH), (7.5, 22.5, 'mostly inference engine', VIOLET),
                             (22.5, 27.5, 'final recipes', INK2)):
        ax.axvspan(a0, a1, color='#eef6f7' if col == ARCH else ('#f1effc' if col == VIOLET else '#f5f6f8'), zorder=0,
                   lw=0)
        ax.text((a0 + a1) / 2, 0.975, lab, transform=ax.get_xaxis_transform(), ha='center', va='top',
                fontproperties=prop('demi', 15), color=col)
    par = P['opus_parent']
    path, cur = [], 23
    while cur:
        path.append(cur)
        cur = par.get(cur)
    path = sorted(path)
    ax.plot(path, [by[k]['dec'] for k in path], color=VIOLET, lw=2.0, alpha=0.55, zorder=2)
    for r in rows:
        unc = 'uncertain' in r['status']
        ax.scatter([r['a']], [r['dec']], s=30, color='white' if unc else '#b9b2ee', edgecolor='#b9b2ee', lw=1.1,
                   zorder=3)
    marks = [(4, 'smaller model', ARCH, (14, -20), 'left'), (10, 'CUDA graph replay', VIOLET, (-12, 18), 'right'),
             (14, 'kernel fusion', VIOLET, (-12, 17), 'right'), (18, 'ring K/V cache', VIOLET, (14, -24), 'left'),
             (19, 'custom attention', VIOLET, (-38, 18), 'right'), (22, 'fused decode step', VIOLET, (2, 24), 'center')]
    for at, lab, col, off, ha in marks:
        v = by[at]['dec']
        unc = 'uncertain' in by[at]['status']
        ax.scatter([at], [v], s=150, color='white' if unc else col, edgecolor=col, lw=2.2, zorder=5)
        ax.annotate(f'A{at} {lab}  {v:.1f}×', (at, v), xytext=off, textcoords='offset points', ha=ha, va='center',
                    fontproperties=prop('demi', 16.5), color=INK, zorder=6,
                    arrowprops=dict(arrowstyle='-', color='#8f96a1', lw=0.9, shrinkA=2, shrinkB=5))
    # the recorded hypothesis next to A10
    hx, hy = 10.6, 1.62
    ax.annotate('', (10, by[10]['dec']), (12.4, 3.15), arrowprops=dict(arrowstyle='-', color=VIOLET, lw=1.0,
                                                                         shrinkA=0, shrinkB=6))
    ax.text(hx, hy, 'A10 hypothesis, recorded before the run:\n' + Q['op10a'], fontproperties=prop('it', 15),
            color=INK, va='bottom', ha='left', linespacing=1.15, zorder=7,
            bbox=dict(boxstyle='round,pad=0.45,rounding_size=0.6', fc='white', ec=VIOLET, lw=1.1))
    for k_, (yk, lab) in enumerate(((0.30, 'open marker: speed-uncertain'), (0.22, 'small dots: other attempts'),
                                    (0.14, 'line: recorded path to A23'))):
        if k_ == 0:
            ax.scatter([0.805], [yk], s=40, color='white', edgecolor='#8f96a1', lw=1.2, transform=ax.transAxes, zorder=7)
        elif k_ == 1:
            ax.scatter([0.805], [yk], s=26, color='#b9b2ee', transform=ax.transAxes, zorder=7)
        else:
            ax.plot([0.79, 0.82], [yk, yk], color=VIOLET, alpha=0.55, lw=2.0, transform=ax.transAxes, zorder=7)
        ax.text(0.83, yk, lab, transform=ax.transAxes, va='center', fontproperties=prop('reg', 13.5), color=INK2)
    ax.set_xlim(0.5, 27.5)
    ax.set_yscale('log')
    ax.set_ylim(lo, hi)
    ax.set_yticks([1, 2, 5, 10, 20, 32])
    ax.set_yticklabels(['1×', '2×', '5×', '10×', '20×', '32×'])
    ax.minorticks_off()
    ax.set_xticks([1, 5, 10, 15, 20, 25, 27])
    ax.set_xlabel('attempt', fontproperties=prop('reg', TK), color=INK2, labelpad=2)
    ax.set_ylabel('decode speed-up over the starting model', fontproperties=prop('reg', 14), color=INK2, labelpad=3)
    ax.grid(axis='y', color='#e3e6ea', lw=0.8, zorder=0)
    style_ax(ax, TK)
    return yy + 6.5 + h + 11


def verified(x, y, w, h):
    yy = panel_title(x, y, 'Independently verified')
    note(x, yy - 0.5, w, '13-recipe test: retrained, seeds 6 to 8, timed on one A40', 14, MUTED, 'it')
    rows = [('baseline', 'Starting model', '8 layers, 50.3M'), ('gpt_a22', 'GPT A22', '4 layers; no inference code'),
            ('opus_a27', 'Opus A27', '3 layers + own inference engine')]
    by0 = yy + 13.5
    bh, gap = 12.5, 13.0
    bw = w - 4
    vmax = 5000.0
    for i, (k, n1, n2) in enumerate(rows):
        r = R[k]
        ry = by0 + i * (bh + gap)
        col = COL[r['group']]
        text(x, ry - 2.2, n1, 17, 'demi', col if k != 'baseline' else INK2)
        text(x + tw(n1 + '  ', 17, 'demi'), ry - 2.2, n2, 14, 'reg', MUTED)
        rect(x, ry, max(1.4, bw * r['dec'] / vmax), bh, col, z=3)
        lab = f"{r['dec']:,.0f} tok/s · {r['dec'] / R['baseline']['dec']:.1f}×"
        if r['dec'] / vmax > 0.6:
            text(x + bw * r['dec'] / vmax - 3, ry + 8.9, lab, 17.5, 'bold', 'white', ha='right')
        else:
            text(x + bw * r['dec'] / vmax + 3, ry + 8.9, lab, 17.5, 'bold', INK)
    ya = by0 + 3 * (bh + gap) - gap + 4
    hline(x, x + bw, ya, color='#9aa0a8', lw=1.0)
    for t in (0, 1000, 2000, 3000, 4000, 5000):
        xx = x + bw * t / vmax
        cv.plot([xx, xx], [ya, ya + 1.4], color='#9aa0a8', lw=1.0)
        text(xx, ya + 6.0, f'{t // 1000}k' if t else '0', 13.5, 'reg', INK2, ha='center')
    text(x + bw, ya + 12.0, 'decode tokens/s (1,024 in, 128 out, batch 1); × = rate ÷ starting model', 13.5, 'reg',
         INK2, ha='right')
    return ya + 15


def section2(y):
    yy = sec_head(X0, y, CW, 2, 'What did the agents discover?', [
        ('Both AutoPareto agents shrank the model; AutoResearch kept its size. ', 'reg', INK2),
        ('Only Claude Opus 5.5 then rewrote the inference engine.', 'demi', INK)])
    lw_ = 352.0
    y1 = lineage(X0, yy + 4, lw_, 90)
    y2 = verified(X0 + lw_ + 16, yy + 4, CW - lw_ - 16, 90)
    return max(y1, y2)


# ================================================================ 3: how Opus made decoding fast
def lane(x0, y, Lrow, kind):
    ch, gh, gap = 8.0, 8.0, 5.0
    cy, gy = y, y + ch + gap
    text(x0 - 2.0, cy + 6.0, 'CPU', 13, 'demi', MUTED, ha='right')
    text(x0 - 2.0, gy + 6.0, 'GPU', 13, 'demi', MUTED, ha='right')
    rect(x0, cy, Lrow, ch, '#f3f0e8', z=1)
    rect(x0, gy, Lrow, gh, LANE, z=1)
    if kind in ('reference', 'eager'):
        k = 24 if kind == 'reference' else 12
        step = Lrow / k
        for i in range(k):
            lx = x0 + i * step + 0.4
            lw_ = step * 0.62
            rect(lx, cy + 1.3, lw_, ch - 2.6, LAUNCH, z=3)
            kx = lx + lw_ + 0.6
            kw = step * 0.2
            rect(kx, gy + 1.3, kw, gh - 2.6, SMALLK, z=3)
            if i % (3 if kind == 'reference' else 2) == 0:
                arrow(lx + lw_ * 0.85, cy + ch - 0.4, kx + kw * 0.5, gy + 0.6, color='#8a909a', lw=0.7, ms=5)
    else:
        fused = kind == 'fused'
        rect(x0 + 0.3, cy + 1.3, 1.8, ch - 2.6, VIOLET, z=3)
        for t_ in (2.7, 3.5):
            cv.plot([x0 + t_, x0 + t_], [cy + 1.3, cy + ch - 1.3], color=LAUNCH, lw=0.7, zorder=3)
        arrow(x0 + 1.2, cy + ch - 0.4, x0 + 1.4, gy + 0.6, color=VIOLET, lw=1.0, ms=6)
        n = 6 if fused else 18
        gx0 = x0 + 0.3
        wk = (Lrow - 0.4) / n
        for i in range(n):
            c = '#9d95ec' if fused else SMALLK
            rect(gx0 + i * wk, gy + 1.3, wk - 0.22, gh - 2.6, c, z=3)
    return gy + gh


def lanes(x, y, w):
    lab_w = 98.0
    x0 = x + lab_w + 10
    Lfull = w - lab_w - 10 - 20
    stages = [('1', 'Original reference decoder', 'starting model: 8 layers, 50.3M', 'baseline', 1.0, 'reference', INK),
              ('2', 'Opus, graph and fusion off', '3 layers; own attention, ring cache on', 'a23-V4_eager_unfused',
               A['V4_eager_unfused'], 'eager', '#5d6577'),
              ('3', '+ CUDA graph replay', 'recorded per prompt, replayed per token', 'a23-V2_no_fusion',
               A['V2_no_fusion'], 'graph', VIOLET),
              ('4', '+ kernel fusion', 'small kernels merged (TorchInductor)', 'a23-V0_as_submitted',
               A['V0_as_submitted'], 'fused', VIOLET)]
    ry = y
    for num, t1, t2, ck, s, kind, col in stages:
        cv.add_patch(Circle((x + 3.6, ry + 4.6), 3.6, fc=col, ec='none', zorder=3))
        text(x + 3.6, ry + 4.75, num, 13, 'bold', 'white', ha='center', va='center', zorder=4)
        text(x + 10, ry + 6.6, t1, 16.5, 'demi', col if col != INK else INK)
        text(x + 10, ry + 12.8, t2, 13.5, 'reg', INK2)
        c = C[ck]
        launches = (f"1 graph + {c['launches']:.0f} launches" if c['graph_launches'] else f"{c['launches']:.0f} launches")
        text(x + 10, ry + 19.0, f"{launches} · {c['kernels']:.0f} kernels", 13.5, 'demi', MUTED)
        Lr = Lfull / s
        lane(x0, ry + 1.0, Lr, kind)
        text(x0 + Lr + 2.5, ry + 15.0, f'{s:.1f}×', 19, 'bold', col)
        ry += 26.0
    # captions inside the time the optimizations saved
    cx_ = x0 + Lfull / A['V2_no_fusion'] + 26
    para(cx_, y + 2 * 26.0 + 1.0, x0 + Lfull - cx_, [('Graph replay cuts CPU launch overhead: ', 'demi', INK),
                                                      ('the same 290 kernels still run on the GPU.', 'reg', INK2)],
         14.5, lead=1.18, justify=False)
    para(cx_, y + 3 * 26.0 + 1.0, x0 + Lfull - cx_, [('Fusion cuts the kernel count: ', 'demi', INK),
                                                      ('290 to 78; it may also reduce intermediate memory traffic '
                                                       '(not measured).', 'reg', INK2)], 14.5, lead=1.18,
         justify=False)
    ta = ry + 3.0
    arrow(x0, ta, x0 + Lfull, ta, color=MUTED, lw=1.0, ms=9)
    text(x0 + Lfull, ta - 1.8, 'lane length = time per decoded token, to scale; blocks schematic', 13.5, 'reg', MUTED,
         ha='right')
    lx = x0
    for col, lab in ((LAUNCH, 'CPU kernel launch'), (SMALLK, 'GPU kernel'), (VIOLET, 'graph replay call'),
                     ('#9d95ec', 'fused GPU kernel')):
        rect(lx, ta + 2.2, 4.5, 3.8, col, z=3)
        text(lx + 6, ta + 5.5, lab, 13.5, 'reg', INK2)
        lx += 6 + tw(lab, 13.5) + 7
    return ta + 9


def ablation(x, y, w, h):
    yy = panel_title(x, y, 'Take one part out of the final engine')
    note(x, yy - 0.5, w, 'ablation session: decode speed-up, medians of Opus A23, A24, A27', 14, MUTED, 'it')
    rows = [('V0_as_submitted', 'full engine', VIOLET), ('V5_linear_kv_cache', 'without ring K/V cache', '#8f86e6'),
            ('V3_sdpa_attention', 'without custom attention', '#8f86e6'), ('V2_no_fusion', 'without kernel fusion',
                                                                           '#8f86e6'),
            ('V1_no_cuda_graph', 'without CUDA graph', '#8f86e6'), ('V4_eager_unfused', 'without graph or fusion',
                                                                    '#8f86e6')]
    lab_w = 70.0
    bx0 = x + lab_w
    bw = w - lab_w - 18
    vmax = 34.0
    y0 = yy + 9
    rh = (h - 22) / len(rows)
    lo_v, hi_v = float(M['CRSixSpeedLo']), float(M['CRSixSpeedHi'])
    rect(bx0 + bw * lo_v / vmax, y0 - 3, bw * (hi_v - lo_v) / vmax, len(rows) * rh + 2, '#d9d4fa', z=1)

    for i, (k, lab, col) in enumerate(rows):
        ry = y0 + i * rh
        v = A[k]
        text(bx0 - 2.5, ry + rh * 0.62, lab, 15.5, 'demi' if i == 0 else 'reg', INK, ha='right')
        rect(bx0, ry + rh * 0.16, bw * v / vmax, rh * 0.66, col, z=3)
        text(bx0 + bw * v / vmax + 1.8, ry + rh * 0.66, f'{v:.1f}×', 16, 'bold', INK)
    ya = y0 + len(rows) * rh + 1
    hline(bx0, bx0 + bw, ya, color='#9aa0a8', lw=1.0)
    for t in (0, 10, 20, 30):
        xx = bx0 + bw * t / vmax
        cv.plot([xx, xx], [ya, ya + 1.4], color='#9aa0a8', lw=1.0)
        text(xx, ya + 6.0, f'{t}×', 13.5, 'reg', INK2, ha='center')
    text(bx0 + bw * hi_v / vmax + 1.0, ya + 12.5, f'shaded: independently verified, {M["CRSixSpeedLo"]} to '
                                                   f'{M["CRSixSpeedHi"]}×', 13.5, 'demi', VIOLET, ha='right')
    return ya + 15


def section3(y):
    yy = sec_head(X0, y, CW, 3, 'How did Opus make decoding 32× faster?', [
        ('At batch 1 each decoded token is hundreds of tiny GPU kernels, each launched by the CPU. ', 'reg', INK2),
        ('Opus applied standard CUDA graph replay, then fused kernels.', 'demi', INK)])
    lw_ = 360.0
    y1 = lanes(X0, yy + 6, lw_)
    y2 = ablation(X0 + lw_ + 14, yy + 4, CW - lw_ - 14, 106)
    yn = note(X0, max(y1, y2) + 1, CW, 'Counts per decoded token: torch.profiler census (Opus A23; starting model). '
                                       'Speed-ups: ablation session on one A40, medians of A23, A24, A27.', 14, MUTED)
    return yn


# ================================================================ 4: verification and the budget
def budget(x, y, w):
    rows = [('baseline', 'Starting model'), ('arref', 'AR-ref'), ('gpt_a22', 'GPT A22'), ('opus_a27', 'Opus A27')]
    lab_w = 40.0
    pa_w = 132.0
    gap = 14.0
    pb_w = (w - lab_w - pa_w - 2 * gap) / 2
    xa = x + lab_w
    xb = xa + pa_w + gap
    xc = xb + pb_w + gap
    panel_title(xa, y, 'Data seen in 300 s')
    panel_title(xb, y, 'Held-out BPB after 300 s')
    panel_title(xc, y, 'Held-out BPB after 201.3M tokens')
    note(xa, y + 7.5, pa_w, 'same time budget', 14, MUTED, 'it')
    note(xb, y + 7.5, pb_w, 'same time budget', 14, MUTED, 'it')
    note(xc, y + 7.5, pb_w, 'separate runs: same amount of data', 14, MUTED, 'it')
    y0 = y + 18
    rh = 11.0
    tmax = 220.0
    lo, hi = 1.090, 1.180
    bx = lambda v, x_, w_: x_ + w_ * (v - lo) / (hi - lo)
    # shaded "not resolved" band behind the three leaders at 300 s
    lead = [R[k]['fix'] for k in ('arref', 'gpt_a22', 'opus_a27')]
    rect(bx(min(lead) - 0.004, xb, pb_w), y0 + rh * 1 - 1, bx(max(lead) + 0.004, xb, pb_w) - bx(min(lead) - 0.004, xb, pb_w),
         3 * rh + 0.5, '#ece9fb', z=0)
    for i, (k, name) in enumerate(rows):
        r = R[k]
        ry = y0 + i * rh
        col = COL[r['group']]
        text(x, ry + rh * 0.62, name, 16, 'demi', col if k != 'baseline' else INK2)
        # tokens in 300 s
        Lb = pa_w * (r['ftok'] / 1e6) / tmax
        rect(xa, ry + rh * 0.18, Lb, rh * 0.64, col, z=3, alpha=0.9)
        text(xa + Lb + 1.8, ry + rh * 0.66, f"{r['ftok'] / 1e6:.0f}M", 15.5, 'demi', INK)
        for key, xp in (('fix', xb), ('tok', xc)):
            mv = r[key]
            if mv > hi:
                text(xp + pb_w, ry + rh * 0.66, f'{mv:.3f} (off scale)', 14, 'demi', INK2, ha='right')
                continue
            seeds = [s_[key] for s_ in r['seeds']]
            cv.scatter([bx(v_, xp, pb_w) for v_ in seeds], [ry + rh * 0.5] * len(seeds), s=26, color=col, alpha=0.45,
                       lw=0, zorder=4)
            cv.scatter([bx(mv, xp, pb_w)], [ry + rh * 0.5], s=120, marker=MARK[r['group']], color=col,
                       edgecolor='white', lw=1.0, zorder=5)
            text(xp + pb_w, ry + rh * 0.66, f'{mv:.3f}', 14.5, 'demi', INK, ha='right')
    yb = y0 + len(rows) * rh + 1.5
    tx = (xa + xa + pa_w * 201.3 / tmax) / 2
    lx = xa + pa_w * 201.3 / tmax

    for x_, w_ in ((xb, pb_w), (xc, pb_w)):
        hline(x_, x_ + w_, yb, color='#9aa0a8', lw=1.0)
        for t in (1.10, 1.12, 1.14, 1.16, 1.18):
            cv.plot([bx(t, x_, w_)] * 2, [yb, yb + 1.4], color='#9aa0a8', lw=1.0)
            text(bx(t, x_, w_), yb + 5.8, f'{t:.2f}', 13.5, 'reg', INK2, ha='center')
        arrow(x_ + w_ * 0.32, yb + 10.5, x_ + 2, yb + 10.5, color=INK2, lw=1.1, ms=9)
        text(x_ + w_ * 0.34, yb + 11.7, 'lower = better language modelling', 13.5, 'it', INK2)
    hline(xa, xa + pa_w, yb, color='#9aa0a8', lw=1.0)
    for t in (0, 100, 200):
        cv.plot([xa + pa_w * t / tmax] * 2, [yb, yb + 1.4], color='#9aa0a8', lw=1.0)
        text(xa + pa_w * t / tmax, yb + 5.8, f'{t}M' if t else '0', 13.5, 'reg', INK2, ha='center')
    text(xa, yb + 11.7, 'training tokens', 13.5, 'it', INK2)
    ci = M['CRSixOpTwSevenTimeCI'].replace('[', '').replace(']', '').replace(', ', ' to ')
    nx_ = bx(min(lead) - 0.004, xb, pb_w) - 9
    text(nx_, y0 + rh * 2.2, 'not resolved:', 14.5, 'demi', VIOLET, ha='right')
    text(nx_, y0 + rh * 2.2 + 5.6, '95% CIs of differences include 0', 13, 'it', VIOLET, ha='right')
    best = bx(R['arref']['tok'], xc, pb_w)
    text(best + 11, y0 + rh * 1 + rh * 0.66, 'best in every seed', 13.5, 'demi', ARC)
    return yb + 15


def pareto(x, y, w, h, key, title, ylim, show_ylabel):
    yy = panel_title(x, y, title)
    ax = ax_at(x + 18, yy + 2, w - 20, h)
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
    ax.fill_between(xs, ys, [hi] * len(xs), color='#eef0f3', zorder=0, lw=0)
    ax.plot(xs[1:-1], ys[1:-1], color=INK2, lw=1.4, ls=(0, (4, 2)), zorder=4)
    for r in P['recipes']:
        q, v, g = r[key], r['dec'], r['group']
        ax.scatter([v], [min(q, hi - 0.002)], s=120 if g == 'opus' else 80, marker=MARK[g], color=COL[g],
                   edgecolor='white', lw=1.0, zorder=5, clip_on=False)
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
    style_ax(ax, 14)
    ax.set_xlabel('decode tokens/s (log scale)', fontproperties=prop('reg', 14), color=INK2, labelpad=2)
    if show_ylabel:
        ax.set_ylabel('held-out BPB (better: up)', fontproperties=prop('reg', 14), color=INK2, labelpad=3)
    fs = 15
    lab = lambda k, txt, dx, dy, ha='left', col=None: ax.annotate(
        txt, (R[k]['dec'], min(R[k][key], hi - 0.002)), xytext=(dx, dy), textcoords='offset points', ha=ha,
        va='center', fontproperties=prop('demi', fs), color=col or COL[R[k]['group']], zorder=7)
    ov = sum(R[k]['dec'] for k in OPUS_IDS) / 3
    oq = max(R[k][key] for k in OPUS_IDS)
    ax.annotate('Opus\nA23, A24, A27', (ov, oq), xytext=(0, -13), textcoords='offset points', ha='center', va='top',
                fontproperties=prop('demi', fs), color=VIOLET, linespacing=1.0)
    if key == 'tok':
        lab('arref', 'AR-ref', 9, 3)
        lab('gpt_a22', 'GPT A22', 9, -9)
        lab('gpt_a24', 'GPT A24', 9, 0)
        lab('gpt_a25', 'GPT A25', 9, -9)
        ax.annotate('starting\nmodel', (R['baseline']['dec'], R['baseline']['tok']), xytext=(-5, -15),
                    textcoords='offset points', ha='left', va='top', fontproperties=prop('demi', 14), color=BASEC,
                    linespacing=1.0, zorder=7)
        ax.text(215, 1.1125, 'classical\noptimizers (5)', fontproperties=prop('demi', fs - 0.5), color=CLC,
                linespacing=1.0, va='center')
    else:
        lab('arref', 'AR-ref', 0, 14, 'center')
        lab('gpt_a22', 'GPT A22', 9, 7)
        lab('gpt_a24', 'GPT A24', 9, 0)
        lab('gpt_a25', 'GPT A25', 9, 0)
        ax.annotate(f'starting model {R["baseline"]["fix"]:.3f}, off scale', (R['baseline']['dec'], hi - 0.002),
                    xytext=(9, 0), textcoords='offset points', ha='left', va='center', fontproperties=prop('reg', 13.5),
                    color=BASEC)
        ax.text(132, 1.1738, 'classical (5)', fontproperties=prop('demi', fs - 0.5), color=CLC, va='center')
        ax.text(0.97, 0.36, 'leaders’ BPB differences\nnot statistically resolved', transform=ax.transAxes,
                ha='right', fontproperties=prop('it', 14), color=INK2, linespacing=1.1)
    ax.text(0.97, 0.05, 'dominated', transform=ax.transAxes, ha='right', fontproperties=prop('it', 13.5),
            color=MUTED)
    return yy + 2 + h + 10


def memory(x, y, w, h):
    yy = panel_title(x, y, 'Third objective: memory')
    note(x, yy - 0.5, w, 'peak inference MiB, lower is better', 14, MUTED, 'it')
    rows = ['arref', 'gpt_a22', 'opus_a27']
    y0 = yy + 10
    rh = (h - 14) / 3
    bw = w - 4
    for i, k in enumerate(rows):
        r = R[k]
        col = COL[r['group']]
        ry = y0 + i * rh
        text(x, ry + 4.5, r['name'], 16, 'demi', col)
        L_ = bw * r['mem'] / 360.0
        rect(x, ry + 6.5, L_, rh * 0.42, col, z=3)
        text(x + L_ + 1.8, ry + 6.5 + rh * 0.36, f"{r['mem']:.0f}", 17, 'bold', INK)
    return y0 + 3 * rh


def section4(y):
    yy = sec_head(X0, y, CW, 4, 'What did independent verification reveal?', [
        ('All 13 recipes were retrained on new seeds 6 to 8 under two separate contracts. ', 'reg', INK2),
        ('A 300 s budget rewards speed with more data; equal tokens reveal learning per token.', 'demi', INK)])
    yb = budget(X0, yy + 5, CW)
    mw = 104.0
    pw = (CW - mw - 2 * 14) / 2
    ph = 80.0
    y1 = pareto(X0, yb + 2, pw, ph, 'tok', 'Quality vs speed: equal tokens (201.3M)', (1.0905, 1.1935), True)
    pareto(X0 + pw + 14, yb + 2, pw, ph, 'fix', 'Equal time (300 s)', (1.1485, 1.1935), False)
    memory(X0 + 2 * pw + 28, yb + 2, mw, ph + 6)
    yn = note(X0, y1 + 4.0, CW, 'All recipes timed on the same benchmark: 1,024 tokens in, 128 out, batch 1, one A40. '
                              'Dashed: frontier of the seed means; shaded: dominated in quality and speed.', 14, MUTED)
    return yn


# ================================================================ 5: lessons
def section5(y):
    yy = sec_head(X0, y, CW, 5, 'What this teaches about autonomous ML research')
    items = [('The contract shapes the search.', '300 s rewards faster training with more data; with decode steps '
                                                 'scored, Opus engineered the inference engine.'),
             ('Verify at matched exposure.', 'At equal tokens AR-ref learns most per token; at equal time the '
                                             'leaders are not separated.'),
             ('Report what was timed.', '32× faster decode steps is 4.6× faster generation: graph capture runs in '
                                        'the unscored prefill.')]
    cwid = (CW - 2 * 12) / 3
    bottoms = []
    for i, (h_, b_) in enumerate(items):
        x = X0 + i * (cwid + 12)
        yh = para(x + 6, yy + 1, cwid - 6, [(h_, 'demi', INK)], 19, lead=1.15, justify=False)
        bottoms.append(para(x + 6, yh + 1.5, cwid - 6, [(b_, 'reg', INK2)], 17, lead=1.2, justify=False))
    for i in range(3):
        rect(X0 + i * (cwid + 12), yy + 1, 2.2, max(bottoms) - yy - 1, VIOLET, z=2)
    return max(bottoms)


def limits(y):
    return para(X0, y, CW, [
        ('Limits. ', 'demi', INK),
        ('Three seeds; each compared search ran once under a different harness, so this is no ranking of models; the '
         '32× compares inference engines as well as models and measures speed, not quality. ', 'reg', INK2),
        ('AI use. ', 'demi', INK),
        ('Claude (Opus 5.5, Claude Code) ran the camera-ready experiments under approved plans and drafted this poster; '
         'authorship is human. ', 'reg', INK2),
        ('Refs. ', 'demi', INK),
        ('Karpathy, autoresearch (2026); Ferreira et al., arXiv:2603.24647 (2026).', 'reg', INK2)], 13, lead=1.2,
        justify=False)


def build():
    header()
    footer()
    ends = {}
    y = section1(HEAD_H + 8); ends['s1'] = y
    y = section2(y + 7); ends['s2'] = y
    y = section3(y + 6); ends['s3'] = y
    y = section4(y + 6); ends['s4'] = y
    y = section5(y + 6); ends['s5'] = y
    yl = limits(y + 3.5)
    print({k: round(v, 1) for k, v in ends.items()})
    print(f'content ends at {yl:.1f} mm; footer at {H - FOOT_H:.1f}')
    OUT.mkdir(exist_ok=True)
    meta = {'Title': 'AutoPareto: NeurIPS 2026 AutoMLR poster (A1 portrait, version 6)', 'Author': 'Rami Alobaidy',
            'Subject': 'Auditing Fixed-Time Autonomous Transformer Search with Matched-Exposure Verification'}
    fig.savefig(OUT / 'autopareto_poster_v6_A1.pdf', metadata=meta)
    fig.savefig(OUT / 'autopareto_poster_v6_A1.svg')
    fig.savefig(OUT / 'autopareto_poster_v6_preview.png', dpi=150)
    fig.savefig(OUT / 'autopareto_poster_v6_preview_small.png', dpi=36)
    print('wrote', OUT)


if __name__ == '__main__':
    build()
