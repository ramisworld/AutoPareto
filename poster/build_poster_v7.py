#!/usr/bin/env python3
"""AutoPareto poster, version 7: A1 portrait (594 x 841 mm), four questions, fewer and larger figures.

    /path/to/paper/.venv/bin/python poster/build_poster_v7.py

Writes poster/out_v7/autopareto_poster_v7_A1.{pdf,svg}, a 150-dpi PNG and a 2-3 m viewing preview. Versions 1 to 6
are unchanged. Every number comes from poster_data.load(), which reads the paper's frozen records and asserts each
printed value against the paper. The CPU/GPU lanes share one time axis: lane length is the time per decoded token in
the ablation session, 1 / (149 tok/s x the median speed-up); one mark stands for 10 profiler-counted CPU launches or
GPU kernels, and mark positions are schematic. The header's QR code (qr/) links to the paper's OpenReview page.
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

OUT = HERE / 'out_v7'
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
HEAD_H, FOOT_H = 89.0, 25.0
SEC, LEDE, LAB, TK, NOTE = 32, 20, 17, 16, 15


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
QR_URL = 'https://openreview.net/forum?id=U9ICjFvtN8'      # the paper's OpenReview page, given by the author


def qr_modules():
    """The QR module matrix for QR_URL, made with macOS CoreImage by qr/qr_matrix.swift (error correction M) and stored
    in qr/openreview_forum_M.txt, so the build needs no network or extra package. CoreImage's 1-module border is cut;
    qr_tile() adds the standard 4-module quiet zone."""
    rows = (HERE / 'qr' / 'openreview_forum_M.txt').read_text().split()
    keep_r = [i for i, r in enumerate(rows) if '1' in r]
    keep_c = [j for j in range(len(rows[0])) if any(r[j] == '1' for r in rows)]
    return [r[keep_c[0]:keep_c[-1] + 1] for r in rows[keep_r[0]:keep_r[-1] + 1]]


def qr_tile(x, y, size, dark='#10131c'):
    """A white tile with the QR code as one vector path (dark runs merged per row, so no hairline seams)."""
    from matplotlib.path import Path as MPath
    from matplotlib.patches import PathPatch
    mods = qr_modules()
    n, quiet = len(mods), 4
    m = size / (n + 2 * quiet)
    box(x, y, size, size, 'white', r=1.6, z=2)
    verts, codes = [], []
    for r, row in enumerate(mods):
        c = 0
        while c < n:
            if row[c] == '1':
                c1 = c
                while c1 < n and row[c1] == '1':
                    c1 += 1
                x0_, y0_ = x + (quiet + c) * m, y + (quiet + r) * m
                x1_, y1_ = x + (quiet + c1) * m, y + (quiet + r + 1) * m + 0.02
                verts += [(x0_, y0_), (x1_, y0_), (x1_, y1_), (x0_, y1_), (x0_, y0_)]
                codes += [MPath.MOVETO, MPath.LINETO, MPath.LINETO, MPath.LINETO, MPath.CLOSEPOLY]
                c = c1
            else:
                c += 1
    cv.add_patch(PathPatch(MPath(verts, codes), fc=dark, ec='none', lw=0, zorder=3))
    return n, m


def header():
    rect(0, 0, W, HEAD_H, NAVY, z=0)
    rect(0, HEAD_H - 2.2, W, 2.2, VIOLET, z=1)
    text(X0, 34, 'AutoPareto', 104, 'bold', 'white')
    text(X0, 49.0, 'An AI research agent shrank the model and rewrote the inference engine:', 30, 'demi', '#cfc8ff')
    text(X0, 61.0, '~32× faster decoding, 4.6× faster generation', 30, 'demi', 'white')
    ft = 'Auditing Fixed-Time Autonomous Transformer Search with Matched-Exposure Verification'
    text(X0, 72.5, ft, 18.5, 'med', '#e7eaf0')
    text(X0, 81.5, 'Rami Alobaidy   ·   NeurIPS 2026 Workshop for Autonomous Machine Learning Research (AutoMLR), Sydney',
         15.5, 'reg', '#b7bfcc')
    qs = 60.0
    qx = X1 - qs
    qr_tile(qx, 8.0, qs)
    text(qx + qs / 2, 75.5, 'Scan for the paper', 16, 'demi', 'white', ha='center')
    text(qx + qs / 2, 81.5, 'openreview.net', 14, 'reg', '#b7bfcc', ha='center')


def footer():
    """Compact footer: the scientific qualifications and disclosure that section 5 used to carry, then contact."""
    y0 = H - FOOT_H
    rect(0, y0, W, FOOT_H, NAVY, z=0)
    rect(0, y0, W, 1.6, VIOLET, z=1)
    lt, wt = '#c9cfda', 'white'
    para(X0, y0 + 3.8, CW, [
        ('Limits. ', 'demi', wt),
        ('Three seeds; each compared search ran once under a different harness, so this is no ranking of models. The '
         '32× compares inference engines as well as models and measures speed, not quality. ', 'reg', lt),
        ('AI use. ', 'demi', wt),
        ('Claude (Opus 5.5, Claude Code) ran the camera-ready experiments under approved plans and drafted this '
         'poster; authorship is human. ', 'reg', lt),
        ('Refs. ', 'demi', wt),
        ('Karpathy, autoresearch (2026); Ferreira et al., arXiv:2603.24647 (2026).', 'reg', lt)], 14, lead=1.25,
        justify=False)
    hline(X0, X1, H - 9.8, color='#3a4466', lw=0.8)
    yb = H - 3.8
    text(X0, yb, 'rameonix@gmail.com   ·   rami.co.nz   ·   @vicilah on X', 14, 'reg', '#dfe3ea')
    text(W / 2, yb, 'Code: github.com/ramisworld/autopareto', 14, 'demi', 'white', ha='center')
    text(X1, yb, 'NeurIPS 2026  ·  AutoMLR Workshop  ·  Sydney', 14, 'reg', '#dfe3ea', ha='right')


# ================================================================ 1: the research loop and one whole generation
def loop(x, y, w):
    ix, iw = x + 10, w - 10                     # the return path runs in the 10 mm on the left
    gap, bh = 11.0, 27.0
    bw = (iw - 2 * gap) / 3
    items = [('Agent', 'GPT-5.6 or Claude Opus 5.5'), ('edits train.py', 'with a recorded hypothesis'),
             ('trains 300 s', 'protected harness, one A40')]
    for i, (t1, t2) in enumerate(items):
        bx = ix + i * (bw + gap)
        box(bx, y, bw, bh, LAV if i == 0 else '#f3f4f6', ec=LAV_EDGE if i == 0 else '#d8dbe0', lw=1.0, r=2)
        text(bx + bw / 2, y + 11.2, t1, 19.5, 'demi', INK, ha='center')
        text(bx + bw / 2, y + 20.0, t2, fit_size(t2, 15.5, 'reg', bw - 5), 'reg', INK2, ha='center')
        if i < 2:
            arrow(bx + bw + 1.0, y + bh / 2, bx + bw + gap - 1.0, y + bh / 2, color='#8f96a3', lw=1.7, ms=14)
    yo = y + bh + 14
    tx_ = ix + 2 * (bw + gap) + bw / 2
    arrow(tx_, y + bh + 1.0, tx_, yo - 1.0, color='#8f96a3', lw=1.7, ms=14)
    text(tx_ - 3.5, y + bh + 9.0, 'scored on', 15.5, 'it', INK2, ha='right')
    objs = [('Quality', 'bits per byte, unseen text', 'lower is better'),
            ('Decode speed', 'tokens/s, batch 1', 'higher is better'),
            ('Memory', 'peak inference MiB', 'lower is better')]
    og, oh = 7.0, 31.0
    ow = (iw - 2 * og) / 3
    for i, (t1, t2, t3) in enumerate(objs):
        bx = ix + i * (ow + og)
        box(bx, yo, ow, oh, 'white', ec=VIOLET, lw=1.4, r=2)
        text(bx + ow / 2, yo + 10.2, t1, 19.5, 'demi', VIOLET, ha='center')
        text(bx + ow / 2, yo + 18.4, t2, fit_size(t2, 15.5, 'reg', ow - 4), 'reg', INK2, ha='center')
        text(bx + ow / 2, yo + 25.6, t3, 15.5, 'it', MUTED, ha='center')
    # the loop closes: every attempt reads all earlier ones and picks a parent (paper section 2)
    yr = yo + oh + 7.0
    for i in range(3):
        sx = ix + i * (ow + og) + ow / 2
        cv.plot([sx, sx], [yo + oh, yr], color=VIOLET, lw=1.7, zorder=4)
    xl = x + 3.0
    cv.plot([ix + 2 * (ow + og) + ow / 2, xl, xl], [yr, yr, y + bh / 2], color=VIOLET, lw=1.7, zorder=4,
            solid_joinstyle='round', solid_capstyle='butt')
    arrow(xl, y + bh / 2, ix - 0.8, y + bh / 2, color=VIOLET, lw=1.7, ms=14)
    text(ix + iw / 2, yr + 7.6, 'repeat: each attempt reads all earlier ones and picks a parent to edit', 16, 'it',
         VIOLET, ha='center')
    yb = note(x, yr + 13.5, w, 'Compared against AutoResearch (scored on quality only) and five classical optimizers.',
              16.5, INK2)
    return yb


def latency(x, y, w):
    ix, iw = x + 8, w - 16
    yy = panel_title(ix, y + 7, 'One complete generation: 1,024-token prompt, 128 new tokens', 18.5)
    lab_w = 33.0
    bx0 = ix + lab_w
    sc = (iw - lab_w - 25) / 880.0
    bh = 16.0
    rows = [('Starting\nmodel', L['base_pre'], L['base_dec'], BASEC, yy + 3), ('Opus', L['op_pre'], L['op_dec'],
                                                                               VIOLET, yy + 26)]
    for name, pre, dec, col, ry in rows:
        lines = name.split('\n')
        for k, ln in enumerate(lines):
            text(ix, ry + bh / 2 + 2.6 + (k - (len(lines) - 1) / 2) * 7.0, ln, 18, 'demi', INK)
        rect(bx0, ry, pre * sc, bh, PREFILL, ec='#9b968c', lw=0.7, z=3, hatch='///')
        rect(bx0 + pre * sc, ry, dec * sc, bh, col, z=3)
        text(bx0 + (pre + dec) * sc + 2.2, ry + bh / 2 + 2.8, f'{pre + dec:.0f} ms', 20, 'bold', INK)
    text(bx0 + (L['base_pre'] + L['base_dec'] / 2) * sc, yy + 3 + bh / 2 + 2.6, f'decode {L["base_dec"]:.0f} ms', 18,
         'demi', 'white', ha='center')
    ry = yy + 26
    pl = f'{L["op_pre"]:.0f} ms'
    text(bx0 + L['op_pre'] * sc / 2, ry + bh / 2 + 2.3, pl, fit_size(pl, 16, 'demi', L['op_pre'] * sc - 5), 'demi', INK,
         ha='center')
    ox = bx0 + (L['op_pre'] + L['op_dec']) * sc + 2.2 + tw(f"{L['op_tot']:.0f} ms", 20, 'bold') + 4
    text(ox, ry + bh / 2 + 2.6, f'decode {L["op_dec"]:.0f} ms', 18, 'demi', VIOLET)
    yk = ry + bh + 7
    rect(bx0, yk, 7, 5.0, PREFILL, ec='#9b968c', lw=0.7, z=3, hatch='///')
    text(bx0 + 9.5, yk + 4.4, 'prefill: Opus captures its CUDA graph here; recorded, not scored', 15.5, 'reg', INK2)
    ys = yk + 10
    half = (iw - 6) / 2
    for i, (num, col, l1, l2, c2) in enumerate((('~32×', VIOLET, 'faster decoding', 'the scored metric', MUTED),
                                                ('4.6×', INK, 'faster complete', 'generation', INK))):
        bxl = ix + i * (half + 6)
        box(bxl, ys, half, 31, 'white', ec='#dcdfe4', lw=0.9, r=2)
        nw = tw(num, 54, 'bold')
        tx_w = max(tw(l1, 18, 'demi'), tw(l2, 18, 'demi'))
        sx = bxl + (half - nw - 4 - tx_w) / 2
        text(sx, ys + 23.0, num, 54, 'bold', col)
        text(sx + nw + 4, ys + 13.6, l1, 18, 'demi', INK)
        text(sx + nw + 4, ys + 21.6, l2, 18, 'demi', c2)
    bot = ys + 31 + 9.0
    text(ix + iw, bot - 3.2, '13-recipe test, seeds 6 to 8; Opus = mean of A23, A24, A27', 14.5, 'it', MUTED,
         ha='right')
    box(x, y, w, bot - y, '#f6f5fd', ec=LAV_EDGE, lw=1.0, r=2.5)
    return bot


def section1(y):
    yy = sec_head(X0, y, CW, 1, 'What does AutoPareto test?',
                  [('An AI agent repeatedly rewrites a Transformer’s training program, trains each version for 300 s, '
                    'and is scored on three objectives at once.', 'reg', INK2)])
    lw_ = 262.0
    y1 = loop(X0, yy + 7, lw_)
    y2 = latency(X0 + lw_ + 18, yy + 3, CW - lw_ - 18)
    return max(y1, y2)


# ================================================================ 2: what the agents discovered
def lineage(x, y, w, h):
    yy = panel_title(x, y, 'Opus’s search, attempt by attempt', 19)
    note(x, yy - 0.5, w, 'historical search records: one timing per attempt', 15.5, MUTED, 'it')
    ax = ax_at(x + 19, yy + 8.5, w - 21, h)
    rows = P['opus_rows']
    by = {r['a']: r for r in rows}
    lo, hi = 0.85, 125
    for a0, a1, lab, col in ((0.5, 7.5, 'smaller model, batch', ARCH), (7.5, 22.5, 'mostly inference engine', VIOLET),
                             (22.5, 27.5, 'final recipes', INK2)):
        ax.axvspan(a0, a1, color='#eef6f7' if col == ARCH else ('#f1effc' if col == VIOLET else '#f5f6f8'), zorder=0,
                   lw=0)
        ax.text((a0 + a1) / 2, 0.972, lab, transform=ax.get_xaxis_transform(), ha='center', va='top',
                fontproperties=prop('demi', 17), color=col)
    par = P['opus_parent']
    path, cur = [], 23
    while cur:
        path.append(cur)
        cur = par.get(cur)
    path = sorted(path)
    ax.plot(path, [by[k]['dec'] for k in path], color=VIOLET, lw=2.4, alpha=0.55, zorder=2)
    for r in rows:
        unc = 'uncertain' in r['status']
        ax.scatter([r['a']], [r['dec']], s=48, color='white' if unc else '#b9b2ee', edgecolor='#b9b2ee', lw=1.3,
                   zorder=3)
    marks = [(4, 'smaller model', ARCH, (18, -26), 'left'), (10, 'CUDA graph replay', VIOLET, (-16, 22), 'right'),
             (14, 'kernel fusion', VIOLET, (-16, 21), 'right'), (18, 'ring K/V cache', VIOLET, (18, -30), 'left'),
             (19, 'custom attention', VIOLET, (-46, 22), 'right'), (22, 'fused decode step', VIOLET, (4, 30),
                                                                      'center')]
    for at, lab, col, off, ha in marks:
        v = by[at]['dec']
        unc = 'uncertain' in by[at]['status']
        ax.scatter([at], [v], s=230, color='white' if unc else col, edgecolor=col, lw=2.6, zorder=5)
        ax.annotate(f'A{at} {lab}  {v:.1f}×', (at, v), xytext=off, textcoords='offset points', ha=ha, va='center',
                    fontproperties=prop('demi', 19), color=INK, zorder=6,
                    arrowprops=dict(arrowstyle='-', color='#8f96a1', lw=1.0, shrinkA=2, shrinkB=6))
    # the recorded hypothesis next to A10
    ax.annotate('', (10, by[10]['dec']), (12.2, 3.3), arrowprops=dict(arrowstyle='-', color=VIOLET, lw=1.2,
                                                                       shrinkA=0, shrinkB=7))
    ax.text(10.5, 1.58, 'A10 hypothesis, recorded before the run:\n' + Q['op10a'], fontproperties=prop('it', 17),
            color=INK, va='bottom', ha='left', linespacing=1.18, zorder=7,
            bbox=dict(boxstyle='round,pad=0.5,rounding_size=0.6', fc='white', ec=VIOLET, lw=1.2))
    for k_, (yk, lab) in enumerate(((0.30, 'open marker: speed uncertain'), (0.22, 'small dots: other attempts'),
                                    (0.14, 'line: recorded path to A23'))):
        if k_ == 0:
            ax.scatter([0.815], [yk], s=60, color='white', edgecolor='#8f96a1', lw=1.3, transform=ax.transAxes,
                       zorder=7)
        elif k_ == 1:
            ax.scatter([0.815], [yk], s=40, color='#b9b2ee', transform=ax.transAxes, zorder=7)
        else:
            ax.plot([0.80, 0.83], [yk, yk], color=VIOLET, alpha=0.55, lw=2.4, transform=ax.transAxes, zorder=7)
        ax.text(0.84, yk, lab, transform=ax.transAxes, va='center', fontproperties=prop('reg', 15.5), color=INK2)
    ax.set_xlim(0.5, 27.5)
    ax.set_yscale('log')
    ax.set_ylim(lo, hi)
    ax.set_yticks([1, 2, 5, 10, 20, 32])
    ax.set_yticklabels(['1×', '2×', '5×', '10×', '20×', '32×'])
    ax.minorticks_off()
    ax.set_xticks([1, 5, 10, 15, 20, 25, 27])
    ax.grid(axis='y', color='#e3e6ea', lw=0.8, zorder=0)
    style_ax(ax, TK)
    ax.set_xlabel('attempt', fontproperties=prop('reg', TK), color=INK2, labelpad=3)
    ax.set_ylabel('decode speed-up over the starting model', fontproperties=prop('reg', 15.5), color=INK2, labelpad=4)
    return yy + 8.5 + h + 15


def verified_line(x, y, w):
    """The independently verified rates, kept as one annotation (the v6 bar chart is gone)."""
    b, g, o = R['baseline'], R['gpt_a22'], R['opus_a27']
    gx, ox = g['dec'] / b['dec'], o['dec'] / b['dec']
    poster_data.need(f'{ox:.1f}' == M['CRSixDecXVsBaseRecipe'] and f'{gx:.1f}' == M['CRSixGptDecRatio'],
                     'verified 31.6x and 1.8x')
    sep = ('·', 'reg', MUTED)
    runs = [('Independently re-timed', 'demi', INK), (' (13-recipe test, seeds 6 to 8):', 'reg', INK2),
            (f"starting model {b['dec']:,.0f} tok/s", 'demi', INK2), sep,
            (f"GPT A22 {g['dec']:,.0f} tok/s ({gx:.1f}×)", 'demi', GPTC), sep,
            (f"Opus A27 {o['dec']:,.0f} tok/s ({ox:.1f}×)", 'demi', VIOLET)]
    return para(x, y, w, runs, 17.5, lead=1.2, justify=False)


def section2(y):
    yy = sec_head(X0, y, CW, 2, 'What did the agents discover?', [
        ('Both AutoPareto agents shrank the model; AutoResearch kept its size. ', 'reg', INK2),
        ('Only Claude Opus 5.5 then rewrote the inference engine.', 'demi', INK)])
    y1 = lineage(X0, yy + 4, CW, 104)
    return verified_line(X0 + 19, y1 + 2.0, CW - 19)


# ================================================================ 3: how Opus made decoding fast
def lane(x0, y, Lr, kind, c):
    """One decoded token on a CPU row and a GPU row. Length = measured time per token; one mark = 10 CPU launches
    or 10 GPU kernels (profiler counts); mark positions are schematic."""
    ch, gh, gap = 9.0, 9.0, 4.0
    cy, gy = y, y + ch + gap
    text(x0 - 2.5, cy + 6.6, 'CPU', 14, 'demi', MUTED, ha='right')
    text(x0 - 2.5, gy + 6.6, 'GPU', 14, 'demi', MUTED, ha='right')
    rect(x0, cy, Lr, ch, '#f3f0e8', z=1)
    rect(x0, gy, Lr, gh, LANE, z=1)
    if kind in ('reference', 'eager'):
        n = round(c['launches'] / 10)
        nk = round(c['kernels'] / 10)
        step = Lr / n
        kstep = Lr / nk
        for i in range(n):
            lx = x0 + i * step + 0.3
            lw_ = step * 0.55
            rect(lx, cy + 1.4, lw_, ch - 2.8, LAUNCH, z=3)
            if i % 6 == 2:
                arrow(lx + lw_ * 0.9, cy + ch - 0.5, lx + lw_ + 0.65, gy + 0.7, color='#8a909a', lw=0.7, ms=5)
        for j in range(nk):
            rect(x0 + j * kstep + 0.3 + step * 0.55 + 0.35, gy + 1.4, max(0.55, kstep * 0.16), gh - 2.8, SMALLK, z=3)
    else:
        fused = kind == 'fused'
        rect(x0 + 0.25, cy + 1.4, 1.9, ch - 2.8, VIOLET, z=3)
        arrow(x0 + 1.2, cy + ch - 0.5, x0 + 1.4, gy + 0.7, color=VIOLET, lw=1.0, ms=6)
        n = round(c['kernels'] / 10)
        wk = (Lr - 0.3) / n
        for i in range(n):
            rect(x0 + 0.3 + i * wk, gy + 1.4, wk * 0.72, gh - 2.8, '#9d95ec' if fused else SMALLK, z=3)
    return gy + gh


def lanes(x, y, w):
    lab_w = 116.0
    x0 = x + lab_w + 12
    col_w = 74.0
    kx = (w - lab_w - 12 - col_w - 34) / 7.0          # mm per ms of time per decoded token
    T0 = 1000.0 / P['abl_base_tps']                   # starting model in the ablation session: 6.71 ms per token
    sx = x + w - col_w / 2                            # speed-up column centre
    stages = [('1', 'Original reference decoder', 'starting model: 8 layers, 50.3M', 'baseline', 1.0, 'reference',
               INK),
              ('2', 'Opus, graph and fusion off', '3 layers; own attention, ring cache on', 'a23-V4_eager_unfused',
               A['V4_eager_unfused'], 'eager', '#5d6577'),
              ('3', '+ CUDA graph replay', 'recorded per prompt, replayed per token', 'a23-V2_no_fusion',
               A['V2_no_fusion'], 'graph', VIOLET),
              ('4', '+ kernel fusion', 'small kernels merged (TorchInductor)', 'a23-V0_as_submitted',
               A['V0_as_submitted'], 'fused', VIOLET)]
    # column heads: what the lane length is, and how the speed-up follows from it
    text(x0, y + 5.5, 'time per decoded token (lane length, to scale)', 16.5, 'demi', INK)
    text(sx, y + 5.5, 'speed-up', 16.5, 'demi', INK, ha='center')
    text(sx, y + 11.8, f'= {T0:.2f} ms ÷ time', 15, 'it', MUTED, ha='center')
    RP = 28.5
    ry = y + 14.0
    for num, t1, t2, ck, s, kind, col in stages:
        cv.add_patch(Circle((x + 4.0, ry + 5.0), 4.0, fc=col, ec='none', zorder=3))
        text(x + 4.0, ry + 5.15, num, 14.5, 'bold', 'white', ha='center', va='center', zorder=4)
        text(x + 11, ry + 7.2, t1, 18, 'demi', col)
        text(x + 11, ry + 14.4, t2, fit_size(t2, 15, 'reg', lab_w - 12), 'reg', INK2)
        c = C[ck]
        if c['graph_launches']:
            same = 'same ' if kind == 'graph' else ''   # graph replay keeps every kernel of stage 2
            cnt = f"1 graph + {c['launches']:.0f} launches · {same}{c['kernels']:.0f} kernels"
        else:
            cnt = f"{c['launches']:.0f} launches · {c['kernels']:.0f} kernels"
        text(x + 11, ry + 21.6, cnt, fit_size(cnt, 15, 'demi', lab_w - 12), 'demi', MUTED)
        T = T0 / s
        Lr = kx * T
        lane(x0, ry + 0.5, Lr, kind, c)
        text(x0 + Lr + 2.8, ry + 14.0, f'{T:.2f} ms', 18, 'demi', INK)
        text(sx, ry + 16.5, f'{s:.1f}×', 28, 'bold', col, ha='center')
        ry += RP
    # the shared time axis
    ya = ry - 4.0
    hline(x0, x0 + 7 * kx, ya, color='#9aa0a8', lw=1.0)
    for t in range(8):
        cv.plot([x0 + t * kx] * 2, [ya, ya + 1.5], color='#9aa0a8', lw=1.0)
        text(x0 + t * kx, ya + 6.4, str(t), 15, 'reg', INK2, ha='center')
    text(x0 + 7 * kx + 3, ya + 6.4, 'ms', 15, 'reg', INK2)
    cv.plot([sx - col_w / 2 + 3] * 2, [y + 14, ya], color=RULE, lw=0.9, ls=(0, (3, 3)), zorder=1)
    # ablation findings, in the space the faster lanes leave free
    bx_ = x0 + 72.0
    bw_ = sx - col_w / 2 - 6 - bx_
    by_ = y + 14.0 + 2 * RP - 3.0
    runs_t = [(f"Take one part out of the final engine ({A['V0_as_submitted']:.1f}×)", 'demi', INK)]
    items = [[('Fusion without graph replay: ', 'reg', INK2), (f"{A['V1_no_cuda_graph']:.1f}×", 'demi', VIOLET),
              ('; the two interact.', 'reg', INK2)],
             [('Without custom attention: ', 'reg', INK2), (f"{A['V3_sdpa_attention']:.1f}×", 'demi', VIOLET),
              ('; without the ring K/V cache: ', 'reg', INK2), (f"{A['V5_linear_kv_cache']:.1f}×", 'demi', VIOLET),
              ('.', 'reg', INK2)],
             [('Re-timed in the 13-recipe test: ', 'reg', INK2),
              (f"{M['CRSixSpeedLo']} to {M['CRSixSpeedHi']}×", 'demi', VIOLET), ('.', 'reg', INK2)]]

    def draw_box(draw):
        yy_ = para(bx_ + 6, by_ + 5, bw_ - 12, runs_t, 17.5, lead=1.2, justify=False, draw=draw)
        for it in items:
            if draw:
                cv.add_patch(Circle((bx_ + 7.5, yy_ + 4.6), 1.0, fc=VIOLET, ec='none', zorder=4))
            yy_ = para(bx_ + 11, yy_ + 2.2, bw_ - 17, it, 16.5, lead=1.2, justify=False, draw=draw)
        return yy_ + 4.0
    bb = draw_box(False)
    box(bx_, by_, bw_, bb - by_, '#f6f5fd', ec=LAV_EDGE, lw=1.0, r=2)
    draw_box(True)
    # legend and sources
    yl = ya + 12.5
    lx = x0
    for col, lab in ((LAUNCH, 'CPU kernel launch'), (SMALLK, 'GPU kernel'), (VIOLET, 'graph replay call'),
                     ('#9d95ec', 'fused GPU kernel')):
        rect(lx, yl - 4.0, 5.0, 4.4, col, z=3)
        text(lx + 7, yl, lab, 15, 'reg', INK2)
        lx += 7 + tw(lab, 15) + 8
    text(lx + 2, yl, 'one mark = 10; positions schematic', 15, 'it', MUTED)
    return note(x, yl + 5.5, w, f'Counts: torch.profiler census per decoded token (Opus A23; starting model). Times: '
                                f'ablation session on one A40, starting model {M["CRIABaseDecode"]} tok/s, medians of '
                                f'Opus A23, A24, A27.', 15, MUTED)


def section3(y):
    yy = sec_head(X0, y, CW, 3, 'How did Opus make decoding 32× faster?', [
        ('At batch 1 each decoded token is hundreds of tiny GPU kernels, each launched by the CPU. ', 'reg', INK2),
        ('Opus applied standard CUDA graph replay, then fused kernels.', 'demi', INK)])
    return lanes(X0, yy + 3, CW)


# ================================================================ 4: verification
def pareto(x, y, w, h, key, title, sub, ylim):
    yy = panel_title(x, y, title, 19)
    yy = note(x, yy - 0.5, w, sub, 16, INK2, 'it')
    ax = ax_at(x + 20, yy + 4, w - 22, h)
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
    ax.plot(xs[1:-1], ys[1:-1], color=INK2, lw=1.6, ls=(0, (4, 2)), zorder=4)
    for r in P['recipes']:
        q, v, g = r[key], r['dec'], r['group']
        ax.scatter([v], [min(q, hi - 0.002)], s=190 if g == 'opus' else 130, marker=MARK[g], color=COL[g],
                   edgecolor='white', lw=1.2, zorder=5, clip_on=False)
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
    style_ax(ax, TK)
    ax.set_xlabel('decode tokens/s (log scale)', fontproperties=prop('reg', TK), color=INK2, labelpad=3)
    ax.set_ylabel('held-out BPB (better: up)', fontproperties=prop('reg', TK), color=INK2, labelpad=4)
    fs = 17

    def lab(k, txt, dx, dy, ha='left', col=None, size=fs, style='demi'):
        ax.annotate(txt, (R[k]['dec'], min(R[k][key], hi - 0.002)), xytext=(dx, dy), textcoords='offset points',
                    ha=ha, va='center', fontproperties=prop(style, size), color=col or COL[R[k]['group']], zorder=7,
                    linespacing=1.05)
    ov = sum(R[k]['dec'] for k in OPUS_IDS) / 3
    oq = max(R[k][key] for k in OPUS_IDS)
    ax.annotate('Opus A23, A24, A27', (ov, oq), xytext=(0, -15), textcoords='offset points', ha='right' if key == 'fix'
                else 'center', va='top', fontproperties=prop('demi', fs), color=VIOLET)
    if key == 'tok':
        lab('arref', 'AR-ref', 11, 0)
        lab('gpt_a22', 'GPT A22', 11, -4)
        lab('gpt_a24', 'GPT A24', 11, 0, size=15.5, style='reg')
        lab('gpt_a25', 'GPT A25', 11, -10, size=15.5, style='reg')
        ax.annotate('starting\nmodel', (R['baseline']['dec'], R['baseline']['tok']), xytext=(0, -13),
                    textcoords='offset points', ha='center', va='top', fontproperties=prop('demi', 15.5), color=BASEC,
                    linespacing=1.0, zorder=7)
        ax.text(222, 1.1125, 'classical\noptimizers (5)', fontproperties=prop('demi', 15.5), color=CLC,
                linespacing=1.0, va='center')
        head = 'held-out BPB at 201.3M tokens'
        rows = [(f"AR-ref {R['arref']['tok']:.3f}: best in every seed", ARC), (f"GPT A22 {R['gpt_a22']['tok']:.3f}", GPTC),
                (f"Opus A27 {R['opus_a27']['tok']:.3f}", VIOLET)]
    else:
        tk = lambda k: f"{R[k]['ftok'] / 1e6:.0f}M"
        lab('arref', 'AR-ref', -4, 15, size=fs)
        lab('gpt_a22', 'GPT A22', 8, 12)
        lab('gpt_a24', 'GPT A24', 11, 0, size=15.5, style='reg')
        lab('gpt_a25', 'GPT A25', 11, 0, size=15.5, style='reg')
        ax.annotate(f"starting model {R['baseline']['fix']:.3f}, off scale", (R['baseline']['dec'], hi - 0.002), xytext=(11, 0),
                    textcoords='offset points', ha='left', va='center', fontproperties=prop('reg', 15), color=BASEC)
        ax.text(133, 1.1752, 'classical (5)', fontproperties=prop('demi', 15.5), color=CLC, va='center')
        head = 'held-out BPB after 300 s · tokens seen'
        rows = [(f"AR-ref {R['arref']['fix']:.3f} · {tk('arref')}", ARC),
                (f"GPT A22 {R['gpt_a22']['fix']:.3f} · {tk('gpt_a22')}", GPTC),
                (f"Opus A27 {R['opus_a27']['fix']:.3f} · {tk('opus_a27')}", VIOLET),
                ('not resolved: 95% CIs of differences include 0', None)]
    # the numbers, as one right-aligned block in the empty dominated corner
    # equal tokens: the empty non-dominated corner (top right); equal time: the dominated corner (bottom right)
    yb_ = 0.43 if key == 'fix' else 0.90
    ax.text(0.975, yb_, head, transform=ax.transAxes, ha='right', va='bottom', fontproperties=prop('it', 15.5),
            color=INK2)
    for i, (t_, c_) in enumerate(rows):
        ax.text(0.975, yb_ - 0.072 * (i + 1), t_, transform=ax.transAxes, ha='right', va='bottom',
                fontproperties=prop('it' if c_ is None else 'demi', 15.5 if c_ is None else 17),
                color=INK2 if c_ is None else c_)
    return yy + 4 + h + 18


def section4(y):
    yy = sec_head(X0, y, CW, 4, 'What did independent verification reveal?', [
        ('All 13 recipes were retrained on new seeds 6 to 8 under two separate contracts. ', 'reg', INK2),
        ('A 300 s budget rewards speed with more data; equal tokens reveal learning per token.', 'demi', INK)])
    gap = 26.0
    pw = (CW - gap) / 2
    ph = 96.0
    y1 = pareto(X0, yy + 5, pw, ph, 'tok', 'Equal tokens: 201.3M each', 'AR-ref learns the most per token',
                (1.0905, 1.1935))
    pareto(X0 + pw + gap, yy + 5, pw, ph, 'fix', 'Equal time: 300 s each',
           'faster-training recipes see more tokens in the same time', (1.1485, 1.1935))
    mem = [('Third objective, peak inference memory (lower is better):', 'demi', INK)]
    for k in ('arref', 'gpt_a22', 'opus_a27'):
        mem.append((f"{R[k]['name']} {R[k]['mem']:.0f} MiB", 'demi', COL[R[k]['group']]))
        if k != 'opus_a27':
            mem.append(('·', 'reg', MUTED))
    ym = para(X0 + 20, y1, CW - 20, mem, 17.5, lead=1.2, justify=False)
    return note(X0 + 20, ym + 2.5, CW - 20, 'One benchmark for all: 1,024 tokens in, 128 out, batch 1, one A40. Dashed: '
                                            'frontier of the seed means; shaded: dominated.', 15, MUTED)


def build():
    header()
    footer()
    ends = {}
    y = section1(HEAD_H + 9); ends['s1'] = y
    y = section2(y + 11); ends['s2'] = y
    y = section3(y + 11); ends['s3'] = y
    y = section4(y + 11); ends['s4'] = y
    print({k: round(v, 1) for k, v in ends.items()})
    print(f'content ends at {y:.1f} mm; footer at {H - FOOT_H:.1f}')
    OUT.mkdir(exist_ok=True)
    meta = {'Title': 'AutoPareto: NeurIPS 2026 AutoMLR poster (A1 portrait, version 7)', 'Author': 'Rami Alobaidy',
            'Subject': 'Auditing Fixed-Time Autonomous Transformer Search with Matched-Exposure Verification'}
    fig.savefig(OUT / 'autopareto_poster_v7_A1.pdf', metadata=meta)
    fig.savefig(OUT / 'autopareto_poster_v7_A1.svg')
    fig.savefig(OUT / 'autopareto_poster_v7_preview.png', dpi=150)
    fig.savefig(OUT / 'autopareto_poster_v7_view_2-3m.png', dpi=14)
    print('wrote', OUT)


if __name__ == '__main__':
    build()
