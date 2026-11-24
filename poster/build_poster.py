#!/usr/bin/env python3
"""AutoPareto A1 poster (NeurIPS 2026 AutoMLR workshop). One vector page, 594 x 841 mm, drawn with matplotlib.

    /path/to/paper/.venv/bin/python poster/build_poster.py

Writes poster/out/autopareto_poster_A1.{pdf,svg}, a full-page PNG preview and a reduced preview. Every number comes
from poster_data.load(), which reads the paper's frozen records and asserts each printed value against the paper.
Fonts: Avenir Next, extracted at build time from the macOS system collection (embedding permitted for print,
fsType 4) and embedded as TrueType (Type 42) subsets. No font file is stored in the repository.
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
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

import poster_data

OUT = HERE / 'out'
W, H = 594.0, 841.0          # A1 portrait, mm
MARGIN = 18.0
X0, X1 = MARGIN, W - MARGIN
PT = 72 / 25.4               # points per mm

# ---------------------------------------------------------------- colour system (method colours = the paper's)
INK, INK2, MUTED, RULE, TINT = '#12161c', '#4a5160', '#858c98', '#dfe2e6', '#f3f4f6'
DARK, ON_DARK, ON_DARK2 = '#10141b', '#eef1f6', '#a3adbd'
HERO_V = '#b8aeff'
COL = {'baseline': '#8a8985', 'autoresearch': '#eb6834', 'classical': '#1baf7a', 'gpt': '#2a78d6', 'opus': '#6250d6'}
MARK = {'baseline': 's', 'autoresearch': 'D', 'classical': 'o', 'gpt': '^', 'opus': 'v'}
GEMV, TINY, PREFILL = '#3d4f6b', '#a9b1bf', '#d8d4cc'
OPUS_LIGHT = '#e9e6fb'
GPT_LIGHT = '#e4eefb'

# ---------------------------------------------------------------- type scale (pt): physical sizes on the A1 sheet
T = dict(title=128, sub=30, hero=100, take=26, h2=40, lede=23, body=21, fig=19, tick=17, note=16)


def load_fonts():
    """Extract Avenir Next faces from the system collection into a temporary directory and register them."""
    from fontTools.ttLib import TTCollection
    want = {'Regular': 'reg', 'Medium': 'med', 'Demi Bold': 'demi', 'Bold': 'bold', 'Heavy': 'heavy',
            'Italic': 'it', 'Medium Italic': 'medit', 'Demi Bold Italic': 'demiit'}
    tmp = Path(tempfile.mkdtemp(prefix='poster_fonts_'))
    fp = {}
    for f in TTCollection('/System/Library/Fonts/Avenir Next.ttc').fonts:
        full = f['name'].getDebugName(4)
        sub = full.replace('Avenir Next', '').strip() or 'Regular'
        if sub in want:
            p = tmp / f"AvenirNext-{sub.replace(' ', '')}.ttf"
            f.save(str(p))
            font_manager.fontManager.addfont(str(p))
            fp[want[sub]] = FontProperties(fname=str(p))
    missing = set(want.values()) - set(fp)
    if missing:
        raise SystemExit(f'fonts missing: {missing}')
    return fp


FP = load_fonts()
plt.rcParams.update({'pdf.fonttype': 42, 'svg.fonttype': 'none', 'font.family': 'Avenir Next',
                     'axes.unicode_minus': True, 'mathtext.default': 'regular', 'hatch.linewidth': 0.8,
                     'path.simplify': False})

fig = plt.figure(figsize=(W / 25.4, H / 25.4), dpi=72)
cv = fig.add_axes([0, 0, 1, 1])
cv.set_xlim(0, W)
cv.set_ylim(H, 0)
cv.axis('off')
REN = fig.canvas.get_renderer()


# ---------------------------------------------------------------- primitives (page coordinates in mm, y down)
def prop(style, size):
    p = FP[style].copy()
    p.set_size(size)
    return p


def tw(s, size, style='reg'):
    """Width of a string in mm."""
    w, _, _ = REN.get_text_width_height_descent(s, prop(style, size), ismath=False)
    return w / fig.dpi * 25.4


def text(x, y, s, size, style='reg', color=INK, ha='left', va='baseline', ax=None, **kw):
    return (ax or cv).text(x, y, s, fontproperties=prop(style, size), color=color, ha=ha, va=va, **kw)


def lh(size, lead=1.22):
    return size / PT * lead


def para(x, y, width, runs, size, style='reg', color=INK, lead=1.24, first_baseline=True, draw=True):
    """Rich paragraph. runs: str or [(text, style, colour)]; y is the top of the first line. Returns the bottom y."""
    if isinstance(runs, str):
        runs = [(runs, style, color)]
    words = []
    prev_space = True
    for r in runs:
        t, st, co = (r + (color,))[:3] if len(r) == 2 else r
        parts = t.split(' ')
        for i, p in enumerate(parts):
            if p == '' and i > 0:
                continue
            glue = i == 0 and not prev_space and t[:1] in ').,;:'
            words.append([p, st or style, co or color, glue])
        prev_space = t.endswith(' ')
    lines, cur, curw = [], [], 0.0
    for wd in words:
        ww = tw(wd[0], size, wd[1])
        sp = tw(' ', size, wd[1]) if cur and not wd[3] else 0.0
        if cur and curw + sp + ww > width:
            lines.append(cur)
            cur, curw = [wd], ww
        else:
            cur.append(wd)
            curw += sp + ww
    if cur:
        lines.append(cur)
    asc = size / PT * 0.74
    yy = y + asc
    for ln in lines:
        if draw:
            xx = x
            segs = []
            for wd in ln:
                if segs and segs[-1][1] == wd[1] and segs[-1][2] == wd[2]:
                    segs[-1][0] += ('' if wd[3] else ' ') + wd[0]
                else:
                    segs.append([wd[0], wd[1], wd[2], wd[3]])
            for k, (s, st, co, _) in enumerate(segs):
                text(xx, yy, s, size, st, co)
                glue_next = k + 1 < len(segs) and segs[k + 1][3]
                xx += tw(s, size, st) + (0 if glue_next else tw(' ', size, st))
        yy += lh(size, lead)
    return yy - lh(size, lead) + size / PT * 0.30


def box(x, y, w, h, fc, ec='none', lw=0.0, r=3.0, z=0, ax=None, **kw):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw, zorder=z, **kw)
    (ax or cv).add_patch(p)
    return p


def rect(x, y, w, h, fc, ec='none', lw=0.0, z=1, ax=None, **kw):
    p = Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=z, **kw)
    (ax or cv).add_patch(p)
    return p


def arrow(x0, y0, x1, y1, color=INK2, lw=1.6, ms=14, style='-|>', rad=0.0, z=5, ax=None, **kw):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                        connectionstyle=f'arc3,rad={rad}', zorder=z, shrinkA=0, shrinkB=0, **kw)
    (ax or cv).add_patch(a)
    return a


def hline(x0, x1, y, color=RULE, lw=1.0, z=1, **kw):
    cv.plot([x0, x1], [y, y], color=color, lw=lw, zorder=z, solid_capstyle='butt', **kw)


def ax_at(x, y, w, h):
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])


def style_ax(ax, size=None):
    size = size or T['tick']
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#9aa0a8')
        ax.spines[s].set_linewidth(1.1)
    ax.tick_params(labelsize=size, length=5, width=1.1, color='#9aa0a8', labelcolor=INK2, pad=4)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontproperties(prop('med', size))
    ax.set_facecolor('none')


def ticklabels(ax, size=None):
    size = size or T['tick']
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontproperties(prop('med', size))
        lab.set_color(INK2)


def up_arrow(x, y, label, color=INK2, size=None, ax=None):
    """A small 'better' arrow pointing up at page position (x, y) (top of the arrow)."""
    size = size or T['tick']
    arrow(x, y + 9, x, y, color=color, lw=1.4, ms=11, ax=ax)
    text(x + 2.2, y + 6.5, label, size, 'demi', color, ax=ax)


def section(x, y, num, title, width, lede=None, lede_size=None):
    r = 6.6
    cv.add_patch(Circle((x + r, y + r), r, fc=INK, ec='none', zorder=3))
    text(x + r, y + r + 0.15, str(num), 24, 'heavy', 'white', ha='center', va='center', zorder=4)
    text(x + 2 * r + 4.5, y + r + 4.9, title, T['h2'], 'bold', INK)
    yy = y + 2 * r + 4.0
    if lede:
        yy = para(x, yy, width, lede, lede_size or T['lede'], 'med', INK2, lead=1.2)
    return yy


def star_marker(x, y, size_mm, color, ax=None, z=6):
    """Five-pointed star (the font has no star glyph)."""
    pts = []
    for i in range(10):
        a = math.pi / 2 + i * math.pi / 5
        rr = size_mm / 2 if i % 2 == 0 else size_mm / 5
        pts.append((x + rr * math.cos(a), y - rr * math.sin(a)))
    (ax or cv).add_patch(Polygon(pts, closed=True, fc=color, ec='none', zorder=z))


P, M = poster_data.load()
R = {r['key']: r for r in P['recipes']}
Q = P['quotes']
GROUP_LABEL = [('baseline', 'Baseline'), ('autoresearch', 'AutoResearch (quality only)'),
               ('classical', 'Classical search'), ('gpt', 'AutoPareto, GPT-5.6'), ('opus', 'AutoPareto, Claude Opus 5.5')]


# ================================================================ header
def header():
    box(X0, 18, X1 - X0, 104, DARK, r=5)
    text(X0 + 15, 33.5, 'NeurIPS 2026  ·  Workshop for Autonomous Machine Learning Research (AutoMLR)  ·  Sydney',
         19, 'demi', ON_DARK2)
    text(X0 + 13, 77, 'AutoPareto', T['title'], 'heavy', 'white')
    text(X0 + 15, 91.5, 'Auditing Fixed-Time Autonomous Transformer Search', T['sub'], 'med', ON_DARK)
    text(X0 + 15, 103.5, 'with Matched-Exposure Verification', T['sub'], 'med', ON_DARK)
    text(X0 + 15, 115.0, 'Rami Alobaidy   ·   rameonix@gmail.com', 19, 'med', ON_DARK2)
    # hero numbers: the same three recipes, two measurement contracts
    xr = X1 - 14
    cap_w = 112
    for (num, col, l1, l2, yb) in (('32×', HERO_V, 'faster decoding', 'decode steps only, the scored metric', 62),
                                   ('4.6×', 'white', 'faster complete generation', 'prompt prefill + 128 new tokens', 107)):
        text(xr, yb, num, T['hero'], 'heavy', col, ha='right')
        nx = xr - tw(num, T['hero'], 'heavy') - 6
        text(nx, yb - 15.5, l1, 21, 'bold', col, ha='right')
        para(nx - cap_w, yb - 11.0, cap_w, [(l2, 'med', ON_DARK2)], 16.5, draw=False)
        yy = yb - 11.0
        # right-aligned two-line caption
        words, lines, cur = l2.split(' '), [], ''
        for w_ in words:
            t_ = (cur + ' ' + w_).strip()
            if tw(t_, 16.5, 'med') > cap_w and cur:
                lines.append(cur)
                cur = w_
            else:
                cur = t_
        lines.append(cur)
        for i, ln in enumerate(lines):
            text(nx, yy + 4.6 + i * 6.6, ln, 16.5, 'med', ON_DARK2, ha='right')
    hline(X1 - 200, X1 - 14, 72.5, color='#2a3140', lw=1.2, z=2)
    text(X1 - 14, 117.0, 'Opus A23, A24, A27 vs the 50.3M-parameter starting model · one A40 · 1,024 tokens in, 128 out',
         14.5, 'med', '#7f8a9c', ha='right')


def takeaway(y):
    rect(X0, y, 3.2, 20.5, COL['opus'], z=2)
    runs = [('The 300 s budget shaped what the agents discovered. ', 'bold', INK),
            ('GPT-5.6 found smaller, faster architectures; Claude Opus 5.5 also rewrote the inference engine. ', 'med',
             INK),
            ('Its 32× faster decoding is 4.6× faster complete generation: the measurement contract matters.', 'bold',
             INK)]
    return para(X0 + 9, y + 0.5, X1 - X0 - 9, runs, T['take'], lead=1.2)


# ================================================================ 1: the budget
def sec_budget(y0):
    y = section(X0, y0, 1, 'The budget shapes the scientist', X1 - X0, [
        ('Each attempt trains for 300 s, so a faster training program also sees more data. ', 'med', INK2),
        ('Every agent found the same lever: ', 'demi', INK),
        ('smaller batches give more optimizer updates, smaller models more tokens. But faster training is not better '
         'learning per token.', 'med', INK2)])
    yv = y + 4
    loop_diagram(X0, yv, 176, 96)
    e1_chart(221, yv + 4, 172, 78)
    quotes(410, yv, X1 - 410)
    return yv + 96


def loop_diagram(x, y, w, h):
    text(x, y + 5, 'How AutoPareto searches', T['fig'] + 1, 'bold', INK)
    bw, bh, g, gv = (w - 14) / 2, 37.0, 14.0, 7.5
    yy = y + 10
    y2 = yy + bh + gv
    B = [(x, yy, '1  Agent proposes', 'reads every earlier attempt, picks a parent, records a hypothesis, edits train.py'),
         (x + bw + g, yy, '2  Protected training', '300 s on one NVIDIA A40; the agent cannot change the harness'),
         (x + bw + g, y2, '3  Three objectives', 'quality (BPB, lower is better), decode speed, inference memory'),
         (x, y2, '4  Pareto archive', 'failures recorded; frontier: no other attempt at least as good on all three and '
                                     'better on one')]
    for bx, by, t1, t2 in B:
        box(bx, by, bw, bh, TINT, ec=RULE, lw=1.0, r=3)
        text(bx + 4, by + 7.8, t1, 18.5, 'bold', INK)
        para(bx + 4, by + 10.8, bw - 7, t2, 15.0, 'med', INK2, lead=1.13)
    c = '#9aa0a8'
    arrow(x + bw + 1.5, yy + bh / 2, x + bw + g - 1.5, yy + bh / 2, color=c, lw=2.0, ms=15)
    arrow(x + bw + g + bw / 2, yy + bh + 0.8, x + bw + g + bw / 2, y2 - 0.8, color=c, lw=2.0, ms=13)
    arrow(x + bw + g - 1.5, y2 + bh / 2, x + bw + 1.5, y2 + bh / 2, color=c, lw=2.0, ms=15)
    arrow(x + bw / 2, y2 - 0.8, x + bw / 2, yy + bh + 0.8, color=c, lw=2.0, ms=13)
    text(x, y2 + bh + 6.8, 'AutoResearch (Karpathy’s autoresearch) runs the same loop but scores quality only.',
         15.5, 'it', INK2)


def e1_chart(x, y, w, h):
    text(x - 14, y - 1, 'GPT A22 learns less per token but sees more tokens', T['fig'] + 1, 'bold', INK)
    ax = ax_at(x, y + 6, w, h - 6)
    E = P['e1']
    for key, col, mk in (('arref', COL['autoresearch'], 'o'), ('gpt_a22', COL['gpt'], 'o')):
        bx = [p['tok'] / 1e6 for p in E[key]['budget']]
        by = [p['bpb'] for p in E[key]['budget']]
        for p in E[key]['budget']:
            ax.scatter([p['tok'] / 1e6] * 3, p['seeds'], s=26, color=col, alpha=0.35, lw=0, zorder=3)
        ax.plot(bx, by, color=col, lw=3.2, zorder=4, solid_capstyle='round')
        ax.scatter(bx, by, s=95, color=col, edgecolor='white', lw=1.4, zorder=5)
        t3 = E[key]['t300']
        ax.scatter([t3['tok'] / 1e6], [t3['bpb']], s=300, marker='D', color='white', edgecolor=col, lw=3.0, zorder=6)
    a3, g3 = E['arref']['t300'], E['gpt_a22']['t300']
    ax.plot([a3['tok'] / 1e6, g3['tok'] / 1e6], [a3['bpb'], g3['bpb']], color=INK2, lw=1.6, ls=(0, (2, 2)), zorder=2)
    ax.set_xlim(72, 214)
    ax.set_ylim(1.205, 1.085)
    ax.set_xticks([88.6, 157.3, 201.3])
    ax.set_xticklabels(['88.6M', '157.3M', '201.3M'])
    ax.text(a3['tok'] / 1e6, -0.20, f'AR-ref in 300 s: {M["CREOneTokAR"]}M', transform=ax.get_xaxis_transform(),
            ha='left', va='top', fontproperties=prop('demi', 14.5), color=COL['autoresearch'])
    ax.text(g3['tok'] / 1e6 + 4, -0.20, f'GPT A22 in 300 s: {M["CREOneTokAP"]}M', transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontproperties=prop('demi', 14.5), color=COL['gpt'])
    ax.set_yticks([1.10, 1.13, 1.16, 1.19])
    ax.set_yticklabels(['1.10', '1.13', '1.16', '1.19'])
    ax.grid(axis='y', color='#eceef1', lw=1.0, zorder=0)
    style_ax(ax)
    ax.set_xlabel('')
    ax.set_ylabel('held-out BPB', fontproperties=prop('med', T['tick']), color=INK2, labelpad=4)
    fs = 15.5
    ax.annotate('AR-ref', (201.3, E['arref']['budget'][2]['bpb']), xytext=(-14, 9), textcoords='offset points',
                ha='right', va='bottom', fontproperties=prop('bold', T['fig']), color=COL['autoresearch'])
    ax.text(127, 1.1885, 'GPT A22', ha='left', va='center', fontproperties=prop('bold', T['fig']), color=COL['gpt'])
    # equal-token gap at 201.3M
    ya, yg = E['arref']['budget'][2]['bpb'], E['gpt_a22']['budget'][2]['bpb']
    ax.annotate('', (207, ya), (207, yg), arrowprops=dict(arrowstyle='<->', color=INK, lw=1.4, shrinkA=1, shrinkB=1))
    ax.text(209, (ya + yg) / 2, f'+{P["e1_gaps"][2]:.3f}', fontproperties=prop('bold', fs), color=INK, va='center',
            ha='left', clip_on=False)
    # 300 s markers: same BPB
    ax.text(g3['tok'] / 1e6 + 5.5, g3['bpb'] + 0.0035, 'not separated\nat 300 s', fontproperties=prop('bold', fs),
            color=INK, ha='left', va='top', linespacing=1.05)
    ci = P['e1_300']['ci']
    ci_s = f'{ci[0]:+.3f} to {ci[1]:+.3f}'.replace('-', '−')
    ax.text(0.985, 0.035, f'GPT A22 minus AR-ref at 300 s: {M["CRSixGptTimeD"]} (95% CI {ci_s})',
            transform=ax.transAxes, fontproperties=prop('med', fs - 0.5), color=INK2, ha='right')
    # legend glyph for the diamonds
    ax.scatter([0.055], [0.86], s=170, marker='D', color='white', edgecolor=INK2, lw=2.2, transform=ax.transAxes,
               clip_on=False, zorder=7)
    ax.text(0.09, 0.86, 'equal time (300 s)', transform=ax.transAxes, fontproperties=prop('med', fs - 0.5),
            color=INK2, va='center')
    ax.scatter([0.055], [0.74], s=60, color=INK2, transform=ax.transAxes, clip_on=False, zorder=7)
    ax.text(0.09, 0.74, 'equal tokens (seeds: faint)', transform=ax.transAxes, fontproperties=prop('med', fs - 0.5),
            color=INK2, va='center')
    return ax


def quotes(x, y, w):
    text(x, y + 5, 'In the agents’ own words', T['fig'] + 1, 'bold', INK)
    items = [(Q['ar5'], 'AutoResearch #5, batch halved', COL['autoresearch']),
             (Q['gpt1'], 'GPT-5.6 A1, batch ¼', COL['gpt']),
             (Q['gpt8'], 'GPT-5.6 A8, four layers', COL['gpt']),
             (Q['op1'], 'Opus A1, batch halved', COL['opus'])]
    yy = y + 11
    for q, who, col in items:
        rect(x, yy + 0.5, 1.6, 15.5, col, z=2)
        yb = para(x + 5, yy, w - 6, [(q, 'medit', INK)], 17.5, lead=1.16)
        text(x + 5, yb + 5.2, who, 14.5, 'demi', col)
        yy = yb + 10.5
    yy += 1
    para(x, yy, w, [('Start: 8 layers, 50.3M. ', 'demi', INK),
                    (f'GPT A22: 4 layers, {P["params_M"]["gpt_a22"]:.1f}M, trains {M["CRSixGptTrainRatio"]}× '
                     f'as many tokens/s as AR-ref. Opus: 3 layers, 22 to 26M.', 'med', INK2)], 15.5, lead=1.18)


# ================================================================ 2: all 13 recipes
def sec_pareto(y0):
    y = section(X0, y0, 2, 'One benchmark for all 13 recipes', X1 - X0, [
        ('A pre-registered test trained all 13 recipes on seeds 6 to 8 (reusing E1 and Opus runs) for 201.3M tokens '
         'and for 300 s; one A40 timed them all. ', 'med', INK2),
        ('At equal tokens AR-ref learns most per token; at 300 s no recipe is resolved as better than AR-ref, so decode '
         'speed decides. No recipe wins everything.', 'demi', INK)])
    # legend
    yl = y + 6.5
    xx = X0 + 16
    for g, lab in GROUP_LABEL:
        cv.scatter([xx], [yl], s=170 if g != 'autoresearch' else 130, marker=MARK[g], color=COL[g], edgecolor='white',
                   lw=1.0, zorder=5)
        text(xx + 4, yl + 2.2, lab, 16.5, 'demi', INK2)
        xx += tw(lab, 16.5, 'demi') + 13
    cv.plot([xx - 3, xx + 9], [yl, yl], color=INK2, lw=1.8, ls=(0, (3, 2)))
    text(xx + 11, yl + 2.2, 'frontier of the seed means', 16.5, 'demi', INK2)
    xx += tw('frontier of the seed means', 16.5, 'demi') + 24
    cv.scatter([xx], [yl], s=95, facecolors='none', edgecolors=INK2, lw=1.6)
    text(xx + 4, yl + 2.2, 'timing flag', 16.5, 'demi', INK2)
    yp = yl + 9
    hp = 104
    fa = pareto_panel(X0 + 24, yp, 186, hp, 'tok', '(a) Equal training tokens: 201.3M', (1.0905, 1.1935), True)
    fb = pareto_panel(X0 + 238, yp, 188, hp, 'fix', '(b) Equal training time: 300 s', (1.1485, 1.1935), False)
    memory_panel(X0 + 452, yp, X1 - (X0 + 452), hp)
    yn = yp + hp + 19.5
    para(X0, yn, X1 - X0, [
        ('Benchmark: ', 'demi', INK), ('BF16, batch 1, greedy, 1,024 tokens in and 128 out; each rate is the median of 8 '
                                       'windows bracketed by baseline windows. ', 'med', INK2),
        ('Rings: ', 'demi', INK), (f'{M["CRSixUnstableRuns"]} of 39 runs carry a timing-stability flag; dropping them moves '
                                   f'no mean by more than {M["CRSixStableShiftMax"]}% and leaves the frontier unchanged. ',
                                   'med', INK2),
        ('Correctness: ', 'demi', INK), (f'every generated token checked against the recipe’s uncached forward pass: '
                                         f'{M["CRSixAgreePct"]}% of {M["CRSixTokensChecked"]} match, '
                                         f'{M["CRSixNearTies"]} BF16 near-ties, 0 hard mismatches.', 'med', INK2)],
         15.5, lead=1.2)
    return yn + 12.5


def place_column(items, lo, hi, gap):
    items = sorted(items, key=lambda t: t[1])
    ys = []
    for _, yv in items:
        ys.append(max(yv, ys[-1] + gap) if ys else yv)
    over = ys[-1] - hi if ys else 0
    if over > 0:
        ys = [v - over for v in ys]
        for i in range(len(ys) - 2, -1, -1):
            ys[i] = min(ys[i], ys[i + 1] - gap)
    return {n: v for (n, _), v in zip(items, ys)}


def pareto_panel(x, y, w, h, key, title, ylim, ylabel):
    text(x, y + 6, title, T['fig'] + 1, 'bold', INK)
    ax = ax_at(x, y + 9, w, h - 9)
    lo, hi = ylim
    pts = []
    off = {}
    for r in P['recipes']:
        q, v, g = r[key], r['dec'], r['group']
        pts.append((r['key'], q, v))
        if q > hi:
            off[r['key']] = q
            ax.scatter([v], [hi - 0.0018], s=150, marker=MARK[g], color=COL[g], edgecolor='white', lw=1.0, zorder=5,
                       clip_on=False)
            continue
        sk = 'tok' if key == 'tok' else 'fix'
        ok = [(s[sk], s['dec']) for s in r['seeds'] if not s['flag']]
        bad = [(s[sk], s['dec']) for s in r['seeds'] if s['flag']]
        ax.scatter([b for _, b in ok], [a for a, _ in ok], s=34, color=COL[g], alpha=0.42, lw=0, zorder=3)
        ax.scatter([b for _, b in bad], [a for a, _ in bad], s=46, facecolors='none', edgecolors=COL[g], alpha=0.85,
                   lw=1.3, zorder=3)
        ax.scatter([v], [q], s=190 if g != 'autoresearch' else 150, marker=MARK[g], color=COL[g], edgecolor='white',
                   lw=1.2, zorder=5)
    front = poster_data.pareto(pts)
    fp = sorted([p for p in pts if p[0] in front], key=lambda p: p[2])
    xs, ys = [], []
    for i, (_, q, v) in enumerate(fp):
        if i:
            xs.append(v)
            ys.append(ys[-1])
        xs.append(v)
        ys.append(q)
    ax.plot(xs, ys, color=INK2, lw=1.8, ls=(0, (3, 2)), zorder=4)
    ax.set_xscale('log')
    ax.set_xlim(112, 9000)
    ax.set_xticks([125, 250, 500, 1000, 2000, 4000])
    ax.set_xticklabels(['125', '250', '500', '1k', '2k', '4k'])
    ax.minorticks_off()
    ax.set_ylim(hi, lo)
    ax.grid(color='#eceef1', lw=1.0, zorder=0)
    ax.set_axisbelow(True)
    style_ax(ax)
    ax.set_xlabel('decode speed, tokens/s (log scale)', fontproperties=prop('med', T['tick']), color=INK2, labelpad=3)
    if ylabel:
        ax.set_ylabel('held-out BPB', fontproperties=prop('med', T['tick']), color=INK2, labelpad=5)
    # labels: non-Opus recipes in a column between the reference cluster and the Opus cluster
    gap = (hi - lo) * 0.066
    col = [(k, min(q, hi - 0.0018)) for k, q, v in pts if not k.startswith('opus')]
    ty = place_column(col, lo + gap * 0.35, hi - gap * 0.55, gap)
    lab_x = 560
    for k, q, v in pts:
        if k.startswith('opus'):
            continue
        yv = min(q, hi - 0.0018)
        t = R[k]['name'] + (f' ({q:.3f}, off scale)' if k in off else '')
        bold = k in ('arref', 'gpt_a22')
        ax.annotate(t, (v, yv), xytext=(lab_x, ty[k]), textcoords='data', ha='left', va='center',
                    fontproperties=prop('bold' if bold else 'med', 15.5), color=INK if bold else INK2, zorder=7,
                    bbox=dict(boxstyle='square,pad=0.12', fc='white', ec='none'),
                    arrowprops=dict(arrowstyle='-', color='#b4b9c1', lw=0.9, shrinkA=1.5, shrinkB=5))
    oq = max(q for k, q, _ in pts if k.startswith('opus'))
    ov = sum(v for k, _, v in pts if k.startswith('opus')) / 3
    ax.annotate('Opus A23,\nA24, A27', (ov, oq), xytext=(0, -16), textcoords='offset points', ha='center', va='top',
                fontproperties=prop('bold', 15.5), color=COL['opus'], linespacing=1.0)
    # reading aids
    ax.annotate('', xy=(0.975, 0.97), xytext=(0.975, 0.78), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>', color=INK2, lw=1.6, mutation_scale=13))
    ax.text(0.955, 0.875, 'better', transform=ax.transAxes, fontproperties=prop('demi', 15), color=INK2, va='center',
            ha='right')
    ax.annotate('', xy=(0.985, 0.045), xytext=(0.83, 0.045), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>', color=INK2, lw=1.6, mutation_scale=13))
    ax.text(0.905, 0.085, 'faster', transform=ax.transAxes, fontproperties=prop('demi', 15), color=INK2, ha='center')
    return front


def memory_panel(x, y, w, h):
    text(x, y + 6, '(c) Third objective: memory', T['fig'] + 1, 'bold', INK)
    ax = ax_at(x + 31, y + 9, w - 42, h - 9)
    n = len(P['recipes'])
    for i, r in enumerate(P['recipes']):
        yy = i
        ax.barh(yy, r['mem'], height=0.66, color=COL[r['group']], lw=0, zorder=3)
        ax.text(r['mem'] + 6, yy, f"{r['mem']:.0f}", va='center', ha='left', fontproperties=prop('demi', 15),
                color=INK, clip_on=False)
        ax.text(-8, yy, r['name'], va='center', ha='right', fontproperties=prop('med', 15), color=INK2, clip_on=False)
    ax.set_ylim(n - 0.4, -0.6)
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 100, 200, 300])
    ax.set_yticks([])
    ax.grid(axis='x', color='#eceef1', lw=1.0, zorder=0)
    style_ax(ax)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('peak MiB at inference', fontproperties=prop('med', T['tick']), color=INK2, labelpad=3)


# ================================================================ 3: two agents
def sec_agents(x0, y0, w, y_end):
    y = section(x0, y0, 3, 'Two agents, two discoveries', w, [
        ('Both agents first shrank the model, which made decoding about 1.9× faster. ', 'med', INK2),
        ('Only Opus then diagnosed the decoder itself and rebuilt it.', 'demi', INK)])
    yt = y + 4
    gx, ox = x0 + 4, x0 + 122          # rail x positions
    gw, ow = ox - gx - 12, x0 + w - ox - 5
    text(gx - 2, yt + 5.5, 'GPT-5.6', 21, 'bold', COL['gpt'])
    text(gx - 2 + tw('GPT-5.6 ', 21, 'bold'), yt + 5.5, 'seed 0, 25 attempts', 15.0, 'med', INK2)
    text(ox - 2, yt + 5.5, 'Claude Opus 5.5', 21, 'bold', COL['opus'])
    text(ox - 2 + tw('Claude Opus 5.5 ', 21, 'bold'), yt + 5.5, 'seed 3, 27 attempts', 15.0, 'med', INK2)
    yb = yt + 9.5
    # fixed band heights for the first two phases; the third takes what is left above the re-timed row
    retimed_h, cave_h = 18.0, 19.0
    h3 = y_end - cave_h - retimed_h - 3 - (yb + 46 + 3 + 28 + 3)
    bands = []
    yy = yb
    for name, hgt in (('shrink batch and model', 46), ('hit a quality limit', 28), ('different next steps', h3)):
        bands.append((yy, hgt, name))
        yy += hgt + 3
    yend = yy - 3
    for i, (by, bh, name) in enumerate(bands):
        rect(x0 - 2, by, w + 2, bh, TINT if i % 2 == 0 else '#fafbfc', z=0)
        text(x0 + w - 2, by + 5.2, name.upper(), 14, 'bold', MUTED, ha='right')
    cv.plot([gx, gx], [yb + 4, yend - 3], color=COL['gpt'], lw=3.0, zorder=2, solid_capstyle='round')
    cv.plot([ox, ox], [yb + 4, yend - 3], color=COL['opus'], lw=3.0, zorder=2, solid_capstyle='round')

    def node(xr, yv, col, big=False):
        r = 2.7 if big else 2.0
        cv.add_patch(Circle((xr, yv), r, fc=col, ec='white', lw=1.2, zorder=4))

    def item(xr, yv, col, label, sub=None, quote=None, width=100, big=False):
        node(xr, yv, col, big=big)
        text(xr + 5, yv + 2.3, label, 17.0, 'bold', INK)
        yb_ = yv + 3.4
        if sub:
            yb_ = para(xr + 5, yb_ + 0.6, width, [(sub, 'med', INK2)], 15.0, lead=1.14)
        if quote:
            yb_ = para(xr + 5, yb_ + 0.6, width, [(quote, 'medit', INK2)], 15.0, lead=1.14)
        return yb_

    b1, b2, b3 = bands
    # phase 1
    yv = b1[0] + 9.5
    yv = item(gx, yv, COL['gpt'], 'A1 batch ¼', quote=Q['gpt1'], width=gw) + 5.5
    node(gx, yv, COL['gpt'])
    text(gx + 5, yv + 2.3, 'A4 to A8 remove layers, 8 to 4', 17.0, 'bold', INK)
    para(gx + 5, yv + 4.0, gw, [('A4: ', 'demi', INK2), (Q['gpt4'], 'medit', INK2)], 15.0, lead=1.14)
    yv = b1[0] + 9.5
    yv = item(ox, yv, COL['opus'], 'A1 batch halved', quote=Q['op1'], width=ow) + 5.5
    item(ox, yv, COL['opus'], 'A3 to A5 depth 6, 4, 3',
         sub=f'decode {P["opus_path"][4]:.1f}× by A4, {P["opus_path"][5]:.1f}× at A5', width=ow)
    # phase 2
    yv = b2[0] + 9.5
    node(gx, yv, COL['gpt'])
    text(gx + 5, yv + 2.3, f'A8 loses {P["gpt_a8_loss"]} BPB', 17.0, 'bold', INK)
    para(gx + 5, yv + 4.0, gw, [('A18 restores width 512 to ', 'med', INK2), (Q['gpt18'], 'medit', INK2)], 15.0,
         lead=1.14)
    node(ox, yv, COL['opus'])
    text(ox + 5, yv + 2.3, f'A6 depth 2: BPB {P["opus_a6_loss"]:+.3f}'.replace('-', '−'), 17.0, 'bold', INK)
    para(ox + 5, yv + 4.0, ow, [('A7 record: ', 'demi', INK2), (Q['op7'], 'medit', INK2)], 15.0, lead=1.14)
    # phase 3: GPT
    yv = b3[0] + 9.5
    yv = item(gx, yv, COL['gpt'], 'A19 to A22 tune learning rates', width=gw) + 6
    yv = item(gx, yv, COL['gpt'], 'A22 = GPT A22', sub=f'4 layers, {P["params_M"]["gpt_a22"]:.1f}M parameters',
              width=gw, big=True) + 8
    yv = para(gx + 5, yv, gw - 2, [('No inference code in any campaign: ', 'demi', INK),
                                   ('0 of 75 attempts, and 0 of 22 in a later campaign, defined the generation hook.',
                                    'med', INK2)], 15.0, lead=1.14)
    para(gx + 5, yv + 6, gw - 2, [('GPT-5.6 tied decode speed to model size.', 'demi', INK2)], 15.0, lead=1.14)
    # phase 3: Opus, each step with its decode speed-up (search records, log scale)
    steps = [(8, 'A8 own inference code', f'memory −{P["opus_a8_mem"]:.0f} MiB'),
             (10, 'A10 CUDA graph replay', None), (13, 'A13 bf16 weights', None),
             (14, 'A14 compiled kernel fusion', None),
             (18, 'A18 ring K/V cache', None), (19, 'A19 custom decode attention', None),
             (22, 'A22 fused decode step', None)]
    yv = b3[0] + 9.5
    bar_w = ow - 22
    lmax = math.log(36)
    for a, lab, sub in steps:
        node(ox, yv, COL['opus'], big=a in (10, 14, 22))
        text(ox + 5, yv + 2.3, lab, 17.0, 'bold', INK)
        if sub:
            text(ox + 5 + tw(lab + '  ', 17.0, 'bold'), yv + 2.3, sub, 15.0, 'demi', INK2)
            yv += 10.5
            continue
        yq = yv + 3.4
        if a == 10:
            yq = para(ox + 5, yq + 0.6, ow - 2, [('hypothesis: ', 'demi', INK2), (Q['op10a'], 'medit', INK2)], 15.0,
                      lead=1.14)
        sx = P['opus_path'][a]
        unc = a in P['opus_uncertain']
        L = bar_w * math.log(sx) / lmax
        rect(ox + 5, yq + 1.2, L, 3.6, 'white' if unc else COL['opus'], ec=COL['opus'], lw=1.3 if unc else 0, z=3,
             hatch='////' if unc else None)
        text(ox + 5 + L + 1.8, yq + 4.6, f'{sx:.1f}×', 16, 'bold', COL['opus'])
        yv = yq + 11.5
    # re-timed row
    yr = yend + 3
    rect(x0 - 2, yr, w + 2, retimed_h, OPUS_LIGHT, z=0)
    text(x0 + w - 2, yr + 5.2, 'RE-TIMED IN THE 13-RECIPE TEST', 14, 'bold', MUTED, ha='right')
    text(gx - 2, yr + 13.5, f'GPT A22  {M["CRNineGptDecX"]}×', 21, 'heavy', COL['gpt'])
    text(ox - 2, yr + 13.5, f'Opus A23, A24, A27  {M["CRSixSpeedLo"]} to {M["CRSixSpeedHi"]}×', 21, 'heavy',
         COL['opus'])
    yc = yr + retimed_h + 2.5
    para(x0, yc, w, [('Bars: search-record decode speed-ups, log scale; steps accumulate, so each gain depends on the '
                      'ones before; hatched: speed-uncertain. ', 'med', MUTED),
                     ('Each search ran once, under different harnesses: an observed difference in strategy, not a '
                      'ranking of models.', 'demi', INK2)], 14.5, lead=1.14)


# ================================================================ 4: why 32x becomes 4.6x
def sec_engine(x0, y0, w):
    y = section(x0, y0, 4, 'Why 32× decoding is 4.6× generation', w, [
        ('Only decode steps were scored. At batch one they are ', 'med', INK2), ('launch-bound', 'demi', INK),
        (': hundreds of tiny GPU kernels per token, each launched by the CPU.', 'med', INK2)])
    C = P['census']
    yy = y + 3
    # reference strip: the starting model
    box(x0, yy, w, 13, TINT, r=2.5)
    text(x0 + 4, yy + 8.6, 'Starting model, 8 layers:', 16.5, 'bold', INK)
    t2 = f'{C["baseline"]["kernels"]:.0f} kernels and {C["baseline"]["launches"]:.0f} CPU launches per token'
    text(x0 + 4 + tw('Starting model, 8 layers: ', 16.5, 'bold'), yy + 8.6, t2, 16.5, 'med', INK2)
    text(x0 + w - 4, yy + 8.9, '1.0×', 20, 'heavy', INK2, ha='right')
    yy += 17
    yy = mechanism_matrix(x0, yy, w)
    yy = fusion_closeup(x0, yy + 3, w)
    return latency(x0, yy + 3, w)


def lane(x, y, w, n_launch, n_kern, graph, fused, scale_note=False):
    """Schematic CPU and GPU lanes for one decoded token (counts are real; widths are not to scale)."""
    text(x, y + 3.7, 'CPU', 14, 'bold', MUTED)
    text(x, y + 13.7, 'GPU', 14, 'bold', MUTED)
    lx = x + 12
    lw_ = w - 13
    cv.plot([lx, lx + lw_], [y + 4.6, y + 4.6], color=RULE, lw=1.0, zorder=1)
    cv.plot([lx, lx + lw_], [y + 14.6, y + 14.6], color=RULE, lw=1.0, zorder=1)
    if graph:
        rect(lx, y + 0.6, 9, 4.0, COL['opus'], z=3)
        text(lx + 11, y + 4.0, '1 graph replay call', 13.5, 'demi', COL['opus'])
    else:
        k = 34 if n_launch > 200 else 12
        for i in range(k):
            xx = lx + i * (lw_ - 1) / k
            cv.plot([xx, xx], [y + 0.6, y + 4.6], color=INK, lw=1.3, zorder=3, solid_capstyle='butt')
    # kernels: unfused = many thin blocks, fused = fewer wider ones; with the graph they run back to back
    k = 34 if not fused else 12
    span = lw_ * (0.42 if graph else 1.0)
    slot = span / k
    kw_ = slot * (0.86 if graph else 0.38)
    for i in range(k):
        xx = lx + i * slot + (0 if graph else slot * 0.5)
        c = COL['opus'] if fused else TINY
        if i % (6 if not fused else 3) == 0:
            c = GEMV
        rect(xx, y + 10.6, kw_, 4.0, c, z=3)


def mechanism_matrix(x, y, w):
    C, A = P['census'], P['abl']
    rl = 13
    cw = (w - rl - 5) / 2
    ch = 35.5
    xs = [x + rl, x + rl + cw + 5]
    text(xs[0] + cw / 2, y + 5, 'without CUDA graph', 17, 'bold', INK, ha='center')
    text(xs[1] + cw / 2, y + 5, 'with CUDA graph replay', 17, 'bold', COL['opus'], ha='center')
    ys = [y + 8, y + 8 + ch + 4]
    for j, lab in enumerate(['unfused', 'fused']):
        text(x + 4.5, ys[j] + ch / 2, lab, 17, 'bold', INK if j == 0 else COL['opus'], ha='center', va='center',
             rotation=90)
    cells = {(0, 0): ('a23-V4_eager_unfused', 'V4_eager_unfused', 'no graph, no fusion'),
             (0, 1): ('a23-V2_no_fusion', 'V2_no_fusion', 'graph only'),
             (1, 0): ('a23-V1_no_cuda_graph', 'V1_no_cuda_graph', 'fusion only'),
             (1, 1): ('a23-V0_as_submitted', 'V0_as_submitted', 'as submitted')}
    for (j, i), (ck, ak, name) in cells.items():
        cx, cy = xs[i], ys[j]
        best = (j, i) == (1, 1)
        box(cx, cy, cw, ch, OPUS_LIGHT if best else TINT, ec=COL['opus'] if best else 'none', lw=2.0 if best else 0,
            r=2.5)
        text(cx + 4, cy + 7.4, name, 15.5, 'demi', INK2)
        text(cx + cw - 4, cy + 9.2, f'{A[ak]:.1f}×', 27, 'heavy', COL['opus'] if best else INK, ha='right')
        lane(cx + 4, cy + 11.5, cw - 8, C[ck]['launches'], C[ck]['kernels'], graph=i == 1, fused=j == 1)
        cnt = (f"{C[ck]['launches']:.0f} CPU launch calls · {C[ck]['kernels']:.0f} GPU kernels per token")
        text(cx + 4, cy + ch - 2.6, cnt, 14.5, 'med', INK2)
    yb = ys[1] + ch + 3
    yb = para(x, yb, w, [('Graph replay removes CPU launches; fusion removes GPU kernels. ', 'demi', INK),
                         ('The two interact: fusion alone goes from 2.0× to 2.7×, fusion on top of the graph from 11.6× '
                          'to 30.8×. Speed-ups: ablation session, medians of Opus A23, A24, A27; '
                          'counts: profiler census of A23. Split-key decode attention adds '
                          f'{P["abl_ranges"]["attn"]}×, the ring K/V cache {P["abl_ranges"]["ring"]}×.',
                          'med', INK2)], 15.0, lead=1.15)
    return yb + 1.5


def fusion_closeup(x, y, w):
    text(x, y + 5.5, 'Kernel fusion inside one layer’s decode step', 18, 'bold', INK)
    text(x + w, y + 5.5, 'schematic, after Opus A23’s source', 14.5, 'med', MUTED, ha='right')
    yy = y + 10
    # (label, kind): mm = matrix-vector multiply (cuBLAS, unchanged); glue = small pointwise/norm work; attn = attention
    seq = [('q, k, v', 'mm'), (None, 'glue'), ('attention', 'attn'), (None, 'glue'), ('out proj', 'mm'), (None, 'glue'),
           ('MLP up', 'mm'), (None, 'glue'), ('MLP down', 'mm')]
    wid = {'mm': 26.5, 'attn': 25.0, 'glue': 16.0}
    for name, fused in (('eager', False), ('fused', True)):
        text(x, yy + 5.4, name, 15.5, 'bold', COL['opus'] if fused else INK2)
        xx = x + 17
        for lab, kind in seq:
            ww = wid[kind]
            if kind == 'mm':
                rect(xx, yy, ww, 7.5, GEMV, z=3)
                text(xx + ww / 2, yy + 5.3, lab, 13.5, 'demi', 'white', ha='center')
            elif kind == 'attn':
                rect(xx, yy, ww, 7.5, COL['opus'] if fused else TINY, z=3)
                text(xx + ww / 2, yy + 5.3, lab, 13.5, 'demi', 'white' if fused else INK, ha='center')
            elif fused:
                rect(xx, yy, ww, 7.5, COL['opus'], z=3)
                text(xx + ww / 2, yy + 5.3, 'fused', 13.5, 'demi', 'white', ha='center')
            else:
                for k in range(5):
                    rect(xx + 0.6 + k * 3.1, yy + 1.2, 2.1, 5.1, TINY, z=3)
            xx += ww + 1.6
        yy += 13.0
    para(x, yy - 1.5, w, [('Between matrix-vector multiplies the eager step runs many small kernels (norms, rotary '
                           'embedding, gates, residual adds, relu²), each reading and writing GPU memory. Opus compiled '
                           'this work with TorchInductor (Triton kernels): helpers from A14, and from A22 regions that span '
                           'everything between two multiplies, attention included; K/V cache writes stay separate. Fewer '
                           'kernels to launch or replay, and intermediates need not round-trip through memory.', 'med',
                           INK2)], 15.0, lead=1.15)
    return yy + 26


def latency(x, y, w):
    L = P['lat']
    text(x, y + 5.5, 'One whole generation, 13-recipe test', 18, 'bold', INK)
    text(x + w, y + 5.5, 'means; Opus = A23, A24, A27', 14.5, 'med', MUTED, ha='right')
    bx0 = x + 28
    scale = (w - 28 - 30) / 900.0
    rows = [('Baseline', L['base_pre'], L['base_dec'], COL['baseline'], y + 10),
            ('Opus', L['op_pre'], L['op_dec'], COL['opus'], y + 23)]
    for name, pre, dec, col, yy in rows:
        text(x, yy + 6.2, name, 17, 'bold', INK)
        rect(bx0, yy, pre * scale, 9, PREFILL, ec='#9b968c', lw=0.8, z=3, hatch='///')
        rect(bx0 + pre * scale, yy, dec * scale, 9, col, z=3)
        text(bx0 + (pre + dec) * scale + 2, yy + 6.4, f'{pre + dec:.0f} ms', 19, 'heavy', INK)
    yb = y + 10
    text(bx0 + L['base_pre'] * scale + L['base_dec'] * scale / 2, yb + 6.4, f'decode {L["base_dec"]:.0f} ms', 15.5,
         'demi', 'white', ha='center')
    yo = y + 23
    text(bx0 + L['op_pre'] * scale / 2, yo + 6.4, f'prefill {L["op_pre"]:.0f} ms', 15.5, 'demi', INK, ha='center')
    text(bx0 + (L['op_pre'] + L['op_dec']) * scale + 33, yo + 6.4, f'(decode {L["op_dec"]:.0f} ms)', 15.5, 'demi',
         COL['opus'])
    yk = yo + 13
    rect(bx0, yk, 6, 4.4, PREFILL, ec='#9b968c', lw=0.8, z=3, hatch='///')
    text(bx0 + 8, yk + 3.8, 'prompt prefill: recorded, not scored', 14.5, 'med', INK2)
    rect(bx0 + 98, yk, 6, 4.4, COL['opus'], z=3)
    text(bx0 + 106, yk + 3.8, 'decode steps: scored', 14.5, 'med', INK2)
    return para(x, yk + 8, w, [('Decode time fell from ', 'med', INK2), (f'{L["base_dec"]:.0f} to {L["op_dec"]:.0f} ms ', 'bold', INK),
                        (f'(scored speed-up {M["CRSixSpeedLo"]} to {M["CRSixSpeedHi"]}×), but prefill rose from '
                         f'{L["base_pre"]:.1f} to {L["op_pre"]:.0f} ms, because Opus records its CUDA graph there (A11: ',
                         'med', INK2), (Q['op11'], 'medit', INK2), ('). So a whole generation is ', 'med', INK2),
                        (f'{M["CRSixEtoEXLo"]}× faster. ', 'bold', INK),
                        ('The benchmark scored what it timed.', 'demi', INK)], 16.0, lead=1.18)


# ================================================================ footer
def footer(y):
    hline(X0, X1, y, color=RULE, lw=1.2)
    span = X1 - X0 - 16
    cws = [span * 0.315, span * 0.295, span * 0.39]
    cols = [
        [('Limits. ', 'bold', INK),
         ('Three seeds. Each compared search ran once, under different harnesses: no ranking of models. The 32× '
          'compares engines as well as models (ten recipes use the protected reference decoder) and measures speed, not '
          'quality. Correctness: greedy, batch one, this workload.', 'med', INK2)],
        [('Records. ', 'bold', INK),
         ('Every number is from the paper’s frozen records; nothing was re-run for this poster. Recipes are compared '
          'in the 13-recipe test; E1 adds two token budgets; search-record speed-ups show each search’s path. '
          'Pre-registrations, amendments and incident logs are in the pending release.', 'med', INK2)],
        [('References and AI use. ', 'bold', INK),
         ('A. Karpathy, autoresearch (2026), github.com/karpathy/autoresearch. Research agents: GPT-5.6, Claude Opus '
          '5.5. Claude (Opus 5.5, Claude Code; same family as the Opus agent, no role in its proposals) wrote and ran the '
          'camera-ready experiments under approved plans, the analysis, draft text and this poster. Authorship is human.',
          'med', INK2)]]
    xx = X0
    for cw, runs in zip(cws, cols):
        para(xx, y + 3.5, cw, runs, 14.0, lead=1.15)
        xx += cw + 8


def build():
    header()
    yt = takeaway(128)
    y1 = sec_budget(yt + 8)
    y2 = sec_pareto(y1 + 9)
    ytop = y2 + 7
    hline(X0, X1, ytop - 3.5, color=RULE, lw=1.0)
    lw_ = 258
    yfoot = H - MARGIN - 30
    sec_agents(X0, ytop, lw_, yfoot - 5)
    ye = sec_engine(X0 + lw_ + 14, ytop, X1 - (X0 + lw_ + 14))
    print(f'takeaway ends {yt:.1f}; section 1 ends {y1:.1f}; section 2 ends {y2:.1f}; section 4 ends {ye:.1f}; '
          f'footer at {yfoot:.1f}')
    footer(yfoot)
    OUT.mkdir(exist_ok=True)
    meta = {'Title': 'AutoPareto: NeurIPS 2026 AutoMLR poster (A1)', 'Author': 'Rami Alobaidy',
            'Subject': 'Auditing Fixed-Time Autonomous Transformer Search with Matched-Exposure Verification'}
    fig.savefig(OUT / 'autopareto_poster_A1.pdf', metadata=meta)
    fig.savefig(OUT / 'autopareto_poster_A1.svg')
    fig.savefig(OUT / 'autopareto_poster_preview.png', dpi=110)
    fig.savefig(OUT / 'autopareto_poster_preview_small.png', dpi=36)
    print('wrote', OUT)


if __name__ == '__main__':
    build()
