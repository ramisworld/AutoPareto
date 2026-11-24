#!/usr/bin/env python3
"""AutoPareto poster, version 2: 61 x 85 cm portrait (the workshop's maximum), far less text than version 1.

    /path/to/paper/.venv/bin/python poster/build_poster_v2.py

Writes poster/out_v2/autopareto_poster_v2_61x85cm.{pdf,svg} and two PNG previews. Version 1 (A1, build_poster.py,
poster/out/) is unchanged. Same data and checks: every number comes from poster_data.load(), which reads the paper's
frozen records and asserts each printed value against the paper. Same fonts (Avenir Next, extracted at build time).
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

OUT = HERE / 'out_v2'
W, H = 610.0, 850.0          # workshop maximum: 61 x 85 cm portrait
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
T = dict(title=140, sub=31, hero=112, take=29, h2=44, lede=26, body=23, fig=22, tick=20, note=18)


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
    r = 7.2
    cv.add_patch(Circle((x + r, y + r), r, fc=INK, ec='none', zorder=3))
    text(x + r, y + r + 0.15, str(num), 26, 'heavy', 'white', ha='center', va='center', zorder=4)
    text(x + 2 * r + 4.5, y + r + 5.4, title, T['h2'], 'bold', INK)
    yy = y + 2 * r + 4.5
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
GAP = 18.0
COLW = (X1 - X0 - GAP) / 2
XR = X0 + COLW + GAP
GROUP_LABEL = [('baseline', 'Baseline'), ('autoresearch', 'AutoResearch (quality only)'),
               ('classical', 'Classical search'), ('gpt', 'AutoPareto, GPT-5.6'), ('opus', 'AutoPareto, Claude Opus 5.5')]


# ================================================================ header and takeaway
def header():
    box(X0, 18, X1 - X0, 108, DARK, r=5)
    text(X0 + 15, 34, 'NeurIPS 2026  ·  Workshop for Autonomous Machine Learning Research (AutoMLR)  ·  Sydney', 20,
         'demi', ON_DARK2)
    text(X0 + 13, 81, 'AutoPareto', T['title'], 'heavy', 'white')
    text(X0 + 15, 96.5, 'Auditing Fixed-Time Autonomous Transformer Search', T['sub'], 'med', ON_DARK)
    text(X0 + 15, 108.5, 'with Matched-Exposure Verification', T['sub'], 'med', ON_DARK)
    text(X0 + 15, 120.0, 'Rami Alobaidy   ·   rameonix@gmail.com', 20, 'med', ON_DARK2)
    xr = X1 - 14
    for num, col, l1, l2, yb in (('32×', HERO_V, 'faster decoding', 'decode steps only (scored)', 64),
                                 ('4.6×', 'white', 'faster generation', 'prefill + 128 new tokens', 114)):
        text(xr, yb, num, T['hero'], 'heavy', col, ha='right')
        nx = xr - tw(num, T['hero'], 'heavy') - 6
        text(nx, yb - 14.5, l1, 23, 'bold', col, ha='right')
        text(nx, yb - 5.5, l2, 18.5, 'med', ON_DARK2, ha='right')
    hline(X1 - 215, X1 - 14, 76, color='#2a3140', lw=1.2, z=2)


def takeaway(y):
    rect(X0, y, 3.4, 21.5, COL['opus'], z=2)
    return para(X0 + 10, y + 0.3, X1 - X0 - 10, [
        ('A 300 s training budget rewards speed, so the agents shrank batches and models. ', 'bold', INK),
        ('Claude Opus 5.5 also rewrote the inference engine: its recipes decode 32× faster than the starting model, yet '
         'a whole generation is only 4.6× faster.', 'med', INK)], T['take'], lead=1.18)


# ================================================================ 1: the budget (E1)
def sec_budget(x, y0, w):
    y = section(x, y0, 1, 'The budget shapes the scientist', w,
                [('In 300 s, GPT A22’s smaller model sees 1.8× more data, but learns less from each token.', 'med', INK2)])
    text(x, y + 9, 'Held-out BPB (lower is better), seeds 6 to 8', T['fig'], 'bold', INK)
    ax = ax_at(x + 21, y + 13, w - 36, 150)
    E = P['e1']
    for key, col in (('arref', COL['autoresearch']), ('gpt_a22', COL['gpt'])):
        bx = [p['tok'] / 1e6 for p in E[key]['budget']]
        by = [p['bpb'] for p in E[key]['budget']]
        ax.plot(bx, by, color=col, lw=4.0, zorder=4, solid_capstyle='round')
        ax.scatter(bx, by, s=150, color=col, edgecolor='white', lw=1.8, zorder=5)
        t3 = E[key]['t300']
        ax.scatter([t3['tok'] / 1e6], [t3['bpb']], s=460, marker='D', color='white', edgecolor=col, lw=3.6, zorder=6)
    a3, g3 = E['arref']['t300'], E['gpt_a22']['t300']
    ax.plot([a3['tok'] / 1e6, g3['tok'] / 1e6], [a3['bpb'], g3['bpb']], color=INK2, lw=2.0, ls=(0, (2, 2)), zorder=2)
    ax.set_xlim(72, 216)
    ax.set_ylim(1.205, 1.085)
    ax.set_xticks([88.6, 157.3, 201.3])
    ax.set_xticklabels(['88.6M', '157.3M', '201.3M'])
    ax.set_yticks([1.10, 1.13, 1.16, 1.19])
    ax.set_yticklabels(['1.10', '1.13', '1.16', '1.19'])
    ax.grid(axis='y', color='#eceef1', lw=1.0, zorder=0)
    style_ax(ax)
    fs = 19
    ax.annotate('AR-ref', (201.3, E['arref']['budget'][2]['bpb']), xytext=(-16, 10), textcoords='offset points',
                ha='right', va='bottom', fontproperties=prop('bold', T['fig'] + 1), color=COL['autoresearch'])
    ax.text(127, 1.1895, 'GPT A22', ha='left', va='center', fontproperties=prop('bold', T['fig'] + 1), color=COL['gpt'])
    ya, yg = E['arref']['budget'][2]['bpb'], E['gpt_a22']['budget'][2]['bpb']
    ax.annotate('', (209, ya), (209, yg), arrowprops=dict(arrowstyle='<->', color=INK, lw=1.8, shrinkA=1, shrinkB=1))
    ax.text(207, (ya + yg) / 2, f'+{P["e1_gaps"][2]:.3f}\nat equal\ntokens', fontproperties=prop('bold', fs - 1),
            color=INK, va='center', ha='right', linespacing=1.0)
    ax.text(g3['tok'] / 1e6 + 5.5, g3['bpb'] + 0.0036, 'not separated\nat 300 s', fontproperties=prop('bold', fs),
            color=INK, ha='left', va='top', linespacing=1.05)
    ax.text(a3['tok'] / 1e6, -0.17, f'AR-ref in 300 s:\n{M["CREOneTokAR"]}M', transform=ax.get_xaxis_transform(),
            ha='left', va='top', fontproperties=prop('demi', 17.5), color=COL['autoresearch'], linespacing=1.05)
    ax.text(g3['tok'] / 1e6 - 4, -0.17, f'GPT A22 in 300 s:\n{M["CREOneTokAP"]}M', transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontproperties=prop('demi', 17.5), color=COL['gpt'], linespacing=1.05)
    # legend: diamond = 300 s run, dot = fixed token budget
    ax.scatter([0.05], [0.90], s=260, marker='D', color='white', edgecolor=INK2, lw=2.6, transform=ax.transAxes,
               clip_on=False, zorder=7)
    ax.text(0.085, 0.90, '300 s run', transform=ax.transAxes, fontproperties=prop('med', fs - 1), color=INK2,
            va='center')
    ax.scatter([0.05], [0.79], s=100, color=INK2, transform=ax.transAxes, clip_on=False, zorder=7)
    ax.text(0.085, 0.79, 'fixed token budget', transform=ax.transAxes, fontproperties=prop('med', fs - 1),
            color=INK2, va='center')


# ================================================================ 2: two agents
def sec_agents(x, y0, w, y_end):
    y = section(x, y0, 2, 'Two agents, two discoveries', w,
                [('Both shrank the model first. ', 'med', INK2), ('Only Opus then rebuilt the decoder.', 'demi', INK)])
    gx, ox = x + 5, x + 108
    gw, ow = ox - gx - 10, x + w - ox - 4
    yt = y + 8
    text(gx - 3, yt, 'GPT-5.6', 23, 'bold', COL['gpt'])
    text(ox - 3, yt, 'Claude Opus 5.5', 23, 'bold', COL['opus'])
    ytop, ybot = yt + 7, y_end - 34
    cv.plot([gx, gx], [ytop, ybot], color=COL['gpt'], lw=3.4, zorder=2, solid_capstyle='round')
    cv.plot([ox, ox], [ytop, ybot], color=COL['opus'], lw=3.4, zorder=2, solid_capstyle='round')

    def node(xr, yv, col, big=False):
        r = 3.0 if big else 2.3
        cv.add_patch(Circle((xr, yv), r, fc=col, ec='white', lw=1.4, zorder=4))

    # GPT-5.6: four steps, spread over the same height
    g_items = [('A1 batch ¼', None), ('A4 to A8 fewer layers', '8 to 4'), ('A18 to A22 width, learning rates', None),
               ('GPT A22', f'4 layers, {P["params_M"]["gpt_a22"]:.1f}M')]
    span = ybot - ytop - 44
    for i, (lab, sub) in enumerate(g_items):
        yv = ytop + 8 + i * span / (len(g_items) - 1)
        node(gx, yv, COL['gpt'], big=i == len(g_items) - 1)
        yb_ = para(gx + 5.5, yv - 3.9, gw, [(lab, 'bold', INK)], 20, lead=1.08)
        if sub:
            yb_ = yb_ + 5.0
            text(gx + 5.5, yb_, sub, 18, 'med', INK2)
    para(gx + 5.5, yb_ + 7, gw, [('No inference code in any campaign.', 'demi', INK2)], 17.5, lead=1.1)
    # Opus: speed-up at each step (search records, log scale)
    steps = [(4, 'A4 four layers'), (10, 'A10 CUDA graph replay'), (14, 'A14 kernel fusion'),
             (18, 'A18 ring K/V cache'), (19, 'A19 custom attention'), (22, 'A22 fused decode step')]
    pitch = (ybot - ytop - 6) / len(steps)
    bar_w = ow - 26
    lmax = math.log(36)
    for i, (a, lab) in enumerate(steps):
        yv = ytop + 5 + i * pitch
        node(ox, yv, COL['opus'], big=a in (10, 14, 22))
        text(ox + 5.5, yv + 2.6, lab, 20, 'bold', INK)
        sx = P['opus_path'][a]
        unc = a in P['opus_uncertain']
        L = bar_w * math.log(sx) / lmax
        rect(ox + 5.5, yv + 5.2, L, 4.6, 'white' if unc else COL['opus'], ec=COL['opus'], lw=1.4 if unc else 0, z=3,
             hatch='////' if unc else None)
        text(ox + 5.5 + L + 2, yv + 9.6, f'{sx:.1f}×', 19, 'bold', COL['opus'])
    # re-timed on the common benchmark
    yr = ybot + 5
    rect(x - 2, yr, w + 2, 18, OPUS_LIGHT, z=0)
    text(x + 2, yr + 6.2, 'Re-timed on the common benchmark', 16.5, 'bold', MUTED)
    text(gx - 3, yr + 14.8, f'GPT A22  {M["CRNineGptDecX"]}×', 22, 'heavy', COL['gpt'])
    text(ox - 3, yr + 14.8, f'Opus  {M["CRSixSpeedLo"]} to {M["CRSixSpeedHi"]}×', 22, 'heavy', COL['opus'])
    para(x, yr + 21.5, w, [('Bars: search-record speed-ups (hatched: speed-uncertain). Each search shown ran once, '
                            'under different harnesses: not a ranking of models.', 'med', MUTED)], 16.5, lead=1.12)


# ================================================================ 3: thirteen recipes
def pareto_panel(x, y, w, h, key, title, ylim, ylabel):
    text(x, y + 6.5, title, T['fig'], 'bold', INK)
    ax = ax_at(x, y + 10, w, h - 10)
    lo, hi = ylim
    pts = []
    for r in P['recipes']:
        q, v, g = r[key], r['dec'], r['group']
        pts.append((r['key'], q, v))
        yy = min(q, hi - 0.002)
        ax.scatter([v], [yy], s=260 if g != 'autoresearch' else 210, marker=MARK[g], color=COL[g], edgecolor='white',
                   lw=1.4, zorder=5, clip_on=False)
    front = poster_data.pareto(pts)
    fp = sorted([p for p in pts if p[0] in front], key=lambda p: p[2])
    xs, ys = [], []
    for i, (_, q, v) in enumerate(fp):
        if i:
            xs.append(v)
            ys.append(ys[-1])
        xs.append(v)
        ys.append(q)
    ax.plot(xs, ys, color=INK2, lw=2.0, ls=(0, (3, 2)), zorder=4)
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
    lab = lambda k, txt, dx, dy, ha='left', col=INK: ax.annotate(
        txt, (R[k]['dec'], min(R[k][key], hi - 0.002)), xytext=(dx, dy), textcoords='offset points', ha=ha,
        va='center', fontproperties=prop('bold', 19), color=col, zorder=7,
        bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none', alpha=0.9))
    if key == 'tok':
        lab('arref', 'AR-ref', 16, 0)
        lab('gpt_a22', 'GPT A22', 12, -14)
    else:
        ax.annotate('AR-ref', (R['arref']['dec'], R['arref']['fix']), xytext=(118, 1.1512), textcoords='data',
                    ha='left', va='center', fontproperties=prop('bold', 19), color=INK, zorder=7,
                    arrowprops=dict(arrowstyle='-', color='#9aa0a8', lw=1.0, shrinkA=1, shrinkB=6))
        lab('gpt_a22', 'GPT A22', 14, 11)
        lab('baseline', f'Baseline ({R["baseline"]["fix"]:.3f}, off scale)', 14, 0, col=INK2)
    oq = max(R[k][key] for k in ('opus_a23', 'opus_a24', 'opus_a27'))
    ov = sum(R[k]['dec'] for k in ('opus_a23', 'opus_a24', 'opus_a27')) / 3
    ax.annotate('Opus A23,\nA24, A27', (ov, oq), xytext=(0, -20), textcoords='offset points', ha='center', va='top',
                fontproperties=prop('bold', 19), color=COL['opus'], linespacing=1.0)
    ax.annotate('', xy=(0.975, 0.97), xytext=(0.975, 0.76), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>', color=INK2, lw=1.8, mutation_scale=15))
    ax.text(0.955, 0.865, 'better', transform=ax.transAxes, fontproperties=prop('demi', 18), color=INK2, va='center',
            ha='right')
    ax.annotate('', xy=(0.985, 0.05), xytext=(0.80, 0.05), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>', color=INK2, lw=1.8, mutation_scale=15))
    ax.text(0.89, 0.10, 'faster', transform=ax.transAxes, fontproperties=prop('demi', 18), color=INK2, ha='center')


def memory_panel(x, y, w, h):
    text(x, y + 6.5, '(c) Memory', T['fig'], 'bold', INK)
    ax = ax_at(x + 33, y + 10, w - 45, h - 10)
    keys = ['arref', 'gpt_a22', 'opus_a23', 'opus_a24', 'opus_a27']
    for i, k in enumerate(keys):
        r = R[k]
        ax.barh(i, r['mem'], height=0.62, color=COL[r['group']], lw=0, zorder=3)
        ax.text(r['mem'] + 8, i, f"{r['mem']:.0f}", va='center', ha='left', fontproperties=prop('bold', 18),
                color=INK, clip_on=False)
        ax.text(-10, i, r['name'], va='center', ha='right', fontproperties=prop('med', 18), color=INK2, clip_on=False)
    ax.set_ylim(len(keys) - 0.45, -0.55)
    ax.set_xlim(0, 380)
    ax.set_xticks([0, 100, 200, 300])
    ax.set_yticks([])
    ax.grid(axis='x', color='#eceef1', lw=1.0, zorder=0)
    style_ax(ax)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('peak MiB at inference', fontproperties=prop('med', T['tick']), color=INK2, labelpad=3)


def sec_pareto(y0):
    y = section(X0, y0, 3, 'One benchmark for all 13 recipes', X1 - X0, [
        ('Equal tokens: AR-ref learns most per token. Equal time (300 s): no recipe is resolved as better than '
         'AR-ref, so decode speed decides.', 'med', INK2)])
    yl = y + 7
    xx = X0 + 4
    for g, labl in GROUP_LABEL:
        cv.scatter([xx], [yl], s=230 if g != 'autoresearch' else 180, marker=MARK[g], color=COL[g], edgecolor='white',
                   lw=1.0, zorder=5)
        text(xx + 4.5, yl + 2.5, labl, 18.5, 'demi', INK2)
        xx += tw(labl, 18.5, 'demi') + 14
    cv.plot([xx - 3, xx + 10], [yl, yl], color=INK2, lw=2.0, ls=(0, (3, 2)))
    text(xx + 12.5, yl + 2.5, 'frontier', 18.5, 'demi', INK2)
    yp = yl + 8
    hp = 128
    pareto_panel(X0 + 30, yp, 190, hp, 'tok', '(a) Equal tokens: 201.3M', (1.0905, 1.1935), True)
    pareto_panel(X0 + 242, yp, 192, hp, 'fix', '(b) Equal time: 300 s', (1.1485, 1.1935), False)
    memory_panel(X0 + 452, yp, X1 - (X0 + 452), hp)
    text(X0, yp + hp + 24, 'Means of seeds 6 to 8; 1,024 tokens in, 128 out, one A40. Per-seed points, timing flags '
                           'and correctness checks: paper, Figure 3.', 16.5, 'med', MUTED)
    return yp + hp + 26


# ================================================================ 4: 32x vs 4.6x
def lane(x, y, w, n_launch, graph, fused):
    """Schematic CPU and GPU lanes for one decoded token: counts are real, widths are not to scale."""
    text(x, y + 4.0, 'CPU', 16, 'bold', MUTED)
    text(x, y + 14.6, 'GPU', 16, 'bold', MUTED)
    lx = x + 14
    lw_ = w - 15
    cv.plot([lx, lx + lw_], [y + 5.0, y + 5.0], color=RULE, lw=1.0, zorder=1)
    cv.plot([lx, lx + lw_], [y + 15.6, y + 15.6], color=RULE, lw=1.0, zorder=1)
    if graph:
        rect(lx, y + 0.8, 10, 4.2, COL['opus'], z=3)
        text(lx + 12, y + 4.4, '1 replay call', 16, 'demi', COL['opus'])
    else:
        k = 30 if n_launch > 200 else 10
        for i in range(k):
            xx = lx + i * (lw_ - 1) / k
            cv.plot([xx, xx], [y + 0.8, y + 5.0], color=INK, lw=1.5, zorder=3, solid_capstyle='butt')
    k = 30 if not fused else 10
    span = lw_ * (0.42 if graph else 1.0)
    slot = span / k
    kw_ = slot * (0.86 if graph else 0.38)
    for i in range(k):
        xx = lx + i * slot + (0 if graph else slot * 0.5)
        c = COL['opus'] if fused else TINY
        if i % (6 if not fused else 3) == 0:
            c = GEMV
        rect(xx, y + 11.4, kw_, 4.2, c, z=3)


def sec_engine(y0):
    y = section(X0, y0, 4, 'Why 32× decoding is only 4.6× generation', X1 - X0, [
        ('Only decode was scored. At batch one it is launch-bound: hundreds of tiny GPU kernels per token, each '
         'launched by the CPU.', 'med', INK2)])
    C, A = P['census'], P['abl']
    # left: graph x fusion grid
    x, w = X0, 316
    yy = y + 4
    text(x, yy + 6, f'Starting model: {C["baseline"]["kernels"]:.0f} kernels, {C["baseline"]["launches"]:.0f} CPU '
                    'launches per token', 18.5, 'demi', INK2)
    yy += 11
    rl = 13
    cw = (w - rl - 5) / 2
    ch = 47.0
    xs = [x + rl, x + rl + cw + 5]
    text(xs[0] + cw / 2, yy + 5.5, 'no CUDA graph', 20, 'bold', INK, ha='center')
    text(xs[1] + cw / 2, yy + 5.5, 'CUDA graph replay', 20, 'bold', COL['opus'], ha='center')
    ys = [yy + 9, yy + 9 + ch + 4]
    for j, labl in enumerate(['unfused', 'fused']):
        text(x + 4.5, ys[j] + ch / 2, labl, 20, 'bold', INK if j == 0 else COL['opus'], ha='center', va='center',
             rotation=90)
    cells = {(0, 0): ('a23-V4_eager_unfused', 'V4_eager_unfused'), (0, 1): ('a23-V2_no_fusion', 'V2_no_fusion'),
             (1, 0): ('a23-V1_no_cuda_graph', 'V1_no_cuda_graph'), (1, 1): ('a23-V0_as_submitted', 'V0_as_submitted')}
    for (j, i), (ck, ak) in cells.items():
        cx, cy = xs[i], ys[j]
        best = (j, i) == (1, 1)
        box(cx, cy, cw, ch, OPUS_LIGHT if best else TINT, ec=COL['opus'] if best else 'none', lw=2.2 if best else 0,
            r=2.5)
        text(cx + cw - 4, cy + 12.5, f'{A[ak]:.1f}×', 36, 'heavy', COL['opus'] if best else INK, ha='right')
        lane(cx + 4, cy + 21.5, cw - 8, C[ck]['launches'], graph=i == 1, fused=j == 1)
        text(cx + 4, cy + 10.5, f"{C[ck]['launches']:.0f} launches · {C[ck]['kernels']:.0f} kernels", 17.5, 'med',
             INK2)
    yb = ys[1] + ch + 3
    para(x, yb, w, [('Graph replay removes CPU launches; fusion removes GPU kernels. ', 'demi', INK),
                    ('Speed-ups: ablations of Opus A23, A24, A27 (medians).', 'med', MUTED)], 16.5, lead=1.12)
    # right: whole generation, to scale
    xr0 = X0 + w + 22
    wr = X1 - xr0
    L = P['lat']
    text(xr0, yy + 6, 'One whole generation (13-recipe test)', 21, 'bold', INK)
    bx0 = xr0 + 30
    scale = (wr - 30 - 34) / 900.0
    rows = [('Baseline', L['base_pre'], L['base_dec'], COL['baseline'], yy + 14),
            ('Opus', L['op_pre'], L['op_dec'], COL['opus'], yy + 31)]
    for name, pre, dec, col, ry in rows:
        text(xr0, ry + 8.0, name, 21, 'bold', INK)
        rect(bx0, ry, pre * scale, 11.5, PREFILL, ec='#9b968c', lw=0.9, z=3, hatch='///')
        rect(bx0 + pre * scale, ry, dec * scale, 11.5, col, z=3)
        text(bx0 + (pre + dec) * scale + 2.5, ry + 8.2, f'{pre + dec:.0f} ms', 23, 'heavy', INK)
    text(bx0 + L['base_pre'] * scale + L['base_dec'] * scale / 2, yy + 14 + 8.2, f'decode {L["base_dec"]:.0f} ms', 19,
         'demi', 'white', ha='center')
    text(bx0 + L['op_pre'] * scale / 2, yy + 31 + 7.9, f'prefill {L["op_pre"]:.0f}', 16.5, 'demi', INK, ha='center')
    text(bx0 + (L['op_pre'] + L['op_dec']) * scale + 37, yy + 31 + 8.2, f'decode {L["op_dec"]:.0f}', 19, 'demi',
         COL['opus'])
    yk = yy + 47
    rect(bx0, yk, 7, 5, PREFILL, ec='#9b968c', lw=0.9, z=3, hatch='///')
    text(bx0 + 9, yk + 4.3, 'prefill: not scored', 17.5, 'med', INK2)
    rect(bx0 + 74, yk, 7, 5, COL['opus'], z=3)
    text(bx0 + 83, yk + 4.3, 'decode: scored', 17.5, 'med', INK2)
    # the two ratios, side by side
    yq = yk + 12
    for i, (num, l1, l2) in enumerate((('32×', 'decode steps', f'{L["base_dec"]:.0f} to {L["op_dec"]:.0f} ms'),
                                       ('4.6×', 'whole generation', f'{L["base_tot"]:.0f} to {L["op_tot"]:.0f} ms'))):
        bxl = xr0 + i * (wr / 2)
        box(bxl, yq, wr / 2 - 5, 33, OPUS_LIGHT if i == 0 else TINT, r=3)
        text(bxl + 5, yq + 21, num, 50, 'heavy', COL['opus'] if i == 0 else INK)
        text(bxl + 5 + tw(num, 50, 'heavy') + 4, yq + 13, l1, 19, 'bold', INK)
        text(bxl + 5 + tw(num, 50, 'heavy') + 4, yq + 21.5, l2, 18, 'med', INK2)
    para(xr0, yq + 37, wr, [(f'Opus records its CUDA graph in the unscored prefill ({L["base_pre"]:.0f} to '
                             f'{L["op_pre"]:.0f} ms). ', 'med', INK2),
                            ('The benchmark scored what it timed.', 'demi', INK)], 18.5, lead=1.15)
    return max(yb + 12, yq + 52)


def footer(y):
    hline(X0, X1, y, color=RULE, lw=1.2)
    para(X0, y + 3.5, X1 - X0, [
        ('Limits: ', 'bold', INK),
        ('three seeds; each compared search ran once, under different harnesses; the 32× compares engines as well as '
         'models and measures speed, not quality. ', 'med', INK2),
        ('Reference: ', 'bold', INK), ('A. Karpathy, autoresearch (2026). ', 'med', INK2),
        ('AI use: ', 'bold', INK),
        ('research agents GPT-5.6 and Claude Opus 5.5. Claude (Opus 5.5, Claude Code; same family as the Opus agent) '
         'wrote and ran the camera-ready experiments under approved plans, the analysis, draft text and this poster. '
         'Authorship is human.', 'med', INK2)], 15.5, lead=1.15)


def build():
    header()
    yt = takeaway(133)
    yr1 = yt + 9
    sec_budget(X0, yr1, COLW)
    y_end1 = yr1 + 236
    sec_agents(XR, yr1, COLW, y_end1)
    hline(X0, X1, y_end1 + 9, color=RULE, lw=1.0)
    y2 = sec_pareto(y_end1 + 18)
    hline(X0, X1, y2 + 8, color=RULE, lw=1.0)
    y3 = sec_engine(y2 + 17)
    yfoot = H - MARGIN - 17
    print(f'takeaway ends {yt:.1f}; row 1 ends {y_end1:.1f}; section 3 ends {y2:.1f}; section 4 ends {y3:.1f}; '
          f'footer at {yfoot:.1f}')
    footer(yfoot)
    OUT.mkdir(exist_ok=True)
    meta = {'Title': 'AutoPareto: NeurIPS 2026 AutoMLR poster (61 x 85 cm, version 2)', 'Author': 'Rami Alobaidy',
            'Subject': 'Auditing Fixed-Time Autonomous Transformer Search with Matched-Exposure Verification'}
    fig.savefig(OUT / 'autopareto_poster_v2_61x85cm.pdf', metadata=meta)
    fig.savefig(OUT / 'autopareto_poster_v2_61x85cm.svg')
    fig.savefig(OUT / 'autopareto_poster_v2_preview.png', dpi=110)
    fig.savefig(OUT / 'autopareto_poster_v2_preview_small.png', dpi=36)
    print('wrote', OUT)


if __name__ == '__main__':
    build()
