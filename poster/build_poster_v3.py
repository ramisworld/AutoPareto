#!/usr/bin/env python3
"""AutoPareto poster, version 3: landscape academic layout (85 x 61 cm), Opus-focused, with math.

    /path/to/paper/.venv/bin/python poster/build_poster_v3.py

Writes poster/out_v3/autopareto_poster_v3_landscape.{pdf,svg} and two PNG previews. Versions 1 and 2 are unchanged.
Layout follows the conventions of a conference poster: header and footer bands, three columns with centred section
titles, justified text, display mathematics (Computer Modern via matplotlib mathtext), numbered figures and tables.
Every number comes from poster_data.load(), which reads the paper's frozen records and asserts each value against the
paper; derived quantities (the Amdahl bound, the interaction ratio) are computed here from those values and labelled.
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

OUT = HERE / 'out_v3'
W, H = 850.0, 610.0                 # landscape, mm
PT = 72 / 25.4
MX, GUT = 22.0, 20.0
COLW = (W - 2 * MX - 2 * GUT) / 3
COLX = [MX + i * (COLW + GUT) for i in range(3)]
BODY_TOP, BODY_BOT = 93.0, 581.0

NAVY, INK, INK2, MUTED, RULE = '#1c2541', '#16181d', '#454b57', '#7d8491', '#c9cdd3'
LAV, LAV_EDGE = '#eeecfa', '#c3bcf0'
COL = {'baseline': '#8a8985', 'autoresearch': '#eb6834', 'classical': '#1baf7a', 'gpt': '#2a78d6', 'opus': '#6250d6'}
MARK = {'baseline': 's', 'autoresearch': 'D', 'classical': 'o', 'gpt': '^', 'opus': 'v'}
GEMV, TINY, PREFILL = '#3d4f6b', '#aab2c0', '#d8d4cc'
VIOLET = COL['opus']
BODY, CAP, TAB, TICK, FLAB = 17.5, 14.5, 15.5, 13.5, 14.5


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
mx = lambda v: f'{v:.1f}' + r'$\times$'   # a speed-up with a math times sign


# ================================================================ header and footer
def header():
    rect(0, 0, W, 80, NAVY, z=0)
    text(W / 2, 27.5, 'AutoPareto: Auditing Fixed-Time Autonomous Transformer Search', 44, 'bold', 'white',
         ha='center')
    text(W / 2, 45.0, 'with Matched-Exposure Verification', 44, 'bold', 'white', ha='center')
    text(W / 2, 60.5, 'Rami Alobaidy', 23, 'med', '#e6e9ef', ha='center')
    text(W / 2, 72.0, 'NeurIPS 2026  ·  Workshop for Autonomous Machine Learning Research (AutoMLR)  ·  Sydney', 17,
         'reg', '#b7bfcc', ha='center')


def footer():
    rect(0, H - 21, W, 21, NAVY, z=0)
    yb = H - 9.2
    text(MX, yb, 'rameonix@gmail.com', 15, 'reg', '#dfe3ea')
    text(W / 2, yb, 'NeurIPS 2026  ·  AutoMLR Workshop', 15, 'reg', '#dfe3ea', ha='center')
    text(W - MX, yb, 'Records and code: release pending', 15, 'reg', '#dfe3ea', ha='right')


# ================================================================ column 1: setting and method
def column1(x, w):
    y = section_title(x, BODY_TOP, w, 'Autonomous research under a training contract')
    y = callout(x, y + 1.5, w, [('An LLM agent rewrites a Transformer’s training program, trains each version for '
                                 '300 s on one GPU, and is scored on ', 'reg', INK),
                                ('quality, decode speed and memory together', 'demi', INK),
                                ('. What does it discover, and does the discovery survive a fair test?', 'reg', INK)])
    y = para(x, y + 4.5, w, [('Each attempt is a program ', 'reg', INK), ('$P$', 'math', INK),
                           (' (the agent’s train.py, as in autoresearch [1]). A protected harness trains it for ',
                            'reg', INK),
                           ('$T=300\\,\\mathrm{s}$', 'math', INK),
                           (' on one NVIDIA A40 and returns three objectives:', 'reg', INK)])
    y = display(x + w / 2, y + 4, r'$f(P)=\left(\mathrm{BPB}(P),\;-v(P),\;m(P)\right)$', 21)
    y = display(x + w / 2, y + 5, r'$\mathrm{BPB}(P)=\frac{1}{B}\sum_{t=1}^{n}-\log_2\,p_{\theta_P}(x_t\mid x_{<t})$',
                21)
    y = para(x, y + 4, w, [('Quality is bits per byte over ', 'reg', INK), ('$B$', 'math', INK),
                           (' unseen bytes (validation text in search, a shard no agent saw in verification); ',
                            'reg', INK), ('$v$', 'math', INK),
                           (' is batch-1 decode throughput (1,024 tokens in, 128 out in Opus’s search and the final '
                            'test); ', 'reg', INK),
                           ('$m$', 'math', INK), (' is peak inference memory. Attempt ', 'reg', INK),
                           ('$P$', 'math', INK), (' dominates ', 'reg', INK), ('$Q$', 'math', INK),
                           (' when it is no worse on every objective and better on one:', 'reg', INK)])
    y = display(x + w / 2, y + 4, r'$P\succ Q\;\Leftrightarrow\;\forall i:\,f_i(P)\leq f_i(Q)\;\;\wedge\;\;'
                                  r'\exists j:\,f_j(P)<f_j(Q)$', 21)
    y = para(x, y + 4, w, [('Each run starts from a recorded hypothesis; failures stay in the archive, whose '
                            'non-dominated set is the Pareto frontier.', 'reg', INK)])
    y = loop_figure(x, y + 6, w)
    y = caption(x, y + 2.5, w, 'Figure 1.', 'The AutoPareto loop. The agent can edit only train.py, including an '
                                            'optional generation hook that the protected benchmark times.')

    y = section_title(x, y + 5.5, w, 'Matched-exposure verification')
    y = para(x, y + 1, w, [('A time budget rewards fast programs with more data. A pre-registered test therefore '
                            'retrains every final recipe on fresh seeds 6 to 8 under two contracts, with model and '
                            'optimizer code unchanged,', 'reg', INK)])
    y = display(x + w / 2, y + 4, r'$\theta_T(P)=\mathrm{Train}(P\mid t\leq 300\,\mathrm{s}),\qquad'
                                  r'\theta_N(P)=\mathrm{Train}(P\mid N=201.3\mathrm{M\ tokens}),$', 19)
    y = para(x, y + 4, w, [('and times all 13 recipes with one bracketed protocol:', 'reg', INK)])
    y = display(x + w / 2, y + 4, r'$\hat{s}(P)=\mathrm{median}_{\,i=1,\ldots,8}\;\frac{r_i(P)}{\bar{r}_i(\mathrm{base})}$',
                21)
    y = para(x, y + 4, w, [('$r_i$', 'math', INK), (' is the candidate’s decode rate in window ', 'reg', INK),
                           ('$i$', 'math', INK), (', ', 'reg', INK), (r'$\bar{r}_i$', 'math', INK),
                           (' the baseline rate interpolated from the windows just before and after it, so drift '
                            'common to both cancels.', 'reg', INK)])
    y = section_title(x, y + 5.5, w, 'Two agents, one generation hook')
    G = R
    rows = [['GPT-5.6', '75 + 22', '0', 'GPT A24, A25', mx(G['gpt_a24']['speedup'])],
            [('Claude Opus 5.5',), '27 (25 valid)', ('19 of 25',), 'Opus A23, A24, A27',
             (f"{M['CRSixSpeedLo']}–{M['CRSixSpeedHi']}" + r'$\times$',)]]
    y = table(x, y + 1, w, [('Agent', 0.25, 'l'), ('Attempts', 0.17, 'r'), ('Own inference\ncode', 0.19, 'r'),
                             ('Fastest recipe,\nre-timed', 0.22, 'r'), ('Decode\nspeed-up', 0.17, 'r')], rows, shade=(1,))
    y = caption(x, y + 2.5, w, 'Table 1.', 'Both agents could write generation code through the same documented '
                                           'hook. Each search ran once, under different harnesses: an observed '
                                           'difference in strategy, not a ranking of models (cf. [2]).')
    y = findings(x, y + 5, w)
    return y


def findings(x, y, w):
    items = [('Budget. ', 'Under 300 s every agent found the same lever, smaller batches and models; none learns '
                          'more per token than AR-ref.'),
             ('Discovery. ', 'Claude Opus 5.5 diagnosed launch-bound decoding and built its own inference engine: '
                             f'{M["CRSixSpeedLo"]} to {M["CRSixSpeedHi"]}× decode speed in {M["CRSixOpMemLo"]} to '
                             f'{M["CRSixOpMemHi"]} MiB, with the same greedy tokens up to BF16 near-ties.'),
             ('Verification. ', 'At equal tokens AR-ref learns most per token; at 300 s no recipe is resolved as '
                                'better than AR-ref, so decode speed decides.'),
             ('Contract. ', f'The scored 32× is {M["CRSixEtoEXLo"]}× for a whole generation.')]
    pad = 5.0
    yy = y + pad
    text(x + pad, yy + mm(18) * 0.76, 'Key findings', 18, 'demi', INK)
    yy += mm(18) + 2.5
    bottoms = []
    for k, (h, b) in enumerate(items):
        text(x + pad, yy + mm(16) * 0.76, f'{k + 1}.', 16, 'demi', VIOLET)
        yy = para(x + pad + 7, yy, w - 2 * pad - 7, [(h, 'demi', INK), (b, 'reg', INK)], 16, lead=1.25,
                  justify=False, draw=False) + 2.2
    box(x, y, w, yy - y + pad - 2.2, LAV, ec=LAV_EDGE, lw=1.0, r=2.0)
    yy = y + pad + mm(18) + 2.5
    for k, (h, b) in enumerate(items):
        yy = para(x + pad + 7, yy, w - 2 * pad - 7, [(h, 'demi', INK), (b, 'reg', INK)], 16, lead=1.25,
                  justify=False) + 2.2
    return yy + pad - 2.2


def loop_figure(x, y, w):
    bh, gap = 22.0, 9.0
    bw = (w - 3 * gap) / 4
    items = [('Agent', 'Claude Opus 5.5\nor GPT-5.6'), ('train.py', 'edit + hypothesis'),
             ('Harness', '300 s on one A40'), ('Archive', r'$f(P)$, Pareto frontier')]
    for i, (t1, t2) in enumerate(items):
        bx = x + i * (bw + gap)
        box(bx, y, bw, bh, LAV if i == 0 else '#f3f4f6', ec=LAV_EDGE if i == 0 else RULE, lw=1.0, r=2)
        text(bx + bw / 2, y + 8.2, t1, 16.5, 'demi', INK, ha='center')
        for k, ln in enumerate(t2.split('\n')):
            text(bx + bw / 2, y + 14.2 + k * 5.0, ln, 13.5, 'reg', INK2, ha='center')
        if i < 3:
            arrow(bx + bw + 1.2, y + bh / 2, bx + bw + gap - 1.2, y + bh / 2, color='#8f96a3', lw=1.4, ms=11)
    # return path: archive back to the agent
    cv.plot([x + 3 * (bw + gap) + bw / 2, x + 3 * (bw + gap) + bw / 2], [y + bh + 1, y + bh + 6], color='#8f96a3',
            lw=1.4)
    cv.plot([x + 3 * (bw + gap) + bw / 2, x + bw / 2], [y + bh + 6, y + bh + 6], color='#8f96a3', lw=1.4)
    arrow(x + bw / 2, y + bh + 6, x + bw / 2, y + bh + 1.0, color='#8f96a3', lw=1.4, ms=11)
    text(x + w / 2, y + bh + 5.0, 'reads every earlier attempt, picks a parent', 13.5, 'it', INK2, ha='center')
    return y + bh + 8


# ================================================================ column 2: what Opus discovered
def column2(x, w):
    y = section_title(x, BODY_TOP, w, 'What Claude Opus 5.5 discovered')
    y = trajectory(x, y + 1, w, 206)
    y = caption(x, y + 2, w, 'Figure 2.', 'The 25 valid attempts of the Opus search (A2 and A9 failed; its own '
                'records: 300 s training, 1,024/128 decode benchmark). (a) After A7, validation BPB improves by only '
                f'{M["CRThreeLateGain"]} over {M["CRThreeLateValid"]} valid attempts. (b) The agent '
                'then rebuilds the decoder; the scored decode speed-up climbs from 2.5× to 32×, while one whole '
                'generation stays near 5× once graph capture runs in the unscored prefill (A10 onwards). '
                '(c) Peak inference memory falls below 100 MiB. Open markers: speed-uncertain records.')
    y = callout(x, y + 5, w, [('A10 hypothesis: ', 'demi', INK), ('“Batch-1 cached decode in attempt 8 is '
                                                                   'launch-bound.”', 'it', INK)], size=17.5)
    y = section_title(x, y + 8, w, 'Anatomy of a decode step')
    y = para(x, y + 1, w, [('At batch 1 a decode step is hundreds of tiny GPU kernels, each launched by the CPU. ',
                            'reg', INK), ('CUDA-graph replay', 'demi', INK),
                           (' records one complete step and replays it with a single call; ', 'reg', INK),
                           ('kernel fusion', 'demi', INK),
                           (' (TorchInductor, Triton) merges the pointwise work between matrix–vector products. '
                            'Removing either from the final recipes shows how the 32× is built:', 'reg', INK)])
    y = timeline(x, y + 5, w)
    y = caption(x, y + 2, w, 'Figure 3.', 'One decoded token, to scale: bar length is time per token relative to '
                                          'the starting model (1/decode speed-up; ablation medians of Opus A23, A24, '
                                          'A27, each row one configuration of the final engine). Counts: profiler '
                                          'census of A23; the drawn ticks and the gap-free GPU band are schematic. '
                                          'Removing split-key decode attention costs '
                                          f'{P["abl_ranges"]["attn"]}×, the ring K/V cache {P["abl_ranges"]["ring"]}×.')
    k0, k1 = P['kappa']
    y = para(x, y + 5, w, [('The two mechanisms are complementary. With gains ', 'reg', INK), ('$S$', 'math', INK),
                           (' measured over the engine with neither (derived from the ablations),', 'reg', INK)])
    y = display(x + w / 2, y + 3, r'$\kappa=\frac{S_{G+F}}{S_G\,S_F}\in[%.2f,\;%.2f]$' % (k0, k1), 21)
    f_alone, f_graph = P['fusion_alone'], P['fusion_on_graph']
    y = para(x, y + 3, w, [('across the three recipes, where ', 'reg', INK), (r'$\kappa=1$', 'math', INK),
                           (f' would mean independent effects. Fusion alone gains {f_alone:.2f}×; on top of graph '
                            f'replay, {f_graph:.2f}× (medians over the three recipes).', 'reg', INK)])
    return y


def trajectory(x, y, w, h):
    rows = P['opus_rows']
    a = [r['a'] for r in rows]
    unc = lambda r: 'uncertain' in r['status']
    ax_h = [h * 0.27, h * 0.39, h * 0.21]
    gaps = [4.0, 4.0]
    lx, aw = x + 19, w - 21
    yy = y
    axs = []
    for i, hh in enumerate(ax_h):
        axs.append(ax_at(lx, yy, aw, hh))
        yy += hh + (gaps[i] if i < 2 else 0)
    phases = [(0.5, 7.5, 'shrink model and batch'), (7.5, 22.5, 'rebuild the decoder'), (22.5, 27.5, 'frontier')]
    for ax in axs:
        for lo, hi, _ in phases:
            ax.axvspan(lo, hi, color='#f1effc' if lo == 7.5 else '#f6f7f8', zorder=0, lw=0)
        ax.set_xlim(0.5, 27.5)
    for lo, hi, lab in phases:
        axs[0].text((lo + hi) / 2, 1.03, lab, transform=axs[0].get_xaxis_transform(), ha='center', va='bottom',
                    fontproperties=prop('demi', 13.5), color=INK2)
    # (a) validation BPB (lower is better: axis inverted so that improvement points up)
    ax = axs[0]
    ax.plot(a, [r['bpb'] for r in rows], color=VIOLET, lw=1.2, alpha=0.45, zorder=2)
    ax.scatter(a, [r['bpb'] for r in rows], s=34, color=VIOLET, edgecolor='white', lw=0.8, zorder=3)
    ax.axhline(P['opus_start']['bpb'], color=MUTED, lw=1.0, ls=(0, (3, 2)), zorder=1)
    ax.text(27.3, P['opus_start']['bpb'], 'starting model', ha='right', va='bottom', fontproperties=prop('reg', 13),
            color=MUTED)
    ax.set_ylim(1.205, 1.090)
    ax.set_yticks([1.10, 1.15, 1.20])
    ax.set_ylabel('(a) val. BPB', fontproperties=prop('demi', FLAB), color=INK, labelpad=4)
    best7 = [r for r in rows if r['a'] == 7][0]['bpb']
    ax.text(15.0, 1.168, f'A8 to A27: {M["CRThreeLateValid"]} valid attempts,\n{M["CRThreeLateGain"]} BPB gained in total',
            ha='center', va='center', fontproperties=prop('reg', 13), color=INK2, linespacing=1.1)
    # (b) speed-ups, log scale
    ax = axs[1]
    for key, col, lab in (('e2e', '#9aa0a8', 'whole generation (unscored prefill + decode)'),
                          ('dec', VIOLET, 'decode steps (scored)')):
        ax.plot(a, [r[key] for r in rows], color=col, lw=1.6 if key == 'dec' else 1.3, zorder=2)
        for r in rows:
            ax.scatter([r['a']], [r[key]], s=36, marker='o', color='white' if unc(r) else col, edgecolor=col,
                       lw=1.2, zorder=3)
    ax.set_yscale('log')
    ax.set_ylim(0.85, 45)
    ax.set_yticks([1, 2, 5, 10, 20, 32])
    ax.set_yticklabels(['1×', '2×', '5×', '10×', '20×', '32×'])
    ax.minorticks_off()
    ax.set_ylabel('(b) speed-up', fontproperties=prop('demi', FLAB), color=INK, labelpad=4)
    ax.text(14.5, 3.0, 'whole generation (prefill + decode)', fontproperties=prop('reg', 13), color=MUTED,
            ha='center', va='top')
    ax.text(1.0, 2.6, 'decode (scored)', fontproperties=prop('demi', 13), color=VIOLET, ha='left', va='bottom')
    marks = [(4, 'fewer layers', (-2, 9), 'center'), (10, 'CUDA graph', (-4, 10), 'right'),
             (14, 'kernel fusion', (-3, 10), 'right'), (18, 'ring K/V cache', (6, -11), 'left'),
             (19, 'split-key attention', (-5, 9), 'right'), (22, 'fused decode step', (2, 9), 'left')]
    byA = {r['a']: r for r in rows}
    for at, lab, off, ha in marks:
        ax.annotate(f'A{at} {lab}', (at, byA[at]['dec']), xytext=off, textcoords='offset points', ha=ha,
                    va='center', fontproperties=prop('demi', 12.5), color=INK)
    # (c) memory
    ax = axs[2]
    ax.plot(a, [r['mib'] for r in rows], color=VIOLET, lw=1.2, alpha=0.45, zorder=2)
    ax.scatter(a, [r['mib'] for r in rows], s=34, color=VIOLET, edgecolor='white', lw=0.8, zorder=3)
    ax.axhline(P['opus_start']['mib'], color=MUTED, lw=1.0, ls=(0, (3, 2)), zorder=1)
    ax.set_ylim(60, 460)
    ax.set_yticks([100, 200, 300, 400])
    ax.set_ylabel('(c) MiB', fontproperties=prop('demi', FLAB), color=INK, labelpad=4)
    ax.annotate('A10: 429 MiB, A11 fixes\nthe graph memory', (10, byA[10]['mib']), xytext=(8, -2),
                textcoords='offset points', ha='left', va='top', fontproperties=prop('reg', 12.5), color=INK2,
                linespacing=1.05)
    ax.set_xlabel('attempt', fontproperties=prop('reg', TICK), color=INK2, labelpad=2)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 27])
    for k, ax in enumerate(axs):
        ax.grid(axis='y', color='#e8eaee', lw=0.7, zorder=0)
        style_ax(ax)
        if k < 2:
            ax.tick_params(labelbottom=False)
    return y + sum(ax_h) + sum(gaps) + 10.5


def timeline(x, y, w):
    C, A = P['census'], P['abl']
    rows = [('Starting model', 'baseline', 1.0, False, False),
            ('Opus, neither', 'a23-V4_eager_unfused', A['V4_eager_unfused'], False, False),
            ('fusion only', 'a23-V1_no_cuda_graph', A['V1_no_cuda_graph'], False, True),
            ('graph only', 'a23-V2_no_fusion', A['V2_no_fusion'], True, False),
            ('graph + fusion', 'a23-V0_as_submitted', A['V0_as_submitted'], True, True)]
    lab_w, num_w = 52.0, 56.0
    x0 = x + lab_w
    full = w - lab_w - num_w
    rh = 13.5
    text(x + w, y + 3.6, 'CPU launches · GPU kernels · speed-up', 13, 'demi', MUTED, ha='right')
    yy = y + 6
    for name, ck, s, graph, fused in rows:
        L_ = full / s
        text(x, yy + 8.2, name, 15, 'demi' if name == 'graph + fusion' else 'reg', INK)
        cv.plot([x0, x0 + L_], [yy + 4.0, yy + 4.0], color='#d5d8de', lw=0.8, zorder=1)
        nl, nk = C[ck]['launches'], C[ck]['kernels']
        if graph:
            rect(x0, yy + 1.0, max(1.2, min(4.0, L_ * 0.25)), 3.0, VIOLET, z=3)
            xt = x0 + max(1.2, min(4.0, L_ * 0.25)) + 0.8
            cv.plot([xt, xt], [yy + 1.0, yy + 4.0], color=INK, lw=0.6, zorder=3, solid_capstyle='butt')
        else:
            k = int(min(nl, 160) * L_ / full) if L_ < full else 160
            k = max(k, 12)
            for i in range(k):
                xx = x0 + (i + 0.5) * L_ / k
                cv.plot([xx, xx], [yy + 1.0, yy + 4.0], color=INK, lw=0.6, zorder=3, solid_capstyle='butt')
        # GPU lane: kernels; with the graph they run back to back (a solid band at this scale)
        gy = yy + 5.8
        if graph:
            rect(x0, gy, L_, 3.4, VIOLET if fused else GEMV, z=3)
        else:
            k = int(min(nk, 160) * L_ / full) if L_ < full else 160
            k = max(k, 12)
            for i in range(k):
                xx = x0 + (i + 0.5) * L_ / k
                rect(xx - 0.18, gy, 0.36 if not fused else 0.75, 3.4, VIOLET if fused else GEMV, z=3)
        text(x + w, yy + 8.2, f'{nl:.0f} · {nk:.0f} · ' + mx(s).replace(r'$\times$', '×'), 15,
             'demi' if fused and graph else 'reg', VIOLET if fused and graph else INK, ha='right')
        yy += rh
    text(x0, yy + 2.5, 'CPU (top) and GPU (bottom) per decoded token; time runs left to right', 13, 'it', MUTED)
    return yy + 4


# ================================================================ column 3: verified results and the contract
def column3(x, w):
    y = section_title(x, BODY_TOP, w, 'Verified on one benchmark')
    y = para(x, y + 1, w, [('All 13 recipes, retrained on seeds 6 to 8 and timed on one A40 with the bracketed '
                            'protocol (means ± SD over seeds; held-out BPB, lower is better).', 'reg', INK)])
    keys = ['baseline', 'arref', 'gpt_a22', 'opus_a23', 'opus_a24', 'opus_a27']
    rows = []
    for k in keys:
        r = R[k]
        rows.append([r['name'], f"{r['params'] / 1e6:.1f}M", f"{r['tok']:.3f} ± {r['tok_sd']:.3f}",
                     f"{r['fix']:.3f} ± {r['fix_sd']:.3f}", f"{r['dec']:,.0f}", f"{r['mem']:.0f}"])
    y = table(x, y + 4, w, [('Recipe', 0.20, 'l'), ('Params', 0.12, 'r'), ('BPB, 201.3M\ntokens', 0.21, 'r'),
                            ('BPB, 300 s', 0.20, 'r'), ('Decode\ntok/s', 0.13, 'r'), ('Memory\nMiB', 0.14, 'r')],
              rows, shade=(3, 4, 5))
    y = caption(x, y + 2.5, w, 'Table 2.', f'At equal tokens AR-ref (AutoResearch) learns most per token: all '
                f'{M["CRSixTokWorseCount"]} other recipes are worse in every seed. At 300 s no recipe is resolved '
                'as better than AR-ref (Opus A27: ' + M['CRSixOpTwSevenTimeD'] + ', 95% CI '
                + M['CRSixOpTwSevenTimeCI'].replace('[', '').replace(']', '').replace(', ', ' to ')
                + f'). Opus decodes 32× faster in {M["CRSixOpMemLo"]} to {M["CRSixOpMemHi"]} instead of '
                f'{M["CRSixARMem"]} MiB.')
    y = pareto_fig(x, y + 5, w, 122)
    y = caption(x, y + 1.5, w, 'Figure 4.', 'Quality against decode speed (log), seed means of all 13 recipes; '
                                            'dashed: frontier of the means in these two objectives. Memory, the third '
                                            'objective, is in Table 2.')
    y = section_title(x, y + 8, w, 'Why 32× decoding is only 4.6× generation')
    y = para(x, y + 1, w, [('The benchmark scored decode steps. A whole generation also includes the prompt ',
                            'reg', INK), ('prefill', 'it', INK), (', which was recorded but not scored:', 'reg', INK)])
    y = display(x + w / 2, y + 4, r'$T_{\mathrm{gen}}=T_{\mathrm{prefill}}+127\,t_{\mathrm{step}},\qquad'
                                  r'S_{\mathrm{gen}}=\frac{%.1f+%.0f}{%.0f+%.0f}=%.1f\times$'
                % (L['base_pre'], L['base_dec'], L['op_pre'], L['op_dec'], L['e2e_x']), 19)
    y = latency_fig(x, y + 5, w)
    bound = L['base_tot'] / L['op_pre']
    y = para(x, y + 3, w, [('Opus captures its CUDA graph inside the prefill (A11: ', 'reg', INK),
                           (Q['op11'], 'it', INK), (f'): {L["op_pre"]:.0f} ms against the baseline’s '
                                                    f'{L["base_pre"]:.1f} ms. With this prefill, even instant '
                                                    'decoding would give at most (derived)', 'reg', INK)])
    y = display(x + w / 2, y + 3, r'$\lim_{t_{\mathrm{step}}\to 0}S_{\mathrm{gen}}=\frac{%.0f}{%.0f}=%.1f\times$'
                % (L['base_tot'], L['op_pre'], bound), 19)
    y = callout(x, y + 5, w, [('Same greedy tokens up to BF16 near-ties. ', 'demi', INK),
                              (f'All {M["CRSixTokensChecked"]} generated tokens (39 runs × 8 prompts × 128) were '
                               'checked against each recipe’s uncached forward pass: '
                               f'{M["CRSixAgreePct"]}% identical, {M["CRSixNearTies"]} near-ties within 0.25 logits, '
                               '0 hard mismatches (agreement of greedy tokens, not identical logits).', 'reg', INK)],
                size=15.5, align='left')
    y = para(x, y + 5, w, [('Limitations. ', 'demi', INK),
                           ('Three seeds; each compared search ran once, under different harnesses; the 32× compares '
                            'inference engines as well as models, and measures speed, not quality. ', 'reg', INK2),
                           ('AI use. ', 'demi', INK),
                           ('Research agents: GPT-5.6 and Claude Opus 5.5. Claude (Opus 5.5, Claude Code; same family as '
                            'the Opus agent) wrote and ran the camera-ready experiments under approved plans and drafted '
                            'this poster. Authorship is human.', 'reg', INK2)], 13.5, lead=1.25)
    y = para(x, y + 4, w, [('References. ', 'demi', INK),
                           ('[1] A. Karpathy. autoresearch. Software repository, 2026. ', 'reg', INK2),
                           ('[2] F. Ferreira, L. Wobbe, A. Krishnakumar, F. Hutter, A. Zela. Can LLMs beat classical '
                            'hyperparameter optimization algorithms? A study on autoresearch. arXiv:2603.24647, 2026.',
                            'reg', INK2)], 13.5, lead=1.25)
    return y


def pareto_fig(x, y, w, h):
    pw = (w - 30) / 2
    out = y
    for j, (key, title, ylim) in enumerate((('tok', '(a) equal tokens, 201.3M', (1.0905, 1.1935)),
                                            ('fix', '(b) equal time, 300 s', (1.1485, 1.1935)))):
        ax = ax_at(x + 18 + j * (pw + 12), y + 6, pw, h - 24)
        text(x + 18 + j * (pw + 12), y + 3.8, title, FLAB, 'demi', INK)
        lo, hi = ylim
        pts = []
        for r in P['recipes']:
            q, v, g = r[key], r['dec'], r['group']
            pts.append((r['key'], q, v))
            ax.scatter([v], [min(q, hi - 0.002)], s=70 if g == 'opus' else 46, marker=MARK[g], color=COL[g],
                       edgecolor='white', lw=0.8, zorder=5, clip_on=False, alpha=1.0 if g == 'opus' else 0.9)
        fr = poster_data.pareto(pts)
        fp = sorted([p for p in pts if p[0] in fr], key=lambda p: p[2])
        xs, ys = [], []
        for i, (_, q, v) in enumerate(fp):
            if i:
                xs.append(v)
                ys.append(ys[-1])
            xs.append(v)
            ys.append(q)
        ax.plot(xs, ys, color=INK2, lw=1.0, ls=(0, (3, 2)), zorder=4)
        ax.set_xscale('log')
        ax.set_xlim(112, 9000)
        ax.set_xticks([125, 500, 2000])
        ax.set_xticklabels(['125', '500', '2k'])
        ax.minorticks_off()
        ax.set_ylim(hi, lo)
        ax.set_yticks([1.10, 1.12, 1.14, 1.16, 1.18] if key == 'tok' else [1.15, 1.16, 1.17, 1.18, 1.19])
        ax.set_yticklabels([f'{t:.2f}' for t in ax.get_yticks()])
        ax.grid(color='#eceef1', lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        style_ax(ax)
        ax.set_xlabel('decode tokens/s', fontproperties=prop('reg', TICK), color=INK2, labelpad=2)
        if j == 0:
            ax.set_ylabel('held-out BPB', fontproperties=prop('reg', TICK), color=INK2, labelpad=3)
        ov = sum(R[k]['dec'] for k in ('opus_a23', 'opus_a24', 'opus_a27')) / 3
        oq = max(R[k][key] for k in ('opus_a23', 'opus_a24', 'opus_a27'))
        ax.annotate('Opus', (ov, oq), xytext=(0, -9), textcoords='offset points', ha='center', va='top',
                    fontproperties=prop('demi', 13.5), color=VIOLET)
        ax.annotate('AR-ref', (R['arref']['dec'], R['arref'][key]), xytext=(7, 3) if key == 'tok' else (2, 13),
                    textcoords='offset points', ha='left' if key == 'tok' else 'center', va='center', fontproperties=prop('demi', 13),
                    color=COL['autoresearch'])
        ax.annotate('GPT A22', (R['gpt_a22']['dec'], R['gpt_a22'][key]), xytext=(6, -9 if key == 'tok' else 8),
                    textcoords='offset points', ha='left', va='center', fontproperties=prop('demi', 13),
                    color=COL['gpt'])
        if key == 'fix':
            ax.annotate('Baseline 1.240 (off scale)', (R['baseline']['dec'], hi - 0.002), xytext=(7, 0),
                        textcoords='offset points', ha='left', va='center', fontproperties=prop('reg', 11.5),
                        color=MUTED)
        out = y + h
    # legend
    lx = x + 18
    for g, lab in (('baseline', 'Baseline'), ('autoresearch', 'AutoResearch'), ('classical', 'Classical (5)'),
                   ('gpt', 'GPT-5.6'), ('opus', 'Opus 5.5')):
        cv.scatter([lx], [out - 2.2], s=44, marker=MARK[g], color=COL[g], edgecolor='white', lw=0.6, zorder=5)
        text(lx + 3.0, out - 0.8, lab, 13, 'reg', INK2)
        lx += tw(lab, 13) + 11
    return out + 1


def latency_fig(x, y, w):
    lx0 = x + 25
    full = w - 25 - 24
    sc = full / 900.0
    for i, (name, pre, dec, col) in enumerate((('Baseline', L['base_pre'], L['base_dec'], COL['baseline']),
                                               ('Opus', L['op_pre'], L['op_dec'], VIOLET))):
        yy = y + i * 13
        text(x, yy + 7.2, name, 15.5, 'demi', INK)
        rect(lx0, yy, pre * sc, 10, PREFILL, ec='#9b968c', lw=0.6, z=3, hatch='///')
        rect(lx0 + pre * sc, yy, dec * sc, 10, col, z=3)
        text(lx0 + (pre + dec) * sc + 2, yy + 7.2, f'{pre + dec:.0f} ms', 15.5, 'demi', INK)
    text(lx0 + (L['base_pre'] + L['base_dec'] / 2) * sc, y + 7.0, f'decode {L["base_dec"]:.0f} ms', 13.5, 'demi',
         'white', ha='center')
    text(lx0 + L['op_pre'] * sc / 2, y + 13 + 7.0, f'prefill {L["op_pre"]:.0f}', 13, 'demi', INK, ha='center')
    yk = y + 28
    rect(lx0, yk, 5.5, 3.8, PREFILL, ec='#9b968c', lw=0.6, z=3, hatch='///')
    text(lx0 + 7, yk + 3.6, 'prefill (not scored)', 13, 'reg', INK2)
    rect(lx0 + 54, yk, 5.5, 3.8, VIOLET, z=3)
    text(lx0 + 61, yk + 3.6, f'decode (scored): {L["base_dec"]:.0f} to {L["op_dec"]:.0f} ms', 13, 'reg', INK2)
    return yk + 5


def build():
    header()
    footer()
    ends = [column1(COLX[0], COLW), column2(COLX[1], COLW), column3(COLX[2], COLW)]
    print('column ends (mm):', [round(e, 1) for e in ends], 'limit', BODY_BOT)
    OUT.mkdir(exist_ok=True)
    meta = {'Title': 'AutoPareto: NeurIPS 2026 AutoMLR poster (landscape, version 3)', 'Author': 'Rami Alobaidy',
            'Subject': 'Auditing Fixed-Time Autonomous Transformer Search with Matched-Exposure Verification'}
    fig.savefig(OUT / 'autopareto_poster_v3_landscape.pdf', metadata=meta)
    fig.savefig(OUT / 'autopareto_poster_v3_landscape.svg')
    fig.savefig(OUT / 'autopareto_poster_v3_preview.png', dpi=110)
    fig.savefig(OUT / 'autopareto_poster_v3_preview_small.png', dpi=40)
    print('wrote', OUT)


if __name__ == '__main__':
    build()
