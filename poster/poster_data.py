#!/usr/bin/env python3
"""Every number on the AutoPareto poster, read from the paper's frozen records and checked against the paper.

Sources (all already used by the camera-ready paper; nothing here re-runs or re-times anything):
  results/camera_ready_2026/phase6/analysis/unified_13.json   13-recipe test (Figures 1 and 3, Table 6)
  paper/camera_ready/data/cr_v2.json                           E1 curves, Opus search records, Phase 5 ablations
                                                               and the profiler census (Figure 4, Appendix G)
  analysis/cr_r9.py: lineage3_data()                           attempt records and quotes of Figure 2; every quote
                                                               is asserted verbatim against its attempt record
  paper/generated/camera_ready/*.tex                           the macros the paper prints

`load()` returns (P, M): P holds plotted data and printed strings, M the paper's macros as plain text. Every string
the poster prints that is also in the paper is asserted equal to the paper's macro here, so a poster number cannot
drift from the paper. Numbers that the paper does not print (for example the profiler's CPU launch counts) come from
the same record files and are listed in P['sources'] for the claims checklist.
"""
from __future__ import annotations

import json
import os
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'poster/.mplcache'))
os.environ.setdefault('AUTOPARETO_EVIDENCE_ROOT', str(ROOT.parent / 'autopareto'))

UNI = ROOT / 'results/camera_ready_2026/phase6/analysis/unified_13.json'
CRV2 = ROOT / 'paper/camera_ready/data/cr_v2.json'
GEN = ROOT / 'paper/generated/camera_ready'
ORDER = ['baseline', 'arref', 'rs', 'tpe', 'motpe', 'nsgaii', 'cmaes', 'gpt_a22', 'gpt_a24', 'gpt_a25',
         'opus_a23', 'opus_a24', 'opus_a27']
OPUS = ('opus_a23', 'opus_a24', 'opus_a27')
IA_MODELS = ('opus_attempt23', 'opus_attempt24', 'opus_attempt27')


def need(cond, what):
    if not cond:
        raise SystemExit(f'poster_data: check failed: {what}')


def tex_plain(v: str) -> str:
    v = v.replace('{,}', ',').replace('$-$', '−').replace('\\,', ' ').replace('$', '')
    return v


def macros() -> dict:
    M = {}
    for f in sorted(GEN.glob('*.tex')):
        for m in re.finditer(r'\\newcommand\{\\([A-Za-z]+)\}\{((?:[^{}]|\{[^{}]*\})*)\}', f.read_text()):
            M[m.group(1)] = tex_plain(m.group(2))
    return M


def group(recipe, family):
    if family in ('baseline', 'autoresearch', 'classical'):
        return family
    return 'gpt' if recipe.startswith('gpt') else 'opus'


def pareto(points):
    """points: [(name, bpb, speed)] -> non-dominated names (lower bpb, higher speed). Same rule as the paper."""
    return [n for n, q, v in points
            if not any(q2 <= q and v2 >= v and (q2 < q or v2 > v) for _, q2, v2 in points)]


def f1(x):
    return f'{x:.1f}'


def load(with_lineage=True):
    M = macros()
    U = json.loads(UNI.read_text())
    S = {s['recipe']: s for s in U['summary']}
    rows = U['rows']
    D = json.loads(CRV2.read_text())
    P = {'sources': {}}
    src = P['sources']

    # ------------------------------------------------------------------ 13-recipe test (Figures 1 and 3)
    uns = {tuple(x) for x in U['inference_stability']['unstable_runs']}
    need(len(uns) == int(M['CRSixUnstableRuns']), 'unstable run count')
    rec = []
    for r in ORDER:
        s = S[r]
        seeds = [{'seed': x['seed'], 'tok': x['heldout_bpb'], 'fix': x['fixed_heldout_bpb'],
                  'dec': x['inf_decode_tok_s'], 'flag': (r, x['seed']) in uns}
                 for x in rows if x['recipe'] == r]
        rec.append({'key': r, 'name': s['display'], 'group': group(r, s['family']), 'params': s['params_mean'],
                    'tok_sd': s['heldout_bpb_sd'], 'fix_sd': s['fixed_heldout_bpb_sd'], 'ftok': s['fixed_tokens_mean'],
                    'tok': s['heldout_bpb_mean'], 'fix': s['fixed_heldout_bpb_mean'],
                    'dec': s['inf_decode_tok_s_mean'], 'mem': s['inf_peak_allocated_mib_mean'],
                    'train': s['fixed_train_tok_s_mean'], 'speedup': s['inf_speedup_vs_bracket_mean'],
                    'seeds': seeds})
    P['recipes'] = rec
    R = {x['key']: x for x in rec}
    P['front_tok'] = pareto([(x['key'], x['tok'], x['dec']) for x in rec])
    P['front_fix'] = pareto([(x['key'], x['fix'], x['dec']) for x in rec])
    # the paper's printed frontier facts
    names = lambda ks: {R[k]['name'] for k in ks}
    need(names(P['front_tok']) >= {'AR-ref', 'TPE', 'GPT A22', 'Opus A27', 'Opus A24', 'NSGA-II'}, 'tok frontier')
    need({'GPT A22', 'Opus A27'} <= names(P['front_fix']), 'fixed-time frontier')
    # printed numbers
    need(f"{R['baseline']['mem']:.0f}" == M['CRSixARMem'] == f"{R['arref']['mem']:.0f}", 'AR-ref memory 337')
    om = [R[k]['mem'] for k in OPUS]
    need((f'{min(om):.0f}', f'{max(om):.0f}') == (M['CRSixOpMemLo'], M['CRSixOpMemHi']), 'Opus memory 96-110')
    od = [R[k]['dec'] for k in OPUS]
    need((f'{min(od):,.0f}', f'{max(od):,.0f}') == (M['CRSixDecodeLo'], M['CRSixDecodeHi']), 'Opus decode')
    osp = [R[k]['speedup'] for k in OPUS]
    need((f1(min(osp)), f1(max(osp))) == (M['CRSixSpeedLo'], M['CRSixSpeedHi']), 'Opus speed-up 31.4-32.1')
    # the paper prints GPT A22's 1.9x from its search records (CRNineGptDecX); the 13-recipe record gives 1.931x,
    # which the poster prints as its re-timed value (from the record; the paper prints 1.8x against AR-ref instead)
    need(M['CRNineGptDecX'] == '1.9' and f1(R['gpt_a22']['speedup']) == '1.9', 'GPT A22 about 1.9x in both sources')
    need(f"{R['gpt_a22']['train'] / R['arref']['train']:.2f}" == M['CRSixGptTrainRatio'], 'GPT A22 trains 1.81x')
    need(f"{R['gpt_a22']['dec'] / R['arref']['dec']:.1f}" == M['CRSixGptDecRatio'], 'GPT A22 decodes 1.8x')
    need(f"{100 * (1 - R['gpt_a22']['mem'] / R['arref']['mem']):.0f}" == M['CRSixGptMemRed'], 'GPT A22 28% memory')
    need(f"{R['gpt_a22']['params'] / 1e6:.1f}" == '29.4' and f"{R['baseline']['params'] / 1e6:.1f}" == '50.3', 'params')
    P['params_M'] = {k: R[k]['params'] / 1e6 for k in ORDER}
    src['13-recipe test'] = str(UNI.relative_to(ROOT))

    # ------------------------------------------------------------------ whole generation (Figure 4c)
    base = S['baseline']
    ops = [S[k] for k in OPUS]
    lat = {'base_pre': base['inf_prefill_ms_mean'], 'base_dec': base['inf_e2e_ms_mean'] - base['inf_prefill_ms_mean'],
           'op_pre': statistics.mean(o['inf_prefill_ms_mean'] for o in ops),
           'op_dec': statistics.mean(o['inf_e2e_ms_mean'] - o['inf_prefill_ms_mean'] for o in ops)}
    lat['base_tot'] = lat['base_pre'] + lat['base_dec']
    lat['op_tot'] = lat['op_pre'] + lat['op_dec']
    lat['e2e_x'] = lat['base_tot'] / lat['op_tot']
    need(f1(lat['e2e_x']) == M['CRSixEtoEXLo'] == '4.6', 'whole generation 4.6x')
    need(f"{lat['base_dec']:.0f}" == M['CRSixBaseDecDms'] and f"{lat['op_dec']:.0f}" == M['CRSixOpDecDms'], '873 -> 28 ms')
    need(f1(lat['base_pre']) == M['CRSixBasePrefill'], 'baseline prefill 7.3 ms')
    pre = [o['inf_prefill_ms_mean'] for o in ops]
    need((f'{min(pre):.0f}', f'{max(pre):.0f}') == (M['CRSixPrefillLo'], M['CRSixPrefillHi']), 'Opus prefill 163-165')
    need(f"{lat['base_tot']:.0f}" == '880' and f"{lat['op_tot']:.0f}" == '192' and f"{lat['op_pre']:.0f}" == '164',
         'Figure 4c labels 880 / 192 / 164')
    P['lat'] = lat

    # ------------------------------------------------------------------ E1 (Appendix Figure 6)
    e = D['e1']
    cur = {}
    for arm, key in (('AR-ref', 'arref'), ('AP-Q', 'gpt_a22')):
        pts = []
        for cell in ('tokens_088604672', 'tokens_157286400', 'tokens_201326592'):
            c = e['curves'][f'{arm}|{cell}']
            pts.append({'tok': c['tokens'][0], 'bpb': statistics.mean(c['heldout']), 'seeds': c['heldout']})
        t = e['curves'][f'{arm}|time_300s']
        cur[key] = {'budget': pts, 't300': {'tok': statistics.mean(t['tokens']), 'bpb': statistics.mean(t['heldout']),
                                            'seed_tok': t['tokens'], 'seeds': t['heldout']}}
    need(f"{cur['gpt_a22']['t300']['tok'] / 1e6:.1f}" == M['CREOneTokAP'], 'E1 158.1M tokens')
    need(f"{cur['arref']['t300']['tok'] / 1e6:.1f}" == M['CREOneTokAR'], 'E1 88.0M tokens')
    hd = e['per_cell']['heldout_bpb']
    gaps = [hd[c]['mean'] for c in ('tokens_088604672', 'tokens_157286400', 'tokens_201326592')]
    need(all(min(hd[c]['values']) > 0 for c in ('tokens_088604672', 'tokens_157286400', 'tokens_201326592')),
         'GPT A22 worse at every budget and seed')
    need(f"{gaps[2]:+.3f}" == M['CRSixGptTokD'], 'E1 gap +0.042 at 201.3M')
    t3 = hd['time_300s']
    need(f"{t3['mean']:+.3f}".replace('-', '−') == M['CRSixGptTimeD'], '300 s gap -0.002')
    need(f"[{t3['ci'][0]:+.3f}, {t3['ci'][1]:+.3f}]".replace('-', '−') == M['CRSixGptTimeCI'], '300 s CI')
    P['e1'] = cur
    P['e1_gaps'] = gaps
    P['e1_300'] = t3
    src['E1'] = 'paper/camera_ready/data/cr_v2.json (e1)'

    # ------------------------------------------------------------------ ablations and census (Figure 4b, §1.5, App. G)
    ia = D['phase5']['ablation']
    sp, cen = ia['speedup'], ia['census']
    med = lambda v: statistics.median(sp[m][v]['decode'] for m in IA_MODELS)
    abl = {v: med(v) for v in ('V0_as_submitted', 'V1_no_cuda_graph', 'V2_no_fusion', 'V3_sdpa_attention',
                               'V4_eager_unfused', 'V5_linear_kv_cache')}
    need([f1(abl[v]) for v in ('V0_as_submitted', 'V5_linear_kv_cache', 'V3_sdpa_attention', 'V2_no_fusion',
                               'V1_no_cuda_graph', 'V4_eager_unfused')] == ['30.8', '29.3', '23.8', '11.6', '2.7', '2.0'],
         'Figure 4b values')
    need(f1(abl['V0_as_submitted']) == M['CRThreeDecX'], 'ablation 30.8x')
    P['abl'] = abl
    # the ablation session's own starting-model decode rate (paper: 149 tok/s, runs 145 to 153); v7 draws time per
    # decoded token as 1 / (this rate x speed-up)
    need(f"{ia['baseline']['decode_tps']:.0f}" == M['CRIABaseDecode'] == '149', 'ablation baseline 149 tok/s')
    P['abl_base_tps'] = ia['baseline']['decode_tps']
    P['census'] = {k: {'kernels': v['gpu_kernels_per_token'], 'launches': v['host_kernel_launch_calls_per_token'],
                       'graph_launches': v['host_graph_launch_calls_per_token']} for k, v in cen.items()}
    C = P['census']
    need(f"{C['baseline']['kernels']:.0f}" == M['CRIAKernBase'] == '625', '625 kernels per token')
    need(f"{C['a23-V4_eager_unfused']['kernels']:.0f}" == M['CRIAKernEager'] == '290', '290 kernels unfused')
    need(f"{C['a23-V1_no_cuda_graph']['kernels']:.0f}" == M['CRIAKernFused'] == '78', '78 kernels fused')
    need(C['a23-V0_as_submitted']['kernels'] == C['a23-V1_no_cuda_graph']['kernels'], 'graph keeps the kernel count')
    need(C['a23-V2_no_fusion']['kernels'] == C['a23-V4_eager_unfused']['kernels'], 'graph keeps the kernel count (unfused)')
    need(f"{C['baseline']['launches']:.0f}" == M['CRIALaunchBase'] == '618', '618 CPU launch calls')
    need(f"{C['a23-V1_no_cuda_graph']['launches']:.0f}" == M['CRIALaunchNG'] == '75', '75 CPU launch calls')
    need(f"{C['a23-V0_as_submitted']['launches']:.0f}" == M['CRIALaunchGraph'] == '2', '2 CPU launch calls')
    for k in ('graph (V0/V1)', 'fusion (V0/V2)', 'graph+fusion (V0/V4)', 'attention kernel (V0/V3)', 'ring buffer (V0/V5)'):
        need(all(k in D['phase5']['ablation']['factors'][m]['decode'] for m in IA_MODELS) if 'factors' in ia else True, k)
    # interaction of graph and fusion, per recipe: kappa = S_(G+F) / (S_G S_F) with gains over the eager engine (V4),
    # i.e. V0 V4 / (V1 V2); the paper's macro CRIAInteract is its reciprocal in the removal framing
    kap = [sp[m]['V0_as_submitted']['decode'] * sp[m]['V4_eager_unfused']['decode']
           / (sp[m]['V1_no_cuda_graph']['decode'] * sp[m]['V2_no_fusion']['decode']) for m in IA_MODELS]
    need((f'{1 / max(kap):.2f}', f'{1 / min(kap):.2f}') == (M['CRIAInteractLo'], M['CRIAInteractHi']), 'kappa = 1/interaction')
    P['kappa'] = (min(kap), max(kap))
    P['fusion_alone'] = statistics.median(sp[m]['V1_no_cuda_graph']['decode'] / sp[m]['V4_eager_unfused']['decode']
                                          for m in IA_MODELS)
    P['graph_alone'] = statistics.median(sp[m]['V2_no_fusion']['decode'] / sp[m]['V4_eager_unfused']['decode']
                                         for m in IA_MODELS)
    P['graph_on_fusion'] = statistics.median(sp[m]['V0_as_submitted']['decode'] / sp[m]['V1_no_cuda_graph']['decode']
                                             for m in IA_MODELS)
    P['fusion_on_graph'] = statistics.median(sp[m]['V0_as_submitted']['decode'] / sp[m]['V2_no_fusion']['decode']
                                             for m in IA_MODELS)
    acc = (ROOT / 'paper/generated/camera_ready/table_access.tex').read_text()
    need('0 of 75' in acc and '0 of 22' in acc and '19 of 25 valid (first: A8)' in acc, 'Table 2 inference-code counts')
    P['abl_ranges'] = {'graph': (M['CRIAGraphLo'], M['CRIAGraphHi']), 'fusion': (M['CRIAFusionLo'], M['CRIAFusionHi']),
                       'arch': (M['CRIAArchLo'], M['CRIAArchHi']), 'attn': M['CRIAAttnLo'], 'ring': M['CRIARingLo']}
    src['ablations, census'] = 'paper/camera_ready/data/cr_v2.json (phase5.ablation)'

    # ------------------------------------------------------------------ Opus search records (Figure 4a)
    orow = {r['attempt']: r for r in D['agents']['opus']['rows']}
    unc = {a for a, r in orow.items() if r['quality_status'] == 'quality_valid_speed_uncertain'}
    path = {a: orow[a]['decode_x'] for a in (4, 5, 10, 13, 14, 18, 19, 22, 23)}
    need([f1(path[a]) for a in (4, 5, 10, 14, 18, 19, 22, 23)] == ['1.9', '2.5', '8.6', '16.8', '24.7', '29.6', '32.0',
                                                                     '30.8'], 'Figure 4a stages')
    need(f1(path[22]) == M['CREightOpTwentyTwoX'], 'A22 32.0x')
    P['opus_path'] = path
    P['opus_rows'] = [{'a': r['attempt'], 'bpb': r['val_bpb'], 'dec': r['decode_x'], 'e2e': r['e2e_x'],
                       'mib': r['peak_mib'], 'pre': r['prefill_ms'], 'status': r['quality_status']}
                      for r in D['agents']['opus']['rows'] if r.get('val_bpb') is not None]
    P['opus_start'] = {'bpb': D['agents']['baseline']['val_bpb'], 'mib': D['agents']['baseline']['peak_mib']}
    late = [r for r in P['opus_rows'] if r['a'] > 7 and r['status'] != 'invalid']
    need(len(late) == int(M['CRThreeLateValid']), '19 valid attempts after A7')
    need(f"{orow[7]['val_bpb'] - min(r['bpb'] for r in late):.4f}" == M['CRThreeLateGain'], 'gain 0.0033 after A7')
    need(abs(orow[8]['prefill_ms'] - 6.4) < 0.05 and abs(orow[10]['prefill_ms'] - 81.3) < 0.05, 'A8 6.4 ms, A10 81 ms prefill')
    P['opus_uncertain'] = sorted(unc)
    P['opus_a23_remeasured'] = D['agents']['opus']['winners']['23']['decode_x']

    # ------------------------------------------------------------------ attempt records and quotes (Figure 2)
    if with_lineage:
        from cr_r9 import _q, lineage3_data
        AR, GP, OP = lineage3_data()
        ar, ap, op = AR['nodes'], GP['nodes'], OP['nodes']
        Q = {
            'ar5': _q(ar, 5, 'twice as many optimizer updates within the fixed training time'),
            'gpt1': _q(ap, 1, 'roughly four times as many optimizer updates'),
            'gpt4': _q(ap, 4, 'materially improving cached decode speed'),
            'gpt8': _q(ap, 8, 'the smaller model processes more tokens'),
            'gpt18': _q(ap, 18, 'recover enough validation quality'),
            'op1': _q(op, 1, 'optimizer-update-limited, not token-limited'),
            'op7': _q(op, 7, 'the quality prediction was falsified'),
            'op10a': _q(op, 10, 'Batch-1 cached decode in attempt 8 is launch-bound'),
            'op10b': _q(op, 10, 'CUDA-graph replay is the only lever that can raise speed without a quality cost'),
            'op11': _q(op, 11, 'Prefill time rose from 6.4 ms to 81 ms'),
        }
        P['quotes'] = Q
        need(f"{ap[8]['val_bpb'] - ap[7]['val_bpb']:.3f}" == M['CREightAPEightWorse'], 'GPT A8 lost 0.061')
        need(not any(re.search(r'build_inference_adapter|CUDA graph|cuda_graph|torch.compile', n['text'])
                     for n in ap.values()), 'GPT seed 0: no inference-code hypothesis')
        P['gpt_a8_loss'] = M['CREightAPEightWorse']
        P['opus_a8_mem'] = op[7]['peak_mib'] - op[8]['peak_mib']
        P['opus_parent'] = {a_: op[a_]['parent'] for a_ in op}
        # the A7 record's "quality prediction was falsified" refers to A6 (depth 2), +0.0215 BPB over A5
        need('Attempt 6 (depth 2' in op[7]['text'] and op[6]['parent'] == 5, 'A7 record reviews A6')
        P['opus_a6_loss'] = op[6]['val_bpb'] - op[5]['val_bpb']
        need(f"{P['opus_a6_loss']:+.3f}" == '+0.021', 'A6 +0.021 as in Figure 2')
        need('HYPOTHESIS (falsifiable) Batch-1 cached decode in attempt 8 is launch-bound' in ' '.join(op[10]['text'].split()),
             'A10 launch-bound statement is a hypothesis')
        src['quotes'] = 'attempt records via analysis/cr_r9.py lineage3_data() (asserted verbatim)'

    # ------------------------------------------------------------------ correctness, flags, misc. printed values
    for k in ('CRSixAgreePct', 'CRSixNearTies', 'CRSixTokensChecked', 'CRSixUnstableRuns', 'CRSixStableShiftMax',
              'CRSixTokDLo', 'CRSixTokDHi', 'CRSixTokWorseCount', 'CRSixOpTrainRatio', 'CRThreeLateValid',
              'CRThreeLateGain', 'CRSixRhoWiki', 'CRSixFlagBaseDrift', 'CRThreeEtoEX', 'CRThreePreOpus'):
        P[k] = M[k]
    return P, M


if __name__ == '__main__':
    P, M = load(with_lineage='--no-lineage' not in sys.argv)
    for x in P['recipes']:
        print(f"{x['name']:14s} {x['group']:12s} tok {x['tok']:.4f} fix {x['fix']:.4f} dec {x['dec']:8.1f} "
              f"mem {x['mem']:6.1f} x{x['speedup']:.2f}")
    print('front_tok', P['front_tok'], 'front_fix', P['front_fix'])
    print('lat', {k: round(v, 2) for k, v in P['lat'].items()})
    print('abl', {k: round(v, 2) for k, v in P['abl'].items()})
    print('census', P['census'])
    print('quotes', P.get('quotes'))
    print('all poster checks passed')
