#!/usr/bin/env python3
"""Camera-ready additions after E1 and the Opus 5.5 arm (2026-11-08).

Reads only committed analysis outputs and read-only campaign records:
  * E1 (pre-registered):  results/camera_ready_2026/amendment1/E1/analysis/e1_analysis.json
  * E1 (exploratory):     results/camera_ready_2026/amendment1/E1/analysis_exploratory/e1_exposure_decomposition.json
  * Phase 2 re-verification and downstream suite: results/camera_ready_2026/phase2/phase2_verification.json
  * Panel and fixed-time re-runs: paper/camera_ready/data/cr_stats.json (written by cr_evidence.py)
  * Phase 5 (pre-registered, acc86d7): results/camera_ready_2026/phase5/analysis/phase5_analysis.json
      5A: Opus frontier retrained at 201.3M tokens on E1's seeds 6-8; 5B: inference ablations on the Pair-A checkpoints
  * Agent arms (read-only, under AUTOPARETO_EVIDENCE_ROOT):
      Claude Opus 5.5:  runs/pair-a-claude-opus55-t-seed3-new-a40-continuation-20261105c/pair-a-fixed-time-seed3/history.jsonl
      Codex gpt-5.6-sol: runs/pair-a-official-20260914-new-a40-continuation-20260916-r2/pair-a-fixed-time-seed3/history.jsonl

Writes paper/camera_ready/data/cr_v2.json, paper/generated/camera_ready/{cr_macros_v2,table_e1,table_agents,table_bench}.tex
and paper/figures/cr_{main,e1,agents,ablation}.{pdf,png} (+ poster copies).  Launches nothing; no GPU; no network.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / 'paper/camera_ready/.mplcache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cr_common import CR_DATA, EVIDENCE, FIG, GEN_CR, NAMES, PANEL_ORDER, POSTER, ROOT, display_names, read_json, sha256_file, write_json

E1_DIR = ROOT / 'results/camera_ready_2026/amendment1/E1'
E1_PRE = E1_DIR / 'analysis/e1_analysis.json'
E1_EXP = E1_DIR / 'analysis_exploratory/e1_exposure_decomposition.json'
P2 = ROOT / 'results/camera_ready_2026/phase2/phase2_verification.json'
CR_STATS = CR_DATA / 'cr_stats.json'
OPUS_ROOT = EVIDENCE / 'runs/pair-a-claude-opus55-t-seed3-new-a40-continuation-20261105c/pair-a-fixed-time-seed3'
CODEX_ROOT = EVIDENCE / 'runs/pair-a-official-20260914-new-a40-continuation-20260916-r2/pair-a-fixed-time-seed3'
PLAN_SHA = '89f2901588fceec18ab93d4e74ed652b8f6c6df9329eb8ba1d3a7ead128f60dc'
CELLS = ['tokens_088604672', 'tokens_157286400', 'tokens_201326592', 'time_300s']
OPUS_WINNERS = ['23', '24', '27']  # extended frontier (doc 20 section 21); re-verified in Phase 2
FROZEN_ATTEMPTS = 25               # the frozen 25-attempt snapshot (doc 20 section 21)
P5 = ROOT / 'results/camera_ready_2026/phase5/analysis/phase5_analysis.json'
P5A_PLAN_SHA = 'a47f81b752d7953d05227bae77451f398ab0fca672f711f21a2fb3523c72993d'
T95 = 4.302652729911275            # t(0.975, df = 2)
OPUS_KEYS = ('OPUS-A23', 'OPUS-A24', 'OPUS-A27')
IA_MODELS = ('opus_attempt23', 'opus_attempt24', 'opus_attempt27')
IA_VARIANTS = [('V0_as_submitted', 'As submitted'), ('V5_linear_kv_cache', 'Linear K/V'),
               ('V3_sdpa_attention', 'SDPA attn.'), ('V2_no_fusion', 'No fusion'),
               ('V1_no_cuda_graph', 'No graph'), ('V4_eager_unfused', 'Neither')]

BLUE, ORANGE, VIOLET, AQUA = '#2a78d6', '#eb6834', '#6250d6', '#1baf7a'
GRAY, INK, INK2, MUTED, GRID = '#8a8985', '#0b0b0b', '#52514e', '#9a9893', '#e6e5e1'


def sgn(x, nd):
    s = f'{x:+.{nd}f}'
    return ('$-$' + s[1:]) if s.startswith('-') else s


def ci(lo, hi, nd):
    return f'[{sgn(lo, nd)}, {sgn(hi, nd)}]'


# ----------------------------------------------------------------------------- extraction
def load_arm(root):
    rows = []
    for line in (root / 'history.jsonl').read_text().splitlines():
        d = json.loads(line)
        r = d.get('result') or {}
        inf = r.get('inference_result') or {}
        cm = inf.get('candidate_only_memory_measurement') or {}
        pd = d.get('pareto_decision') or {}
        status = pd.get('quality_status')
        rows.append({
            'attempt': d['campaign_attempt'], 'agent': d.get('agent_model_identifier'),
            'valid': r.get('status') == 'completed' and status is not None,
            'quality_status': status, 'decision': pd.get('decision'),
            'category': d.get('change_category'),
            'val_bpb': r.get('validation_bpb'), 'tokens': r.get('total_tokens'),
            'batch': r.get('total_batch_size'), 'params': r.get('num_params'),
            'updates': r.get('optimizer_updates'),
            'decode_x': inf.get('normalized_decode_throughput'),
            'decode_tok_s': inf.get('cached_decode_tokens_per_second_median'),
            'e2e_tok_s': cm.get('end_to_end_tokens_per_second'),
            'prefill_ms': cm.get('prefill_milliseconds'),
            'peak_mib': (inf.get('peak_allocated_inference_memory_bytes') or 0) / 2 ** 20 or None,
        })
    view = read_json(root / 'campaign_view.json')
    return rows, view['baseline']


def best(rows, upto=None):
    pool = [r for r in rows if r['valid'] and (upto is None or r['attempt'] <= upto)]
    return min(pool, key=lambda r: r['val_bpb'])


def extract():
    pre, exp, p2, S = read_json(E1_PRE), read_json(E1_EXP), read_json(P2), read_json(CR_STATS)
    assert pre['plan_sha256'] == PLAN_SHA and exp['plan_sha256'] == PLAN_SHA, 'E1 plan hash mismatch'
    assert pre['runs_used'] == 24 and not pre['synthetic'] and not exp['synthetic']
    assert exp['cross_check']['passed'] and exp['cross_check']['max_abs_difference'] < 1e-12
    m = {k: pre['metrics'][k] for k in ('heldout_bpb', 'validation_bpb')}
    # the exploratory paired deltas must equal the pre-registered per-cell values
    for met in m:
        for c in CELLS:
            a, b = m[met]['per_cell'][c]['values'], exp['paired_delta'][met][c]['values']
            assert max(abs(x - y) for x, y in zip(a, b)) < 1e-12, (met, c)
    desc = exp['descriptive']
    e1 = {
        'per_cell': {met: {c: m[met]['per_cell'][c] for c in CELLS} for met in m},
        'H_E1a': {met: m[met]['H_E1a'] for met in m}, 'H_E1b': {met: m[met]['H_E1b'] for met in m},
        'crossover': {met: m[met]['crossover'] for met in m},
        'crossover_bootstrap': {met: m[met]['crossover_bootstrap'] for met in m},
        'outcome': {met: m[met]['outcome_table']['outcome'] for met in m},
        'fixed_time_vs_history': pre['fixed_time_vs_history'],
        'decomposition': {met: exp['decomposition_300s'][met]['summary'] for met in m},
        'replication': exp['replication_vs_phase1_panel'],
        'seed_consistency': exp['seed_consistency'],
        'curves': {f'{rec}|{c}': {'tokens': desc[f'{rec}|{c}']['tokens']['values'],
                                  'heldout': desc[f'{rec}|{c}']['heldout_bpb']['values'],
                                  'validation': desc[f'{rec}|{c}']['validation_bpb']['values'],
                                  'tok_s': desc[f'{rec}|{c}']['steady_state_tokens_per_second']['values']}
                   for rec in ('AP-Q', 'AR-ref') for c in CELLS},
        'seeds': exp['seeds'],
    }
    pos = sum(1 for c in CELLS[:3] for v in e1['per_cell']['heldout_bpb'][c]['values'] if v > 0)
    e1['matched_positive_pairs'] = pos
    tsA = statistics.mean(e1['curves']['AP-Q|time_300s']['tok_s'])
    tsR = statistics.mean(e1['curves']['AR-ref|time_300s']['tok_s'])
    e1['throughput_ratio_mean_of_means'] = tsA / tsR

    opus, base = load_arm(OPUS_ROOT)
    codex, base_c = load_arm(CODEX_ROOT)
    assert base == base_c, 'the two arms must share the registered baseline'
    assert all(r['agent'] == 'claude-opus-5-5' for r in opus) and all(r['agent'] == 'gpt-5.6-sol' for r in codex)
    probe = p2['baseline_recipe_probe']['median']
    e2e_ref = probe['e2e_tok_s']
    for rows in (opus, codex):
        for r in rows:
            r['e2e_x'] = r['e2e_tok_s'] / e2e_ref if r['e2e_tok_s'] else None
    win = {}
    for a in OPUS_WINNERS:
        w = p2['winners'][a]
        assert w['protected_bpb_delta'] == 0.0
        key = f'opus55_t_attempt{a}'
        dm = p2['downstream'][key]['single']['metrics']
        win[a] = {'val_bpb': w['protected_bpb_remeasured'], 'tokens': w['discovery']['tokens'],
                  'decode_x': w['summary']['speedup_median'], 'stable': w['summary']['stable_repeats'],
                  'e2e_x': w['summary']['e2e_rep_median_tok_s_median'] / probe['e2e_rep_median_tok_s'],
                  'stable_repeats': w['summary']['stable_repeats'], 'repeats': w['summary']['repeats'],
                  'prefill_ms': w['summary']['prefill_ms_median'], 'peak_mib': w['summary']['peak_alloc_mb_median'],
                  'heldout': dm['heldout_climbmix_bpb.bits_per_byte']['value'],
                  'lambada': dm['lambada_openai.exact_match']['value'], 'blimp': dm['blimp.macro_accuracy']['value'],
                  'piqa': dm['piqa.accuracy']['value'], 'arc_e': dm['arc_easy.accuracy']['value'],
                  'params': next(r['params'] for r in opus if str(r['attempt']) == a)}
    bdm = p2['downstream']['pair_a_baseline_t']['single']['metrics']
    ar201 = p2['downstream']['autoresearch_compatible_tokens201M']['seed_summary']
    ap201 = p2['downstream']['autopareto_292_tokens201M']['seed_summary']
    first_adapter = min(r['attempt'] for r in opus if r['valid'] and (r['prefill_ms'] or 0) > 50)
    pre_adapter = max((r for r in opus if r['valid'] and r['attempt'] < first_adapter), key=lambda r: r['attempt'])
    agents = {
        'baseline': {'val_bpb': base['validation_bpb'], 'tokens': base['total_tokens'],
                     'decode_tok_s': base['inference_result']['cached_decode_tokens_per_second_median'],
                     'peak_mib': base['inference_result']['peak_allocated_inference_memory_bytes'] / 2 ** 20,
                     'e2e_ref_tok_s': e2e_ref, 'e2e_rep_ref_tok_s': probe['e2e_rep_median_tok_s'], 'prefill_ref_ms': probe['prefill_ms'],
                     'experiment_id': base['baseline_experiment_id'],
                     'heldout_t_checkpoint': bdm['heldout_climbmix_bpb.bits_per_byte']['value'],
                     'lambada': bdm['lambada_openai.exact_match']['value'], 'blimp': bdm['blimp.macro_accuracy']['value']},
        'opus': {'rows': opus, 'attempts': len(opus), 'valid': sum(r['valid'] for r in opus),
                 'valid_frozen': sum(r['valid'] for r in opus if r['attempt'] <= FROZEN_ATTEMPTS),
                 'speed_uncertain': sum(1 for r in opus if r['quality_status'] == 'quality_valid_speed_uncertain'),
                 'best': best(opus), 'best_first22': best(opus, 22), 'best_frozen': best(opus, FROZEN_ATTEMPTS),
                 'max_tokens': max(r['tokens'] for r in opus if r['valid']),
                 'winners': win, 'first_adapter_attempt': first_adapter, 'pre_adapter': pre_adapter,
                 'first_adapter': next(r for r in opus if r['attempt'] == first_adapter)},
        'codex': {'rows': codex, 'attempts': len(codex), 'valid': sum(r['valid'] for r in codex),
                  'speed_uncertain': sum(1 for r in codex if r['quality_status'] == 'quality_valid_speed_uncertain'),
                  'best': best(codex), 'max_tokens': max(r['tokens'] for r in codex if r['valid']),
                  'max_decode_x': max(r['decode_x'] for r in codex if r['valid']),
                  'max_prefill_ms': max(r['prefill_ms'] for r in codex if r['valid'])},
        'matched_token_refs': {'ar_ref_201M_heldout': ar201['heldout_climbmix_bpb.bits_per_byte']['mean'],
                               'ap_q_201M_heldout': ap201['heldout_climbmix_bpb.bits_per_byte']['mean'],
                               'ar_ref_201M_blimp': ar201['blimp.macro_accuracy']['mean']},
    }
    ev = [json.loads(l) for l in (E1_DIR / 'stage_events.jsonl').read_text().splitlines()]
    t0 = next(e['utc'] for e in ev if e['event'] == 'stage_started')
    t1 = next(e['utc'] for e in ev if e['event'] == 'stage_finished')
    from datetime import datetime
    e1['a40_hours'] = (datetime.fromisoformat(t1) - datetime.fromisoformat(t0)).total_seconds() / 3600
    e1['runs_tok_s'] = {}
    for rec, tag in (('AP-Q', 'apq'), ('AR-ref', 'arref')):
        vals = [read_json(f)['metrics']['steady_state_tokens_per_second']
                for f in sorted((E1_DIR / 'runs').glob(f'e1-{tag}-*/camera_ready_run_record.json'))]
        assert len(vals) == 12 and all(vals), (rec, len(vals))
        e0 = read_json(next((E1_DIR.parent / 'E0/runs').glob(f'e0-{tag}-*/camera_ready_run_record.json')))
        e1['runs_tok_s'][rec] = {'min': min(vals), 'max': max(vals), 'e0': e0['metrics']['steady_state_tokens_per_second']}
    def frontier(upto):
        pool = [r for r in opus if r['attempt'] <= upto and r['quality_status'] == 'quality_valid_speed_valid']
        dom = lambda x, y: (y['val_bpb'] <= x['val_bpb'] and y['decode_x'] >= x['decode_x'] and y['peak_mib'] <= x['peak_mib']
                            and (y['val_bpb'] < x['val_bpb'] or y['decode_x'] > x['decode_x'] or y['peak_mib'] < x['peak_mib']))
        return [r['attempt'] for r in pool if not any(dom(r, o) for o in pool if o is not r)]
    agents['opus']['frontier_frozen'] = frontier(FROZEN_ATTEMPTS)
    agents['opus']['frontier_extended'] = frontier(10 ** 6)
    assert agents['opus']['frontier_extended'] == [int(a) for a in OPUS_WINNERS], agents['opus']['frontier_extended']
    panel = {'fixed_time_apq_minus_ar': S['fixed_time_apq_minus_ar'], 'matched_heldout': S['matched_heldout'],
             'matched_val': S['matched_val'],
             'fixed_tokens_ar': statistics.mean(S['fixed_time']['68']['tokens_by_seed'].values()),
             'fixed_tokens_apq': statistics.mean(S['fixed_time']['134']['tokens_by_seed'].values())}
    inputs = {str(p.relative_to(ROOT) if str(p).startswith(str(ROOT)) else p): sha256_file(p)
              for p in (E1_PRE, E1_EXP, P2, CR_STATS, OPUS_ROOT / 'history.jsonl', CODEX_ROOT / 'history.jsonl',
                        OPUS_ROOT / 'campaign_view.json', CODEX_ROOT / 'campaign_view.json')}
    return {'format': 'autopareto_camera_ready_v2', 'e1': e1, 'agents': agents, 'panel': panel, 'inputs': inputs}


def paired_t(vals):
    m = statistics.mean(vals)
    h = T95 * statistics.stdev(vals) / len(vals) ** 0.5
    return {'mean': m, 'ci': [m - h, m + h], 'values': vals}


def extract_phase5(D):
    """Phase 5 from its committed analysis (scripts/analyze_phase5.py), cross-checked against E1."""
    P = read_json(P5)
    a, b = P['phase5a'], P['phase5b']
    assert a['plan_sha256'] == P5A_PLAN_SHA, '5A plan hash mismatch'
    assert len(a['runs_passed']) == 10 and not a['runs_missing_or_failed']
    dc = a['drift_control']
    assert dc['tokens_equal'] and dc['updates_equal'] and not dc['session_confounded']
    # the comparators are E1's own 201.3M and 300 s runs
    for met, key in (('heldout_bpb', 'heldout'), ('validation_bpb', 'validation')):
        e1v = D['e1']['per_cell'][met]['tokens_201326592']['values']
        assert max(abs(x - y) for x, y in zip(a['contrasts'][f'AP-Q - AR-ref [{key}]']['values'], e1v)) < 1e-12, met
    seeds = [str(x) for x in D['e1']['seeds']]
    ar300 = [a['arref_300s_heldout']['by_seed'][x] for x in seeds]
    assert max(abs(x - y) for x, y in zip(ar300, D['e1']['curves']['AR-ref|time_300s']['heldout'])) < 1e-12
    C = a['contrasts']
    mt = {'recipes': {}, 'drift': dc, 'seeds': seeds, 'apq_k': a['throughput_tradeoff']['AP-Q']['k']['mean']}
    for k in ('AR-ref', 'AP-Q') + OPUS_KEYS:
        r = a['recipes'][k]
        mt['recipes'][k] = {'heldout': r['heldout'], 'validation': r['validation'], 'params': r['parameter_count'],
                            'tok_s': r['training_tokens_per_second']['mean'],
                            'heldout_by_seed': [r['by_seed'][x]['heldout'] for x in seeds]}
    for k in OPUS_KEYS:
        R = mt['recipes'][k]
        R['vs_ar'] = {m: C[f'{k} - AR-ref [{m}]'] for m in ('heldout', 'validation')}
        R['vs_apq'] = {m: C[f'{k} - AP-Q [{m}]'] for m in ('heldout', 'validation')}
        R['rho'] = a['throughput_tradeoff'][k]['rho']['mean']
        R['k_lower_bounds'] = [v['k_lower_bound'] for v in a['throughput_tradeoff'][k]['k_by_seed'].values()
                               if v.get('bound') == 'lower']
        R['k_extrapolated_mean'] = statistics.mean(v['k_linear_extrapolation'] for v in a['throughput_tradeoff'][k]['k_by_seed'].values())
        R['blimp_vs_ar'] = a['downstream']['paired'][f'{k} - AR-ref [blimp.macro_accuracy]']
        R['blimp_vs_apq'] = a['downstream']['paired'][f'{k} - AP-Q [blimp.macro_accuracy]']
        # post hoc, descriptive: Opus at 201.3M tokens (about its 300 s exposure) minus AR-ref at 300 s, by seed
        R['post_hoc_vs_ar300'] = paired_t([x - y for x, y in zip(R['heldout_by_seed'], ar300)])
    mt['other_tasks_ci_span_zero'] = all(
        (lambda c: c[0] <= 0 <= c[1])(a['downstream']['paired'][f'{k} - AR-ref [{t}]']['ci95'])
        for k in OPUS_KEYS for t in ('lambada_openai.exact_match', 'piqa.accuracy', 'arc_easy.accuracy'))
    assert not b['correctness_failures'] and len(b['runs']) == 39
    F = b['factors']
    assert all(F[m]['greedy_samples_identical_across_variants'] for m in IA_MODELS)
    base = b['cells']['pair_a_baseline/canonical']
    ia = {'factors': {m: {'decode': F[m]['decode_tps'], 'e2e': F[m]['e2e_tps']} for m in IA_MODELS},
          'speedup': {m: {v: {'decode': b['cells'][f'{m}/{v}']['decode_tps'] / base['decode_tps'],
                              'e2e': b['cells'][f'{m}/{v}']['e2e_tps'] / base['e2e_tps']}
                          for v, _ in IA_VARIANTS} for m in IA_MODELS},
          'cells': {k: {f: v[f] for f in ('decode_tps', 'e2e_tps', 'prefill_ms', 'peak_mib', 'kv_mib')}
                    for k, v in b['cells'].items()},
          'baseline': {f: base[f] for f in ('decode_tps', 'decode_tps_values', 'e2e_tps', 'prefill_ms', 'peak_mib', 'kv_mib')},
          'census': b['kernel_census'], 'unstable': b['unstable_runs'], 'runs': len(b['runs'])}
    return {'matched_tokens': mt, 'ablation': ia}


# ----------------------------------------------------------------------------- macros and tables
def macros_phase5(D):
    mt, ia = D['phase5']['matched_tokens'], D['phase5']['ablation']
    R = mt['recipes']
    O = [R[k] for k in OPUS_KEYS]
    lohi = lambda xs, fmt: (format(min(xs), fmt), format(max(xs), fmt))
    d_ar = [r['vs_ar']['heldout'] for r in O]
    d_ap = [r['vs_apq']['heldout'] for r in O]
    post = [r['post_hoc_vs_ar300'] for r in O]
    bl = [-r['blimp_vs_ar']['mean'] for r in O]
    fac = lambda key, kind='decode': [ia['factors'][m][kind][key] for m in IA_MODELS]
    cell = lambda v, f: [ia['cells'][f'{m}/{v}'][f] for m in IA_MODELS]
    cen = ia['census']
    capture = [1 - x / y for x, y in zip(cell('V1_no_cuda_graph', 'prefill_ms'), cell('V0_as_submitted', 'prefill_ms'))]
    M = {
        # 5A: matched tokens (pre-registered)
        'CRMTRuns': '10', 'CRMTPairs': str(sum(v > 0 for r in O for v in r['vs_ar']['heldout']['values'])),
        'CRMTDLo': lohi([c['mean'] for c in d_ar], '+.3f')[0], 'CRMTDHi': lohi([c['mean'] for c in d_ar], '+.3f')[1],
        'CRMTDLoAbs': lohi([c['mean'] for c in d_ar], '.3f')[0], 'CRMTDHiAbs': lohi([c['mean'] for c in d_ar], '.3f')[1],
        'CRMTCIMin': f"{min(c['ci95'][0] for c in d_ar):+.3f}",
        'CRMTHolm': f"{max(c['p_holm_primary_family'] for c in d_ar):.3f}",
        'CRMTDAPLo': lohi([c['mean'] for c in d_ap], '+.3f')[0], 'CRMTDAPHi': lohi([c['mean'] for c in d_ap], '+.3f')[1],
        'CRMTDAPCIMin': f"{min(c['ci95'][0] for c in d_ap):+.3f}",
        'CRMTValLo': lohi([r['vs_ar']['validation']['mean'] for r in O], '+.3f')[0],
        'CRMTValHi': lohi([r['vs_ar']['validation']['mean'] for r in O], '+.3f')[1],
        'CRMTHeldLo': lohi([r['heldout']['mean'] for r in O], '.3f')[0],
        'CRMTHeldHi': lohi([r['heldout']['mean'] for r in O], '.3f')[1],
        'CRMTARHeld': f"{R['AR-ref']['heldout']['mean']:.3f}", 'CRMTAPHeld': f"{R['AP-Q']['heldout']['mean']:.3f}",
        'CRMTTpsLo': lohi([r['tok_s'] / 1e3 for r in O], '.0f')[0], 'CRMTTpsHi': lohi([r['tok_s'] / 1e3 for r in O], '.0f')[1],
        'CRMTTpsAR': f"{R['AR-ref']['tok_s'] / 1e3:.0f}", 'CRMTTpsAP': f"{R['AP-Q']['tok_s'] / 1e3:.0f}",
        'CRMTRhoLo': lohi([r['rho'] for r in O], '.2f')[0], 'CRMTRhoHi': lohi([r['rho'] for r in O], '.2f')[1],
        'CRMTKBound': f"{min(x for r in O for x in r['k_lower_bounds']):.2f}",
        'CRMTKExtLo': lohi([r['k_extrapolated_mean'] for r in O], '.2f')[0],
        'CRMTKExtHi': lohi([r['k_extrapolated_mean'] for r in O], '.2f')[1],
        'CRMTPostLo': lohi([c['mean'] for c in post], '+.3f')[0], 'CRMTPostHi': lohi([c['mean'] for c in post], '+.3f')[1],
        'CRMTPostCILo': sgn(min(c['ci'][0] for c in post), 3), 'CRMTPostCIHi': sgn(max(c['ci'][1] for c in post), 3),
        'CRMTBlimpLo': lohi(bl, '.3f')[0], 'CRMTBlimpHi': lohi(bl, '.3f')[1],
        'CRMTBlimpLoPts': lohi([100 * x for x in bl], '.1f')[0], 'CRMTBlimpHiPts': lohi([100 * x for x in bl], '.1f')[1],
        'CRMTKAPQ': f"{mt['apq_k']:.2f}",
        'CRMTDrift': sgn(mt['drift']['heldout'][2], 4), 'CRMTDriftAbs': f"{abs(mt['drift']['heldout'][2]):.4f}",
        'CRMTParamsLo': lohi([r['params'] / 1e6 for r in O], '.1f')[0], 'CRMTParamsHi': lohi([r['params'] / 1e6 for r in O], '.1f')[1],
        # 5B: inference ablations (existing checkpoints)
        'CRIARuns': str(ia['runs']), 'CRIAUnstable': str(len(ia['unstable'])),
        'CRIATotalLo': lohi(fac('total (V0/baseline)'), '.1f')[0], 'CRIATotalHi': lohi(fac('total (V0/baseline)'), '.1f')[1],
        'CRIAGraphLo': lohi(fac('graph (V0/V1)'), '.1f')[0], 'CRIAGraphHi': lohi(fac('graph (V0/V1)'), '.1f')[1],
        'CRIAFusionLo': lohi(fac('fusion (V0/V2)'), '.1f')[0], 'CRIAFusionHi': lohi(fac('fusion (V0/V2)'), '.1f')[1],
        'CRIAJointLo': lohi(fac('graph+fusion (V0/V4)'), '.1f')[0], 'CRIAJointHi': lohi(fac('graph+fusion (V0/V4)'), '.1f')[1],
        'CRIAInteractLo': lohi(fac('interaction (joint / (graph x fusion))'), '.2f')[0],
        'CRIAInteractHi': lohi(fac('interaction (joint / (graph x fusion))'), '.2f')[1],
        'CRIAAttnLo': lohi(fac('attention kernel (V0/V3)'), '.2f')[0], 'CRIAAttnHi': lohi(fac('attention kernel (V0/V3)'), '.2f')[1],
        'CRIARingLo': lohi(fac('ring buffer (V0/V5)'), '.2f')[0], 'CRIARingHi': lohi(fac('ring buffer (V0/V5)'), '.2f')[1],
        'CRIAArchLo': lohi(fac('architecture, eager (V4/baseline)'), '.2f')[0],
        'CRIAArchHi': lohi(fac('architecture, eager (V4/baseline)'), '.2f')[1],
        # headline values: the medians over the three checkpoints that Figure 2c prints (draw_ablation)
        'CRIATotalMed': f"{statistics.median(ia['speedup'][m]['V0_as_submitted']['decode'] for m in IA_MODELS):.0f}",
        'CRIAEtoEMed': f"{statistics.median(ia['speedup'][m]['V0_as_submitted']['e2e'] for m in IA_MODELS):.1f}",
        'CRIAEtoELo': lohi(fac('total (V0/baseline)', 'e2e'), '.1f')[0], 'CRIAEtoEHi': lohi(fac('total (V0/baseline)', 'e2e'), '.1f')[1],
        'CRIAGraphEtoELo': lohi(fac('graph (V0/V1)', 'e2e'), '.1f')[0], 'CRIAGraphEtoEHi': lohi(fac('graph (V0/V1)', 'e2e'), '.1f')[1],
        'CRIAFusionEtoEHi': f"{max(fac('fusion (V0/V2)', 'e2e')):.2f}",
        'CRIAPrefillLo': lohi(cell('V0_as_submitted', 'prefill_ms'), '.0f')[0],
        'CRIAPrefillHi': lohi(cell('V0_as_submitted', 'prefill_ms'), '.0f')[1],
        'CRIAPrefillNGLo': lohi(cell('V1_no_cuda_graph', 'prefill_ms'), '.1f')[0],
        'CRIAPrefillNGHi': lohi(cell('V1_no_cuda_graph', 'prefill_ms'), '.1f')[1],
        'CRIAPrefillBase': f"{ia['baseline']['prefill_ms']:.1f}",
        'CRIACapture': f"{100 * min(capture):.0f}",
        'CRIAKVOp': f"{statistics.mean(cell('V0_as_submitted', 'kv_mib')):.1f}",
        'CRIAKVLin': f"{statistics.mean(cell('V5_linear_kv_cache', 'kv_mib')):.1f}",
        'CRIABaseDecode': f"{ia['baseline']['decode_tps']:.0f}",
        'CRIABaseDecodeLo': f"{min(ia['baseline']['decode_tps_values']):.0f}",
        'CRIABaseDecodeHi': f"{max(ia['baseline']['decode_tps_values']):.0f}",
        'CRIAKernBase': f"{cen['baseline']['gpu_kernels_per_token']:.0f}",
        'CRIALaunchBase': f"{cen['baseline']['host_kernel_launch_calls_per_token']:.0f}",
        'CRIAKernEager': f"{cen['a23-V4_eager_unfused']['gpu_kernels_per_token']:.0f}",
        'CRIAKernFused': f"{cen['a23-V1_no_cuda_graph']['gpu_kernels_per_token']:.0f}",
        'CRIALaunchNG': f"{cen['a23-V1_no_cuda_graph']['host_kernel_launch_calls_per_token']:.0f}",
        'CRIALaunchGraph': f"{cen['a23-V0_as_submitted']['host_kernel_launch_calls_per_token']:.0f}",
    }
    # stage durations from the workspace console log of the Phase 5 chain
    import re
    from datetime import datetime
    log = (ROOT / 'results/camera_ready_2026/phase5/workspace_logs/camera_ready_phase5_console.log').read_text()
    ts = lambda pat: datetime.fromisoformat(re.search(r'^\[([0-9T:-]+)Z\] ' + pat, log, re.M).group(1))
    M['CRMTHours'] = f"{(ts('5A downstream exit 0') - ts('5A train start')).total_seconds() / 3600:.2f}"
    M['CRIAHours'] = f"{(ts('PHASE5B DONE') - ts('phase5b start')).total_seconds() / 3600:.2f}"
    assert cen['a23-V0_as_submitted']['gpu_kernels_per_token'] == cen['a23-V1_no_cuda_graph']['gpu_kernels_per_token']
    assert cen['a23-V0_as_submitted']['host_graph_launch_calls_per_token'] == 1.0
    return M


def macros(D):
    e1, ag, pn = D['e1'], D['agents'], D['panel']
    H, V = e1['per_cell']['heldout_bpb'], e1['per_cell']['validation_bpb']
    dh, dv = e1['decomposition']['heldout_bpb'], e1['decomposition']['validation_bpb']
    rep = e1['replication']['heldout_bpb']['welch_e1_minus_panel']
    sc = e1['seed_consistency']['heldout_bpb']
    op, cx, b = ag['opus'], ag['codex'], ag['baseline']
    W = op['winners']
    matched = [H[c]['mean'] for c in CELLS[:3]]
    tA = statistics.mean(e1['curves']['AP-Q|time_300s']['tokens'])
    tR = statistics.mean(e1['curves']['AR-ref|time_300s']['tokens'])
    rng = lambda xs, nd: (f'{min(xs):.{nd}f}', f'{max(xs):.{nd}f}')
    dec_lo, dec_hi = rng([w['decode_x'] for w in W.values()], 1)
    e2e_lo, e2e_hi = rng([w['e2e_x'] for w in W.values()], 1)
    mem_lo, mem_hi = rng([w['peak_mib'] for w in W.values()], 0)
    held_lo, held_hi = rng([w['heldout'] for w in W.values()], 3)
    pre_lo, pre_hi = rng([w['prefill_ms'] for w in W.values()], 0)
    tok_lo, tok_hi = rng([w['tokens'] / 1e6 for w in W.values()], 0)
    dlam = [w['lambada'] - b['lambada'] for w in W.values()]
    dbl = [w['blimp'] - b['blimp'] for w in W.values()]
    dho = [w['heldout'] - b['heldout_t_checkpoint'] for w in W.values()]
    fa, pa = op['first_adapter'], op['pre_adapter']
    M = {
        # E1, pre-registered
        'CREOneHeldA': sgn(H[CELLS[0]]['mean'], 4), 'CREOneHeldB': sgn(H[CELLS[1]]['mean'], 4),
        'CREOneHeldC': sgn(H[CELLS[2]]['mean'], 4),
        'CREOneMatchedLo': f'{min(matched):+.3f}', 'CREOneMatchedHi': f'{max(matched):+.3f}',
        'CREOneMatchedLoAbs': f'{min(matched):.3f}', 'CREOneMatchedHiAbs': f'{max(matched):.3f}',
        'CREOneTimeHeld': sgn(H['time_300s']['mean'], 4), 'CREOneTimeHeldThree': sgn(H['time_300s']['mean'], 3),
        'CREOneTimeHeldCI': ci(*H['time_300s']['ci'], 3),
        'CREOneTimeHeldLo': sgn(H['time_300s']['ci'][0], 3), 'CREOneTimeHeldHi': sgn(H['time_300s']['ci'][1], 3),
        'CREOneTimeVal': sgn(V['time_300s']['mean'], 4), 'CREOneTimeValThree': sgn(V['time_300s']['mean'], 3),
        'CREOneTimeValCI': ci(*V['time_300s']['ci'], 3),
        'CREOneTimeHeldWins': str(sum(v < 0 for v in H['time_300s']['values'])),
        'CREOneContrast': sgn(e1['H_E1a']['heldout_bpb']['contrast_largest_minus_smallest_budget']['mean'], 4),
        'CREOneContrastCI': ci(*e1['H_E1a']['heldout_bpb']['contrast_largest_minus_smallest_budget']['ci'], 4),
        'CREOnePairs': str(e1['matched_positive_pairs']), 'CREOneMaxSD': f"{sc['max_within_recipe_sd']:.4f}",
        'CREOneReplDiff': sgn(rep['difference'], 4), 'CREOneReplCI': ci(*rep['ci'], 4),
        'CREOneTokAP': f'{tA / 1e6:.1f}', 'CREOneTokAR': f'{tR / 1e6:.1f}', 'CREOneTokRatio': f'{tA / tR:.2f}',
        'CREOneHistWelch': sgn(e1['fixed_time_vs_history']['welch_e1_minus_history']['difference'], 4),
        'CREOneResamples': str(e1['crossover_bootstrap']['heldout_bpb']['resamples']),
        # E1, exploratory decomposition (held-out unless named Val)
        'CREOneExposure': sgn(dh['exposure_on_AR_curve']['mean'], 3),
        'CREOneExposureCI': ci(*dh['exposure_on_AR_curve']['ci'], 3),
        'CREOneArch': sgn(dh['architecture_at_AP_exposure']['mean'], 3),
        'CREOneArchCI': ci(*dh['architecture_at_AP_exposure']['ci'], 3),
        'CREOneSched': sgn(dh['schedule_residual']['mean'], 3),
        'CREOneExposureVal': sgn(dv['exposure_on_AR_curve']['mean'], 3),
        'CREOneArchVal': sgn(dv['architecture_at_AP_exposure']['mean'], 3),
        'CREOneRho': f"{dh['throughput_ratio_rho']['mean']:.2f}",
        'CREOneK': f"{dh['sample_efficiency_multiple_k']['mean']:.2f}",
        'CREOneKCI': f"[{dh['sample_efficiency_multiple_k']['ci'][0]:.2f}, {dh['sample_efficiency_multiple_k']['ci'][1]:.2f}]",
        'CREOneKVal': f"{dv['sample_efficiency_multiple_k']['mean']:.2f}",
        # fixed-time history (seeds 0-2) for the table
        'CRFixedTokAR': f"{pn['fixed_tokens_ar'] / 1e6:.1f}", 'CRFixedTokAPQ': f"{pn['fixed_tokens_apq'] / 1e6:.1f}",
        # two agents, one contract
        'CRAgBaseBPB': f"{b['val_bpb']:.4f}", 'CRAgBaseTok': f"{b['tokens'] / 1e6:.1f}",
        'CRAgBaseMem': f"{b['peak_mib']:.0f}", 'CRAgBaseDecode': f"{b['decode_tok_s']:.0f}",
        'CRAgEtoERef': f"{b['e2e_ref_tok_s']:.0f}", 'CRAgEtoERefRep': f"{b['e2e_rep_ref_tok_s']:.0f}",
        'CRAgPrefillRef': f"{b['prefill_ref_ms']:.1f}",
        'CROpStableRepeats': str(sum(w['stable_repeats'] for w in W.values())),
        'CROpRepeats': str(sum(w['repeats'] for w in W.values())),
        'CROpAttempts': str(op['attempts']), 'CROpValid': str(op['valid']), 'CROpValidFrozen': str(op['valid_frozen']),
        'CROpUncertain': str(op['speed_uncertain']),
        'CROpBestBPB': f"{op['best']['val_bpb']:.4f}", 'CROpBestAttempt': str(op['best']['attempt']),
        'CROpBestFrozen': f"{op['best_frozen']['val_bpb']:.4f}",
        'CROpBestTwentyTwo': f"{op['best_first22']['val_bpb']:.4f}",
        'CROpBestTwentyTwoAttempt': str(op['best_first22']['attempt']),
        # where Opus's quality came from: best by A7 (batch and depth edits only) vs its overall best
        'CROpBestSeven': f"{best(D['agents']['opus']['rows'], 7)['val_bpb']:.4f}",
        'CROpLateGain': f"{best(D['agents']['opus']['rows'], 7)['val_bpb'] - op['best']['val_bpb']:.4f}",
        'CROpUpdLo': f"{min(r['updates'] for r in D['agents']['opus']['rows'] if str(r['attempt']) in OPUS_WINNERS):,}",
        'CROpUpdHi': f"{max(r['updates'] for r in D['agents']['opus']['rows'] if str(r['attempt']) in OPUS_WINNERS):,}",
        'CROpLateDone': str(sum(1 for r in D['agents']['opus']['rows'] if r['attempt'] > 7 and r['valid'])),
        'CROpLateInf': str(sum(1 for r in D['agents']['opus']['rows'] if r['attempt'] > 7 and r['valid'] and (r['category'] or '').startswith('inference'))),
        'CROpMaxTok': f"{op['max_tokens'] / 1e6:.0f}", 'CROpTokLo': tok_lo, 'CROpTokHi': tok_hi,
        'CROpTokRatio': f"{statistics.mean(w['tokens'] for w in W.values()) / b['tokens']:.1f}",
        'CROpDecodeLo': dec_lo, 'CROpDecodeHi': dec_hi, 'CROpEtoELo': e2e_lo, 'CROpEtoEHi': e2e_hi,
        'CROpPrefillLo': pre_lo, 'CROpPrefillHi': pre_hi, 'CROpMemLo': mem_lo, 'CROpMemHi': mem_hi,
        'CROpHeldLo': held_lo, 'CROpHeldHi': held_hi,
        'CROpARHeld': f"{ag['matched_token_refs']['ar_ref_201M_heldout']:.3f}",
        'CROpAPQHeld': f"{ag['matched_token_refs']['ap_q_201M_heldout']:.3f}",
        'CROpTokFrac': f"{100 * statistics.mean(w['tokens'] for w in W.values()) / 201326592:.0f}",
        'CROpAdapterAttempt': str(fa['attempt']),
        'CROpPreAdapterDecode': f"{pa['decode_x']:.1f}", 'CROpPreAdapterPrefill': f"{pa['prefill_ms']:.0f}",
        'CROpAdapterDecode': f"{fa['decode_x']:.1f}", 'CROpAdapterPrefill': f"{fa['prefill_ms']:.0f}",
        'CROpDLambadaLo': f'{min(dlam):+.3f}', 'CROpDLambadaHi': f'{max(dlam):+.3f}',
        'CROpDBlimpLo': f'{min(dbl):+.3f}', 'CROpDBlimpHi': f'{max(dbl):+.3f}',
        'CROpDHeldLo': f'{-max(dho):.3f}', 'CROpDHeldHi': f'{-min(dho):.3f}',
        'CREOneHours': f"{e1['a40_hours']:.2f}",
        'CREOneTputAR': f"{e1['runs_tok_s']['AR-ref']['min'] / 1e3:.1f}--{e1['runs_tok_s']['AR-ref']['max'] / 1e3:.1f}k",
        'CREOneTputAP': f"{e1['runs_tok_s']['AP-Q']['min'] / 1e3:.1f}--{e1['runs_tok_s']['AP-Q']['max'] / 1e3:.1f}k",
        'CREZeroTputAR': f"{e1['runs_tok_s']['AR-ref']['e0'] / 1e3:.1f}k",
        'CREZeroTputAP': f"{e1['runs_tok_s']['AP-Q']['e0'] / 1e3:.1f}k",
        'CROpFrontierFrozen': ', '.join(f'A{a}' for a in op['frontier_frozen']),
        'CROpFrontierExtended': ', '.join(f'A{a}' for a in op['frontier_extended']),
        'CRCxAttempts': str(cx['attempts']), 'CRCxValid': str(cx['valid']), 'CRCxUncertain': str(cx['speed_uncertain']),
        'CRCxBestBPB': f"{cx['best']['val_bpb']:.4f}", 'CRCxBestAttempt': str(cx['best']['attempt']),
        'CRCxBestTok': f"{cx['best']['tokens'] / 1e6:.1f}", 'CRCxMaxTok': f"{cx['max_tokens'] / 1e6:.0f}",
        'CRCxBestDecode': f"{cx['best']['decode_x']:.2f}", 'CRCxMaxDecode': f"{cx['max_decode_x']:.2f}",
        'CRCxMaxPrefill': f"{cx['max_prefill_ms']:.0f}",
    }
    M.update(macros_phase5(D))
    B = bench(D)
    M['CRBenchSDMax'] = f"{B['max_acc_sd_pts']:.1f}"
    M['CRBenchSDMed'] = f"{B['median_acc_sd_pts']:.1f}"
    out = ['% Generated by analysis/cr_v2.py from paper/camera_ready/data/cr_v2.json. Do not edit by hand.']
    out += [f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in M.items()]
    (GEN_CR / 'cr_macros_v2.tex').write_text('\n'.join(out) + '\n')
    return M


def table_e1(D):
    e1, pn = D['e1'], D['panel']
    H, V = e1['per_cell']['heldout_bpb'], e1['per_cell']['validation_bpb']
    cur = e1['curves']
    tok = lambda rec, c: statistics.mean(cur[f'{rec}|{c}']['tokens']) / 1e6
    wins = lambda vals: f"{sum(v < 0 for v in vals)}/3"
    ft, mh, mv = pn['fixed_time_apq_minus_ar'], pn['matched_heldout'], pn['matched_val']
    rows = [
        f"AP-Q & 300\\,s, re-runs & 0--2 & {pn['fixed_tokens_ar'] / 1e6:.1f} / {pn['fixed_tokens_apq'] / 1e6:.1f} & n/a & "
        f"\\textbf{{{sgn(ft['mean'], 4)}}} {ci(*ft['ci95'], 4)} & {wins(ft['diffs'])} \\\\",
        f"AP-Q & 300\\,s (E1) & 6--8 & {tok('AR-ref', 'time_300s'):.1f} / {tok('AP-Q', 'time_300s'):.1f} & "
        f"\\textbf{{{sgn(H['time_300s']['mean'], 4)}}} {ci(*H['time_300s']['ci'], 4)} & "
        f"{sgn(V['time_300s']['mean'], 4)} {ci(*V['time_300s']['ci'], 4)} & {wins(H['time_300s']['values'])} \\\\",
        '\\midrule',
    ]
    for c, lab in zip(CELLS[:3], ['88.6M tok. (E1)', '157.3M tok. (E1)', '201.3M tok. (E1)']):
        rows.append(f"AP-Q & {lab} & 6--8 & {tok('AR-ref', c):.1f} / {tok('AP-Q', c):.1f} & "
                    f"\\textbf{{{sgn(H[c]['mean'], 4)}}} {ci(*H[c]['ci'], 4)} & {sgn(V[c]['mean'], 4)} {ci(*V[c]['ci'], 4)} & "
                    f"{wins(H[c]['values'])} \\\\")
    rows.append(f"AP-Q & 201.3M tok. (panel) & 3--5 & 201.3 / 201.3 & \\textbf{{{sgn(mh['mean'], 4)}}} {ci(*mh['ci95'], 4)} & "
                f"{sgn(mv['mean'], 4)} {ci(*mv['ci95'], 4)} & {wins(mh['diffs'])} \\\\")
    rows.append('\\midrule')
    R = D['phase5']['matched_tokens']['recipes']
    for k in OPUS_KEYS:
        h, v = R[k]['vs_ar']['heldout'], R[k]['vs_ar']['validation']
        rows.append(f"Opus A{k[-2:]} & 201.3M tok. & 6--8 & 201.3 / 201.3 & \\textbf{{{sgn(h['mean'], 4)}}} {ci(*h['ci95'], 4)} & "
                    f"{sgn(v['mean'], 4)} {ci(*v['ci95'], 4)} & {wins(h['values'])} \\\\")
    s = ('% Generated by analysis/cr_v2.py (E1 e1_analysis.json; cr_stats.json; Phase 5 phase5_analysis.json). Do not edit by hand.\n'
         '\\begin{tabular}{lllcllc}\n\\toprule\n'
         'Recipe & Contract & Seeds & Tokens (M) & $\\Delta$ held-out [95\\% CI] & $\\Delta$ validation [95\\% CI] & Better \\\\\n'
         '\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')
    (GEN_CR / 'table_e1.tex').write_text(display_names(s))


def table_agents(D):
    ag = D['agents']
    b, op, cx = ag['baseline'], ag['opus'], ag['codex']
    W = op['winners']
    cb = cx['best']
    lines = [
        f"Registered baseline (calibration run) & n/a & {b['val_bpb']:.4f} & {b['tokens'] / 1e6:.1f} & 50.3 & 1.00 & 1.00 & {b['peak_mib']:.0f} \\\\",
        f"Codex, best (A{cb['attempt']}) & {cx['valid']}/{cx['attempts']} & {cb['val_bpb']:.4f} & {cb['tokens'] / 1e6:.1f} & "
        f"{cb['params'] / 1e6:.1f} & {cb['decode_x']:.2f} & {cb['e2e_x']:.2f} & {cb['peak_mib']:.0f} \\\\",
    ]
    for i, (a, w) in enumerate(W.items()):
        lead = f"{op['valid']}/{op['attempts']}" if i == 0 else ''
        lines.append(f"Opus 5.5, A{a} & {lead} & {w['val_bpb']:.4f} & {w['tokens'] / 1e6:.1f} & {w['params'] / 1e6:.1f} & "
                     f"{w['decode_x']:.1f} & {w['e2e_x']:.1f} & {w['peak_mib']:.0f} \\\\")
    s = ('% Generated by analysis/cr_v2.py (Pair-A T-arm histories; Phase 2 re-verification). Do not edit by hand.\n'
         '\\begin{tabular}{lcccrccr}\n\\toprule\n'
         ' & Valid / & Val BPB & Tokens & Params & \\multicolumn{2}{c}{Speed-up vs baseline} & Peak \\\\\n'
         '\\cmidrule(lr){6-7}\n'
         'Agent, attempt & attempts & $\\downarrow$ & in 300\\,s (M) & (M) & decode only & end to end & MiB \\\\\n'
         '\\midrule\n' + '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n')
    (GEN_CR / 'table_agents.tex').write_text(s)


BENCH = [('heldout', 'heldout_climbmix_bpb/bits_per_byte', 'heldout_climbmix_bpb.bits_per_byte'),
         ('arc_easy', 'arc_easy/accuracy', 'arc_easy.accuracy'),
         ('piqa', 'piqa/accuracy', 'piqa.accuracy'),
         ('lambada', 'lambada_openai/exact_match', 'lambada_openai.exact_match'),
         ('blimp', 'blimp/macro_accuracy', 'blimp.macro_accuracy')]


def bench(D):
    """Every recipe on every benchmark at 201.3M training tokens.
    Panel: seeds 3-5 (cr_evidence.json panel_downstream, cross-checked against cr_stats.json).
    Opus retraining (Phase 5A): E1's seeds 6-8 (phase5_analysis.json downstream.by_recipe)."""
    E, S = read_json(CR_DATA / 'cr_evidence.json'), read_json(CR_STATS)
    by = {}
    for r in E['panel_downstream']:
        by.setdefault((r['model'], f"{r['task']}/{r['metric']}"), {})[r['seed']] = r['value']
    panel = []
    for m in PANEL_ORDER:
        row = {'name': NAMES[m], 'params': S['table1'][m]['params'] / 1e6}
        for key, pk, _ in BENCH:
            vals = [by[(m, pk)][s] for s in (3, 4, 5)]
            assert abs(statistics.mean(vals) - S['downstream_means'][m][pk]['mean']) < 1e-12, (m, pk)
            row[key] = (statistics.mean(vals), statistics.stdev(vals))
        assert max(abs(x - y) for x, y in zip([by[(m, BENCH[0][1])][s] for s in (3, 4, 5)],
                                               S['table1'][m]['heldout_bpb']['values'])) < 1e-12, m
        panel.append(row)
    P = read_json(P5)['phase5a']
    br = P['downstream']['by_recipe']
    p5 = []
    for k in ('AR-ref', 'AP-Q') + OPUS_KEYS:
        row = {'name': k if not k.startswith('OPUS') else f'Opus A{k[-2:]}', 'params': P['recipes'][k]['parameter_count'] / 1e6}
        for key, _, qk in BENCH:
            row[key] = (br[k][qk]['mean'], br[k][qk]['sd'])
        assert abs(row['heldout'][0] - D['phase5']['matched_tokens']['recipes'][k]['heldout']['mean']) < 1e-12, k
        p5.append(row)
    acc_sd = [r[k][1] for r in panel + p5 for k in ('arc_easy', 'piqa', 'lambada', 'blimp')]
    return {'panel': panel, 'phase5': p5, 'max_acc_sd_pts': 100 * max(acc_sd),
            'median_acc_sd_pts': 100 * statistics.median(acc_sd)}


def table_bench(D):
    B = bench(D)
    cell = lambda key, v: f'{v[0]:.4f}' if key == 'heldout' else f'{100 * v[0]:.1f}'
    def block(title, rows):
        out = [r'\multicolumn{7}{l}{\emph{' + title + r'}} \\']
        for r in rows:
            out.append(r'\quad ' + f"{r['name']} & {r['params']:.1f} & " + ' & '.join(cell(k, r[k]) for k, _, _ in BENCH) + r' \\')
        return out
    lines = block('Panel (seeds 3--5)', B['panel']) + ['\\midrule'] + block('Opus retraining (E1 seeds 6--8)', B['phase5'])
    s = ('% Generated by analysis/cr_v2.py (cr_evidence.json panel_downstream; Phase 5 phase5_analysis.json). Do not edit by hand.\n'
         '\\begin{tabular}{lrrrrrr}\n\\toprule\n'
         ' & Params & Held-out & ARC-Easy & PIQA & LAMBADA & BLiMP \\\\\n'
         'Recipe & (M) & BPB $\\downarrow$ & acc.\\,$\\uparrow$ & acc.\\,$\\uparrow$ & exact\\,$\\uparrow$ & acc.\\,$\\uparrow$ \\\\\n'
         '\\midrule\n' + '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n')
    (GEN_CR / 'table_bench.tex').write_text(s)
    return B


def table_reasoning(D):
    """Verbatim excerpts of the agents' recorded reasoning next to the measured outcome of the same attempt.
    Each excerpt is asserted to occur verbatim in that attempt's record (hypothesis, intended change or patch);
    outcomes are read from the same records (parent -> attempt)."""
    hist = {x['id']: x for x in read_json(ROOT / 'paper/generated/historical_experiments.json')}
    def pair_rec(root, a):
        for line in (root / 'history.jsonl').read_text().splitlines():
            d = json.loads(line)
            if d['campaign_attempt'] == a:
                return '\n'.join((d.get('hypothesis') or '', d.get('intended_change') or '', d.get('candidate_patch') or ''))
        raise KeyError(a)
    O = {r['attempt']: r for r in D['agents']['opus']['rows']}
    C = {r['attempt']: r for r in D['agents']['codex']['rows']}
    B = D['agents']['baseline']
    b_upd = B['tokens'] / 2 ** 19
    assert abs(b_upd - round(b_upd)) < 1e-9
    m = lambda x: f'{x / 1e6:.0f}M'
    tr = lambda p, c: f"BPB {p['val_bpb']:.4f}$\\to${c['val_bpb']:.4f}; tokens {m(p['tokens'])}$\\to${m(c['tokens'])}"
    th = lambda p, c: (f"BPB {hist[p]['val_bpb']:.4f}$\\to${hist[c]['val_bpb']:.4f}; tokens "
                       f"{m(hist[p]['total_training_tokens'])}$\\to${m(hist[c]['total_training_tokens'])}")
    rows = [
        ('AutoPareto A1', hist[113]['hypothesis'], 'roughly four times as many optimizer updates within the fixed training window',
         th(8, 113) + ' (largest single gain on the path)'),
        ('AutoPareto A7', hist[119]['hypothesis'], 'the additional optimization steps possible within the fixed training budget will keep validation BPB competitive',
         th(118, 119)),
        ('AutoPareto A8', hist[120]['hypothesis'], 'the smaller model processes more tokens in the fixed training time',
         th(119, 120) + ' (width 256: prediction failed; A18 restored width 512)'),
        ('Codex A3', pair_rec(CODEX_ROOT, 3), 'the resulting higher token throughput within Contract T should partly offset the explicit risk of worse BPB',
         tr(C[2], C[3])),
        ('Codex A4', pair_rec(CODEX_ROOT, 4), 'the additional trained tokens may offset reduced capacity', tr(C[3], C[4])),
        ('Opus A1 (code comment)', pair_rec(OPUS_ROOT, 1), '2x more updates in the fixed 300 s budget',
         f"updates {b_upd:.0f}$\\to${O[1]['updates']}; BPB {B['val_bpb']:.4f}$\\to${O[1]['val_bpb']:.4f}"),
        ('Opus A3', pair_rec(OPUS_ROOT, 3), 'Depth 6 at width 512 should give about 1.2x more updates and tokens',
         f"tokens $\\times${O[3]['tokens'] / O[1]['tokens']:.2f}; BPB {O[1]['val_bpb']:.4f}$\\to${O[3]['val_bpb']:.4f}"),
        ('Opus A7', pair_rec(OPUS_ROOT, 7), 'Smaller, higher-loss models have smaller critical batch sizes',
         f"predicted BPB 1.103--1.114; measured {O[7]['val_bpb']:.4f}; updates {O[5]['updates']}$\\to${O[7]['updates']}"),
        ('Opus A10', pair_rec(OPUS_ROOT, 10), 'CUDA-graph replay is the only lever that can raise speed without a quality cost',
         f"decode-only {O[8]['decode_x']:.1f}$\\times{{\\to}}${O[10]['decode_x']:.1f}$\\times$; prefill {O[8]['prefill_ms']:.0f}$\\to${O[10]['prefill_ms']:.0f} ms"),
        ('Opus A11', pair_rec(OPUS_ROOT, 11), 'Prefill time rose from 6.4 ms to 81 ms (graph capture per prefill)',
         'capture stayed in prefill through A27 (\\CRIACapture\\% of frontier prefill)'),
        ('Opus A21', pair_rec(OPUS_ROOT, 21), 'Prefill/TTFT: may drop slightly (diagnostic only)', 'prefill is recorded but not scored'),
        ('Opus A26 (on A25)', pair_rec(OPUS_ROOT, 26), 'training throughput collapsed 657k→360k tok/s (825 vs 1,504 updates)',
         f"A25 BPB {O[25]['val_bpb']:.4f} on {m(O[25]['tokens'])} tokens; A26 kept A23's training path"),
    ]
    out = []
    for who, rec, quote, outcome in rows:
        assert quote in rec, (who, quote)
        assert '\u2014' not in quote and '\u2013' not in quote
        q = quote.replace('→', '$\\to$').replace('%', '\\%')
        out.append(f"{who} & ``{q}'' & {outcome} \\\\")
    s = ('% Generated by analysis/cr_v2.py (table_reasoning): excerpts asserted verbatim against the attempt records. Do not edit by hand.\n'
         '\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{2.1cm}>{\\raggedright\\arraybackslash}p{6.1cm}>{\\raggedright\\arraybackslash}p{4.9cm}@{}}\n\\toprule\n'
         'Agent, attempt & Recorded text (verbatim excerpt) & Measured outcome (parent$\\to$attempt) \\\\\n\\midrule\n'
         + '\n'.join(out) + '\n\\bottomrule\n\\end{tabular}\n')
    (GEN_CR / 'table_reasoning.tex').write_text(s)


def table_env_v2():
    """The panel-era environment table plus an E1 row read from the E1 stage manifest."""
    import re
    man = read_json(next(E1_DIR.glob('stage_manifest-*.json')))
    env = man['environment']['training_interpreter']
    det = next(c['detail'] for c in man['checks'] if c['check'] == 'gpu_idle')
    driver = re.search(r'NVIDIA A40, ([0-9.]+)', det).group(1)
    cuda = env['torch'].split('+cu')[1]
    # the two reported Pair-A arms: their own run manifests (runtime block), read-only under the evidence root
    row = ''
    for lab, man_path in (('Pair-A Codex arm, continuation (Sep 16)', CODEX_ROOT.parent / 'provenance/final_pair_a_continuation_manifest_20260916_r2.json'),
                          ('Pair-A Opus 5.5 arm (Nov)', OPUS_ROOT.parent / 'provenance/claude_t_continuation_manifest_20261105c.json')):
        rt = read_json(man_path)['runtime']
        row += f"{lab} & {rt['python_version']} & {rt['torch']} & {rt['cuda']} & not recorded \\\\\n"
    row += (f"E1 token sweep, 24 runs (Nov 7) & {env['python']} & {env['torch']} & {cuda[:-1]}.{cuda[-1]} & {driver} \\\\\n")
    # Phase 5: the 5A stage manifest (same probe as E1) and the 5B kernel-census records plus the 5B driver log
    p5 = ROOT / 'results/camera_ready_2026/phase5'
    m5 = read_json(next((p5 / 'A').glob('stage_manifest-*.json')))['e1_environment']
    m5env = m5['environment']
    env5 = m5env['training_interpreter']
    det5 = next(c['detail'] for c in m5['checks'] if c['check'] == 'gpu_idle')
    drv5 = re.search(r'NVIDIA A40, ([0-9.]+)', det5).group(1)
    cu5 = env5['torch'].split('+cu')[1]
    row += (f"Opus retraining, 10 runs (Nov 8) & {env5['python']} & {env5['torch']} & {cu5[:-1]}.{cu5[-1]} & {drv5} \\\\\n")
    tv = {read_json(f)['torch'] for f in (p5 / 'B/kernel_census').glob('*.json')}
    assert len(tv) == 1, tv
    tb = tv.pop()
    cub = tb.split('+cu')[1]
    drvb = re.search(r'GPU-[0-9a-f-]+, ([0-9.]+);', (p5 / 'B/driver.log').read_text().splitlines()[0]).group(1)
    row += (f"Inference ablations, 39 runs (Nov 8) & not recorded & {tb} & {cub[:-1]}.{cub[-1]} & {drvb} \\\\\n")
    # 13-recipe test (Phase 6, Nov 16-17), read from its own run records
    p6 = Path(__file__).resolve().parents[1] / 'results/camera_ready_2026/phase6'
    if p6.is_dir():
        tr = [read_json(f) for f in sorted(p6.glob('runs/p6-*/run_metadata.json'))]
        sv = {(m['software_versions']['python'], m['software_versions']['torch']) for m in tr}
        recs = [read_json(f) for f in sorted(p6.glob('runs/p6-*/phase6_run_record.json'))]
        first = {}
        for r in sorted(recs, key=lambda r: r['finished_utc']):
            for g in r['gpu']:
                first.setdefault(g.split(', ')[-1], r['finished_utc'])
        assert len(sv) == 1 and 1 <= len(first) <= 2, (sv, first)
        (py6, t6) = sv.pop()
        drivers6 = sorted(first, key=first.get)  # a host restart on Nov 16-17 changed the driver
        d6 = drivers6[0] + (r'$^\dagger$' if len(drivers6) > 1 else '')
        row += (f"13-recipe training, {sum(r['integrity']['passed'] for r in recs)} runs (Nov 16--17) & {py6} & {t6} & {t6.split('+cu')[1][:-1]}.{t6.split('+cu')[1][-1]} & {d6} \\\\\n")
        inf = [dict(x['environment'], _t=x.get('finished_utc', '')) for x in
               (read_json(f) for f in sorted(p6.glob('inference/runs/*.json')) if not f.name.endswith('-attempt2.json'))]
        iv = {(e['python'], e['torch'], e['cuda_runtime']) for e in inf}
        assert len(iv) == 1, iv
        ipy, it, icu = iv.pop()
        idrvs = list(dict.fromkeys(e['nvidia_smi'].split(', ')[2] for e in sorted(inf, key=lambda e: e.get('_t', ''))))
        idrv = idrvs[0] + (r'$^\dagger$' if len(idrvs) > 1 else '')
        row += (f"13-recipe inference, {len(inf)} runs (Nov 16--17) & {ipy} & {it} & {icu} & {idrv} \\\\\n")
        lm = [read_json(f) for f in sorted(p6.glob('lmeval/runs/*.json'))]
        lv = {(x['python'], x['torch'], x['lm_eval_version']) for x in lm}
        assert len(lv) == 1, lv
        lpy, lt, lver = lv.pop()
        row += (f"13-recipe lm-eval {lver}, {len(lm)} runs & {lpy} & {lt} & {lt.split('+cu')[1][:-1]}.{lt.split('+cu')[1][-1]} & not recorded \\\\\n")
    src = (GEN_CR / 'table_env.tex').read_text()
    assert '\\bottomrule' in src
    out = src.replace('% Generated by analysis/cr_tables.py', '% Generated by analysis/cr_v2.py from table_env.tex (cr_tables.py) plus the E1 and Phase 5 records;', 1)
    out = out.replace('\\bottomrule', row + '\\bottomrule', 1)
    (GEN_CR / 'table_env_v2.tex').write_text(out)


# ----------------------------------------------------------------------------- figures
def style(scale=1.0):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.0 * scale, 'axes.labelsize': 7.0 * scale,
                         'axes.titlesize': 7.5 * scale, 'xtick.labelsize': 6.5 * scale, 'ytick.labelsize': 6.5 * scale,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': INK2,
                         'axes.linewidth': 0.6, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.labelcolor': INK,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})


def save(fig, name, poster=False):
    meta = {'Creator': 'AutoPareto camera-ready analysis (analysis/cr_v2.py)', 'Author': ''}
    tgt = POSTER if poster else FIG
    tgt.mkdir(parents=True, exist_ok=True)
    fig.savefig(tgt / f'{name}.pdf', bbox_inches='tight', pad_inches=0.03, metadata=meta)
    fig.savefig(tgt / f'{name}.png', dpi=300 if poster else 220, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)


def draw_e1(ax, D, title='(a) Token response, seeds 6–8; diamonds: 300 s'):
    cur = D['e1']['curves']
    for rec, col in (('AR-ref', ORANGE), ('AP-Q', BLUE)):
        xs = [statistics.mean(cur[f'{rec}|{c}']['tokens']) / 1e6 for c in CELLS[:3]]
        ys = [statistics.mean(cur[f'{rec}|{c}']['heldout']) for c in CELLS[:3]]
        for c in CELLS[:3]:
            ax.scatter([t / 1e6 for t in cur[f'{rec}|{c}']['tokens']], cur[f'{rec}|{c}']['heldout'], s=8,
                       facecolors='none', edgecolors=col, linewidths=0.6, zorder=3)
        ax.plot(xs, ys, color=col, lw=1.5, zorder=2)
        ax.scatter(xs, ys, s=14, color=col, zorder=4, edgecolors='white', linewidths=0.6)
        tx = [t / 1e6 for t in cur[f'{rec}|time_300s']['tokens']]
        ty = cur[f'{rec}|time_300s']['heldout']
        ax.scatter([statistics.mean(tx)], [statistics.mean(ty)], marker='D', s=26, color=col, zorder=5,
                   edgecolors=INK, linewidths=0.7)
        ax.annotate(rec, (xs[-1], ys[-1]), xytext=(4, 0), textcoords='offset points', va='center', color=INK)
        ax.annotate('300 s', (statistics.mean(tx), statistics.mean(ty)),
                    xytext=(6, -8) if rec == 'AR-ref' else (5, 6), textcoords='offset points',
                    ha='left', color=INK2, fontsize=plt.rcParams['font.size'] * 0.9)
    ax.set_xscale('log')
    ax.set_xticks([88.6, 157.3, 201.3])
    ax.set_xticklabels(['88.6M', '157.3M', '201.3M'])
    ax.minorticks_off()
    ax.set_xlim(78, 275)
    ax.set_xlabel('Training tokens (log scale)')
    ax.set_ylabel('Held-out BPB (lower is better)')
    ax.grid(axis='y', color=GRID, lw=0.5)
    ax.set_title(title, loc='left', color=INK)


def draw_split(bx, D, title='(b) 300 s gap split (exploratory)'):
    dh = D['e1']['decomposition']['heldout_bpb']
    comps = [('Extra tokens', dh['exposure_on_AR_curve'], INK2),
             ('Architecture,\nequal tokens', dh['architecture_at_AP_exposure'], INK2),
             ('Schedule', dh['schedule_residual'], INK2),
             ('Observed\n300 s gap', dh['gap_300s'], None)]
    for i, (lab, st, col) in enumerate(comps):
        y = len(comps) - 1 - i
        if col:
            bx.barh(y, st['mean'], height=0.56, color=col, zorder=2)
        else:
            bx.barh(y, st['mean'], height=0.56, facecolor='white', edgecolor=INK, linewidth=0.8, zorder=2)
        bx.plot(st['ci'], [y, y], color=INK, lw=0.9, zorder=3)
        for v in st['values']:
            bx.scatter(v, y, s=6, facecolors='white', edgecolors=INK, linewidths=0.5, zorder=4)
        off = 4 if st['mean'] >= 0 else -4
        bx.annotate(sgn(st['mean'], 3).replace('$-$', '−'), (st['ci'][1] if st['mean'] >= 0 else st['ci'][0], y),
                    xytext=(off, 0), textcoords='offset points', va='center', ha='left' if off > 0 else 'right',
                    color=INK)
    bx.axvline(0, color=INK2, lw=0.7)
    bx.set_yticks(range(len(comps)))
    bx.set_yticklabels([c[0] for c in comps][::-1])
    bx.set_xlim(-0.088, 0.078)
    bx.set_xticks([-0.04, 0, 0.04])
    bx.set_xticklabels(['−0.04', '0', '+0.04'])
    bx.set_xlabel('Component of AP-Q − AR-ref held-out BPB')
    bx.grid(axis='x', color=GRID, lw=0.5)
    bx.spines['left'].set_visible(False)
    bx.tick_params(axis='y', length=0)
    bx.set_title(title, loc='left', color=INK)


def arms_of(D):
    ag = D['agents']
    return (('Codex', ag['codex']['rows'], AQUA), ('Opus', ag['opus']['rows'], VIOLET))


def draw_tokens(ax, D, title='(a) Training tokens per 300 s attempt'):
    b = D['agents']['baseline']
    for name, rows, col in arms_of(D):
        v = [r for r in rows if r['valid']]
        ax.plot([r['attempt'] for r in v], [r['tokens'] / 1e6 for r in v], color=col, lw=1.3, marker='o', ms=2.6,
                zorder=3)
        label = 'Claude Opus 5.5' if name == 'Opus' else 'Codex (gpt-5.6-sol)'
        ax.annotate(label, (v[-1]['attempt'], v[-1]['tokens'] / 1e6), xytext=(4, 0), textcoords='offset points',
                    va='center', color=INK, fontsize=plt.rcParams['font.size'] * 0.95)
    ax.axhline(b['tokens'] / 1e6, color=GRAY, lw=0.8, ls='--', zorder=1)
    ax.annotate('registered baseline', (36, b['tokens'] / 1e6), xytext=(0, 3), textcoords='offset points',
                ha='right', va='bottom', color=INK2, fontsize=plt.rcParams['font.size'] * 0.9)
    ax.set_xlabel('Attempt (valid attempts only)')
    ax.set_ylabel('Tokens trained in 300 s (M)')
    ax.set_xlim(0, 36)
    ax.set_xticks([1, 5, 10, 15, 20, 25])
    ax.grid(axis='y', color=GRID, lw=0.5)
    ax.set_title(title, loc='left', color=INK)


def draw_speeds(bx, D, title='(b) Decode-only (solid) vs end-to-end (dashed)', compact=False):
    ag = D['agents']
    for name, rows, col in arms_of(D):
        v = [r for r in rows if r['valid'] and r['decode_x'] and r['e2e_x']]
        bx.plot([r['attempt'] for r in v], [r['decode_x'] for r in v], color=col, lw=1.3, zorder=3)
        bx.plot([r['attempt'] for r in v], [r['e2e_x'] for r in v], color=col, lw=1.1, ls=(0, (3, 1.6)), zorder=3)
    fs = plt.rcParams['font.size'] * 0.95
    op = [r for r in ag['opus']['rows'] if r['valid']]
    bx.annotate('Opus,\ndecode only' if compact else 'Opus, decode only', (op[-1]['attempt'], op[-1]['decode_x']),
                xytext=(3, 0), textcoords='offset points', va='center', color=INK, fontsize=fs)
    bx.annotate('Opus,\nend to end' if compact else 'Opus, end to end', (op[-1]['attempt'], op[-1]['e2e_x']),
                xytext=(3, 0), textcoords='offset points', va='center', color=INK, fontsize=fs)
    cv = [r for r in ag['codex']['rows'] if r['valid']]
    bx.annotate('Codex,\nboth' if compact else 'Codex, both', (cv[-1]['attempt'], cv[-1]['decode_x']),
                xytext=(3, -2), textcoords='offset points', va='center', color=INK, fontsize=fs)
    fa = ag['opus']['first_adapter_attempt']
    bx.axvline(fa, color=GRAY, lw=0.7, ls=':')
    bx.annotate(f'A{fa}: graph capture\nin untimed prefill' if compact else
                f'A{fa}: decode graph\ncaptured in untimed prefill', (fa, 2.45), xytext=(3, 0),
                textcoords='offset points', ha='left', va='center', color=INK2,
                fontsize=plt.rcParams['font.size'] * 0.85)
    bx.set_yscale('log')
    bx.set_yticks([1, 2, 5, 10, 30])
    bx.set_yticklabels(['1×', '2×', '5×', '10×', '30×'])
    bx.minorticks_off()
    bx.set_xlim(0, 45 if compact else 39)
    bx.set_xticks([1, 10, 20] if compact else [1, 5, 10, 15, 20, 25])
    bx.set_xlabel('Attempt (valid only)' if compact else 'Attempt (valid attempts only)')
    bx.set_ylabel('Speed-up vs baseline')
    bx.grid(axis='y', color=GRID, lw=0.5)
    bx.set_title(title, loc='left', color=INK)


def draw_ablation(cx, D, title='(c) Opus inference ablations', compact=True):
    """5B: decode-only (filled) and end-to-end (hollow) speed-up over the same-session baseline, per variant.
    Marker: median of the three checkpoints; bar: their range."""
    sp = D['phase5']['ablation']['speedup']
    n = len(IA_VARIANTS)
    for i, (v, lab) in enumerate(IA_VARIANTS):
        y = n - 1 - i
        # end to end: larger hollow ring underneath; decode: smaller filled dot on top (coincident = ringed dot)
        for kind, face, size, z in (('e2e', 'white', 24, 3), ('decode', VIOLET, 11, 4)):
            xs = [sp[m][v][kind] for m in IA_MODELS]
            cx.plot([min(xs), max(xs)], [y, y], color=VIOLET, lw=1.0, zorder=2, solid_capstyle='butt')
            cx.scatter([statistics.median(xs)], [y], s=size, marker='o', facecolors=face, edgecolors=VIOLET,
                       linewidths=0.9, zorder=z)
        if v == 'V0_as_submitted':
            fs = plt.rcParams['font.size'] * 0.85
            for kind, nd in (('decode', 0), ('e2e', 1)):
                xm = statistics.median(sp[m][v][kind] for m in IA_MODELS)
                cx.annotate(f'{xm:.{nd}f}×', (xm, y), xytext=(0, 4), textcoords='offset points', ha='center',
                            va='bottom', color=INK, fontsize=fs)
    cx.axvline(1, color=GRAY, lw=0.7, ls='--', zorder=1)
    cx.set_xscale('log')
    cx.set_xticks([1, 2, 5, 10, 30])
    cx.set_xticklabels(['1×', '2×', '5×', '10×', '30×'])
    cx.minorticks_off()
    cx.set_xlim(0.8, 52)
    cx.set_ylim(-0.7, n - 0.35)
    cx.set_yticks(range(n))
    cx.set_yticklabels([lab for _, lab in IA_VARIANTS][::-1])
    cx.tick_params(axis='y', length=0)
    cx.spines['left'].set_visible(False)
    cx.grid(axis='x', color=GRID, lw=0.5)
    cx.set_xlabel('Speed-up vs baseline (log)')
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls='none', marker='o', ms=2.9, mfc=VIOLET, mec=VIOLET, label='decode only'),
         Line2D([], [], ls='none', marker='o', ms=4.2, mfc='white', mec=VIOLET, label='end to end')]
    cx.legend(handles=h, loc='lower right', bbox_to_anchor=(1.0, 0.0), frameon=False,
              fontsize=plt.rcParams['font.size'] * 0.85, handletextpad=0.1, borderaxespad=0.1, labelspacing=0.2)
    cx.set_title(title, loc='left', color=INK)


def fig_ablation(D, poster=False):
    style(1.45 if poster else 1.0)
    fig, ax = plt.subplots(figsize=(3.6, 2.0) if not poster else (6.0, 3.4))
    draw_ablation(ax, D, 'Opus 5.5 inference ablations (3 checkpoints)')
    fig.tight_layout()
    save(fig, 'cr_ablation', poster)


def _ancestry(parent_of, start):
    """Attempt numbers from the first edit to `start`, following recorded parent links (baseline excluded)."""
    path, a = [], start
    while a is not None:
        path.append(a)
        a = parent_of.get(a)
    return path[::-1]


def _pair_a_parents(root):
    out = {}
    for line in (root / 'history.jsonl').read_text().splitlines():
        d = json.loads(line)
        p = d.get('parent_experiment_id') or ''
        out[d['campaign_attempt']] = int(p.rsplit('-', 1)[1]) if '-attempt-' in p else None
    return out


def fig_paths(D):
    """Part 1 Figure 1: validation BPB against tokens trained in 300 s, with each campaign's recorded
    parent path from the baseline to its best recipe.  (a) historical AutoPareto seed 0 (AP-Q lineage);
    (b) Pair-A seed 3, Codex and Opus.  All values are read from the attempt records."""
    style(0.9)
    hist = read_json(ROOT / 'paper/generated/historical_experiments.json')
    seed0 = {x['id']: x for x in hist if x['campaign_id'] == 'autopareto-seed0-20260731'}
    hbase = next(x for x in hist if x['id'] == 8)
    rows_h = {**seed0, 8: hbase}
    path, i = [], 134                                     # AP-Q = exp 134 (A22)
    while i is not None:
        path.append(i)
        i = rows_h[i]['parent_experiment_id'] if i != 8 else None
    path = path[::-1]
    assert path == [8, 113, 116, 118, 119, 120, 130, 131, 132, 134], path   # the lineage figure's main path
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.62, 1.78))
    tok = lambda x: x['total_training_tokens'] / 1e6
    fs = plt.rcParams['font.size'] * 0.92

    def draw_path(a, xs, ys, col):
        a.plot(xs, ys, color=col, lw=0.9, zorder=3)
        for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
            if abs(x1 - x0) > 15 or abs(y1 - y0) > 0.012:   # arrowheads only on large moves
                a.annotate('', xy=(x1, y1), xytext=((x0 + x1) / 2, (y0 + y1) / 2),
                           arrowprops={'arrowstyle': '-|>', 'mutation_scale': 6.5, 'lw': 0, 'color': col}, zorder=3)
        a.scatter(xs, ys, s=11, color=col, zorder=4)

    def note(a, t, xy, xyt, col=INK, bold=False, ha='left'):
        a.annotate(t, xy, xytext=xyt, textcoords='data', ha=ha, va='center', fontsize=fs, color=col,
                   fontweight='bold' if bold else 'normal',
                   arrowprops={'arrowstyle': '-', 'lw': 0.5, 'color': MUTED, 'shrinkA': 1, 'shrinkB': 3})

    others = [x for k, x in seed0.items() if k not in path and x.get('status') == 'completed' and x.get('val_bpb')]
    ax.scatter([tok(x) for x in others], [x['val_bpb'] for x in others], s=8, facecolor='white', edgecolor=MUTED, lw=0.6, zorder=2)
    P = {k: (tok(rows_h[k]), rows_h[k]['val_bpb']) for k in path}
    draw_path(ax, [P[k][0] for k in path], [P[k][1] for k in path], BLUE)
    ax.scatter(*P[8], s=14, color=INK, zorder=5)
    note(ax, 'B: baseline', P[8], (110, 1.2137))
    note(ax, 'A1: batch ÷4 (4× updates)', P[113], (100, 1.165))
    note(ax, 'A4–A7: depth 8→5', P[118], (118, 1.135))
    note(ax, 'A8: width 256', P[120], (196, 1.185), ha='right')
    note(ax, 'A18–A22: width 512,\nLR tweaks → AP-Q', P[134], (209, 1.127), BLUE, True)
    ax.set_ylim(1.086, 1.224)
    ax.set_xlim(75, 268)
    ax.set_title('(a) AutoPareto, seed 0: path to AP-Q', loc='left', color=INK)
    base = D['agents']['baseline']
    B0 = (base['tokens'] / 1e6, base['val_bpb'])
    ends = {}
    for name, rows, col in (('Codex', D['agents']['codex']['rows'], AQUA), ('Opus', D['agents']['opus']['rows'], VIOLET)):
        par = _pair_a_parents(CODEX_ROOT if name == 'Codex' else OPUS_ROOT)
        valid = {r['attempt']: r for r in rows if r['valid']}
        b = min(valid.values(), key=lambda r: r['val_bpb'])['attempt']
        anc = _ancestry(par, b)
        assert all(x in valid for x in anc), (name, anc)
        oth = [r for k, r in valid.items() if k not in anc]
        bx.scatter([r['tokens'] / 1e6 for r in oth], [r['val_bpb'] for r in oth], s=8, facecolor='white', edgecolor=col, lw=0.6, alpha=0.7, zorder=2)
        xs = [B0[0]] + [valid[k]['tokens'] / 1e6 for k in anc]
        ys = [B0[1]] + [valid[k]['val_bpb'] for k in anc]
        draw_path(bx, xs, ys, col)
        ends[name] = {k: (valid[k]['tokens'] / 1e6, valid[k]['val_bpb']) for k in anc}
        ends[name]['best'] = b
        print(f'fig_paths: {name} path to A{b}:', anc)
    bx.scatter(*B0, s=14, color=INK, zorder=5)
    note(bx, 'B: baseline', B0, (112, 1.2))
    E = ends['Codex']
    note(bx, 'Codex A12:\nbatch ÷2', E[12], (78, 1.135), AQUA)
    note(bx, f"Codex best (A{E['best']})", E[E['best']], (78, 1.093), AQUA, True)
    E = ends['Opus']
    note(bx, 'Opus A3–A5: depth 8→3', E[4], (170, 1.165), VIOLET)
    note(bx, f"Opus best (A{E['best']})", E[E['best']], (205, 1.125), VIOLET, True)
    bx.set_ylim(1.084, 1.212)
    bx.set_xlim(75, 255)
    bx.set_title('(b) Pair-A, seed 3: Codex and Opus', loc='left', color=INK)
    for a in (ax, bx):
        a.set_xlabel('Tokens trained in 300 s (M)')
        a.grid(color=GRID, lw=0.5)
    ax.set_ylabel('Validation BPB ↓')
    fig.tight_layout(w_pad=1.0)
    save(fig, 'cr_paths')


def fig_main(D):
    """Part 1 figure: (a) E1 token response, (b) exploratory 300 s split, (c) Opus inference ablations (5B)."""
    style(0.9)  # drawn at print size (text width 5.5 in): 6.3 pt text, 5.9 pt ticks
    fig, axs = plt.subplots(1, 3, figsize=(5.62, 1.48), gridspec_kw={'width_ratios': [0.95, 1.0, 1.05]})
    draw_e1(axs[0], D, '(a) E1, seeds 6–8')
    axs[0].set_xticklabels(['88.6', '157', '201'])
    draw_split(axs[1], D, '(b) 300 s gap split')
    axs[1].set_xlim(-0.112, 0.088)
    draw_ablation(axs[2], D, '(c) Opus inference ablations')
    axs[0].set_ylabel('Held-out BPB ↓')
    from matplotlib.ticker import FormatStrFormatter, MultipleLocator
    axs[0].yaxis.set_major_locator(MultipleLocator(0.02))
    axs[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[0].set_xlabel('Training tokens (M, log)')
    axs[1].set_xlabel('Component of ΔBPB')
    fig.tight_layout(w_pad=0.3)
    save(fig, 'cr_main')


def fig_e1(D, poster=False):
    style(1.45 if poster else 1.0)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.15) if not poster else (11, 3.6),
                                 gridspec_kw={'width_ratios': [1.25, 1]})
    draw_e1(ax, D)
    draw_split(bx, D)
    fig.tight_layout(w_pad=1.2)
    save(fig, 'cr_e1', poster)


def fig_agents(D, poster=False):
    style(1.45 if poster else 1.0)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.0) if not poster else (11, 3.4))
    draw_tokens(ax, D)
    draw_speeds(bx, D)
    fig.tight_layout(w_pad=1.5)
    save(fig, 'cr_agents', poster)


def main():
    D = extract()
    D['phase5'] = extract_phase5(D)
    D['inputs'][str(P5.relative_to(ROOT))] = sha256_file(P5)
    GEN_CR.mkdir(parents=True, exist_ok=True)
    write_json(CR_DATA / 'cr_v2.json', D)
    macros(D)
    table_e1(D)
    table_agents(D)
    table_bench(D)
    table_reasoning(D)
    table_env_v2()
    fig_main(D)
    fig_paths(D)
    for poster in (False, True):
        fig_e1(D, poster)
        fig_agents(D, poster)
        fig_ablation(D, poster)
    print('cr_v2: wrote cr_v2.json, cr_macros_v2.tex, table_e1.tex, table_agents.tex, table_bench.tex, cr_main.*, cr_e1.*, cr_agents.*, cr_ablation.*')


if __name__ == '__main__':
    main()
