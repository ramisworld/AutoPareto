#!/usr/bin/env python3
"""Extract and recompute every camera-ready number from primary records (read-only).

Inputs (read-only):
  SNAPDB   results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db
  P1JSON   .../phase1/phase1_equal_token_results.json  (canonical validation and held-out values)
  per-run  .../phase1/runs/phase1-<model>-seed<k>/{training_metrics,final_inference}.json
  selection manifests via `git show autopareto/research-infrastructure:results/...`
Outputs:
  paper/camera_ready/data/cr_evidence.json   extracted records with provenance hashes
  paper/camera_ready/data/cr_stats.json      derived statistics + cross-check against the audit JSONs

Every derived quantity is compared with audit/camera_ready_2026/supporting/{statistical_audit,
claims_recomputation}.json where the audit computed the same thing; a mismatch above 1e-9 aborts.
"""
from __future__ import annotations

import ast
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lineage_labels import module_constants
from cr_common import (AUDIT_CLAIMS, AUDIT_STATS, CR_DATA, DISCOVERY_EXP, EVIDENCE, P1JSON, PANEL_ORDER, PLAN_REF,
                       SELECTION_REF, SNAP_PHASE1, SNAPDB, TOKEN_STREAM, connect_ro, git_commit_of, git_show,
                       paired, read_json, sha256_bytes, sha256_file, write_json)

AP_CAMPAIGNS = ['autopareto-seed42-20260727', 'autopareto-seed0-20260731', 'autopareto-seed1-20260731']
ARCHIVE_CAMPAIGNS = AP_CAMPAIGNS + ['autoresearch-seed42-20260723', 'autoresearch-diagnostic5-seed42-20260724',
                                    'tpe-seed42-20260805', 'motpe-seed42-20260805', 'random-search-seed42-20260805',
                                    'nsgaii-seed42-20260805', 'cmaes-seed42-20260805']
FIXED_TIME_CANDIDATES = [65, 68, 124, 134, 136, 137]


def single(values):
    vals = set(values)
    if len(vals) != 1:
        raise ValueError(f'expected one value, got {sorted(vals)[:5]}')
    return vals.pop()


def extract():
    con = connect_ro()
    p1 = read_json(P1JSON)
    panel = []
    for r in p1['runs']:
        rd = SNAP_PHASE1 / 'runs' / f"phase1-{r['model_id']}-seed{r['seed']}"
        tm = read_json(rd / 'training_metrics.json')
        fi = read_json(rd / 'final_inference.json')
        reps = [x for x in fi['inference_repetitions'] if not x.get('is_warmup')]
        panel.append({
            'model': r['model_id'], 'seed': r['seed'], 'heldout_bpb': r['heldout_bpb'],
            'val_bpb': r['final_validation_bpb'], 'decode_tok_s': r['stabilized_decode_tokens_per_second'],
            'peak_alloc_mib': r['peak_inference_cuda_memory_mb'], 'optimizer_steps': r['optimizer_steps'],
            'tokens': tm['total_training_tokens'], 'params': tm['parameter_count'],
            'steady_s': r['steady_state_equivalent_seconds'],
            'kv_mib': single(x['incremental_kv_cache_memory_mb'] for x in reps),
            'mem_all_reps': single(x['peak_allocated_cuda_memory_mb'] for x in reps),
            'n_measured_reps': len(reps), 'source_sha256': tm['source_sha256'],
            'excluded_initial_tokens_timing_only': tm['excluded_initial_tokens'],
        })
    fixed = [dict(r) for r in con.execute(
        'select c.id conf_id, c.candidate_experiment_id cand, c.seed, c.selection_role role, c.status conf_status, '
        'c.created_at, t.id training_run_id, t.experiment_id exp_row, t.val_bpb, t.total_training_tokens tokens, '
        't.status train_status from confirmation_runs c join training_runs t on t.id=c.training_run_id order by c.id')]
    cols = ('id, campaign_id, experiment_number, method, training_seed, status, val_bpb, parameter_count, '
            'total_training_tokens, median_cached_decode_tokens_per_second, inference_tokens_per_second, '
            'peak_inference_cuda_memory_mb, decision, change_category, parent_experiment_id, inference_status, '
            'valid_result, exclude_from_analysis, started_at, finished_at')
    archive = [dict(r) for r in con.execute(
        f'select {cols} from experiments where campaign_id in ({",".join("?" * len(ARCHIVE_CAMPAIGNS))}) order by id',
        ARCHIVE_CAMPAIGNS)]
    baselines = [dict(r) for r in con.execute(f'select {cols} from experiments where id between 3 and 13 order by id')]
    lineage = [dict(r) for r in con.execute(
        'select id, experiment_number, parent_experiment_id, status, decision, change_category, hypothesis, '
        'intended_change, agent_interpretation, val_bpb, inference_tokens_per_second, '
        'peak_inference_cuda_memory_mb, parameter_count, total_training_tokens, crash_or_failure_reason, '
        'started_at, finished_at from experiments where campaign_id=? or id=8 order by id',
        ('autopareto-seed0-20260731',))]
    for x in lineage:
        if x['crash_or_failure_reason']:
            x['crash_or_failure_reason'] = x['crash_or_failure_reason'].strip().splitlines()[-1][:300]
    campaigns = [dict(r) for r in con.execute(
        'select campaign_id, created_at from campaigns where campaign_id like "autopareto-primary-%" order by created_at')]
    sel_bytes = git_show(SELECTION_REF)
    plan_bytes = git_show(PLAN_REF)
    sel = json.loads(sel_bytes)
    plan = json.loads(plan_bytes)
    selection = {
        'selection_ref': SELECTION_REF, 'selection_sha256': sha256_bytes(sel_bytes),
        'selection_commit': git_commit_of('autopareto/research-infrastructure', 'results/autopareto_confirmation_selection.json'),
        'selection_rule': sel['selection_rule'],
        'representatives': sorted(int(r['experiment_id'] if 'experiment_id' in r else r['id']) for r in sel['representatives']),
        'plan_ref': PLAN_REF, 'plan_sha256': sha256_bytes(plan_bytes),
        'plan_commit': git_commit_of('autopareto/research-infrastructure', 'results/autopareto_primary_confirmation_plan.json'),
        'plan_rule': plan['selection_rule'], 'plan_source_selection_sha256': plan['source_selection_sha256'],
        'plan_recipes': [{'exp': r['candidate_experiment_id'], 'role': r['role']} for r in plan['recipes']],
        'confirmation_campaigns_created_at_db': campaigns,
    }
    down = []
    for r in con.execute(
            'select e.method model, e.training_seed seed, t.task_name task, t.metric_name metric, t.metric_value value, '
            't.metric_standard_error se, t.sample_count n, t.split split, t.raw_json raw from downstream_task_results t '
            'join downstream_evaluation_runs d on d.id=t.downstream_run_id join experiments e on e.id=d.experiment_id '
            "where e.phase='phase1_downstream' and d.status='completed'"):
        row = dict(r)
        raw = json.loads(row.pop('raw') or '{}')
        if row['task'] == 'blimp' and row['metric'] == 'macro_accuracy':
            row['configurations'] = {k: v['accuracy'] for k, v in (raw.get('configurations') or {}).items()}
        down.append(row)
    depth = {}
    for r in panel:
        if r['model'] in depth:
            continue
        rd = SNAP_PHASE1 / 'runs' / f"phase1-{r['model']}-seed{r['seed']}"
        src = read_json(rd / 'training_metrics.json')['source_path'].replace('/workspace/autopareto/', '')
        text = (EVIDENCE / src).read_text()
        depth[r['model']] = {'source': src, 'sha256': sha256_file(EVIDENCE / src),
                             'DEPTH': module_constants(ast.parse(text))['DEPTH']}
    ts = read_json(TOKEN_STREAM)
    ev = {
        'provenance': {'snapdb': str(SNAPDB), 'snapdb_sha256': sha256_file(SNAPDB), 'p1json': str(P1JSON),
                       'p1json_sha256': sha256_file(P1JSON), 'token_stream_verification': str(TOKEN_STREAM),
                       'token_stream_status': ts.get('status'), 'sqlite_mode': 'ro&immutable=1'},
        'panel_runs': panel, 'fixed_time_confirmations': fixed, 'archive': archive, 'baselines': baselines,
        'seed0_lineage': lineage, 'selection': selection, 'panel_downstream': down, 'panel_depth': depth,
        'training_runs_left_running': [dict(r) for r in con.execute(
            "select id, experiment_id from training_runs where status='running'")],
        'panel_token_budget': p1['token_budget'], 'panel_seeds': p1['seeds'], 'panel_status': p1['status'],
    }
    write_json(CR_DATA / 'cr_evidence.json', ev)
    return ev


def dominated(a, b):
    """b dominates a on (held-out BPB down, decode up, memory down)."""
    return (b['bpb'] <= a['bpb'] and b['dec'] >= a['dec'] and b['mem'] <= a['mem']
            and (b['bpb'] < a['bpb'] or b['dec'] > a['dec'] or b['mem'] < a['mem']))


def stats(ev):
    S = {}
    panel = ev['panel_runs']
    by = {(r['model'], r['seed']): r for r in panel}
    seeds = sorted({r['seed'] for r in panel})
    # Table 1 summaries
    t1 = {}
    for m in PANEL_ORDER:
        rs = [by[(m, s)] for s in seeds]
        def ms(k):
            v = [r[k] for r in rs]
            return {'mean': statistics.mean(v), 'sd': statistics.stdev(v), 'values': v}
        t1[m] = {k: ms(k) for k in ('heldout_bpb', 'val_bpb', 'decode_tok_s', 'peak_alloc_mib', 'steady_s', 'kv_mib')}
        t1[m]['params'] = single(r['params'] for r in rs)
        t1[m]['optimizer_steps'] = single(r['optimizer_steps'] for r in rs)
        t1[m]['tokens'] = single(r['tokens'] for r in rs)
    S['table1'] = t1
    ap, ar = 'autopareto_292', 'autoresearch_compatible'
    S['matched_heldout'] = paired([by[(ap, s)]['heldout_bpb'] - by[(ar, s)]['heldout_bpb'] for s in seeds])
    S['matched_val'] = paired([by[(ap, s)]['val_bpb'] - by[(ar, s)]['val_bpb'] for s in seeds])
    S['decode_ratio'] = t1[ap]['decode_tok_s']['mean'] / t1[ar]['decode_tok_s']['mean']
    S['decode_ratio_by_seed'] = [by[(ap, s)]['decode_tok_s'] / by[(ar, s)]['decode_tok_s'] for s in seeds]
    S['param_reduction_pct'] = 100 * (1 - t1[ap]['params'] / t1[ar]['params'])
    S['memory_reduction_pct'] = 100 * (1 - t1[ap]['peak_alloc_mib']['mean'] / t1[ar]['peak_alloc_mib']['mean'])
    S['kv_ratio'] = t1[ap]['kv_mib']['mean'] / t1[ar]['kv_mib']['mean']
    # Frontier membership per seed and on means
    fr = {}
    for key in [str(s) for s in seeds] + ['means']:
        pts = {}
        for m in PANEL_ORDER:
            if key == 'means':
                pts[m] = {'bpb': t1[m]['heldout_bpb']['mean'], 'dec': t1[m]['decode_tok_s']['mean'], 'mem': t1[m]['peak_alloc_mib']['mean']}
            else:
                r = by[(m, int(key))]
                pts[m] = {'bpb': r['heldout_bpb'], 'dec': r['decode_tok_s'], 'mem': r['peak_alloc_mib']}
        fr[key] = sorted(m for m in PANEL_ORDER if not any(dominated(pts[m], pts[o]) for o in PANEL_ORDER if o != m))
    S['frontier'] = fr
    # Fixed-time confirmations (300 s, seeds 0-2); training completed for every row used
    fx = {}
    for row in ev['fixed_time_confirmations']:
        fx.setdefault(row['cand'], {})[row['seed']] = row
    S['fixed_time'] = {str(c): {'val_by_seed': {str(s): fx[c][s]['val_bpb'] for s in (0, 1, 2)},
                                'tokens_by_seed': {str(s): fx[c][s]['tokens'] for s in (0, 1, 2)},
                                'mean': statistics.mean(fx[c][s]['val_bpb'] for s in (0, 1, 2)),
                                'sd': statistics.stdev(fx[c][s]['val_bpb'] for s in (0, 1, 2)),
                                'conf_status_by_seed': {str(s): fx[c][s]['conf_status'] for s in (0, 1, 2)},
                                'created_at_by_seed': {str(s): fx[c][s]['created_at'] for s in (0, 1, 2)}}
                       for c in FIXED_TIME_CANDIDATES}
    S['fixed_time_apq_minus_ar'] = paired([fx[134][s]['val_bpb'] - fx[68][s]['val_bpb'] for s in (0, 1, 2)])
    S['pooled_fixed_time_sd'] = math.sqrt(statistics.mean(S['fixed_time'][str(c)]['sd'] ** 2 for c in FIXED_TIME_CANDIDATES))
    S['pooled_panel_heldout_sd'] = math.sqrt(statistics.mean(t1[m]['heldout_bpb']['sd'] ** 2 for m in PANEL_ORDER))
    S['max_panel_heldout_sd'] = max(t1[m]['heldout_bpb']['sd'] for m in PANEL_ORDER)
    # Discovery archive
    arch = {r['id']: r for r in ev['archive']}
    S['discovery_apq_minus_ar'] = arch[134]['val_bpb'] - arch[68]['val_bpb']
    bests = {}
    for c in ARCHIVE_CAMPAIGNS:
        rs = [r for r in ev['archive'] if r['campaign_id'] == c and r['val_bpb'] is not None
              and r['valid_result'] and not r['exclude_from_analysis']]
        rs_all = [r for r in ev['archive'] if r['campaign_id'] == c]
        # Top-5 gaps use every row that recorded a BPB (as the audit does, 05 section 3.5);
        # the archive best uses valid, non-excluded rows only.
        top = sorted(r['val_bpb'] for r in rs_all if r['val_bpb'] is not None)[:5]
        b = min(rs, key=lambda r: r['val_bpb'])
        bests[c] = {'best_exp': b['id'], 'best_val': b['val_bpb'], 'seed': b['training_seed'], 'attempts': len(rs_all),
                    'with_bpb': sum(r['val_bpb'] is not None for r in rs_all), 'valid_with_bpb': len(rs),
                    'completed': sum(r['status'] == 'completed' for r in rs_all),
                    'gap_best_to_2nd': top[1] - top[0], 'gap_best_to_5th': top[4] - top[0]}
    S['archive_bests'] = bests
    compat68 = arch[68]['val_bpb']
    S['seed42_bests'] = {'AutoPareto': bests['autopareto-seed42-20260727']['best_val'], 'AR-ref (best compatible)': compat68,
                         'TPE': bests['tpe-seed42-20260805']['best_val'], 'MOTPE': bests['motpe-seed42-20260805']['best_val']}
    S['seed42_spread'] = max(S['seed42_bests'].values()) - min(S['seed42_bests'].values())
    base = {r['id']: r for r in ev['baselines']}
    S['baseline_seed0_minus_seed42'] = base[8]['val_bpb'] - base[13]['val_bpb']
    # Session offset under the time contract: same baseline code and seeds, two campaigns (ids 3-7 vs 8-12)
    S['baseline_session_offset_by_seed'] = {str(k): base[8 + k]['val_bpb'] - base[3 + k]['val_bpb'] for k in range(5)}
    S['baseline_session_tokens'] = {'early': [base[3 + k]['total_training_tokens'] for k in range(5)],
                                    'cached_fresh': [base[8 + k]['total_training_tokens'] for k in range(5)]}
    # Selection optimism and token exposure under the time contract
    S['optimism'] = {str(c): {'discovery': arch[c]['val_bpb'], 'rerun_mean': S['fixed_time'][str(c)]['mean'],
                              'optimism': arch[c]['val_bpb'] - S['fixed_time'][str(c)]['mean']}
                     for c in FIXED_TIME_CANDIDATES}
    S['token_spread_pct'] = {}
    for c in FIXED_TIME_CANDIDATES:
        toks = [arch[c]['total_training_tokens']] + [fx[c][s]['tokens'] for s in (0, 1, 2)]
        mu = statistics.mean(toks)
        S['token_spread_pct'][str(c)] = 100 * (max(toks) - min(toks)) / mu  # range / mean (audit 05 section 2.4)
    S['exp136_tokens'] = {'discovery': arch[136]['total_training_tokens'], 'reruns': [fx[136][s]['tokens'] for s in (0, 1, 2)]}
    # Change categories over the 75 AutoPareto attempts (agent-recorded field)
    cats = {}
    ap_rows = [r for r in ev['archive'] if r['campaign_id'] in AP_CAMPAIGNS]
    for r in ap_rows:
        cats[r['change_category']] = cats.get(r['change_category'], 0) + 1
    S['ap_attempts'] = len(ap_rows)
    S['ap_change_categories'] = cats
    # Lineage step sizes after A18
    lin = {r['experiment_number']: r for r in ev['seed0_lineage'] if r['id'] != 8}
    S['lineage_steps_after_A18'] = {f'A{p}->A{c}': lin[p]['val_bpb'] - lin[c]['val_bpb'] for p, c in [(18, 19), (19, 20), (20, 22)]}
    S['A09_A10'] = {'bpb': [lin[9]['val_bpb'], lin[10]['val_bpb']], 'decode': [lin[9]['inference_tokens_per_second'], lin[10]['inference_tokens_per_second']],
                    'params': [lin[9]['parameter_count'], lin[10]['parameter_count']]}
    S['A25_decode'] = lin[25]['inference_tokens_per_second']
    S['exp109'] = {k: arch[109][k] for k in ('val_bpb', 'parameter_count', 'total_training_tokens', 'training_seed')}
    S['token_ratio_panel_over_fixed'] = {str(c): ev['panel_token_budget'] / statistics.mean(fx[c][s]['tokens'] for s in (0, 1, 2))
                                         for c in (68, 134)}
    # Downstream (panel, seeds 3-5): seed means per recipe, ARC-E standard error, BLiMP per-configuration split
    D = {}
    for row in ev['panel_downstream']:
        D.setdefault((row['model'], row['task'], row['metric']), []).append(row)
    dm = {}
    for (m, t, k), rows in D.items():
        dm.setdefault(m, {})[f'{t}/{k}'] = {'mean': statistics.mean(r['value'] for r in rows), 'n_runs': len(rows),
                                            'mean_se': statistics.mean(r['se'] for r in rows if r['se'] is not None) if any(r['se'] is not None for r in rows) else None,
                                            'sample_count': rows[0]['n'], 'split': rows[0]['split']}
    S['downstream_means'] = dm
    def cfg_mean(m):
        rows = D[(m, 'blimp', 'macro_accuracy')]
        keys = rows[0]['configurations'].keys()
        return {k: statistics.mean(r['configurations'][k] for r in rows) for k in keys}
    a, b = cfg_mean(ap), cfg_mean(ar)
    S['blimp_config_split'] = {'apq_lower': sum(a[k] < b[k] for k in a), 'tied': sum(a[k] == b[k] for k in a),
                               'apq_higher': sum(a[k] > b[k] for k in a), 'n_configs': len(a)}
    import numpy as np
    blimp = [dm[m]['blimp/macro_accuracy']['mean'] for m in PANEL_ORDER]
    held = [t1[m]['heldout_bpb']['mean'] for m in PANEL_ORDER]
    dep = [ev['panel_depth'][m]['DEPTH'] for m in PANEL_ORDER]
    S['heldout_minus_val_201M'] = {m: t1[m]['heldout_bpb']['mean'] - t1[m]['val_bpb']['mean'] for m in PANEL_ORDER}
    S['corr_blimp_heldout'] = float(np.corrcoef(blimp, held)[0, 1])
    S['corr_blimp_depth'] = float(np.corrcoef(blimp, dep)[0, 1])
    S['exp68_tokens'] = arch[68]['total_training_tokens']
    return S


def cross_check(S):
    A = read_json(AUDIT_STATS)
    C = read_json(AUDIT_CLAIMS)
    pairs = [
        ('matched_heldout.mean', S['matched_heldout']['mean'], A['matched_token_AP292_minus_AR']['heldout']['mean']),
        ('matched_heldout.ci_lo', S['matched_heldout']['ci95'][0], A['matched_token_AP292_minus_AR']['heldout']['ci95'][0]),
        ('matched_heldout.ci_hi', S['matched_heldout']['ci95'][1], A['matched_token_AP292_minus_AR']['heldout']['ci95'][1]),
        ('matched_val.mean', S['matched_val']['mean'], A['matched_token_AP292_minus_AR']['validation']['mean']),
        ('matched_val.ci_lo', S['matched_val']['ci95'][0], A['matched_token_AP292_minus_AR']['validation']['ci95'][0]),
        ('matched_val.ci_hi', S['matched_val']['ci95'][1], A['matched_token_AP292_minus_AR']['validation']['ci95'][1]),
        ('fixed_time.mean', S['fixed_time_apq_minus_ar']['mean'], A['fixed_time_AP292_minus_AR_val_seeds012']['mean']),
        ('fixed_time.ci_lo', S['fixed_time_apq_minus_ar']['ci95'][0], A['fixed_time_AP292_minus_AR_val_seeds012']['ci95'][0]),
        ('fixed_time.ci_hi', S['fixed_time_apq_minus_ar']['ci95'][1], A['fixed_time_AP292_minus_AR_val_seeds012']['ci95'][1]),
        ('pooled_fixed_time_sd', S['pooled_fixed_time_sd'], A['seed_variability']['pooled_fixed_time_val_sd']),
        ('pooled_panel_heldout_sd', S['pooled_panel_heldout_sd'], A['seed_variability']['pooled_panel_heldout_sd']),
        ('discovery_gap', S['discovery_apq_minus_ar'], A['reversal_table'][0]['delta_AP292_minus_AR']),
        ('decode_ratio', S['decode_ratio'], C['headline']['decode_ratio']),
        ('param_reduction_pct', S['param_reduction_pct'], C['headline']['param_reduction_pct']),
        ('memory_reduction_pct', S['memory_reduction_pct'], C['headline']['memory_reduction_pct']),
    ]
    for i, r in enumerate(S['decode_ratio_by_seed']):
        pairs.append((f'decode_ratio_seed{i}', r, A['matched_token_AP292_minus_AR']['decode_ratio_by_seed'][i]))
    for c in FIXED_TIME_CANDIDATES:
        pairs.append((f'optimism_{c}', S['optimism'][str(c)]['optimism'], A['selection_optimism'][str(c)]['optimism']))
        pairs.append((f'token_spread_{c}', S['token_spread_pct'][str(c)], A['fixed_time_token_exposure_variation'][str(c)]['max_rel_dev_pct']))
    for camp, v in A['top_candidate_gaps_vs_noise']['campaigns'].items():
        pairs.append((f'gap2_{camp}', S['archive_bests'][camp]['gap_best_to_2nd'], v['gap_best_to_2nd']))
        pairs.append((f'gap5_{camp}', S['archive_bests'][camp]['gap_best_to_5th'], v['gap_best_to_5th']))
    for m, dd in C['downstream_means_201M'].items():
        for key, val in dd.items():
            mine = S['downstream_means'].get(m, {}).get(key.replace(':', '/'), {}).get('mean') if isinstance(val, (int, float)) else None
            if mine is not None:
                pairs.append((f'downstream_{m}_{key}', mine, val))
    for k in range(5):
        pairs.append((f'session_offset_seed{k}', S['baseline_session_offset_by_seed'][str(k)],
                      C['baseline_seed_effects']['cached_fresh_seeds0_4'][str(k)] - C['baseline_seed_effects']['early_seeds0_4'][str(k)]))
    pairs.append(('corr_blimp_heldout', S['corr_blimp_heldout'], C['correlations_7_recipes']['blimp_vs_heldout_pearson']))
    pairs.append(('corr_blimp_depth', S['corr_blimp_depth'], C['correlations_7_recipes']['blimp_vs_depth_pearson']))
    for k, m in [('3', '3'), ('4', '4'), ('5', '5'), ('means', 'means')]:
        pairs.append((f'frontier_{k}', float(S['frontier'][k] == sorted(A['panel_frontier_by_seed'][m])), 1.0))
    out = [{'quantity': q, 'camera_ready': a, 'audit': b, 'abs_diff': abs(a - b)} for q, a, b in pairs]
    worst = max(x['abs_diff'] for x in out)
    if worst > 1e-9:
        bad = [x for x in out if x['abs_diff'] > 1e-9]
        raise SystemExit(f'cross-check failed: {bad}')
    return {'n_checked': len(out), 'max_abs_diff': worst, 'items': out}


def main():
    ev = extract()
    S = stats(ev)
    S['cross_check_vs_audit'] = cross_check(S)
    write_json(CR_DATA / 'cr_stats.json', S)
    print(f"extracted {len(ev['panel_runs'])} panel runs, {len(ev['fixed_time_confirmations'])} fixed-time rows; "
          f"cross-checked {S['cross_check_vs_audit']['n_checked']} quantities vs audit (max |diff| {S['cross_check_vs_audit']['max_abs_diff']:.2e})")
    return ev, S


if __name__ == '__main__':
    main()
