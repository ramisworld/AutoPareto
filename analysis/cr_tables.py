#!/usr/bin/env python3
"""Write the camera-ready LaTeX macros and tables from cr_evidence/cr_stats JSON.

Outputs (only these paths; the legacy refresh pipeline never writes here):
  paper/generated/camera_ready/cr_macros.tex
  paper/generated/camera_ready/table_main.tex       Table 1 (matched-exposure panel, means +- SD)
  paper/generated/camera_ready/table_reversal.tex   reversal decomposition (Part 1)
  paper/generated/camera_ready/table_funnel.tex     selection funnel (Part 2)
  paper/generated/camera_ready/table_env.tex        per-era environment (Appendix A)
"""
from __future__ import annotations

import glob
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cr_common import CR_DATA, EVIDENCE, GEN_CR, NAMES, PANEL_ORDER, SNAP_PHASE1, display_names, read_json, write_json


def f(x, nd):
    return f'{x:.{nd}f}'


def sgn(x, nd):
    s = f'{x:+.{nd}f}'
    return ('$-$' + s[1:]) if s.startswith('-') else s


def ci(lo, hi, nd):
    return f'[{sgn(lo, nd)}, {sgn(hi, nd)}]'


def macros(S, ev):
    t1 = S['table1']
    ap, ar = t1['autopareto_292'], t1['autoresearch_compatible']
    ft = S['fixed_time_apq_minus_ar']
    mh, mv = S['matched_heldout'], S['matched_val']
    opt = [abs(v['optimism']) for v in S['optimism'].values()]
    agent_camps = [c for c in S['archive_bests'] if c.startswith('autopareto') or c.startswith('autoresearch')]
    gap2 = max(S['archive_bests'][c]['gap_best_to_2nd'] for c in agent_camps)
    gap5 = max(S['archive_bests'][c]['gap_best_to_5th'] for c in agent_camps)
    cats = S['ap_change_categories']
    m = {
        # Matched-exposure headline (P1JSON, seeds 3-5)
        'CRDecodeRatio': f(S['decode_ratio'], 2),
        'CRDecodeRatioSeeds': ', '.join(f(x, 2) for x in S['decode_ratio_by_seed']),
        'CRDecodeRatioMin': f(min(S['decode_ratio_by_seed']), 2),
        'CRDecodeRatioMax': f(max(S['decode_ratio_by_seed']), 2),
        'CRParamReduction': f(S['param_reduction_pct'], 1),
        'CRMemoryReduction': f(S['memory_reduction_pct'], 1),
        'CRAPQSpeed': f(ap['decode_tok_s']['mean'], 1), 'CRARSpeed': f(ar['decode_tok_s']['mean'], 1),
        'CRAPQHeld': f(ap['heldout_bpb']['mean'], 4), 'CRARHeld': f(ar['heldout_bpb']['mean'], 4),
        'CRAPQParams': f(ap['params'] / 1e6, 2), 'CRARParams': f(ar['params'] / 1e6, 2),
        'CRAPQMem': f(ap['peak_alloc_mib']['mean'], 1), 'CRARMem': f(ar['peak_alloc_mib']['mean'], 1),
        'CRAPQKV': f(ap['kv_mib']['mean'], 3), 'CRARKV': f(ar['kv_mib']['mean'], 3),
        'CRHeldDiff': sgn(mh['mean'], 3), 'CRHeldDiffAbs': f(mh['mean'], 3),
        'CRHeldDiffFour': sgn(mh['mean'], 4),
        'CRHeldCI': ci(mh['ci95'][0], mh['ci95'][1], 3), 'CRHeldCIFour': ci(mh['ci95'][0], mh['ci95'][1], 4),
        'CRHeldCILo': f(mh['ci95'][0], 3), 'CRHeldCIHi': f(mh['ci95'][1], 3),
        'CRValDiff': sgn(mv['mean'], 3), 'CRValDiffFour': sgn(mv['mean'], 4), 'CRValCIFour': ci(mv['ci95'][0], mv['ci95'][1], 4),
        'CRHeldSeeds': ', '.join(sgn(x, 4) for x in mh['diffs']), 'CRValSeeds': ', '.join(sgn(x, 4) for x in mv['diffs']),
        # Fixed-time re-runs (SNAPDB confirmation_runs 7-9 vs 13-15, seeds 0-2)
        'CRFixedDiff': sgn(ft['mean'], 4), 'CRFixedDiffThree': sgn(ft['mean'], 3),
        'CRFixedCIFour': ci(ft['ci95'][0], ft['ci95'][1], 4),
        'CRFixedSeeds': ', '.join(sgn(x, 4) for x in ft['diffs']),
        'CRDiscGap': sgn(S['discovery_apq_minus_ar'], 4),
        # Variability and selection
        'CRPooledSD': f(S['pooled_fixed_time_sd'], 4),
        'CRMaxHeldSD': f(S['max_panel_heldout_sd'], 4),
        'CRBaseSeedEffect': f(S['baseline_seed0_minus_seed42'], 3),
        'CRSeedFortyTwoAP': f(S['seed42_bests']['AutoPareto'], 4),
        'CRSeedFortyTwoAR': f(S['seed42_bests']['AR-ref (best compatible)'], 4),
        'CRSeedFortyTwoTPE': f(S['seed42_bests']['TPE'], 4),
        'CRSeedFortyTwoMOTPE': f(S['seed42_bests']['MOTPE'], 4),
        'CRSeedFortyTwoSpread': f(S['seed42_spread'], 4),
        'CROptMin': f(min(opt), 4), 'CROptMax': f(max(opt), 4),
        'CRGapTwoMax': f(gap2, 4), 'CRGapFiveMax': f(gap5, 4),
        'CRTokenSpreadMax': f(max(S['token_spread_pct'].values()), 1),
        'CRExpOneThreeSixDisc': f(S['exp136_tokens']['discovery'] / 1e6, 1),
        'CRExpOneThreeSixRerunLo': f(min(S['exp136_tokens']['reruns']) / 1e6, 1),
        'CRExpOneThreeSixRerunHi': f(max(S['exp136_tokens']['reruns']) / 1e6, 1),
        'CRExpOneZeroNineBPB': f(S['exp109']['val_bpb'], 4),
        'CRExpOneZeroNineTokens': f(S['exp109']['total_training_tokens'] / 1e6, 1),
        'CRExpSixEightTokens': f(S['exp68_tokens'] / 1e6, 1),
        'CRAPAttempts': str(S['ap_attempts']),
        'CRCatArch': str(cats['architecture']), 'CRCatOptimizer': str(cats['optimizer']),
        'CRCatOptimization': str(cats['optimization']), 'CRCatBatching': str(cats['batching']),
        'CRCatSchedule': str(cats['schedule']),
        'CRStepNineteenTwenty': f(S['lineage_steps_after_A18']['A19->A20'], 4),
        'CRStepTwentyTwentyTwo': f(S['lineage_steps_after_A18']['A20->A22'], 5),
        'CRAnineBPB': f(S['A09_A10']['bpb'][0], 4), 'CRAtenBPB': f(S['A09_A10']['bpb'][1], 4),
        'CRAnineDec': f(S['A09_A10']['decode'][0], 1), 'CRAtenDec': f(S['A09_A10']['decode'][1], 1),
        'CRAtwentyfiveDec': f(S['A25_decode'], 1),
        'CRFrontierN': str(len(S['frontier']['means'])),
        'CRPanelBudget': f"{ev['panel_token_budget']:,}".replace(',', '{,}'),
        'CRARTokRatio': f(S['token_ratio_panel_over_fixed']['68'], 2),
        'CRAPQTokRatio': f(S['token_ratio_panel_over_fixed']['134'], 2),
        'CRFixedSixtyFive': f(S['fixed_time']['65']['mean'], 4), 'CRFixedSixtyEight': f(S['fixed_time']['68']['mean'], 4),
        'CRBLiMPLower': str(S['blimp_config_split']['apq_lower']), 'CRBLiMPTied': str(S['blimp_config_split']['tied']),
        'CRBLiMPHigher': str(S['blimp_config_split']['apq_higher']), 'CRBLiMPConfigs': str(S['blimp_config_split']['n_configs']),
        'CRNCrossChecked': str(S['cross_check_vs_audit']['n_checked']),
        'CRSessionLo': f(min(S['baseline_session_offset_by_seed'].values()), 4),
        'CRSessionHi': f(max(S['baseline_session_offset_by_seed'].values()), 4),
        'CRHeldValOffLo': f(min(S['heldout_minus_val_201M'].values()), 3),
        'CRHeldValOffHi': f(max(S['heldout_minus_val_201M'].values()), 3),
        'CRCorrBLiMPHeld': f(S['corr_blimp_heldout'], 3), 'CRCorrBLiMPDepth': f(S['corr_blimp_depth'], 3),  # both used inside math mode
    }
    dm = S['downstream_means']
    for model, tag in [('autopareto_292', 'APQ'), ('autoresearch_compatible', 'AR')]:
        x = dm[model]
        m[f'CR{tag}ARCE'] = f(100 * x['arc_easy/accuracy']['mean'], 2)
        m[f'CR{tag}PIQA'] = f(100 * x['piqa/accuracy']['mean'], 2)
        m[f'CR{tag}LAMBADA'] = f(100 * x['lambada_openai/exact_match']['mean'], 2)
        m[f'CR{tag}BLiMP'] = f(100 * x['blimp/macro_accuracy']['mean'], 2)
    m['CRARCESE'] = f(100 * statistics.mean([dm['autopareto_292']['arc_easy/accuracy']['mean_se'], dm['autoresearch_compatible']['arc_easy/accuracy']['mean_se']]), 1)
    lines = ['% Generated by analysis/cr_tables.py from paper/camera_ready/data/cr_stats.json. Do not edit by hand.']
    lines += ['\\newcommand{\\' + k + '}{' + v + '}' for k, v in m.items()]
    (GEN_CR / 'cr_macros.tex').write_text('\n'.join(lines) + '\n')
    write_json(CR_DATA / 'cr_macros.json', m)
    return m


def table_main(S):
    t1 = S['table1']
    rows = []
    for k in [m for m in PANEL_ORDER if m not in ('autopareto_294', 'autopareto_299')] + ['autopareto_294', 'autopareto_299']:
        x = t1[k]
        rows.append(f"{NAMES[k]} & {x['params'] / 1e6:.2f} & {x['heldout_bpb']['mean']:.4f}\\,$\\pm$\\,{x['heldout_bpb']['sd']:.4f} & "
                    f"{x['decode_tok_s']['mean']:.1f}\\,$\\pm$\\,{x['decode_tok_s']['sd']:.1f} & {x['peak_alloc_mib']['mean']:.1f} & "
                    f"{x['steady_s']['mean']:.1f} \\\\")
    s = ('% Generated by analysis/cr_tables.py (source: P1JSON + per-run training_metrics/final_inference). Do not edit by hand.\n'
         '\\begin{tabular}{lrrrrr}\n\\toprule\n'
         'Recipe & Params (M) & Held-out BPB $\\downarrow$ & Decode tok/s $\\uparrow$ & Memory MiB $\\downarrow$ & Train s (est.) \\\\\n'
         '\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')
    (GEN_CR / 'table_main.tex').write_text(display_names(s))


def table_reversal(S):
    ft, mv, mh = S['fixed_time_apq_minus_ar'], S['matched_val'], S['matched_heldout']
    rows = [
        f"300\\,s (discovery archive) & 0 vs 42 & val & {sgn(S['discovery_apq_minus_ar'], 4)} & one run each & n/a \\\\",
        f"300\\,s re-runs & 0, 1, 2 & val & \\textbf{{{sgn(ft['mean'], 4)}}} & {', '.join(sgn(x, 4) for x in ft['diffs'])} & {ci(*ft['ci95'], 4)} \\\\",
        f"201M tokens & 3, 4, 5 & val & \\textbf{{{sgn(mv['mean'], 4)}}} & {', '.join(sgn(x, 4) for x in mv['diffs'])} & {ci(*mv['ci95'], 4)} \\\\",
        f"201M tokens & 3, 4, 5 & held-out & \\textbf{{{sgn(mh['mean'], 4)}}} & {', '.join(sgn(x, 4) for x in mh['diffs'])} & {ci(*mh['ci95'], 4)} \\\\",
    ]
    s = ('% Generated by analysis/cr_tables.py (SNAPDB experiments 134/68; confirmation_runs 7-9, 13-15; P1JSON). Do not edit by hand.\n'
         '\\begin{tabular}{llllll}\n\\toprule\n'
         'Contract & Seeds & Data & $\\Delta$ BPB & Per seed & 95\\% CI \\\\\n\\midrule\n'
         + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')
    (GEN_CR / 'table_reversal.tex').write_text(s)


def table_funnel(S, ev):
    sel = ev['selection']
    reps = ', '.join(str(x) for x in sel['representatives'])
    plan = ', '.join(str(r['exp']) for r in sel['plan_recipes'])
    b = S['archive_bests']
    ap_n = sum(b[c]['attempts'] for c in b if c.startswith('autopareto'))
    ar_main, ar_diag = b['autoresearch-seed42-20260723']['attempts'], b['autoresearch-diagnostic5-seed42-20260724']['attempts']
    tpe, motpe = b['tpe-seed42-20260805']['attempts'], b['motpe-seed42-20260805']['attempts']
    panel_ap = sorted({r['model'] for r in ev['panel_runs'] if r['model'].startswith('autopareto')})
    panel_ap_exp = ', '.join(str(x) for x in sorted({134: 1, 136: 1, 137: 1}))
    assert panel_ap == ['autopareto_292', 'autopareto_294', 'autopareto_299']
    rows = [
        f"Discovery attempts (Jul 22--Aug 7) & {ap_n} (seeds 0, 1, 42) & {ar_main}\\,+\\,{ar_diag} (seed 42) & {tpe}\\,+\\,{motpe} (seed 42) \\\\",
        f"Per-campaign representatives (rule fixed Jul 27; output Aug 5) & {len(sel['representatives'])}: exp {reps} & exp 65, 68 & -- \\\\",
        f"Global four-role selector (Aug 10, after discovery) & {len(sel['plan_recipes'])}, all seed 0: exp {plan} & 68 (65 incompatible) & best BPB: 183, 234 \\\\",
        f"201M-token panel (set Aug 21--25; run Aug 25--27) & 3: exp {panel_ap_exp} (exp 124 dropped; no rule recorded) & 68 & 183, 234 \\\\",
    ]
    s = ('% Generated by analysis/cr_tables.py (SNAPDB counts; selection manifest sha256 '
         f"{sel['selection_sha256'][:16]}; plan sha256 {sel['plan_sha256'][:16]}; P1JSON). Do not edit by hand.\n"
         '\\begin{tabular}{p{1.42in}p{1.78in}p{0.88in}p{0.82in}}\n\\toprule\n'
         'Stage & AutoPareto & AutoResearch & TPE / MOTPE \\\\\n\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')
    (GEN_CR / 'table_funnel.tex').write_text(s)


def table_env():
    """Per-era environment.  Panel and Pair-A rows are read from their records; other rows
    are transcribed from REPRODUCIBILITY_AND_PROVENANCE.md section D.1 (sources named there)."""
    runs = SNAP_PHASE1 / 'runs'
    sv = set()
    drivers = set()
    for p in glob.glob(str(runs / 'phase1-*/run_metadata.json')):
        d = json.loads(Path(p).read_text())['software_versions']
        sv.add((d['python'], d['torch'], d['torch_cuda']))
    for p in glob.glob(str(runs / 'phase1-*/final_inference.json')):
        for t in json.loads(Path(p).read_text()).get('gpu_telemetry') or []:
            if isinstance(t, dict) and t.get('driver_version'):
                drivers.add(t['driver_version'])
    (py, torch, cuda), = sv
    drv, = drivers
    man = json.loads((EVIDENCE / 'runs/pair-a-official-20260906-fixed-warmup-v2/provenance/'
                      'final_pre_study_manifest_pair_a_20260906_fixed_warmup_v2.json').read_text())['runtime']
    rows = [
        ('Discovery and fixed-time re-runs (Jul 22--Aug 12)', '3.10 (from tracebacks)', 'not recorded', 'not recorded', 'not recorded'),
        ('Inference canary, session 4 (Aug 16)', '3.12.3', '2.8.0+cu128', '12.8', '570.195.03'),
        ('Matched-token panel, 21 runs (Aug 25--27)', py, torch, cuda, drv),
        ('Pair-A pilot (Sep 6--7)', man['python_version'], man['torch'], man['cuda'], 'not recorded'),
    ]
    body = '\n'.join(' & '.join(r) + ' \\\\' for r in rows)
    s = ('% Generated by analysis/cr_tables.py: panel row from run_metadata.json/final_inference.json (21 runs); '
         'Pair-A row from the v2 manifest; other rows from audit REPRO section D.1. Do not edit by hand.\n'
         '\\begin{tabular}{p{2.1in}llll}\n\\toprule\nEra & Python & PyTorch & CUDA & Driver \\\\\n\\midrule\n'
         + body + '\n\\bottomrule\n\\end{tabular}\n')
    (GEN_CR / 'table_env.tex').write_text(s)
    return {'panel': [py, torch, cuda, drv], 'pair_a': man}


def main():
    GEN_CR.mkdir(parents=True, exist_ok=True)
    ev = read_json(CR_DATA / 'cr_evidence.json')
    S = read_json(CR_DATA / 'cr_stats.json')
    m = macros(S, ev)
    table_main(S)
    table_reversal(S)
    table_funnel(S, ev)
    env = table_env()
    print(f'wrote {len(m)} macros and 4 tables to {GEN_CR}; env panel={env["panel"]}')


if __name__ == '__main__':
    main()
