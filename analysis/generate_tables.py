#!/usr/bin/env python3
from common import *

def table(name,header,rows,alignment):
 s='\\begin{tabular}{'+alignment+'}\n\\toprule\n'+' & '.join(header)+' \\\\\n\\midrule\n'
 s+='\n'.join(' & '.join(str(x) for x in row)+' \\\\' for row in rows)+'\n\\bottomrule\n\\end{tabular}\n'
 (OUT/name).write_text(s)
def main():
 ms=read(OUT/'pareto_analysis.json')['models'];m={x['model']:x for x in ms};ds=read(OUT/'downstream_analysis.json');dc=read(OUT/'downstream_contrast.json');dm=read(OUT/'downstream_mechanism.json');b=read(OUT/'budget_analysis.json');runs=read(OUT/'equal_token_runs.json')
 macro={}
 for model,prefix in [('autopareto_292','AP'),('autoresearch_compatible','AR')]:
  x=m[model]
  for key,suffix,fmt,div in [('parameters','Params','.2f',1e6),('bpb','BPB','.6f',1),('speed','Speed','.2f',1),('steady_seconds','Time','.2f',1),('scientific_seconds','Wall','.2f',1),('memory','Memory','.2f',1)]:macro[prefix+suffix]=format(x[key]['mean']/div,fmt)
 macro['DecodeRatio']=f"{m['autopareto_292']['speed']['mean']/m['autoresearch_compatible']['speed']['mean']:.2f}"
 macro['MemoryReduction']=f"{100*(1-m['autopareto_292']['memory']['mean']/m['autoresearch_compatible']['memory']['mean']):.1f}"
 macro['ParameterReduction']=f"{100*(1-m['autopareto_292']['parameters']['mean']/m['autoresearch_compatible']['parameters']['mean']):.1f}"
 macro['BPBCost']=f"{m['autopareto_292']['bpb']['mean']-m['autoresearch_compatible']['bpb']['mean']:.6f}"
 macro['PanelN']=str(sum(x['eligible'] for x in runs));macro['BudgetN']=str(len(b['rows']));macro['TimeN']=str(b['arms']['T']['attempts']);macro['TokenN']=str(b['arms']['N']['attempts'])
 macro['TimeBest']=f"{b['arms']['T']['best_bpb']:.6f}";macro['TokenBest']=f"{b['arms']['N']['best_bpb']:.6f}"
 macro['BudgetParetoN']=str(sum(x['pareto_eligible'] for x in b['rows']))
 macro['BLiMPLower']=str(dc['autopareto_lower']);macro['BLiMPTied']=str(dc['tied']);macro['BLiMPHigher']=str(dc['autopareto_higher'])
 ap_runs=[x for x in runs if x['eligible'] and x['model']=='autopareto_292'];ar_runs=[x for x in runs if x['eligible'] and x['model']=='autoresearch_compatible']
 ap_kv=sum(x['incremental_kv_memory'] for x in ap_runs)/len(ap_runs);ar_kv=sum(x['incremental_kv_memory'] for x in ar_runs)/len(ar_runs)
 macro['APKV']=f"{ap_kv:.3f}";macro['ARKV']=f"{ar_kv:.3f}";macro['KVReduction']=f"{100*(1-ap_kv/ar_kv):.0f}"
 macro['BLiMPHeldoutCorr']=f"{dm['correlations']['blimp_vs_heldout_bpb']:.3f}";macro['BLiMPDepthCorr']=f"{dm['correlations']['blimp_vs_depth']:.3f}";macro['BLiMPParamCorr']=f"{dm['correlations']['blimp_vs_parameters']:.3f}"
 ablations=read(OUT/'db_ablation_runs.json');training={x['id']:x for x in read(OUT/'db_training_runs.json')}
 for name,prefix in [('batch_only','BatchOnly'),('nonbatch_only','NonBatchOnly')]:
  ar=[training[x['training_run_id']] for x in ablations if x['ablation_name']==name and x['status']=='completed']
  macro[prefix+'BPB']=f"{sum(x['val_bpb'] for x in ar)/len(ar):.4f}";macro[prefix+'Tokens']=f"{sum(x['total_training_tokens'] for x in ar)/len(ar)/1e6:.1f}"
 (OUT/'results_macros.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macro.items())+'\n');write(OUT/'headline_values.json',macro)
 table('main_results_table.tex',['Recipe','M params','BPB $\\downarrow$','Decode $\\uparrow$','MiB $\\downarrow$','Steady s (est.)'],[[LABELS[x['model']],f"{x['parameters']['mean']/1e6:.2f}",f"{x['bpb']['mean']:.4f}",f"{x['speed']['mean']:.1f}",f"{x['memory']['mean']:.1f}",f"{x['steady_seconds']['mean']:.1f}"] for x in ms],'lrrrrr')
 hist=read(OUT/'historical_experiments.json')
 search_rows=[]
 search_specs=[
  ('AutoPareto','autopareto','3 $\\times$ 25','Source code','Quality, decode, memory'),
  ('AutoResearch','autoresearch','50 + 5','Source code','Validation BPB'),
  ('Random','random_search','25','Fixed vector','Validation BPB'),
  ('TPE','tpe','25','Fixed vector','Validation BPB'),
  ('MOTPE','motpe','25','Fixed vector','Multiple objectives'),
  ('NSGA-II','nsgaii','25','Fixed vector','Multiple objectives'),
  ('CMA-ES','cmaes','25','Fixed vector','Validation BPB')]
 for label,method,attempts,surface,objective in search_specs:
  candidates=[x for x in hist if x['method']==method and x['valid_result'] and not x['exclude_from_analysis'] and x['val_bpb'] is not None]
  best=min((x['val_bpb'] for x in candidates),default=None)
  search_rows.append([label,attempts,surface,objective,f"{best:.4f}" if best is not None else 'n/a'])
 table('search_comparison_table.tex',['Method','Attempts','Surface','Selection signal','Best archived val. BPB (diagnostic)'],search_rows,'lp{.5in}p{.62in}p{1.05in}r')
 table('budget_results_table.tex',['Trial','Depth','M params','M tokens','Updates','BPB'],[[x['condition']+str(x['attempt']),x['depth'],f"{x['parameters']/1e6:.2f}",f"{x['tokens']/1e6:.3f}",x['steps'],f"{x['bpb']:.4f}"] for x in sorted(b['rows'],key=lambda x:(x['condition']=='N',x['attempt'])) if x['quality_eligible']],'lrrrrr')
 keys=[('arc_easy','accuracy'),('arc_easy','length_normalized_accuracy'),('piqa','accuracy'),('piqa','length_normalized_accuracy'),('lambada_openai','exact_match'),('lambada_openai','perplexity'),('blimp','macro_accuracy'),('blimp','micro_accuracy')]
 table('downstream_table.tex',['Recipe','ARC','ARC-n','PIQA','PIQA-n','L-EM','L-PPL','BL-M','BL-m'],[[LABELS[model]]+[f"{ds[model][a+'/'+k]['mean']*(1 if k=='perplexity' else 100):.2f}" for a,k in keys] for model in ORDER],'lrrrrrrrr')
 ap,ar=ds['autopareto_292'],ds['autoresearch_compatible']
 table('downstream_comparison_table.tex',['Task','Metric','AP292','AutoResearch'],[
  ['ARC-Easy','Accuracy',f"{100*ap['arc_easy/accuracy']['mean']:.2f}",f"{100*ar['arc_easy/accuracy']['mean']:.2f}"],
  ['ARC-Easy','Length-normalized',f"{100*ap['arc_easy/length_normalized_accuracy']['mean']:.2f}",f"{100*ar['arc_easy/length_normalized_accuracy']['mean']:.2f}"],
  ['PIQA','Accuracy',f"{100*ap['piqa/accuracy']['mean']:.2f}",f"{100*ar['piqa/accuracy']['mean']:.2f}"],
  ['PIQA','Length-normalized',f"{100*ap['piqa/length_normalized_accuracy']['mean']:.2f}",f"{100*ar['piqa/length_normalized_accuracy']['mean']:.2f}"],
  ['LAMBADA','Exact match',f"{100*ap['lambada_openai/exact_match']['mean']:.2f}",f"{100*ar['lambada_openai/exact_match']['mean']:.2f}"],
  ['LAMBADA','Perplexity',f"{ap['lambada_openai/perplexity']['mean']:.2f}",f"{ar['lambada_openai/perplexity']['mean']:.2f}"],
  ['BLiMP','Accuracy',f"{100*ap['blimp/macro_accuracy']['mean']:.2f}",f"{100*ar['blimp/macro_accuracy']['mean']:.2f}"]],
  'llrr')
 controls=[['Initial code / loader','Identical executed input bytes','Identical executed input bytes'],['Scientist / seed','gpt-5.6-sol, high / 3','gpt-5.6-sol, high / 3'],['Training stop','300 s; complete final update','88,604,672 counted tokens'],['LR progress','Elapsed-time fraction','Counted-token fraction'],['Frozen prefix',f"{b['arms']['T']['attempts']} completed attempts",f"{b['arms']['N']['attempts']} completed attempts"],['Inference / memory','Shared bracketed protocol','Shared bracketed protocol'],['Context / hardware','Separate history; one A40','Separate history; same A40'],['Parent policy (executed)','Quality-valid parents allowed','Quality-valid parents allowed']]
 table('controls_table.tex',['Control','Fixed time (T)','Fixed tokens (N)'],controls,'p{1.2in}p{1.8in}p{1.8in}')
 examples=[]
 descriptions={('N',1):'Increase depth and width',('N',2):'Increase capacity again',('N',3):'Increase update count at fixed tokens',('T',2):'Halve batch again',('T',3):'Smaller batch',('T',4):'Scale learning rate'}
 for x in b['rows']:
  if (x['condition'],x['attempt']) in descriptions and x['quality_eligible']:
   examples.append([x['condition']+str(x['attempt']),descriptions[(x['condition'],x['attempt'])],f"{x['parameters']/1e6:.2f}M; BPB {x['bpb']:.4f}; {x['steps']} updates"])
 table('representative_discoveries.tex',['Trial','Agent prediction / edit','Observed outcome'],examples,'lp{2.6in}p{1.7in}')
 # Complete SD/per-seed output is machine-readable and in the internal report.
if __name__=='__main__':main()
