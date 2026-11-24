#!/usr/bin/env python3
"""Independent arithmetic checks on raw timing rows and public summaries."""
from common import *
import statistics,math

def main():
 records=[];panel=read(OUT/'equal_token_runs.json');public=read(ROOT/'results/phase1/phase1_equal_token_results.json');down=read(ROOT/'results/phase1/public_downstream_metrics.json')['models'];errs=[]
 for r in panel:
  t=read(ROOT/r['provenance']['training']);lines=[json.loads(x) for x in (ROOT/r['provenance']['timing']).read_text().splitlines() if x.strip()]
  total=sum(x['tokens_processed'] for x in lines);wall=sum(x['step_wall_seconds'] for x in lines);steady=sum(x['step_wall_seconds'] for x in lines if x['zero_based_step_index']>=11);tokens=sum(x['tokens_processed'] for x in lines if x['zero_based_step_index']>=11)
  checks={'token_sum':total==r['tokens'],'step_wall_sum':math.isclose(wall,r['scientific_seconds'],rel_tol=1e-10),'steady_equivalent':math.isclose(steady*total/tokens,r['steady_seconds'],rel_tol=1e-10),'derived_source_hash':sha(Path(r['provenance']['training']).parent/'phase1_train.py')==t['derived_source_sha256']}
  records.append({'id':r['id'],'checks':checks})
  if not all(checks.values()):errs.append(records[-1])
 for pm in public['models']:
  model=pm['model_id'];rs=[x for x in panel if x['model']==model]
  for ours,pub in [('bpb','heldout_bpb'),('speed','stabilized_decode_tokens_per_second'),('steady_seconds','steady_state_equivalent_seconds')]:
   a=statistics.mean(x[ours] for x in rs);b=pm[pub]['mean']
   if not math.isclose(a,b,rel_tol=1e-10):errs.append({'model':model,'metric':ours,'raw_mean':a,'public_mean':b})
  for task,metrics in down[model]['metrics'].items():
   for metric,values in metrics.items():
    xs=[d['metric_value'] for r in rs for d in r['downstream'] if d['task_name']==task and d['metric_name']==metric]
    if len(xs)!=3 or abs(statistics.mean(xs)-values['mean'])>0.00000051:errs.append({'model':model,'task':task,'metric':metric,'error':'public downstream mismatch'})
 write(OUT/'independent_verification.json',{'records':records,'public_summary_comparison':'three principal metrics at full precision, all downstream metrics to published rounding','errors':errs,'passed':not errs})
 print('Independent verification:', 'PASS' if not errs else errs)
 if errs:raise SystemExit(1)
if __name__=='__main__':main()
