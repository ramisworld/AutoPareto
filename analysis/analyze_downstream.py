#!/usr/bin/env python3
from common import *
import numpy as np
def main():
 rows=[x for x in read(OUT/'equal_token_runs.json') if x['eligible']];out={};expected=None
 for model in ORDER:
  rs=[x for x in rows if x['model']==model]; groups={}
  for r in rs:
   keys={(d['task_name'],d['metric_name']) for d in r['downstream']}
   if expected is None:expected=keys
   if keys!=expected:raise ValueError('Incomplete downstream suite: '+r['id'])
   for d in r['downstream']:groups.setdefault(d['task_name']+'/'+d['metric_name'],[]).append(d)
  out[model]={k:dict(summary([x['metric_value'] for x in ds]),sample_count=sorted(set(x['sample_count'] for x in ds)),split=ds[0]['split'],dataset=ds[0]['dataset_name'],revision=ds[0]['dataset_revision'],higher_is_better=ds[0]['higher_is_better']) for k,ds in groups.items()}
 write(OUT/'downstream_analysis.json',out)
 # BLiMP configuration-level contrast, averaged over the same three confirmation
 # seeds. This prevents narrative counts from being hand-transcribed.
 per_model={}
 for model in ('autopareto_292','autoresearch_compatible'):
  configs={}
  for r in [x for x in rows if x['model']==model]:
   item=next(d for d in r['downstream'] if d['task_name']=='blimp' and d['metric_name']=='macro_accuracy')
   raw=json.loads(item['raw_json'])['configurations']
   for name,value in raw.items():configs.setdefault(name,[]).append(value['accuracy'])
  per_model[model]={name:sum(values)/len(values) for name,values in configs.items()}
 deltas={name:per_model['autopareto_292'][name]-per_model['autoresearch_compatible'][name] for name in per_model['autopareto_292']}
 contrast={
  'autopareto_higher':sum(delta>0 for delta in deltas.values()),
  'tied':sum(delta==0 for delta in deltas.values()),
  'autopareto_lower':sum(delta<0 for delta in deltas.values()),
  'largest_deficits':sorted(deltas.items(),key=lambda x:x[1])[:12],
  'deltas':deltas,
 }
 write(OUT/'downstream_contrast.json',contrast)
 # Recipe-level diagnostic. Seven selected recipes are descriptive units, so the
 # correlations identify the strongest observed account rather than a cause.
 recipe_rows=[]
 for model in ORDER:
  rs=[x for x in rows if x['model']==model]
  recipe_rows.append({'model':model,'depth':rs[0]['constants'].get('DEPTH'),'parameters':sum(x['parameters'] for x in rs)/len(rs),'heldout_bpb':sum(x['bpb'] for x in rs)/len(rs),'blimp':out[model]['blimp/macro_accuracy']['mean']})
 def corr(a,b): return float(np.corrcoef(a,b)[0,1])
 mechanism={'n_recipes':len(recipe_rows),'recipes':recipe_rows,'correlations':{
  'blimp_vs_heldout_bpb':corr([x['blimp'] for x in recipe_rows],[x['heldout_bpb'] for x in recipe_rows]),
  'blimp_vs_depth':corr([x['blimp'] for x in recipe_rows],[x['depth'] for x in recipe_rows]),
  'blimp_vs_parameters':corr([x['blimp'] for x in recipe_rows],[x['parameters'] for x in recipe_rows])}}
 write(OUT/'downstream_mechanism.json',mechanism)
if __name__=='__main__':main()
