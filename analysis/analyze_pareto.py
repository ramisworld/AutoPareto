#!/usr/bin/env python3
from common import *
def main():
 rows=[x for x in read(OUT/'equal_token_runs.json') if x['eligible']];models=[]
 for model in ORDER:
  rs=[x for x in rows if x['model']==model]
  if not rs:continue
  if len({x['protocol'] for x in rs})!=1:raise ValueError('Mixed protocols within recipe: '+model)
  m={'model':model,'protocol':rs[0]['protocol'],'seeds':[x['seed'] for x in rs]}
  for k in ['parameters','tokens','steps','steady_seconds','scientific_seconds','interval_seconds','train_speed','bpb','validation_bpb','speed','memory']:m[k]=summary([x[k] for x in rs])
  models.append(m)
 points=[dict(model=m['model'],protocol=m['protocol'],bpb=m['bpb']['mean'],speed=m['speed']['mean'],memory=m['memory']['mean']) for m in models]
 for m,p in zip(models,points):m['nondominated']=not any(dominates(q,p) for q in points)
 write(OUT/'pareto_analysis.json',{'dominance':'minimize heldout BPB and peak allocated MiB; maximize cached decode tok/s; raw units; strict improvement in at least one objective','aggregation':'descriptive model means over seeds 3,4,5; not the search selection frontier','hypervolume':'not used; no reference-point claim','models':models})
if __name__=='__main__':main()
