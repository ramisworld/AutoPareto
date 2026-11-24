#!/usr/bin/env python3
from common import *
import collections,difflib,hashlib

def classify(row,parent):
 a=row.get('constants',{});b=parent or {}; cats=[]
 groups={'capacity/depth/width':['DEPTH','ASPECT_RATIO','HEAD_DIM'],'attention':['WINDOW_PATTERN'],'batching':['TOTAL_BATCH_SIZE','DEVICE_BATCH_SIZE'],'learning-rate/schedule':['EMBEDDING_LR','UNEMBEDDING_LR','MATRIX_LR','SCALAR_LR','WARMUP_RATIO','WARMDOWN_RATIO','FINAL_LR_FRAC'],'regularization':['WEIGHT_DECAY'],'optimizer':['ADAM_BETAS']}
 for group,keys in groups.items():
  if any(k in a and k in b and a[k]!=b[k] for k in keys):cats.append(group)
 return cats or ['other / interface edit; manual review required']

def main():
 rows=read(OUT/'budget_attempts.json');cur=[x for x in rows if x['current'] and x['checks']['cutoff_bound']]; inv=read(OUT/'artifact_inventory.json'); reports={}; snapshots={}
 for arm in ['T','N']:
  rs=sorted([x for x in cur if x['condition']==arm],key=lambda x:x['attempt']);byid={x['raw_record']['result'].get('experiment_id'):x for x in rs}
  for x in rs:
   req=x['raw_record'].get('scientist',{}).get('handoff',{}).get('agent_request',{})
   parent=safe_constants(req.get('context_files',{}).get('train.py',{}).get('content',''))
   x['edit_categories']=classify(x,parent)
   x['constant_changes']={k:{'before':parent.get(k),'after':v} for k,v in x['constants'].items() if k in parent and parent[k]!=v}
   x['human_classification_verified']=False
   x['depth']=x['constants'].get('DEPTH'); x['width']=x['depth']*x['constants'].get('ASPECT_RATIO',64) if x['depth'] else None
   x['heads']=x['width']//x['constants'].get('HEAD_DIM',128) if x['width'] else None
   parent_id=x.get('parent') or ''
   entry=req.get('context_files',{}).get('train.py',{})
   staged_hash=hashlib.sha256(entry.get('content','').encode()).hexdigest()
   expected=(x['raw_record'].get('starting_source_sha256') if parent_id.startswith('baseline-') else byid.get(parent_id,{}).get('raw_record',{}).get('candidate_source_after_sha256'))
   x['lineage_verified']=bool(expected and staged_hash==entry.get('sha256')==x['raw_record'].get('candidate_source_before_sha256')==expected and (parent_id.startswith('baseline-') or byid[parent_id]['attempt']<x['attempt']))
   if x['attempt']==1:
    snapshots[arm]=req
    for name,entry in req.get('context_files',{}).items():
     if entry.get('content') is not None:
      target=ROOT/'paper/audit/prompts'/arm/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(entry['content'])
    program=req.get('program',{});content=program.get('content') if isinstance(program,dict) else program
    if content: (ROOT/'paper/audit/prompts'/arm/'program.md').write_text(content)
  reports[arm]={'attempts':len(rs),'quality_valid':sum(x['quality_eligible'] for x in rs),'pareto_eligible':sum(x['pareto_eligible'] for x in rs),'failures':sum(x['status']!='completed' for x in rs),'best_bpb':min((x['bpb'] for x in rs if x['quality_eligible']),default=None),'parameters':summary([x['parameters'] for x in rs if x['quality_eligible']]),'categories':dict(collections.Counter(c for x in rs for c in x['edit_categories']))}
 # Exact executed instruction and staged-context diff, never substitute today's prompt.
 for name in ['program.md','train.py','campaign_view.json','prepare.py','study_runtime.py']:
  paths=[ROOT/'paper/audit/prompts'/a/name for a in ['T','N']]
  if all(p.exists() for p in paths):
   diff=''.join(difflib.unified_diff(paths[0].read_text().splitlines(True),paths[1].read_text().splitlines(True),fromfile='T/'+name,tofile='N/'+name))
   (ROOT/'paper/audit'/('executed_'+name.replace('.','_')+'.diff')).write_text(diff)
 checks={}
 for name in ['train.py','prepare.py','study_runtime.py']:
  a=snapshots['T']['context_files'][name];b=snapshots['N']['context_files'][name];checks[name]={'equal_bytes':a['content']==b['content'],'T_sha256':hashlib.sha256(a['content'].encode()).hexdigest(),'N_sha256':hashlib.sha256(b['content'].encode()).hexdigest()}
 write(OUT/'budget_analysis.json',{'arms':reports,'controls':checks,'unit':'one incomplete matched adaptive pair; no independent replication','current_namespace':rel(CURRENT),'rows':[{k:v for k,v in x.items() if k!='raw_record'} for x in cur]})
 print('Budget:',reports)
if __name__=='__main__':main()
