#!/usr/bin/env python3
"""Read immutable evidence, preserve failed attempts, and validate metric provenance."""
import argparse, collections, csv, difflib, hashlib, json, sqlite3, subprocess
from datetime import datetime,timezone
from pathlib import Path
from common import *

def inventory():
 records=[]; errors=[]
 for base in ['data','results','runs','study','configs','protocol','evaluation','scripts','protected','docs','search','queue']:
  for p in sorted((ROOT/base).rglob('*')):
   if not p.is_file() or p.name.startswith('._') or '__pycache__' in p.parts:continue
   st=p.stat(); rec=dict(path=rel(p),bytes=st.st_size,mtime_utc=datetime.fromtimestamp(st.st_mtime,timezone.utc).isoformat())
   # Full hashes of all evidence, including checkpoint bytes; no deserialization.
   rec['sha256']=sha(p)
   if p.suffix=='.json':
    try:read(p)
    except Exception as e:errors.append(dict(path=rel(p),error=str(e)))
   records.append(rec)
 for p in sorted(ROOT.iterdir()):
  if p.is_file() and p.suffix in ['.py','.md','.toml','.json','.yaml','.yml']:
   st=p.stat();records.append({'path':rel(p),'bytes':st.st_size,'mtime_utc':datetime.fromtimestamp(st.st_mtime,timezone.utc).isoformat(),'sha256':sha(p)})
 write(OUT/'artifact_inventory.json',records);write(OUT/'parse_errors.json',errors)
 return records,errors

def build(cutoff):
 OUT.mkdir(parents=True,exist_ok=True)
 c=sqlite3.connect('file:'+str(DB)+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
 tables={t:[dict(r) for r in c.execute('select * from "'+t+'"')] for (t,) in c.execute("select name from sqlite_master where type='table'") if t!='sqlite_sequence'}
 # Keep tables and raw field names, including agent text and all failures.
 for t,rows in tables.items():write(OUT/('db_'+t+'.json'),rows)
 experiments=tables['experiments']; campaigns={x['campaign_id']:x for x in tables['campaigns']}
 downstream=tables['downstream_task_results']; dsruns=tables['downstream_evaluation_runs']
 panel=[];issues=[]
 pm={m['model_id']:m for m in read(ROOT/'evaluation/phase1_public_manifest.json')['models']}
 for p in sorted((PHASE/'runs').glob('phase1-*')):
  if not (p/'training_metrics.json').exists():continue
  m=read(p/'run_metadata.json');t=read(p/'training_metrics.json');inf=read(p/'final_inference.json');ev=read(p/'checkpoint_evaluations.json')
  model=m['model_identifier'];seed=m['seed'];e=next(x for x in experiments if x['method']==model and x['training_seed']==seed and x['phase']=='phase1_downstream')
  ds=next(x for x in dsruns if x['experiment_id']==e['id'] and x['status']=='completed')
  dr=[x for x in downstream if x['downstream_run_id']==ds['id']]
  source=ROOT/pm[model]['source_path'];const=safe_constants(source.read_text())
  timing=[json.loads(l) for l in (p/'step_timing.jsonl').read_text().splitlines() if l.strip()]
  checks={'tokens':t['total_training_tokens']==201326592,'step_count':t['optimizer_steps']*t['tokens_per_optimizer_step']==201326592,'timing_rows':len(timing)==t['optimizer_steps'],'source_hash':sha(source)==t['source_sha256']==pm[model]['source_sha256'],'inference':inf['status']=='completed' and inf['correctness']['passed'],'checkpoint_pairing':ds['checkpoint_sha256']==t['checkpoint_sha256'],'complete_suite':len(dr)==9,'cutoff':before_cutoff(ds['finished_at'],cutoff)}
  pass_summaries=inf['measurement_pass_summaries']
  measured_repetitions=inf.get('inference_repetitions',[])
  if not measured_repetitions: raise ValueError('Missing measured inference repetitions: '+p.name)
  prefill_kv=summary([x['prefill_kv_cache_memory_mb'] for x in measured_repetitions])
  incremental_kv=summary([x['incremental_kv_cache_memory_mb'] for x in measured_repetitions])
  # Preserve summaries; individual timing repetitions are not all present in this bundle.
  row=dict(id=p.name,model=model,seed=seed,source=rel(source),source_sha256=sha(source),experiment_id=e['id'],protocol=inf['protocol_version'],constants=const,parameters=t['parameter_count'],tokens=t['total_training_tokens'],steps=t['optimizer_steps'],steady_seconds=t['steady_state_equivalent_seconds'],scientific_seconds=t['total_scientific_step_wall_seconds'],interval_seconds=t['total_scientific_wall_seconds'],train_speed=t['steady_state_training_tokens_per_second'],bpb=next(x['metric_value'] for x in dr if x['task_name']=='heldout_climbmix_bpb'),validation_bpb=t['val_bpb'],speed=inf['median_cached_decode_tokens_per_second'],memory=inf['peak_inference_cuda_memory_mb'],prefill_kv_memory=prefill_kv['mean'],incremental_kv_memory=incremental_kv['mean'],downstream=dr,milestones=ev['checkpoints'],checks=checks,eligible=all(checks.values()),finished_at=ds['finished_at'],provenance={k:rel(p/n) for k,n in [('training','training_metrics.json'),('timing','step_timing.jsonl'),('inference','final_inference.json'),('evaluation','checkpoint_evaluations.json'),('environment','run_metadata.json')]},downstream_source=rel(DB)+'#downstream_run='+str(ds['id']))
  panel.append(row)
  if not row['eligible']:issues.append(dict(id=row['id'],failed=[k for k,v in checks.items() if not v]))
 write(OUT/'equal_token_runs.json',panel)
 # Every history namespace stays separate. Never deduplicate by bare experiment ID.
 histories=[]
 observation_path=ROOT/'paper/audit/first_observed_records.json'
 observations=read(observation_path) if observation_path.exists() else {}
 observation_time=datetime.now(timezone.utc).isoformat()
 for p in sorted((ROOT/'runs').rglob('history.jsonl')):
  for line,l in enumerate(p.read_text().splitlines(),1):
   try:d=json.loads(l)
   except Exception as e:issues.append(dict(path=rel(p),line=line,error=str(e)));continue
   r=d.get('result',{});inf=r.get('inference_result',{});decision=r.get('pareto_decision',{})
   source=p.parent/'sources'/((r.get('experiment_id') or f"{d.get('campaign_id')}-attempt-{d.get('campaign_attempt',0):02}")+'.train.py')
   constants=safe_constants(source.read_text()) if source.exists() else {}
   runroot=p.parent.parent; official=runroot==CURRENT
   record_hash=hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest()
   observations.setdefault(record_hash,observation_time)
   observed=observations[record_hash]
   # First observation is an upper bound on completion; file mtime is not authoritative.
   checks=budget_result_checks(d,sha(source) if source.exists() else None,observed,cutoff)
   checks['token_step_product']=bool(constants.get('TOTAL_BATCH_SIZE') and positive_finite(r.get('optimizer_updates')) and r.get('total_tokens')==r['optimizer_updates']*constants['TOTAL_BATCH_SIZE'])
   if official:
    manifest_path=next((CURRENT/'provenance').glob('*.json'));manifest=read(manifest_path)
    checks['executed_manifest']=d.get('manifest_sha256')==sha(manifest_path)
    checks['inference_protocol']=inf.get('protocol_version')==manifest['inference']['protocol_version']
   row=dict(id=rel(runroot)+':'+str(d.get('contract_id'))+':'+str(d.get('campaign_attempt')),namespace=rel(runroot),current=official,condition=d.get('contract_id'),attempt=d.get('campaign_attempt'),status=d.get('status'),parameters=r.get('num_params'),tokens=r.get('total_tokens'),steps=r.get('optimizer_updates'),seconds=r.get('protected_measured_training_seconds'),bpb=r.get('validation_bpb'),speed=inf.get('absolute_candidate_tok_s',inf.get('cached_decode_tokens_per_second_median')),normalized_speed=inf.get('normalized_decode_throughput'),memory=inf.get('peak_allocated_inference_memory_bytes'),speed_status=inf.get('normalized_inference_status',inf.get('inference_status')),decision=decision,protocol=inf.get('protocol_version'),manifest_sha256=d.get('manifest_sha256'),program_sha256=d.get('program_sha256'),starting_source_sha256=d.get('starting_source_sha256'),agent=d.get('agent_model_identifier'),reasoning=d.get('reasoning_setting'),seed=d.get('training_seed'),runtime_sha256=d.get('study_runtime_sha256'),parent=d.get('parent_experiment_id'),hypothesis=d.get('hypothesis'),intended_change=d.get('intended_change'),category=d.get('change_category'),constants=constants,warmup=r.get('warmup_lifecycle'),validation_provenance=r.get('protected_validation_provenance'),error=r.get('error'),checks=checks,quality_eligible=all(checks.values()),pareto_eligible=all(checks.values()) and decision.get('pareto_eligible',decision.get('decision') in ('retained','dominated')) and 'invalid' not in decision.get('reason',''),source=rel(source) if source.exists() else None,history_source=rel(p),history_line=line,observed_by_utc=observed,timestamp_basis='first audit observation upper bound; no authoritative completion timestamp',candidate_patch=d.get('candidate_patch'),raw_record=d)
   histories.append(row)
 write(observation_path,observations)
 assert_unique_ids(histories)
 write(OUT/'budget_attempts.json',histories)
 # Historical experiments retain database columns and add code constants/provenance.
 for e in experiments:
  e['source_database']=rel(DB);e['campaign_record']=campaigns.get(e['campaign_id']);e['constants']={}
  name=f"{e['experiment_number']:04}-train.py";p=ROOT/'results/candidates'/e['campaign_id']/name
  if p.exists():
   e['source_path']=rel(p);e['source_sha256']=sha(p);source_text=p.read_text();e['constants']=safe_constants(source_text)
   try:ast.parse(source_text)
   except (SyntaxError,ValueError) as error:e['source_parse_error']=str(error);issues.append({'path':rel(p),'error':str(error),'use':'historical constants missing; raw record retained'})
  e['within_cutoff']=before_cutoff(e.get('finished_at'),cutoff)
 write(OUT/'historical_experiments.json',experiments)
 write(OUT/'validation_report.json',dict(cutoff=cutoff,issues=issues,panel_eligible=sum(x['eligible'] for x in panel),panel_total=len(panel),current_attempts=sum(x['current'] for x in histories),current_quality_eligible=sum(x['current'] and x['quality_eligible'] for x in histories)))
 print('Panel:',len(panel),'eligible:',sum(x['eligible'] for x in panel),'current attempts:',sum(x['current'] for x in histories),'issues:',len(issues))
 if issues:print(str(issues)[:2000])

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--cutoff',default=DEFAULT_CUTOFF);a.add_argument('--inventory',action='store_true');x=a.parse_args();build(x.cutoff)
 if x.inventory: inventory()
