"""Independent read-only recomputation of Table 1 and headline ratios from raw per-run files + snapshot DB."""
import json, sqlite3, statistics, hashlib, math
from pathlib import Path
ROOT = Path('/Users/Rami/Documents/dev/autopareto')
PH = ROOT/'results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1'
DB = PH/'downstream_200m.db'
ORDER = ['baseline','autoresearch_compatible','tpe_quality','motpe_quality','autopareto_292','autopareto_299','autopareto_294']
LAB = dict(zip(ORDER,['Baseline','AutoResearch','TPE','MOTPE','AP292','AP299','AP294']))
c = sqlite3.connect(f'file:{DB}?mode=ro&immutable=1', uri=True); c.row_factory = sqlite3.Row
manifest = json.loads((ROOT/'evaluation/phase1_public_manifest.json').read_text())
pm = {m['model_id']: m for m in manifest['models']}
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows = []
for model in ORDER:
    for seed in (3,4,5):
        d = PH/'runs'/f'phase1-{model}-seed{seed}'
        t = json.loads((d/'training_metrics.json').read_text())
        inf = json.loads((d/'final_inference.json').read_text())
        meta = json.loads((d/'run_metadata.json').read_text())
        timing = [json.loads(l) for l in (d/'step_timing.jsonl').read_text().splitlines() if l.strip()]
        # experiments row for this run in the snapshot DB
        e = c.execute("select * from experiments where method=? and training_seed=? and phase='phase1_downstream'", (model, seed)).fetchall()
        assert len(e)==1, (model, seed, len(e))
        e = e[0]
        ds = c.execute("select * from downstream_evaluation_runs where experiment_id=? and status='completed'", (e['id'],)).fetchall()
        assert len(ds)==1
        ds = ds[0]
        dr = c.execute("select task_name, metric_name, metric_value, sample_count from downstream_task_results where downstream_run_id=?", (ds['id'],)).fetchall()
        held = [x['metric_value'] for x in dr if x['task_name']=='heldout_climbmix_bpb']
        assert len(held)==1
        reps = [r for r in inf['inference_repetitions'] if not r.get('is_warmup')]
        rep_speed = [r['decode_tokens_per_second'] for r in reps]
        rep_mem = [r.get('peak_allocated_cuda_memory_mb') for r in reps]
        rep_kv = [r['incremental_kv_cache_memory_mb'] for r in reps]
        steady = sum(x['step_wall_seconds'] for x in timing if x['zero_based_step_index']>=11)
        steady_tok = sum(x['tokens_processed'] for x in timing if x['zero_based_step_index']>=11)
        total_tok = sum(x['tokens_processed'] for x in timing)
        src = ROOT/pm[model]['source_path']
        rows.append(dict(model=model, seed=seed,
            params=t['parameter_count'], tokens=t['total_training_tokens'], steps=t['optimizer_steps'], tok_per_step=t['tokens_per_optimizer_step'],
            heldout_bpb_db=held[0], heldout_bpb_ckpt=json.loads((d/'checkpoint_evaluations.json').read_text()).get('heldout_bpb',{}).get('metric_value'),
            val_bpb=t['val_bpb'],
            speed_reported=inf['median_cached_decode_tokens_per_second'], speed_recomputed_median=statistics.median(rep_speed), n_reps=len(reps),
            mem_reported=inf['peak_inference_cuda_memory_mb'], mem_reps=sorted(set(rep_mem)),
            kv_reps=sorted(set(rep_kv)), kv_mean=statistics.mean(rep_kv),
            steady_reported=t['steady_state_equivalent_seconds'], steady_recomputed=steady*total_tok/steady_tok,
            step_wall_sum=t['total_scientific_step_wall_seconds'], interval=t['total_scientific_wall_seconds'],
            timing_rows=len(timing), token_sum=total_tok,
            src_sha_file=sha(src), src_sha_train=t['source_sha256'], src_sha_manifest=pm[model]['source_sha256'],
            ckpt_pair=ds['checkpoint_sha256']==t['checkpoint_sha256'], n_ds=len(dr),
            py=meta['software_versions']['python'], torch=meta['software_versions']['torch'], cuda=meta['software_versions']['torch_cuda'], git=meta['software_versions']['git_commit'],
            driver=sorted({g.get('driver_version') for g in inf.get('gpu_telemetry',[]) if g.get('driver_version')}),
            gpu=sorted({g.get('name') for g in inf.get('gpu_telemetry',[]) if g.get('name')}),
            protocol=inf['protocol_version'], correctness=inf['correctness'].get('passed'),
            ds_finished=ds['finished_at'], e_id=e['id'], ds_id=ds['id']))
json.dump(rows, open('table1_per_run.json','w'), indent=1)
print('n runs', len(rows))
bad = [ (r['model'],r['seed']) for r in rows if not (r['tokens']==201326592 and r['steps']*r['tok_per_step']==201326592 and r['timing_rows']==r['steps'] and r['token_sum']==201326592 and r['src_sha_file']==r['src_sha_train']==r['src_sha_manifest'] and r['ckpt_pair'] and r['n_ds']==9 and r['correctness'] is True)]
print('integrity failures:', bad)
print('max |speed_reported - median(reps)|:', max(abs(r['speed_reported']-r['speed_recomputed_median']) for r in rows))
print('max |steady_reported - recomputed|:', max(abs(r['steady_reported']-r['steady_recomputed']) for r in rows))
print('heldout db vs ckpt json max diff:', max(abs(r['heldout_bpb_db']-r['heldout_bpb_ckpt']) for r in rows if r['heldout_bpb_ckpt'] is not None))
print('env sets: py', {r['py'] for r in rows}, 'torch', {r['torch'] for r in rows}, 'cuda', {r['cuda'] for r in rows}, 'git', {r['git'] for r in rows}, 'driver', {tuple(r['driver']) for r in rows}, 'gpu', {tuple(r['gpu']) for r in rows}, 'protocol', {r['protocol'] for r in rows})
print('ds finished range', min(r['ds_finished'] for r in rows), max(r['ds_finished'] for r in rows))
print()
summ = {}
print(f"{'Recipe':13s} {'Mparams':>8s} {'BPB':>8s} {'Decode':>7s} {'MiB':>7s} {'Steady':>7s} | sdBPB sdDec | KV mean | mem set")
for model in ORDER:
    rs = [r for r in rows if r['model']==model]
    m = lambda k: statistics.mean(r[k] for r in rs)
    s = lambda k: statistics.stdev(r[k] for r in rs)
    summ[model] = dict(params=m('params'), bpb=m('heldout_bpb_db'), speed=m('speed_reported'), mem=m('mem_reported'), steady=m('steady_reported'), sd_bpb=s('heldout_bpb_db'), sd_speed=s('speed_reported'), kv=m('kv_mean'), wall_step=m('step_wall_sum'), interval=m('interval'), valbpb=m('val_bpb'))
    x = summ[model]
    print(f"{LAB[model]:13s} {x['params']/1e6:8.2f} {x['bpb']:8.4f} {x['speed']:7.1f} {x['mem']:7.1f} {x['steady']:7.1f} | {x['sd_bpb']:.4f} {x['sd_speed']:.2f} | {x['kv']:.4f} | {sorted({v for r in rs for v in r['mem_reps']})} kv{sorted({v for r in rs for v in r['kv_reps']})}")
ap, ar = summ['autopareto_292'], summ['autoresearch_compatible']
print()
print('DecodeRatio', ap['speed']/ar['speed'])
print('ParameterReduction %', 100*(1-ap['params']/ar['params']))
print('MemoryReduction %', 100*(1-ap['mem']/ar['mem']))
print('KVReduction %', 100*(1-ap['kv']/ar['kv']), ap['kv'], ar['kv'])
print('BPBCost', ap['bpb']-ar['bpb'], ap['bpb'], ar['bpb'])
print('APTime ARTime', ap['steady'], ar['steady'], 'APWall(step-sum) ARWall', ap['wall_step'], ar['wall_step'], 'interval', ap['interval'], ar['interval'])
print('APSpeed ARSpeed', ap['speed'], ar['speed'], 'APMem ARMem', ap['mem'], ar['mem'], 'APParams ARParams', ap['params'], ar['params'])
json.dump(summ, open('table1_summary.json','w'), indent=1)
