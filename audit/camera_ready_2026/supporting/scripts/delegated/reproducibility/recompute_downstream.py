import json, sqlite3, statistics, re, math
from pathlib import Path
ROOT=Path('/Users/Rami/Documents/dev/autopareto')
DB=ROOT/'results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db'
c=sqlite3.connect(f'file:{DB}?mode=ro&immutable=1',uri=True); c.row_factory=sqlite3.Row
ORDER=['baseline','autoresearch_compatible','tpe_quality','motpe_quality','autopareto_292','autopareto_299','autopareto_294']
man={m['model_id']:m for m in json.loads((ROOT/'evaluation/phase1_public_manifest.json').read_text())['models']}
def depth_of(src):
    t=Path(src).read_text()
    m=re.search(r'^DEPTH\s*=\s*(\d+)', t, re.M)
    return int(m.group(1)) if m else None
rows={}
blimp_cfg={}
for model in ORDER:
    res=[]
    for e in c.execute("select id, training_seed, parameter_count from experiments where method=? and phase='phase1_downstream' order by training_seed",(model,)):
        ds=c.execute("select id from downstream_evaluation_runs where experiment_id=? and status='completed'",(e['id'],)).fetchone()
        tr={(r['task_name'],r['metric_name']):r for r in c.execute("select * from downstream_task_results where downstream_run_id=?",(ds['id'],))}
        res.append(dict(seed=e['training_seed'], tr=tr))
    rows[model]=res
# model-level means
summ={}
for model in ORDER:
    rs=rows[model]
    keys=sorted(rs[0]['tr'].keys())
    summ[model]={f'{a}/{b}':statistics.mean(r['tr'][(a,b)]['metric_value'] for r in rs) for a,b in keys}
    summ[model]['_samples']={f'{a}/{b}':sorted({r['tr'][(a,b)]['sample_count'] for r in rs}) for a,b in keys}
    summ[model]['_split']={f'{a}/{b}':rs[0]['tr'][(a,b)]['split'] for a,b in keys}
    summ[model]['_rev']={f'{a}/{b}':rs[0]['tr'][(a,b)]['dataset_revision'] for a,b in keys}
print('metric keys:', [k for k in summ['baseline'] if not k.startswith('_')])
print('samples:', summ['baseline']['_samples'])
print('splits:', summ['baseline']['_split'])
print('revisions:', summ['baseline']['_rev'])
print()
fmt=[('arc_easy/accuracy',100),('arc_easy/length_normalized_accuracy',100),('piqa/accuracy',100),('piqa/length_normalized_accuracy',100),('lambada_openai/exact_match',100),('lambada_openai/perplexity',1),('blimp/macro_accuracy',100),('blimp/micro_accuracy',100),('heldout_climbmix_bpb/bits_per_byte',1)]
for model in ORDER:
    print(f"{model:24s}", ' '.join(f"{summ[model].get(k,float('nan'))*m:8.4f}" for k,m in fmt))
# BLiMP configuration contrast
per={}
for model in ('autopareto_292','autoresearch_compatible'):
    cfg={}
    for r in rows[model]:
        raw=json.loads(r['tr'][('blimp','macro_accuracy')]['raw_json'])['configurations']
        for name,v in raw.items(): cfg.setdefault(name,[]).append(v['accuracy'])
    per[model]={k:sum(v)/len(v) for k,v in cfg.items()}
    print(model, 'n configs', len(cfg), 'per-config n seeds', sorted({len(v) for v in cfg.values()}))
d={k:per['autopareto_292'][k]-per['autoresearch_compatible'][k] for k in per['autopareto_292']}
print('BLiMP AP lower', sum(x<0 for x in d.values()), 'tied', sum(x==0 for x in d.values()), 'higher', sum(x>0 for x in d.values()))
# tolerance check of ties
print('near-ties |delta|<1e-12:', sum(abs(x)<1e-12 for x in d.values()), ' |delta|<=0.0005:', sum(abs(x)<=0.0005 for x in d.values()))
# total blimp pairs
raw=json.loads(rows['autopareto_292'][0]['tr'][('blimp','macro_accuracy')]['raw_json'])
print('blimp raw keys', list(raw.keys())[:10])
tot=sum(v.get('sample_count', v.get('n', 0)) for v in raw['configurations'].values())
print('blimp total pairs from configs:', tot, 'example cfg:', list(raw['configurations'].items())[0])
# correlations
def corr(a,b):
    ma,mb=statistics.mean(a),statistics.mean(b)
    num=sum((x-ma)*(y-mb) for x,y in zip(a,b)); den=math.sqrt(sum((x-ma)**2 for x in a)*sum((y-mb)**2 for y in b)); return num/den
bl=[summ[m]['blimp/macro_accuracy'] for m in ORDER]
hb=[summ[m]['heldout_climbmix_bpb/bits_per_byte'] for m in ORDER]
dp=[depth_of(ROOT/man[m]['source_path']) for m in ORDER]
pc=[statistics.mean(e['parameter_count'] for e in c.execute("select parameter_count from experiments where method=? and phase='phase1_downstream'",(m,))) for m in ORDER]
print('depths', dict(zip(ORDER,dp)))
print('r(BLiMP, heldout BPB) =', corr(bl,hb))
print('r(BLiMP, depth) =', corr(bl,dp))
print('r(BLiMP, params) =', corr(bl,pc))
# Spearman as robustness
def rank(v):
    s=sorted(range(len(v)), key=lambda i:v[i]); r=[0]*len(v)
    i=0
    while i<len(s):
        j=i
        while j+1<len(s) and v[s[j+1]]==v[s[i]]: j+=1
        for k in range(i,j+1): r[s[k]]=(i+j)/2+1
        i=j+1
    return r
print('spearman(BLiMP, heldout)=', corr(rank(bl),rank(hb)), ' spearman(BLiMP, depth)=', corr(rank(bl),rank(dp)))
json.dump({m:{k:v for k,v in summ[m].items() if not k.startswith('_')} for m in ORDER}, open('downstream_summary.json','w'), indent=1)
