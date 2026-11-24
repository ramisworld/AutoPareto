import ast, json, sqlite3, hashlib
from pathlib import Path
ROOT=Path('/Users/Rami/Documents/dev/autopareto')
DB=ROOT/'results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db'
c=sqlite3.connect(f'file:{DB}?mode=ro&immutable=1',uri=True); c.row_factory=sqlite3.Row
def consts(text):
    env={}
    def val(n):
        if isinstance(n,ast.Constant): return n.value
        if isinstance(n,ast.Name): return env[n.id]
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.USub): return -val(n.operand)
        if isinstance(n,ast.BinOp):
            a,b=val(n.left),val(n.right)
            import operator as o
            ops={ast.Pow:o.pow,ast.Mult:o.mul,ast.Add:o.add,ast.Sub:o.sub,ast.Div:o.truediv,ast.FloorDiv:o.floordiv}
            return ops[type(n.op)](a,b)
        if isinstance(n,ast.Tuple): return tuple(val(x) for x in n.elts)
        raise ValueError
    try: tree=ast.parse(text)
    except SyntaxError: return None
    for n in tree.body:
        if isinstance(n,ast.Assign):
            try: v=val(n.value)
            except Exception: continue
            for t in n.targets:
                if isinstance(t,ast.Name) and t.id.isupper(): env[t.id]=v
    return env
rows={r['id']:dict(r) for r in c.execute("select id, campaign_id, experiment_number, status, val_bpb, inference_tokens_per_second, peak_inference_cuda_memory_mb, parameter_count, parent_experiment_id, decision, crash_or_failure_reason, length(code_diff) cd from experiments where campaign_id='autopareto-seed0-20260731' or id=8")}
def src(r):
    if r['id']==8: p=ROOT/'results/candidates'/r['campaign_id']/f"{r['experiment_number']:04}-train.py"
    else: p=ROOT/'results/candidates'/r['campaign_id']/f"{r['experiment_number']:04}-train.py"
    return p
LBL={1:'batch 2^19->2^17',2:'matrix LR .04->.03',3:'KV heads 4->2',4:'depth 8->7',5:'matrix LR .04->.03',6:'depth 7->6; width 512->384',7:'depth 6->5; width 384->320',8:'depth 5->4; width 320->256',9:'width 256->320',10:'width 320->384',11:'matrix LR .04->.03',12:'matrix LR .04->.035',13:'embed LR .6->.5',14:'embed LR .5->.4',15:'head LR .004->.0035',16:'head LR .004->.0045',17:'depth 4->3',18:'width 256->512',19:'matrix LR .04->.035',20:'embed LR .6->.5',21:'embed LR .5->.45',22:'head LR .004->.0035',23:'KV heads 4->2',24:'depth 4->3; width 512->384',25:'width 384->512'}
FIG_BPB=[1.2137,1.1063,1.1061,1.1071,1.1005,1.1003,1.0978,1.0966,1.1573,1.1068,1.1054,1.0964,1.0963,1.0954,1.0973,1.0958,1.0992,None,1.0957,1.0954,1.0934,1.0938,1.0933,1.0995,1.1163,1.0974]
FIG_TPS=[164,165,163,156,185,184,217,251,312,325,291,223,258,252,216,257,226,None,313,312,311,320,313,287,351,408]
order=[8]+sorted([i for i in rows if i!=8], key=lambda i: rows[i]['experiment_number'])
mism=[]
cache={}
for k,i in enumerate(order):
    r=rows[i]; p=src(r)
    cache[i]=consts(p.read_text()) if p.exists() else None
for k,i in enumerate(order):
    r=rows[i]
    bpb=None if r['val_bpb'] is None else round(r['val_bpb'],4)
    tps=None if r['inference_tokens_per_second'] is None else round(r['inference_tokens_per_second'])
    fb, ft = FIG_BPB[k], FIG_TPS[k]
    ok = (bpb==fb or (r['status']!='completed' and fb is None)) and (tps==ft or (r['status']!='completed' and ft is None))
    if not ok: mism.append((i, bpb, fb, tps, ft))
    par=r['parent_experiment_id']
    diff={}
    if i!=8 and cache.get(i) is not None and cache.get(par) is not None:
        a,b=cache[par],cache[i]
        diff={kk:(a.get(kk),b.get(kk)) for kk in sorted(set(a)|set(b)) if a.get(kk)!=b.get(kk)}
    width = None
    if cache.get(i):
        cst=cache[i]
        if 'DEPTH' in cst and 'ASPECT_RATIO' in cst: width=cst['DEPTH']*cst['ASPECT_RATIO']
    print(f"A{r['experiment_number'] if i!=8 else 0:02d} id={i} parent={par} status={r['status']} dec={r['decision']} bpb={bpb} fig={fb} tps={tps} fig={ft} params={r['parameter_count']} src={'yes' if cache.get(i) is not None else 'MISSING'} | label='{LBL.get(r['experiment_number'],'baseline') if i!=8 else 'baseline'}' | const_diff={diff}")
print('figure value mismatches:', mism)
main_path=[8,113,116,118,119,120,130,131,132,134]
for a,b in zip(main_path, main_path[1:]):
    print('path edge', a,'->',b, 'recorded parent of', b, '=', rows[b]['parent_experiment_id'], 'OK' if rows[b]['parent_experiment_id']==a else 'MISMATCH')
for a,b in [(134,136),(136,137)]:
    print('sec edge', a,'->',b, 'recorded parent =', rows[b]['parent_experiment_id'])
print('A17 row:', {k:rows[129][k] for k in ['status','crash_or_failure_reason','cd']} if 129 in rows else 'n/a')
