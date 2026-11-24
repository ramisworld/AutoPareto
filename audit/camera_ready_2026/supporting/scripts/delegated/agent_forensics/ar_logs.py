import re, sys, glob, os
sys.path.insert(0, os.path.dirname(__file__))
from q import conn, DB
c = conn(DB)
L = "/Users/Rami/Documents/dev/autopareto/results/logs"
rows = c.execute("select id, experiment_number n, status, decision, parent_experiment_id par, val_bpb from experiments where campaign_id='autoresearch-seed42-20260723' order by n").fetchall()
bpb_by_id = {r['id']: r['val_bpb'] for r in c.execute("select id, val_bpb from experiments")}
for r in rows:
    n = r['n']
    tl = f"{L}/autoresearch-seed42-20260723-{n:04d}.log"
    il = f"{L}/autoresearch-seed42-20260723-{n:04d}-inference.log"
    log_bpb = None; tb = None; inf = None
    if os.path.exists(tl):
        t = open(tl, errors='replace').read()
        m = re.findall(r"val_bpb:\s+([0-9.]+)", t)
        log_bpb = float(m[-1]) if m else None
        m2 = re.findall(r"total_batch_size\D*(\d+)|TOTAL_BATCH_SIZE\D*(\d+)", t)
        last = t.strip().splitlines()[-1][:120] if t.strip() else ''
    else:
        last = 'NO TRAIN LOG'
    if os.path.exists(il):
        it = open(il, errors='replace').read()
        m = re.findall(r"(RuntimeError: .*|AssertionError.*)", it)
        inf = m[-1][:110] if m else 'ok?'
    par_bpb = bpb_by_id.get(r['par'])
    better = (log_bpb is not None and par_bpb is not None and log_bpb < par_bpb)
    print(n, r['id'], r['status'], r['decision'], 'par', r['par'], par_bpb and round(par_bpb,6), 'db', r['val_bpb'] and round(r['val_bpb'],6), 'log', log_bpb, 'BETTER_THAN_PARENT' if better and r['decision']!='retained' else '', '|', inf if r['status']!='completed' else '', '|', last if log_bpb is None else '')
