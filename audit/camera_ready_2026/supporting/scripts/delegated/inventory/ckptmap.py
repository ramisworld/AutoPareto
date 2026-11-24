#!/usr/bin/env python3
"""Map local checkpoints to experiments; compare with DB paths and git-preserved manifests (read-only).
Usage: ckptmap.py REPO PT_HASHES_TXT"""
import sqlite3, os, sys, json, subprocess, collections, re
repo, hfile = sys.argv[1], sys.argv[2]
os.chdir(repo)
local = {}
for line in open(hfile):
    h, s, m, p = line.split(None, 3)
    local[p.strip()] = (h, int(s), m)
con = sqlite3.connect('file:results/results.db?mode=ro&immutable=1', uri=True)
exp = {(r[1], r[2]): r for r in con.execute('select id,campaign_id,experiment_number,status,checkpoint_path,training_seed,val_bpb,method,phase,valid_result,exclude_from_analysis from experiments')}
byid = {r[0]: r for r in exp.values()}
trs = collections.defaultdict(list)
for r in con.execute('select id,experiment_id,run_kind,seed,status,checkpoint_path from training_runs'):
    trs[r[1]].append(r)
# git manifests
man = {}
names = subprocess.run(['git', 'ls-tree', '--name-only', 'autopareto/research-infrastructure', 'results/preserved_checkpoints/'], capture_output=True, text=True).stdout.split()
for n in names:
    d = json.loads(subprocess.run(['git', 'show', f'autopareto/research-infrastructure:{n}'], capture_output=True, text=True).stdout)
    for c in d.get('checkpoints', []):
        eid = c.get('id') or c.get('experiment_id')
        man.setdefault(eid, []).append((n.split('/')[-1], c.get('checkpoint_sha256') or c.get('sha256'), c.get('checkpoint_bytes') or c.get('bytes'), c.get('checkpoint_present'), c.get('important'), c.get('role')))
# downstream evaluation checkpoint hashes
ds = collections.defaultdict(list)
for r in con.execute('select id, experiment_id, checkpoint_path, checkpoint_sha256, status from downstream_evaluation_runs'):
    ds[r[1]].append(r)
rows = []
for p, (h, s, m) in sorted(local.items()):
    if not p.startswith('results/checkpoints/'):
        continue
    camp, f = p.split('/')[2], p.split('/')[3]
    num = int(f.split('.')[0])
    e = exp.get((camp, num))
    eid = e[0] if e else None
    dbp = e[4] if e else None
    trp = [t for t in trs.get(eid, []) if t[5]]
    mm = man.get(eid, [])
    mstat = 'no-manifest'
    if mm:
        mstat = 'MATCH' if any(x[1] == h for x in mm) else 'DIFF:' + ','.join((x[1] or 'None')[:12] for x in mm)
    dstat = ''
    if ds.get(eid):
        dstat = 'ds:' + ','.join(('MATCH' if r[3] == h else 'diff:' + r[3][:12]) for r in ds[eid])
    rows.append((p, eid, e[3] if e else None, e[5] if e else None, s, h[:12], (dbp or '')[-60:], len(trp), mstat, dstat))
for r in rows:
    print('\t'.join(map(str, r)))
# experiments whose checkpoint_path is set but file not local
print('\n# DB experiments with checkpoint_path but no local file (by campaign)')
miss = collections.defaultdict(list)
for (camp, num), e in sorted(exp.items(), key=lambda kv: kv[1][0]):
    if e[4]:
        lp = f'results/checkpoints/{camp}/{num:04d}.pt'
        if lp not in local:
            miss[camp].append(e[0])
for c, ids in miss.items():
    print(c, len(ids), ids)
print('\n# manifest entries (git) total', sum(len(v) for v in man.values()), 'distinct exps', len(man))
mf_missing = [(eid, v) for eid, v in man.items() if eid not in [r[1] for r in rows]]
print('# manifest-listed experiments with NO local checkpoint:', sorted(e for e, _ in mf_missing if e))
