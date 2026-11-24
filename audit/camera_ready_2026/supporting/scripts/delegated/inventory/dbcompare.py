#!/usr/bin/env python3
"""Read-only chronological comparison of results-family SQLite stores.
Usage: dbcompare.py LABEL=PATH ... (in chronological order)"""
import sqlite3, sys, json

def ro(p):
    return sqlite3.connect(f'file:{p}?mode=ro&immutable=1', uri=True)

def compress(ids):
    ids = sorted(ids)
    out = []; i = 0
    while i < len(ids):
        j = i
        while j + 1 < len(ids) and ids[j + 1] == ids[j] + 1:
            j += 1
        out.append(str(ids[i]) if i == j else f'{ids[i]}-{ids[j]}')
        i = j + 1
    return ','.join(out)

def load(p):
    con = ro(p)
    tabs = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    d = {}
    ecols = [r[1] for r in con.execute('pragma table_info(experiments)')]
    sel = ['id', 'campaign_id', 'experiment_number', 'method', 'status', 'val_bpb', 'phase', 'valid_result', 'exclude_from_analysis']
    sel = [c for c in sel if c in ecols]
    d['exp'] = {r[0]: r[1:] for r in con.execute(f'select {",".join(sel)} from experiments')}
    d['exp_cols'] = sel[1:]
    d['tr'] = {r[0]: r[1:] for r in con.execute('select id, experiment_id, run_kind, seed, status, val_bpb from training_runs')}
    ccols = [r[1] for r in con.execute('pragma table_info(campaigns)')]
    d['camp'] = {r[0]: r[1:] for r in con.execute('select campaign_id, method, status' + (', dashboard_visible, archive_reason' if 'archive_reason' in ccols else '') + ' from campaigns')}
    d['cs'] = {r[0]: r[1:] for r in con.execute('select campaign_id, status, next_experiment_number, attempt_limit from campaign_state')}
    d['ev'] = {r[0]: r[1:] for r in con.execute('select id, campaign_id, experiment_id, training_run_id, event_type, event_time, source from experiment_events')}
    for t in ('confirmation_runs', 'ablation_runs', 'deployment_sessions', 'downstream_evaluation_runs'):
        if t in tabs:
            d[t] = {r[0]: r[1:] for r in con.execute(f'select * from {t}')}
    return d

args = [a.split('=', 1) for a in sys.argv[1:]]
stores = [(lab, load(p)) for lab, p in args]
report = {}
for (la, a), (lb, b) in zip(stores, stores[1:]):
    print(f'\n##### {la}  ->  {lb}')
    for key, name in (('exp', 'experiments'), ('tr', 'training_runs'), ('camp', 'campaigns'), ('cs', 'campaign_state'), ('ev', 'experiment_events'),
                      ('confirmation_runs', 'confirmation_runs'), ('ablation_runs', 'ablation_runs'), ('deployment_sessions', 'deployment_sessions'), ('downstream_evaluation_runs', 'downstream_evaluation_runs')):
        A = a.get(key, {}); B = b.get(key, {})
        added = sorted(set(B) - set(A), key=str); removed = sorted(set(A) - set(B), key=str)
        changed = sorted([k for k in set(A) & set(B) if A[k] != B[k]], key=str)
        if not (added or removed or changed):
            continue
        def fmt(ks):
            if ks and all(isinstance(k, int) for k in ks):
                return compress(ks)
            return ', '.join(map(str, ks))
        print(f'  {name}: +{len(added)} -{len(removed)} ~{len(changed)}')
        if added: print(f'    added: {fmt(added)[:1500]}')
        if removed:
            print(f'    REMOVED: {fmt(removed)[:3000]}')
            if key in ('exp', 'tr', 'camp', 'cs'):
                for k in removed[:60]:
                    print(f'      removed {k}: {A[k]}')
            if key == 'ev':
                from collections import Counter
                c = Counter((A[k][0], A[k][4]) for k in removed)
                print('      removed events by (campaign,event_type):', c.most_common(25))
                c2 = Counter(A[k][1] for k in removed)
                print('      removed events by experiment_id:', sorted(c2.items(), key=lambda x: (x[0] is None, x[0]))[:60])
                c3 = Counter(A[k][2] for k in removed)
                print('      removed events by training_run_id:', sorted(c3.items(), key=lambda x: (x[0] is None, x[0]))[:60])
        if changed:
            print(f'    changed: {fmt(changed)[:1500]}')
            if key in ('exp', 'camp', 'cs', 'tr'):
                for k in changed[:40]:
                    print(f'      ~{k}: {A[k]} -> {B[k]}')
