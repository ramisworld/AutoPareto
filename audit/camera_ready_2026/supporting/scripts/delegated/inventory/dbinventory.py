#!/usr/bin/env python3
"""Read-only SQLite inventory. Opens every DB with mode=ro&immutable=1.
Usage: dbinventory.py OUT_JSON DB [DB ...]"""
import sqlite3, sys, json, os, hashlib

TIME_COLS = ('started_at', 'finished_at', 'created_at', 'updated_at', 'event_time',
             'measured_at', 'archived_at', 'completed_at', 'timestamp', 'recorded_at')

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def inv(path):
    con = sqlite3.connect(f'file:{path}?mode=ro&immutable=1', uri=True)
    out = {'path': path, 'bytes': os.path.getsize(path), 'sha256': sha(path), 'tables': {}}
    names = [r[0] for r in con.execute("select name from sqlite_master where type='table' order by name")]
    gmin, gmax = None, None
    for t in names:
        cols = [r[1] for r in con.execute(f'pragma table_info("{t}")')]
        n = con.execute(f'select count(*) from "{t}"').fetchone()[0]
        rec = {'rows': n, 'cols': len(cols)}
        if 'id' in cols:
            rec['max_id'] = con.execute(f'select max(id) from "{t}"').fetchone()[0]
            rec['min_id'] = con.execute(f'select min(id) from "{t}"').fetchone()[0]
        tmin, tmax = None, None
        for c in cols:
            if c in TIME_COLS:
                a, b = con.execute(f'select min("{c}"), max("{c}") from "{t}" where "{c}" is not null and "{c}" != \'\'').fetchone()
                if a is not None:
                    a, b = str(a), str(b)
                    tmin = a if tmin is None or a < tmin else tmin
                    tmax = b if tmax is None or b > tmax else tmax
        if tmin:
            rec['tmin'] = tmin; rec['tmax'] = tmax
            gmin = tmin if gmin is None or tmin < gmin else gmin
            gmax = tmax if gmax is None or tmax > gmax else gmax
        out['tables'][t] = rec
    if 'sqlite_sequence' in names:
        out['sqlite_sequence'] = dict(con.execute('select name, seq from sqlite_sequence').fetchall())
    out['tmin'] = gmin; out['tmax'] = gmax
    con.close()
    return out

res = []
for p in sys.argv[2:]:
    try:
        res.append(inv(p))
    except Exception as e:
        res.append({'path': p, 'error': repr(e)})
json.dump(res, open(sys.argv[1], 'w'), indent=1)
for r in res:
    if 'error' in r:
        print(r['path'], 'ERROR', r['error']); continue
    t = r['tables']
    e = t.get('experiments', {}); tr = t.get('training_runs', {}); ev = t.get('experiment_events', {}); c = t.get('campaigns', {})
    print(f"{os.path.relpath(r['path'])}\n  bytes={r['bytes']} sha256={r['sha256'][:16]} tables={len(t)} "
          f"experiments={e.get('rows')} max_exp_id={e.get('max_id')} training_runs={tr.get('rows')} max_tr={tr.get('max_id')} "
          f"events={ev.get('rows')} max_ev={ev.get('max_id')} campaigns={c.get('rows')} t=[{r['tmin']} .. {r['tmax']}]")
    print('  ' + '; '.join(f"{k}:{v['rows']}" for k, v in t.items()))
