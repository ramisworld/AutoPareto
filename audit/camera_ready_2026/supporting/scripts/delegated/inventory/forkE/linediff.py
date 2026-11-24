import json, sys, re
a, b, arm = sys.argv[1], sys.argv[2], sys.argv[3]
fa = open(f'{a}/pair-a-fixed-{arm}-seed3/history.jsonl').read().splitlines()
fb = open(f'{b}/pair-a-fixed-{arm}-seed3/history.jsonl').read().splitlines()
def walk(o, p=''):
    if isinstance(o, dict):
        for k, v in o.items(): yield from walk(v, p + '.' + k)
    elif isinstance(o, list):
        for i, v in enumerate(o): yield from walk(v, p + f'[{i}]')
    else: yield p, o
def norm(s, ns):
    return s.replace(ns, '<NS>') if isinstance(s, str) else s
for i in range(min(len(fa), len(fb))):
    ra, rb = json.loads(fa[i]), json.loads(fb[i])
    da = dict(walk(ra)); db = dict(walk(rb))
    diffs = [k for k in set(da) | set(db) if da.get(k) != db.get(k)]
    ndiffs = [k for k in diffs if norm(da.get(k), a) != norm(db.get(k), b)]
    print(f'line {i+1} pos={ra.get("global_position")}/{rb.get("global_position")}: raw-diff-fields={len(diffs)} after-namespace-normalization={len(ndiffs)} {sorted(ndiffs)[:8]}')
