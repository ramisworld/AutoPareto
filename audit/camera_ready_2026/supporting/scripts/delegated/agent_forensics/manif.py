import json, sys, re
REDACT = re.compile(r'[A-Za-z0-9._-]+@ssh\.runpod\.io|ssh\.runpod\.io|\b\d{1,3}(?:\.\d{1,3}){3}\b')
def red(s):
    return REDACT.sub('<runpod-ssh-endpoint>', s)
def flat(d, pre=''):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f'{pre}{k}.'))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(flat(v, f'{pre}{i}.'))
    else:
        out[pre[:-1]] = d
    return out
prev = None
for p in sys.argv[1:]:
    d = json.load(open(p))
    f = flat(d)
    print('=' * 100)
    print(p.split('/runs/')[-1])
    keys = [k for k in f if re.search(r'(^|\.)(schema|status|frozen_at_utc|created_at_utc|reason|amendment|purpose|description|change|summary|scope|decision|requeue|abandon|rationale|note|notes)(\.|$)', k, re.I)]
    for k in keys[:60]:
        v = f[k]
        s = red(json.dumps(v))
        print(f'  {k}: {s[:400]}')
    if prev is not None:
        diffs = [k for k in set(f) | set(prev) if f.get(k) != prev.get(k) and ('immutable' in k or 'program' in k or 'sha256' in k)]
        print('  -- hash/program diffs vs previous file:', len(diffs))
        for k in sorted(diffs)[:40]:
            print('     ', k, str(prev.get(k))[:16], '->', str(f.get(k))[:16])
    prev = f
