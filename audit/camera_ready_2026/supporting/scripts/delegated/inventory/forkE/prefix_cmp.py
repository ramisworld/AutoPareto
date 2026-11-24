import json, sys, itertools
S = json.load(open('ns_summary.json'))
H = {}
for ns, s in S.items():
    for h in s['histories']:
        rel = h['path'].split('/runs/')[1]
        arm = 'T' if 'fixed-time' in rel else 'N'
        key = rel.rsplit('/', 2)[0] if rel.count('/') >= 2 else ns
        H[(rel.rsplit('/pair-a-fixed', 1)[0], arm)] = h['line_sha256']
keys = sorted(H)
# For each pair with same arm, report common prefix length if > 0
for arm in 'NT':
    ks = [k for k in keys if k[1] == arm and H[k]]
    print(f'== arm {arm}')
    for a, b in itertools.permutations(ks, 2):
        la, lb = H[a], H[b]
        n = 0
        while n < min(len(la), len(lb)) and la[n] == lb[n]:
            n += 1
        if n and len(la) <= len(lb):
            tag = 'IDENTICAL' if la == lb else ('PREFIX' if n == len(la) else f'diverge@{n+1}')
            if a < b or tag != 'IDENTICAL':
                print(f'  {a[0]} ({len(la)}) vs {b[0]} ({len(lb)}): common={n} {tag}')
