#!/usr/bin/env python3
"""Compare queue/<campaign>/N-train.py with results/candidates/<campaign>/NNNN-train.py (read-only)."""
import os, hashlib, re, sys
root = sys.argv[1]
def h(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()
for d in sorted(os.listdir(os.path.join(root, 'queue'))):
    qd = os.path.join(root, 'queue', d)
    cd = os.path.join(root, 'results', 'candidates', d)
    same = differ = nocand = 0; difflist = []
    for f in sorted(os.listdir(qd)):
        if not f.endswith('.py'):
            continue
        m = re.match(r'(\d+)-train\.py$', f)
        if m:
            c = os.path.join(cd, f'{int(m.group(1)):04d}-train.py')
            if os.path.exists(c):
                if h(os.path.join(qd, f)) == h(c): same += 1
                else: differ += 1; difflist.append(f)
            else:
                nocand += 1; difflist.append(f + '(no cand)')
        else:
            # candidate-train.py: compare against each candidate file
            hs = h(os.path.join(qd, f))
            matches = [g for g in (os.listdir(cd) if os.path.isdir(cd) else []) if g.endswith('.py') and h(os.path.join(cd, g)) == hs]
            print(f'  {d}/{f}: sha256={hs[:16]} matches candidates {matches}')
            continue
    if same or differ or nocand:
        print(f'{d}: identical={same} differ={differ} no_candidate={nocand} {difflist[:10]}')
