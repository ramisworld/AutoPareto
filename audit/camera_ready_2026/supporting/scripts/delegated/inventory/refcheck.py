#!/usr/bin/env python3
"""Extract repo-relative path references from docs and check where they exist (read-only).
Usage: refcheck.py REPO RI_FILES_TXT SNAP_FILES_TXT DOC [DOC ...]"""
import os, re, sys, collections
repo, rif, snapf = sys.argv[1], sys.argv[2], sys.argv[3]
docs = sys.argv[4:]
ri = set(l.strip() for l in open(rif))
ri_dirs = set()
for p in ri:
    parts = p.split('/')
    for i in range(1, len(parts)):
        ri_dirs.add('/'.join(parts[:i]))
snap = set(l.strip() for l in open(snapf))
pat = re.compile(r'`([A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-*]+)+/?)`')
prefixes = ('results/', 'runs/', 'study/', 'evaluation/', 'docs/', 'scripts/', 'paper/', 'protected/', 'data/', 'configs/', 'plots/', 'analysis/', 'queue/', 'search/', 'tests/', 'protocol/', 'programs/', 'dashboard/', 'prelaunch_audit/')
summary = collections.Counter()
for d in docs:
    txt = open(os.path.join(repo, d), errors='replace').read()
    refs = sorted(set(m.group(1).rstrip('/') for m in pat.finditer(txt)))
    out = []
    for r in refs:
        if not r.startswith(prefixes) or '*' in r or '<' in r:
            continue
        if os.path.exists(os.path.join(repo, r)):
            st = 'WT'
            if os.path.isfile(os.path.join(repo, r)) and os.path.getsize(os.path.join(repo, r)) == 0:
                st = 'WT-EMPTY'
        elif r in ri or r in ri_dirs:
            st = 'GIT-ONLY(research-infrastructure)'
        elif r in snap or ('results/' + r.split('results/', 1)[-1]) in snap:
            st = 'SNAPSHOT-ONLY'
        else:
            st = 'ABSENT'
        summary[(d, st)] += 1
        if st not in ('WT',):
            out.append(f'    {st}: {r}')
    print(f'== {d}: {len(refs)} backticked refs')
    print('\n'.join(out[:80]))
print()
for k, v in sorted(summary.items()):
    print(k, v)
