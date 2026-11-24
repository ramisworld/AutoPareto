import json, sys, os, hashlib
ns = sys.argv[1]
tot = ok = missing = mismatch = 0
for arm in ('tokens', 'time'):
    d = f'{ns}/pair-a-fixed-{arm}-seed3'
    for line in open(f'{d}/history.jsonl'):
        r = json.loads(line)
        sc = r.get('scientist') or {}
        for which, diag_key, sha_key in (('child', 'process_diagnostics', 'stdout_sha256'), ('parent', 'parent_selection_process_diagnostics', 'parent_selection_stdout_sha256')):
            pdx = sc.get(diag_key) or {}
            sp = pdx.get('stdout_path')
            if not sp:
                continue
            tot += 1
            inv = sp.split('/scientist_diagnostics/')[1].split('/')[0]
            local = f'{d}/scientist_diagnostics/{inv}/stdout.jsonl'
            if not os.path.exists(local):
                missing += 1; print('MISSING', arm, r.get('global_position'), which, inv); continue
            h = hashlib.sha256(open(local, 'rb').read()).hexdigest()
            if h == sc.get(sha_key):
                ok += 1
            else:
                mismatch += 1; print('MISMATCH', arm, r.get('global_position'), which, inv, h[:12], str(sc.get(sha_key))[:12])
print(ns, 'stdout refs', tot, 'present+hash-match', ok, 'missing', missing, 'hash-mismatch', mismatch)
