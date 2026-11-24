import json, os, glob, collections
S = json.load(open('ns_summary.json'))
R = '/Users/Rami/Documents/dev/autopareto/runs/'
for ns, s in S.items():
    st = s['stats']
    main = [c for c in s['states'] if 'error' not in c and os.path.basename(c['path']) == 'controller_state.json']
    cids = len(glob.glob(R + ns + '/**/*.cid', recursive=True))
    diags = len(glob.glob(R + ns + '/**/scientist_diagnostics/scientist-*', recursive=True))
    rows = []
    for c in main:
        arms = c['arms']
        N = arms.get('pair-a-fixed-tokens-seed3', {}); T = arms.get('pair-a-fixed-time-seed3', {})
        rm = [it.get('result_mtime') for it in c['positions_detail'] if it.get('result_mtime')]
        nonp = {k: v for k, v in c['status_counts'].items() if k != 'pending'}
        rows.append(f"  state={os.path.relpath(c['path'], R)} gpu={str(c['a40_uuid'])[4:12]} n*={c['n_star']} model={c['scientist']['model']}/{c['scientist']['reasoning']} N={dict(N)} T={dict(T)} results={min(rm) if rm else '-'}..{max(rm) if rm else '-'}")
    hl = {os.path.relpath(h['path'], R): h['lines'] for h in s['histories']}
    print(f"{ns} | {st['bytes']/1048576:.1f}MB | files={st['files']} | {st['mtime_min']}..{st['mtime_max']} | cid={cids} diag={diags}")
    for r in rows: print(r)
    for k, v in hl.items(): print(f"  history {k}: {v} lines")
