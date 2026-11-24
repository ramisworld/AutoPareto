#!/usr/bin/env python3
import json, sys, collections
S = json.load(open(sys.argv[1]))
only = sys.argv[2:]
for ns, s in S.items():
    if only and ns not in only:
        continue
    st = s['stats']
    print(f"\n=== {ns}  {st['bytes']/1048576:.1f}MB files={st['files']} mtime={st['mtime_min']}..{st['mtime_max']}")
    print('   subdirs:', s['subdirs'])
    for c in s['states']:
        if 'error' in c:
            print('   STATE ERROR', c['path'], c['error']); continue
        rp = c['path'].split('/runs/')[1]
        print(f"   [state] {rp} {c['bytes']/1048576:.1f}MB mtime={c['mtime']} schema={c['schema']} study={c['study_id']} gpu={str(c['a40_uuid'])[:12]} n_star={c['n_star']} manifest={c['manifest_sha256']} sched={str(c['schedule_sha256'])[:8]}")
        print(f"      scientist={c['scientist']} exec_models={c['exec_models']} exec_reasoning={c['exec_reasoning']}")
        if 'amendments' in c:
            print(f"      amendments={c['amendments']} {c.get('amendment_ids')}")
        print(f"      status_counts={c['status_counts']} arms={c['arms']}")
        for it in c['positions_detail']:
            extra = {k: v for k, v in it.items() if k not in ('pos', 'arm', 'attempt', 'status', 'result_file', 'result_exists', 'result_mtime')}
            rm = it.get('result_mtime')
            arm = (it['arm'] or '').replace('pair-a-fixed-', '').replace('-seed3', '')
            print(f"        p{it['pos']:02d} {arm}{it['attempt']} {it['status']} res={'Y' if it.get('result_exists') else ('N' if 'result_file' in it else '-')} {rm or ''} {json.dumps(extra)[:300] if extra else ''}")
    for h in s['histories']:
        rp = h['path'].split('/runs/')[1]
        recs = h['records']
        print(f"   [history] {rp} lines={h['lines']} {h['bytes']/1048576:.1f}MB mtime={h['mtime']}")
        for r in recs:
            print(f"        pos={r.get('pos')} att={r.get('attempt')} {r.get('status')} {r.get('model')}/{r.get('reasoning')} dec={r.get('decision')} q={r.get('quality_status')} bpb={r.get('bpb')} tok/s={r.get('tok_s')} tb={r.get('total_batch_size')} cat={r.get('change_category')} parent={r.get('parent')} after={r.get('after_sha')} man={r.get('manifest_sha256')} nstar={r.get('n_star')} fail={r.get('failure')}")
    for k, v in s['position_files'].items():
        v = sorted(v)
        print(f"   [position files] {k}: {len(v)} files, {v[0][0][:12]}..{v[-1][0][:12]} mtimes {min(x[1] for x in v)}..{max(x[1] for x in v)}")
