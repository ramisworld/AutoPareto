#!/usr/bin/env python3
"""Read-only summary of runs/ namespaces. Writes JSON summary to argv[2]."""
import os, sys, json, hashlib, time, collections, re

RUNS = sys.argv[1]
OUT = sys.argv[2]

def fmt(ts):
    return time.strftime('%Y-%m-%d %H:%M', time.localtime(ts)) if ts else None

def dstats(path):
    size = n = 0; lo = hi = None; ext = collections.Counter()
    for dp, dns, fns in os.walk(path):
        for f in fns:
            p = os.path.join(dp, f)
            try: st = os.lstat(p)
            except OSError: continue
            size += st.st_size; n += 1
            ext[os.path.splitext(f)[1] or '(none)'] += 1
            lo = st.st_mtime if lo is None or st.st_mtime < lo else lo
            hi = st.st_mtime if hi is None or st.st_mtime > hi else hi
    return {'bytes': size, 'files': n, 'mtime_min': fmt(lo), 'mtime_max': fmt(hi), 'ext': dict(ext.most_common(10))}

def summarize_state(path):
    d = json.load(open(path))
    out = {'path': path, 'bytes': os.path.getsize(path), 'mtime': fmt(os.path.getmtime(path))}
    for k in ('schema', 'study_id', 'a40_uuid', 'n_star', 'schedule_sha256'):
        out[k] = d.get(k)
    out['manifest_sha256'] = (d.get('manifest_sha256') or '')[:16]
    out['top_keys'] = list(d.keys())
    sc = d.get('scientist_config') or {}
    out['scientist'] = {k: sc.get(k) for k in ('model', 'reasoning', 'codex_version', 'timeout_seconds', 'container_image', 'execution_backend')}
    oma = d.get('operational_manifest_amendments')
    if oma is not None:
        out['amendments'] = len(oma)
        out['amendment_ids'] = [ (a.get('amendment_id') or a.get('id') or a.get('name') or a.get('manifest_path','')[-80:]) if isinstance(a, dict) else str(a)[:80] for a in oma]
    pos = d.get('positions', {})
    sched = {str(s['global_position']): s for s in d.get('schedule', [])}
    out['n_positions'] = len(pos)
    out['status_counts'] = dict(collections.Counter(p.get('status') for p in pos.values()))
    arms = collections.defaultdict(lambda: collections.Counter())
    detail = []
    models = collections.Counter(); reason = collections.Counter()
    for k in sorted(pos, key=int):
        p = pos[k]; s = sched.get(k, {})
        arm = p.get('campaign_id') or s.get('campaign_id')
        arms[arm][p.get('status')] += 1
        ex = p.get('execution') or {}
        res = ex.get('reservation') or {}
        if res.get('agent_model_identifier'): models[res['agent_model_identifier']] += 1
        if res.get('reasoning_setting'): reason[res['reasoning_setting']] += 1
        if p.get('status') != 'pending':
            item = {'pos': int(k), 'arm': arm, 'attempt': p.get('campaign_attempt') or s.get('campaign_attempt'), 'status': p.get('status')}
            if p.get('outcome'): item['outcome'] = p['outcome']
            if p.get('result_file'):
                rf = p['result_file']; item['result_file'] = rf
                item['result_exists'] = os.path.exists(rf)
                if os.path.exists(rf): item['result_mtime'] = fmt(os.path.getmtime(rf))
            for extra in ('reconciliation_failure', 'last_transport_error', 'non_consuming_infrastructure_events', 'scientific_failure', 'failure'):
                if extra in p:
                    v = p[extra]
                    item[extra] = json.dumps(v)[:400]
            if ex.get('dispatch_status'): item['dispatch_status'] = ex['dispatch_status']
            detail.append(item)
    out['arms'] = {a: dict(c) for a, c in arms.items()}
    out['positions_detail'] = detail
    out['exec_models'] = dict(models); out['exec_reasoning'] = dict(reason)
    return out

def summarize_history(path):
    lines = open(path, 'rb').read().splitlines()
    out = {'path': path, 'bytes': os.path.getsize(path), 'lines': len(lines), 'mtime': fmt(os.path.getmtime(path))}
    hashes = [hashlib.sha256(l).hexdigest() for l in lines]
    out['line_sha256'] = hashes
    recs = []
    for l in lines:
        try:
            r = json.loads(l)
        except Exception as e:
            recs.append({'parse_error': repr(e)[:80]}); continue
        res = r.get('result') or {}
        pd = r.get('pareto_decision') or res.get('pareto_decision') or {}
        recs.append({
            'pos': r.get('global_position'), 'attempt': r.get('campaign_attempt'), 'status': r.get('status'),
            'model': r.get('agent_model_identifier'), 'reasoning': r.get('reasoning_setting'),
            'decision': pd.get('decision'), 'quality_status': pd.get('quality_status'),
            'bpb': res.get('protected_validation_bpb'), 'tok_s': (res.get('inference_result') or {}).get('cached_decode_tokens_per_second_median'),
            'mem': (res.get('inference_result') or {}).get('peak_allocated_inference_memory_bytes'),
            'total_tokens': res.get('total_tokens'), 'total_batch_size': res.get('total_batch_size'),
            'change_category': r.get('change_category'), 'parent': r.get('parent_experiment_id'),
            'exp_id': res.get('experiment_id'), 'res_status': res.get('status'),
            'after_sha': (r.get('candidate_source_after_sha256') or '')[:8],
            'manifest_sha256': (r.get('manifest_sha256') or '')[:12],
            'n_star': (r.get('contract') or {}).get('n_star'),
            'checkpoint': (res.get('checkpoint') or '')[:120],
            'failure': (res.get('failure_reason') or res.get('error') or r.get('failure_reason') or '')[:160] if isinstance(res.get('failure_reason') or res.get('error') or r.get('failure_reason') or '', str) else json.dumps(res.get('failure_reason') or res.get('error'))[:160],
        })
    out['records'] = recs
    return out

summary = {}
for ns in sorted(os.listdir(RUNS)):
    p = os.path.join(RUNS, ns)
    if not os.path.isdir(p):
        continue
    s = {'stats': dstats(p), 'states': [], 'histories': [], 'position_files': {}, 'subdirs': sorted(os.listdir(p))}
    for dp, dns, fns in os.walk(p):
        for f in fns:
            fp = os.path.join(dp, f)
            if f == 'controller_state.json' or (f.startswith('controller_state.json.') and f.endswith('.json')):
                try:
                    s['states'].append(summarize_state(fp))
                except Exception as e:
                    s['states'].append({'path': fp, 'error': repr(e)})
            elif f == 'history.jsonl':
                s['histories'].append(summarize_history(fp))
            elif re.match(r'position-\d{3}-.*\.json$', f) and os.path.basename(dp).startswith('pair-a-fixed'):
                key = os.path.relpath(dp, RUNS)
                s['position_files'].setdefault(key, []).append((f, fmt(os.path.getmtime(fp))))
    summary[ns] = s
    print('done', ns, file=sys.stderr)

json.dump(summary, open(OUT, 'w'), indent=1, default=str)
