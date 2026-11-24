import sqlite3, json, collections
DB='/Users/Rami/Documents/dev/autopareto/results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db'
c=sqlite3.connect(f'file:{DB}?mode=ro&immutable=1',uri=True)
print('== downstream_evaluation_runs environment_json')
cnt=collections.Counter()
for rid, eid, env, ev, evsha, fin in c.execute('select id, experiment_id, environment_json, evaluator_git_commit, evaluator_sha256, finished_at from downstream_evaluation_runs'):
    e=json.loads(env) if env else {}
    key=(e.get('python') or e.get('python_version'), e.get('torch') or e.get('torch_version'), e.get('cuda') or e.get('torch_cuda') or e.get('cuda_version'), e.get('driver_version') or e.get('nvidia_driver'), (ev or '')[:8], (evsha or '')[:8], (fin or '')[:10])
    cnt[key]+=1
for k,v in sorted(cnt.items(), key=lambda x: x[0][-1]): print(v, k)
print('sample env keys:', sorted(json.loads(c.execute('select environment_json from downstream_evaluation_runs order by id desc limit 1').fetchone()[0]).keys()))
print(json.dumps(json.loads(c.execute('select environment_json from downstream_evaluation_runs order by id desc limit 1').fetchone()[0]))[:1500])
print('== deployment_sessions')
for row in c.execute('select id, session_name, protocol_version, environment_json, hardware_json, started_at from deployment_sessions'):
    env=json.loads(row[3]) if row[3] else {}
    hw=json.loads(row[4]) if row[4] else {}
    print(row[0], row[1], row[2], row[5], {k: env.get(k) for k in list(env)[:12]}, '| hw:', json.dumps(hw)[:300])
print('== experiments evaluator commits by method/phase')
for r in c.execute("select method, phase, count(*), group_concat(distinct substr(evaluator_git_commit,1,8)), group_concat(distinct substr(git_commit,1,8)), min(started_at), max(finished_at) from experiments group by method, phase order by min(started_at)"):
    print(r)
