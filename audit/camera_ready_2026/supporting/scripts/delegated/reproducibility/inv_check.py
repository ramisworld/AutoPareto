import json, hashlib, re
from pathlib import Path
ROOT=Path('/Users/Rami/Documents/dev/autopareto')
inv=json.loads((ROOT/'paper/generated/artifact_inventory.json').read_text())
byp={r['path']:r for r in inv}
print('inventory records:', len(inv))
pats=[r'results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m\.db$',
      r'results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/runs/phase1-[a-z_0-9]+-seed[345]/(training_metrics\.json|final_inference\.json|step_timing\.jsonl|checkpoint_evaluations\.json|run_metadata\.json)$',
      r'evaluation/phase1_public_manifest\.json$', r'results/phase1/.*', r'runs/pair-a-official-20260906-fixed-warmup-v2/.*/history\.jsonl$',
      r'runs/pair-a-official-20260906-fixed-warmup-v2/provenance/.*\.json$',
      r'results/candidates/(autopareto-seed0-20260731|baseline-a40-seed0|autoresearch-diagnostic5-seed42-20260724|tpe-seed42-20260805|motpe-seed42-20260805)/00(02|05|21|22|24|25)-train\.py$',
      r'results/results\.db$']
checked=0; changed=[]; missing=[]
for path,r in byp.items():
    if any(re.match(p,path) for p in pats):
        p=ROOT/path
        if not p.exists(): missing.append(path); continue
        h=hashlib.sha256(p.read_bytes()).hexdigest(); checked+=1
        if h!=r['sha256']: changed.append(path)
print('checked', checked, 'changed', len(changed), changed[:10], 'missing', missing[:10])
# new history files not in inventory
hist=[str(p.relative_to(ROOT)) for p in ROOT.glob('runs/**/history.jsonl')]
print('history.jsonl now:', len(hist), 'in inventory:', sum(1 for h in hist if h in byp))
print('runs namespaces now:', len([p for p in (ROOT/'runs').iterdir() if p.is_dir()]))
