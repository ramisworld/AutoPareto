import json, os, glob, hashlib, collections
R = '/Users/Rami/Documents/dev/autopareto/'
# index all json files under runs/ and study/ by sha256
idx = {}
for p in glob.glob(R + 'runs/**/*.json', recursive=True) + glob.glob(R + 'study/**/*.json', recursive=True):
    if os.path.getsize(p) > 5_000_000: continue
    idx.setdefault(hashlib.sha256(open(p, 'rb').read()).hexdigest(), []).append(os.path.relpath(p, R))
out = collections.defaultdict(lambda: collections.Counter())
for h in glob.glob(R + 'runs/**/history.jsonl', recursive=True):
    ns = os.path.relpath(h, R + 'runs').split('/pair-a-fixed')[0]
    for l in open(h):
        r = json.loads(l)
        ms, mp = r.get('manifest_sha256'), r.get('manifest_path')
        at_path = os.path.exists(mp) and hashlib.sha256(open(mp, 'rb').read()).hexdigest() == ms if mp else False
        anywhere = ms in idx
        out[ns][('path_ok' if at_path else 'path_bad') + '/' + ('found_by_hash' if anywhere else 'NOT_FOUND')] += 1
for ns, c in sorted(out.items()):
    print(ns, dict(c))
