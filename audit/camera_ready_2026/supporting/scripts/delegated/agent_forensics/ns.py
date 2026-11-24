import json, os, glob, sys, hashlib
sys.path.insert(0, os.path.dirname(__file__))
from hist import summarize
R = "/Users/Rami/Documents/dev/autopareto/runs"
nss = sorted(d for d in os.listdir(R) if d.startswith("pair-a") and os.path.isdir(os.path.join(R, d)))
for ns in nss:
    for arm in ("pair-a-fixed-time-seed3", "pair-a-fixed-tokens-seed3"):
        p = os.path.join(R, ns, arm, "history.jsonl")
        if not os.path.exists(p):
            continue
        rows = summarize(p)
        sig = [(r['att'], r['src'][:8], r['status'][:9], (round(r['bpb'],4) if r['bpb'] else None)) for r, d in rows]
        r0, d0 = rows[0]
        n_star = (d0.get('contract') or {}).get('n_star')
        print(f"{ns[len('pair-a-official-'):] if ns.startswith('pair-a-official-') else ns:70s} {arm[7:18]:12s} n={len(rows):2d} parent0={r0['parent']} prog={r0['prog']} nstar={n_star}")
        print("     ", sig)
