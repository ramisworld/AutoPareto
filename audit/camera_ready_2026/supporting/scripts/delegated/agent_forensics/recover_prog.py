import json, sys, hashlib, glob, os
# Recover program.md text printed by the agent via `sed -n '1,240p' program.md && printf '\n__CAMPAIGN_VIEW__\n' ...`
want = sys.argv[1]  # hash prefix
outdir = sys.argv[2]
for p in sorted(glob.glob(sys.argv[3] + '/**/stdout.jsonl', recursive=True)):
    for line in open(p):
        try: r = json.loads(line)
        except Exception: continue
        it = r.get('item') or {}
        if it.get('type') != 'command_execution' or it.get('exit_code') != 0: continue
        cmd = it.get('command') or ''
        out = it.get('aggregated_output') or ''
        cands = []
        if "program.md" in cmd and "__CAMPAIGN_VIEW__" in out:
            cands.append(out.split('\n__CAMPAIGN_VIEW__\n')[0])
        elif cmd.strip().endswith("program.md\"") or "cat program.md" in cmd or ("sed -n" in cmd and "program.md" in cmd and "&&" not in cmd and ";" not in cmd):
            cands.append(out)
        for c in cands:
            for variant in (c, c + '\n', c.rstrip('\n') + '\n'):
                h = hashlib.sha256(variant.encode()).hexdigest()
                if h.startswith(want):
                    open(os.path.join(outdir, want + '.md'), 'w').write(variant)
                    print('RECOVERED', want, 'from', p)
                    sys.exit(0)
print('not recovered', want)
