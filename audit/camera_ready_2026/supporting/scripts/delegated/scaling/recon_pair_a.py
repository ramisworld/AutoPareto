#!/usr/bin/env python3
"""Reconstruct Pair-A candidate sources by replaying recorded unified diffs on the registered baseline.
Writes only into the scratch folder given as argv[3]. Verifies candidate_source_after_sha256."""
import hashlib, json, subprocess, sys, shutil
from pathlib import Path
hist = Path(sys.argv[1]); base = Path(sys.argv[2]); out = Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
sources = {"baseline-calibration-new-a40-20260914-run1": base}
ok = bad = 0
for line in hist.read_text().splitlines():
    r = json.loads(line)
    res = r.get("result") or {}
    eid = res.get("experiment_id") or f"{r['campaign_id']}-attempt-{r['campaign_attempt']:02d}"
    parent = sources.get(r.get("parent_experiment_id"))
    if parent is None or not r.get("candidate_patch"):
        print(r["campaign_attempt"], "skip (no parent source or patch)"); continue
    if sha(parent) != r["candidate_source_before_sha256"]:
        print(r["campaign_attempt"], "parent hash mismatch"); bad += 1; continue
    dst = out / f"{r['campaign_id']}-{r['campaign_attempt']:02d}.train.py"
    shutil.copy(parent, dst)
    patch = out / "tmp.patch"; patch.write_text(r["candidate_patch"])
    p = subprocess.run(["patch", "-s", "-p0", str(dst), str(patch)], capture_output=True, text=True)
    got = sha(dst)
    good = got == r["candidate_source_after_sha256"]
    ok += good; bad += (not good)
    print(r["campaign_attempt"], "OK" if good else f"MISMATCH rc={p.returncode} {p.stdout[:80]} {p.stderr[:80]}")
    sources[eid] = dst
print("verified", ok, "failed", bad)
