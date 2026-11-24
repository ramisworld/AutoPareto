#!/usr/bin/env bash
# Replacement-A40 validation with the acceptance rule recorded in doc 17 §8 before any result:
# both smokes pass the BPB halt rule (G2 script, unchanged) AND all three bracketed inference
# repeats are stable (normalized drift <= 0.02). Prints ACCEPT or REJECT. No campaign state is touched.
set -uo pipefail
TAG=${1:?tag, e.g. 20261105_pod3}
V=/var/lib/autopareto/pair_a_claude_opus55_a40_validation_$TAG
OUT=/workspace/g2_logs/a40_validation_$TAG
[ ! -e "$V" ] || { echo "refusing: $V exists"; exit 1; }
echo "[$(date -u +%FT%TZ)] host: $(nvidia-smi --query-gpu=uuid --format=csv,noheader) | load $(cut -d' ' -f1-3 /proc/loadavg) | cpu pressure $(head -1 /proc/pressure/cpu)"
cd /workspace/autopareto-camera-ready
G2_STATE="$V" bash scripts/pod/g2_reregister_pair_a_baseline.sh > "$OUT.out" 2>&1
g2=$?
echo "[$(date -u +%FT%TZ)] G2 exit $g2: $(grep -E '^G2 (T PASS|HALT|DONE)' "$OUT.out" | tr '\n' ' ')"
[ "$g2" = 0 ] && grep -q '^G2 DONE' "$OUT.out" || { echo "REJECT: baseline smoke failed or halted"; bash scripts/pod/fresh_a40_bootstrap.sh mirror >/dev/null; exit 3; }
bash /workspace/autopareto-camera-ready/scripts/pod/bracketed_inference_repeats.sh "$V" > "${OUT}_inference_repeats.out" 2>&1
/opt/autopareto_runtime/bin/python - "$V" <<'PY'
import json, pathlib, sys
V = pathlib.Path(sys.argv[1]); ok = True
for name in ("t_smoke", "n_smoke"):
    r = json.load(open(V / "nonstudy_tests" / f"{name}.json"))["result"]
    print(f"  {name}: bpb {r['protected_validation_bpb']:.5f} tokens {r['total_tokens']} updates {r['optimizer_updates']} inference {(r.get('inference_result') or {}).get('inference_status')}")
for i in (1, 2, 3):
    p = V / "inference_repeats" / f"repeat_{i}.json"
    try:
        d = json.load(open(p))
    except Exception as exc:
        print(f"  repeat {i}: no result ({exc})"); ok = False; continue
    st, dr = d.get("normalized_inference_status"), d.get("normalized_relative_drift")
    print(f"  repeat {i}: normalized drift {dr:.4f} -> {st}")
    ok = ok and st == "stable" and dr is not None and dr <= 0.02
print("ACCEPT: all three bracketed repeats stable" if ok else "REJECT: at least one bracketed repeat unstable")
sys.exit(0 if ok else 4)
PY
rc=$?
bash scripts/pod/fresh_a40_bootstrap.sh mirror >/dev/null && echo "[$(date -u +%FT%TZ)] mirrored to /workspace"
echo "VALIDATION FINISHED rc=$rc"
exit $rc
