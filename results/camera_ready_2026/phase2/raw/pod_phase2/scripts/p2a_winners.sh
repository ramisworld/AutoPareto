#!/usr/bin/env bash
# Phase 2A (2026-11-07): independent re-verification of Opus attempts 23, 24, 27 on this A40 with the frozen
# protected harness (Env B, trusted tree, registered cache), unchanged protocol and thresholds.
#  1. checkpoint + source identity against the campaign records;
#  2. protected validation BPB via train_launcher._evaluate_protected_bpb (the campaign's own in-process evaluator);
#  3. three bracketed inference repeats per winner (protocol autopareto_search_inference_bracketed_v2: correctness,
#     warm-up, 8 baseline/candidate/baseline blocks against the paired T checkpoint f7ca17c6), rotated across winners.
# Also re-evaluates the paired Pair-A baseline T checkpoint's protected BPB. Writes only into a new directory.
set -uo pipefail
TS=$(date -u +%Y%m%dT%H%M%SZ)
R=/var/lib/autopareto/phase2_winner_verification_$TS
IN=/workspace/autopareto-camera-ready/results/camera_ready_2026/phase2/inputs/sources
C=/var/lib/autopareto/pair_a_claude_opus55_t_seed3_continuation_20261105c/attempts
BASE=/var/lib/autopareto/pair_a_claude_opus55_baseline_attempt2/nonstudy_tests/t_smoke_artifacts/result.checkpoint.pt
BCTX=/opt/autopareto_trusted/study/contracts_2026/historical_reference/baseline_source
RC=/var/cache/autopareto/pair_a_registered_cache
install -d -o root -g root -m 0700 "$R" "$R/contexts"
export HOME=/root AUTOPARETO_REGISTERED_CACHE=$RC HF_HOME=$RC/huggingface HF_HUB_CACHE=$RC/huggingface/hub HF_HUB_OFFLINE=1
export AUTOPARETO_TRUSTED_ROOT=/opt/autopareto_trusted AUTOPARETO_STUDY_VENV=/opt/autopareto_runtime
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
PY=/opt/autopareto_runtime/bin/python
cd /opt/autopareto_trusted
declare -A POS=([23]=046 [24]=047 [27]=052)
declare -A CKSHA=([23]=f098e0bd5b19fcc7cbfdf8f41d464cdd7297c392224560f4a053ccf516436941 [24]=e5687149ac6243e48b6691d2aa595687a0a2c2404288d02e221c258039cf1d6b [27]=520af249fd6d1d157596a039a44b6402d2ebe8ae7ca0fcefe99c4e11dfc84d35)
declare -A SRCSHA=([23]=408228f4cd45e13395a43b6de75e4e0578f550480b32e4d8c1a2ae352ed83def [24]=d7c7d51b5117bfeac27bcd969f66210867f67a21cf7b33e54118f880d55be6b9 [27]=b8eaf5e93838b007aaaa66b6bb5a5ed15127f4c1c10c1f35633f2d075f8f8e12)
declare -A CK
log() { echo "[$(date -u +%FT%TZ)] $* | load $(cut -d' ' -f1-3 /proc/loadavg)"; }
log "phase2A start on $(nvidia-smi --query-gpu=uuid --format=csv,noheader) -> $R"
echo "$(sha256sum "$BASE" | cut -c1-64)" | grep -q '^f7ca17c624f8847e28a0d6abf9a33491023ea1bee72bf794fe7d8c6b343d43d9$' || { log "FAIL baseline checkpoint hash"; exit 1; }
for a in 23 24 27; do
  ck=$(ls $C/claude-opus55-position-${POS[$a]}-pair-a-fixed-time-seed3-*/position-${POS[$a]}-pair-a-fixed-time-seed3.checkpoint.pt)
  CK[$a]=$ck
  [ "$(sha256sum "$ck" | cut -c1-64)" = "${CKSHA[$a]}" ] || { log "FAIL checkpoint hash attempt $a"; exit 1; }
  install -d -m 0700 "$R/contexts/attempt-$a" "$R/attempt-$a"
  install -m 0600 "$IN/opus_attempt$a.train.py" "$R/contexts/attempt-$a/train.py"
  [ "$(sha256sum "$R/contexts/attempt-$a/train.py" | cut -c1-64)" = "${SRCSHA[$a]}" ] || { log "FAIL source hash attempt $a"; exit 1; }
  # A candidate context is train.py plus the registered prepare.py (the loader puts the context on sys.path),
  # exactly like the registered baseline context.
  install -m 0600 "$BCTX/prepare.py" "$R/contexts/attempt-$a/prepare.py"
  [ "$(sha256sum "$R/contexts/attempt-$a/prepare.py" | cut -c1-64)" = 4f2ba9cbb8ba8c4a3d35be405a913e2f3be3af9aea103ed52ef7b2a662058150 ] || { log "FAIL prepare hash"; exit 1; }
done
log "identity: 3 checkpoints and 3 sources match the campaign records; paired baseline f7ca17c6 verified"

bpb() {  # bpb <label> <context> <checkpoint> <out>
  log "protected BPB $1 start"
  "$PY" - "$2" "$3" "$4" "$R" <<'PYEOF'
import json, pathlib, sys, time
from study.contracts_2026 import train_launcher as tl
ctx, ck, out, run = (pathlib.Path(x) for x in sys.argv[1:5])
t = time.time()
value, prov = tl._evaluate_protected_bpb(context=ctx, checkpoint=ck, run_directory=run, batch_size=64)
json.dump({"protected_validation_bpb": value, "provenance": prov, "seconds": time.time() - t,
           "context_train_sha256": tl._sha256_file(ctx / "train.py"), "checkpoint_sha256": tl._sha256_file(ck)},
          open(out, "w"), indent=1)
print(f"  protected BPB {value:.10f}")
PYEOF
  log "protected BPB $1 exit $?"
}
for a in 23 24 27; do bpb "attempt-$a" "$R/contexts/attempt-$a" "${CK[$a]}" "$R/attempt-$a/protected_bpb.json"; done
install -d -m 0700 "$R/pair_a_baseline_t"
bpb "pair-a baseline T" "$BCTX" "$BASE" "$R/pair_a_baseline_t/protected_bpb.json"

bracket() {  # bracket <attempt> <repeat>
  local a=$1 i=$2
  log "bracketed attempt-$a repeat $i start"
  "$PY" -m study.contracts_2026.inference_benchmark --protocol autopareto_search_inference_bracketed_v2 \
    --context "$R/contexts/attempt-$a" --candidate train.py --checkpoint "${CK[$a]}" --output "$R/attempt-$a/repeat_$i.json" \
    --bracketed-baseline-context "$BCTX" --bracketed-baseline-checkpoint "$BASE" > "$R/attempt-$a/repeat_$i.log" 2>&1
  log "bracketed attempt-$a repeat $i exit $?"
  "$PY" - "$R/attempt-$a/repeat_$i.json" <<'PYEOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as exc:
    print("  no result:", exc); raise SystemExit
c = d.get("candidate_only_memory_measurement") or {}
print("  ", {k: d.get(k) for k in ("status", "normalized_inference_status", "normalized_relative_drift",
      "cached_decode_tokens_per_second_median", "normalized_decode_throughput", "prefill_milliseconds")},
      {"correctness": (d.get("correctness") or {}).get("passed"), "e2e_tok_s": c.get("end_to_end_tokens_per_second"),
       "peak_mb": c.get("peak_allocated_cuda_memory_mb")})
PYEOF
}
for order in "23 24 27" "24 27 23" "27 23 24"; do
  i=$(( ${i:-0} + 1 ))
  for a in $order; do bracket "$a" "$i"; done
done
log "PHASE2A DONE $R"
echo "$R" > /workspace/autopareto-camera-ready/results/camera_ready_2026/phase2/winner_verification_dir.txt
