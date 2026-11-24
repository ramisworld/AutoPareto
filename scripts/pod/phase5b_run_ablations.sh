#!/usr/bin/env bash
# Phase 5B (pre-registered in evaluation/PHASE5_PREREG.md and evaluation/phase5b_inference_ablation_plan.json):
# inference-only ablations of the Opus 5.5 frontier adapters on their existing Pair-A checkpoints, measured by the
# frozen protected benchmark (candidate-only mode) on this A40. No training; no protected file is edited; the
# historical checkpoints are read only. Writes only into a new results directory.
set -uo pipefail
REPO=/workspace/autopareto-camera-ready
PLAN=$REPO/evaluation/phase5b_inference_ablation_plan.json
OUT=$REPO/results/camera_ready_2026/phase5/B
SRC=$REPO/results/camera_ready_2026/phase2/raw/pod_phase2/inputs/sources
C=/var/lib/autopareto/pair_a_claude_opus55_t_seed3_continuation_20261105c/attempts
BASE=/var/lib/autopareto/pair_a_claude_opus55_baseline_attempt2/nonstudy_tests/t_smoke_artifacts/result.checkpoint.pt
BCTX=/opt/autopareto_trusted/study/contracts_2026/historical_reference/baseline_source
P2REC=$(ls /var/lib/autopareto/phase2_winner_verification_20261107T032813Z/attempt-24/repeat_1.json)
RC=/var/cache/autopareto/pair_a_registered_cache
export HOME=/root AUTOPARETO_REGISTERED_CACHE=$RC HF_HOME=$RC/huggingface HF_HUB_CACHE=$RC/huggingface/hub HF_HUB_OFFLINE=1
export AUTOPARETO_TRUSTED_ROOT=/opt/autopareto_trusted AUTOPARETO_STUDY_VENV=/opt/autopareto_runtime
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
PY=/opt/autopareto_runtime/bin/python
log() { echo "[$(date -u +%FT%TZ)] $* | load $(cut -d' ' -f1-3 /proc/loadavg)" | tee -a "$OUT/driver.log"; }
[ -e "$OUT/runs" ] && { echo "refusing: $OUT/runs exists"; exit 1; }
mkdir -p "$OUT/runs" "$OUT/kernel_census"
log "phase5b start on $(nvidia-smi --query-gpu=uuid,driver_version --format=csv,noheader); plan sha256 $(sha256sum "$PLAN" | cut -c1-64)"
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || { log "FAIL GPU not idle"; exit 1; }

# Identity: checkpoints, sources, registered prepare, and the protected benchmark Phase 2A used.
declare -A POS=([23]=046 [24]=047 [27]=052)
declare -A CK
read_plan() { python3 -c "import json,sys; d=json.load(open('$PLAN')); print(eval(sys.argv[1]))" "$1"; }
for a in 23 24 27; do
  ck=$(ls $C/claude-opus55-position-${POS[$a]}-pair-a-fixed-time-seed3-*/position-${POS[$a]}-pair-a-fixed-time-seed3.checkpoint.pt)
  CK[$a]=$ck
  [ "$(sha256sum "$ck" | cut -c1-64)" = "$(read_plan "d['checkpoints']['opus_attempt$a']['sha256']")" ] || { log "FAIL checkpoint $a"; exit 1; }
  [ "$(sha256sum "$SRC/opus_attempt$a.train.py" | cut -c1-64)" = "$(read_plan "d['checkpoints']['opus_attempt$a']['source_sha256']")" ] || { log "FAIL source $a"; exit 1; }
done
[ "$(sha256sum "$BASE" | cut -c1-64)" = "$(read_plan "d['checkpoints']['pair_a_baseline']['sha256']")" ] || { log "FAIL baseline checkpoint"; exit 1; }
BENCH=/opt/autopareto_trusted/study/contracts_2026/inference_benchmark.py
P2SHA=$(python3 -c "import json; print(json.load(open('$P2REC'))['evaluator_sha256'])" 2>/dev/null || echo none)
[ "$(sha256sum "$BENCH" | cut -c1-64)" = "$P2SHA" ] || { log "FAIL benchmark sha differs from Phase 2A ($P2SHA)"; exit 1; }
[ "$(sha256sum $REPO/scripts/phase5b_inference_ablation.py | cut -c1-64)" = "$(read_plan "d['variant_generator']['sha256']")" ] || { log "FAIL generator hash"; exit 1; }
[ "$(sha256sum $REPO/scripts/phase5b_kernel_census.py | cut -c1-64)" = "$(read_plan "d['kernel_census']['sha256']")" ] || { log "FAIL census hash"; exit 1; }
log "identity ok: 3 Opus checkpoints + sources, baseline T, benchmark $P2SHA (= Phase 2A), generator, census; driver $(sha256sum "$0" | cut -c1-64)"

python3 $REPO/scripts/phase5b_inference_ablation.py make --sources "$SRC" --prepare "$BCTX/prepare.py" --out "$OUT/contexts" > "$OUT/contexts.log" 2>&1 || { log "FAIL context generation"; exit 1; }
python3 - "$PLAN" "$OUT/contexts/variant_contexts.json" <<'PYEOF' || { log "FAIL variant hashes"; exit 1; }
import json, sys
plan, made = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))["contexts"]
bad = [k for k, v in plan["variant_train_sha256"].items() if made.get(k, {}).get("train_sha256") != v]
print("variant hashes:", "ok" if not bad else bad); sys.exit(1 if bad else 0)
PYEOF
log "18 variant contexts generated and hash-verified"

cd /opt/autopareto_trusted
run_one() {  # run_one <id> <context> <checkpoint>
  local id=$1 ctx=$2 ck=$3
  log "run $id start"
  "$PY" -m study.contracts_2026.inference_benchmark --context "$ctx" --candidate train.py --checkpoint "$ck" \
     --output "$OUT/runs/$id.json" --sample "$OUT/runs/$id.sample.txt" > "$OUT/runs/$id.log" 2>&1
  local rc=$?
  "$PY" - "$OUT/runs/$id.json" "$rc" "$id" <<'PYEOF' | tee -a "$OUT/driver.log"
import json, statistics, sys
path, rc, rid = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    d = json.load(open(path))
except Exception as exc:
    print(f"  {rid}: exit {rc}, no result ({exc})"); raise SystemExit
e2e = statistics.median(r["end_to_end_tokens_per_second"] for r in d["inference_repetitions"])
c = d.get("correctness") or {}
print(f"  {rid}: exit {rc} status={d.get('inference_status')} decode={d['cached_decode_tokens_per_second_median']:.1f} "
      f"prefill_ms={d['prefill_milliseconds']:.2f} e2e={e2e:.1f} peak_mib={d['peak_allocated_inference_memory_bytes']/2**20:.1f} "
      f"kv_mib={d['kv_cache_bytes']/2**20:.2f} correct={c.get('passed')} mode={c.get('mode') or c.get('correctness_mode')}")
PYEOF
}
python3 -c "import json; [print(r['id'], r['model'], r['variant']) for r in json.load(open('$PLAN'))['order']]" > "$OUT/order.txt"
while read -r id model variant; do
  if [ "$model" = "pair_a_baseline" ]; then
    run_one "$(printf '%02d' $(grep -n "^$id " "$OUT/order.txt" | cut -d: -f1))_$id" "$BCTX" "$BASE"
  else
    a=${model#opus_attempt}
    run_one "$(printf '%02d' $(grep -n "^$id " "$OUT/order.txt" | cut -d: -f1))_$id" "$OUT/contexts/attempt-$a/$variant" "${CK[$a]}"
  fi
done < "$OUT/order.txt"
log "timing runs done"

# Diagnostic kernel census (not timing): attempt 23, every variant, and the baseline.
for v in V0_as_submitted V1_no_cuda_graph V2_no_fusion V3_sdpa_attention V4_eager_unfused V5_linear_kv_cache; do
  "$PY" $REPO/scripts/phase5b_kernel_census.py --context "$OUT/contexts/attempt-23/$v" --checkpoint "${CK[23]}" \
     --output "$OUT/kernel_census/a23-$v.json" > "$OUT/kernel_census/a23-$v.log" 2>&1
  log "census a23 $v exit $? $(tail -c 300 "$OUT/kernel_census/a23-$v.log" | tr '\n' ' ' | cut -c1-240)"
done
"$PY" $REPO/scripts/phase5b_kernel_census.py --context "$BCTX" --checkpoint "$BASE" --output "$OUT/kernel_census/baseline.json" \
   > "$OUT/kernel_census/baseline.log" 2>&1
log "census baseline exit $? $(tail -c 300 "$OUT/kernel_census/baseline.log" | tr '\n' ' ' | cut -c1-240)"
log "PHASE5B DONE"
