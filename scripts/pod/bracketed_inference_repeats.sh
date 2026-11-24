#!/usr/bin/env bash
# Inference-only diagnostic: repeat the campaign's bracketed decode measurement on this A40.
# Candidate = the new-A40 T validation checkpoint (baseline source); bracket = the campaign's paired T checkpoint.
set -uo pipefail
V=${1:?validation state directory}
OUT=$V/inference_repeats
CAND=$V/nonstudy_tests/t_smoke_artifacts/result.checkpoint.pt
BASE=/var/lib/autopareto/pair_a_claude_opus55_baseline_attempt2/nonstudy_tests/t_smoke_artifacts/result.checkpoint.pt
CTX=/opt/autopareto_trusted/study/contracts_2026/historical_reference/baseline_source
RC=/var/cache/autopareto/pair_a_registered_cache
install -d -o root -g root -m 0700 "$OUT"
export HOME=/root AUTOPARETO_REGISTERED_CACHE=$RC HF_HOME=$RC/huggingface HF_HUB_CACHE=$RC/huggingface/hub HF_HUB_OFFLINE=1
export AUTOPARETO_TRUSTED_ROOT=/opt/autopareto_trusted AUTOPARETO_STUDY_VENV=/opt/autopareto_runtime
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
cd /opt/autopareto_trusted
for i in 1 2 3; do
  echo "[$(date -u +%FT%TZ)] repeat $i start"
  /opt/autopareto_runtime/bin/python -m study.contracts_2026.inference_benchmark --protocol autopareto_search_inference_bracketed_v2 \
    --context "$CTX" --candidate train.py --checkpoint "$CAND" --output "$OUT/repeat_$i.json" \
    --bracketed-baseline-context "$CTX" --bracketed-baseline-checkpoint "$BASE" > "$OUT/repeat_$i.log" 2>&1
  echo "[$(date -u +%FT%TZ)] repeat $i exit $?"
  /opt/autopareto_runtime/bin/python - "$OUT/repeat_$i.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as exc:
    print("  no result:", exc); raise SystemExit
keys = ("inference_status", "stability_failure_reason", "decode_block_relative_drift", "cached_decode_coefficient_of_variation",
        "cached_decode_tokens_per_second_median", "normalized_decode_throughput")
print("  ", {k: d.get(k) for k in keys})
PY
done
echo "REPEATS DONE" > "$OUT/DONE"
