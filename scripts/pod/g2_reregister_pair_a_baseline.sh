#!/usr/bin/env bash
# G2: re-register the Pair-A paired baseline checkpoints on a fresh A40.
#
# The checkpoints the Codex campaign bracketed against (T 0ccea0c4..., N 23be533b...)
# were lost with the old pod. This runs the registered baseline (b2fb8788..., seed 3)
# through the unchanged trusted smoke tools, under the production launcher
# environment, into a new root-only state directory. Halt rule (approved): if the
# T run's protected validation BPB differs from the registered 1.19989 by more
# than 0.005, stop before the N run.
#
# Run as root on the pod after scripts/pod/fresh_a40_bootstrap.sh all.
set -euo pipefail

# A failed attempt keeps its state directory as evidence; retries pass a new one.
STATE=${G2_STATE:-/var/lib/autopareto/pair_a_claude_opus55_baseline}
REGISTERED_T_BPB=1.1998857791422985
TOLERANCE=0.005
N_STAR=85458944
PY=/opt/autopareto_runtime/bin/python

[ ! -e "$STATE" ] || { echo "refusing: $STATE exists" >&2; exit 1; }
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { echo "GPU is busy" >&2; exit 1; }
install -d -o root -g root -m 0700 "$STATE"

export AUTOPARETO_TRUSTED_ROOT=/opt/autopareto_trusted
export AUTOPARETO_TRUSTED_STATE_ROOT="$STATE"
export AUTOPARETO_STUDY_VENV=/opt/autopareto_runtime
export AUTOPARETO_REGISTERED_CACHE=/var/cache/autopareto/pair_a_registered_cache
export AUTOPARETO_CANDIDATE_CONTEXT_ROOT=/opt/autopareto_candidate_contexts
export AUTOPARETO_CANDIDATE_ATTEMPT_ROOT=/opt/autopareto-candidate-attempts
export AUTOPARETO_CANDIDATE_ROOT=/var/lib/autopareto/candidate_attempts
export HF_HOME=/opt/autopareto_candidate_inputs/cache/huggingface
export HF_HUB_CACHE=/opt/autopareto_candidate_inputs/cache/huggingface/hub
export HF_HUB_OFFLINE=1
export HOME=/root
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF

summary() {  # summary <evidence json>
  "$PY" - "$1" <<'PY'
import hashlib, json, sys
record = json.load(open(sys.argv[1]))
result = record["result"]
out = {key: result.get(key) for key in ("status", "protected_validation_bpb", "total_tokens", "optimizer_updates", "measured_training_seconds")}
inference = result.get("inference_result") or {}
out["inference"] = {key: inference.get(key) for key in ("inference_status", "normalized_decode_throughput", "peak_allocated_inference_memory_bytes")}
out["persisted"] = {name: item["sha256"] for name, item in record["persisted_artifacts"].items()}
out["checkpoint"] = next((item for name, item in record["persisted_artifacts"].items() if name.endswith(".pt")), None)
print(json.dumps(out, indent=2, default=str))
PY
}

cd /opt/autopareto_trusted
echo "[$(date -u +%FT%TZ)] T smoke (300 s, seed 3) start"
"$PY" -m study.contracts_2026.tools.run_t_smoke > "$STATE/t_smoke.stdout" 2>&1
echo "[$(date -u +%FT%TZ)] T smoke done"
summary "$STATE/nonstudy_tests/t_smoke.json"
T_BPB=$("$PY" -c "import json; print(json.load(open('$STATE/nonstudy_tests/t_smoke.json'))['result']['protected_validation_bpb'])")
if ! "$PY" -c "import sys; sys.exit(0 if abs($T_BPB - $REGISTERED_T_BPB) <= $TOLERANCE else 1)"; then
  echo "G2 HALT: T baseline BPB $T_BPB differs from registered $REGISTERED_T_BPB by more than $TOLERANCE"
  exit 3
fi
echo "G2 T PASS: |$T_BPB - $REGISTERED_T_BPB| <= $TOLERANCE"

echo "[$(date -u +%FT%TZ)] N smoke (N* = $N_STAR, seed 3) start"
"$PY" -m study.contracts_2026.tools.run_n_smoke --n-star "$N_STAR" > "$STATE/n_smoke.stdout" 2>&1
echo "[$(date -u +%FT%TZ)] N smoke done"
summary "$STATE/nonstudy_tests/n_smoke.json"
echo "G2 DONE"
