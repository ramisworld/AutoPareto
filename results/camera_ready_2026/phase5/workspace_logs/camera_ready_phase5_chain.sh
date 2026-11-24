#!/usr/bin/env bash
# Phase 5 chain (pre-registered order, evaluation/PHASE5_PREREG.md @ acc86d7): 5A training -> 5A downstream -> 5B.
# Each stage starts only if the previous one exited 0 (stop rule: report, do not improvise).
set -uo pipefail
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
cd /workspace/autopareto-camera-ready
L=/workspace/camera_ready_phase5_console.log
P5A=a47f81b752d7953d05227bae77451f398ab0fca672f711f21a2fb3523c72993d
stamp() { echo "[$(date -u +%FT%TZ)] $*" >> "$L"; }
date -u +%FT%TZ > /workspace/camera_ready_phase5_launch_time.txt
stamp "5A train start (plan $P5A)"
uv run --locked python scripts/run_phase5a_opus_fixed_tokens.py --stage train --execute --expect-plan-sha256 $P5A >> "$L" 2>&1
rc=$?; stamp "5A train exit $rc"; [ $rc -eq 0 ] || { stamp "STOP after 5A train"; exit $rc; }
uv run --locked python scripts/run_phase5a_opus_fixed_tokens.py --stage downstream --execute --expect-plan-sha256 $P5A >> "$L" 2>&1
rc=$?; stamp "5A downstream exit $rc"; [ $rc -eq 0 ] || { stamp "STOP after 5A downstream"; exit $rc; }
bash scripts/pod/phase5b_run_ablations.sh >> "$L" 2>&1
rc=$?; stamp "5B exit $rc"
stamp "PHASE5_CHAIN_DONE"
