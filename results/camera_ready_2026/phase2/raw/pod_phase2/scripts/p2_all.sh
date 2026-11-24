#!/usr/bin/env bash
# Phase 2 master: A (protected BPB + bracketed timing, CPU-sensitive, runs alone first), B (frozen downstream suite), C (context-position test + BLiMP map).
set -uo pipefail
P=/workspace/autopareto-camera-ready/results/camera_ready_2026/phase2
echo "[$(date -u +%FT%TZ)] PHASE2 START"
bash $P/scripts/p2a_winners.sh
bash $P/scripts/p2b_downstream.sh
cd /workspace/autopareto-camera-ready
for m in opus55_t_attempt23 opus55_t_attempt24 opus55_t_attempt27 pair_a_baseline_t autoresearch_compatible_time300s_seed0 autopareto_292_tokens201M_seed3 autopareto_292_tokens201M_seed4 autopareto_292_tokens201M_seed5 autoresearch_compatible_tokens201M_seed3 blimp_map; do
  echo "[$(date -u +%FT%TZ)] context-position $m"
  /opt/uv-tool/bin/uv run --locked python $P/scripts/p2c_context_position.py $P/context_position $m < /dev/null 2>&1 | grep -v -i 'fetching\|warn\|it/s\]' | tail -3
done
echo "[$(date -u +%FT%TZ)] PHASE2 ALL DONE"
