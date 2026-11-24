#!/usr/bin/env bash
# Phase 2B (2026-11-07): the frozen downstream suite (downstream_evaluator.py 696f9c46, suite 3dce2c73: LAMBADA,
# BLiMP, PIQA, ARC-Easy, held-out ClimbMix BPB, qualitative generations) on six checkpoints, in Env A exactly as the
# August 201M panel ran it. Results go to a NEW database and output directory; no historical database is touched.
set -uo pipefail
REPO=/workspace/autopareto-camera-ready
P2=$REPO/results/camera_ready_2026/phase2/downstream
DB=$P2/phase2_downstream.db
C=/var/lib/autopareto/pair_a_claude_opus55_t_seed3_continuation_20261105c/attempts
IN=$REPO/results/camera_ready_2026/phase2/inputs
log() { echo "[$(date -u +%FT%TZ)] $*"; }
[ -e "$DB" ] && { log "refusing: $DB exists"; exit 1; }
mkdir -p "$P2/runs"
cd "$REPO"
[ "$(sha256sum downstream_evaluator.py | cut -c1-64)" = 696f9c4677d3f6538c8b92cedec08691175aa30e8b8e48e96b080f2d43c107d5 ] || { log "FAIL evaluator hash"; exit 1; }
[ "$(sha256sum evaluation/downstream_suite.json | cut -c1-64)" = 3dce2c736a7108a855d1150e1f39e2d3d13ca50cfadfff4f568dfed80ed72ef3 ] || { log "FAIL suite hash"; exit 1; }
# id|method|seed|candidate|checkpoint|checkpoint sha256|contract note
MODELS=$(cat <<EOF
1|opus55_t_attempt23|3|$IN/sources/opus_attempt23.train.py|$(ls $C/claude-opus55-position-046-*/position-046-*.checkpoint.pt)|f098e0bd5b19fcc7cbfdf8f41d464cdd7297c392224560f4a053ccf516436941|Pair-A fixed time 300 s, 197.1M tokens
2|opus55_t_attempt24|3|$IN/sources/opus_attempt24.train.py|$(ls $C/claude-opus55-position-047-*/position-047-*.checkpoint.pt)|e5687149ac6243e48b6691d2aa595687a0a2c2404288d02e221c258039cf1d6b|Pair-A fixed time 300 s, 198.4M tokens
3|opus55_t_attempt27|3|$IN/sources/opus_attempt27.train.py|$(ls $C/claude-opus55-position-052-*/position-052-*.checkpoint.pt)|520af249fd6d1d157596a039a44b6402d2ebe8ae7ca0fcefe99c4e11dfc84d35|Pair-A fixed time 300 s, 198.0M tokens
4|pair_a_baseline_t|3|$REPO/study/contracts_2026/historical_reference/baseline_source/train.py|/var/lib/autopareto/pair_a_claude_opus55_baseline_attempt2/nonstudy_tests/t_smoke_artifacts/result.checkpoint.pt|f7ca17c624f8847e28a0d6abf9a33491023ea1bee72bf794fe7d8c6b343d43d9|Pair-A fixed time 300 s, 84.9M tokens (paired T checkpoint)
5|autoresearch_compatible_time300s|0|$REPO/results/camera_ready_2026/amendment1/E0/runs/e0-arref-time300s-seed0/isolate/train.py|$REPO/results/camera_ready_2026/amendment1/E0/runs/e0-arref-time300s-seed0/final.pt|c034269894bfc1964bf027b0025e5ee8d573b687928ae279e69618e9ad67b9ad|Phase-1 fixed time 300 s (E0 AR-ref), 87.6M tokens
6|autopareto_292_tokens201M_repro|3|$REPO/results/candidates/autopareto-seed0-20260731/0022-train.py|$IN/phase1_checkpoints/phase1-autopareto_292-seed3/final.pt|9dbea197499a5be43814e1eeb6c649f2cb0591189f8d7de1af6d80cc9e65f8f7|Phase-1 matched tokens 201.3M; re-evaluation of August downstream run 14
EOF
)
PYA="/opt/uv-tool/bin/uv run --locked python"
$PYA - "$DB" "$MODELS" <<'PYEOF' || exit 1
import sqlite3, sys
from autopareto_db import migrate
db = sqlite3.connect(sys.argv[1]); db.row_factory = sqlite3.Row
db.execute("""CREATE TABLE experiments (
    id INTEGER PRIMARY KEY, campaign_id TEXT, experiment_number INTEGER,
    method TEXT, training_seed INTEGER, status TEXT, description TEXT,
    git_commit TEXT, code_diff TEXT, val_bpb REAL,
    inference_tokens_per_second REAL, peak_inference_cuda_memory_mb REAL,
    training_tokens_per_second REAL, total_training_tokens INTEGER,
    peak_training_cuda_memory_mb REAL, parameter_count INTEGER,
    training_seconds REAL, inference_seconds REAL, total_runtime_seconds REAL,
    checkpoint_path TEXT, sample_path TEXT, log_path TEXT,
    crash_or_failure_reason TEXT, protected_manifest_sha256 TEXT,
    started_at TEXT, finished_at TEXT, phase TEXT, valid_result INTEGER,
    exclude_from_analysis INTEGER, UNIQUE(campaign_id, experiment_number))""")
db.execute("""CREATE TABLE campaign_state (campaign_id TEXT PRIMARY KEY, method TEXT, status TEXT,
    attempt_limit INTEGER, next_experiment_number INTEGER)""")
migrate(db)
for line in sys.argv[2].splitlines():
    i, method, seed, cand, ck, sha, note = line.split("|")
    db.execute("""INSERT INTO experiments (id, campaign_id, experiment_number, method, training_seed, status,
                  description, git_commit, code_diff, checkpoint_path, phase, training_status, inference_status)
                  VALUES (?, 'phase2-downstream-20261107', ?, ?, ?, 'completed', ?, '8ce879d', '', ?, 'phase2', 'completed', 'compatible')""",
               (int(i), int(i), method, int(seed), note + "; registered only to drive the frozen evaluator (status taken from the source campaign record)", ck))
db.commit(); print("phase2 downstream db created:", db.execute("select count(*) from experiments").fetchone()[0], "experiments")
PYEOF
while IFS='|' read -r i method seed cand ck sha note; do
  [ "$(sha256sum "$ck" | cut -c1-64)" = "$sha" ] || { log "FAIL checkpoint hash $method"; continue; }
  log "downstream $i $method start ($note)"
  $PYA downstream_evaluator.py --experiment-id "$i" --candidate "$cand" --checkpoint "$ck" --db "$DB" \
     --output-dir "$P2/runs" --display-name "$method" --selection-role "phase2: $note" > "$P2/run_$i.log" 2>&1 < /dev/null
  log "downstream $i $method exit $? ($(tail -c 300 "$P2/run_$i.log" | tr '\n' ' ' | cut -c1-200))"
done <<< "$MODELS"
log "PHASE2B DONE"
