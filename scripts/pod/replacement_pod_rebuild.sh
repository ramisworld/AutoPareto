#!/usr/bin/env bash
# Rebuild the replacement A40's container-disk state from the /workspace volume (no training, no GPU use).
# Follows scripts/pod/fresh_a40_bootstrap.sh; deviations are only where /workspace already holds
# state from the first pod (Env A venv and persistent data cache), which is verified instead of rebuilt.
set -euo pipefail
REPO=/workspace/autopareto-camera-ready
B="$REPO/scripts/pod/fresh_a40_bootstrap.sh"
LOG_DIR=/workspace/pod_setup_logs
ASSETS=/workspace/assets
UV_LOCK_SHA256=262a4d55eaa24f47c98ca44cc82ff1660d5daba73df0b1e78246cceac2c429b4
T_SHA=f7ca17c624f8847e28a0d6abf9a33491023ea1bee72bf794fe7d8c6b343d43d9
N_SHA=72d05e979d5dcf32a795931dcb579e27549fe57de2d72a7d8ab140e127d51667
log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG_DIR/bootstrap.log"; }
die() { log "FAIL: $*"; exit 1; }
cd "$REPO"

log "=== replacement-pod migration starts: $(nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader)"
bash "$B" check

# --- assets: verify the bundle, then verify (not rebuild) the persistent Env A cache on /workspace
(cd "$ASSETS" && sha256sum --quiet -c SHA256SUMS) || die "asset hashes do not match"
log "assets verified: $(wc -l < "$ASSETS/SHA256SUMS") files"
for f in data/shard_00000.parquet data/shard_00001.parquet data/shard_00002.parquet data/shard_00003.parquet \
         data/shard_00004.parquet data/shard_00005.parquet data/shard_00006.parquet data/shard_00007.parquet \
         data/shard_00008.parquet data/shard_00009.parquet data/shard_06542.parquet \
         tokenizer/tokenizer.pkl tokenizer/token_bytes.pt; do
  cmp -s "$ASSETS/autoresearch_cache/$f" "/workspace/.cache/autoresearch/$f" || die "persistent cache differs: $f"
done
log "persistent Env A cache on /workspace matches the asset bundle (13 files); contents: $(ls /workspace/.cache/autoresearch/data | wc -l) data, $(ls /workspace/.cache/autoresearch/tokenizer | wc -l) tokenizer files"
# container-disk parts of install_assets.sh env-a: runtime symlink, held-out shard, HF kernel cache
[ -e /root/.cache/autoresearch ] && die "/root/.cache/autoresearch already exists"
install -d -m 0755 /root/.cache /root/.cache/autopareto/downstream /root/.cache/huggingface/hub
ln -s /workspace/.cache/autoresearch /root/.cache/autoresearch
DOWN=915333b4f8b8684f39aeaafea600fea6f43fb703-shard_06541.parquet
install -m 0644 "$ASSETS/autopareto_downstream/$DOWN" "/root/.cache/autopareto/downstream/$DOWN"
cmp -s "$ASSETS/autopareto_downstream/$DOWN" "/root/.cache/autopareto/downstream/$DOWN" || die "held-out copy differs"
KREPO=models--kernels-community--flash-attn3
# Not cp -a (nor cp --preserve=mode): on an NFS /workspace, coreutils also copies the NFSv4 ACL, which the
# container disk refuses (exit 1). rsync -rlpt sets modes with chmod and keeps symlinks; contents are
# checked byte for byte below.
rsync -rlpt "$ASSETS/huggingface/hub/$KREPO/" "/root/.cache/huggingface/hub/$KREPO/"
(cd "$ASSETS/huggingface/hub/$KREPO" && find . \( -type f -o -type l \) -print0 | while IFS= read -r -d '' p; do
   cmp -s "$p" "/root/.cache/huggingface/hub/$KREPO/$p" || { echo "kernel cache differs: $p"; exit 1; }
 done) || die "kernel cache copy differs"
[ "$(cat /root/.cache/huggingface/hub/$KREPO/refs/main)" = e29f138fc363b396e5d2706c8a5f6fa7d36f41e0 ] || die "refs/main is not e29f138f"
log "Env A runtime view, held-out shard and kernel cache e29f138f installed under /root/.cache"

# --- Env A: restore the uv-managed interpreter the /workspace venv points to, then check it.
# On a fresh volume restored from the Mac backup (which excludes the rebuildable .venv), build Env A
# with the runbook's own env-a stage instead.
if [ ! -e "$REPO/.venv" ]; then
  log "no Env A venv on this volume: building it with the runbook's env-a stage"
  bash "$B" env-a
else
  [ -e /opt/uv-tool ] && die "/opt/uv-tool already exists"
  echo "$UV_LOCK_SHA256  $REPO/uv.lock" | sha256sum --quiet -c - || die "uv.lock hash mismatch"
  python3 -m venv /opt/uv-tool
  /opt/uv-tool/bin/pip install --quiet "uv==0.9.0"
  /opt/uv-tool/bin/uv python install 3.10.18
  # --inexact keeps the venv exactly as installed on 2026-11-04 (including datasets==5.0.1 outside the lock)
  (cd "$REPO" && /opt/uv-tool/bin/uv sync --frozen --inexact --python 3.10.18)
fi
PYA="$REPO/.venv/bin/python"
"$PYA" - <<'PY' || die "Env A pins"
import importlib.metadata as m, sys
assert sys.version.split()[0] == "3.10.18", sys.version
want = {"torch": "2.9.1+cu128", "triton": "3.5.1", "kernels": "0.12.1", "tiktoken": "0.12.0", "rustbpe": "0.1.0",
        "pyarrow": "23.0.1", "numpy": "2.2.6", "huggingface-hub": "1.5.0", "matplotlib": "3.10.8", "datasets": "5.0.1"}
bad = {k: (m.version(k), v) for k, v in want.items() if m.version(k) != v}
assert not bad, bad
print("Env A pins OK:", ", ".join(f"{k}=={v}" for k, v in want.items()))
PY
log "Env A restored and pinned"

# --- Env B, Pair-A trusted state (unchanged stages)
bash "$B" env-b
bash "$B" pair-a

# --- restore the paired-baseline state (G2 attempt 2 and attempt 1) from the /workspace mirror, hash-checked
for d in pair_a_claude_opus55_baseline pair_a_claude_opus55_baseline_attempt2; do
  [ -e "/var/lib/autopareto/$d" ] && die "/var/lib/autopareto/$d already exists"
  rsync -a --no-owner --no-group "/workspace/mirror/var_lib_autopareto/$d/" "/var/lib/autopareto/$d/"
done
chown -R root:root /var/lib/autopareto
find /var/lib/autopareto -mindepth 1 -maxdepth 1 -name 'pair_a_claude_opus55_baseline*' -exec chmod -R go-rwx {} +
P=/var/lib/autopareto/pair_a_claude_opus55_baseline_attempt2/nonstudy_tests
echo "$T_SHA  $P/t_smoke_artifacts/result.checkpoint.pt" | sha256sum --quiet -c - || die "T checkpoint hash mismatch"
echo "$N_SHA  $P/n_smoke_artifacts/result.checkpoint.pt" | sha256sum --quiet -c - || die "N checkpoint hash mismatch"
log "paired-baseline checkpoints restored from the mirror; T f7ca17c6 and N 72d05e97 verified"

# --- verify stage: Linux sandbox tests, native tokenizer retrain, E0/E1 validate-only, receipt
bash "$B" verify
(cd "$REPO" && HF_HUB_OFFLINE=1 /opt/uv-tool/bin/uv run --locked python scripts/run_camera_ready_e0_e1.py \
  --expect-plan-sha256 89f2901588fceec18ab93d4e74ed652b8f6c6df9329eb8ba1d3a7ead128f60dc \
  > "$LOG_DIR/e0_e1_validate_only_amendment1_$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1) || die "amendment-1 validate-only failed"
log "amendment-1 validate-only: $(grep -h "^summary" "$LOG_DIR"/e0_e1_validate_only_amendment1_*.log | tail -1)"
log "=== replacement-pod migration finished"
