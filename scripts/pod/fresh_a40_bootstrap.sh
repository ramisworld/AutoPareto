#!/usr/bin/env bash
# Reproducible setup of a fresh RunPod A40 for the AutoPareto camera-ready work.
#
# Run as root on the pod, from /workspace/autopareto-camera-ready, after the
# Mac-side transfer in audit/camera_ready_2026/13_FRESH_POD_SETUP.md.
#
#   bash scripts/pod/fresh_a40_bootstrap.sh check|assets|env-a|env-b|pair-a|verify|mirror|all
#
# No stage trains a model; the GPU is only queried with nvidia-smi. Each stage
# refuses to overwrite existing state, and every input is verified by SHA-256.
set -euo pipefail

REPO=/workspace/autopareto-camera-ready
ASSETS=/workspace/assets                 # copy of the Mac's autopareto_assets_20261102
LOG_DIR=/workspace/pod_setup_logs
UV_VERSION=0.9.0
PY_A=3.10.18
PLAN_SHA256=806a63970258627092d929d4602e54e36a2412e0656a63298e17307c6c829340
UV_LOCK_SHA256=262a4d55eaa24f47c98ca44cc82ff1660d5daba73df0b1e78246cceac2c429b4
RUNTIME_B=/opt/autopareto_runtime
TRUSTED=/opt/autopareto_trusted
REGISTERED_CACHE=/var/cache/autopareto/pair_a_registered_cache
CANDIDATE_INPUTS=/opt/autopareto_candidate_inputs
CANDIDATE_UID=2001
TOKENIZER_PKL_SHA256=888c9fa865c72c25f9c5e9467136a9cc8655253db459fb97481deb326681c629
TOKEN_BYTES_SHA256=23517e0590d6aa9389c3263fab1f6062b2810b7c398a7b380291857639ac8d98

mkdir -p "$LOG_DIR"
log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG_DIR/bootstrap.log"; }
die() { log "FAIL: $*"; exit 1; }
need_absent() { [ ! -e "$1" ] || die "$1 already exists; refusing to overwrite (inspect it first)"; }
pin() {  # pin <python> <module> <expected version>
  local got
  got=$("$1" -c "import importlib.metadata as m; print(m.version('$2'))")
  [ "$got" = "$3" ] || die "$2 is $got, expected $3"
  log "pinned $2==$got"
}

stage_check() {
  [ "$(id -u)" = 0 ] || die "must run as root"
  [ "$(uname -m)" = x86_64 ] || die "candidate sandbox has a syscall map only for x86_64"
  command -v setpriv >/dev/null || die "setpriv (util-linux) is missing"
  local gpu
  gpu=$(nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader)
  [ "$(printf '%s\n' "$gpu" | wc -l)" = 1 ] || die "expected exactly one GPU: $gpu"
  case "$gpu" in *A40*) ;; *) die "GPU is not an A40: $gpu" ;; esac
  local cuda
  cuda=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -1)
  python3 - "$cuda" <<'PY' || die "driver CUDA $cuda is older than 12.8"
import sys
major, minor = (int(x) for x in sys.argv[1].split(".")[:2])
sys.exit(0 if (major, minor) >= (12, 8) else 1)
PY
  python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' || die "base image Python is not 3.12"
  python3 -c 'import torch; assert torch.__version__ == "2.8.0+cu128", torch.__version__' \
    || die "base image torch is not 2.8.0+cu128 (use runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404)"
  [ -d /workspace ] || die "/workspace volume is missing"
  local root_free ws_free
  root_free=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
  ws_free=$(df -BG --output=avail /workspace | tail -1 | tr -dc 0-9)
  [ "$root_free" -ge 40 ] || die "container disk has ${root_free}G free; need >= 40G"
  [ "$ws_free" -ge 100 ] || die "/workspace has ${ws_free}G free; need >= 100G"
  [ -z "${PYTORCH_CUDA_ALLOC_CONF:-}${PYTORCH_ALLOC_CONF:-}" ] || die "unset PYTORCH_CUDA_ALLOC_CONF/PYTORCH_ALLOC_CONF"
  log "gpu: $gpu | driver CUDA $cuda | kernel $(uname -r) | free: / ${root_free}G, /workspace ${ws_free}G"
  log "seccomp state of this shell: $(grep Seccomp: /proc/self/status | tr -s ' \t' ' ')"
}

stage_assets() {
  [ -f "$ASSETS/SHA256SUMS" ] || die "missing $ASSETS/SHA256SUMS (transfer the asset folder first)"
  (cd "$ASSETS" && sha256sum --quiet -c SHA256SUMS) || die "asset hashes do not match"
  log "assets verified: $(wc -l < "$ASSETS/SHA256SUMS") files"
  [ -f "$ASSETS/install_assets.sh" ] || die "missing $ASSETS/install_assets.sh"
  # The asset folder carries its own layout-specific installer (see its ASSETS.md):
  # it places the data cache, tokenizer and HF kernel cache for Env A under ~/.cache.
  bash "$ASSETS/install_assets.sh" env-a
}

stage_env_a() {
  [ -f "$REPO/uv.lock" ] || die "missing $REPO/uv.lock (copy it from autopareto/research-infrastructure)"
  echo "$UV_LOCK_SHA256  $REPO/uv.lock" | sha256sum --quiet -c - || die "uv.lock hash mismatch"
  need_absent "$REPO/.venv"
  need_absent /opt/uv-tool
  python3 -m venv /opt/uv-tool
  /opt/uv-tool/bin/pip install --quiet "uv==$UV_VERSION"
  local uv=/opt/uv-tool/bin/uv
  "$uv" python install "$PY_A"
  (cd "$REPO" && "$uv" sync --frozen --python "$PY_A")
  # Outside uv.lock, as recorded for the downstream suite (disclosed deviation).
  "$uv" pip install --python "$REPO/.venv/bin/python" "datasets==5.0.1"
  local py="$REPO/.venv/bin/python"
  "$py" -c 'import sys; assert sys.version.split()[0] == "3.10.18", sys.version' || die "Env A Python is not 3.10.18"
  pin "$py" torch 2.9.1+cu128; pin "$py" triton 3.5.1; pin "$py" kernels 0.12.1
  pin "$py" tiktoken 0.12.0; pin "$py" rustbpe 0.1.0; pin "$py" pyarrow 23.0.1
  pin "$py" numpy 2.2.6; pin "$py" huggingface-hub 1.5.0; pin "$py" matplotlib 3.10.8
  pin "$py" datasets 5.0.1
}

stage_env_b() {
  need_absent "$RUNTIME_B"
  python3 -m venv --system-site-packages "$RUNTIME_B"
  "$RUNTIME_B/bin/pip" install --quiet pyarrow==23.0.1 rustbpe==0.1.0 tiktoken==0.12.0 \
    huggingface_hub==1.30.0 kernels==0.12.1 cmaes==0.12.0 matplotlib==3.10.8 numpy==2.2.6 \
    optuna==4.5.0 pandas==2.3.3 requests==2.32.5
  "$RUNTIME_B/bin/pip" install --quiet triton==3.4.0
  local py="$RUNTIME_B/bin/python"
  pin "$py" torch 2.8.0+cu128; pin "$py" triton 3.4.0; pin "$py" kernels 0.12.1
  pin "$py" huggingface-hub 1.30.0; pin "$py" tiktoken 0.12.0; pin "$py" pyarrow 23.0.1
}

stage_pair_a() {
  getent group autopareto-candidate >/dev/null || groupadd --system --gid "$CANDIDATE_UID" autopareto-candidate
  # The sandbox uses only this uid/gid; each run gets its own candidate home.
  getent passwd autopareto-candidate >/dev/null || useradd --system --uid "$CANDIDATE_UID" --gid "$CANDIDATE_UID" \
    --home-dir /nonexistent --shell /usr/sbin/nologin autopareto-candidate
  [ "$(id -u autopareto-candidate)" = "$CANDIDATE_UID" ] || die "candidate uid is not $CANDIDATE_UID"
  install -d -o root -g root -m 0700 /var/lib/autopareto
  install -d -o root -g root -m 0755 /opt/autopareto-candidate-attempts /opt/autopareto_candidate_contexts
  install -d -o "$CANDIDATE_UID" -g "$CANDIDATE_UID" -m 0700 /opt/autopareto-candidate-tmp

  need_absent "$TRUSTED"
  # The candidate sandbox requires a root-only trusted tree (mode & 077 == 0).
  install -d -o root -g root -m 0700 "$TRUSTED"
  (cd "$REPO" && rsync -a --files-from=scripts/pod/trusted_files.txt ./ "$TRUSTED/")
  chown -R root:root "$TRUSTED"
  chmod 0700 "$TRUSTED"
  log "trusted repository: $(find "$TRUSTED" -type f | wc -l) files (expected 122); hashes are checked by the controller's run gate"

  need_absent "$REGISTERED_CACHE"
  bash "$ASSETS/install_assets.sh" pair-a "$REGISTERED_CACHE"
  chown -R root:root "$REGISTERED_CACHE"
  find "$REGISTERED_CACHE" -type d -exec chmod 0555 {} +
  find "$REGISTERED_CACHE" -type f -exec chmod 0444 {} +

  # Same layout as prepare_clean_room.py produced: cache/{autoresearch,huggingface}, read-only.
  need_absent "$CANDIDATE_INPUTS"
  install -d -o root -g root -m 0755 "$CANDIDATE_INPUTS" "$CANDIDATE_INPUTS/per_attempts" \
    "$CANDIDATE_INPUTS/cache" "$CANDIDATE_INPUTS/cache/autoresearch"
  rsync -a --exclude huggingface "$REGISTERED_CACHE/" "$CANDIDATE_INPUTS/cache/autoresearch/"
  cp -a "$REGISTERED_CACHE/huggingface" "$CANDIDATE_INPUTS/cache/huggingface"
  chown -R root:root "$CANDIDATE_INPUTS/cache"
  find "$CANDIDATE_INPUTS/cache" -type d -exec chmod 0555 {} +
  find "$CANDIDATE_INPUTS/cache" -type f -exec chmod 0444 {} +
  # The candidate boundary rejects ACLs on its read-only inputs.
  if find "$REGISTERED_CACHE" "$CANDIDATE_INPUTS/cache" -print0 | xargs -0 ls -ld | awk '{print $1}' | grep -q '+'; then
    die "ACLs present on the registered cache or candidate inputs"
  fi
  log "pair-a state created"
}

stage_verify() {
  local receipt="$LOG_DIR/readiness_$(date -u +%Y%m%dT%H%M%SZ).json"
  # Linux-only sandbox tests that are skipped on the Mac run here. The Docker
  # timeout test is Codex-only and is expected to error (no Docker on the pod).
  (cd "$REPO" && "$RUNTIME_B/bin/python" -m unittest discover -s study/contracts_2026/tests -t . \
    > "$LOG_DIR/unittest_env_b.log" 2>&1) || true
  # Native tokenizer confirmation: the installed token_bytes.pt was derived on macOS (see the
  # asset folder's ASSETS.md); retraining here must reproduce both registered hashes.
  local scratch
  scratch=$(mktemp -d)
  install -d "$scratch/.cache/autoresearch/data"
  cp /workspace/.cache/autoresearch/data/*.parquet "$scratch/.cache/autoresearch/data/"
  (cd "$REPO" && HOME="$scratch" "$REPO/.venv/bin/python" prepare.py > "$LOG_DIR/tokenizer_native.log" 2>&1) \
    || die "native tokenizer retraining failed (see $LOG_DIR/tokenizer_native.log)"
  echo "$TOKENIZER_PKL_SHA256  $scratch/.cache/autoresearch/tokenizer/tokenizer.pkl" | sha256sum --quiet -c - \
    || die "natively retrained tokenizer.pkl does not match the registered hash"
  echo "$TOKEN_BYTES_SHA256  $scratch/.cache/autoresearch/tokenizer/token_bytes.pt" | sha256sum --quiet -c - \
    || die "natively retrained token_bytes.pt does not match the registered hash"
  rm -rf "$scratch"
  log "tokenizer retrained natively; both registered hashes reproduce"
  (cd "$REPO" && HF_HUB_OFFLINE=1 /opt/uv-tool/bin/uv run --locked python scripts/run_camera_ready_e0_e1.py \
    --expect-plan-sha256 "$PLAN_SHA256" > "$LOG_DIR/e0_e1_validate_only.log" 2>&1) || true
  python3 - "$receipt" "$LOG_DIR" "$TRUSTED" <<'PY'
import hashlib, json, pathlib, re, subprocess, sys
receipt, logs, trusted = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
                     capture_output=True, text=True, check=True).stdout.strip()
tests = (logs / "unittest_env_b.log").read_text()
validate = (logs / "e0_e1_validate_only.log").read_text()
amended = ["bridge", "campaign_controller", "candidate_process", "isolation", "scientist_config", "train_launcher"]
payload = {
    "gpu": gpu,
    "kernel": subprocess.run(["uname", "-r"], capture_output=True, text=True).stdout.strip(),
    "unittest_env_b": (re.findall(r"^(Ran .*|OK.*|FAILED.*)$", tests, re.M) or ["no summary"])[-2:],
    "unittest_errors": re.findall(r"^(?:ERROR|FAIL): (\S+)", tests, re.M),
    "e0_e1_validate_only": (re.findall(r"^summary: .*$", validate, re.M) or ["no summary"])[-1],
    "trusted_amended_sha256": {
        name: hashlib.sha256((trusted / f"study/contracts_2026/{name}.py").read_bytes()).hexdigest()
        for name in amended
    },
}
receipt.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
  log "readiness receipt: $receipt"
}

stage_mirror() {
  # Container disk is wiped when a pod stops: mirror durable pod state to the volume.
  # /workspace (MooseFS over FUSE) refuses chown and ignores chmod (directories stay 0777), so
  # the mirror is a backup, not a trust boundary: verify hashes before restoring from it.
  # Candidates cannot reach it: /workspace is outside the deny-by-default Landlock allowlist.
  install -d -m 0700 /workspace/mirror
  rsync -a --no-owner --no-group /var/lib/autopareto/ /workspace/mirror/var_lib_autopareto/
  log "mirrored /var/lib/autopareto to /workspace/mirror"
}

case "${1:-}" in
  check) stage_check ;;
  assets) stage_assets ;;
  env-a) stage_env_a ;;
  env-b) stage_env_b ;;
  pair-a) stage_pair_a ;;
  verify) stage_verify ;;
  mirror) stage_mirror ;;
  all) stage_check; stage_assets; stage_env_a; stage_env_b; stage_pair_a; stage_verify ;;
  *) echo "usage: $0 check|assets|env-a|env-b|pair-a|verify|mirror|all" >&2; exit 2 ;;
esac
