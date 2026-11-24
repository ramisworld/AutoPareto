#!/usr/bin/env bash
# install_assets.sh -- install the verified AutoPareto data/tokenizer/kernel assets on a pod.
#
#   install_assets.sh env-a
#       Phase-1 / E0 / E1 environment (Env A). Installs, refusing to overwrite anything:
#         ${AUTOPARETO_PERSISTENT_CACHE:-/workspace/.cache/autoresearch}/data/shard_000{00..09}.parquet
#         ${AUTOPARETO_PERSISTENT_CACHE:-/workspace/.cache/autoresearch}/data/shard_06542.parquet
#         ${AUTOPARETO_PERSISTENT_CACHE:-/workspace/.cache/autoresearch}/tokenizer/{tokenizer.pkl,token_bytes.pt}
#         $HOME/.cache/autoresearch -> the persistent cache above (symlink, exactly what
#             scripts/ensure_persistent_cache.py creates; an existing symlink to the same
#             target is accepted, anything else at that path is refused)
#         $HOME/.cache/autopareto/downstream/915333b4f8b8684f39aeaafea600fea6f43fb703-shard_06541.parquet
#         <hub>/models--kernels-community--flash-attn3/{blobs,snapshots/e29f138f...,refs/main}
#             where <hub> = $HF_HUB_CACHE, else $HF_HOME/hub, else ${XDG_CACHE_HOME:-$HOME/.cache}/huggingface/hub
#             (the huggingface_hub resolution order). refs/main = e29f138f..., so an unpinned
#             kernels.get_kernel("kernels-community/flash-attn3") resolves offline
#             (HF_HUB_OFFLINE=1) to build/torch29-cxx11-cu128-x86_64-linux under torch 2.9
#             and build/torch28-cxx11-cu128-x86_64-linux under torch 2.8.
#
#   install_assets.sh pair-a <registered_cache_dir>
#       Pair-A registered cache (Env B). <registered_cache_dir> must not exist
#       (normally /var/cache/autopareto/pair_a_registered_cache). Creates:
#         <dir>/data/*.parquet, <dir>/tokenizer/*     (data_manifest.json relative paths, 13 files)
#         <dir>/huggingface/hub/models--kernels-community--flash-attn3/{blobs,snapshots/e29f138f...,refs/main}
#       Files are created 0644 and directories 0755; the caller then sets root ownership,
#       directories 0555 and files 0444 (and checks there are no ACLs).
#
# Every installed file is re-hashed (sha256sum) after copying and compared with the values
# embedded below (the same values as SHA256SUMS). Snapshot entries are relative symlinks
# into blobs/ and are checked to resolve inside the repository cache directory.
# Uses bash + coreutils only. Exit status is non-zero on any problem.
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
KREV="e29f138fc363b396e5d2706c8a5f6fa7d36f41e0"
DREV="915333b4f8b8684f39aeaafea600fea6f43fb703"
REPO="models--kernels-community--flash-attn3"

die() { echo "install_assets.sh: ERROR: $*" >&2; exit 1; }
say() { echo "install_assets.sh: $*"; }

# kind|path relative to the asset folder|sha256 (for links: of the resolved file)|link target (links only)
MANIFEST='file|autoresearch_cache/data/shard_00000.parquet|054ddbd98abf30d773c54de578fc9d579bafeb6c14e04e97bd36aa90e825bf9b|
file|autoresearch_cache/data/shard_00001.parquet|d4cfe1de19f4fd976022a9047458655968bc5583582ec5fa714801ec2be2c93c|
file|autoresearch_cache/data/shard_00002.parquet|5e80d7596d53be648d841ea5811082747281c39f6851e10dd90375a84bc509a7|
file|autoresearch_cache/data/shard_00003.parquet|b349c85d4b993d98e2328833ecfd3734692e0395a592d89858bae34a4ba73c59|
file|autoresearch_cache/data/shard_00004.parquet|87dae96fe1a6116c54c7873aadb81832178a36755991befd1b3b641c76244474|
file|autoresearch_cache/data/shard_00005.parquet|3d9aebf138ef69cff2585a686d7f21585b7ffc0de4655f8e7723590418583bd6|
file|autoresearch_cache/data/shard_00006.parquet|6a3f2afde8f1a39f9283ea9d9e9ee6431634d6881b63798176bf64efc3f33576|
file|autoresearch_cache/data/shard_00007.parquet|731eea337d941e5b18a9a4e2bbe73a517dac45d91bb93ae2e69346c68c5f37f6|
file|autoresearch_cache/data/shard_00008.parquet|e78379132451838afabed9034b5483a2f3bc5a422bed8b55b99f2bbb9a128755|
file|autoresearch_cache/data/shard_00009.parquet|f22410fe8f13b6976a2860a8a82ee71e838f4ee9b605d61584aaa963f966e120|
file|autoresearch_cache/data/shard_06542.parquet|769fe59d108dfd2cfa186c63173b83fbfb90a7adb3519519ba6eaa6ca9889f94|
file|autoresearch_cache/tokenizer/token_bytes.pt|23517e0590d6aa9389c3263fab1f6062b2810b7c398a7b380291857639ac8d98|
file|autoresearch_cache/tokenizer/tokenizer.pkl|888c9fa865c72c25f9c5e9467136a9cc8655253db459fb97481deb326681c629|
file|autopareto_downstream/915333b4f8b8684f39aeaafea600fea6f43fb703-shard_06541.parquet|51da1a9bc052dfd5a77988d6ca4d1cbe62b89eedb247b56850fc1b9b45e56c73|
file|huggingface/hub/models--kernels-community--flash-attn3/blobs/02dbf81201d16b6791a086b87a8b46dee7f45122|8f5f15c7e10c1f7128bda2eaee5254fdf6db78d9abadb02a1faa246dce0f7c36|
file|huggingface/hub/models--kernels-community--flash-attn3/blobs/03dbc1afe1cf156661a2b1b22003cd5f599a0309|0757fb1e297d4d48dfb3fe1ee969744dd1079c322d1227017527d8f791a6a62a|
file|huggingface/hub/models--kernels-community--flash-attn3/blobs/0cd58331b2a989b68be4ec5676383437fca8687b|aa0f55007ee70e2a79b9b3538b50daee4026794e9bdf7c811bba8cf3ffbba479|
file|huggingface/hub/models--kernels-community--flash-attn3/blobs/1d9ac17ca94ff06dfccb0a48aeefbdf75e28043c|cf66e040c4e61bb0c875e5c393202e6e7a7a48a7d89f4b28a633c4746d759570|
file|huggingface/hub/models--kernels-community--flash-attn3/blobs/2391775caf0ae40d0fd5d06a011b60d47380ce42527ea84ef0cff3e88330a95c|2391775caf0ae40d0fd5d06a011b60d47380ce42527ea84ef0cff3e88330a95c|
file|huggingface/hub/models--kernels-community--flash-attn3/blobs/2e2a3cdd3e89cbc59bf6c956454e2776c305e0e5|cb46c3d89606305817ea20eb7b7cc9376b013eeabfd7637a328e7967e1d008d4|
file|huggingface/hub/models--kernels-community--flash-attn3/blobs/385c5960f738f6c2f217ca93a46fb9be003e7076d60a5711cef8bebdf19eee20|385c5960f738f6c2f217ca93a46fb9be003e7076d60a5711cef8bebdf19eee20|
file|huggingface/hub/models--kernels-community--flash-attn3/blobs/3d12edd447e84d6e65c4fa9ab91ae3c3192884c1|29756640933e2a159cd94a9625eb2ea4268ffa6a80bd3c7bc86b9e1404e09471|
file|huggingface/hub/models--kernels-community--flash-attn3/blobs/6c1ffe7402bae1934fd296c76aadd8fc89b2d77e|483782b5b780bbb856d227a330281d58c55c8fed59ba0155e955ca487a6d6b20|
file|huggingface/hub/models--kernels-community--flash-attn3/blobs/9cf5deed9898dce769f4cc73913d3530b92a0bd8|8b3dd246b49a19923c0ffbddf525fb031af9c9cc683ca618756ad7ed632cd50c|
file|huggingface/hub/models--kernels-community--flash-attn3/blobs/c4b7130324be2c4cabee7ada7000e170a2a81865|bb1a3e7b231c0e2b7fffc603692e11b64e414cc2b34d22bdb4e2938dd5957bfc|
file|huggingface/hub/models--kernels-community--flash-attn3/blobs/d7b48ccbd63561f4f1e903fc5c7cb8fb864e003f|685a205d769a6b2305afb7094a5c3a44c7492361927ebe8138d485ad0eea1c39|
file|huggingface/hub/models--kernels-community--flash-attn3/blobs/e0cc0c4f14abd10162c22499be39c75e6d922f94|bf7062340012da017522a7f276f972958e7c8317417b43ff19cfd3a482d63827|
file|huggingface/hub/models--kernels-community--flash-attn3/refs/main|6081bac91c1c94b81632c860d5e5459a9af6faf4b8669b1ed1ee4d1e99b0302a|
link|huggingface/hub/models--kernels-community--flash-attn3/snapshots/e29f138fc363b396e5d2706c8a5f6fa7d36f41e0/.gitattributes|aa0f55007ee70e2a79b9b3538b50daee4026794e9bdf7c811bba8cf3ffbba479|../../blobs/0cd58331b2a989b68be4ec5676383437fca8687b
link|huggingface/hub/models--kernels-community--flash-attn3/snapshots/e29f138fc363b396e5d2706c8a5f6fa7d36f41e0/README.md|483782b5b780bbb856d227a330281d58c55c8fed59ba0155e955ca487a6d6b20|../../blobs/6c1ffe7402bae1934fd296c76aadd8fc89b2d77e
link|huggingface/hub/models--kernels-community--flash-attn3/snapshots/e29f138fc363b396e5d2706c8a5f6fa7d36f41e0/benchmark.py|685a205d769a6b2305afb7094a5c3a44c7492361927ebe8138d485ad0eea1c39|../../blobs/d7b48ccbd63561f4f1e903fc5c7cb8fb864e003f
link|huggingface/hub/models--kernels-community--flash-attn3/snapshots/e29f138fc363b396e5d2706c8a5f6fa7d36f41e0/benchmarks/benchmark.py|685a205d769a6b2305afb7094a5c3a44c7492361927ebe8138d485ad0eea1c39|../../../blobs/d7b48ccbd63561f4f1e903fc5c7cb8fb864e003f
link|huggingface/hub/models--kernels-community--flash-attn3/snapshots/e29f138fc363b396e5d2706c8a5f6fa7d36f41e0/build/torch28-cxx11-cu128-x86_64-linux/__init__.py|29756640933e2a159cd94a9625eb2ea4268ffa6a80bd3c7bc86b9e1404e09471|../../../../blobs/3d12edd447e84d6e65c4fa9ab91ae3c3192884c1
link|huggingface/hub/models--kernels-community--flash-attn3/snapshots/e29f138fc363b396e5d2706c8a5f6fa7d36f41e0/build/torch28-cxx11-cu128-x86_64-linux/_flash_attn3_5630d32.abi3.so|2391775caf0ae40d0fd5d06a011b60d47380ce42527ea84ef0cff3e88330a95c|../../../../blobs/2391775caf0ae40d0fd5d06a011b60d47380ce42527ea84ef0cff3e88330a95c
link|huggingface/hub/models--kernels-community--flash-attn3/snapshots/e29f138fc363b396e5d2706c8a5f6fa7d36f41e0/build/torch28-cxx11-cu128-x86_64-linux/_ops.py|8f5f15c7e10c1f7128bda2eaee5254fdf6db78d9abadb02a1faa246dce0f7c36|../../../../blobs/02dbf81201d16b6791a086b87a8b46dee7f45122
link|huggingface/hub/models--kernels-community--flash-attn3/snapshots/e29f138fc363b396e5d2706c8a5f6fa7d36f41e0/build/torch28-cxx11-cu128-x86_64-linux/flash_attn3/__init__.py|0757fb1e297d4d48dfb3fe1ee969744dd1079c322d1227017527d8f791a6a62a|../../../../../blobs/03dbc1afe1cf156661a2b1b22003cd5f599a0309
link|huggingface/hub/models--kernels-community--flash-attn3/snapshots/e29f138fc363b396e5d2706c8a5f6fa7d36f41e0/build/torch28-cxx11-cu128-x86_64-linux/flash_attn_config.py|bb1a3e7b231c0e2b7fffc603692e11b64e414cc2b34d22bdb4e2938dd5957bfc|../../../../blobs/c4b7130324be2c4cabee7ada7000e170a2a81865
link|huggingface/hub/models--kernels-community--flash-attn3/snapshots/e29f138fc363b396e5d2706c8a5f6fa7d36f41e0/build/torch28-cxx11-cu128-x86_64-linux/flash_attn_interface.py|cb46c3d89606305817ea20eb7b7cc9376b013eeabfd7637a328e7967e1d008d4|../../../../blobs/2e2a3cdd3e89cbc59bf6c956454e2776c305e0e5
link|huggingface/hub/models--kernels-community--flash-attn3/snapshots/e29f138fc363b396e5d2706c8a5f6fa7d36f41e0/build/torch28-cxx11-cu128-x86_64-linux/metadata.json|8b3dd246b49a19923c0ffbddf525fb031af9c9cc683ca618756ad7ed632cd50c|../../../../blobs/9cf5deed9898dce769f4cc73913d3530b92a0bd8
link|huggingface/hub/models--kernels-community--flash-attn3/snapshots/e29f138fc363b396e5d2706c8a5f6fa7d36f41e0/build/torch29-cxx11-cu128-x86_64-linux/__init__.py|29756640933e2a159cd94a9625eb2ea4268ffa6a80bd3c7bc86b9e1404e09471|../../../../blobs/3d12edd447e84d6e65c4fa9ab91ae3c3192884c1
link|huggingface/hub/models--kernels-community--flash-attn3/snapshots/e29f138fc363b396e5d2706c8a5f6fa7d36f41e0/build/torch29-cxx11-cu128-x86_64-linux/_flash_attn3_cuda_2d990fe.abi3.so|385c5960f738f6c2f217ca93a46fb9be003e7076d60a5711cef8bebdf19eee20|../../../../blobs/385c5960f738f6c2f217ca93a46fb9be003e7076d60a5711cef8bebdf19eee20
link|huggingface/hub/models--kernels-community--flash-attn3/snapshots/e29f138fc363b396e5d2706c8a5f6fa7d36f41e0/build/torch29-cxx11-cu128-x86_64-linux/_ops.py|cf66e040c4e61bb0c875e5c393202e6e7a7a48a7d89f4b28a633c4746d759570|../../../../blobs/1d9ac17ca94ff06dfccb0a48aeefbdf75e28043c
link|huggingface/hub/models--kernels-community--flash-attn3/snapshots/e29f138fc363b396e5d2706c8a5f6fa7d36f41e0/build/torch29-cxx11-cu128-x86_64-linux/flash_attn3/__init__.py|0757fb1e297d4d48dfb3fe1ee969744dd1079c322d1227017527d8f791a6a62a|../../../../../blobs/03dbc1afe1cf156661a2b1b22003cd5f599a0309
link|huggingface/hub/models--kernels-community--flash-attn3/snapshots/e29f138fc363b396e5d2706c8a5f6fa7d36f41e0/build/torch29-cxx11-cu128-x86_64-linux/flash_attn_config.py|bb1a3e7b231c0e2b7fffc603692e11b64e414cc2b34d22bdb4e2938dd5957bfc|../../../../blobs/c4b7130324be2c4cabee7ada7000e170a2a81865
link|huggingface/hub/models--kernels-community--flash-attn3/snapshots/e29f138fc363b396e5d2706c8a5f6fa7d36f41e0/build/torch29-cxx11-cu128-x86_64-linux/flash_attn_interface.py|cb46c3d89606305817ea20eb7b7cc9376b013eeabfd7637a328e7967e1d008d4|../../../../blobs/2e2a3cdd3e89cbc59bf6c956454e2776c305e0e5
link|huggingface/hub/models--kernels-community--flash-attn3/snapshots/e29f138fc363b396e5d2706c8a5f6fa7d36f41e0/build/torch29-cxx11-cu128-x86_64-linux/metadata.json|bf7062340012da017522a7f276f972958e7c8317417b43ff19cfd3a482d63827|../../../../blobs/e0cc0c4f14abd10162c22499be39c75e6d922f94'

usage() {
  awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0" >&2
  exit 2
}

[ "$#" -ge 1 ] || usage
MODE="$1"

for tool in sha256sum install ln readlink mkdir dirname; do
  command -v "$tool" >/dev/null 2>&1 || die "required tool not found: $tool"
done

# The tokenizer files are regenerated artifacts; refuse loudly if they are not in the bundle.
for f in autoresearch_cache/tokenizer/tokenizer.pkl autoresearch_cache/tokenizer/token_bytes.pt; do
  [ -f "$SRC/$f" ] || die "tokenizer file missing from the asset bundle: $SRC/$f (the tokenizer was not reproduced; regenerate it with prepare.py and verify 888c9fa8.../23517e05... before installing)"
done

# --- planning: build "kind|source|destination|sha256" lines -------------------------------
PLAN=""
add() { PLAN+="$1|$2|$3|$4|$5"$'\n'; }

map_entries() { # $1 = prefix for autoresearch_cache, $2 = hub dir, $3 = downstream dir (or empty)
  local kind rel sha target dest
  while IFS='|' read -r kind rel sha target; do
    [ -n "$kind" ] || continue
    case "$rel" in
      autoresearch_cache/*) dest="$1/${rel#autoresearch_cache/}" ;;
      autopareto_downstream/*)
        [ -n "$3" ] || continue
        dest="$3/${rel#autopareto_downstream/}" ;;
      huggingface/hub/*) dest="$2/${rel#huggingface/hub/}" ;;
      *) die "unmapped manifest entry: $rel" ;;
    esac
    add "$kind" "$rel" "$dest" "$sha" "$target"
  done <<< "$MANIFEST"
}

case "$MODE" in
  env-a)
    [ "$#" -eq 1 ] || usage
    : "${HOME:?HOME must be set}"
    PERSIST="${AUTOPARETO_PERSISTENT_CACHE:-/workspace/.cache/autoresearch}"
    RUNTIME="$HOME/.cache/autoresearch"
    if [ -n "${HF_HUB_CACHE:-}" ]; then
      HUB="$HF_HUB_CACHE"
    elif [ -n "${HF_HOME:-}" ]; then
      HUB="$HF_HOME/hub"
    else
      HUB="${XDG_CACHE_HOME:-$HOME/.cache}/huggingface/hub"
    fi
    if [ -n "${HF_HUB_CACHE:-}" ] && [ "$HF_HUB_CACHE" != "${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}/hub" ]; then
      say "WARNING: HF_HUB_CACHE ($HF_HUB_CACHE) differs from HF_HOME/hub; the E0/E1 executor records the kernel revision from HF_HOME/hub."
    fi
    DOWN="$HOME/.cache/autopareto/downstream"
    # ~/.cache/autoresearch must be absent or already the persistent-cache symlink.
    if [ -L "$RUNTIME" ]; then
      [ "$(readlink -f "$RUNTIME")" = "$(readlink -m "$PERSIST")" ] \
        || die "$RUNTIME is a symlink to $(readlink -f "$RUNTIME"), expected $PERSIST"
      MAKE_RUNTIME_LINK=0
    elif [ -e "$RUNTIME" ]; then
      die "$RUNTIME exists and is not a symlink to $PERSIST (scripts/ensure_persistent_cache.py would also refuse it); move it away first"
    else
      MAKE_RUNTIME_LINK=1
    fi
    if [ -e "$HUB/$REPO" ] || [ -L "$HUB/$REPO" ]; then
      die "$HUB/$REPO already exists; refusing to merge into an existing kernel cache (inspect or move it away first)"
    fi
    map_entries "$PERSIST" "$HUB" "$DOWN"
    ;;
  pair-a)
    [ "$#" -eq 2 ] || usage
    TARGET="$2"
    case "$TARGET" in /*) ;; *) die "registered cache dir must be an absolute path: $TARGET" ;; esac
    if [ -e "$TARGET" ] || [ -L "$TARGET" ]; then die "target exists, refusing: $TARGET"; fi
    map_entries "$TARGET" "$TARGET/huggingface/hub" ""
    ;;
  -h|--help|help) usage ;;
  *) usage ;;
esac

# --- preflight: every source present, no destination present ------------------------------
while IFS='|' read -r kind rel dest sha target; do
  [ -n "$kind" ] || continue
  if [ "$kind" = file ]; then
    [ -f "$SRC/$rel" ] || die "missing source file: $SRC/$rel"
  fi
  if [ -e "$dest" ] || [ -L "$dest" ]; then die "destination exists, refusing to overwrite: $dest"; fi
done <<< "$PLAN"

# --- install --------------------------------------------------------------------------------
if [ "$MODE" = pair-a ]; then
  install -d -m 0755 "$TARGET"
fi
while IFS='|' read -r kind rel dest sha target; do
  [ -n "$kind" ] || continue
  install -d -m 0755 "$(dirname "$dest")"
  case "$kind" in
    file) install -m 0644 "$SRC/$rel" "$dest" ;;
    link)
      # recreate the relative symlink exactly as huggingface_hub lays it out
      # (snapshots/<rev>/<path> -> ../../blobs/<etag>, depth-adjusted)
      case "$target" in ../*blobs/*) ;; *) die "unexpected link target for $rel: $target" ;; esac
      ln -s "$target" "$dest"
      ;;
    *) die "unknown kind $kind" ;;
  esac
done <<< "$PLAN"
if [ "$MODE" = env-a ] && [ "$MAKE_RUNTIME_LINK" = 1 ]; then
  install -d -m 0755 "$(dirname "$RUNTIME")"
  ln -s "$PERSIST" "$RUNTIME"
fi

# --- verify ---------------------------------------------------------------------------------
fail=0; n=0
while IFS='|' read -r kind rel dest sha target; do
  [ -n "$kind" ] || continue
  n=$((n+1))
  if [ "$kind" = link ]; then
    [ -L "$dest" ] || { echo "NOT A SYMLINK: $dest" >&2; fail=1; continue; }
    repo_dir="${dest%%/snapshots/*}"
    resolved="$(readlink -f "$dest")"
    case "$resolved" in "$(readlink -f "$repo_dir")"/blobs/*) ;; *) echo "SYMLINK ESCAPES: $dest -> $resolved" >&2; fail=1; continue ;; esac
  fi
  actual="$(sha256sum "$dest" | cut -d' ' -f1)"
  if [ "$actual" != "$sha" ]; then echo "SHA256 MISMATCH: $dest ($actual != $sha)" >&2; fail=1; fi
done <<< "$PLAN"
# the env-a runtime view must resolve to the same files
if [ "$MODE" = env-a ]; then
  for f in tokenizer/tokenizer.pkl tokenizer/token_bytes.pt data/shard_06542.parquet; do
    [ -f "$RUNTIME/$f" ] || { echo "RUNTIME VIEW MISSING: $RUNTIME/$f" >&2; fail=1; }
  done
fi
[ "$fail" = 0 ] || die "verification failed; see messages above"
say "installed and verified $n entries (mode $MODE)"
case "$MODE" in
  env-a)
    say "data cache: $PERSIST (runtime view $RUNTIME)"
    say "held-out:   $DOWN/$DREV-shard_06541.parquet"
    say "kernel:     $HUB/$REPO (refs/main = $KREV)"
    ;;
  pair-a)
    say "registered cache: $TARGET  (next: chown -R root:root; find dirs -> 0555, files -> 0444; check no ACLs)"
    ;;
esac
