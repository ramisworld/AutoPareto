#!/usr/bin/env bash
# Doc 16/17 cache fix: let the kernel's online tree listing exist before the campaign, readable.
# Performs the same online load the trusted post-training stage performs (Env B, HF_HOME = registered
# cache), then makes the tree-listing file 0444, then re-checks the sandbox's registered-cache rule.
set -euo pipefail
RC=/var/cache/autopareto/pair_a_registered_cache
K=$RC/huggingface/hub/models--kernels-community--flash-attn3
REV=e29f138fc363b396e5d2706c8a5f6fa7d36f41e0
TREE=$K/trees/$REV.json
LOG=/workspace/pod_setup_logs
TS=$(date -u +%Y%m%dT%H%M%SZ)
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { echo "GPU is busy" >&2; exit 1; }
[ ! -e "$TREE" ] || { echo "tree file already exists:"; ls -l "$TREE"; exit 1; }
listing() { (cd "$1" && find . -printf '%m %u %p\n' | sort -k3); }
listing "$RC" > "$LOG/registered_cache_before_kernelfix_$TS.txt"
listing /opt/autopareto_candidate_inputs/cache > "$LOG/candidate_inputs_cache_before_kernelfix_$TS.txt"
HOME=/root HF_HOME=$RC/huggingface HF_HUB_CACHE=$RC/huggingface/hub /opt/autopareto_runtime/bin/python -c "
from kernels import get_kernel
k = get_kernel('kernels-community/flash-attn3')
print('loaded', k.__file__)
" 2>&1 | grep -v 'Fetching'
listing "$RC" > "$LOG/registered_cache_after_kernelload_$TS.txt"
echo "--- entries created by the online load:"
diff "$LOG/registered_cache_before_kernelfix_$TS.txt" "$LOG/registered_cache_after_kernelload_$TS.txt" || true
[ -f "$TREE" ] || { echo "tree file was not created" >&2; exit 1; }
chmod 0444 "$TREE"
[ "$(cat "$K/refs/main")" = "$REV" ] || { echo "refs/main moved" >&2; exit 1; }
listing "$RC" > "$LOG/registered_cache_after_kernelfix_$TS.txt"
/opt/autopareto_runtime/bin/python - "$RC" /opt/autopareto_candidate_inputs/cache <<'PY'
import pathlib, sys
for arg in sys.argv[1:]:
    root = pathlib.Path(arg); bad = []
    st = root.stat()
    if st.st_uid != 0 or st.st_mode & 0o022 or st.st_mode & 0o005 != 0o005:
        bad.append(("root", str(root)))
    for child in root.rglob("*"):
        if child.is_symlink():
            if not child.resolve(strict=True).is_relative_to(root):
                bad.append(("escape", str(child)))
            continue
        s = child.lstat(); m = s.st_mode & 0o777
        if s.st_uid != 0 or m & 0o022:
            bad.append(("writable", str(child)))
        elif child.is_dir() and m & 0o005 != 0o005:
            bad.append(("dir-unreadable", str(child)))
        elif child.is_file() and not m & 0o004:
            bad.append(("file-unreadable", str(child)))
    print(f"sandbox cache rule on {root}: {'no violations' if not bad else bad}")
    if bad:
        sys.exit(1)
PY
listing /opt/autopareto_candidate_inputs/cache > "$LOG/candidate_inputs_cache_after_kernelfix_$TS.txt"
diff -q "$LOG/candidate_inputs_cache_before_kernelfix_$TS.txt" "$LOG/candidate_inputs_cache_after_kernelfix_$TS.txt" \
  && echo "candidate inputs cache unchanged"
echo "tree file: $(stat -c '%a %U' "$TREE") sha256 $(sha256sum "$TREE" | cut -c1-16); refs/main $(cat "$K/refs/main")"
echo "CACHE FIX DONE ($TS)"
