# 19 — Pod storage correction, verified off-pod backup, move to a network volume (2026-11-05)

## Correction
The second camera-ready pod's `/workspace` is a **pod-local 150 GB volume disk**, not a detachable network
volume; the PI confirmed this in RunPod's Volumes tab. It is mounted through MooseFS
(`mfs#eu-se-1.runpod.net`), but it belongs to the pod and is deleted when the pod is terminated. An
earlier chat message called it a network volume; that was wrong. Doc 13 §2 ("Volume disk (`/workspace`)")
describes this pod-local disk. The next pod uses a real RunPod **network volume** instead.

## Verified backup (no pod stopped; nothing changed on the pod)
Location: `/Users/Rami/Documents/dev/autopareto_pod_backups/pod2_GPU-9f05dd40_20261105/` (outside git;
`README.md` indexes it, `SHA256SUMS_ALL.txt` covers its 7,424 files).

| State | Where | Check |
|---|---|---|
| `/workspace`, except the 9.0 GB rebuildable Env A `.venv` | `workspace/` | 7,370/7,370 files identical by SHA-256, pod vs Mac; 18/18 symlinks identical |
| container `/var/lib/autopareto` | `var_lib_autopareto/` | 36/36 files identical |
| rebuildable container trees (registered cache, trusted tree, Env A/B package sets) | `pod_manifests/` (hash lists and freezes) | rebuilt from assets + repo + `uv.lock` by the scripts |
| Opus campaign state (authoritative) | Mac `runs/pair-a-claude-opus55-t-seed3/` | unchanged, sha `c821de3a…` |

**Resume readiness**, from the backup and the Mac only (`RESUME_READINESS.json`):
- The completed-prefix audit passes. Attempt 1's result `7d903b23…` and both source-registry rows
  re-hash, and attempt 2's candidate `ab63dbad…` is intact.
- Both paired checkpoints match the frozen manifest.
- 13/13 registered data files and 87/87 asset files verify. `uv.lock` matches. 122/122 trusted files
  match the worktree.
- The frozen manifest is unchanged (`c6e3ca06…`).

`scripts/pod/replacement_pod_rebuild.sh` now builds Env A with the runbook's own `env-a` stage when no
`.venv` exists, as on a fresh volume restored from this backup.

## Migration order (the current pod stays up until the last step)
1. The PI creates a network volume and a replacement A40 pod that mounts it at `/workspace`.
2. Run the 60-s CPU screen (doc 18 §6).
3. Restore `workspace/` from this Mac backup to the network volume. Verify every file against
   `pod_manifests/workspace.sha256`.
4. Rebuild the container state (`replacement_pod_rebuild.sh`), then run the predeclared validation (doc 17 §8).
5. Only after the restored data verifies on the new volume may the current pod be terminated.
