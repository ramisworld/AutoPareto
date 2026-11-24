# 21 — Fourth camera-ready A40 pod: restore, rebuild and readiness checks (2026-11-07, Phase 1)

Phase 1 of the PI's three-phase plan (restore and verify; then winner re-benchmarking; then E1). No protocol, threshold,
timer, seed or scientific setting was changed. No training ran. The search was not restarted. SSH endpoints and pod
identifiers are not recorded here.

## 1. Why a fourth pod
The RunPod balance ran out on 2026-11-06. Pod3 (`GPU-b74ec8bd…`) stopped between 03:10 and 08:42 UTC. That was after the
Opus arm completed (02:27 UTC) and after the last hash-verified pull to the Mac (03:10 UTC, 182 + 14,738 files).
The PI deployed a replacement pod; its `/workspace` volume arrived holding pod3's volume contents.

## 2. Pod
- NVIDIA A40 **`GPU-3adf8a47-baa4-0808-d8e8-3068de291340`**, driver 595.91.07 (CUDA 13.2), kernel 6.8.0-147, CA-MTL-1.
- Same image as before (Python 3.12.3, torch 2.8.0+cu128), 60 GB container disk.
- CPU: Xeon Gold 6342 (2 × 24 cores × 2 threads), governor `schedutil`, container quota 7.65 CPUs. This is the
  same hardware class as pods 2 and 3.
- `/workspace`: 140 GB NFSv4 mount (`ca-mtl-1-vast.runpod.net`, `podvolumes/…` path, the same pattern as pod3's).
  **Treated as a pod-local volume disk** (deleted on terminate) until the PI confirms otherwise in the RunPod console.
- Container disk (`/var/lib/autopareto`, `/opt`, `/var/cache`) was empty, as expected after a stop.

## 3. Host CPU screen (doc 18 §6; rule unchanged): **fails, report before GPU**
`scripts/pod/host_cpu_screen.py` (60 × 1-s chunks, no GPU), at 01:31 UTC, load 23.7, CPU pressure `some` ≈ 35%:

| Host | IQR/median (limit 0.02) | Rolling 8-chunk sets > 0.02 (limit 0.5) | Screen |
|---|---|---|---|
| Pod2 (reference) | 0.172 | 75% | report before GPU |
| Pod3 | 0.0157 | 32% | pass |
| **Pod4** | **0.0535** | **85%** | **report before GPU** |

The screen is advisory, and the acceptance test for decode timing is the bracketed measurement itself. An inference-only
host probe (§8) measured that directly. The noisy host matters for the bracketed decode benchmark. It matters little for
E1, which runs on the Phase-1 path and compares paired arms on one host.

## 4. Restore
1. **`/workspace` (no transfer needed).** Every file was hashed on the pod (without `.venv`, `pod_backups`, `mirror`)
   and compared with pod3's last verified list (`autopareto_pod_backups/pod3_GPU-b74ec8bd_live/verified_state.json`,
   sha `586bab4d…`): **14,738/14,738 identical, none missing**.
2. **Repository copy.** 122/122 trusted files were already identical to the Mac worktree (HEAD `8ce879d`). The 25
   non-trusted files committed after pod3's last sync were copied over: doc 20, the ledger, the extension, requeue and
   ledger scripts and their tests, the pull script and the E0 result logs. Their hashes are appended to
   `/workspace/pod_setup_logs/post_restore_repo_sync.log`, and all 25 verify on the pod.
3. **`/var/lib/autopareto`.**
   - Pod3's container state was taken from the Mac backup, after re-hashing the Mac copy: 182/182 match the
     verified list.
   - It was uploaded to `/workspace/pod_backups/pod3_GPU-b74ec8bd_20261106/` and verified there (182/182).
   - The two paired-baseline folders came from the volume mirror, via the rebuild script. The two campaign
     state roots (`…continuation_20261105b`, `…continuation_20261105c`) and pod3's validation folder came from the
     uploaded copy.
   - Result: **182/182 files identical to pod3's last verified state, no extra files.** This includes every
     checkpoint of attempts 3–27, with Phase 2's targets `f098e0bd…` (23), `e5687149…` (24) and `520af249…` (27).
   - Log: `/workspace/pod_setup_logs/pod4/restore_var_lib_*.log`.
   - **Upload incident.** The single rsync stream to the pod's direct TCP port fell to about 19 KB/s, while the pod's NFS
     wrote at 274 MB/s and the Mac's uplink measured 342 Mbps. It was stopped and resumed as six parallel streams
     (about 14 MB/s). No partial files remained.

## 5. Rebuild (`scripts/pod/replacement_pod_rebuild.sh`, unchanged; 01:34–01:37 UTC)
- Asset bundle 87/87; persistent Env A data cache matches the bundle (13 files).
- **Env A** restored onto the existing volume `.venv` (uv 0.9.0, CPython 3.10.18, `uv sync --frozen --inexact`, lock
  `262a4d55…`). All 10 pins hold: torch 2.9.1+cu128, triton 3.5.1, kernels 0.12.1, tiktoken 0.12.0, rustbpe 0.1.0,
  pyarrow 23.0.1, numpy 2.2.6, huggingface-hub 1.5.0, matplotlib 3.10.8, datasets 5.0.1.
- **Env B** (`/opt/autopareto_runtime`): torch 2.8.0+cu128, triton 3.4.0, kernels 0.12.1, huggingface-hub 1.30.0,
  tiktoken 0.12.0, pyarrow 23.0.1.
- Pair-A state: trusted tree 122 files; registered cache and candidate inputs rebuilt from the asset bundle.
  Paired checkpoints T `f7ca17c6…` and N `72d05e97…` restored from the mirror and verified.
- Linux sandbox tests: **193 run, OK (5 skipped)**. Pod3 ran 178; the extra tests came from campaign-time commits.
- Tokenizer retrained natively, and both registered hashes reproduce.
- E0/E1 validate-only:
  - The superseded original plan hash `806a6397…` is refused, as on pods 2 and 3.
  - **Amendment-1 plan `89f29015…`: 18 OK, 0 FAIL.** That covers 2 E0 + 24 E1 runs, 20 implementation files
    hash-matched, data and held-out shards present, E0 verdict PASS, and no E1 run directory yet.
- **Registered-cache kernel fix** (`registered_cache_kernel_tree_fix.sh`, unchanged, doc 17 §6). The online load
  created exactly the entries seen on pods 2 and 3. Tree file `8fe3d67e…` is 0444 root, `refs/main` is still
  `e29f138f`, the sandbox cache rule finds no violation, and the candidate inputs are unchanged.

## 6. Verification against frozen records
**Pod side.** The final campaign manifest (`35e50181…`) was checked file by file, with **135/135 identical**:
- 122 immutable-deployment files under `/opt/autopareto_trusted`, with no extra file;
- the 11 registered-cache closure files of flash-attn3 `e29f138f`;
- both paired baseline checkpoints.

The manifest's GPU UUID is pod3's, so the campaign's own launch gate would refuse this pod. That is correct: the
campaign is complete and is not relaunched.

**Mac side.** A read-only audit (no file changed) found the following.

| Check | Result |
|---|---|
| Controller state uses the final manifest `35e50181…` | pass |
| Positions: 27 completed, 25 not in study | pass |
| History: 27 rows, attempts 1–27 | pass |
| Source registry: 26 rows (baseline + 25 valid attempts), every source re-hashes | pass |
| Each result consistent with state, history, registry and parent lineage | pass |
| 27 attempts, 25 valid, failures [2, 9] | pass |
| Frozen-25 snapshot tar `20f04761…` and its inner state, history and view hashes (`42badb5e…`, `c50cfb5e…`, `58488ac3…`) | pass |
| Frozen-25 counts: 23 valid, failures [2, 9] | pass |
| Winner: attempt 23 (`408228f4…`, BPB 1.09620) in both the frozen-25 and the 27-attempt pool | pass |
| Frozen documents and sources: confirmation pre-registration `adf8ce71…`, G4 decision `f5a59d52…`, E1X pre-registration `a1f6365f…`, G4 source `b612b6e8…`, baseline source `b2fb8788…` | pass |
| Baseline registration present; the manifest's registered source is `b2fb8788…` | pass |
| Checkpoint of every valid attempt on the Mac | **24/25: attempt 1 missing** (§9) |

## 7. GPU sanity checks (no training)
- Both environments see the A40 (44.4 GiB, sm86) under CUDA 12.8.
- flash-attn3 loads offline from the registered cache (Env B) and from `~/.cache` (Env A). Its output matches SDPA to
  a maximum absolute error of 0.008.
- bf16 matmul after warm-up: about 110 TFLOP/s in both environments.
- SM clock 1,740 MHz, no thermal throttling.

## 8. Inference-only host probe (non-study; no training)
The same three bracketed repeats as pod3's predeclared validation, with the unchanged command of
`bracketed_inference_repeats.sh`:
- candidate: pod3's T-validation checkpoint (baseline source);
- bracket: paired T checkpoint `f7ca17c6…`;
- output: a new folder, `/var/lib/autopareto/nonstudy_pod4_host_probe_*`.

The inputs are identical to pod3's, so this is a direct host comparison.

Run 01:56–02:09 UTC, load 14–21:

| Repeat | Pod4: normalized drift (limit 0.02) | Pod4 candidate tok/s | Pod4 normalized throughput | Pod3 drift, same inputs |
|---|---|---|---|---|
| 1 | 0.0195, stable | 139.4 | 0.991 | 0.0073 |
| 2 | 0.0183, stable | 143.3 | 1.006 | 0.0176 |
| 3 | **0.0428, unstable** | 146.2 | 1.010 | 0.0035 |

- **Doc 17 §8 decode-timing acceptance rule (all three stable): REJECT on this host.** Pod3 passed it; pod2 failed it.
- The bracket still normalizes well. Candidate and bracket are the same recipe, and the normalized throughput is within
  1% of 1.0 in every repeat, so speed ratios measured here look unbiased. Stability verdicts, however, would often come
  out `speed_uncertain`.
- This affects Phase 2's decode and end-to-end timing, not BPB, correctness or E1. How to proceed is the PI's decision.

## 9. Missing artifacts (none needed for Phase 1)
- **Codex G4 attempt-14 checkpoint** (`63bf0276…`, discovery on the September Codex pods): not on the Mac (305 large
  `.pt` files hashed) or on the pod. Only its result record and frozen source (`b612b6e8…`) survive. Phase 2 can
  include G4 only by retraining its frozen source on this pod, which is GPU training and needs PI approval.
- **Opus attempt 1 checkpoint** (first camera-ready pod, `GPU-4702993a…`, container disk lost on 2026-11-04): the result
  record is intact; the checkpoint is gone. It is not a Phase 2 target.

## 10. Off-pod protection
- The pod3 pull loop, which had been failing since its pod stopped, was stopped.
- `scripts/pod/pull_pod_state_to_mac.py`, unchanged, now targets this pod. Destination:
  `/Users/Rami/Documents/dev/autopareto_pod_backups/pod4_GPU-3adf8a47_live/`, seeded with an APFS clone of the pod3
  copy.
- First cycle (01:54 UTC), with every file hashed on both sides: **182/182 `/var/lib/autopareto` and 14,774/14,774
  `/workspace` files verified**.
- It then runs every 15 minutes. At 02:10 UTC it verified 189 + 14,776 files, including the probe outputs. The E1
  output root (`/workspace/autopareto-camera-ready/results/camera_ready_2026/amendment1`) is inside the pulled tree.

## 11. Readiness (end of Phase 1)
| Check | Verdict |
|---|---|
| GPU, driver, CUDA, both pinned environments, flash-attn3 | PASS |
| `/workspace` identical to pod3's last verified state | PASS |
| `/var/lib/autopareto` identical to pod3's last verified state | PASS |
| Trusted tree, registered cache and paired checkpoints against the final campaign manifest | PASS |
| 27 attempt records, frozen-25 snapshot, protocols, baseline registration | PASS |
| Sandbox tests; tokenizer; E0/E1 amendment-1 validate-only | PASS |
| Off-pod backup (15-min pull, hash-verified) | PASS |
| Host CPU screen | FAIL (advisory) |
| Decode-timing stability (doc 17 §8 rule, inference-only probe) | FAIL (1 of 3 repeats unstable) |
| Checkpoints for Phase 2: Opus 23, 24 and 27, plus baseline | PASS |
| Checkpoint for Phase 2: Codex G4 | MISSING (needs approved retraining) |
| E1 training environment | PASS. No training was run in Phase 1. The plan's own estimate is 6.44 A40-h, plus 0.15–0.35 h per retry. |
