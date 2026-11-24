# 16 — Opus campaign incident: attempt 2 (global position 4), infrastructure failure before training

Status: **campaign paused, not reconciled.** The live campaign state is untouched. The fix and the
reconciliation below await PI approval.

## What happened
- **11:03:16 UTC.** Attempt 1 completed: validation BPB 1.1791, retained, inference stable.
- **Attempt 2.** Both agent calls completed. The candidate source is `ab63dbad…`, its parent is attempt 1, and the metadata is `0b6c6edc…`. The candidate and metadata were transferred to the pod, and the durable launch was requested.
- **11:07:00 UTC.** The remote worker started. **2 s later** the sandbox's pre-launch permission check refused to start the candidate:
  `ProcessSandboxUnavailableError: registered cache file is not readable: …/hub/models--kernels-community--flash-attn3/trees/e29f138fc363b396e5d2706c8a5f6fa7d36f41e0.json`.
  The launcher classified the failure origin as unknown. The durable receipt is `FAILED` (`pair-a-004-ccb492e3…`). The controller marked the position `reconciliation_required` and exited, and the supervisor alerted (`paused_reconciliation_required`).
- **Nothing ran.** No candidate code ran, no training happened and no result was produced. `bridge_results/` holds only position 2. Attempt 2 was **not** consumed.

## Cause
1. **The trusted harness runs part of its work online.** `remote_runner.py` gives the trusted training
   launcher an allowlisted environment. It sets `HF_HOME` to the registered cache but does not pass
   `HF_HUB_OFFLINE` (lines 496–539). When the trusted post-training stage of attempt 1 loaded the
   flash-attn3 kernel (10:57:18 UTC), huggingface_hub 1.30.0 took its online path and wrote three things:
   - `trees/e29f138f….json`, a tree-listing cache created through `mkstemp`, hence mode **0600**;
   - a refreshed `refs/main` (unchanged content, `e29f138f`, mode 0444);
   - `.agent_harnesses.json` (0644).

   The protected inference subprocess itself does set `HF_HUB_OFFLINE=1` (`train_launcher.py` lines 499–505).
2. **The sandbox fails closed.** It requires every registered-cache file to be root-owned, not group- or
   other-writable, and other-readable (`candidate_sandbox.py` lines 840–860). The new 0600 file
   breaks that rule for the next launch.
3. **Why attempt 1 was not affected.** The file did not exist when attempt 1 launched. Every fresh pod
   would hit this at its second attempt. The Codex campaign ran the same frozen `remote_runner.py`,
   so its trusted stage was online too. How its cache avoided the 0600 file is not recoverable,
   because that pod is gone.

## Validity
- **Kernel identity is unchanged.** The Hub's `main` for `kernels-community/flash-attn3` is `e29f138f`
  (last modified 2026-09-01), `refs/main` still reads `e29f138f`, and `e29f138f` is the only snapshot.
  The kernel used is the same revision as Codex and E0/E1.
- **Attempt 1's result is valid.**
- **Disclosure for the paper.** The trusted (non-candidate) evaluation stage resolves the kernel online
  in both agent arms. That would become a kernel-identity risk only if the Hub's `main` moved; a new
  snapshot would also create a new 0600 tree file and fail the next launch closed.

## Proposed reconciliation (needs approval)
1. **Pod.** `chmod 0444` that single tree file, so it is root-owned, read-only and readable, like the rest
   of the cache. huggingface_hub writes the tree cache only when it is missing, and rewrites
   `.agent_harnesses.json` in place (keeping 0644), so the fix holds for this revision.
   - A read-only scan, which excludes symlinks as the sandbox does, finds no other violation.
2. **Requeue tool.** Retry attempt 2 with the **same** persisted candidate and payload under a new
   invocation ID: `amend_control_plane_manifest.py --retry-existing-execution`. This is the
   documented path for a verified post-remote harness failure; the Codex campaign used it (e.g. its position 14, `position-014-adapterfix-…`).
   - The tool predated single-arm studies: it rejected the `not_in_study` N positions. It now skips them only when the state declares `active_contracts`, and two-arm states are checked as before.
   - The tool is not part of the frozen deployment. Tests pass 6 of 6, including the new single-arm case and the unchanged two-arm rejection.
   - No new agent calls are made, and the Scientist's proposal is preserved.
3. **Dry run on a copy of the state.**
   - No harness file changed hash.
   - The amended manifest passes the controller's `_verify_final_manifest`.
   - `_initialize_state` accepts it for resume.
   - Position 4 becomes `running` / `launch_requested`, with new invocation `pair-a-004-retry-…` linked to the failed one.
   - Counts: 1 completed, 1 running, 23 pending, 25 not in study.
   - The pod payload for position 4 still matches `ab63dbad…` and `0b6c6edc…`.
4. **Resume.** Run the launch command with the amended manifest's path and sha.
   - The failed invocation stays on record as a non-consuming infrastructure event, with its receipt.
   - The manifest records `scientific_contract_changed: false`.
