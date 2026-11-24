# 20 — Third camera-ready A40 pod: host screen, verified restore, validation ACCEPT, campaign resumed (2026-11-05)

No protocol, threshold, timer or scientific setting was changed. SSH endpoints are not recorded here.
The Opus campaign resumed at attempt 2 (global position 4) at 04:41 UTC with the same candidate and no new agent call.

## 1. Pod
- NVIDIA A40 **`GPU-b74ec8bd-38c7-1555-5cb2-1244c55954d8`**, driver 595.91.07, kernel 6.8.0-147, datacenter
  CA-MTL-1, same image (`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`), 60 GB container disk.
- CPU: Xeon Gold 6342 (2 × 24 cores × 2 threads), governor `schedutil`, container quota 7.65 CPUs — the
  same hardware class as the second pod.
- `/workspace`: 150 GB NFSv4 mount (`ca-mtl-1-vast.runpod.net`). **The PI confirmed in the RunPod console
  that it is a pod-local volume disk, not a network volume**, so it is deleted when the pod is terminated.
- PI decision: keep this pod, because its host passed the CPU screen (§2), and protect its state with a
  hash-verified pull to the Mac every 15 minutes (§5) rather than redeploy onto a network volume and
  risk a noisier host.

## 2. Host CPU screen (doc 18 §6; rule fixed before use)
`scripts/pod/host_cpu_screen.py`, 60 × 1-s chunks of fixed pure-Python work, no GPU:

| Host | IQR/median (limit 0.02) | Rolling 8-chunk sets > 0.02 (limit 0.5) | Screen |
|---|---|---|---|
| Second pod (reference) | 0.172 | 75% | report before GPU |
| **This pod** (03:15 UTC, load 9.1) | **0.0157** | **32%** | **pass** |

Rate CV is 6.2% here because of a few outlier chunks; the screen is advisory and the validation
acceptance rule (doc 17 §8) is the real test. Report: Mac `runs/replacement_a40_pod3_20261105/`.

## 3. Restore from the verified off-pod backup (doc 19)
- Before the upload, no file on the second pod had changed since the backup (none newer than 01:30 UTC).
- `workspace/` of the Mac backup → this pod's `/workspace` (rsync, 03:16–03:35 UTC). The backup folder's own
  files (the `/var/lib/autopareto` copy, hash lists, receipts) → `/workspace/pod_backups/pod2_GPU-9f05dd40_20261105/`.
- `scripts/pod/verify_workspace_restore.py` (tested first on a clean and a tampered APFS clone):
  **7,370/7,370 files identical** to the hash list computed on the second pod, none missing or extra;
  **18/18 symlinks identical**; **54/54 backup-folder files identical**; the Mac-side and pod-side lists agree.
- The second pod's research state therefore exists, verified, on the Mac and on this pod.

## 4. Rebuild
- Four repository files committed after the backup (doc 19, the rebuild script, the timing-diagnostic
  script, the restore verifier) were copied from the Mac worktree; their hashes are logged in
  `/workspace/pod_setup_logs/post_restore_repo_sync.log`.
- **NFS fix.** The rebuild script's `cp -a` of the HF kernel cache from `/workspace` failed: coreutils
  copies the NFSv4 ACL along with the mode, and the container disk refuses it (`cp --preserve=mode` fails
  the same way). The old MooseFS volume had no such ACL. The script now copies with `rsync -rlpt` (tested
  on a scratch copy: contents, modes and link targets identical); the byte-for-byte check after it is
  unchanged. The two failed runs created only container-disk files under `/root/.cache`, which were
  removed before the rerun.
- Result (03:51–03:55 UTC): asset bundle 87/87; Env A built by the runbook's own stage, all 10 pins; Env B
  pins; trusted tree 122 files, identical to the Mac worktree; registered cache and candidate inputs rebuilt;
  paired checkpoints T `f7ca17c6…` and N `72d05e97…` verified; Linux sandbox tests 178 run, OK (4 skipped);
  tokenizer retrained natively, both hashes reproduce; amendment-1 validate-only 18 OK, 0 FAIL.

## 5. Off-pod protection while the campaign runs
`scripts/pod/pull_pod_state_to_mac.py`, every 15 minutes, on the Mac:
- pulls `/var/lib/autopareto` and `/workspace` (without the rebuildable `.venv`, the backup folder and the
  pod-side mirror) into `/Users/Rami/Documents/dev/autopareto_pod_backups/pod3_GPU-b74ec8bd_live/`;
- only adds or updates files on the Mac, never deletes;
- hashes every new or changed file on both sides (pod side at nice 19 / idle I/O, to stay out of the
  CPU-bound decode measurement), re-pulls any mismatch by checksum, and reports anything unverified in
  `status.json` with a macOS notification.

It replaces the pod-side mirror loop of doc 17 §7 step 5: the copy now leaves the pod.

First pull (03:55 UTC, before any GPU work): 14 `/var/lib/autopareto` files and 7,344 `/workspace` files verified on
both sides. A Mac-side file corrupted with its size and modification time kept (which rsync's quick check
misses) was caught, re-pulled by checksum and re-verified. Incremental cycles take about 20 s.

## 6. Predeclared validation (doc 17 §8 rule, scripts unchanged; 03:58–04:30 UTC): **ACCEPT**
State `/var/lib/autopareto/pair_a_claude_opus55_a40_validation_20261105_pod3`; G2 script `15975b14…`. Evidence on the
Mac: `runs/replacement_a40_pod3_20261105/` (`SHA256SUMS`; the 17 state files are identical to the pod's).

| Run | BPB | Tokens / updates | Inference |
|---|---|---|---|
| T smoke (300 s) | **1.19685** (\|Δ\| 0.0030 vs 1.19989: halt rule passes) | 87,031,808 / 166 in 301.8 s | stable |
| N smoke (N\*) | 1.19908 | 85,458,944 / 163 | stable |

| Bracketed repeat (T validation checkpoint vs paired `f7ca17c6…`) | Normalized drift (limit 0.02) | Candidate tok/s |
|---|---|---|
| 1 | 0.0073, stable | 147.1 |
| 2 | 0.0176, stable | 142.5 |
| 3 | 0.0035, stable | 146.8 |

- Decode speed (142–151 tok/s) is close to the Codex comparator host's median of 150.3 (doc 18 §3).
- **Disclosure: fixed-time update count.** This A40 trains 166 updates in the 300-s window, against 164 on the
  Codex comparator's validation and 162 on the first Opus pod (attempt 1): +1.2% and +2.5% tokens (doc 18 §4).
  The baseline's own T smoke scores 1.19685 here, against 1.19910 on the comparator's validation and 1.20110 on the
  first pod. That is inside the predeclared halt rule (|Δ| ≤ 0.005 from the registered 1.19989), but a gap of
  about 0.002 BPB is the size of a small candidate improvement. Candidates are scored against the registered
  baseline, not a same-host one (the paired baseline only normalizes decode speed), so this host slightly favours
  attempts 2–25. The paper should state it with the two-A40 note and report a sensitivity check against this
  host's baseline (1.19685).

## 7. Cache fix, continuation, launch
1. **Cache fix** (`scripts/pod/registered_cache_kernel_tree_fix.sh`, 04:31 UTC): the online kernel load created the
   same entries as the scratch test (doc 17 §6); `trees/e29f138f….json` is now 0444 root; `refs/main` unchanged; the
   sandbox cache rule finds no violation on the registered cache or the candidate inputs.
2. **Calibration record** `runs/pair-a-continuation-preparation-20261104/provenance/reused_calibration_20261105.json`
   (`53d0cfd0…`, read-only): the frozen N\* reused as on 2026-09-16, with the validation evidence above recorded by hash.
3. **First preparation, not used** (`runs/pair-a-claude-opus55-t-seed3-new-a40-continuation-20261105/`, manifest
   `6ab2a11a…`). Its controller checks and launch gates passed, but its launch stopped within seconds, before any
   agent call or remote action: the controller resumes a re-armed position only once `arm_continuation_payload` has
   transferred and verified its payload, and that function picked the first schedule position not `completed`,
   which in a single-arm study is the inactive position 1. Its state is unchanged since preparation and kept as is.
4. **Second single-arm fix** (commit 79e0715): `arm_continuation_payload` now skips `not_in_study` positions like the
   rest of the module. The single-arm test now also arms and re-arms; it fails on the previous code. Mac: 9/9
   continuation tests; the full Mac suite gives the same result before and after (its 8 failures need torch or
   Docker). Pod (Linux, Env B): 178 tests OK. Deployed to the pod's repo copy and trusted tree (log:
   `post_restore_repo_sync.log`). `continuation.py` is now `03425d40…` (frozen manifest: `a3046502…`; first
   continuation: `efdee709…`); it is still the only immutable-deployment change.
5. **Continuation** `runs/pair-a-claude-opus55-t-seed3-new-a40-continuation-20261105b/`, manifest
   `claude_t_continuation_manifest_20261105b.json` **`b9abe985…`** (read-only), remote state
   `/var/lib/autopareto/pair_a_claude_opus55_t_seed3_continuation_20261105b`. The live source state is unchanged
   (`c821de3a…`).
   - Controller: `_verify_final_manifest` and `_initialize_state` accept it; counts 1 completed, 1 running, 23
     pending, 25 not in study; scientist, contracts, inference and Pareto settings identical to the frozen manifest.
   - Launch gates over the proxy, in the gate script's order (its fresh-state-root check omitted, as a continuation
     is not fresh): manifest, transport and state config, 122 trusted files, GPU `GPU-b74ec8bd…`, 15 absolute files
     (registered cache and both paired checkpoints), baseline record. Report:
     `runs/replacement_a40_pod3_20261105/launch_gates_continuation_20261105b.json`.
   - **Arming.** The first attempt (04:39 UTC) lost the SSH session before the result frame: the files had been
     written and verified on the pod, but the controller could not confirm it and left the state unarmed. Eight
     non-study probes of the same transport, including transfer-sized input, all succeeded, so it was treated as a
     transient proxy drop. The retry armed position 4 (`launch_requested`); the pod confirmed `train.py` `ab63dbad…`
     and `registered_metadata.json` `e9400448…`. A pre-arm copy of the state is kept.
6. **Launch** at 04:41 UTC under the supervisor (`…-20261105b.supervisor`), with the command of doc 15 §3 and only
   the state root, manifest, pod host, remote state root and supervisor directory changed.

**Disclosure for the paper (replaces doc 17 §7's):** the Claude arm spans two A40s. Attempt 1 ran on
`GPU-4702993a…`; attempts 2–25 run on `GPU-b74ec8bd…`. Attempt 2 is the original Scientist proposal from
2026-11-04, re-armed without a new agent call.

## 8. Attempt 2 outcome (global position 4, 04:41–04:48 UTC): candidate-caused failure, as the frozen rule requires
- Candidate `ab63dbad…` (attempt 1's code with DEPTH 8 → 6 at fixed width 512) ran on this A40: 423 timed updates
  at about 0.71 s each.
- The candidate keeps its own 300-s fallback `break` on a clock it starts just before its loop and reads after
  each update. That clock leads the protected one, which starts just before the first scientific forward
  pass. Its loop broke at its own 300.1 s while the protected elapsed time was still under 300 s, so the
  harness recorded `CandidateExitedBeforeStop` ("candidate worker exited before protected completion").
- Outcome `scientific_failure`, `failure_class: candidate`, not Pareto-eligible. This is the Claude-backend
  amendment of `candidate_process.py` (doc 18 §1), frozen before the campaign: a body that returns before the
  protected stop is a candidate failure. In the Codex arm the same event re-ran the body and produced an
  untrained checkpoint (BPB 3.2159), so it also counted against the candidate. Nothing was changed.
- Attempt 1 has the same fallback break and stopped on the right side of the boundary. Whether it fires first
  depends on where the last update lands against a clock offset of milliseconds. It is the candidate's design
  risk, now visible to the Scientist in its history.
- The controller moved on to attempt 3 (global position 5) with no reconciliation needed. Seat usage at
  04:50 UTC was 30% (7-day window), with no overage.

## 9. Attempt 3 is the timer-fixed retry of attempt 2; transfer drop, requeue, resume (04:50–06:28 UTC)
**Attempt 3's proposal.** Opus 5.5 proposed it itself, under the frozen protocol (Scientist calls at 04:47 and
04:50 UTC). It chose parent `attempt-01`, the parent attempt 2 also used. From attempt 2's failure record and
diff, it diagnosed the clock race of §8: the candidate's fallback break at 300 s on a clock that leads the
protected one. It kept attempt 2's change exactly (DEPTH 8 → 6 at fixed MODEL_DIM 512) and moved the fallback
break to 300 s + `STOP_GUARD_S` = 10 s, so the protected stop always ends training. Its candidate `d37c50f2…`
differs from attempt 2's `ab63dbad…` only in that guard and its comment, plus a whitespace change and an
equivalent assert (`MODEL_DIM` for `model_dim`, the same value). This is the retry the PI asked for at
06:15 UTC ("retry experiment 2 with the same six-layer architecture, fixing only the conflicting timer"),
produced inside the protocol, with no human edit to agent code.

**PI request and how it was met.**
- Attempt 2's recorded outcome is not changed. The rule against altering results applies, and no BPB,
  speed or memory exists for it.
- Its failure stays in the record and in the Scientist's view; the papers should describe it as a
  millisecond clock race at the stop boundary, not a property of the six-layer model.
- Attempt 3 runs the six-layer model with the timer fixed, and the standard pipeline evaluates it.
- If it is non-dominated, it enters the parent pool for attempt 4 onward. The Scientist then chooses among
  the pool as usual.
- Open PI decision: whether to add one attempt at the end so the arm has 25 valid results. The Codex arm's
  budget was 25 attempts, failures included, so an extra attempt would be an asymmetry to disclose.

**Transfer drop.**
- At 04:51:17 UTC the payload transfer for attempt 3 (global position 5) lost the PTY session over the
  RunPod proxy: `script: write master: Input/output error`.
- The controller paused the position as `reconciliation_required` (phase `payload_transfer`, dispatch
  `prepared`). It sat paused until 06:17 UTC.
- That is the second such drop in four payload transfers on this pod (the first was the arming at 04:39,
  §7). Both happened while streaming the payload in; launches, polls and result fetches have not dropped.
- Nothing reached the pod. Evidence: `runs/replacement_a40_pod3_20261105/position005_undelivered_payload_absence_20261105.json`
  (`dd8266c9…`, 06:27 UTC). It records no payload directory, no invocation receipt, no matching process and
  an idle GPU.

**Requeue.**
- `scripts/requeue_undelivered_payload.py` (new) puts back a position whose transfer failed before
  delivery. It is fail-closed:
  - phase `payload_transfer`, dispatch `prepared`, no remote payload directory or result;
  - every earlier position completed and every later one pending;
  - candidate and metadata hashes match;
  - the absence evidence names the same run and invocation and proves nothing arrived.
- It keeps a read-only copy of the state (`controller_state.json.pre_requeue_position_005_20261105T062726Z.json`,
  `57abc907…`). It moves the failure into the position's `requeued_transport_failures` and sets the position
  back to `running`.
- The candidate and metadata are unchanged, and no agent call is made.
- `arm_continuation_payload` then transferred and verified the payload on the pod: `train.py` `d37c50f2…`,
  `registered_metadata.json` `1f5b26a4…`, `launch_requested`.
- The supervisor relaunched at 06:28 UTC with the command, manifest (`b9abe985…`) and state root of §7.
- Test: `scripts/tests/test_requeue_undelivered_payload.py` runs the real controller loop offline. A transfer for
  attempt 4 fails; the tool refuses four kinds of bad evidence and a second requeue; the controller refuses to
  resume an unarmed requeue and leaves the state alone. After arming, the campaign completes 25 attempts, each
  trained once. Attempt 4 trains the original candidate, and no new Scientist call is made for it. The supervisor
  suite (8 tests) still passes with the harness knob added.

**Verification of parent selection and Scientist context (PI question, 06:15 UTC).**
- Every attempt makes two Scientist calls.
  - `select_parent` must name one `parent_experiment_id` from the campaign view.
  - The controller checks it against the source registry: same arm, parent-eligible, source hash.
  - `edit_child` then gets that parent's exact `train.py` and must implement the selected plan.
- So a child is not taken from the latest attempt by default.
- The parent pool is the baseline plus every non-dominated, quality-valid result (this includes results whose
  speed was uncertain). Dominated and failed results drop out, but their lineage stays.
- One gap: the controller's registry check uses eligibility at commit time, so a parent dominated later would
  still be accepted. The view and prompt steer against it.
- Replay of every selection against the pool at that time:
  - Claude arm: 3 of 3 inside the pool. Attempt 3 skipped the failed latest attempt 2 and branched from
    attempt 1.
  - Codex state `pair-a-official-20260914-new-a40-continuation-20260916-r2`: 22 of 22 inside the pool, 19 from
    the latest member, 3 branches to an older one (attempts 8, 17, 21).
- Both calls get the same `campaign_view.json`:
  - lineage of every attempt (parent, source hash, status);
  - for every completed or failed attempt: hypothesis, intended change, category, the exact diff against its
    parent, validation BPB, decode speed with intervals and drift, memory, training statistics and the Pareto
    decision;
  - for failures, the class and error message;
  - the frontier, the parent pool and a failure list.
- By design, raw logs, checkpoints and sealed downstream metrics are not shown.
- The attempt-3 tool log shows both calls reading `campaign_view.json`. The parent-selection reasoning reviews all
  three records and cites attempt 2's failure reason.

**Attempt 3 result (06:28–06:39 UTC).**
- 424 updates in 300.394 s of protected time, so the timer fix worked and the protected stop ended training.
- BPB **1.16222**, against 1.17909 for attempt 1 and 1.19989 for the registered baseline.
- Decode was stable: normalized **1.327** [1.316, 1.338], 197.4 tok/s against the bracketed baseline's 148.7.
- Peak inference memory **303,241,728 B** (attempt 1: 353,574,400), KV cache 14,143,488 B.
- Pareto decision `retained`. Attempt 3 dominates attempt 1, which leaves the frontier and the parent pool.
  Attempt 3 is now the only parent for attempt 4.
- The host disclosure of §6 applies: about +2.5% tokens against the first pod.

## 10. PI amendment: one extra attempt (campaign attempt 26) after attempt 25 (approved 2026-11-05)
**Decision.** The PI asked "Add a 26th attempt at the end so the Opus arm has 25 valid results?" and answered
"Yes, add one attempt". The reason is attempt 2's CandidateExitedBeforeStop, which produced no measurement.

**Asymmetry to disclose.** The Codex arm ran 25 attempts with no extension. The paper should report:
- the frozen-budget result (attempts 1–25, as scheduled);
- the extended result (attempts 1–26), with this amendment.

**How it runs, with nothing frozen changed.**
- `scripts/run_extension_attempt.py` (new) adds the attempt as global position 51, outside the frozen
  50-position schedule.
- No file under `study/` changes. The schedule, program, contract, manifest (`b9abe985…`), trusted tree and
  Scientist configuration also stay the same.
- It drives the controller's own position functions: `_run_position`, `_resume_running_position` and the commit.
- It runs the launch gates in `run_campaign`'s order.
- It refuses to run until all 25 scheduled attempts are complete.

**`prepare`, after attempt 25:**
- keeps a read-only tar snapshot of the whole 25-attempt state, the basis of the frozen-budget analysis;
- keeps a read-only copy of the terminal campaign view;
- changes only the view's `attempt_number` (25 → 26), which the controller requires to match the position;
- adds pending position 51;
- writes a read-only amendment record, `provenance/extension_attempt26_amendment.json`. It holds the approval,
  the outcome counts, the hashes of everything above and the driver's own hash.

**Scientist prompt.** Each of the two Scientist calls for attempt 26 gets the same prompt as every other attempt,
plus one disclosed note. `program.md` says "Run exactly 25 attempts" and "Stop after attempt 25", so the note
says the PI authorized one extra attempt after attempt 2's CandidateExitedBeforeStop, that the orchestration
layer owns numbering, that everything else in `program.md` applies, and that attempt 26 is the last. The note is
added in the driver's process only.

**Transfer drop at attempt 26.** Run `scripts/requeue_undelivered_payload.py`, then `run_extension_attempt.py arm`,
then `run` again. `continuation.arm_continuation_payload` only covers scheduled positions.

**Test.** `scripts/tests/test_run_extension_attempt.py` runs the real controller loop offline:
- `prepare` refuses while attempt 25 is paused by a transfer drop, and refuses a second time;
- the snapshot holds the exact pre-extension state, and only `attempt_number` changes in the view;
- attempt 26 survives a transfer drop through requeue and arm;
- 26 attempts each train once, with 2 Scientist calls each;
- the note is in both attempt-26 prompts and in no other prompt (a mutation that drops the note fails the test);
- the history ends with 26 records, and a rerun reports `already_completed`.

All script tests pass (10).

**Not yet executed.** It runs after attempt 25, as one more step of the approved campaign:
1. `prepare`;
2. `run` under the supervisor, with the campaign's own `run` arguments.

## 11. Attempt 8 (global position 16): protected-harness defect in the FP32 correctness adjudication (08:02 UTC). Campaign paused; nothing changed
**Attempts 4–7.**
- Attempt 4: parent 3, depth. BPB 1.12731, decode 1.931, 255 MB.
- Attempt 5: parent 4, depth. BPB 1.11766, decode 2.486, 232 MB.
- Attempt 6: parent 5, depth. BPB 1.13912, decode 3.644, 202 MB. Not dominated: the frontier now has two points.
- Attempt 7: parent **5**, not the latest attempt. Batch 2^18 → 2^17. BPB 1.09953, decode 2.474, 232 MB. It
  dominates attempt 5; the frontier is now {6, 7}.

**Attempt 8.**
- Parent attempt 7, category `inference_adapter_memory`. The training path is byte-identical to attempt 7's.
- The candidate adds `build_inference_adapter`, a cached-inference adapter that uses SDPA with an explicit
  sliding-window mask instead of FlashAttention 3.
- Training completed: 1,475 updates, about 300 s. The checkpoint is on the pod, under the attempt directory in
  `/opt/autopareto-candidate-attempts/`.
- The bracketed inference then failed with `FlashAttention only supports fp16, bf16, and fp8_e4m3 data type`.
  The remote runner classified the failure origin as `unknown`, and the controller paused the position as
  `RemoteInfrastructureFailure`. The position is not consumed.

**Root cause (code).** `inference_benchmark.precision_aware_correctness` first checks the cache in BF16.
- If that check misses tolerance, it adjudicates in FP32: it reloads the model with `.float()` and calls
  `protected/inference_adapter.check_cached_correctness`.
- For a candidate-supplied adapter, that function uses the candidate's own `GPT.forward` as the reference. The
  canonical adapter uses its own SDPA prefill.
- That forward still calls `fa3.flash_attn_func`, inherited from the baseline, so FP32 Q/K/V reach FlashAttention
  3, which rejects them. The `sdpa_kernel(MATH)` context does not cover FA3.
- `inference_model_host.py` already avoids this exact problem in its `reference` operation (see its comment). The
  adjudication path does not.
- So any candidate that supplies its own adapter and misses the BF16 tolerance cannot be adjudicated, whether its
  adapter is right or wrong.
- Comparing an SDPA adapter against an FA3 forward in BF16 also measures backend round-off. The docstring of
  `check_cached_correctness` names this effect for the canonical path.

**Prior results unaffected.**
- No completed result in either arm used this path.
- In the Codex state (`pair-a-official-20260914-new-a40-continuation-20260916-r2`), 6 results were FP32-adjudicated
  (T attempts 9, 16, 19, 20, 22 and N attempt 7). All used the protected canonical adapter, and none supplied
  `build_inference_adapter`.
- No Opus result before attempt 8 supplied an adapter.

**Status.** Campaign paused at 08:02 UTC. Nothing was changed. PI decision requested between:
1. a narrow harness fix and a rerun of attempt 8 with the same candidate;
2. charging the failure to the candidate;
3. a new attempt 8 without a fix.

## 12. PI decision (09:00 UTC): fix the FP32 adjudication defect, test it on attempt 8's checkpoint, rerun the exact attempt 8
The PI's instruction:
- verify the attention backend first;
- test the fix independently on attempt 8's saved checkpoint, keeping attention semantics, tolerances and protocol;
- rerun the same candidate, hypothesis, parent and code, with no Scientist call;
- keep all logs, document the fix, and verify BPB, decode speed and memory;
- if all succeeds, update the frontier normally and resume.

**Attention backends on this A40.** Non-study probe, `runs/replacement_a40_pod3_20261105/attempt8_fp32_fix/backend_probe.json`:
- The device is `NVIDIA A40`, compute capability (8, 6), torch 2.8.0+cu128.
- The candidates' own rule (`varunneal/flash-attention-3` only on (9, 0)) selects the registered
  `kernels-community/flash-attn3` build. Its extension `_flash_attn3_5630d32.abi3.so` holds 102 sm_80 and 302 sm_90a
  cubins, and no PTX.
- **FlashAttention-3 is in use, but through its Ampere path:** the kernel that runs is
  `flash::FlashAttnFwdSm80` (`enable_sm80_to_sm89`). The Hopper kernels cannot run here. That path accepts
  fp16/bf16/fp8 only, and FP32 reproduces the error in isolation.
- So this A40 runs the following attention code:

| Path | Kernel |
|---|---|
| Candidate `GPT.forward`: training, validation BPB, BF16 reference | FA3 `FlashAttnFwdSm80` |
| SDPA with an explicit mask in BF16 (attempt 8's prefill; the canonical adapter, by code) | PyTorch memory-efficient attention (`fmha_cutlassF … Sm80`) |
| SDPA without a mask in BF16 (decode-like) | PyTorch FlashAttention-2 split-KV (`pytorch_flash::flash_fwd_splitkv_kernel`) |
| SDPA in FP32 under `SDPBackend.MATH` | `ampere_sgemm` + softmax |

- The BF16 cache check therefore compares different kernels, and round-off can exceed its tolerance. FP32
  adjudication exists for exactly that case. The manifest already allows it:
  `inference.correctness.fp32_sdpa_adjudication_allowed: true`.

**Fix** (`study/contracts_2026/inference_benchmark.py`, `f05d47a1…` → `c9b632e6…`, commit f2873fc).
- It applies during FP32 adjudication, and only for a candidate-supplied adapter.
- Every FlashAttention interface in the candidate module (here, `fa3`) runs FP32 inputs through
  `fp32_flash_attention_reference`. That is SDPA with FlashAttention's documented semantics:
  - layout (B, T, H, D);
  - default scale 1/√D;
  - GQA head grouping;
  - causal mask and sliding window `[i + Tk − Tq − left, i + Tk − Tq + right]`, aligned bottom-right;
  - zeros for a query with no visible key.
- Half-precision calls still go to the kernel. FP32 options that change the math (softcap, qv, descale,
  attention_chunk, return_attn_probs) are refused.
- Interfaces are restored afterwards, including on error. The canonical adapter's path is untouched.
- The result records `fp32_reference_attention` (mode, replaced globals, call count).
- Unchanged: the BF16 stage, all tolerances, prompts, prefixes, decode steps, measurement and Pareto rules.
- Tests: `study/contracts_2026/tests/test_fp32_reference_attention.py`, 9 tests.
  - The stand-in matches a row-by-row restatement of FlashAttention's docstring at float64 precision (1e-10).
  - It matches the registered FA3 kernel in BF16 on the candidate shapes (T 2048/1024, 4 heads of 128,
    windows 1024/2048/512): maximum difference < 2e-2, mean < 2e-3.
  - The swap is scoped and restored.
  - The full pod suite passes: 191 tests, 4 skipped.

**Independent check on attempt 8's saved checkpoint** (non-study, read-only copies, hashes in `SHA256SUMS`):
- Original evaluator: fails exactly at `precision_aware_correctness` → `check_cached_correctness` (line 424,
  `model(common_tokens)`) → FA3, with the same error.
- Fixed evaluator, full bracketed inference run, completed 09:14 UTC:
  - BF16 check: missed tolerance. Mean logit error 0.0351 against a limit of 0.03 at the 64-token prefix, and one
    greedy-token mismatch at 256.
  - **FP32 adjudication: passed.** Max error 1.57e-5 (limit 1e-4), mean 2.4e-6 (limit 1e-5), greedy tokens
    match, 192 FP32 attention calls through the stand-in, only `fa3` replaced. Attempt 8's adapter is correct.
  - Decode: stable, normalized **2.566** [2.557, 2.569], 383.5 tok/s against the bracketed baseline's 149.5.
  - Peak inference memory **177,542,656 B** (attempt 7: 231,937,536), KV cache 7,864,320 B.
  - The paired baseline passed on the BF16 fast path.

**Re-arm without a new Scientist call: continuation 20261105c.**
- Manifest `claude_t_continuation_manifest_20261105c.json` **`d3465088…`**, read-only.
- Remote state `/var/lib/autopareto/pair_a_claude_opus55_t_seed3_continuation_20261105c`.
- The 20261105b state is unchanged (`ed3bae00…`). So are its failed receipt and log on the pod, and the failed
  run's checkpoint.
- Amendment record `runs/replacement_a40_pod3_20261105/attempt8_fp32_fix/amendment_record_20261105c.json`
  (`bfa9ea7c…`). Its transition is `protected_harness_fix_same_a40`, and the abandoned invocation is recorded
  truthfully (durable receipt present, `FAILED`).
- Identical to the paused attempt 8: candidate `0d1dcd8c…`, parent `attempt-07`, hypothesis, intended change,
  category and the Scientist record.
- Changed in the metadata: only `evaluator_hashes.inference_benchmark.py` and the metadata's own hash.
- Changed in the manifest: three immutable files (inference_benchmark.py, continuation.py, the preparation tool)
  and the inference section's evaluator pin.
- Preparing a chained continuation past a recorded candidate failure exposed three more gaps in
  `continuation.py`. Each is fixed with tests, fail-closed:
  - commit f915705: the prefix audit now accepts a recorded failure, which has no source registry row;
  - commit 83c4f81: a chained continuation keeps its predecessor's `migration/` records;
  - commit b4facee: a manifest-pinned file hash moves only with a listed amendment.
- A dry preparation in scratch was accepted by the controller's own `_verify_final_manifest` and
  `_initialize_state` before the real one.
- Launch gates (`launch_gates_continuation_20261105c.json`): manifest, transport and state config, 122 trusted
  files, GPU, 14 absolute files, baseline record, controller state. All passed, and the controller state was
  unchanged by the checks.
- Armed: the pod confirmed `train.py` `0d1dcd8c…` and metadata `edb6fff5…`, `launch_requested`. A pre-arm copy
  of the state is kept.
- Supervisor launched at 09:22:45 UTC.

**Rerun result: attempt 8, global position 16, 09:22–09:33 UTC. Valid.**
- Same candidate `0d1dcd8c…` and parent `attempt-07`; invocation `pair-a-016-7368fbd3…`.
- Training: 1,474 updates in 300.144 s of protected time, 193,200,128 tokens.
- Validation BPB **1.09949**. Attempt 7, whose training path is byte-identical, scored 1.09953.
- Decode stable: normalized **2.571** [2.563, 2.581], 386.9 tok/s against the bracketed baseline's 150.6.
- Peak inference memory **177,542,656 B**, KV cache 7,864,320 B, reserved 207,618,048 B.
- Cache correctness: this retrained checkpoint passed the unchanged BF16 check directly (mean error 0.0224 against
  0.03, greedy tokens match). The rerun took the `bf16_fast_path` and did not need FP32 adjudication.
  - The fixed FP32 path was exercised only by the independent check on the original checkpoint, where it passed.
  - The two checkpoints differ because GPU training is not bitwise reproducible. Their decode and memory figures
    agree: 2.571 against 2.566, and identical memory.
- Pareto decision `retained`. Attempt 8 dominates attempt 7: lower BPB, faster and smaller.
- Frontier and parent pool are now attempts 6 and 8. The controller updated them normally.

The campaign resumed at once with attempt 9 (global position 18). Totals so far: 8 attempts, 7 valid, 1 candidate
failure (attempt 2).
- The failed first run of attempt 8 is kept unchanged, with no result committed for it:
  - 20261105b state;
  - on the pod, its receipt `FAILED` and its log;
  - its checkpoint, and the non-study copy in `/workspace/nonstudy_diagnostics/attempt8_fp32_adjudication_20261105/`.
- The extension tool of §10 now runs against the 20261105c state root.

## 13. Attempt 9 (global position 18): Scientist timeout before any candidate, requeued as in the Codex comparator (09:52–10:27 UTC)
**What happened.**
- The parent-selection call completed: 09:33:45–09:42:48, 543 s, parent `attempt-08`, category
  `inference_adapter_speed`.
- The child-edit call reached the frozen 600-s Scientist bound (09:42:48–09:52:49) and was stopped (SIGTERM,
  return code 143). It had made 16 tool calls, mostly reads of `campaign_view.json` and the parent, plus 10 reasoning
  blocks, and had not yet edited `train.py`.
- The controller paused the position as `ScientistExecutionTimeout` in phase `pre_remote_execution`. There was no
  pending execution, no candidate and no A40 work.
- The pause went unnoticed for 35 minutes. It landed in the seconds between one 30-minute event watch expiring and
  the next starting from the end of the log. The watch now resumes from a saved log offset.

**The bound.** The watchdog (`isolation._supervised_codex`) waits on a monotonic clock, which on macOS stops during
sleep. Calls whose wall time exceeds 600 s and still completed spanned Mac sleep:
- Opus attempt 1, 833 s;
- Codex attempts 6, 8 and 14: 1,200, 1,798 and 1,098 s.

The bound was the same 600 s of awake time in both arms. Every invocation records `timeout_seconds: 600`.

**Trend.** Opus parent-selection calls grow with the campaign view: 132–255 s at attempts 2–7, 395 s at 8, 543 s
at 9. Child-edit calls were 50–88 s at attempts 2–7, then 294 s at 8 (the adapter) and 600+ s at 9. Further timeouts
are likely as the view grows.

**Handling: the Codex comparator's own precedent.**
- In the comparator lineage, position 12 (`operational_amendments[1]` of
  `final_pair_a_continuation_manifest_20260916_r2_requeue031.json`) was handled this way: "Scientist parent-selection
  invocation timed out before any remote execution or scientific attempt … Requeue the same unconsumed scheduled
  position without changing scientific components or timeout."
- Position 31 was requeued after a Scientist capacity failure.
- The same handling here: `scripts/requeue_scientist_timeout.py` (new, fail-closed, commit 8750a6b). It requires:
  - a `ScientistExecutionTimeout` in `pre_remote_execution`;
  - no execution, pending candidate or result;
  - every earlier position completed and every later one pending;
  - the named diagnostics to record the bounded timeout for this attempt.
- It keeps a read-only copy of the state (`controller_state.json.pre_requeue_scientist_timeout_position_018_20261105T102711Z.json`),
  records the timeout inside the position (`requeued_scientist_timeouts`), and sets the position back to `pending`.
- Timeout, manifest (`d3465088…`), code and prompt are unchanged. The timed-out call's diagnostics stay in place.
- Test: `scripts/tests/test_requeue_scientist_timeout.py`, the real controller loop offline with a forced child-edit
  timeout. After the requeue the campaign completes 25 attempts trained once each, and attempt 4 shows the failed
  call plus two fresh ones.
- Relaunched at 10:27 UTC with the same command. Attempt 9 gets fresh parent-selection and child-edit calls.
- If timeouts recur, the PI decides the next step. A larger bound or a compacted view would differ from the
  comparator's 600 s and full view.

**Second timeout of attempt 9 (10:27–10:42 UTC), and the PI's decision.**
- Fresh parent selection: 10:27:21, 282 s, parent `attempt-08`, category `inference_adapter_decode_speed`.
- Child edit: 10:32:03, stopped at 600 s after 20 tool calls, 3 of them edits. The unfinished edit was discarded
  with its temporary context.
- The campaign view is now 2,215,335 B. Nothing reached the A40. Seat usage was 34% of the 7-day window.
- Options given to the PI: requeue again at 600 s (the comparator's own handling), raise the bound to 900 s for the
  rest of the arm (disclosed asymmetry), or charge the attempt as a failure.
- The PI chose "Requeue, keep 600 s". The same tool requeued it at 10:55:52 UTC (state copy
  `…pre_requeue_scientist_timeout_position_018_20261105T105552Z.json`; the position now records 2 requeued timeouts).
- Relaunched with the unchanged command, manifest `d3465088…` and the 600-s bound.

**Third timeout of attempt 9 (10:55–11:10 UTC). Paused for the PI.**
- Parent selection: 10:56:03, 283 s, parent `attempt-08`, `inference_adapter_speed`.
- Child edit: 11:00:46, stopped at 600 s after 18 tool calls, 3 of them edits.
- All three tries chose an inference-adapter speed change on attempt 8, and none finished its child edit in 600 s.
- Seat usage 35% (7-day window), 18% (5-hour window). Not requeued again: the PI's decision is needed.

## 14. PI amendment: Scientist timeout 600 → 900 s for the rest of the Opus arm (from attempt 9, 11:39 UTC)
**Decision.**
- After three consecutive child-edit timeouts at attempt 9 (§13), the PI chose "Raise to 900 s".
- The choice was made at 00:3x NZDT on 2026-11-06, which is 2026-11-05 11:3x UTC. The approval text inside the
  amendment carries the PI's local date.

**Disclosure.**
- Every Codex comparator result ran at 600 s.
- Opus attempts 1–8 ran at 600 s; attempts 9–25 (and the extension attempt 26) run at 900 s.
- An earlier, abandoned comparator lineage made the same change for the same reason:
  `final_operational_manifest_pair_a_20260909_t6_timeout900.json`, "raise only the bounded Scientist timeout to
  900 seconds and resume the same scheduled position".
- The paper must state the asymmetry. It should also report that 3 of attempt 9's child-edit calls (and none before
  it in this arm) reached the 600-s bound.

**How it was applied** (`scripts/amend_scientist_timeout.py`, new, commit dab88a2, unit-tested):
1. `requeue_scientist_timeout.py` requeued position 18 for the third time: state copy
   `…pre_requeue_scientist_timeout_position_018_20261105T113901Z.json`, with 3 requeued timeouts now recorded in the
   position.
2. New manifest `claude_t_continuation_manifest_20261105c_timeout900.json` **`75b090e6…`**, read-only. It is identical
   to `d3465088…` except:
   - `scientist.configuration.timeout_seconds` 600 → 900;
   - one `operational_amendments` entry in the comparator's form: schema `autopareto_control_plane_amendment_v1`,
     effective at position 18 and attempt 9, `scientist_timeout_seconds` {600, 900}, no changed immutable paths,
     scientific components and contract unchanged, with the reason and the PI's approval.
3. The controller state now points to the new manifest and `scientist_config.timeout_seconds` = 900. The amendment
   is recorded in the state, and a pre-amendment copy is kept. Position 18's last requeue record notes the amendment.
- A dry run on a scratch copy was checked first. The controller's own `_verify_final_manifest` and
  `_initialize_state` accept the amended manifest and state with a 900-s launch, and reject a 600-s one.
- Launch gates with the 900-s configuration: all passed, state unchanged
  (`launch_gates_continuation_20261105c_timeout900.json`).
- Relaunched at 11:39:43 UTC with `--manifest … _timeout900.json --manifest-sha256 75b090e6… --scientist-timeout-seconds 900`.
  Nothing else in the command changed.

## 15. Attempt 9 (global position 18), 900-s bound: candidate cache-size accounting failed the protected integrity check (12:00 UTC). Paused, nothing changed
**What happened.**
- Both Scientist calls completed under the 900-s bound. Category `inference_adapter_decode_speed`, parent `attempt-08`,
  candidate `98b693ed…`: attempt 8's training path byte-identical, plus a per-state CUDA-graph decode step in its adapter.
- Training completed: 1,473 updates (11:53–11:59 UTC).
- The decode warm-up then failed in the protected check `protected/inference_adapter.validated_cache_nbytes`, which
  exists to "reject fabricated cache-size reports": the candidate's `cache_nbytes` reported 7,864,320 B, while the
  protected traversal of the candidate's own decode state found 7,907,344 B.
- The candidate's state holds `{"k", "v", "pos", "graph", "io"}`, but its `cache_nbytes` counts only `k` and `v`.
  The 43,024 uncounted bytes are the CUDA graph's static I/O tensors in `state["io"]`. This is a candidate-side
  accounting error, and the check worked as designed.
- The remote runner attributes a post-training failure to the candidate only for listed messages
  (`train_launcher._CANDIDATE_POST_TRAINING_FAILURE_MARKERS` and the value-embedding shape detail). This message is
  not listed, so the origin is `unknown`, giving `RemoteInfrastructureFailure` and a pause. The failed receipt and
  log are kept on the pod. Neither arm had hit this check before.

**Comparator precedent.** In the comparator lineage, two post-training protected-inference failures of unknown origin
(N19 at position 37, N20 at position 39) were reviewed and **charged as candidate scientific failures**. Each was
handled by an operational amendment that added the proven message to the train_launcher classification and reconciled
the position once, in `final_pair_a_continuation_manifest_20260917_posttraining_failure_receipt_fix.json`. Unrecognized
origins stayed fail-closed.

**Status.** Paused, PI decision requested.

## 16. PI decision: attempt 9 charged as a candidate failure, extension to 25 valid results, relaunch from attempt 10 (13:41 UTC)
**Decision (PI, 2026-11-06 local / 2026-11-05 UTC, verbatim).** "Charge attempt 9 as a candidate failure, preserve its
logs and exact failure reason in the full lineage so Opus can learn from it, and apply the tested error-classification
fix. Resume from attempt 10 without changing Opus's code manually. Also extend the campaign to target 25 valid
experiments (currently 27 total attempts), while preserving the original 25-attempt results separately and
documenting all failures and extra compute for fair reporting."

1. **Classification fix** (commit 87de802). `"candidate cache_nbytes reported"` was added to
   `train_launcher._CANDIDATE_POST_TRAINING_FAILURE_MARKERS`. Only `protected/inference_adapter.validated_cache_nbytes`
   raises that message, and only when a candidate adapter's own `cache_nbytes(state)` disagrees with the protected
   traversal of the state that adapter built. Every other unrecognized origin stays fail-closed.
   - Tests: attempt 9's exact error, and the real message of the protected check.
   - A40 runtime: 27/27 in a non-study copy (`/workspace/nonstudy_diagnostics/attempt9_cache_marker_20261106/`) and
     27/27 in the pod's repo copy.
   - Mac: same failing set as before (torch and Docker are absent) plus the new torch test.
   - `train_launcher.py` `0bf6da10…` → `92183ae0…`. Deployed to the trusted tree and the repo copy at 13:40:11 UTC
     (`/workspace/pod_setup_logs/post_restore_repo_sync.log`). It is the only immutable-deployment change.
2. **Evidence**, copied byte-identical to `runs/replacement_a40_pod3_20261105/attempt9_cache_failure/pod/`:
   - durable receipt `1d0dc7e5…`, log `58321cfb…` (78 lines), worker receipt `696b32ec…`, launcher output `910b033c…`;
   - the absence check at 13:29:43 UTC (`absence.json`): no worker, no bridge result, GPU idle.
   - The checkpoint that was trained but never measured (`b25cb2f9…`) stays on the pod's container disk. It is part of
     no result.
3. **Reconciliation, once**, by `scripts/reconcile_post_training_candidate_failure.py` (commit 304a338). It follows the
   comparator's N19/N20 form.
   - **Method.** First rehearsed on a copy of the state. On the live state it re-checked the receipt and evidence,
     classified the unedited error with the repository's launcher (`candidate`), then committed the runner's own
     `scientific_failure` result through the controller's `_commit_pending_result`.
   - **New manifest.** `claude_t_continuation_manifest_20261105c_timeout900_attempt9_candidate_failure.json`,
     **`35e50181…`**, read-only. Only the `train_launcher.py` pin changes, and one `operational_amendments` entry is
     appended with the reason, approval, evidence hashes and precedent.
   - **State copy.** `controller_state.json.pre_reconcile_position_018_20261105T133930Z.json` (`6ec3d8df…`), read-only.
   - **Position 18.** Now `completed` / `scientific_failure`, with its original `reconciliation_failure` record kept.
     Result file `position-018-…json` (`7026a807…`).
   - **What Opus sees for attempt 10 onward.** Attempt 9's entry in `failures` with the exact error, and its full
     `same_arm_evidence` record (hypothesis, intended change, 9.8 KB patch, parent `attempt-08`, invalid).
   - **Unchanged.** Frontier and parent pool (attempts 6 and 8). No candidate was rerun, no Scientist call was made,
     and no Opus code was edited.
4. **Extension to 25 valid results** (`scripts/run_extension_attempt.py`, commit 304a338).
   - **Valid result.** A completed attempt whose outcome is not `scientific_failure`.
   - **How extra attempts run.** One at a time after attempt 25 (positions 51, 52, …; attempts 26, 27, …), until 25 are
     valid. Each opens only after the previous one completes. A recorded safety cap of **30 total attempts** stops the
     run (exit status 3) for the PI.
   - **Expected size.** With failures 2 and 9 and no further failure, this means 27 attempts.
   - **Frozen 25-attempt result.** `prepare` keeps a read-only tar snapshot of the 25-attempt state root, plus hashes of
     its view, history and state, in `provenance/extension_attempts_amendment.json`.
   - **Per extra attempt.** A read-only copy of the view, and the view's `attempt_number` as its only change. Its two
     Scientist prompts end with a note naming the failed attempts and the valid count. No other prompt has one.
   - **One process.** The new `campaign` step runs the remaining scheduled attempts, then the extension.
   - **Tests.** 3/3 extension tests: a failed extra attempt opens the next one, the cap stops it, and the run works in
     one process. 16/16 tests across `scripts/tests`.
5. **Launch gates** at 13:40:59 UTC: all passed. 122 trusted files, including `train_launcher.py` `92183ae0…`. The state
   is valid and unchanged by the check, and the next pending position is 20.
   Output: `runs/replacement_a40_pod3_20261105/launch_gates_continuation_20261105c_attempt9_candidate_failure.json`.
6. **Relaunch at 13:41:39 UTC** (supervisor pid 7345, log `….supervisor.nohup.5.log`). The command is the §14 command,
   except the program is `scripts/run_extension_attempt.py campaign` and the manifest is the new one (`35e50181…`).
   Attempt 10 (position 20) started at once, and its prompt carries no extension note.
7. **Failure and compute ledger.** `scripts/campaign_compute_ledger.py` writes
   `supporting/opus_arm_compute_ledger.{md,json}` from the four state roots and the pod receipt backups. It lists every
   attempt, every Scientist call (used or extra) and every A40 invocation (counted or extra). Regenerate it at the end.
   - At 13:44 UTC: 9 attempts completed, 7 valid, failures 2 and 9.
   - 6 extra Scientist calls (48.5 min), all in attempt 9: three completed parent selections superseded by requeues,
     and three child-edit timeouts.
   - 2 extra A40 invocations: attempt 2's 2-s sandbox refusal on the first pod (doc 16), and attempt 8's FP32
     crash (6.6 min).
   - Attempt 9's failed run (6.6 min) counts as its attempt.

**Disclosures for the paper** (in addition to those of §§12–14):
- attempt 9 was charged as a candidate failure after a PI-approved classification amendment, as the comparator did for
  N19 and N20;
- the Opus arm reports both the frozen 25-attempt result and the extended result (25 valid results). The Codex arm had
  no extension, and its scientific failures were not replaced;
- the ledger's extra Scientist and A40 compute.

## 17. Attempt 18 (global position 35): child-edit Scientist timeout at the 900-s bound (17:21 UTC). Paused, nothing changed
- From the 13:41 relaunch, attempts 10–17 completed, all of them valid.
  - Retained: 10, 11, 13, 14, 17.
  - Quality-valid but speed-uncertain: 12, 15, 16, because the bracketed normalized drift exceeded the frozen 2% limit
    (2.10%, 2.85%, 2.02%).
  - The frontier is now attempt 17 alone: BPB 1.09915, decode 17.02×, 127.0 MB.
- Attempt 18's parent selection completed in 481 s. Its child edit then reached the 900-s bound (900.6 s, 1.2 MB of
  streamed output); the diagnostics are `scientist_diagnostics/scientist-1793898409345907000-ce51babccdfc/`.
- The position is `reconciliation_required` / `ScientistExecutionTimeout` / `pre_remote_execution`. There is no pending
  candidate directory, no execution and no A40 work, and no Scientist or controller process is left. The supervisor
  exited with `paused_reconciliation_required`.
- Since the 900-s amendment, 19 of 20 Opus calls finished within the bound (child edits 61–760 s, parent selections
  156–852 s).
- **Status.** Paused, PI decision requested: requeue at the unchanged 900-s bound (`scripts/requeue_scientist_timeout.py`,
  as in §13), or amend the bound.

## 18. PI decision: requeue attempt 18 at the unchanged 900-s bound; relaunch (20:03–20:04 UTC)
- **Decision.** The PI chose "Requeue, keep 900 s". The bound stays at 900 s, and the manifest stays `35e50181…`.
- **Requeue** (`scripts/requeue_scientist_timeout.py --global-position 35`, 20:03:35 UTC). This is the procedure of §13 and
  the Codex comparator's position 12.
  - Record: `phase_timed_out: edit_child`, `candidate_existed: false`, `a40_work_existed: false`,
    `scientific_position_consumed: false`, `timeout_changed: false`.
  - Diagnostics hash `99ec883d…`.
  - Read-only state copy `controller_state.json.pre_requeue_scientist_timeout_position_035_20261105T200335Z.json`
    (`2137c4c2…`).
  - Before the requeue the pod was idle: GPU at 0 %, no study process, and the last bridge result was position 33.
- **First relaunch: no effect** (20:03:47 UTC, log `….supervisor.nohup.6.log`).
  - The controller's startup check of the pod's trusted files failed: the RunPod SSH proxy closed the PTY session
    mid-command ("Connection to ssh.runpod.io closed by remote host"), with returncode 255. The supervisor classified this
    `controller_failed` after 3.4 s.
  - The check runs before any position work, so nothing ran: no Scientist call, no A40 work, no change to the state.
    Position 35 stayed `pending`, with no reconciliation failure.
- **Second relaunch** (20:04:32 UTC, log `….supervisor.nohup.7.log`). The command was identical to §16 step 6. The
  trusted-file check passed, and attempt 18 restarted at position 35 with a fresh parent-selection call.
- **Disclosure.** Attempt 18 used one extra Scientist call (the timed-out child edit, 900.6 s) plus its superseded parent
  selection (481 s), as attempt 9 did. The compute ledger counts both as extra.

## 19. Attempt 18 result; attempt 19 (global position 38): parent-selection Scientist timeout at the 900-s bound (20:49 UTC). Paused, nothing changed
- **Attempt 18** completed at 20:35 UTC (KV-cache change, parent attempt 17).
  - Result: BPB 1.09965, peak inference memory 116.9 MB.
  - Verdict: `quality_valid_speed_uncertain`. The normalized drift was 6.92%, against the frozen limit of 2%.
  - The candidate's own decode drift was 0.14%. The two baseline brackets drifted 5.49% (before) and 9.93% (after),
    the largest in the arm.
  - Across the arm since attempt 10, candidate drift is 0.01–0.14% and baseline-bracket drift is 0.3–9.9%. All four
    speed-uncertain verdicts (attempts 12, 15, 16, 18) come from baseline drift.
  - At 20:35 UTC, with our GPU idle, the host's load average was 59 on 96 cores (about 30 over 5 min).
  - Nothing was changed. The 2% rule and the bracketing are frozen protocol.
- **Attempt 19's parent selection** reached the 900-s bound: 900.8 s, 954 kB of streamed output. Diagnostics:
  `scientist_diagnostics/scientist-1793910891837664000-ee77ec5f7b8b/`.
  - The position is `reconciliation_required` / `ScientistExecutionTimeout` / `pre_remote_execution`.
  - Nothing was left behind: no `pending/position-038` directory, no Scientist or controller process. The pod is idle
    (GPU 0 %), and its last bridge result is position 35.
  - The supervisor exited with `paused_reconciliation_required`.
- **Scientist calls since the 900-s amendment.** 22 calls in total; 2 reached the bound (attempt 18's child edit and this
  call).
  - Parent selections in the last six attempts took 852, 209, 650, 481, 502 and 900+ s. They grow with the lineage.
- **Status.** Paused, PI decision requested.

## 20. PI decision: automatic requeue of pre-candidate Scientist timeouts at 900 s; attempt 19 requeued (21:35 UTC)
- **Decision.** The PI chose "Auto-requeue at 900 s" (2026-11-06 NZDT).
  - From now on, a Scientist timeout that leaves no candidate and no A40 work is requeued without waiting for the PI,
    with the same procedure as §§13 and 18.
  - The third timeout of one attempt stops the run for the PI, and so does every other kind of pause.
  - The bound (900 s), the manifest (`35e50181…`), the code and the command stay unchanged.
- **Wrapper** `scripts/auto_requeue_scientist_timeouts.py` (commit cc604b1). It runs the identical supervisor command
  and acts only when both of these hold:
  - the supervisor classifies the exit `paused_reconciliation_required`;
  - the single paused position failed with `ScientistExecutionTimeout` in `pre_remote_execution`.
  It then calls `requeue_scientist_timeout.requeue_scientist_timeout`, which itself refuses a position with a pending
  candidate, an execution record, a result, or diagnostics that do not record the bounded timeout.
  - It counts every earlier requeue of that position, including manual ones. It does not requeue a third timeout of
    the same position, and it stops after 12 automatic requeues in one run.
  - A completed campaign exits 0; any stop for the PI exits 4.
  - Each requeue is logged in `<supervisor-dir>/auto_requeue.jsonl`. The position's requeue record carries
    `requeued_by_wrapper` and the approval, and a read-only copy of the state before the requeue is kept, as before.
  - Tests: 3/3, end to end with the real supervisor and the dry controller.
    1. A single timeout is requeued and the campaign completes: 25 trainings, none repeated.
    2. A recurring timeout is requeued twice and the third stops for the PI.
    3. A post-training failure of unknown origin is not requeued.
- **Attempt 19** (position 38) was requeued by hand at 21:34:59 UTC, as the PI decided (first timeout of the position:
  `select_parent`).
  - State copy: `controller_state.json.pre_requeue_scientist_timeout_position_038_20261105T213459Z.json`.
  - Diagnostics hash `0f14546e…`.
- **Relaunch** at 21:35:11 UTC under the wrapper (log `….supervisor.nohup.8.log`). The supervisor and controller
  arguments are identical to §16 step 6. Attempt 19 restarted with a fresh parent selection.
- **Disclosure.** Every requeued call is an extra Scientist call in the compute ledger. The paper reports how many
  timeouts each arm had and how they were handled. The Codex comparator requeued one, at position 12.

## 21. Campaign complete: 27 attempts, 25 valid results (02:27 UTC, 2026-11-06)
- **End.** The wrapper relaunched nothing after §20: there were no more Scientist timeouts and 0 automatic requeues.
  Attempt 27 completed at 02:27:18 UTC. The supervisor then exited `campaign_completed` ("all 27 positions
  completed") and the wrapper `stop_completed`, both with return code 0.
- **Tally.** 27 attempts and 25 valid results. The scientific failures are attempts 2 and 9.
  - **Frozen 25-attempt result** (the `prepare` snapshot `….snapshot_25_attempts_20261106T011555Z.tar`, with hashes in
    `provenance/extension_attempts_amendment.json`): 23 valid; frontier attempts 21, 23 and 24.
  - **Extended result:** frontier attempts 23, 24 and 27. Attempt 27 dominates attempt 21.

  | Frontier | BPB | Decode tok/s (× baseline) | End-to-end tok/s | Peak MiB | Change |
  |---|---|---|---|---|---|
  | 23 | 1.09620 | 4573 (30.84×) | 696 | 113.0 | token-shift embedding |
  | 24 | 1.09946 | 4588 (31.40×) | 851 | 96.9 | shared value embeddings |
  | 27 | 1.09623 | 4580 (30.54×) | 720 | 99.1 | int8 embedding-table storage |

  - Baseline: BPB 1.17909, about 145 tok/s decode and end to end.
- **Speed-uncertain results.** 7 of the 25 valid results are speed-uncertain: attempts 12, 15, 16, 18, 19, 22 and 26.
  In every case the normalized drift came from the baseline brackets; candidate drift stayed at 0.01–0.14% from
  attempt 10 on. The frozen 2% rule was applied unchanged. Report the rate next to the Codex arm's.
- **Read-only audit of the speed numbers** (attempt 22, during the run; nothing changed).
  - The decode timing is GPU-synchronized and each step depends on the previous one. Each `decode_one` replays a
    per-state CUDA graph with nothing carried between requests. FP32 logits match the model forward (max 1.4e-5).
    BPB comes from the protected evaluator over 2.1e7 tokens, on a hash-verified checkpoint loaded in a fresh
    process.
  - **Caveat for the paper.** From attempt 10, the adapter captures its decode CUDA graph (and, later, its compiled
    regions) inside `prefill`, which the decode-only metric (Design B) does not time. Prefill grew from 3–7 ms to
    93–152 ms. End-to-end speed (prefill plus 128 tokens) went 4.5× → 5.1× over attempts 10–27, while decode-only
    went 8.6× → 32×. Report both, and check whether the Codex arm shows the same pattern.
  - **Harness gaps** (no effect on the speed numbers): the outputs of the timed windows are not checked, and the
    correctness prefixes (64 and 256 tokens) do not exercise the adapter's prefill ring wrap at T = 1024.
- **Ledger** (regenerated 02:40 UTC, after the verified pod pull of 02:39:24Z;
  `supporting/opus_arm_compute_ledger.{md,json}`).
  - Scientist calls: 63 calls, 383.5 min. 54 were used. 9 were extra, totalling 86.5 min (5 timed out, the rest were
    superseded by requeues).
  - A40 invocations with a durable receipt: 28 (367.9 wall-min). 26 were counted. 2 were extra, totalling 6.6 min.
  - The extension (attempts 26–27) used 33.3 Scientist-min and 36.1 A40-min.
