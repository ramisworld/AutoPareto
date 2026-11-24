# Phase 6 incidents

Infrastructure events during the Phase 6 runs. This file is not part of the frozen plan (plan sha256
dc1e605e...), so writing it changes no hashed input.

## 1. P0a stopped by a network-volume failure (2026-11-16)

**What happened**
- The chain (`camera_ready_phase6_chain.sh`) started P0a at 03:58:38 UTC. The runner exited with
  status 120 at 05:31:59 UTC, and the chain stopped as designed. Python returns 120 when flushing
  stdout fails at interpreter shutdown.
- 9 of the 15 P0a runs had completed, each with a complete run record (status `completed`,
  0 hard mismatches):
  - opus_a24-seed6, gpt_a22-seed6, arref-seed6
  - opus_a23-seed6, opus_a27-seed6
  - gpt_a22-seed7, arref-seed7
  - opus_a23-seed7, opus_a27-seed7
- The runner exited during preparation for the 10th run (opus_a24-seed7), before it wrote that run's
  `inference_started` event. The GPU was idle from 05:32 to 05:36 UTC.

**What was lost**
- In the chain console log (`/workspace/camera_ready_phase6_console.log`, sha256 3c4aa3ca...),
  about 590 bytes are NUL in place of output written after 05:08. That output was:
  - the console summary lines of opus_a23-seed7 and opus_a27-seed7;
  - any error the runner printed before exiting.

  The runner's error therefore cannot be recovered.
- The pattern matches the volume stalls seen earlier the same day, when single 64 MiB chunks stalled
  for minutes. There is also a 4.4 min gap between opus_a23-seed7 finishing (05:08:38) and
  opus_a27-seed7 starting (05:13:02).
- The volume also served stale reads. At 05:33, `inference_events.jsonl` showed 17 lines and no
  `inference_finished` event for opus_a27-seed7. At 05:36, the same file showed that event with
  timestamp 05:26:03.

**Why no measurement was lost or changed**
- Both seed-7 Opus run records parse completely. Their sha256 values:
  - opus_a23-seed7-native.json: 9f840293...
  - opus_a27-seed7-native.json: 982319c1...
- Each per-run log ends with the summary line that the console would have shown:
  - opus_a23-seed7: 4,580 tok/s, 31.5x, token agreement 1,022 of 1,024.
  - opus_a27-seed7: 4,586 tok/s, 30.9x, token agreement 1,021 of 1,024.
- The original console log is kept unchanged.

**How the run resumed**
- At 05:36:29 UTC, `camera_ready_phase6_chain_r2.sh` re-ran the same commands from P0a, with the
  same frozen plan and the same stop rule.
- The runner skips any run whose record already exists with status `completed`, so the 9 completed
  runs are not repeated.
- The only change is where the chain console log lives. It now goes to the pod's local overlay disk
  (`/root/phase6_console_r2.log`) and is mirrored to `/workspace/camera_ready_phase6_console_r2.log`
  every 5 minutes.
- Run records and events still go to the volume, as the frozen runner specifies.

## 2. Slow training steps in P1 GPT A24 seed 6 (2026-11-16, about 07:10 UTC)

**What happened**
- 23 steps between steps 1,051 and 1,154 took 1.1 to 2.9 s each, against a median of 0.196 s.
- They fell every 4 to 5 steps over 51 s, about 33 s of excess time in total.
- The pattern fits stalled reads of training data from the network volume. This is an inference;
  nothing was logged to confirm it.

**Effect**
- Held-out BPB is unaffected: the run saw the same tokens in the same order.
- This run's steady-state training throughput is about 10% lower than it would be without the
  slow steps.
- The value is kept and reported as measured, per the protocol; this note goes alongside it.

**Comparison**
- No other P1 run so far (5 runs) has any step longer than twice its median.
- Neither has any E1 matched-token run on pod4.

## 3. P1 MOTPE seed 8 failed both attempts on a network-volume error (2026-11-16, 12:17 UTC)

**What happened**
- Attempt 1 ran from 12:17:08 to 12:18:21 UTC and attempt 2 from 12:18:21 to 12:19:27 UTC. Both
  failed before the first training step, while torch.compile was building Triton kernels.
- In both, a compile worker raised `OSError: [Errno 116] Stale file handle` from `os.makedirs` while
  writing the Triton cache inside the run directory on the network volume
  (`<run_dir>/compile_cache/triton/...`). The kernels named were `triton_poi_fused_add_mul_9`
  (attempt 1) and `triton_poi_fused_add_mul_6` (attempt 2).
- The runner recorded both attempts as failed, wrote `run_reported_failed` and went on to baseline
  seed 8, as the frozen one-retry rule specifies.

**Why this is an infrastructure fault, not a property of the recipe**
- The failing call is a directory creation on the network volume, not model code.
- MOTPE seeds 6 and 7 trained with the same code and environment and passed every gate.

**Consequences under the frozen plan**
- The P1 training stage will exit 1 at its end, because one run is reported failed. The chain's stop
  rule then stops before P1 downstream, P0b, controls and anchors, and the A1 activation script
  exits without switching storage or starting the pilot.
- The frozen downstream and inference stages look only at attempts 1 and 2. Without an amendment,
  MOTPE seed 8 has no checkpoint, no held-out BPB, no downstream scores and no inference record.
- Both failed attempt directories are kept unchanged.

**Throughput scan of the other P1 runs (to 12:30 UTC)**
- Steps after step 10 that took more than twice the run's median were summed as excess time.
- GPT A24 seed 6: 32.6 s of excess (incident 2). Its throughput is about 12% below GPT A24 seed 7's
  (614k against 697k tok/s), which fits the stall.
- Every other completed run: at most 0.4 s of excess.

## 4. P1 downstream crashed before evaluating anything; GPU idle 13:35 to 20:31 UTC (2026-11-16)

**What happened**
- P1 training ended at 13:34:03 UTC: 26 of 27 runs completed, and MOTPE seed 8 was reported failed
  (incident 3). The r2 chain stopped as designed.
- The r3 resume script checked that MOTPE seed 8 was the only failure, then started P1 downstream
  at 13:35:02 UTC.
- P1 downstream exited 1 at 13:35:03 UTC with
  `TypeError: tuple indices must be integers or slices, not str`, raised in
  `autopareto_db._add_columns`, which `migrate` calls.

**Cause**
- The frozen `execute_downstream` in `scripts/run_phase6.py` opens its new database without
  `db.row_factory = sqlite3.Row`.
- Phase 5A's function of the same purpose sets that line, and `migrate` needs it.
- This is a coding error in the frozen Phase 6 runner, not an infrastructure fault.

**What it left**
- `downstream_frozen/phase6_downstream.db`, with its tables created and 0 rows in every table.
- An empty `downstream_frozen/runs/`.
- No checkpoint was evaluated, and no measurement was taken.

**Idle time**
- The stop rule held the GPU idle while the author's decision was requested. The decision arrived at
  about 20:30 UTC.
- The GPU was therefore idle from 13:35 to 20:31 UTC, about 6.9 pod-hours.

**Resolution**
- Amendment A3 (`PHASE6_AMENDMENT_A3.md`).
- `camera_ready_phase6_chain_r4.sh` started at 20:31:06 UTC. It runs P0b, controls and anchors
  first, then A3 downstream, the A1 storage switch, A2 and the A1 pilot.

## Incident 5: pod restart during the controls (2026-11-16 to 11-17)

**What happened**
- The last write before the restart was at 23:16:03 UTC. `opus_a24-seed6-eager` had just started:
  its log was created with 0 bytes and no result file.
- The pod restarted. Its uptime counts from about 23:29 UTC.
- All processes stopped, including r4 and the armed P2 script r5. There was no exit stamp and no
  error in the console.
- The restart was found at 01:10 UTC on 11-17, when the author asked for a status check.
- **Unchanged:** the GPU (same UUID, GPU-6de81321), `/opt`, `/root/phase6_stage` and the volume.

**Effect on results**
- None. 23 P0b runs and 15 controls had completed before the restart, and the frozen runner
  skips them.
- `opus_a24-seed6-eager` has no result. It reruns as attempt 1, which overwrites its empty log.
- `opus_a23-seed6-canonical` finished before the restart as `incorrect_not_timed` (30 hard
  mismatches, 978/1,024 tokens agree). This is the preregistered expectation (PHASE6_PREREG.md,
  Modes), not a failure.

**Stale lock**
- The first relaunch, at 01:11:32 UTC, was refused with "GPU lock held". The flock on
  `phase6/gpu.lock` was still held by the dead pre-restart MooseFS client session. No local
  process, no `/proc/locks` entry and no GPU process held it.
- The lock file (0 bytes) was renamed to `gpu.lock.stale-incident5`, not deleted.
- That refusal wrote "STOP after P0b" to the r4 console, and r5 halted on it, as written.

**Resolution**
- r4 (`camera_ready_phase6_chain_r4.sh`, unchanged, sha256 13ab24b1...) was relaunched at 01:12:18
  UTC. P0b finished in 6 s (all skipped), and the controls resumed at `opus_a24-seed6-eager`.
- P2 is armed as `camera_ready_phase6_p2_r6.sh` (sha256 df2b2560...), which is r5 with two changes:
  - It reads r4's console only from r4's latest "r4 start" line.
  - It reads the budget cap from `/root/phase6_cap_hours` (35) at each guard.
- The pilot checks, the stages and their order are unchanged.

**Idle time**
- About 23:16 to 01:12 UTC, about 1.9 pod-hours.

## Incident 6: P2 lm-eval could not load the candidate's flash-attn3 kernel (2026-11-17)

**What happened**
- P2 fixed time completed: 34 of 34 runs, all on attempt 1 with no timing anomaly.
- At 08:12:05 UTC, P2-lmeval started its first cross-check model, `arref-seed6`. The run exited 1
  at model load.
- The candidate's `train.py` calls `kernels.get_kernel("kernels-community/flash-attn3")`. That call
  looked in lm-eval's HF cache (`/root/phase6_lmeval_cache/hf`, datasets only) with
  `HF_HUB_OFFLINE=1`, and raised `LocalEntryNotFoundError`.
- No checkpoint was scored, and no measurement was taken.

**Cause**
- The A1 lm-eval environment points every Hugging Face cache at the dataset cache.
- The kernel snapshot lives in the registered HF hub cache, which training and inference use.
- The lm-eval stage was not piloted, so this was not caught earlier.

**Idle time: 08:12 to 10:05 UTC (about 1.9 pod-hours)**
- The Mac-side watchers never fired. They ran ssh through an unsplit zsh variable, so every check
  failed silently (the same error as earlier in Phase 6).
- The stop was found when the author asked for progress.
- The watchers were replaced with inline ssh commands. Each check was tested on a running section
  and on a stopped one.

**Resolution: `camera_ready_phase6_p2_r7.sh` (sha256 3827ea93...), started at 10:05:30 UTC**
- It runs the unchanged P2 lm-eval stage with one addition: `KERNELS_CACHE` points at the
  registered HF hub cache. That cache holds the same flash-attn3 snapshot (e29f138f, torch29 build)
  that training and inference load.
- Unchanged: the runner, the A1 plan (validation passed), the tasks, the dataset cache and its
  manifest.
- The failed log was kept as `lmeval/crosscheck/arref-seed6.log.incident6`.
- A watchdog stops lm-eval if pod hours reach the cap in `/root/phase6_cap_hours`. The runner skips
  finished outputs, so it can resume.

## Incident 7: the lm-eval cross-check gate failed on BLiMP only (2026-11-17)

**What happened**
- r7 ran A1's cross-check from 10:05 to 10:17 UTC on AR-ref, GPT A22 and Opus A23, all seed 6.
- ARC-Easy, PIQA and LAMBADA matched the frozen evaluator exactly: 0.000 points on all 9
  comparisons.
- BLiMP was higher by 1.007, 1.140 and 1.045 points, against a tolerance of 0.5.
- A1 stopped the stage before the 39-checkpoint run, as written. `crosscheck_report.json` is kept
  unchanged.

**Cause, established by a diagnostic run from 10:24 to 10:28 UTC on the idle GPU**
- lm-eval 0.4.13 scores BLiMP as `("", " " + sentence)` and counts a tie as correct.
- The frozen evaluator scores `("", sentence)` and counts a tie as wrong.
- One scoring pass of GPT A22 seed 6 reproduces both values within 0.0015 points
  (`evaluation/phase6_a4_blimp_diagnostic.json`).
- The leading space accounts for the gap; only 7 of 67,000 pairs tie.
- Neither the frozen BLiMP results nor the checkpoints are in error.

**Proposed resolution: amendment A4 (`PHASE6_AMENDMENT_A4.md`)**
- The gate compares ARC-Easy, PIQA and LAMBADA. BLiMP is reported with the explanation above.
- `camera_ready_phase6_p2_r8.sh` (sha256 722386cf...) is armed. It waits for the A4 plan, which is
  written only after the author approves.
- A4's validation passed in a dry run with an in-memory plan. No plan file was written.
- **Author decision (11:45 UTC):** "Approve A4, run it". The A4 plan
  (`phase6_amendment_a4_plan.json`, sha256 852c799f...) was written with that approval, and r8
  started.
- **Timing probe before the decision** (11:14 to 11:17 UTC; output in `/root/lmeval_probe`, outside
  the results tree, not counted): one checkpoint with all 7 tasks took 3.0 min including load. The
  39-checkpoint run is therefore about 2 h.

## Completion (2026-11-17, 12:40 UTC)

- **r8 under A4.**
  - The gate compared ARC-Easy, PIQA and LAMBADA. It passed with a maximum difference of 0.0
    points, and BLiMP's gaps were recorded as explained by A4.
  - All 39 lm-eval checkpoints then finished with exit 0.
  - `PHASE6_P2_DONE` was stamped at 12:40:20 UTC.
- **No run was dropped.** Every planned measurement in P0, P1 and P2 was recorded.
  - **P0 inference:** 39 native runs, 19 controls and 3 anchors.
  - **Matched tokens:** 39 runs, including the reused E1 and Phase 5A runs, plus 3 drift controls.
  - **300 s:** 39 runs, including the reused E1 runs, plus 1 drift control.
  - **Frozen downstream:** 39 checkpoints.
  - **lm-eval:** 39 checkpoints.
- **Results copied to the Mac** (`results/camera_ready_2026/phase6/`, checksums identical to the
  pod):
  - every record and log, except checkpoints and compile caches;
  - the pod-local console logs, kept in `console_logs/`;
  - the A4 BLiMP diagnostic output.
  The one-checkpoint lm-eval timing probe is kept there too, marked NOT_COUNTED.
