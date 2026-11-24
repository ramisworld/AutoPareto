# Phase 6 budget recompute after the P0 pilots (2026-11-16 04:00 UTC)

Required by the brief ("After the first training and downstream pilot, recompute the projected total
A40-hours"). This file is not part of the frozen plan (plan sha256 dc1e605e...), so writing it changes
no hashed input. Times are pod wall time, which is what is billed.

## Inputs

| Input | Source | Value |
|---|---|---|
| Pod hours already used | pod start about 01:09 UTC to 03:59 UTC | 2.8 h (setup, pilots, and 1.7 h idle after a pilot-script bug) |
| Opus native inference run | pilot-opus23-s6-native-full: 2,312 timed + 722 warm-up repetitions, each paying about 160 ms of prefill and graph capture for about 28 ms of decode | about 11.5 min without the storage stall (912 s with it) |
| Other native or uncached inference run | pilot-arref-s6-uncached-full: 271 s | about 5 min |
| Matched-token training, AR-ref-sized | E1 attempt walls 18.6 to 18.9 min | 19 min |
| Matched-token training, GPT A22-sized | E1 attempt walls 11.7 to 12.3 min | 12 min |
| Matched-token training, Opus | Phase 5A attempt walls 10.4 to 11.1 min | 10.6 min |
| Fixed-time 300 s run | E1 time300s attempt walls 7.9 to 9.1 min | 8.7 min |
| Frozen 4-task downstream | Phase 5A: 15 checkpoints in 41.4 min | 2.8 min per checkpoint |
| lm-eval 11-task suite | not piloted yet | 5 min per checkpoint (assumption) |

Classical recipes and the baseline are costed as AR-ref-sized; GPT A24 and A25 as A22-sized.

## Projection

| Block | Runs | Hours |
|---|---|---|
| Used so far | | 2.8 |
| P0a inference, 15 existing checkpoints | 6 x 5 min + 9 Opus x 11.5 min | 2.2 |
| P1 training | 18 x 19 + 6 x 12 + drift 19 + 12 + 10.6 min | 7.6 |
| P1 frozen downstream | 24 x 2.8 min | 1.1 |
| P0b inference, 24 new checkpoints | 24 x 5 min | 2.0 |
| Controls | 13 uncached, 3 Opus eager reference, 3 canonical-forced correctness | 1.4 |
| Anchors | 3 | 0.4 |
| P2 fixed-time | 34 x 8.7 min | 4.9 |
| P2 lm-eval, 39 checkpoints, plus dataset preparation and pilot | 39 x 5 min + 0.3 h | 3.6 |
| P3 lineage re-measure | 15 x 5 min | 1.3 |
| P3 supplementary workloads | 39 runs, about 6 min each | 3.9 |
| Total | | about 31 (range 29 to 33) |

The projection exceeds the 24 h planning budget and the 30 h contingency. Per the brief, the author
is alerted before the budget is exceeded, and no required result is dropped without the author's
decision. P0 and P1 are required under every option and run first (about 15 h); the decision only
affects P2 and P3 scope.

## Update, 2026-11-16 07:05 UTC (after P0a and the first P1 run)

- **P0a.** It ran from 03:58 to 06:38 UTC, including the volume incident (see PHASE6_INCIDENTS.md).
  Opus runs took 12.5 to 13.8 min each, not 11.5. Hours used through the end of P0a: 5.5, against
  the 5.0 projected.
- **First P1 run** (p6-baseline-tokens201326592-seed6). Total wall time was 23.5 min, against the
  19 min projected:
  - Training itself is in line with E1 AR-ref seed 6: 670 s against 692 s of training time, and
    828 s against 794 s for the whole training process.
  - The extra time is in the held-out evaluation of the 4 checkpoints after training: 9.6 min,
    against about 5.3 min in E1.
  - The likely cause is the degraded network volume, from which the evaluator reads each 159 MB
    checkpoint. This is an inference only; it was not tested while training runs, to avoid
    disturbing the timings.
  - It affects wall time, not any measurement.
- **If the overhead holds for every run with milestone evaluation:**
  - P1 training rises from 7.6 to about 9.5 h (27 runs, plus 4.3 min each).
  - P2 fixed-time rises from 4.9 to about 7.3 h (34 runs).

| Block | Hours |
|---|---|
| Used through P0a (actual) | 5.5 |
| P1 training | 9.5 |
| P1 downstream, P0b, controls, anchors | 4.9 |
| P2 fixed-time | 7.3 |
| P2 lm-eval | 3.6 |
| P3 lineage and supplementary | 5.2 |
| Total | about 36 (range 34 to 38) |

- **Required P0 and P1 come to about 20 h.**
- **Possible mitigation.** P2 needs a plan amendment anyway, for lm-eval. That amendment could
  pre-register local-disk checkpoints for the fixed-time stage, as already done for inference.
  This would recover most of the 2.4 h P2 overhead, for a total of about 34 h.
- **P1 is left as frozen.** Changing its storage mid-stage would make its runs inconsistent with
  each other.

## Update, 2026-11-16 08:00 UTC (after 4 P1 runs)

| Run | Wall time | Projected |
|---|---|---|
| baseline seed 6 | 23.5 min | 19 min |
| GPT A24 seed 6 | 19.0 min | 12 min |
| NSGA-II seed 6 | 21.6 min | 19 min |
| TPE seed 6 | 16.7 min | 19 min |
| **Total** | **80.8 min** | **69 min (+17%)** |

- Scaling the original 7.6 h P1 estimate by 17% gives about 8.9 h, so the total stays at about 35 to 36 h.
- All 4 runs completed on their first attempt, with no failures.

## Update, 2026-11-16 08:30 UTC (correction)

**Fixed-time cost was over-estimated in the 07:05 update.**
- That update added 4.3 min per fixed-time run for checkpoint evaluation.
- But a fixed-time run evaluates 1 final checkpoint, not 4, so P2 fixed-time is about 5.4 h, not
  7.3 h.
- That update also blamed the extra time on checkpoint reads. The 08:15 to 08:30 measurements in
  `PHASE6_AMENDMENT_LOCAL_STORAGE_DRAFT.md` do not support that.

**Remaining work**

| Block | Runs left | Hours left |
|---|---|---|
| Equal-token training (P1) | 21 (18 new, 3 drift controls) | 6.0 |
| Primary 1,024/128 inference (P0b) | 24 | 1.8 |
| Inference controls and anchors | 22 | 1.7 |
| Frozen 4-task downstream | 24 checkpoints | 1.2 |
| Fixed-time 300 s (P2) | 34 (33 new, 1 drift control) | 5.4 |
| lm-eval suite (P2, not piloted) | 39 checkpoints | 3.6 |
| **Required, remaining** | | **19.7** |
| Optional: lineage re-measure (P3) | 15 | 1.2 to 3.3 |
| Optional: supplementary workloads (P3) | 39 | 3.9 to 4.5 |

**Totals**
- Used by 08:30 UTC: 7.35 h.
- Required total: about 27 h.
- With all optional P3 runs: about 32 to 35 h.

## Author decision (2026-11-16, about 09:00 UTC)

- **Budget:** target about 27 to 30 A40-hours in total. Do not exceed 30 h without approval.
- **P1:** continue and finish as planned.
- **Amendment A1 (local storage):** approved for P2, provided all protocols, hashes and
  measurements are preserved. It must be validated before P2 (see `PHASE6_AMENDMENT_A1.md`).
- **P2:** approved. It covers:
  - the missing 300 s runs;
  - primary 1,024/128 inference and its controls;
  - held-out BPB;
  - downstream benchmarks for all 13 models on seeds 6 to 8.
- **lm-eval:** ARC-Easy, PIQA, LAMBADA and BLiMP are not repeated on all 39 checkpoints when the
  frozen evaluations are valid. A small cross-check is run instead.
- **Optional P3** (lineage re-measures, supplementary workloads): not approved for now.
- **GPT A24 seed 6 stall:** investigate it, flag unreliable throughput, and keep storage stalls out
  of the fixed-time comparisons.
- **Reporting:** report at once any essential result that is missing or invalid.

## Update, 2026-11-16 12:55 UTC (incident 3, amendment A2)

- **Failure.** P1 MOTPE seed 8 failed both attempts on a network-volume error (`PHASE6_INCIDENTS.md`,
  incident 3).
- **Author's decision.** Amendment A2 is approved: one extra attempt, under A1 storage
  (`PHASE6_AMENDMENT_A2.md`).
- **Plan hashes.**
  - The A1 plan was regenerated to `2b9fdc3e...`. Only the hashes of `run_phase6_p2.py` and
    `PHASE6_AMENDMENT_A1.md` changed. The original plan is kept as
    `phase6_amendment_a1_plan.v1-f8907d14.json`.
  - The A2 plan is `2ed01afe...`.
- **Execution.** `camera_ready_phase6_chain_r3.sh` (sha256 9ec2b9c2...) is armed. In order, it runs
  the remaining frozen stages, the A1 storage switch, A2 and the A1 pilot. Then it stops for review.

| Block | Hours |
|---|---|
| Used by 12:55 UTC | 11.8 |
| Rest of P1 training (baseline seed 8, GPT A24 seed 8, 3 drift controls) | 0.9 |
| P1 downstream (23), P0b (23), controls and anchors | 4.5 |
| A2 (train, downstream, inference) and A1 pilot | 0.7 |
| P2 lm-eval (39) and fixed time (34) | 9.0 |
| **Projected total** | **about 27 (cap 30)** |

## Update, 2026-11-16 20:35 UTC (incident 4: 6.9 idle hours)

- **What happened.** P1 downstream crashed at 13:35 UTC on a coding error in the frozen runner
  (incident 4). The GPU stayed idle until the author's decision arrived at about 20:30 UTC.
- **Hours used by 20:35 UTC:** 19.4.
- **Remaining time below** uses the measured P0a durations: 4.5 min per non-Opus run and 13 min
  per Opus run.

| Block | Hours |
|---|---|
| Used by 20:35 UTC | 19.4 |
| P0b (23 runs) | 1.7 |
| Controls (13 uncached, 3 Opus eager, 3 canonical) and anchors (3) | 2.2 |
| P1 downstream under A3 (23 checkpoints) | 1.1 to 1.5 |
| A2 and A1 pilot | 0.8 |
| P2 fixed time (34 runs) | 4.9 |
| P2 lm-eval (39 checkpoints, not piloted) | 3.6 |
| **Projected total** | **about 34 (33 to 35)** |

- **Against the cap.** This exceeds the author's 30 h cap. Without lm-eval, the total is about
  30.4 h.
- **Author's decision:** requested.

## Author decision (2026-11-16, about 20:40 UTC)

- **Cap raised to 35 A40-hours** ("Raise cap to 35 h"). The scope is unchanged: P2 fixed time and
  P2 lm-eval both run.

## P2 armed (2026-11-16 21:36 UTC)

- **Script.** `camera_ready_phase6_p2_r5.sh` (sha256 c62e2f79...) waits for r4 to finish the A1
  pilot. It starts P2 only if all of these hold:
  - the pilot completed on its first attempt with every gate passed;
  - the timing-anomaly gate parsed at least 100 steps and passed;
  - the copy to the volume was verified;
  - the held-out BPB is in [1.14, 1.18] (Phase 5A Opus A27 seeds 6 to 8 at 201.3M tokens: 1.156
    to 1.159).
- **Budget guards.** Before each stage it checks used hours plus the stage estimate against the 35 h
  cap: 4.9 h for fixed time, then 3.6 h for lm-eval.
- **Order.** Fixed time runs first, then lm-eval.
- **Purpose.** This avoids idle GPU time between the pilot and P2.
- **Review.** The pilot is still reviewed by hand.

## Update, 2026-11-17 01:15 UTC (incident 5: pod restart, about 1.9 idle hours)

| Block | Hours |
|---|---|
| Used by 01:12 UTC | 24.05 |
| Rest of the controls (4 Opus) and anchors (3) | 0.5 |
| P1 downstream under A3 (23 checkpoints) | 1.1 to 1.5 |
| A2 and A1 pilot | 0.8 |
| P2 fixed time (34 runs) | 4.9 |
| P2 lm-eval (39 checkpoints, not piloted) | 3.6 |
| **Projected total** | **about 35.2 (34.9 to 35.6)** |

- **Against the cap.** This is at or just over the 35 h cap. The r6 guard stops before lm-eval if the
  used hours plus 3.6 would exceed the cap. Fixed time is not affected.
- **Author's decision:** requested.

## Author decision (2026-11-17, about 01:15 UTC)

- **Cap raised to 36.5 A40-hours** ("Raise to 36.5 h"). The scope is unchanged: P2 fixed time and
  P2 lm-eval both run.
- **Where it is applied.** `/root/phase6_cap_hours` now holds 36.5. r6 reads this file at each
  budget guard.

## Update, 2026-11-17 10:05 UTC (incident 6: about 1.9 idle hours)

- **Done:** P2 fixed time, all 34 runs.
- **Hours used when P2 lm-eval restarted (r7):** 32.94.
- **Remaining:** P2 lm-eval, 3 cross-check models then 39 checkpoints, estimated at 3.6 h and not
  piloted.
- **Projected total:** about 36.5 h, at the cap. The r7 watchdog stops lm-eval at the cap. The
  runner skips finished outputs, so it can resume if the cap is raised.
- **Author's decision:** requested.

## Author decision (2026-11-17, after 10:05 UTC)

- **Cap raised to 37.5 A40-hours** ("Raise to 37.5 h"). The scope is unchanged: P2 lm-eval runs in
  full.
- **Where it is applied.** `/root/phase6_cap_hours` now holds 37.5. The r7 watchdog reads it every
  minute.

## Update, 2026-11-17 10:31 UTC (incident 7)

- **P2 lm-eval:** stopped at the cross-check gate.
- **Hours used:** 33.4. The cap is 37.5.
- **If A4 is approved:** the 39-checkpoint run (7 tasks each) is estimated at 2.3 to 3.6 h, for a
  projected total of 35.7 to 37.0 h.
- **If the author skips it:** GPU work ends at about 33.4 h.

## Author decision (2026-11-17, 11:45 UTC)

- **A4 approved:** "Approve A4, run it".
- **Measured rate:** 3.0 min per checkpoint (probe at 11:14 UTC).
- **Estimate:** the 39-checkpoint lm-eval takes about 2 h, for a projected total of about 36.7 h
  against the 37.5 h cap.

## Final (2026-11-17, 12:40 UTC): Phase 6 complete

- **Last stage.** `PHASE6_P2_DONE` was stamped at 12:40:20 UTC. The 39-checkpoint lm-eval finished in
  54 min (11:45:53 to 12:40:20), at 1.4 min per checkpoint, against the probe's estimate of 3.0
  min.
- **Pod hours at completion:** 35.52 (from about 01:09 UTC on 2026-11-16), inside the author's cap
  of 37.5 h.
- **Idle time inside that total:** about 12 h, across four incidents:
  - 6.9 h while a decision was awaited (incident 4);
  - 1.9 h from the pod restart (incident 5);
  - 1.9 h from lm-eval stopping while the watchers had failed (incident 6);
  - about 1.3 h between the gate stop and the A4 approval (incident 7, 10:17 to 11:45 UTC), not
    counting a 4-min BLiMP diagnostic and a 3-min timing probe.
- **The pod is idle now and still billing.** Stopping it is the author's decision.
  - Every result except checkpoints (`*.pt`) and compile caches is copied to the Mac and
    checksum-verified. That includes the pod-local console logs.
  - The checkpoints stay on the network volume.
