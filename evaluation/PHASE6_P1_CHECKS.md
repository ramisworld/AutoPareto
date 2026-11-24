# Phase 6 P1 checks (2026-11-16)

## Drift controls (pre-registered rule, `PHASE6_PREREG.md` section 4)

Seed 6, 201.3M tokens. Pod 5 (A40 GPU-6de81321) against the reused pod-4 values.

| Recipe | Held-out BPB pod 4 | Held-out BPB pod 5 | Difference (limit 0.005) | Throughput pod 4 | Throughput pod 5 | Difference (limit 3%) |
|---|---|---|---|---|---|---|
| AR-ref (E1) | 1.09473 | 1.09677 | +0.0020 | 288,689 tok/s | 288,869 tok/s | +0.06% |
| GPT A22 (E1) | 1.14127 | 1.14144 | +0.0002 | 522,838 tok/s | 522,854 tok/s | +0.00% |
| Opus A23 (Phase 5A) | 1.15605 | 1.15525 | -0.0008 | 655,399 tok/s | 656,625 tok/s | +0.19% |

- **Result.** All three are within both limits, so the reused E1 and Phase 5A matched-token values
  stand.
- **Checkpoints.** The pod-5 checkpoints differ in hash from the pod-4 ones, as expected:
  training is not bitwise deterministic.

## Throughput reliability

- **Measure.** Excess time from steps more than twice the run's median, after step 10. This is A1's
  timing-anomaly measure.
- **Scope.** 26 completed P1 runs, plus 3 drift controls.

| Run | Finding |
|---|---|
| GPT A24 seed 6 | 32.6 s of excess (incident 2). Its throughput is unreliable, about 12% low. Its held-out BPB is valid. |
| All others scanned (20 runs by 12:30 UTC) | 0.4 s of excess or less |

The last 6 P1 runs and the drift controls are checked again before the final tables are built.

## Not run under the frozen plan

- **MOTPE seed 8.** Both attempts failed on a volume error (incident 3). Covered by amendment A2.
