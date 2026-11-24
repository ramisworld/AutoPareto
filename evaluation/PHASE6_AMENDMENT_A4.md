# Phase 6 amendment A4: the lm-eval cross-check gate excludes BLiMP

- **Status:** written before the author's decision, while the stage was stopped. The approval is
  recorded in `evaluation/phase6_amendment_a4_plan.json`, which is written only after the author
  approves.
- **Parent plan:** sha256 dc1e605e..., unchanged.
- **A1 plan:** sha256 2b9fdc3e..., unchanged.

## 1. What happened

- **The run.** `camera_ready_phase6_p2_r7.sh` ran A1's lm-eval cross-check from 10:05 to 10:17 UTC
  on 2026-11-17. It used three checkpoints, all seed 6: AR-ref, GPT A22 and Opus A23.
- **ARC-Easy, PIQA and LAMBADA.** lm-eval matched the frozen evaluator exactly: 0.000 points on all
  9 comparisons.
- **BLiMP.** lm-eval was higher by 1.007, 1.140 and 1.045 points. The tolerance is 0.5.
- **What A1 did.** It stopped the stage before the 39-checkpoint run, as written.
- **The report.** `lmeval/crosscheck_report.json` is kept unchanged, and its sha256 is recorded in
  the A4 plan.

## 2. Cause

**How the two evaluators score BLiMP**
- **Frozen evaluator.** Scores `("", sentence)`. A pair is correct only if the good sentence has a
  strictly higher log-likelihood.
- **lm-eval 0.4.13.** Its BLiMP template uses the default target delimiter `" "`, so it scores
  `("", " " + sentence)`. Accuracy is an argmax, so a tie counts as correct.
- **Common to both:** the dataset, `nyu-mll/blimp` at revision 877fba08, and the macro average over
  67 configs.

**Diagnostic**
- Code: `scripts/phase6_a4_blimp_diagnostic.py`. Result: `evaluation/phase6_a4_blimp_diagnostic.json`.
- One scoring pass of GPT A22 seed 6 over all 67,000 pairs reproduces both reported values within
  0.0015 points:

| Rule | Recomputed | Reported |
|---|---|---|
| Frozen (no space, tie counts as wrong) | 0.725463 | 0.725478 |
| lm-eval (leading space, tie counts as correct) | 0.736896 | 0.736881 |
| Leading space only (tie counts as wrong) | 0.736791 | n/a |

- **Which difference matters.** The leading space causes almost all of the gap. Only 7 of 67,000
  pairs tie.
- **The residual.** The 0.0015 points left over is bf16 batching noise, about one pair.

**Conclusion.** The BLiMP gap comes from a scoring convention, not from an error in the harness, the
checkpoints or the frozen BLiMP results.

## 3. What A4 changes

- **The gate.** It compares ARC-Easy, PIQA and LAMBADA only, with the same 0.5-point tolerance.
  - lm-eval's BLiMP values stay in the cross-check outputs.
  - They are reported with the explanation above.
- **Code.**
  - `scripts/run_phase6_a4.py` calls the A1 runner's `execute_lmeval` unchanged.
  - It passes a copy of the A1 plan whose `crosscheck_metrics` leave out BLiMP.
  - It also writes its own A4 gate report next to the original one.
- **Unchanged:**
  - the 39 checkpoints and their tasks: ARC-Challenge, HellaSwag, WinoGrande, OpenBookQA, SciQ,
    BoolQ and WikiText;
  - the dataset cache and its manifest;
  - the environment, with `KERNELS_CACHE` set as in incident 6;
  - the cap watchdog.
- **BLiMP in the paper.** The frozen evaluator's values, already computed for all 39 checkpoints,
  stay the paper's BLiMP numbers. The 39-checkpoint lm-eval run does not include BLiMP.
