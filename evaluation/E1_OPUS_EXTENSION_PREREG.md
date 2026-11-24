# Pre-registration: E1 extension with the selected Opus winner

Frozen 2026-11-02, before any Opus 5.5 campaign attempt exists. This is a separate study. It does not
alter the E1 plan (`evaluation/camera_ready_e0_e1_plan.json`, sha256 `806a6397…c829340`), which runs
exactly as registered. The commit that adds this file and its SHA-256 are recorded in
`audit/camera_ready_2026/12_GPU_READINESS_REPORT.md`. Any later change is a disclosed deviation.

## 1. Question
Across the E1 budgets and paired seeds, where does the recipe selected from the Opus 5.5 fixed-time
campaign fall relative to the AutoResearch reference and AP292? Does its ranking depend on the
training contract, as AP292's does?

## 2. Recipes. The two baselines are different recipes and are not interchangeable.

| Role | Recipe | Source | Harness |
|---|---|---|---|
| AutoResearch comparator (**AR-ref**) | exp 68 | `results/candidates/autoresearch-diagnostic5-seed42-20260724/0005-train.py`, sha `b3a961ee…` | E1 runs, reused unchanged |
| **AP292** (AP-Q, exp 134) | Phase-1 `autopareto_292` | `results/candidates/autopareto-seed0-20260731/0022-train.py`, sha `55a7fd8b…` | E1 runs, reused unchanged |
| **Opus winner** | role 1 ("winner") of `CLAUDE_CAMPAIGN_CONFIRMATION_PREREG.md` (sha `adf8ce71…`) | its exact Pair-A source, mapped by `scripts/pair_a_to_phase1_source.py` | new runs, same executor path as E1 |

The **Pair-A baseline** (`historical_reference/baseline_source/train.py`, sha `b2fb8788…`) is not a
comparator here. It is the comparator only in the Opus confirmation study, in the Pair-A harness.
It is the Phase-1 panel baseline (sha `428e22f1…`) with the in-script evaluation removed, so it is
not AR-ref.

## 3. Source mapping (fixed before the winner is known)
`scripts/pair_a_to_phase1_source.py` restores only what Pair-A removed: the `evaluate_bpb` import,
the final evaluation block and the `val_bpb` print. It also deletes bare `contract_progress` /
`contract_update_completed` hook statements, whose return values must be unused. The result goes to
the unchanged protected `phase1_source_transform`, exactly as for AR-ref and AP292. No hand edits
are permitted.

Gates, all local and checked at freeze time:
* **A1.** Applied to the Pair-A baseline, the adapter reproduces the panel baseline byte for byte
  (sha `428e22f1…`). Passed.
* **A2.** The protected transform accepts that result. Passed.
* **A3 (feasibility).** 23/23 Codex Pair-A fixed-time candidates are accepted after mapping.
* If the transform or adapter refuses the Opus winner, the extension is reported as not
  executable for that recipe. There is no fallback recipe and no edit.

Budgets: if the winner's `TOTAL_BATCH_SIZE` does not divide a token budget, that cell uses the
largest multiple not exceeding it. The exact tokens and updates are reported.

## 4. Design

| Factor | Levels |
|---|---|
| Budgets | 88,604,672; 157,286,400; 201,326,592 tokens; 300 s fixed time. These are E1's cells, with LR progress following E1's rule: token fraction in token cells, time fraction in the 300 s cell. |
| Seeds | 6, 7, 8, paired with E1. They are fresh for the winner: not used in discovery (seed 3) or selection. |
| New runs | Opus winner × 4 budgets × 3 seeds = 12 |
| Session-drift control | First in the extension block: AR-ref at 201,326,592 tokens, seed 6, and AR-ref at 300 s, seed 6 (2 runs), compared with the same E1 cells |
| Held constant | Executor functions, launchers, evaluator, validation split, held-out shard `shard_06541.parquet`, data identity gate, A40 environment (E0-verified) |

## 5. Outcomes
* **Primary:** held-out BPB, matching E1.
* **Also reported:** validation BPB, tokens, updates, steady-state seconds, peak training memory.
* On the 201,326,592-token checkpoints:
  * stabilized cached-decode throughput and peak inference memory, once per recipe on the seed-6
    checkpoint, since they are deterministic in architecture;
  * the panel downstream suite on all nine checkpoints (3 recipes × seeds 6–8).

## 6. Analysis (fixed)
* Per budget, paired Δ = BPB(winner) − BPB(AR-ref) and Δ = BPB(winner) − BPB(AP292), each with mean
  and t-based 95% CI (n = 3, df = 2). All 8 intervals are reported, with no selection among them.
* **Primary estimand:** Δ(winner − AR-ref) at 201,326,592 tokens on held-out BPB, the paper's
  matched-exposure comparison.
* **Secondary:**
  * a crossover against AR-ref, using linear interpolation of mean Δ in ln(tokens) with the exact
    27-resample seed bootstrap, as in E1;
  * the 300 s cell compared with the token cells, i.e. contract dependence.
* Holm-adjusted p-values across the 8 comparisons are reported as a secondary column.
* Wording:
  * A CI excluding 0 is a "confirmed difference at that budget"; otherwise "inconclusive".
  * Discovery-stage (seed 3) values are never called confirmed.
  * Downstream results are descriptive.
* **Drift control:** if either control run differs from its E1 cell by more than 0.005 BPB (validation
  or held-out), or the token-cell control's tokens or updates differ, every winner-versus-comparator
  contrast is labelled session-confounded and reported descriptively only.

## 7. Integrity and order
* Runs only after E0 passes, E1 completes, and the winner is fixed by the frozen selection rule.
* Each run is retried once on the same seed, otherwise reported as failed. No seeds are added.
* Results go to a new directory under `results/camera_ready_2026/E1X/`; nothing in E1 or earlier
  evidence is touched.
* Implementation: a thin entry point that reuses the E1 executor's functions with a separate plan
  file. It must pass validate-only, and the GPU work needs approval.
* Estimated cost: 14 training runs ≈ 3.7 A40-h, plus inference and downstream ≈ 1 h; total ≈ 4.7 A40-h,
  ≈ $2.1.
