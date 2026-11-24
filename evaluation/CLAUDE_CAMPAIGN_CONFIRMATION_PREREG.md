# Pre-registration: selecting and confirming the Claude-campaign winner

Frozen 2026-11-02, before any Claude campaign attempt exists. The commit that adds this file and its
SHA-256 are recorded in `audit/camera_ready_2026/12_GPU_READINESS_REPORT.md`. Any later change
is a disclosed deviation.

## 1. Pool
The 25 attempts of the single-arm Contract-T campaign `pair-a-claude-opus55-t-seed3`
(Claude Opus 5.5, xhigh, discovery seed 3). Eligible: status `completed`, quality valid,
inference `stable` with correctness passed, i.e. the controller's own `pareto_eligible` flag.
Failed, harness-invalid or inference-incompatible attempts are reported but never selected.
Duplicates by candidate source SHA-256 are collapsed to the earliest attempt.

## 2. Roles (the same four roles as the earlier AutoPareto primary confirmation)
1. **Winner (primary):** lowest protected validation BPB. Ties go to the earlier attempt.
2. Fastest cached decode within 1% of the winner's validation BPB.
3. Lowest peak inference memory within 1% of the winner's validation BPB.
4. Largest remaining exclusive hypervolume contribution on the campaign's Pareto front.

Only the winner is confirmed by default. Roles 2–4 are reported as discovery-stage results and are
confirmed only with separate approval. If no attempt is eligible, nothing is confirmed and the
campaign is reported as having no valid improvement.

## 3. Confirmation design
| Factor | Levels |
|---|---|
| Recipe | Winner (exact source bytes); Pair-A seed-3 baseline `historical_reference/baseline_source/train.py` (sha `b2fb8788…`), the baseline the campaign was scored against |
| Contract | T: 300 s fixed time (the discovery contract); N: fixed tokens at N_STAR = 85,458,944 (the baseline's 300 s exposure, i.e. matched exposure) |
| Seeds | 6, 7, 8: fresh, not used in discovery (3) or in selection |
| Harness | The same protected Pair-A worker, evaluator and bracketed inference benchmark as discovery, with no agent |
| Runs | 2 recipes × 2 contracts × 3 seeds = 12 |

Optional, approval required: the Codex G4 T-arm best-quality candidate on the same 12-run design,
for a same-harness agent comparison.

## 4. Metrics
Protected validation BPB (primary); held-out BPB on `shard_06541.parquet` with the panel
evaluator; the panel's downstream suite; bracketed cached-decode throughput; peak inference memory;
parameter count; tokens and optimizer updates.

## 5. Analysis and decision rule
For each contract: paired Δ = BPB(winner) − BPB(baseline) per seed, with mean and t-based 95% CI
(n = 3, t = 4.303).
* **Confirmed improvement:** CI entirely below 0.
* **Confirmed regression:** CI entirely above 0.
* **Inconclusive:** otherwise. This is a possible and reportable outcome at n = 3.

T and N are reported separately. If T improves and N does not, that is reported as a
contract-dependent gain, the same pattern as the paper's reversal. Discovery-stage values
(seed 3) are never described as confirmed. Inference and memory ratios are reported with their
bracketed CIs. Downstream results are descriptive.

## 6. Integrity
* A failed run is re-run once on the same seed, otherwise reported as failed.
* No seeds are added in response to results.
* Results are written under a new directory, never into existing evidence.

Execution path: a parametrised version of `tools/run_t_smoke.py` / `run_n_smoke.py` (source, seed,
contract) on the same protected worker. It must pass a local dry run and receive approval before
launch. Estimated 12 × ~11 min ≈ 2.2 A40-h, plus held-out and downstream evaluation ≈ 0.5 h:
about 2.7 A40-h, ≈ $1.2.
