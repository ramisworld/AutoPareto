# 23. Phase 3: E1 token-response experiment (24 runs, seeds 6–8)

**Status:** complete, 2026-11-07. 24 of 24 runs passed their integrity gates on the first attempt.
- SSH endpoints and pod identifiers are not recorded here.
- Nothing in the protocol was changed.
- No optional experiment was started: no E1b, scaling, search or G4 retraining.

## 1. What ran

| Item | Value |
|---|---|
| Plan | `evaluation/camera_ready_e0_e1_plan_amendment1.json`, sha256 `89f29015…` (amendment 1, unchanged since 2026-11-04) |
| Executor / analysis | `scripts/run_camera_ready_e0_e1.py` `aee0ab38…` / `scripts/analyze_camera_ready_e1.py` `1be6e8e5…`, both as pre-registered |
| E0 precondition | The E0 verdict recorded on 2026-11-04 under this plan hash: PASS. The executor refuses to re-run E0 ("E0 verdict already recorded"). |
| Pre-launch validate-only | 18 ok, 0 FAIL, 0 POD_ONLY (details below) |
| Hardware | A40 `GPU-3adf8a47…`, driver 595.91.07, CA-MTL-1 (the fourth camera-ready pod) |
| Launch / finish | 2026-11-07 11:59:52Z / 16:45:19Z: **4.76 A40-h** (stage start 12:00:01Z to finish 16:45:19Z) (the plan estimated 6.44) |
| Outcome | 24/24 completed; 0 retries; 0 integrity failures; `reported_failed: []` |
| Cost | ≈ $2.09 at the plan's $0.44/h (confirm the rate in the console). The pod has been idle since 16:45 and still bills until it is stopped. |

The validate-only checks covered:
- the 20 implementation hashes;
- the 13 data files and the held-out shard;
- uv.lock `262a4d55…`, Python 3.10.18 and torch 2.9.1+cu128;
- the flash-attn3 cache at `e29f138f…`, the same revision Pair-A recorded.

The output is in `results/camera_ready_2026/amendment1/camera_ready_E1_amendment1_prelaunch_validate.json`.

**Deviation to disclose (hardware).**
- The plan's `hardware_note` names `GPU-4702993a…`, the pod-1 card on which E0 passed. That pod no longer exists, so E1 ran on `GPU-3adf8a47…` with the same driver.
- E0 was not repeated on this card, because the executor allows one E0 per plan.
- Fixed-token cells do not depend on throughput.
- Training throughput on this card matched the earlier epochs:

| Recipe | E1 range (24 runs) | Pod-1 E0 | July/September panel |
|---|---|---|---|
| AR-ref | 288.5–289.7k tok/s | 287.0k | 289.8k |
| AP-Q | 522.8–524.9k tok/s | 521.8k | 522.7k |

- The host's failed CPU-noise screen (doc 21) therefore did not slow training.
- AP-Q at 201.3M tokens: E1 seeds 6–8 give held-out 1.1410 ± 0.0036; E0 seed 3 gave 1.1391.

**Not repeated:** the CPU token-stream replay.
- The stream is fixed by the 13 SHA-verified data files and the hash-verified loader code.
- It replayed exactly on 2026-11-04 (`180e0a5b…`, seeds 3 and 6).

## 2. Pre-registered results (`results/camera_ready_2026/amendment1/E1/analysis/e1_analysis.md`)

Δ = BPB(AP-Q) − BPB(AR-ref), paired by seed. The interval is the 95% t-CI with n = 3 (t = 4.303).

| Cell | Tokens (AR-ref / AP-Q) | Δ held-out (primary) | Δ validation | Per-seed held-out Δ |
|---|---|---|---|---|
| 88.6M tokens, 676 updates | 88.6M / 88.6M | **+0.0385** [+0.0222, +0.0547] | +0.0418 [+0.0343, +0.0493] | +0.0437, +0.0311, +0.0405 |
| 157.3M tokens, 1,200 updates | 157.3M / 157.3M | **+0.0396** [+0.0226, +0.0565] | +0.0430 [+0.0341, +0.0519] | +0.0454, +0.0321, +0.0412 |
| 201.3M tokens, 1,536 updates | 201.3M / 201.3M | **+0.0415** [+0.0287, +0.0544] | +0.0445 [+0.0364, +0.0527] | +0.0465, +0.0362, +0.0418 |
| 300 s fixed time | 88.0M / 158.1M | **−0.0022** [−0.0205, +0.0160] | −0.0044 [−0.0124, +0.0036] | +0.0037, −0.0105, +0.0001 |

| Rule (from the plan) | Held-out | Validation |
|---|---|---|
| H-E1a: Δ increases with tokens | **inconclusive**. The means rise strictly; Δ(201.3M) − Δ(88.6M) = +0.0031 [−0.0017, +0.0078], and all three seeds are positive. | **supported** |
| H-E1b: Δ ≤ 0 at 88.6M (matched discovery-scale exposure) | **contradicted** (AR-ref better) | **contradicted** |
| Crossover in [88.6M, 201.3M] | none: Δ > 0 throughout; P = 0 in all 27 seed resamples | none |
| Outcome table | **"exposure explains the fixed-time advantage"** | same |
| 300 s vs seeds 0–2 (in-process validation) | n/a | −0.0044 vs −0.0067; Welch +0.0023 [−0.0045, +0.0090] → **consistent** |
| Schedule effect (300 s vs the token cell with similar exposure) | not detected. AP-Q −0.0001 [−0.0014, +0.0012]; AR-ref +0.0019 [+0.0005, +0.0033], below the 0.005 margin. | n/a |

## 3. Exploratory analysis: exposure versus architecture (not pre-registered)

This analysis was written after the plan and before any E1 result was seen.
- Script: `scripts/analyze_e1_exposure_decomposition.py`.
- Outputs: `results/camera_ready_2026/amendment1/E1/analysis_exploratory/`.
- Inputs: the same run selection, read through the pre-registered script's functions.
- Cross-check: its paired Δs match `e1_analysis.json` with max |diff| = 0.

**Method.** For each seed, each recipe's three fixed-token BPBs define a curve, piecewise linear in ln(tokens). The 300 s gap G is split at the tokens each recipe actually processed in 300 s (T_A for AP-Q, T_R for AR-ref):
- **exposure** = AR-ref's BPB at T_A minus its BPB at T_R;
- **architecture** = AP-Q − AR-ref at T_A;
- **schedule** = the residual.

The three components sum to G exactly. AR-ref's 300 s exposure, 88.0M tokens, is 0.7% below the 88.6M cell, so that point is a short linear extension of the first segment.

| Held-out BPB | Mean [95% t-CI] | Seeds 6, 7, 8 |
|---|---|---|
| Observed 300 s gap | −0.0022 [−0.0205, +0.0160] | +0.0037, −0.0105, +0.0001 |
| **Exposure** (AP-Q's 1.80× tokens) | **−0.0406** [−0.0423, −0.0388] | −0.0403, −0.0414, −0.0401 |
| **Architecture at equal tokens** | **+0.0396** [+0.0227, +0.0565] | +0.0454, +0.0321, +0.0412 |
| Schedule residual | −0.0012 [−0.0019, −0.0006] | −0.0015, −0.0012, −0.0010 |
| Interaction (architecture gap at T_A minus at T_R) | +0.0011 [−0.0002, +0.0025] | small |

The validation BPB split agrees:
- exposure −0.0474 [−0.0494, −0.0453];
- architecture +0.0430 [+0.0341, +0.0520];
- schedule −0.0001.

**The same result as two ratios.**
- AP-Q's steady-state throughput is ρ = **1.81×** AR-ref's [1.80, 1.82]. That gives it 1.80× the tokens in 300 s.
- To match AR-ref's 300 s quality, AP-Q needs k = **1.79×** as many tokens [1.29, 2.28] on held-out data, or 1.70× [1.54, 1.86] on validation.
- So ρ ≈ k: the faster architecture's extra tokens buy back almost exactly what it loses per token.

**Replication of the Phase-1 panel.** At 201.3M tokens, the panel's +0.0420 [+0.0297, +0.0544] on seeds 3–5 reappears as +0.0415 [+0.0287, +0.0544] on fresh seeds 6–8, run on a different card.
- Welch difference, E1 − panel, held-out: −0.0005 [−0.0120, +0.0110].
- Validation: −0.0005 [−0.0075, +0.0064].

## 4. Consistency across seeds

- **Matched tokens:** AP-Q is worse on all 9 seed × budget pairs. Within-recipe SD is at most 0.0045 in every cell, about one ninth of the Δ.
- **Shape:** each seed's held-out Δ curve rises strictly from 88.6M to 201.3M, but the rise (+0.003) is within the n = 3 interval.
- **Fixed time:** the sign is not stable on held-out data (1 of 3 seeds favours AP-Q). On validation all 3 seeds favour AP-Q, by 0.002–0.008.
- **Why the two metrics differ at 300 s:** the held-out − validation offset depends on the recipe, as already noted in the camera-ready changelog §3. At matched tokens both metrics agree on sign and size.

## 5. Conclusions for the paper (the reviewers' main question)

1. **The fixed-time quality edge comes entirely from token exposure.**
   - At equal tokens and updates, AP-Q is worse than AR-ref by about 0.04 BPB at every budget tested, including the 88.6M-token, 676-update discovery scale.
   - The pre-registered outcome is "exposure explains the fixed-time advantage".
2. **The architecture is less sample-efficient.** The 300 s comparison is a near-cancellation of two ≈0.04 effects:
   - extra tokens: −0.041;
   - architecture at equal tokens: +0.040.
   - The schedule term is about 0.001.
   - AP-Q processes 1.81× the tokens per second but needs about 1.7–1.8× the tokens for equal quality.
3. **At 300 s the two recipes are statistically tied on seeds 6–8.**
   - Held-out: −0.002 [−0.021, +0.016].
   - Validation: −0.004 [−0.012, +0.004].
   - Both are consistent with the earlier seeds 0–2 result (−0.0067). The submitted claim that AP-Q is better at fixed time should be stated as small and not robust.
4. **No crossover in range.** The matched-token deficit grows slightly with tokens (+0.003 from 88.6M to 201.3M; supported on validation, inconclusive on held-out data). The reversal is not a large-budget artefact.
5. **The reversal replicates** on fresh seeds and a different A40, matching the panel to within 0.0005 BPB.

AP-Q's efficiency results (about 2× decode, lower memory) are measured elsewhere and are not affected by E1.

## 6. Caveats

- n = 3 seeds per cell, so the t-intervals have df = 2 and are wide. The decomposition's architecture interval inherits this.
- The decomposition (section 3) is post hoc. Its exposure term assumes AR-ref's fixed-token curve applies at AP-Q's exposure, with interpolation in ln(tokens).
- The 300 s token counts depend on hardware. They are reported per run: AR-ref 87.9–88.1M; AP-Q 157.7–158.3M, which is 1,203–1,208 updates.
- E1 ran on a different card from E0 (see section 1).

## 7. Evidence

**Committed with this doc** (`results/camera_ready_2026/amendment1/`, about 22 MB):
- `E1/`: 303 files. These are every run's record, metrics, logs, isolate sources, evaluations, stage events, the stage manifest and the execution summary. Checkpoints and compile caches are excluded, as for E0.
- All 303 files match the pod copy by SHA-256 (`E1/SHA256SUMS`); the three `/workspace` logs match too (`E1_workspace_logs.SHA256SUMS`).
- `E1/analysis/` (pre-registered) and `E1/analysis_exploratory/`: JSON, Markdown, `e1_tables.tex`, and the PDF/PNG figures `e1_delta_vs_tokens.*` and `e1_exposure_vs_architecture.*`.
- The console log, launch time and pre-launch validation, copied from `/workspace`.

**Not committed:**
- **Checkpoints** (78 `.pt` files, 9.82 GB: `final.pt` in all 24 runs plus 3 milestones in each of the 18 token runs; hashes in `E1_checkpoints.SHA256SUMS`). They are kept off Git in verified Mac storage, `autopareto_pod_backups/pod4_GPU-3adf8a47_live/workspace/autopareto-camera-ready/results/camera_ready_2026/amendment1/E1/runs/`.

- **Bytecode:** 24 `isolate/**/__pycache__/*.pyc` files are listed in `E1/SHA256SUMS` but are git-ignored. They are generated, and they are in the Mac backup, so `shasum -c` on a fresh clone reports exactly these 24 as missing.

**Off-pod backup verification (2026-11-08 00:00 UTC).**
- The 15-minute pull loop reached `all_verified: true` at 22:35 UTC (52,907 workspace files and 224 `/var/lib/autopareto` files) and stayed clean on four later passes.
- An independent check hashed every file on the pod's `/workspace` (79,401 files) and compared it with the Mac copy: 52,907 of 52,907 matched by SHA-256, with none missing. All 78 E1 checkpoints are among them.
- The remaining pod files are the rebuildable `.venv` (26,203 files, recreated from `uv.lock`) and 291 files in `mirror/` and `pod_backups/`. Those are earlier-pod copies restored in Phase 1, and all 223 distinct hashes are present in the Mac's pod-2 and pod-3 backups.

**Incident (no effect on E1).**
- The Mac slept on battery from about 14:50 to 20:30 UTC. The event watcher and the backup pull paused during that time.
- From about 18:16 to 21:00 UTC, SSH to the pod failed: first time-outs, then kex resets.
- E1 itself ran to completion on the pod, as its stage events show.
