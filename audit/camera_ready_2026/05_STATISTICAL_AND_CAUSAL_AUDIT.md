# 05 — Statistical and Causal Audit

**Covers:** the ranking reversal, confounding, training variance, candidate-selection bias, and the limits of causal inference. Everything here is computed offline from existing records. No new training was run.

**Sources**
- `SNAPDB` = `results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db`
- `P1JSON` = `…/phase1/phase1_equal_token_results.json`
- `STATS` = `supporting/statistical_audit.json`
- `CLAIMS` = `supporting/claims_recomputation.json`
- Scripts are in `supporting/scripts/`.

**Labels**
- OBSERVED: read from a record.
- DERIVED: computed by this audit.
- INFERRED: an interpretation.
- n = 3 confidence intervals use t with 2 degrees of freedom (critical value 4.303).

## 0. Bottom line

1. **The reversal is real, and the contract explains it better than seeds or data do.**
   - At equal wall-clock time on fresh seeds 0–2, AP292 beats the AutoResearch comparator (AR) by 0.0067 BPB on all three seeds.
   - At equal tokens (201M, seeds 3–5), AR beats AP292 by 0.045 BPB on the *same* validation data, and by 0.042 on held-out data.
   - So neither the change of evaluation data nor the change of seeds explains the flip.
   - What remains is the change of training contract. Tokens and optimizer updates moved together, so existing data cannot separate them.
2. **The paper's discovery-stage comparison is confounded by seed.** The baseline varies by 0.024 BPB between seed 0 and seed 42, while the reported discovery gap is 0.0017.
3. **Within-campaign selection is noise-dominated near the top.** The best-to-second gap is at most 0.0003 BPB, against a seed SD of about 0.002. Every re-run candidate scored worse than its discovery value, by 0.0014–0.0087.
4. **The candidate selection skipped a recipe that may undermine the "no quality gain" reading.** The AutoPareto seed-42 campaign found full-size (50.3M-parameter) recipes, exp 109 and its siblings.
   - They scored 1.0940 on *fewer* tokens than AR (84.1M vs 88.9M), at the same seed.
   - This is an exposure-neutral quality candidate that was never confirmed.
5. **Pair-A supports no contract effect.** There are five separate campaign lineages, each one adaptive trajectory per arm, run under drifting protocols.
   - Batch-size reduction shows up in both arms in most lineages.
   - The "fixed-token agent never changed batch size" pattern occurs in exactly one lineage.

---

## 1. The quality-ranking reversal

### 1.1 What changed between discovery and confirmation (AP292 = exp 134 vs AR = exp 68)

| Variable | Discovery (archive) | Fixed-time re-runs (not in paper) | 201M matched-token panel (paper) | Source |
|---|---|---|---|---|
| Stopping rule | 300 s wall clock | 300 s | 201,326,592 counted tokens | `SNAPDB experiments.training_seconds`; `P1JSON token_budget` |
| Tokens: AP292 / AR | 159.8M / 88.9M | 152.2–159.3M / 88.5M | 201.3M / 201.3M | `SNAPDB` ids 134, 68; `confirmation_runs` → `training_runs` |
| Updates: AP292 / AR | 1,219 / 678 (DERIVED, tokens ÷ 131,072) | ~1,197 / 675 | 1,536 / 1,536 | `P1JSON optimizer_steps` |
| Batch | 131,072 for both | same | same | frozen sources (`02_EXPERIMENT_REGISTRY.csv`) |
| Seeds | 0 (AP292) vs 42 (AR); not matched | 0, 1, 2 (paired) | 3, 4, 5 (paired) | |
| Evaluation data | validation split | validation | validation **and** held-out shard_06541 | `P1JSON final_validation_bpb`, `heldout_bpb` |
| LR progress variable | elapsed-time fraction | elapsed-time fraction | counted-token fraction | paper §1.2, Table 2; `P1JSON` |
| Session, protocol | discovery harness, Aug 3–4 | confirmation campaigns, Aug 11–12 | phase-1 runner, Aug 24–27, `autopareto_inference_stabilized_v2` | `SNAPDB` timestamps; `P1JSON` |

**Constant throughout:** the frozen recipe source (`candidate_sha256`), hardware class (A40), dataset, and tokenizer.

### 1.2 Evidence

DERIVED, from `STATS reversal_table`, `fixed_time_AP292_minus_AR_val_seeds012` and `matched_token_AP292_minus_AR`.

| Comparison (AP292 − AR) | Contract | Seeds | Data | Δ BPB | Per-seed values | 95% CI |
|---|---|---|---|---|---|---|
| Discovery archive | 300 s | 0 vs 42 | val | −0.0017 | one run each | n/a |
| Fixed-time re-runs | 300 s | 0, 1, 2 | val | **−0.0067** | −0.0054, −0.0051, −0.0096 | [−0.0130, −0.0003] |
| Fixed-time checkpoints | 300 s | 1 vs 2 | held-out | −0.0041 | one run each | n/a |
| Matched tokens | 201M | 3, 4, 5 | val | **+0.0451** | +0.0440, +0.0467, +0.0445 | [+0.0415, +0.0486] |
| Matched tokens | 201M | 3, 4, 5 | held-out | **+0.0420** | +0.0387, +0.0477, +0.0397 | [+0.0297, +0.0544] |

Row sources:
- Fixed-time re-runs: `SNAPDB confirmation_runs` rows 13–15 (exp 134) and 7–9 (exp 68). Run 8 trained to completion; its *inference* was incompatible, which does not affect BPB.
- Fixed-time checkpoints: `downstream_evaluation_runs` 4 (exp 292) and 2 (exp 77).

**Reconciliation with `07`.** `07` quotes the 300-second contrast as "AP292 better by 0.0054". That is an **unpaired** average: AR uses ids 68 (seed 42), 75 and 77 (seeds 0 and 2), against AP292 ids 291–293 (seeds 0–2). The seed-paired estimate above (−0.0067, seeds 0–2 for both, including AR's seed-1 run whose training completed) is preferred. Both have the same sign and fall inside the same interval.

### 1.3 Explanation by explanation

**(a) More vs. matched training tokens (exposure).**
- *Supporting:*
  - The sign flips exactly when tokens are equalized (§1.2).
  - Across five recipes, the change in validation BPB from the fixed-time runs to the 201M runs tracks the token ratio (§1.4; Pearson −0.92 against log2 ratio).
  - AR gains 0.069 from 2.28× tokens; AP292 gains 0.017 from 1.28× tokens.
- *Contradicting:* none found.
- *Already calculable:* the size of the flip and the per-recipe exposure response (§1.4).
- *Unidentified:*
  - Tokens and updates cannot be separated (see d).
  - The crossover token count, where AR overtakes AP292, is unknown somewhere between 88.5M and 201M.
- *Smallest decisive experiment:* **E1** in `08_EXPERIMENT_PLAN.md`. Train AR and AP292 under the fixed-token contract at 88.5M, about 157M, and 201M tokens on seeds 0–2. That is 18 runs, about 4.5 A40-hours. Same seeds, same validation data, only tokens varied.

**(b) Seed variability.**
- *Supporting:* seed main effects are large (§2.1). For the baseline, seed 3 is about 0.015 easier and seed 42 about 0.023 easier than the mean of seeds 0–4.
- *Contradicting:*
  - Both comparisons are paired, so seed main effects cancel.
  - Within-recipe seed SDs are 0.0008–0.0027 (fixed time) and 0.0004–0.0060 (201M).
  - A seed explanation needs a seed-by-recipe interaction of about 0.05, which is 10–25 times the observed within-recipe spread.
  - All 3 + 3 paired differences have consistent signs.
- *Unidentified:* no fixed-time runs at seeds 3–5, and no 201M runs at seeds 0–2. So "seed set × contract" is not crossed.
- *Smallest experiment:* already covered by E1, which runs 201M at seeds 0–2.

**(c) Change of evaluation data (validation → held-out shard).**
- *Contradicting:* the 201M runs reverse on the validation set too (+0.0451 vs +0.0420 held-out). At fixed time, the held-out shard also favors AP292 (−0.0041, single runs).
- *Verdict:* **ruled out** as the cause of the flip. Validation and held-out differences agree within 0.003.

**(d) Different optimizer-update counts.**
- *Supporting, as a general mechanism:* in the batch-only ablation, quartering the batch at nearly fixed tokens (88.1M vs ~90.2M) improves BPB by 0.113 (§3.6). Update count is a first-order driver at this scale.
- *Unidentified for this pair:* AR's updates rise 2.28× and AP292's 1.28×, in lockstep with tokens.
- *Smallest experiment:* **E1b**. Run AR and AP292 at 201M tokens with batch 65,536 (3,072 updates) on seeds 0–2: 6 runs. Or run them at 88.5M with batch 65,536: 6 runs. This crosses tokens with updates.

**(e) Architecture/recipe × exposure interaction.**
- AR has 50.3M parameters; AP292 has 29.4M.
- Larger models benefit more from extra tokens. That is consistent with standard scaling behaviour and with the exposure response in §1.4: the 50M models gain most.
- INFERRED: this is the likely mechanism behind (a), not a separate explanation. E1 measures it directly.

**(f) A lucky discovery-stage measurement.**
- *Supporting:* every re-run candidate scored worse than at discovery (§3.3). AR's discovery value (seed 42) was optimistic by 0.0065; AP292's by 0.0015.
- *Contradicting:* the paired fixed-time re-runs show AP292 ahead *after* removing discovery noise and seed mismatch.
- *Verdict:* luck inflated the paper's cross-seed discovery gap, but it does not explain the reversal.

**(g) LR progress variable (elapsed-time vs token fraction).**
- With near-constant throughput the two are proportional, so the warm-down covers the same fraction of tokens.
- *Unverified:* throughput stability within individual fixed-time runs was not checked.
- *Covered by E1,* which uses the token contract at every budget.

**(h) Session and hardware throughput variation under the time contract.**
- *Observed:* the same recipe sees different token counts across sessions under the 300 s rule (§2.4). For AP292, the spread is ±4.8% around 157M.
- This adds noise to the fixed-time comparisons but cannot produce a 0.05 flip.

### 1.4 Exposure response across recipes

DERIVED from `STATS exposure_response`. Fixed-time means use seeds 0–2; 201M means use seeds 3–5.

| Recipe | Fixed-time val (tokens) | 201M val | Token ratio | Δ val |
|---|---|---|---|---|
| Baseline (50.3M, batch 524,288) | 1.2165 (90.2M) | 1.0934 | 2.23 | −0.1231 |
| AR exp 68 (50.3M) | 1.1014 (88.5M) | 1.0327 | 2.28 | −0.0688 |
| AP292 exp 134 (29.4M) | 1.0947 (156.9M) | 1.0777 | 1.28 | −0.0170 |
| AP299 exp 137 (26.2M) | 1.0988 (195.9M) | 1.0971 | 1.03 | −0.0017 |
| AP294 exp 136 (17.9M) | 1.1251 (215.3M) | 1.1284 | 0.94 | +0.0033 |

**Caveats:** seed sets differ (0–2 vs 3–5); n = 5 recipes; the baseline also changes update count with a large batch.

**Interpretation (INFERRED):** under the 300 s contract, the shrunken AutoPareto recipes had already consumed most of the exposure the 201M panel gives them. Their apparent quality advantage was largely "more data per second". This is the mechanism behind the paper's H1, quantified.

### 1.5 What the paper can and cannot say

**Can say**
- "AP292's quality ranking depends on the contract.
  - At equal wall-clock time on the A40 it beats the AutoResearch recipe: −0.0067 BPB, 3 fresh seeds, all the same sign.
  - At equal tokens and updates it loses: +0.042 held-out BPB.
- The reversal appears on both the validation and held-out data, so it is not an artifact of the evaluation split. Within-recipe seed variation (≤0.006) is far too small to produce it."

**Cannot say yet**
- Whether tokens or updates drive the reversal.
- Where the crossover lies.
- That AutoPareto produced *no* exposure-controlled quality gain (see §3.6).

---

## 2. Training-quality variability

### 2.1 Seed effects and within-recipe spread

| Quantity | Value | Source |
|---|---|---|
| Baseline, fixed time, seeds 0–4 (`baseline-a40-cached-fresh-20260722`) | 1.2137, 1.2189, 1.2170, 1.1967, 1.2131; SD 0.0088 | `SNAPDB` ids 8–12 |
| Baseline, earlier run, seeds 0–4 | 1.2067, 1.2127, 1.2123, 1.1932, 1.2095 | ids 3–7 |
| Baseline, seed 42 | 1.1893 (−0.0226 vs mean of seeds 0–4) | id 13 |
| Within-recipe SD, fixed time (6 recipes, seeds 0–2) | 0.0008–0.0027; pooled **0.0021** | `STATS seed_variability` |
| Within-recipe held-out SD at 201M (7 recipes) | 0.0004–0.0060; pooled **0.0037** | same |
| Same seed, same recipe, different session | exp 134 at seed 0: 1.0933 (discovery) vs 1.0975 (re-run), with different token counts (159.8M vs 152.2M) | ids 134, 291 |
| Same baseline code and seeds, two campaigns | `baseline-a40-seed0` (ids 3–7) vs `baseline-a40-cached-fresh-20260722` (ids 8–12). The second is worse at every seed, by 0.0035–0.0070 BPB, and processed about 3% fewer tokens (92.3–93.3M vs 90.2–90.7M). This is a session-level exposure offset under the time contract. | ids 3–12 |

The last row shows that even a fixed seed is not reproducible under the time contract, because throughput varies between sessions.

### 2.2 Statistical units

- **Decode throughput:** each run's value is a median of 30 repetitions, which are repeated measurements. The independent units are the 3 training runs per recipe. Per-seed ratios of AP292 to AR are 1.969, 1.958 and 2.029, so the claim is robust.
- **BPB:** one value per training run. n = 3 per recipe in the panel.
- **Downstream:**
  - The local records hold no per-item outcomes. `downstream_task_results` stores aggregates, and the BLiMP raw JSON stores per-configuration accuracies (67 configurations × 1,000 pairs).
  - Paired item-level confidence intervals for ARC-E, PIQA and LAMBADA therefore cannot be computed.
  - The ±1-point differences on those three tasks are within plausible noise. ARC-E has 2,376 items, so its binomial SE is about 1.0 point.

### 2.3 What n = 3 can resolve

- With a paired-difference SD of about 0.002–0.005, the 95% CI half-width at n = 3 is about 0.005–0.012 BPB.
- **Resolvable:** the 0.042 held-out cost, the 0.045 validation cost, and the AP292 vs AR decode ratio.
- **Marginal:** the −0.0067 fixed-time advantage (its CI barely excludes 0).
- **Not resolvable:**
  - discovery-archive differences of 0.001–0.003;
  - lineage steps such as A20 → A22 (0.00009);
  - the top-k ordering within any campaign;
  - downstream differences of about 1 point.

### 2.4 Exposure noise under the fixed-time contract

DERIVED (`STATS fixed_time_token_exposure_variation`): the maximum relative spread in tokens for the same recipe across discovery and the three re-runs.

| Experiment | Spread |
|---|---|
| exp 65 | 0.3% |
| exp 68 | 0.4% |
| exp 137 | 1.2% |
| exp 134 | 4.8% |
| exp 124 | 6.0% |
| exp 136 | **18.7%** (251.8M at discovery vs 209.7–220.5M on re-runs) |

INFERRED: a time contract couples measured quality to session-level throughput. This is additional, unmodeled variance, and it supports the paper's thesis.

### 2.5 A natural replicate inside the discovery archive

- **The pair.** AP seed-0 attempts A09 (exp 121) and A10 (exp 122) train the same model: 19,661,000 parameters, depth 4, width 384. Their sources differ only in `ASPECT_RATIO` (80 vs 96), and both values round to width 384.
- **What differed.** Same seed (0), same contract, consecutive sessions:

  | Quantity | A09 | A10 | Difference |
  |---|---|---|---|
  | Validation BPB | 1.106798 | 1.105390 | 0.0014 |
  | Tokens | 202.5M | 204.3M | +0.9% |
  | Discovery-protocol cached decode (tok/s) | 324.7 | 291.0 | **10.4%** |

- **How the archive treated it.** The agent's own post-hoc note calls A10 "an effective architecture no-op". The archive still retained A10 as a new non-dominated trade-off, because its BPB was 0.0014 better and its decode 10% worse.
- **Implications (INFERRED):**
  - Under the discovery-era protocol, a ~10% decode difference or a 0.0014 BPB difference between candidates is within measurement noise.
  - Pareto acceptance without noise margins admits pure-noise "trade-offs" into the archive.

Sources: `SNAPDB experiments` ids 121 and 122 (`val_bpb`, `median_cached_decode_tokens_per_second`, `agent_interpretation`, `decision`).

---

## 3. Candidate-selection bias

### 3.1 Eligible pools

From `SNAPDB experiments`, 300 s contract:

| Campaign | Attempts | With BPB | Notes |
|---|---|---|---|
| AP seed 0 | 25 | 24 | 1 crash (exp 129) |
| AP seed 1 | 25 | 25 | |
| AP seed 42 | 25 | 25 | ids 84–112; ids 88–90 deleted; 91 a setup crash |
| AR main, seed 42 | 50 | 19 | 29 crash, 2 interrupted |
| AR diagnostic follow-up, seed 42 | 5 | 5 | exp 64 checkpoint infrastructure failure; exp 65 inference-incompatible |
| TPE, MOTPE, RS, NSGA-II, CMA-ES (seed 42) | 25 each | 23–25 | OOMs |

### 3.2 Selection steps actually applied

The code-level rules are reconstructed in `06_AUTONOMOUS_SEARCH_AUDIT.md`, Part D.

1. **AR reference.** Exp 65 was the lowest-BPB candidate (1.09486). It was confirmed on seeds 0–2, but all three runs were inference-incompatible. Exp 68 was then used as "best_compatible" (`confirmation_runs` 4–9). Both come from a 5-attempt follow-up campaign, not the 50-attempt AR campaign.
2. **AutoPareto roles.** Four roles were confirmed at fixed time on seeds 0–2: exp 124, 134, 136, 137 (`confirmation_runs` 10–21). The 201M panel then carried three of them; exp 124 was dropped.
3. **TPE and MOTPE.** Each campaign's minimum-BPB candidate was used (exp 183, exp 234).
4. **Cross-campaign "best quality".** Exp 134 (seed 0, 1.0933) outranks exp 109 (seed 42, 1.0940) only by a cross-seed comparison. The seed effect is about 0.02, so this ordering is not meaningful.

### 3.3 Discovery optimism (winner's curse)

DERIVED (`STATS selection_optimism`). Optimism is the discovery value minus the fixed-time re-run mean on seeds 0–2; negative means discovery was optimistic.

| Experiment | Discovery seed | Discovery | Re-run mean | Optimism |
|---|---|---|---|---|
| 65 (AR) | 42 | 1.09486 | 1.10172 | −0.0069 |
| 68 (AR) | 42 | 1.09498 | 1.10143 | −0.0065 |
| 124 (AP) | 0 | 1.09630 | 1.09774 | −0.0014 |
| 134 (AP292) | 0 | 1.09329 | 1.09474 | −0.0015 |
| 136 (AP294) | 0 | 1.11632 | 1.12506 | −0.0087 |
| 137 (AP299) | 0 | 1.09743 | 1.09882 | −0.0014 |

- All six are optimistic.
- For reference, a null model with identical candidates and noise SD 0.0021 predicts best-of-N optimism of 0.0025 (N = 5), 0.0042 (N = 25) and 0.0048 (N = 50) (`STATS null_best_of_N_optimism`).
- The AP seed-0 values (≈0.0015) are smaller, partly because seed 0 is both the discovery seed and one of the three re-run seeds.
- The AR values mix selection with the seed-42 effect.
- Exp 136 also lost 15–19% of its training tokens on re-run (§2.4).

### 3.4 Rank agreement

- Discovery vs fixed-time re-run, 6 candidates: Spearman ρ = 0.49 (Pearson 0.97, driven by AP294's outlying value).
- Discovery vs 201M held-out, 7 panel recipes: Spearman ρ = 0.68 (Pearson 0.24).

Order changes from discovery to the 201M held-out ranking:

| Recipe | Discovery rank | 201M held-out rank |
|---|---|---|
| AP292 | 1st | 3rd |
| AR | 2nd | 1st |
| AP299 | 3rd | 6th |
| TPE | 4th | 2nd |
| MOTPE | 5th | 5th |
| AP294 | 6th | 7th |
| Baseline | 7th | 4th |

The larger models (AR, TPE, baseline) rise; the shrunken ones fall. This matches §1.4.

### 3.5 Noise floor vs top-candidate gaps

DERIVED (`STATS top_candidate_gaps_vs_noise`). Gaps are in BPB; the pooled seed SD is 0.0021.

| Campaign | Best → 2nd | Best → 5th |
|---|---|---|
| AP seed 0 | 0.00009 | 0.0022 |
| AP seed 1 | 0.0003 | 0.0011 |
| AP seed 42 | 0.00005 | 0.0010 |
| AR main | 0.00001 | 0.0009 |
| AR diagnostic | 0.0001 | 0.0005 |
| TPE | 0.0023 | 0.0056 |
| MOTPE | 0.0027 | 0.0105 |

Choosing "the best" among the top five of any agent campaign is a choice among statistically indistinguishable candidates. Many top candidates are near-duplicates: one learning-rate step apart along a lineage.

### 3.6 Unselected candidates that might do better under confirmation

| Candidate | Why it matters | Evidence |
|---|---|---|
| **Exp 109** (AP seed 42), and siblings 97, 111, 112 | Full 50.3M model; batch 65,536; matrix LR 0.02; embedding LR 0.6; weight decay 0.1; 1.0940 at **84.1M tokens**, *fewer* than AR's 88.9M at the same seed. The most plausible exposure-neutral AutoPareto quality gain, and never confirmed. | `02_EXPERIMENT_REGISTRY.csv` rows `db:97/109/111/112` |
| **Exp 124** (lowest memory within 1%) | Confirmed at fixed time (1.0977 mean) but dropped from the 201M panel without stated reason | `confirmation_runs` 10–12 |
| **Exp 162** (AP seed 1 best, 1.0958) | Never confirmed; 24.6M parameters at 181M tokens. Likely exposure-reliant, like AP292. | id 162 |
| **Exp 65** (AR lowest BPB) | Excluded for inference incompatibility; its fixed-time re-run quality equals exp 68's (1.1017 vs 1.1014), so the exclusion did not change AR's quality | `confirmation_runs` 4–6 |

**Batch-size mechanism.** In the batch-only ablation, changing only `TOTAL_BATCH_SIZE` from 2^19 to 2^17 in the root baseline gives 1.1040 on seeds 0–2. The baseline mean is 1.2165, and the full AR recipe scores 1.1014.

| Quantity | Value |
|---|---|
| Batch-only gain | 0.1126 |
| AR's total gain over baseline | 0.1151 |
| Share of AR's gain from batch alone | **97.8%** |
| AR recipe with its batch restored (non-batch-only) | 1.2188, slightly *worse* than baseline |

Source: `STATS ablation_decomposition_seeds0_2`. Exp 109 goes further, to 2^16. So the strongest exposure-neutral lever in the whole archive is optimizer-update count. That is a hyperparameter, not an architecture.

**Reconciliation with `07` §B6.** That section reports the batch share as 96% and the "everything except batch" effect as +0.0119. It uses a different baseline set (ids 3–7, 5 seeds, mean 1.2069) and AR runs that mix seed 42 with seeds 0 and 2 (ids 68, 75, 77).

The figures above are **seed-matched**: all four arms use seeds 0–2, the baseline is the cached-fresh campaign (ids 8–10), and AR uses the training results of confirmation runs 75–77. This is the preferred estimate. Both calculations agree on the conclusion: the batch change accounts for 96–98% of AR's fixed-time gain.

The spread in the non-batch effect (+0.002 to +0.012) is itself the size of the session offset between the two baseline campaigns (§2.1).

### 3.7 Pareto frontier stability

- With 3 objectives and 7 recipes, the nondominated set is identical on every seed and on the means: everything except the baseline (`STATS panel_frontier_by_seed`).
- The frontier is stable, but uninformative. Report trade-off sizes (Δ BPB per ×decode) rather than membership.

### 3.8 What the archive can and cannot estimate

| Quantity | Estimable now? | How |
|---|---|---|
| Discovery optimism | Partially: 6 candidates, all from top-of-campaign selections | §3.3 |
| Discovery → confirmation rank correlation | Weakly: 6 or 7 recipes, seed sets not crossed | §3.4 |
| Training-score variability | Yes | §2 |
| Frontier sensitivity to noise | Yes, but trivial | §3.7 |
| How often strong results fail verification | **No.** Only top candidates were ever re-run, so there is no sample from the middle of the distribution. | — |

**Smallest informative selection-bias experiment (E2 in 08).**
- Re-run 12 archived candidates, stratified by discovery rank (top, middle, bottom across AP seed 0/1/42 and AR), on seeds 0–2 at fixed time.
- Also re-run the exp 109 family on seeds 0–2 *and* under the token contract at 88.5M and 201M.
- About 36 + 12 runs, roughly 6–8 A40-hours.

---

## 4. Limits of causal inference

1. **Adaptive trajectories.**
   - Every campaign is a sequence. Attempt k depends on attempts 1..k−1.
   - Attempts are not independent samples, so candidate-level significance tests are invalid.
   - The unit for any contract effect is the campaign.
2. **No randomized replication of contracts.**
   - Each Pair-A lineage has one trajectory per arm.
   - The five lineages differ in protocol (N*, A40 instance, warm-up handling, manifest amendments), so they are not exchangeable replicates (06, Part B).
3. **Proprietary agent.** `gpt-5.6-sol` via Codex has hidden sampling state, so trajectories are not reproducible.
4. **Retrospective hypotheses.** H1 was formulated after inspection, which the paper discloses. The analyses in this audit are also post hoc and should be labeled as such.
5. **Multiple comparisons.** Many recipes, tasks and metrics were inspected. Only the pre-specified headline (AP292 vs AR, held-out BPB, decode, memory) should carry inferential weight.

---

## 5. Pair-A: what the data can and cannot say

DERIVED from `STATS pair_a_group_summary` and `02_EXPERIMENT_REGISTRY.csv`.

| Lineage (namespaces) | N* | Arm | Attempts (with BPB) | Best BPB | Batch changes (attempt: from → to) | Degenerate (>3 BPB) |
|---|---|---|---|---|---|---|
| G0 `pair-a` | 88,604,672 | T | 2 (0) | — | — | — |
| G0 | | N | 2 (1) | 1.1400 | first completed attempt already at 131,072 | 0 |
| G1 `…20260906-fresh` | 88,604,672 | T | 2 (2) | 1.1752 | a1 at 262,144 | 1 |
| G1 | | N | 2 (2) | 1.1402 | a1 at 131,072; a2 → 65,536 | 0 |
| G2 `…20260906-fixed-warmup-v2` (+0907 continuations; **paper prefix = a1–a4**) | 88,604,672 | T | 8 (8) | 1.1034 | a1 262,144; a2 → 131,072; a3 → 65,536; a6 → 131,072 | **6** |
| G2 | | N | 8 (8) | 1.1243 | a3 524,288 → 262,144; a4 → 131,072; a5, a7, a8 back and forth | 0 |
| G3 `…20260909-new-a40` | 85,458,944 | T | 7 (7) | 1.1846 | **none** | 0 |
| G3 | | N | 7 (7) | 1.1883 | a3 524,288 → 262,144 | 0 |
| G4 `…20260914-new-a40` (+0916, 0916-r2) | 85,458,944 | T | 22 (22) | 1.1016 | a2 → 262,144; a12 → 131,072 | 0 |
| G4 | | N | 22 (20) | 1.2030 | **none** | 0 |

**What the table shows**
- Batch reduction appears in the fixed-token arm of 4 of the 5 lineages.
- It is absent from the fixed-time arm of G3 and the fixed-token arm of G4.
- **No contract-dependent pattern is identifiable.** The single G4 contrast, cited in an earlier analysis, is not replicated.
- Fixed-time arms in G1 and G2 contain degenerate runs; fixed-token arms do not. Whether that is a contract-specific harness or runtime fault is examined in 06, Part B.

**A further validity problem in the later fixed-token arms (OBSERVED).**
- **The rule.** The fixed-token program requires the LR schedule to use token progress, `progress = min(total_tokens / N_STAR, 1)` (`study/contracts_2026/programs/program_fixed_tokens.md` line 14).
- **The protected hook does not enforce it.** `contract_progress` (`candidate_process.py` lines 185–216) only marks optimizer boundaries and counts tokens. Each candidate's own `train.py` computes the learning-rate progress.
- **The starting source is time-based.** It uses `total_training_time / TIME_BUDGET`, so token progress has to be added by the agent.
- **What the agents did,** from scanning `candidate_patch` for schedule edits:

  | Lineage | Attempts with token progress |
  |---|---|
  | G2 (paper prefix) | N1 added it; N2–N4 descend from N1 |
  | G3 | none of 7 |
  | G4 | only attempt 2 (also `07` B5) |

- **Consequence.** In G3/G4 most fixed-token runs decayed the learning rate on a 300-second clock while stopping at N* tokens.
  - For near-baseline models (about 300 s to reach N*) the effect is small.
  - For faster candidates the decay is incomplete, which **penalizes exactly the efficiency edits the fixed-token contract is meant to treat neutrally**.
- **Implication.** G3/G4 fixed-token quality outcomes, including the G4 arm's failure to beat its baseline, are confounded by this schedule deviation. That is a further reason not to read them as contract effects. The paper's Table 2 ("counted-token fraction") is accurate for the reported G2 prefix only.

**Required design:** **E3** in 08. Add a harness check that rejects candidates whose learning-rate progress does not follow the contract.
- At least 5 campaigns per contract, paired by seed.
- One frozen protocol and a fixed agent version.
- Pre-declared outcomes.
- Campaign as the unit of analysis.
