# 03 — Claims and Evidence

**Scope.** Every important numerical and qualitative claim in the submitted AutoPareto paper is checked against the raw records.
- Submitted PDF: `/Users/Rami/Documents/dev/automlr_autopareto.pdf`.
- Source: `paper/main.tex` and `paper/generated/*`.
- The audit is read-only. No experiment was run.

**Labels**
- **OBSERVED**: read directly from a record.
- **DERIVED**: calculated by this audit. The calculation is shown or scripted.
- **INFERRED**: interpretation.
- **UNVERIFIED**: the evidence was not found locally.

**Confidence levels**
- **High**: deterministic, or recomputed exactly.
- **Medium**: recomputed, with caveats.
- **Low**: indirect.

## Abbreviations

| Name | Path or meaning |
|---|---|
| `SNAPDB` | `results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db`. Latest SQLite: 318 experiments, a superset of `results/results.db`, and identical on the 297 shared ids. |
| `P1JSON` | `results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/phase1_equal_token_results.json`. The 21 matched-token runs. |
| `CLAIMS` | `audit/camera_ready_2026/supporting/claims_recomputation.json`, produced by `supporting/scripts/recompute_claims.py`. |
| `STATS` | `audit/camera_ready_2026/supporting/statistical_audit.json`, produced by `supporting/scripts/statistical_audit.py`. |
| AR | The AutoResearch comparator: experiment 68, source `b3a961ee`. |
| AP292 | Experiment 134, source `55a7fd8b`. |

To re-run the audit:
```
python3 -I supporting/scripts/recompute_claims.py <repo> supporting/claims_recomputation.json
python3 -I supporting/scripts/statistical_audit.py <repo> supporting/claims_recomputation.json supporting/statistical_audit.json
```

---

## A. Verdict summary

| # | Claim (where in the paper) | Verdict | Confidence |
|---|---|---|---|
| C1 | AP292 has 1.99× the cached-decode throughput of AR at matched tokens, updates and batch (abstract, §1.4) | **Supported** for this workload, GPU and protocol | High |
| C2 | 41.7% fewer parameters | **Supported** (exact) | High |
| C3 | 35.1% less peak allocated inference memory | **Supported** (exact; deterministic) | High |
| C4 | 50% less incremental KV-cache memory (1.992 vs 3.984 MiB) | **Supported** | High |
| C5 | Held-out BPB cost of 0.042034 (1.141837 vs 1.099803) | **Supported**. Paired 95% CI is [0.0297, 0.0544]; 3 of 3 seeds agree in sign. | High |
| C6 | "The discovery-stage quality advantage … does not persist" under matched-exposure verification | **Supported, but incomplete.** Seed-matched fixed-time re-runs that the paper does not report show the advantage is real at equal wall-clock time. It reverses only when tokens are matched. | High |
| C7 | H1 is supported | **Supported.** The confounds are narrower than the paper admits: existing data rule out the evaluation-set change and, very plausibly, the seed change (see 05). | Medium-high |
| C8 | Archive bests: AP 1.0933 < AR 1.0950 < TPE 1.0977 < MOTPE 1.0981 < RS 1.1038 < NSGA-II 1.1077 < CMA-ES 1.1083 (§1.2) | **Numerically correct, scientifically weak.** It compares seed 0 against seed 42; the baseline differs by 0.024 BPB between those two seeds. It also mixes unequal attempt budgets, and the AR value comes from a 5-attempt follow-up campaign. | High (numbers) / Low (meaning) |
| C9 | 201,326,592 = 192·2^20 = 1,536 × 131,072 = 384 × 524,288 | **Supported** (arithmetic) | High |
| C10 | Confirmation seeds 3–5 are disjoint from discovery seeds (0, 1, 42) and earlier confirmation seeds (0–2) | **Supported** | High |
| C11 | Every Table 1 entry | **Supported.** All 35 cells reproduce. | High |
| C12 | Memory is exactly constant across 30 repetitions and 3 seeds (SD = 0) | **Supported across seeds.** Repetition-level constancy not re-checked. | Medium |
| C13 | Edit labels 32/21/10/8/4 over 75 attempts come from a "source-diff audit" | **Counts supported.** The provenance wording is doubtful: the counts equal the database field `change_category`, not an independent diff audit. | Medium |
| C14 | Lineage of A22: baseline → A1 → A4 → A6 → A7 → A8 → A18 → A19 → A20 → A22, plus the BPB milestones | **Supported** | High |
| C15 | Downstream ARC-E 36.91/36.50, PIQA 58.29/58.67, LAMBADA 21.50/20.44, BLiMP 72.02/74.56; AP292 lower on 51 of 67 BLiMP configurations | **Supported**. BLiMP split is 51 lower, 1 tied, 15 higher. | High |
| C16 | r = −0.969 (BLiMP vs held-out BPB); r = 0.717 (BLiMP vs depth) | **Supported** (n = 7 recipe means; descriptive) | High |
| C17 | H2 prefix: N arm 50.33M → 85.85M → 135.27M, then returns to the N1 parent with halved batch; T arm "keeps 50.33M and changes batching and LR"; 0 of 8 Pareto-eligible | **Partially supported.** Counts and parameters are correct. The text omits that 2 of the 4 reported fixed-time attempts are degenerate (BPB 3.2159) and that the controller still labelled them `quality_valid … parent_eligible: true`. | High |
| C18 | N* = 88,604,672 is the "largest multiple of the quantum not exceeding the median of three seed-3 baseline runs" | **Unverified.** The rule exists in code; the three calibration runs were not located. | Medium |
| C19 | "Selection … recorded in campaign identifiers created on 2026-08-10 and executed from 2026-08-12, so selection preceded confirmation" | **Partially supported.** It applies to the fixed-time confirmations, not to the 201M panel. A fourth selected role (exp 124) was confirmed but silently dropped from the panel. | Medium |
| C20 | AutoPareto "attains" a frontier point; Figure 2 marks the nondominated set | **Supported, but weakly informative.** 6 of 7 panel recipes are nondominated on every seed; only the baseline is dominated. | High |
| C21 | Hardware and software pins (A40 driver 570.211.01, PyTorch 2.9.1+cu128, CUDA 12.8, Python 3.12) | **Era-dependent.** July-era tracebacks use a `python3.10` venv. See 06, Part B. | Medium |

---

## B. Detailed matrix

### C1. Cached-decode throughput ratio, 1.99×

**Paper:** abstract; §1.3 "Matched-exposure outcome"; §1.4.

**Evidence (OBSERVED)**
- `P1JSON` `runs[*].stabilized_decode_tokens_per_second`.
- AP292 seeds 3/4/5: 382.02, 378.70, 388.34 tok/s.
- AR seeds 3/4/5: 193.99, 193.44, 191.37 tok/s.

**Calculation (DERIVED)**
- Means: 383.02 / 192.93 = **1.9853**.
- Per-seed ratios: 1.969, 1.958, 2.029.

**Conditions**
- Both recipes: 201,326,592 tokens, 1,536 updates, batch 131,072, seeds 3–5.
- Inference protocol `autopareto_inference_stabilized_v2`: BF16, batch 1, 256 + 256 tokens, median of 30 repetitions, one A40.

**Controlled?** Yes, for training exposure. The recipes still differ in architecture: depth 4 vs 8, window pattern SSSL vs SSSS, and learning-rate constants (`02_EXPERIMENT_REGISTRY.csv` rows `db:134` and `db:68`).

**Statistical units:** 3 independent training runs per recipe. The 30 repetitions per run are repeated measurements, not independent units.

**Uncertainty and confounds**
- The ratio is mostly a depth effect (half the layers), which the paper acknowledges.
- It is specific to batch-1 decoding of 256 tokens on an A40. It is not a cross-hardware or cross-workload claim.

**Verdict:** Supported. **Confidence:** High.

### C2–C4. Parameters, memory and KV cache

**Evidence (OBSERVED)**
- `SNAPDB experiments.parameter_count`: exp 134 = 29,360,392; exp 68 = 50,332,176.
- `paper/generated/equal_token_runs.json`, `memory` / `incremental_kv_memory`:
  - AP292: 162.8798828125 MiB and 1.9921875 MiB, identical on all 3 seeds.
  - AR: 250.8818359375 MiB and 3.984375 MiB.

**Calculation (DERIVED)**
- Parameters: 1 − 29,360,392 / 50,332,176 = **41.667%**.
- Memory: 1 − 162.880 / 250.882 = **35.077%**.
- KV: 1.9922 / 3.9844 = **0.500**.

**Verdict:** Supported. These are deterministic properties of the architectures at this workload. **Confidence:** High.

**Note.** The memory definition is `max_memory_allocated` with weights resident; reserved memory is excluded. The 35% saving partly reflects 42% fewer weights, so it is not a pure activation or KV saving. The paper's definition is accurate; readers may still conflate the two.

### C5. Held-out quality cost, 0.042034

**Evidence (OBSERVED):** `P1JSON` `heldout_bpb`.
- AP292: 1.139616, 1.148668, 1.137228.
- AR: 1.100921, 1.100929, 1.097560.

**Calculation (DERIVED)**
- Mean difference: **+0.042034**.
- Paired by seed: +0.038695, +0.047739, +0.039668.
- SD = 0.004964; t(2) = 14.67; 95% CI **[0.0297, 0.0544]**.

**Cross-check:** `SNAPDB downstream_task_results` (`heldout_climbmix_bpb`) equals `P1JSON` exactly (maximum absolute difference 0.0).

**Verdict:** Supported. **Confidence:** High.

### C6–C7. The quality "reversal" and H1

**Paper:** §1.1 (H1); §1.3; §2.8: "the discovery-to-confirmation ranking reversal cannot be attributed to exposure alone".

**Evidence:** the full decomposition is in `05_STATISTICAL_AND_CAUSAL_AUDIT.md` §1.

| Comparison of AP292 − AR | Contract | Seeds | Evaluation data | Δ BPB | Source |
|---|---|---|---|---|---|
| Discovery archive | 300 s | 0 vs 42 (unmatched) | validation | −0.0017 | `SNAPDB experiments` id 134 vs 68 |
| Fixed-time re-runs | 300 s | 0, 1, 2 (paired) | validation | **−0.0067** (CI [−0.0130, −0.0003]; 3/3 seeds) | `SNAPDB confirmation_runs` ids 13–15 vs 7–9 → `training_runs` |
| Fixed-time checkpoints | 300 s | 1 vs 2 (single runs) | held-out | −0.0041 | `SNAPDB downstream_evaluation_runs` 4 (exp 292) vs 2 (exp 77) |
| Matched tokens | 201M tokens | 3, 4, 5 (paired) | validation | **+0.0451** (CI [0.0415, 0.0486]) | `P1JSON final_validation_bpb` |
| Matched tokens | 201M tokens | 3, 4, 5 (paired) | held-out | **+0.0420** (CI [0.0297, 0.0544]) | `P1JSON heldout_bpb` |

**What this establishes (INFERRED from the observed rows above)**
1. The fixed-time advantage of AP292 is reproducible on fresh seeds. It is not only a lucky discovery measurement.
2. The reversal at 201M tokens is present on the same validation data used for discovery. So the switch to held-out data does not explain it.
3. Seed-to-seed differences inside one recipe are about 0.002–0.005 BPB. A seed explanation would need a seed-by-recipe interaction of about 0.05, which is implausible.
4. What changes between the two regimes is the training contract:
   - AR's exposure goes from about 88.5M tokens and 675 updates to 201M tokens and 1,536 updates (×2.27).
   - AP292's goes from about 157M tokens and 1,197 updates to 201M and 1,536 (×1.28).
   - Tokens and updates move together, because batch is 131,072 for both. So **tokens and updates cannot be separated** with existing data.

**Verdict.** The paper's statement is correct but under-claims and under-explains. The defensible statement is that the quality ranking depends on the contract:
- At equal wall-clock time on the A40, AP292 is better: −0.0067, 3 seeds.
- At equal tokens and updates, AR is better: +0.042 held-out, 3 seeds.
- The existing records exclude the evaluation-set change, and make the seed change implausible, as explanations.

**Confidence:** High for the measurements; medium-high for the causal attribution.

**Contradictory evidence:** none found.

**Unresolved:** whether tokens or updates drive the change, and where the crossover lies (see 08, experiment E1).

### C8. Archive best list

**Evidence (OBSERVED):** `SNAPDB experiments`, minimum `val_bpb` per campaign.

| Recipe | Experiment | Val BPB | Seed | Notes |
|---|---|---|---|---|
| AP (seed 0) | exp 134 | 1.093294 | 0 | |
| AP (seed 1) | exp 162 | 1.095755 | 1 | |
| AP (seed 42) | exp 109 | 1.093981 | 42 | |
| AR, main 50-attempt campaign | exp 61 | 1.096754 | 42 | |
| AR, 5-attempt diagnostic follow-up | exp 65 | 1.094856 | 42 | `inference_status = incompatible`; cached/full logits mismatch, max abs error 0.115 |
| AR, 5-attempt diagnostic follow-up | exp 68 | 1.094976 | 42 | Compatible; used as AR |
| TPE | exp 183 | 1.097702 | 42 | |
| MOTPE | exp 234 | 1.098112 | 42 | |
| RS | exp 247 | 1.103809 | 42 | |
| NSGA-II | exp 275 | 1.107744 | 42 | |
| CMA-ES | exp 202 | 1.108287 | 42 | |

**Calculation (DERIVED):** every number printed in the paper reproduces.

**Problems**
1. **Seed confound.** The unchanged baseline scores 1.189298 at seed 42 (exp 13) but 1.213735 at seed 0 (exp 8): a 0.0244 BPB seed effect. That is about 14× the 0.0017 gap the paper reports between AP and AR.
2. **Seed-matched alternative.** AP seed 42 scored 1.0940 against AR's 1.0950, still lower, but by 0.0010, about half a seed SD.
3. **Unequal budgets.** AP had 25 attempts per seed (75 total). AR had 50 plus a 5-attempt follow-up. Classical methods had 25 each.
4. **AR's value comes from the follow-up campaign.** "AutoResearch (1.0950)" is the best *compatible* value from `autoresearch-diagnostic5-seed42-20260724`. It is not from the 50-attempt campaign (best 1.0968). The lowest archived AR value overall is 1.0949 (exp 65), excluded for inference incompatibility.

The paper does caveat "space, budget, seed, and failure history". Even so, listing values to four decimals invites a ranking reading that the seed effect makes meaningless.

**Verdict:** numerically correct; interpretively misleading. **Confidence:** High.

### C9–C10. Token arithmetic and seed disjointness

- **Tokens (DERIVED):** 192 × 1,048,576 = 201,326,592; 1,536 × 131,072 = 201,326,592; 384 × 524,288 = 201,326,592.
- **Seeds (OBSERVED):** `P1JSON seeds = [3, 4, 5]`. `SNAPDB confirmation_runs.seed ∈ {0, 1, 2}`. Discovery campaigns use seeds 0, 1 and 42.
- **Verdict:** Supported.

### C11. Table 1, all seven rows recomputed from `P1JSON`

| Recipe | Params (M) | Held-out BPB | Decode tok/s | MiB | Steady s | Matches paper? |
|---|---|---|---|---|---|---|
| Baseline | 50.33 | 1.146590 | 196.82 | 250.88 | 688.5 | Yes |
| AutoResearch | 50.33 | 1.099803 | 192.93 | 250.88 | 696.2 | Yes |
| TPE | 39.85 | 1.119402 | 258.89 | 205.88 | 540.5 | Yes |
| MOTPE | 26.35 | 1.148253 | 270.01 | 145.57 | 398.2 | Yes |
| AP292 | 29.36 | 1.141837 | 383.02 | 162.88 | 385.0 | Yes |
| AP299 | 26.21 | 1.159309 | 488.26 | 142.38 | 312.8 | Yes |
| AP294 | 17.89 | 1.190057 | 495.49 | 106.94 | 295.7 | Yes |

**Minor discrepancy.** Validation BPB for the panel runs differs between `SNAPDB experiments.val_bpb` (ids 301–321) and `P1JSON final_validation_bpb`, by up to **2.43e-5**. Held-out BPB matches exactly. The paper uses only held-out values, so it is unaffected. The final artifact should still name which validation value is canonical.

### C12. Memory constancy

**OBSERVED:** `peak_mem_values` has exactly one value per recipe across the 3 seeds (`CLAIMS table1.*.peak_mem_values`).

Repetition-level constancy (30 repetitions) was not re-derived. It requires the per-repetition stabilized-inference records under `…/phase1/stabilized_inference/` and the run folders.

**Verdict:** Supported across seeds; repetition level is unverified by this audit. **Confidence:** Medium.

### C13. Edit-category counts

**OBSERVED:** `SNAPDB experiments.change_category` over the three AutoPareto campaigns (75 rows): architecture 32, optimizer 21, optimization 10, batching 8, schedule 4. These match the paper.

**Provenance**
- The paper calls these "the source-diff audit labels".
- No independent diff classifier that produced them was found.
- In the Pair-A harness, `change_category` is a field the agent returns itself (`study/contracts_2026/bridge.py:229`, `campaign_controller.py:416`).
- For the historical campaigns, the field is stored in the database, and whoever assigned it is not documented in the code inspected.

**Verdict:** counts supported; provenance wording should become "recorded change categories" unless an independent diff audit is located. **Confidence:** Medium.

### C14. Lineage of exp 134 (Figure 1)

**OBSERVED** (`SNAPDB experiments.parent_experiment_id` chain):

| Step | Exp | Attempt | Val BPB |
|---|---|---|---|
| Baseline | 8 | — | 1.2137 |
| A1 | 113 | 1 | 1.1063 |
| A4 | 116 | 4 | 1.1005 |
| A6 | 118 | 6 | 1.0978 |
| A7 | 119 | 7 | 1.0966 |
| A8 | 120 | 8 | 1.1573 |
| A18 | 130 | 18 | 1.0957 |
| A19 | 131 | 19 | 1.0954 |
| A20 | 132 | 20 | 1.0934 |
| A22 | 134 | 22 | 1.0933 |

A25 (exp 137) decode = 408.40 tok/s.

**Verdict:** Supported.

**Caveat.** The A20 → A22 step improves BPB by only 0.00009 (1.093382 → 1.093294), about 1/24 of a seed SD (0.0021). The "successive learning-rate edits reached 1.0933" narrative describes noise-level moves (05 §4).

### C15–C16. Downstream results and correlations

**OBSERVED:** `SNAPDB downstream_task_results` joined to experiments 301–321, averaged over seeds.

| Task | AP292 | AR |
|---|---|---|
| ARC-E accuracy | 0.3691 | 0.3650 |
| PIQA | 0.5829 | 0.5867 |
| LAMBADA exact match | 0.2150 | 0.2044 |
| BLiMP | 0.7202 | 0.7456 |

**BLiMP per configuration** (`raw_json.configurations`, averaged over seeds): AP292 lower on 51, tied on 1, higher on 15 (67 total).

**Correlations (DERIVED; 7 recipe means)**
- BLiMP vs held-out: Pearson **−0.9688**.
- BLiMP vs depth: **0.7173**.
- BLiMP vs parameters: 0.6805.

**Verdict:** Supported.

**Caveats**
- n = 7 recipe means. These are descriptive correlations with no inference.
- Depth is read from the frozen sources.
- Downstream differences on ARC-E, PIQA and LAMBADA are within about 1 point. No per-example traces exist locally to compute paired CIs (`downstream_generations` holds 576 generations, not per-item scores).

### C17. H2 prefix

**OBSERVED:** `runs/pair-a-official-20260906-fixed-warmup-v2/pair-a-fixed-{time,tokens}-seed3/history.jsonl`. This is the paper's reported prefix: `TimeBest` = 1.103402 and `TokenBest` = 1.124312 match `paper/FINDINGS_FOR_HUMAN.md`.

| Arm | Attempt | Params | Batch | Val BPB | Controller decision |
|---|---|---|---|---|---|
| N | 1 | 85,852,980 | 524,288 | 1.1631 | quality_valid_speed_uncertain, parent_eligible = true |
| N | 2 | 135,267,480 | 524,288 | 1.1643 | same |
| N | 3 | 85,852,980 | 262,144 | 1.1243 | same |
| N | 4 | 85,852,980 | 131,072 | 1.1402 | same |
| T | 1 | 50,332,176 | 262,144 | 1.1748 | same |
| T | 2 | 50,332,176 | 131,072 | 1.1034 | same |
| T | 3 | 50,332,176 | 65,536 | **3.2159** | same; still `parent_eligible = true` |
| T | 4 | 50,332,176 | 65,536 | **3.2159** | same |

**Verdict:** partially supported.

**What is not disclosed**
1. Half of the reported fixed-time prefix are degenerate runs. 3.2159 BPB is far above the baseline's ~1.20 and indicates training that failed.
2. The harness classified those runs as quality-valid parents.
3. The same namespace family, continued to 8 attempts (`…-20260907-new-a40-r2`), produced 6 of 8 degenerate fixed-time attempts. These include architecture-only edits at batch 131,072 (attempts 6–8).

The root cause is investigated in 06, Part B.

### C18. N* derivation

- **OBSERVED:** `runs/pair-a-seed3-baseline-registration.json` records one baseline run with `total_tokens` = 88,604,672 (= 169 × 524,288).
- **OBSERVED:** `study/contracts_2026/calibration.py:54-55` computes N* = floor(median(token_counts) / quantum) × quantum.
- The three seed-3 calibration runs described in the paper were not located in the paper-era files inspected.

**Verdict:** unverified. Either cite the calibration record or soften the description.

### C19. Selection timing and roles

**OBSERVED:** `SNAPDB campaigns.created_at` for the `autopareto-primary-*-20260810` confirmation campaigns runs from 2026-08-11 23:43:02 to 2026-08-12 01:15:33. The timestamps carry no explicit timezone; the paper's "executed from 2026-08-12" plausibly refers to NZ time.

Four roles were selected:
- `lowest_memory_within_1pct_quality` (exp 124);
- `best_quality` (134);
- `largest_exclusive_hypervolume_contribution` (136);
- `fastest_within_1pct_quality` (137).

All four were confirmed at fixed time on seeds 0–2 (`SNAPDB confirmation_runs` ids 10–21). Only three entered the 201M panel; **exp 124 was dropped**. The paper lists three roles and does not mention the fourth.

The 201M panel itself was assembled later. `P1JSON` and `docs/PHASE1_200M_RESULTS.md` date its runs to Aug 24–27.

**Verdict:** the timing claim holds for the fixed-time confirmations. The panel selection policy needs a sentence (06, Part D).

### C20. Frontier membership (Figure 2)

**DERIVED** (`STATS panel_frontier_by_seed`):
- With 3 objectives (held-out BPB down, decode up, memory down), the nondominated set is {AP292, AP294, AP299, AR, MOTPE, TPE} on each seed and on the means.
- Only the baseline is dominated.

**Verdict:** the figure is accurate. But "frontier point" carries little information here: 6 of 7 recipes qualify. Report trade-off magnitudes, not membership.

### C21. Software and hardware

The paper states Python 3.12, PyTorch 2.9.1+cu128, CUDA 12.8 and driver 570.211.01. July-era failure tracebacks stored in `SNAPDB experiments.crash_or_failure_reason` (e.g. exp 65) reference `/workspace/autopareto/.venv/lib/python3.10/site-packages`. The pins are therefore era-specific. See 06, Part B.

---

## C. Findings the data support but the paper omits

| # | Finding | Evidence | Confidence |
|---|---|---|---|
| U1 | AP292 beats AR at equal wall-clock time on fresh seeds: −0.0067 BPB, 3 of 3 seeds | `SNAPDB confirmation_runs` 7–9, 13–15 | High |
| U2 | The 201M reversal also appears on the discovery validation set (+0.045), not only on held-out data | `P1JSON final_validation_bpb` | High |
| U3 | Across 5 recipes, the change in BPB from fixed time to 201M tracks the change in tokens (Pearson −0.92 against log2 token ratio) | `STATS exposure_response` | Medium (n = 5, seeds differ) |
| U4 | 97.8% of AR's gain over the baseline comes from one batch-size change (2^19 → 2^17) | `SNAPDB ablation_runs` + training_runs 78–83; baseline ids 8–10 | High |
| U5 | Seed main effects are large: seed 42 baseline is 0.023 below the mean of seeds 0–4, and seed 3 is easy too | `SNAPDB` ids 3–13 | High |
| U6 | Under the fixed-time contract, the *same recipe* sees up to 18.7% different token counts across sessions (exp 136: 251.8M at discovery vs 209.7–220.5M on re-run) | `STATS fixed_time_token_exposure_variation` | High |
| U7 | Top candidates within a campaign are separated by ≤0.0003 BPB, against a seed SD of 0.0021 | `STATS top_candidate_gaps_vs_noise` | High |
| U8 | The AR comparator was selected as "best compatible" after the lowest-BPB candidate (exp 65) failed the cached-decode correctness check. Fixed-time re-runs make the two indistinguishable (1.1017 vs 1.1014). | `SNAPDB` ids 65, 68; `confirmation_runs` 4–9 | High |
| U9 | Experiments 88–90 are absent from every retained database snapshot; they were deleted inside the seed-42 AP campaign's id range | `supporting/registry_build_report.json` | High |
| U10 | Pair-A has five campaign lineages, not one. The 22-attempt histories are a later relaunch (new A40, N* = 85,458,944), not a continuation of the paper's prefix. | `02_EXPERIMENT_REGISTRY.csv`, `pair_a_group` column | High |

## D. Discrepancy log

| # | Discrepancy | Records | Action |
|---|---|---|---|
| D1 | Panel validation BPB differs by up to 2.4e-5 between the DB and the JSON | `SNAPDB` ids 301–321 vs `P1JSON` | Name the canonical source |
| D2 | AR's "lowest archived" value: 1.0950 (compatible) vs 1.0949 (incompatible exp 65) | `SNAPDB` ids 65, 68 | Say "best compatible" |
| D3 | Python 3.12 (paper) vs python3.10 venv (July tracebacks) | `SNAPDB` exp 65 failure text | Give per-era environment pins |
| D4 | "Source-diff audit labels" vs the `change_category` field | `SNAPDB experiments` | Reword |
| D5 | N* described as the median of 3 runs; 1 run recorded | `runs/pair-a-seed3-baseline-registration.json` | Cite the calibration or reword |
| D6 | Fourth selected role (exp 124) omitted from the panel and the text | `SNAPDB confirmation_runs` 10–12 | Disclose |
| D7 | H2 prefix text omits the degenerate 3.2159 fixed-time attempts | `runs/pair-a-official-20260906-fixed-warmup-v2/…` | Disclose; explain |
| D8 | Two AR training runs left in `running` state (training_runs 14973 and 16716) | `SNAPDB training_runs` | Mark them interrupted in the artifact |

## E. Further discrepancies from the reproducibility and provenance audit

Sources: `supporting/REPRODUCIBILITY_AND_PROVENANCE.md` (its M-numbers), cross-checked by this audit where marked ✔.

| # | Discrepancy | Evidence | Severity | Action |
|---|---|---|---|---|
| D9 ✔ | **Figure 1 width labels for A07–A10 are wrong.** The labels are hand-typed in `analysis/generate_figures.py` (the `edit=` dict, line ~44) and ignore that width is rounded up to a multiple of `HEAD_DIM` = 128 (`build_model_config`, line 471 of each recipe). Actual widths: A07 384→**384** (label says 384→320); A08 **384**→256 (label 320→256); A09 256→**384** (label 256→320); A10 384→**384**, a no-op (label 320→384). | `results/candidates/autopareto-seed0-20260731/0006-0010-train.py`: DEPTH × ASPECT_RATIO, rounded | **High**: a factual error in a figure | Generate the labels from the sources |
| D10 ✔ | **A09 and A10 build the same model.** Both have 19,661,000 parameters; the sources differ only in `ASPECT_RATIO` 80 vs 96, which round to the same width. The agent's own post-hoc interpretation calls A10 "an effective architecture no-op". The archive still *retained* it, because noise made it non-dominated: BPB 1.10539 vs 1.10680, and decode 291.0 vs 324.7 tok/s. | `SNAPDB experiments` ids 121 and 122 (`agent_interpretation`, `decision`) | Medium | Report this as a natural replicate (05 §2.5) |
| D11 | Appendix A says counted tokens "exclude the eleven disposable warm-up updates". REPRO M3 reports this is false for the 21-run panel: the 11-update figure is the steady-state *timing* window, not a disposable warm-up. | `P1JSON timing_note`; REPRO M3 | Medium | Reword |
| D12 | Python 3.12 (paper) vs 3.10.18 (panel environment); drivers and torch versions differ across eras | REPRO M4 | Low-medium | Give a per-era environment table |
| D13 | The Figure 2 caption says the horizontal error bars are narrower than the markers. Not true for AP299 or MOTPE. | REPRO M7 | Low | Fix the caption |
| D14 | "Every table and vector figure … is regenerated" overstates. Table 2, Figure 3, the Figure 1 labels and the archive numbers in the prose are hard-coded. | REPRO M8 | Medium | Reword, or generate them |
| D15 | Training data is downloaded from the dataset's `main` branch, not the revision the paper pins (`915333b4…`) | REPRO M14 | Medium | Pin the revision; disclose |
| D16 | **Panel checkpoints.** All 21 exist locally as hash-verified LFS objects in `.git/lfs/objects`. The working-tree snapshot copy has only 10 complete files, 1 truncated (`phase1-autoresearch_compatible-seed4/final.pt`) and 10 empty folders. The **discovery checkpoints of AR-ref (exp 68) and MOTPE (exp 234) are missing**; downstream evaluation used substitutes (exp 77, a seed-2 re-run of exp 68's source; exp 300, a seed-42 recovery re-training of exp 234). | `01` §0 item 5 and §C.3; REPRO M15 | Medium | Restore the panel checkpoints from LFS; disclose the substitutes |
| D17 | A17's "failed run" failed on missing infrastructure (a tokenizer file), not on its edit. The same failure in seed 42 was not counted as an attempt. | REPRO M16; `SNAPDB` ids 91 and 129 | Low-medium | Describe the failure accurately; count consistently |
| D18 | **The submitted abstract's source is only `paper_backups/automlr_overleaf_reconstructed/`.** `paper/generated/abstract.tex` and the zips differ, so a pipeline refresh would overwrite the submitted wording. | REPRO M1 | **High for camera-ready editing** | Edit the camera-ready from a reconciled source |
| D19 | The confirmation campaigns were created 2026-08-11/12 (DB), not 2026-08-10. Proof that selection came first exists only in private git history (branch `autopareto/research-infrastructure`). | REPRO M6; 06 D.2 | Low-medium | Scope the claim; release the plan files |
| D20 | The executed panel harness is not the recorded commit. Two harness files differ from their preflight hashes, and 13 executed Pair-A files are not preserved locally. | REPRO M13 | Medium | Disclose |
| D21 | **The token-stream verification exists**, although the project docs say it is missing. `results/phase1/token_stream_verification.json` (git `e74771d` and the RunPod snapshot) reports `RUNTIME_VERIFIED` and `all_seed_stream_hashes_equal: true`. `evaluation/phase1_token_stream_audit.json`, `docs/RESEARCH_LEDGER.md` and `FORENSIC_RESEARCH_AUDIT.md` say it was unobserved. | `01` §0 item 3 | Positive for the paper | Cite it as evidence of common token streams |
| D22 | **The identity of the submitted PDF is uncertain.** The local `automlr_autopareto.pdf` (sha `60b419b1…`, created 2026-09-06 22:59Z) matches no other local build. The locally recorded deadline (2026-09-06 11:59 UTC) predates every 10-page build. | `01` §0 item 10 and §G | High for camera-ready editing | Download the submitted PDF from OpenReview and diff it before editing |
| D23 | The public `data/experiment_records.csv` has 284 rows (the 08-10 snapshot), not 297 | `01` §0 item 11 | Low-medium | Regenerate for the release |
| D24 | **Uncommitted edits to the protected evaluator** `protected/inference_adapter.py` (+125/−6, mtime 2026-09-16), plus `docs/RESEARCH_LEDGER.md` | `01` §0 item 12; `git status` | Medium (evaluator provenance) | Commit or revert, with hashes, before any new run (E0) |
| D25 | `results/candidates/baseline-a40-seed0/0003-train.py` (exp 5) contains 2,178 NUL bytes | `01` §0 item 6 | Low (not a published recipe) | Note in the release |
| D26 | **RunPod SSH endpoints appear in plain text** in about 69 files under `runs/` and `study/`, plus `docs/RESEARCH_LEDGER.md:853` | `01` §0 item 13 | **Release blocker** | Redact before publishing any Pair-A artifact |
| D27 | **The archive branch `autopareto/research-infrastructure` is local-only.** It holds 877 files absent from the working tree (archive DBs, checkpoint manifests, phase-1 run data) and has never been pushed. | `01` §0 item 1 and §H | **Data-loss risk** | Back up to a private remote before any cleanup; do not `git gc` |
