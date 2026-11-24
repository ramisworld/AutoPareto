# 04 — Reviewer Response Matrix

**Scope.** For each reviewer and area-chair (AC) concern, this file records:
- what the manuscript says now;
- what the existing evidence establishes;
- which wording is unjustified;
- the smallest correction;
- whether a new experiment is needed. Experiment IDs refer to `08_EXPERIMENT_PLAN.md`.

**Page budget.** Part 1 ≤ 4 pages and is currently full (it ends on p. 4). Part 2 ≤ 4 pages, currently about 2.6, so about 1.4 pages are free. Part 3 ≤ 1 page, currently about 0.5. Every Part 1 addition below is paired with a cut.

**Evidence references**
- `03` = `03_CLAIMS_AND_EVIDENCE.md`
- `05` = `05_STATISTICAL_AND_CAUSAL_AUDIT.md`
- `06` = `06_AUTONOMOUS_SEARCH_AUDIT.md`
- `07` = `07_SCALING_FEASIBILITY.md`
- `REPRO` = `supporting/REPRODUCIBILITY_AND_PROVENANCE.md`

## Summary

| # | Concern | Can it be answered with existing data? | New runs needed for camera-ready? |
|---|---|---|---|
| 1 | Confirmation changes several variables at once | **Mostly yes.** Evaluation data is ruled out; seeds are implausible. | **E1** recommended (≈4.5 A40-h) to separate tokens from contract and find the crossover |
| 2 | Selection on noisy discovery scores is optimistic | **Yes**, quantified for the 6 re-run candidates | Optional: E2 |
| 3 | Verification may exclude better recipes | **Partly.** Exclusions are documented; one strong candidate (exp 109) was never tested. | **E4** recommended (≈2.2 A40-h) |
| 4 | Fixed-time vs fixed-token comparison too small | **Yes.** The honest answer is that no contract effect is estimable. | No. Reframe; future E3. |
| 5 | Training-quality variability insufficiently quantified | **Yes**: per-seed values, paired CIs, seed effects | Optional: E5 |
| 6 | Baselines use different spaces and budgets | **Yes**, documentable | No |
| 7 | Search and selection policies unclear | **Yes** (06 Parts A and D) | No |
| 8 | Reproducibility limited | **Partly.** Release records; proprietary agent stays a limit. | No |
| 9 | Hard to read; unexplained terms | **Yes**, editorial | No |
| 10 | AutoPareto not introduced in its own section (AC) | **Yes**, editorial | No |

---

## 1. "The confirmation experiment changes several variables simultaneously"

**Manuscript now**
- §1.3: "Under matched-token retraining on new seeds and held-out data the discovery-stage ordering reverses…"
- §2.8: "The confirmation changes exposure, seed, and evaluation data at once, so the discovery-to-confirmation ranking reversal cannot be attributed to exposure alone."

**What existing data establish** (05 §1; 03 C6–C7)

| Variable | Status | Evidence |
|---|---|---|
| Evaluation data | **Ruled out.** At 201M tokens the reversal is +0.0451 BPB on the same validation data used in discovery, and +0.0420 on held-out data. | `P1JSON` |
| Seeds | **Implausible.** Seed-matched fixed-time re-runs on seeds 0–2 give AP292 − AR = −0.0067 (CI [−0.0130, −0.0003]; 3/3 seeds). A seed account needs a seed × recipe interaction of about 0.05, against observed within-recipe spread ≤ 0.006. | `SNAPDB confirmation_runs` 7–9, 13–15 |
| Discovery luck | **Present but not explanatory.** All re-run candidates were optimistic by 0.0014–0.0087. The paired fixed-time advantage survives. | 05 §3.3 |
| Contract | **Remaining explanation.** Exposure response across 5 recipes: Pearson −0.92 against log token ratio. | 05 §1.4 |
| Tokens vs updates | **Not separable** for this pair (same batch) | — |

**Unjustified wording**
- "cannot be attributed to exposure alone" is now too weak. The data rule out evaluation data and make seeds implausible.
- "AP292 held the better validation BPB during fixed-time discovery" compares seed 0 against seed 42. Replace it with the seed-matched fixed-time result.

**Minimal correction**
- Add a 5-row "reversal decomposition" table to §1.3. It is the table in 05 §1.2 (about 9 lines). Make room by moving the "Identifiers" paragraph and half of "Objectives and measurement" to Part 2.
- Rewrite the §2.8 limitation, for example: "Existing re-runs exclude the evaluation-data change and make seed effects implausible as explanations. Tokens and optimizer updates changed together, so we cannot say which drives the reversal."

**New experiment:** **E1**, the token-response crossover. AR and AP292 under the token contract at 88.6M, 157.3M and 201.3M tokens, plus a 300 s cell, on fresh seeds 6–8. 24 runs, about 4.5 A40-h. It turns "contract" into "exposure", locates the crossover, and removes the seed-set confound. **E1b** (optional, batch 65,536 at fixed tokens) separates tokens from updates.

---

## 2. "Discovery-stage candidate selection may create optimistic results"

**Manuscript now**
- §1.2: "Across those archives AutoPareto attains the lowest archived validation BPB, 1.0933, ahead of AutoResearch (1.0950), TPE (1.0977)…"
- The Figure 1 narrative: "successive matrix, embedding, and output-head learning-rate edits reached 1.0933 at A22."

**What existing data establish**
- All six re-run candidates were optimistic, by 0.0014–0.0087 BPB. Null best-of-25 optimism is about 0.0042 (05 §3.3).
- Top-candidate gaps are ≤ 0.0003 BPB, against a seed SD of 0.0021. Choosing "the best" among the top five is choosing among indistinguishable candidates (05 §3.5).
- The archive list compares seed 0 (AP) with seed 42 (everything else). Seed effects reach 0.024 BPB.
- Seed-matched, AP seed 42 scores 1.0940 vs AR 1.0950: inside noise (03 C8).
- The lineage steps A19 → A20 → A22 are 0.0020 and 0.00009 BPB, i.e. noise-level moves.

**Unjustified wording**
- The four-decimal ranked list in §1.2.
- "Reached 1.0933" as though each LR edit caused an improvement.

**Minimal correction**
- Replace the §1.2 list with one sentence: "All agent and classical archives reach validation BPB within 0.005 of each other at seed 42 (AutoPareto 1.0940, AutoResearch 1.0950, TPE 1.0977, MOTPE 1.0981). Cross-seed comparisons are not interpretable because seed effects reach 0.024."
- Move a full seed-labelled table to Part 2 §2.5.
- In Part 2 §2.5 add: "Every candidate re-run on fresh seeds scored worse than at discovery (by 0.0014–0.0087 BPB). Within-campaign top-5 gaps (≤ 0.002) are inside seed noise (SD 0.0021)."
- Edit the Figure 1 caption to say that steps after A18 are within seed noise.

**New experiment:** none required. **E2** (a stratified re-run of 12 candidates × 3 seeds, about 5 A40-h) would estimate optimism across the rank distribution. Recommended for the follow-up paper; optional for camera-ready.

---

## 3. "Verification may exclude recipes that perform better after confirmation"

**Manuscript now:** §1.2: "three AutoPareto recipes selected before confirmation under the archive's primary roles…"

**What existing data establish** (06 Part D; 05 §3.6)
- **Preregistered rule.** A pooled-per-campaign rule committed 2026-07-27, before AutoPareto ran, chose **11** representatives (06 D.1):
  - exp 103, 108, **109**, 124, 134, 136, 137, 147, 152, 158, 162.
- **11 → 4 (2026-08-10).** A global selector applied *after* all discovery cut this to 4, all seed 0: exp 124, 134, 136, 137 (06 D.2). This departs from the preregistered rule.
- **4 → 3 (around 2026-08-21/25).** The 201M panel then dropped exp 124 (lowest memory within 1%) with no recorded rule, after the seeds-0–2 results were known (06 D.3).
- **Exp 109 and siblings (AP seed 42).** Full 50.3M model, batch 65,536. Validation 1.0940 on 84.1M tokens, *fewer* than AR's 88.9M at the same seed. Preregistered as a representative, then cut, and never confirmed. It is the most plausible exposure-neutral AutoPareto quality candidate.
- **AR exp 65** (lowest BPB) was excluded for inference incompatibility, judged under a reference later deemed faulty (06 D.4). Its fixed-time re-run quality equals exp 68's (1.1017 vs 1.1014), so the exclusion did not change AR's quality.

**Unjustified wording**
- "selected before confirmation" applies only to the seeds-0–2 confirmation.
- The panel's role subset was chosen with those confirmation results known.

**Minimal correction:** a "selection funnel" paragraph and table in Part 2 §2.5 (about 10 lines):
- 75 AutoPareto attempts → 11 preregistered representatives → 4 by the post-discovery global selector → 3 in the 201M panel.
- AutoResearch: exp 65 → exp 68.
- State dates and reasons. Name the dropped roles and exp 109.

**New experiment:** **E4**. Retrain exp 109 (and optionally exp 111) at 201M tokens on seeds 3–5, directly comparable to the existing panel, plus at 88.6M and 157.3M tokens on seeds 6–8 to join E1. 9–12 runs, about 2.2 A40-h.
- *If exp 109 beats AR at matched tokens:* the paper gains an exposure-controlled AutoPareto quality result and a stronger story.
- *If not:* the negative result stands with the selection question closed.

---

## 4. "The fixed-time vs. fixed-token comparison is too small as reported"

**Manuscript now:** §1.1 H2; §1.3 "For H2 we report the first four completed attempts of each arm…"; Table 2.

**What existing data establish** (06 Part B; 05 §5)
- There are **five** Pair-A lineages, with protocol drift: N* = 88.6M vs 85.5M, different A40s, adapter amendments, a timeout change.
- The paper's lineage reaches 8 attempts per arm. Only 4 existed at the 2026-09-06 evidence cutoff.
- **2 of the 4 reported fixed-time attempts are harness-invalid** (BPB 3.2159), as are 6 of 8 in the continuation. Cause:
  - `study/contracts_2026/candidate_process.py:660` re-executes the training body when the candidate's own loop exits first. A freshly initialized model is checkpointed (candidate-reported step 0 vs 1,264 observed updates).
  - The controller still marked these runs `quality_valid … parent_eligible: true`.
  - The agent then read the bogus collapse as LR instability.
- Batch-reduction behaviour is inconsistent across lineages. The 22-attempt "fixed-token agent never changed batch size" pattern occurs in one restart only.
- In the later restarts (G3 and G4), most fixed-token candidates kept the time-based learning-rate schedule. The protected hook does not enforce token progress. Their quality outcomes are therefore confounded (`05` §5). The paper's prefix (G2) is not affected.

**Unjustified wording**
- Presenting the prefix as descriptive evidence about contracts without disclosing that half the fixed-time arm is invalid.
- "the fixed-time trajectory keeps the 50.33M-parameter architecture and changes batching and learning rate". It also crashed into the harness bug.

**Minimal correction**
- **Move H2 out of Part 1.** Make Part 1 about H1 and the verified trade-off, with the reversal decomposition.
- In Part 2 add a short "Pair-A pilot" paragraph that reports:
  - the harness defect and its signature;
  - that only 2 fixed-time attempts in the reported prefix are valid;
  - that later restarts disagree on agent behaviour;
  - that no contract effect can be estimated from one adaptive trajectory per arm.
- Keep H2 as a stated hypothesis for future work.
- **Do not** cite the 22-attempt restart as evidence of a contract effect.

**New experiment:** none for camera-ready. A proper test is **E3**:
- at least 5 seed-paired campaigns per contract under one frozen, bug-fixed harness;
- about 100 attempts in total;
- about 1M input and 14k output agent tokens per attempt (from Pair-A transcripts).

This is follow-up-paper work.

---

## 5. "Training-quality variability is insufficiently quantified"

**Manuscript now**
- Table 1 caption: "entries are means over n = 3 training seeds"
- §2.8: "n = 3 supports point estimates with sample SD rather than significance claims."

**What existing data establish** (05 §2)
- Per-seed values exist for every panel run.
- Paired 95% CIs (n = 3, t with 2 df):
  - held-out cost [0.0297, 0.0544];
  - validation cost [0.0415, 0.0486];
  - fixed-time advantage [−0.0130, −0.0003].
- Pooled within-recipe SD: 0.0021 (fixed time) and 0.0037 (held-out at 201M).
- Seed main effects are large: seed 42 baseline −0.023; seed 3 −0.015.
- Same recipe and seed, different session: tokens vary up to 18.7% under the time contract.
- Decode ratio per seed: 1.969, 1.958, 2.029.

**Minimal correction**
- Add ±SD columns to Table 1, or a footnote with per-seed headline values.
- Report the paired CIs for the headline contrasts in §1.4.
- Add one sentence on seed main effects and session-throughput noise under the time contract (this supports the paper's thesis).
- Keep "no significance claims beyond the headline".

**New experiment:** optional **E5** (seeds 6–8 for AP292 and AR at 201M; 6 runs, about 1.3 A40-h). n = 3 already resolves the 0.042 gap (t = 14.7). Only worth running if the discussant is likely to insist.

---

## 6. "Baseline comparisons may involve different search spaces and budgets"

**Manuscript now**
- §1.2: "The AutoResearch campaigns differ from AutoPareto in prompt text, commits, wrappers, and returned feedback…"
- §1.2: "the archives differ in space, budget, seed, and failure history…"

**What existing data establish** (06 A.1–A.3, A.6; 03 C8)

| Campaign | Attempts | Seeds |
|---|---|---|
| AutoPareto | 75 | 0, 1, 42 |
| AutoResearch (main + diagnostic) | 50 + 5 | 42 |
| Classical searches | 25 each | 42 only; seeds 0 and 1 registered but never run |

- The classical searches are fixed-vector spaces. The agents edit free-form code.
- 27 of AR50's trained candidates were discarded on a faulty inference check (06 A.1).
- The paper already disclaims superiority, which is correct.

**Unjustified wording:** listing all archives on one ranked line (see concern 2).

**Minimal correction:** a compact "comparator budget" table in Part 2. Columns:
- method;
- seeds run;
- attempts;
- completed with BPB;
- search space (fixed-vector dimensionality vs free-form code);
- program hash;
- inference-gating rule.

Also qualify "quality-only AutoResearch" as described in 06 D.4.

**New experiment:** none. A fair tournament (seed-matched, equal budget, repeated) is follow-up work.

---

## 7. "Autonomous search and candidate-selection policies need clearer descriptions"

**Manuscript now:** §2.3 Research Loop describes parent eligibility in one dense paragraph. The historical agent identity is given as "Codex CLI 0.153.0 running gpt-5.6-sol".

**What existing data establish** (06 Parts A, C, D)
- Parent rules, Pareto logic, failure handling and selection roles are reconstructable from code.
- **The Codex 0.153.0 / gpt-5.6-sol description is verified only for Pair-A.**
  - For the historical campaigns, including AP292's discovery, the model labels are runner defaults (`gpt-5.6-sol`/`high`), recorded even for classical searches.
  - The handoff allowed "Codex or Cursor".
  - No transcripts or token records exist for the 75 AutoPareto attempts (06 A.5).

**Unjustified wording**
- Stating the historical agent's identity as fact.
- "fresh ephemeral container for every attempt" for the historical campaigns. Verified for Pair-A only.

**Minimal correction**
- Add an "AutoPareto loop" box in Part 2 §2.3 (pseudo-code, about 12 lines): stage parent → agent proposes hypothesis, patch and category → validate hashes → train under contract → protected evaluation → record → update archive → choose next parent (rule), plus how failures are recorded.
- Give per-era agent identity: Pair-A verified; historical self-declared, Codex or Cursor permitted, model label unverified.
- Add the selection funnel (concern 3).
- Add the seed-42 rollback and replay to §2.4 "Human interventions" (06 C.1).

**New experiment:** none.

---

## 8. "Reproducibility is limited by missing raw records and other artifacts"

**Manuscript now:** Appendix A: "checkpoints, raw logs, the private database, and the dataset cache are excluded."

**What existing data establish:** see `REPRO` for the per-artifact classification and the public-release diff, and 01 for the full inventory.
- Much of the missing evidence exists locally and can be released without new runs:
  - an anonymized per-experiment registry (`02_EXPERIMENT_REGISTRY.csv`);
  - the fixed-time confirmation and ablation records;
  - the 21-run panel JSON;
  - Pair-A histories, including failures;
  - the selection files (`results/autopareto_confirmation_selection.json`, `results/autopareto_primary_confirmation_plan.json`, the phase-1 preflight);
  - the audit scripts.

**Minimal correction:** expand the public artifact (ranked list in `REPRO` §F and `09` §Appendix). Update Appendix A to list exactly what is released and what remains private: provider-side agent state, the dataset cache, and large checkpoints.

**New experiment:** none.

---

## 9. "The manuscript is difficult to understand and uses unexplained terminology"

**Problem terms** (from `paper/main.tex`)
- resource contract
- exposure / matched-exposure
- registered
- protected
- bracketed protocol
- quality-valid
- speed-uncertain
- Pareto-eligible
- exclusive hypervolume contribution
- AP292 / AP294 / AP299 (numerals inherited from database rows)
- A22 / exp 134 / archive row 292
- steady s (est.)
- campaign view
- arm

**Minimal correction**
1. A 10-entry glossary box in Part 2, or as a footnote block in Part 1:
   - **contract** = the rule that stops training (fixed time or fixed tokens);
   - **exposure** = training tokens processed;
   - **matched-exposure** = retrained to the same token count;
   - **Pareto-eligible**, **campaign**, **attempt**, **parent**: define each.
2. **Rename recipes descriptively**, with the old IDs kept once for traceability:
   - AP292 → **AP-Q** (quality);
   - AP299 → **AP-F** (fast);
   - AP294 → **AP-S** (small);
   - AR → **AutoResearch-ref**.
3. Remove internal numerals from Part 1 except in the Identifiers table.
4. Replace "steady s (est.)" with "training time for 201M tokens (s, estimated)".
5. Split long multi-clause sentences in §1.1–1.2.

**New experiment:** none.

---

## 10. AC: "AutoPareto itself was not introduced in a sufficiently clear section"

**Manuscript now:** Part 1 opens with "1.1 Hypothesis". AutoPareto is defined only in one abstract sentence and in Part 2.

**Minimal correction:** add **"1.0 AutoPareto in brief"** (about 120 words) as the first subsection of Part 1. Draft:

> *AutoPareto is an autonomous research loop for small language models. In each attempt, an LLM coding agent reads the results so far, picks a parent training script, states a hypothesis, and edits one file (`train.py`). A protected harness the agent cannot modify trains the edited model on one A40 under a fixed resource contract (300 s of training in our historical campaigns). It then measures three objectives: validation bits-per-byte (quality), cached-decode tokens/s (speed), and peak inference memory. Results are appended to a lineage archive that keeps failures. The agent sees only its own campaign's history. We ran three 25-attempt campaigns and compared them with a quality-only AutoResearch agent and five classical optimizers.*

Also point to the loop figure (Figure 3, Part 2) from this paragraph.

**Space:** cut "Identifiers" (moved to the glossary) and compress "Objectives and measurement" in §1.2 to two sentences. Protocol details move to Part 2 or the appendix.

---

## Cross-cutting: statements that should not survive camera-ready

1. Any implication that the archive ranking shows AutoPareto's quality superiority (cross-seed).
2. "Selection preceded confirmation" without scoping it to the seeds-0–2 confirmation.
3. The historical agent identity stated as verified.
4. "Source-diff audit labels" (it is the recorded `change_category` field) (03 C13).
5. The H2 prefix as described, with the harness-invalid attempts undisclosed.
6. "Median of three seed-3 baseline runs" for N*, unless the calibration record is cited. 06 C.1 places three calibration runs in `docs/RESEARCH_LEDGER.md` lines 640–669; cite that, or the manifest.
7. "Best completed attempt of the quality-only AutoResearch archive" without "best inference-compatible attempt of a 5-attempt follow-up diagnostic".
