# 10 — Executive Brief: AutoPareto camera-ready audit

**For:** the research team and the advising model (ChatGPT). Readable on its own.
**Date:** 2026-10-28.
**Camera-ready deadline:** 2026-11-13 (NeurIPS 2026 AutoMLR workshop, poster).
**Method:** a read-only audit. No experiment, GPU job, paid API call, or edit to an existing artifact. Every number traces to a file, record, or script listed in `AUDIT_INDEX.md`.

## What AutoPareto is (for a newcomer)

- An LLM coding agent repeatedly edits one training script (`train.py`) for a small GPT-style language model.
- Each edit is trained under a fixed resource contract on one NVIDIA A40. In the historical campaigns the contract is 300 seconds of training.
- Protected code the agent cannot modify scores each run on validation bits-per-byte (BPB; quality), cached-decode tokens/s (speed), and peak inference memory.
- There were three 25-attempt campaigns (seeds 0, 1, 42).
- The paper's best recipe, called **AP292** here and **AP-Q** in the proposed revision, was retrained with 6 other recipes for exactly 201,326,592 tokens on seeds 3–5.
- **Paper's claim.** AP292 keeps 1.99× decode speed, 41.7% fewer parameters and 35.1% less memory than the quality-only AutoResearch recipe (**AR-ref**). Its discovery-stage quality advantage does not survive equal-token retraining (held-out BPB 1.1418 vs 1.0998).

## The 10 most important findings

1. **The headline numbers are exactly right.**
   - All 35 cells of Table 1, the 1.99× / 41.7% / 35.1% / 50% KV headline, the 0.042034 BPB cost, the downstream scores, the 51/67 BLiMP count, and r = −0.969 / 0.717 all reproduce from raw records.
   - All 21 panel runs pass integrity checks.
   - Sources: `03` §A–B; `supporting/REPRODUCIBILITY_AND_PROVENANCE.md`.
2. **The quality reversal is real, and the existing data already explain most of it.**
   - Seed-matched fixed-time re-runs, which are not in the paper, show **AP292 beats AR-ref at equal wall-clock time**: −0.0067 BPB, seeds 0–2, all three seeds agree, 95% CI [−0.0130, −0.0003].
   - At equal tokens, AR-ref wins by +0.045 on the *same validation data* and by +0.042 on held-out data, again on all three seeds.
   - So the evaluation split does not explain the flip, and seed variation (≤ 0.006) is far too small to. The contract change does: tokens and updates moved together.
   - Source: `05` §1.
3. **The quality advantage was mostly "more data per second".**
   - Under the 300 s contract, AP292 trained on about 157M tokens against AR-ref's 88.5M.
   - Across five recipes, the BPB change from fixed time to 201M tokens tracks the token ratio (Pearson −0.92).
   - The AutoPareto recipes are essentially compute-optimal model sizes for a 300 s budget: three sizes of one recipe (`07` §A).
   - The model-shrinking edits are neutral or harmful per token.
4. **Seed effects swamp the paper's discovery-stage comparisons.**
   - The unchanged baseline differs by 0.024 BPB between seeds 0 and 42.
   - The paper's ranked archive list (AutoPareto 1.0933 < AutoResearch 1.0950 < …) compares seed 0 with seed 42.
   - Seed-matched, AutoPareto seed 42 scores 1.0940 vs AutoResearch's 1.0950: inside noise.
   - Source: `03` C8.
5. **Selection near the top is noise.**
   - Within-campaign top-5 gaps are ≤ 0.002 BPB, against a seed SD of 0.0021.
   - Every candidate re-run on fresh seeds was worse than at discovery, by 0.0014–0.0087.
   - A natural replicate shows two identical models in the archive differing by 0.0014 BPB and 10% in measured decode speed. The archive kept the noise "trade-off".
   - Source: `05` §2.5, §3.
6. **Selection narrowed after results were known, and dropped a promising candidate.**
   - A preregistered rule, committed before AutoPareto ran, chose 11 representatives.
   - After discovery, a global selector cut them to 4 (all seed 0). The 201M panel then dropped one more role (exp 124) with no recorded rule.
   - **Exp 109**, a full-size AutoPareto recipe from seed 42, scored 1.0940 on *fewer* tokens than AR-ref at the same seed. It was preregistered and never confirmed. It is the most plausible *exposure-neutral* AutoPareto quality gain.
   - Sources: `06` Part D; `05` §3.6.
7. **AutoResearch's entire advantage is one hyperparameter.**
   - In the July ablation (seeds 0–2), changing only the batch size from 2¹⁹ to 2¹⁷ in the baseline yields 96–98% of AR-ref's gain.
   - AR-ref's other edits alone make the baseline slightly worse.
   - The batch gain also shrinks with more training: the AR–baseline validation gap is −0.115 at 300 s (seeds 0–2, paired) and −0.061 at 201M (seeds 3–5, paired). See `05` §3.6 and `07` B1; `07` quotes −0.107 at 300 s using a different baseline set.
8. **The reported fixed-time vs fixed-token prefix (H2) contains harness-invalid runs.**
   - 2 of the 4 reported fixed-time attempts (BPB 3.2159) are untrained models.
   - Mechanism: the worker re-executes the training code when the candidate's own loop exits first (`study/contracts_2026/candidate_process.py:660`), and a fresh model is checkpointed.
   - Signature: candidate-reported step 0 vs 1,264 observed updates.
   - The controller marked these runs quality-valid and parent-eligible, and the agent misread them as learning-rate instability.
   - The paper does not disclose this.
   - Source: `06` §B.1.
9. **Pair-A is five separate restart lineages, not one long campaign.**
   - The 22-attempt histories come from a later restart (new A40, different token budget 85.46M vs 88.60M, new programs). They are not a continuation of the paper's prefix.
   - Batch-size exploration varies between lineages in both arms.
   - **No contract effect on agent behavior is estimable from existing data.** This corrects an earlier, unaudited claim that the fixed-token agent systematically "never explores batch size".
   - Sources: `05` §5; `06` §B.2–B.8.
10. **Provenance gaps the paper overstates.**
    - The historical agent identity ("Codex CLI 0.153.0, gpt-5.6-sol") is a runner default label, verified only for Pair-A. No transcripts exist for the 75 AutoPareto attempts.
    - The seed-42 AutoPareto campaign was partly rolled back and re-run (ids 88–90 deleted) without disclosure.
    - Figure 1's width labels for A07–A10 are wrong.
    - Training data was read from an unpinned dataset revision.
    - Python was 3.10.18, not 3.12.
    - All 21 panel checkpoints survive only as git LFS objects. The working-tree copies are incomplete, and AR-ref's own discovery checkpoint is missing.
    - The archive branch holding the historical databases is local-only and has never been pushed.
    - Which PDF was actually submitted cannot be confirmed locally.
    - Sources: `06`; `03` §E; REPRO.

## Solid claims vs. weaker than we thought

| Solid | Weaker than stated |
|---|---|
| The efficiency trade-off at matched tokens (1.99× decode, −41.7% params, −35.1% memory, −50% KV) | "AutoPareto attains the lowest archived validation BPB" (cross-seed; inside noise when seed-matched) |
| The quality cost at matched tokens (+0.042 held-out; CI excludes 0) | "Selection preceded confirmation" (true only for the seeds-0–2 confirmation; the panel subset was chosen afterwards) |
| H1: the discovery advantage does not survive matched exposure | The H2 prefix description (half the fixed-time arm is invalid) |
| Downstream pattern and BLiMP correlations (descriptive) | Historical agent identity and container description |
| Every Table 1 value | "Source-diff audit labels" (the agent's own category field) |
| The AP292 lineage itself has no recorded human intervention | Figure 1 width labels; Appendix warm-up sentence; Python version; "every table regenerated" |

## Valuable historical evidence missing from the paper

- **Fixed-time, seed-matched re-runs** of AP292, AR-ref, AP299, AP294 and exp 124 on seeds 0–2. These give the time-contract half of the reversal.
- **Validation BPB at 201M.** This shows the reversal is not caused by the held-out split.
- **The July batch ablations.** They show AR-ref's gain is the batch size.
- **Baseline seed runs (seeds 0–4 and 42).** They quantify the seed effects.
- **The A09/A10 natural replicate** (noise floor).
- **The selection funnel:** 11 → 4 → 3.
- **The full Pair-A lineages,** including the harness defect.

## Reviewer criticisms answerable without new experiments

Details are in `04`.

| Concern | Status without new runs |
|---|---|
| #2 selection optimism | Answerable |
| #4 Pair-A too small | Answerable, by reframing it as a pilot and disclosing |
| #5 variability | Answerable |
| #6 baselines | Answerable |
| #7 policy descriptions | Answerable |
| #8 reproducibility | Answerable, by releasing records already on disk |
| #9 clarity | Answerable |
| #10 AutoPareto introduction | Answerable |
| #1 confounded confirmation | **Mostly answerable** (evaluation split ruled out; seeds implausible). E1 finishes it. |
| #3 excluded recipes | **Partly answerable** (funnel documented). E4 tests the one recipe that matters. |

## Minimum additional experiments for camera-ready

Details are in `08`. All are GPU-only; none needs agent API calls.

| ID | What | Size and cost | Status |
|---|---|---|---|
| E0 | Reproduce one panel cell and one fixed-time cell in the restored environment | 2 runs, 0.5 A40-h | Halt if the deviation exceeds 0.005 BPB |
| E1 | Token-response crossover: AR-ref vs AP292 at 88.6M / 157.3M / 201.3M tokens plus 300 s, fresh seeds 6–8 | 24 runs, about 4.1 A40-h, about $1.80 | Essential |
| E4 | Exp 109 (exposure-neutral AutoPareto candidate) at 201M tokens, seeds 3–5, plus two budgets on seeds 6–8 | 9 runs, about 2.2 A40-h, about $1 | Essential; the only experiment that can add a positive quality result |
| E1b | Tokens vs updates | 6 runs | Optional |
| E2 | Selection-bias re-runs | 36 runs | Optional |
| E6 | 4× scaling pilot | about 18 A40-h; cap 22; cutoff Nov 6 | Optional |

**Preconditions**
- Restore the A40 environment from the 2026-09-17 snapshot.
- Pin the dataset revision.
- Pre-register.
- Add integrity gates, including checkpoint step = observed updates.

## Strongest achievable new scientific contribution (honest)

**For the camera-ready.** "The training contract decides which agent-discovered recipe wins." The paper can show:
- a seed-controlled and evaluation-controlled reversal: AP292 better at equal time, worse at equal tokens;
- its mechanism, exposure, which the agent bought by choosing compute-optimal small models for 300 s;
- with E1, the measured token crossover.

Two further pieces:
- **E4 could add a positive result:** an exposure-controlled quality gain from AutoPareto's full-size seed-42 recipe, flagged as a post-hoc test of a preregistered representative.
- **The harness failure is itself a contribution** for the workshop's verification theme: a protected harness accepted an untrained checkpoint as quality-valid, and the agent built hypotheses on it.

**For the follow-up paper** ("Do AI-discovered optimizations survive scaling?"): the archive already suggests the answer is "mostly not, by construction".
- The agent's gains decompose into update count, which shrinks with training; compute-optimal sizing, which moves with compute; and noise-level LR tweaks.
- A preregistered promotion-ladder study (`07` §F; about 860–1,190 A40-h) with stratified selection-bias measurement (E2) would make this rigorous.

## Is scaling transfer feasible near-term?

**Yes as a research direction, no as a camera-ready claim.**
- Compute is cheap: promoting six recipe families to 4/16/64× C0 costs about 130 A40-h.
- The constraints are engineering and conceptual.
  - The runtime is hard-coded to the 201M plan.
  - The tokenizer is not local, and the data cache size is unverified.
  - The three AutoPareto recipes are sizes of one recipe, which makes a "promotion operator" ambiguous (depth dial vs width dial).
- Only a 4×-token pilot (E6) fits before Nov 13, and only after E0/E1/E4.

## Issues that could invalidate existing conclusions

- **None invalidates the efficiency headline or the matched-token quality cost.** Both reproduce, and both are deterministic or resolved at n = 3.
- **Invalid as written:** any interpretation of the H2 prefix. It contains harness-invalid runs.
- **Could change the narrative:**
  - exp 109 (E4) may show AutoPareto *did* find an exposure-neutral quality gain;
  - E1 may show AP292's fixed-time advantage was *entirely* exposure.
- **Undermine autonomy and provenance statements, not numbers:**
  - the unverifiable historical agent identity;
  - the undisclosed seed-42 rollback;
  - post-hoc selection narrowing.

## Missing or inaccessible evidence

Details are in `01` and REPRO.
- The RunPod volume snapshot `/workspace/PRESERVED_20260917T021347Z` is remote, and the pod is stopped. It holds the dataset cache, tokenizer, 11 of the 21 panel checkpoints, milestone checkpoints, and executed Pair-A harness files.
- Provider-side agent state and historical agent transcripts: none exist for the 75 AutoPareto attempts.
- Per-item downstream outputs.
- The OpenReview reviews were not machine-readable to this audit; the concerns listed by the team were used.
- The three-run N* calibration record (cited in `docs/RESEARCH_LEDGER.md` lines 640–669) is not stored as a data file.

## Action plan through 2026-11-13

1. **Oct 28–29.**
   - **Protect the evidence first.**
     - Back up the local-only `autopareto/research-infrastructure` branch and `.git/lfs` to a private remote.
     - Do not run `git gc` or other cleanup until that is done.
     - Commit or revert the uncommitted edits to `protected/inference_adapter.py`.
   - Download the exact submitted PDF from OpenReview and diff it against local builds.
   - Reconcile the manuscript source: the submitted abstract lives only in `paper_backups/automlr_overleaf_reconstructed/`.
   - Fix the Figure 1 labels and the other mechanical errors (`09` §0.4).
   - Restore the environment; pin the data revision; pre-register E1 and E4.
2. **Oct 30.** E0 smoke test. Stop if it fails.
3. **Oct 30 – Nov 1.** E1 and E4 (33 runs, about 6.3 A40-h).
4. **Nov 2.** Optional E1b or E2.
5. **Nov 3–6.** Analysis and new figure. Rewrite Part 1 (`09`): add "AutoPareto in brief", the reversal table or crossover figure, and the new §1.4. Run E6 only if all of this is done.
6. **Nov 7–10.**
   - Human rewrite of Part 2: agent identity per era; interventions, including the seed-42 rollback and the harness defect; selection funnel; Pair-A pilot section; glossary.
   - Release the public artifacts (REPRO §F).
7. **Nov 11–12.** Compile, check per-part page limits, run PDF QA, and copy to `/Users/Rami/Documents/dev/autopareto_automlr_FINAL.pdf`.
8. **Nov 13.** Submit.
