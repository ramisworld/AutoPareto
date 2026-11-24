# 09 — Paper Revision Plan (camera-ready, due 2026-11-13)

**Principles**
1. Keep the accepted contribution: matched-exposure verification of an agent-discovered recipe, and the claim that the contract determines what is "discovered". Strengthen and correct it; do not replace it.
2. Every sentence must be backed by `03` (claims), `05` (statistics) or `06` (search and selection).
3. Respect the workshop structure: Part 1 ≤ 4 pp (the autonomous result), Part 2 ≤ 4 pp (human-written system design), Part 3 ≤ 1 pp. Page accounting is given at the end.
4. Prefer plain language. Use one name per object.

Sections below give **REPLACE / ADD / DELETE / MOVE** instructions against `paper/main.tex` and `paper/generated/*.tex`. Draft text is a suggestion; the human author must rewrite and verify Part 2, as the workshop requires.

---

## 0. Global edits

### 0.1 Naming

Use these names everywhere, and keep the old ID once in the glossary.

| Object | New name | Old IDs |
|---|---|---|
| AutoPareto quality recipe | **AP-Q** | A22, exp 134, AP292 |
| AutoPareto fast recipe | **AP-F** | A25, exp 137, AP299 |
| AutoPareto small recipe | **AP-S** | A24, exp 136, AP294 |
| AutoResearch reference | **AR-ref** | exp 68 |
| Contracts | **fixed-time** (300 s) and **fixed-token** (N* tokens) | "T/N arms" in Part 2 only |

### 0.2 Glossary box

Put the glossary in Part 2, before §2.1, with one line per term:
- contract
- exposure (training tokens processed)
- matched-exposure retraining
- campaign
- attempt
- parent
- Pareto-eligible
- held-out vs validation BPB
- cached decode
- seed

### 0.3 Abstract

REPLACE `generated/abstract.tex`. Keep it to about 180 words.

> *Autonomous ML researchers are usually scored under a fixed wall-clock budget, where a faster training program also sees more data before it is evaluated. We study what this does to discovery in AutoPareto, an agent that edits a small Transformer's training script and is scored on validation bits-per-byte, cached-decode speed, and inference memory on one A40. The agent's best recipe (AP-Q) beats a quality-only AutoResearch recipe at equal training time on three fresh seeds (−0.007 BPB). At equal training tokens and updates, it loses by 0.042 held-out BPB (95% CI 0.030–0.054). The flip appears on both validation and held-out data and far exceeds seed variation, so the resource contract, not noise or the evaluation split, decides which recipe looks better. AP-Q's efficiency gains survive matched exposure: 1.99× decode throughput, 42% fewer parameters, 35% less inference memory. Auditing the full archive also shows that seed effects (up to 0.024 BPB) dwarf discovery-stage gaps, and that time contracts tie measured quality to hardware throughput. We argue the contract belongs in the methodology of autonomous ML research.*

If E1 is run, replace "the resource contract … decides" with the measured crossover statement.

### 0.4 Mechanical corrections

These are must-do factual fixes, independent of new experiments. IDs refer to the discrepancy log in `03` §D–E.

1. **Reconcile the manuscript source before editing anything (D18).**
   - The submitted abstract exists only in `paper_backups/automlr_overleaf_reconstructed/`.
   - `paper/generated/abstract.tex` differs from it, and `paper/refresh_paper.sh` would overwrite the submitted wording.
   - Create one camera-ready source of truth first.
2. **Figure 1 labels (D9).** Generate the edit labels from the frozen sources, using `build_model_config` rounding, instead of the hand-typed dictionary in `analysis/generate_figures.py`. Correct labels:
   - A07: "depth 6→5 (width stays 384)"
   - A08: "depth 5→4; width 384→256"
   - A09: "aspect 64→80; width 256→384"
   - A10: "aspect 80→96 (no effective change)"
3. **Figure 2 caption (D13).** Delete "the horizontal bars are narrower than the markers". It is false for AP299 and MOTPE.
4. **Appendix A warm-up sentence (D11).** Replace "counted tokens exclude the eleven disposable warm-up updates" with what the panel actually did. The first 11 updates are excluded only from the steady-state *timing* estimate (see `P1JSON timing_note`).
5. **Environment (D12).** Per-era Python, PyTorch, CUDA and driver table. The panel used Python 3.10.18.
6. **Dataset revision (D15).** State that training data was read from the `main` branch of `karpathy/climbmix-400b-shuffle`. Only the held-out shard's revision is pinned.
7. **"Every table and vector figure … is regenerated" (D14).** Narrow it to the generated items. Table 2, Figure 3 and the Figure 1 labels are hand-authored.
8. **A17 (D17).** Describe it as "an infrastructure failure (a missing tokenizer file), preserved in the record". Note that the identical failure in seed 42 was not counted as an attempt.
9. **Selection timing (D19).** "Campaign identifiers dated 2026-08-10 (plan committed 2026-08-10), created in the database 2026-08-11/12 UTC".

---

## Part 1 — Autonomous research result (≤ 4 pp; currently full)

### §1.0 AutoPareto in brief — ADD (new first subsection; about 120 words)

Use the draft in `04` concern 10. It answers the area chair directly. End it with "Figure 3 (Part 2) shows the loop and its trust boundary."

### §1.1 Hypothesis — TIGHTEN

- Keep H1, but replace "need not persist" with a falsifiable form: *"H1: a recipe's discovery-stage advantage under the fixed-time contract may reverse when competing recipes are retrained to equal training tokens."*
- Keep the disclosure that H1 is retrospective.
- **H2:** replace the paragraph with one sentence: *"A prospective test of whether the contract also changes which hypotheses the agent pursues (H2) was piloted; the pilot is reported in Part 2 (§2.9) because it is too small and, for the fixed-time arm, partly invalid."*

### §1.2 Method — COMPRESS

**"Task and discovery contract."** Keep. Add: *"Campaigns used training seeds 0, 1 and 42; seed effects on the unchanged baseline reach 0.024 BPB (Part 2, §2.5), so we compare recipes only on shared seeds."*

**"Objectives and measurement."** REPLACE with two sentences:

> *Validation BPB measures quality. Cached-decode tokens/s measures speed (BF16, batch 1, 256 prompt + 256 generated tokens, median of 30 repetitions). Peak allocated CUDA memory measures inference memory (weights included). Full protocol: Appendix A.*

MOVE the rest to Appendix A.

**"Comparators and related work."**
- Keep the related-work sentences.
- REPLACE the ranked list ("AutoPareto attains the lowest archived validation BPB, 1.0933, ahead of …") with:

> *At the shared seed 42, the best AutoPareto, AutoResearch, TPE and MOTPE recipes lie within 0.004 BPB of each other (1.0940, 1.0950, 1.0977, 1.0981). The archives differ in search space, budget and failure history, so we draw no search-policy conclusion from them.*

**"Matched-exposure confirmation."**
- Keep.
- After "…three AutoPareto recipes", add a pointer: *"(the selection funnel from 75 attempts to these recipes is in Part 2, §2.5)."*
- Replace "the best completed attempt of the quality-only AutoResearch archive" with: *"the best inference-compatible recipe from AutoResearch's follow-up diagnostic campaign (AR-ref)."*

**"Identifiers."** DELETE; it moves to the glossary.

### §1.3 Evidence

**Figure 1 (lineage)**
- Keep. Reduce the width to 0.70 if space is needed.
- Caption ADD: *"Steps after A18 change validation BPB by less than one seed SD (0.002)."*

**"What the agent searched."**
- REPLACE "the source-diff audit labels" with "the recorded change categories are".
- REPLACE "successive matrix, embedding, and output-head learning-rate edits reached 1.0933 at A22" with "learning-rate edits then moved BPB within seed noise to 1.0933 at A22".

**NEW Table (insert before "Matched-exposure outcome")**

Title: *"Which variable flips the ranking? AP-Q minus AR-ref, BPB; negative = AP-Q better."*

| Contract | Seeds | Eval data | Δ BPB | 95% CI |
|---|---|---|---|---|
| 300 s (archive) | 0 vs 42 | val | −0.002 | — |
| 300 s re-run | 0, 1, 2 | val | **−0.007** | [−0.013, −0.0003] |
| 201M tokens | 3, 4, 5 | val | **+0.045** | [+0.042, +0.049] |
| 201M tokens | 3, 4, 5 | held-out | **+0.042** | [+0.030, +0.054] |

Values come from `05` §1.2. If E1 is run, REPLACE this table with the token-response figure (see "If new experiments are run" below).

**"Matched-exposure outcome."** REPLACE the second and third sentences with:

> *At equal training time on fresh seeds, AP-Q is better than AR-ref on all three seeds. At equal tokens and updates (1,536 at batch 131,072), it is worse on all three seeds, on the same validation data used during discovery and on held-out data. Recipe-level seed variation is at most 0.006 BPB, an order of magnitude smaller than the flip, and the flip persists on the discovery validation data, so neither seeds nor the evaluation split explain it. Tokens and update counts changed together, so we do not attribute the flip to one of them.*

Keep the efficiency sentence. Add the per-seed decode ratios (1.97, 1.96, 2.03).

**"Downstream and budget checks."**
- Keep the downstream sentences.
- ADD: *"ARC-E, PIQA and LAMBADA differences are within about one point, comparable to the ARC-E binomial SE (≈1.0)."*
- DELETE the H2 sentences; they move to §2.9.

**Figure 2 (Pareto)**
- Keep.
- Caption ADD: *"Six of seven recipes are nondominated on every seed; we therefore report trade-off sizes rather than frontier membership."*

### §1.4 Primary Result — REPLACE `generated/primary_result.tex`

> *Which recipe "wins" depends on the training contract. The agent-discovered recipe AP-Q beats AR-ref at equal wall-clock training time (−0.0067 validation BPB; seeds 0–2; all seeds agree). It loses at equal training tokens and updates (+0.042 held-out BPB, 95% CI 0.030–0.054; seeds 3–5; all seeds agree). The flip is not explained by the evaluation split or by seed variation. What survives every contract is efficiency: at matched tokens AP-Q delivers 1.99× cached-decode throughput (per-seed 1.96–2.03), 41.7% fewer parameters, 35.1% less peak inference memory and half the KV-cache memory. The agent's autonomous contribution is the source-edit lineage that produced AP-Q; matched-exposure retraining established which part of its discovery-stage score the edit earned and which part the contract supplied.*

---

## Part 2 — System design and meta-analysis (≤ 4 pp; about 1.4 pp free)

### §2.1 Agent and Harness — CORRECT

- Pair-A: keep the Codex CLI 0.153.0 / gpt-5.6-sol / high / 600 s description, and add the 900 s timeout change in restart 2 (06 A.5).
- Historical campaigns: *"run by a repository-level coding session (the handoff permitted Codex or Cursor). The recorded model label (gpt-5.6-sol, high) is a runner default and is not independently verified. No transcripts survive for these campaigns."* Source: 06 A.5.

### §2.2 Tools — CORRECT

Give environment pins per era. The July runs used a Python 3.10 virtualenv. See `REPRO` §D for exact pins.

### §2.3 Research Loop — ADD an algorithm box (about 12 lines)

1. Stage parent and campaign view.
2. Agent returns hypothesis, patch and category.
3. Validate hashes and the editable surface.
4. Train under the contract.
5. Run the protected BPB, decode and memory evaluation.
6. Append an immutable record.
7. Update the archive using the Pareto rule (quality-valid; speed-certain for Pareto eligibility).
8. Choose the next parent by the eligibility rule. Cite the code path, from 06 A.2.

### §2.4 Human Interventions — ADD (from 06 C.1; about 8 lines)

1. **Seed-42 AutoPareto rollback.** Attempts 5–8 ran on 2026-07-28 and were deleted. Attempt 4 was re-measured on a recreated checkpoint and its decision changed from dominated to retained. Attempt 10's decision was re-adjudicated after the FP32 evaluator amendment. The program text changed after attempt 4. Seeds 0 and 1 were launched after humans reviewed the seed-42 results.
2. **AutoResearch.** 27 trained candidates were discarded by a cache-correctness check later found faulty. The reference comes from a 5-attempt follow-up diagnostic under a revised program.
3. **Pair-A.** Inference-adapter amendments, requeues, model-identity changes before launch, and inconsistent failure adjudication.
4. **The AP-Q lineage** (seed 0) itself has no recorded intervention. Keep this sentence, because it protects the headline.

### §2.5 Verification — ADD the selection funnel and variability (about 0.4 pp)

**Selection funnel table**

| Stage | AutoPareto | AutoResearch | TPE / MOTPE |
|---|---|---|---|
| Discovery attempts | 75 | 50 + 5 | 25 + 25 |
| Representatives by the preregistered rule (2026-07-27) | 11 (exp 103, 108, 109, 124, 134, 136, 137, 147, 152, 158, 162) | exp 65, 68 | — |
| After the global role selector (2026-08-10, post-discovery) | 4, all seed 0 (124, 134, 136, 137) | 68 (65 inference-incompatible) | best BPB each (183, 234) |
| 201M panel (about 2026-08-21/25) | 3 (134, 136, 137); exp 124 dropped, no rule recorded | 68 | 183, 234 |

**Two sentences**

> *Every re-run candidate scored 0.0014–0.0087 BPB worse than at discovery. Within-campaign top-5 gaps (≤ 0.002) lie inside seed noise (SD 0.0021), so "best" within a campaign is not resolvable.*

**Seed effects**

> *the unchanged baseline varies by 0.024 BPB between seeds 0 and 42; under the fixed-time contract the same recipe processed up to 18.7% different token counts in different sessions.*

**Data integrity**
- Name the canonical validation value (`P1JSON`; it differs from DB rows by ≤ 2.4e-5).
- Note the two AutoResearch runs left in "running" state.
- Note that experiment ids 88–90 are missing (explained by the seed-42 rollback).

### §2.6 Significance — REWRITE

Replace "removes the apparent quality win" with "reveals that the quality ranking itself depends on the contract".

### §2.7 Interpretability — ADD one caveat

Agent hypotheses and patches are retained for all campaigns. Agent transcripts survive only for Pair-A.

### §2.8 Limitations — UPDATE

- Replace the "exposure alone" sentence (see `04` concern 1).
- Add: tokens and updates are confounded.
- Add: selection narrowed after discovery (§2.5).
- Add: historical agent identity is unverified.
- Add: no item-level downstream data.
- Add: one GPU type.

### §2.9 Pair-A pilot — NEW (moved from Part 1; about 0.35 pp)

Content:
- Five launch lineages, from 2026-09-05 to 2026-09-17, each a single adaptive trajectory per arm, with protocol changes between them.
- In the reported lineage, **2 of 4 fixed-time attempts are invalid**. The harness re-executed the training body after the candidate's own loop ended first and checkpointed a freshly initialized model (signature: candidate-reported step 0 vs about 1,264 observed updates). The controller still marked these runs quality-valid.
- Later restarts disagree on whether either arm explores batch size.
- Conclusion: *no contract effect on agent behavior can be estimated from these data.*
- The fix: treat a normal script exit as a failure, and assert that the checkpoint's step count equals the observed updates.
- The planned replicated design (E3).

---

## Part 3 — Reflections (≤ 1 pp; about 0.5 pp free)

ADD a "contract card" proposal (about 6 lines). An autonomous-ML paper should report:
1. the stopping rule and the LR-progress variable;
2. seeds used for discovery and for comparison;
3. tokens, updates and FLOPs per scored run;
4. session-throughput variation under time contracts;
5. the selection funnel from attempts to reported recipes, with dates;
6. harness invariants (checkpoint lineage, failure semantics).

ADD a "lesson" sentence: *"Our own harness accepted a checkpoint of an untrained model as quality-valid; invariant checks belong in the trust boundary, not in post-hoc audits."*

---

## Appendix A — Reproducibility — UPDATE

- **List what is released** (from `REPRO` §F):
  - the per-experiment registry;
  - the fixed-time confirmation and ablation records;
  - the 21-run panel JSON;
  - the Pair-A histories with failures;
  - the selection files;
  - the analysis and audit scripts;
  - environment pins.
- **What remains private:** provider state, dataset cache, large checkpoints.
- **Cite the N* calibration record** (06 C.1: `docs/RESEARCH_LEDGER.md` lines 640–669 / the manifest), or reword "median of three seed-3 baseline runs".
- **Fix:** Python version per era.
- **Before release:**
  - Redact the RunPod SSH endpoints found in about 69 `runs/` and `study/` files (`01` §0 item 13).
  - Regenerate the stale public `data/experiment_records.csv`, which has 284 of 297 rows.
  - Disclose that downstream checkpoints for AR-ref and MOTPE are substitutes (exp 77, exp 300).
  - Cite `results/phase1/token_stream_verification.json` (`RUNTIME_VERIFIED`) as evidence of common token streams.

## Before editing: identify the submitted version

The local `automlr_autopareto.pdf` cannot be confirmed as the submitted file (`03` D22; `01` §G).
1. Download the submitted PDF from OpenReview.
2. Diff it against `paper_backups/automlr_overleaf_reconstructed/` and `paper/main.tex`.
3. Start the camera-ready from the matching source.

---

## Things to DELETE or never claim

1. The ranked archive list as evidence of anything.
2. "Selection preceded confirmation" unscoped. Scope it to the seeds-0–2 confirmation.
3. The historical agent identity stated as verified.
4. "Source-diff audit labels".
5. Any statement that the fixed-token agent "never explored batch size". That holds in one restart only.
6. "Best completed attempt of the quality-only AutoResearch archive".
7. Any H2 description that omits the harness-invalid attempts.

---

## If new experiments are run (`08`)

| Experiment | Where it goes | What changes |
|---|---|---|
| **E1** (token-response crossover) | Part 1 §1.3 | Replace the reversal table with a 1-column figure: Δ BPB (AP-Q − AR-ref) against training tokens (88.6M, 157.3M, 201.3M) plus the 300 s point, with per-seed dots, seeds 6–8. State the crossover. Update the abstract and §1.4: "AP-Q is better below about X M tokens and worse above." |
| **E1b** (tokens vs updates) | Part 1 §1.3, one sentence | "At fixed tokens, halving the batch (doubling updates) changes the gap by Y." |
| **E4** (exp 109 family) | Table 1: one extra row ("AP-seed42, full-size") if it is in the panel at seeds 3–5; one sentence in §1.4 | If it beats AR-ref at matched tokens, AutoPareto has an exposure-controlled quality gain: make it a co-headline. If not, add "the preregistered full-size representative also does not beat AR-ref at matched tokens", which closes concern 3. |
| **E2** (stratified re-runs) | Part 2 §2.5 | One small table (optimism by discovery rank) |
| **E6** (scaling pilot, see 07) | Part 3, or one Part 2 sentence | Only if it finishes before 2026-11-09 |

---

## Page budget check

| Part | Additions | Cuts | Net |
|---|---|---|---|
| Part 1 | §1.0 (+0.20 pp); reversal table (+0.18 pp); caption additions (+0.05 pp) | Identifiers (−0.10 pp); measurement paragraph compression (−0.15 pp); H2 paragraph and H2 sentences (−0.22 pp); archive list (−0.03 pp) | ≈ −0.07 pp, fits |
| Part 2 | glossary (+0.20); algorithm box (+0.30); interventions (+0.15); funnel and variability (+0.40); §2.9 (+0.35); limitation edits (+0.05) | — | ≈ +1.45 pp against about 1.4 pp free. **Tight.** Shrink Figure 3 to 0.85 width, or move the glossary to Appendix A. |
| Part 3 | contract card + lesson (+0.25) | — | fits |

After editing, recompile with `tectonic main.tex`. Check per-part page counts, which `paper/EDITING_GUIDE.md` requires checking independently. Then copy the PDF to `/Users/Rami/Documents/dev/autopareto_automlr_FINAL.pdf`, per the project convention.
