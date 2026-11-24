# 08 — Experiment Plan (ranked by information per unit of compute)

**Ranking rule.** Experiments are ordered by how much they reduce uncertainty in the paper's claims, or answer a reviewer concern, per A40-hour and per day of calendar time. Novelty is not a criterion.

**Cost model**
- Measured A40 training throughput (steady-state tok/s, `P1JSON`):
  | Recipe | Steady-state tok/s |
  |---|---|
  | AR-ref | 289k |
  | AP-Q | 523k |
  | Baseline | 292k |
  | TPE | 372k |
  | MOTPE | 506k |
  | AP-F | 644k |
  | AP-S | 681k |
  | exp 109 (discovery, `SNAPDB experiments.training_tokens_per_second`) | 278k |
- A 300-second fixed-time run takes about 8.2 minutes end to end (`SNAPDB` ids 288–299).
- Overhead per run (warm-up, validation, checkpoint, inference) is about 3–5 minutes.
- Price is $0.44 per A40-hour, the RunPod public rate the project recorded on 2026-07-20 (`PROJECT_STATE.md`). Re-check it.
- Agent API cost is measured from Pair-A Codex transcripts (`runs/*/…/scientist_diagnostics/*/stdout.jsonl`, latest lineage). Each call is about 0.49–0.50M input tokens (about 0.43–0.44M of them cached) and 6–7k output tokens, including about 3.1–3.5k reasoning tokens. An attempt is about 2 calls.
- **None of the camera-ready experiments below need agent API calls.**

**Global preconditions (P)**

| ID | Precondition | Why |
|---|---|---|
| P1 | Restore the A40 execution environment: the pod snapshot `/workspace/PRESERVED_20260917T021347Z`, or a rebuilt environment matching the panel's pins. The panel ran on Python 3.10.18 (REPRO M4). | The panel harness, dataset cache and tokenizer are not in the local checkout (REPRO M12–M13) |
| P2 | **Pin the training-data revision.** Training currently reads `main` of `karpathy/climbmix-400b-shuffle` (REPRO M14). | Otherwise new runs may read different data than the panel did |
| P3 | Commit a one-page pre-registration of E1/E4 (hypotheses, run list, analysis) *before* launch, and record its hash | Avoids post-hoc analysis choices |
| P4 | Integrity gates on every run: exact token count; the checkpoint's step count equals the observed optimizer updates (catches the B.1 harness bug class); recipe hash; evaluator hash | The Pair-A harness accepted an untrained checkpoint (06 B.1) |
| P5 | **Protect the evidence before touching the environment.** Back up the local-only branch `autopareto/research-infrastructure` and `.git/lfs` (all 21 panel checkpoints live only there) to a private remote. Do not run `git gc`. Commit or revert the uncommitted edits to `protected/inference_adapter.py` and record its hash. | `01` §0 items 1, 5, 8, 12 |
| P6 | Redact RunPod SSH endpoints (about 69 files under `runs/` and `study/`, plus `docs/RESEARCH_LEDGER.md:853`) before releasing any Pair-A or provenance artifact | `01` §0 item 13 |

---

## Summary table

| Rank | ID | Question | Reviewer concern | Status | Runs | A40-h | $ (A40) | API |
|---|---|---|---|---|---|---|---|---|
| 0 | **E0** | Does the restored environment reproduce the panel? | precondition | **Essential** | 2 | 0.5 | 0.2 | none |
| 1 | **E1** | Where does the AP-Q vs AR-ref ranking cross over as tokens grow, on shared fresh seeds? | 1, 5 | **Essential** | 24 | ~4.1 | ~1.8 | none |
| 2 | **E4** | Does the preregistered full-size AutoPareto recipe (exp 109) beat AR-ref at matched tokens? | 3 (and the headline) | **Essential (highest upside)** | 9 | ~2.2 | ~1.0 | none |
| 3 | E1b | Tokens or updates? | 1 | Optional | 6 | ~1.0 | ~0.4 | none |
| 4 | E2 | How optimistic is discovery-stage selection across the rank distribution? | 2 | Optional (camera-ready) / core (follow-up) | 36 | ~4.9 | ~2.2 | none |
| 5 | E6 | Does AP-Q's quality cost and efficiency gain persist at more tokens or larger models? | follow-up | Optional pilot, only if time remains (see 07) | see 07 | see 07 | see 07 | none |
| — | E8 | Fix and test harness invariants | 4, 7 | Essential before any new *agent* campaign | 0 (unit tests and 1 smoke run) | <0.5 | — | none |
| — | E3 | Does the contract change the agent's hypotheses? A replicated study. | 4 | **Future paper** | ~100 attempts + verification | ~19 | ~8 | ~100M input (~88M cached) + ~1.4M output tokens |
| — | E7 | Framing vs evidence: counterfactual next-proposal resampling | 4 | Future (cheap) | 180 agent calls | 0 | 0 | ~90M input (~79M cached) + ~1.2M output tokens |
| — | E9 | Is the 1.99× decode ratio specific to the A40? | follow-up | Future | ~6 measurements | ~1 | ~0.5 | none |
| — | E10 | Item-level downstream with paired CIs | 5 | Future | re-evaluate 6+ checkpoints | ~2–3 | ~1 | none |

---

## E0 — Environment reproduction smoke (essential precondition)

**Question.** Does the restored environment reproduce a known panel cell and a known fixed-time cell, within seed-level noise?

**Runs**
1. AP-Q (source `55a7fd8b`), 201,326,592 tokens, seed 3. Compare with `P1JSON phase1-autopareto_292-seed3`: held-out 1.139616, validation 1.076589.
2. AR-ref (source `b3a961ee`), 300 s fixed time, seed 0. Compare with `SNAPDB` training_run 23731: validation 1.102901.

**Stopping rule.** If the held-out or validation difference exceeds 0.005 BPB, or tokens or updates differ, **halt** and diagnose environment drift (data revision, torch version, kernels) before anything else. The reported within-recipe SD is ≤ 0.006, so a deviation larger than 0.005 is a red flag.

**Cost:** about 0.5 A40-h.

---

## E1 — Token-response crossover (essential)

**Scientific question.** At equal tokens and updates, on the same fresh seeds and the same evaluation data, how does AP-Q − AR-ref change as training tokens grow? Where does the sign flip?

**Reviewer concern.** #1: several variables changed at once. Also #5 (variability), since E1 adds 3 independent seeds per cell.

**Hypotheses** (pre-register)
- **H-E1a.** Δ = BPB(AP-Q) − BPB(AR-ref) increases monotonically with training tokens.
- **H-E1b.** At 88.6M tokens with equal updates (676), Δ ≤ 0: AP-Q is at least as good when exposure is matched at the *discovery* scale.
  - If Δ > 0 already at 88.6M, AP-Q's fixed-time advantage came entirely from extra exposure.
- **Alternatives:**
  - a seed-set effect (ruled in or out by using fresh seeds 6–8 for every cell);
  - a schedule-variable effect (checked by the 300 s cell versus the 157.3M cell, which has a similar token count for AP-Q).

**Design**

| Factor | Levels |
|---|---|
| Recipe | AR-ref (exp 68, `results/candidates/autoresearch-diagnostic5-seed42-20260724/0005-train.py`, sha `b3a961ee…`); AP-Q (exp 134, `results/candidates/autopareto-seed0-20260731/0022-train.py`, sha `55a7fd8b…`) |
| Contract / budget | fixed tokens 88,604,672 (676 updates at 131,072); 157,286,400 (1,200 updates); 201,326,592 (1,536 updates); **plus** 300 s fixed time |
| Seeds | 6, 7, 8. Fresh for both recipes; disjoint from discovery (0, 1, 42), the confirmations (0–2) and the panel (3–5). |
| Held constant | Recipe source bytes; batch 131,072; device batch 64; LR progress = token fraction (fixed-time cell uses the time fraction); evaluator version; validation split; held-out shard `shard_06541.parquet`; A40 |

All three token budgets divide by both 131,072 and 65,536.

**Runs:** 2 recipes × 4 budgets × 3 seeds = **24 runs**.

**Cost**
- Training: AR-ref ≈ 307 + 544 + 696 + 300 s and AP-Q ≈ 169 + 301 + 385 + 300 s per seed, for about 2.5 h.
- Overhead: about 24 × 4 min ≈ 1.6 h.
- Total **≈ 4.1 A40-h ≈ $1.8**. No API.

**Harness changes**
- Run the panel's fixed-token runner with configurable budget and seed. The panel used one fixed budget, so check that the runner accepts others.
- Run the historical fixed-time runner for the 300 s cell, with the same evaluator.
- Record steady-state seconds.

**Outputs:** validation BPB, held-out BPB, tokens, updates, steady seconds, peak memory. Decode is optional; it is deterministic in architecture and already measured.

**Analysis**
- Paired difference per budget (n = 3; t-based 95% CI).
- Crossover by linear interpolation of the mean Δ in log tokens, with a bootstrap over seeds.
- Plot Δ against log tokens, with per-seed points.
- Compare the 300 s cell with the earlier seeds-0–2 result (−0.0067).

**What each outcome establishes**

| Outcome | Interpretation | Paper text |
|---|---|---|
| Δ < 0 at 88.6M, > 0 at 201M | A genuine **crossover**. AP-Q is better per token at small budgets; AR-ref wins at larger ones. The reversal is an exposure × capacity interaction. | "AP-Q is better below ≈X M tokens and worse above" |
| Δ ≈ 0 or > 0 at 88.6M | AP-Q's fixed-time advantage came **from exposure**. At equal tokens it never wins. | "The fixed-time advantage is explained by extra exposure" |
| 300 s cell differs from the 157M cell for AP-Q | The time-based schedule variable or session throughput matters | Add a sentence on the schedule |

**Essential for camera-ready?** **Yes.** It is the cheapest decisive answer to concern #1. With 3 more seeds at 201M it also gives n = 6 for the headline, which makes **E5 redundant**.

---

## E4 — The exposure-neutral AutoPareto candidate (essential, highest upside)

**Scientific question.** The AutoPareto seed-42 campaign produced exp 109: full 50.3M model, batch 65,536, matrix LR 0.02, embedding LR 0.6, weight decay 0.1. It scored validation 1.0940 on **84.1M tokens**, fewer than AR-ref's 88.9M at the same seed. It was one of the 11 preregistered representatives and was cut by the post-discovery global selector (06 D.1–D.2). Does it beat AR-ref at matched tokens?

**Reviewer concern.** #3: verification may exclude better recipes. It also bears on the headline claim that AutoPareto shows no exposure-controlled quality gain.

**Hypotheses**
- **H-E4.** BPB(exp 109) − BPB(AR-ref) < 0 at 201M tokens, seeds 3–5.
- **Alternatives:**
  - exp 109's discovery edge (0.0010) is seed noise (SD 0.0021);
  - its smaller batch (3,072 vs 1,536 updates at 201M) helps at 88M but not at 201M, since the critical batch size grows with training.

**Design**
- exp 109 (`results/candidates/autopareto-seed42-20260727/0022-train.py`, sha `0f99e3c8…`) at 201,326,592 tokens, seeds 3, 4, 5. That is 3 runs, directly paired with the existing panel's AR-ref runs: same seeds, same budget, same evaluator.
- exp 109 at 88,604,672 and 157,286,400 tokens, seeds 6–8, joining E1: 6 runs.
- Optional: sibling exp 111 (`…/0024-train.py`, sha `126f0785…`) at 201M, seeds 3–5: 3 runs.

**Runs:** 9 (12 with exp 111).

**Cost:** exp 109 at about 278k tok/s takes about 724 s at 201M, plus overhead. **About 2.2 A40-h ≈ $1.0.** No API.

**Analysis**
- Paired Δ against AR-ref at seeds 3–5 (held-out and validation). Use the existing panel AR-ref runs as the comparator, or re-run AR-ref if E0 shows drift.
- In the E1 frame: a Δ curve against tokens.

**What each outcome establishes**

| Outcome | Interpretation | Paper text |
|---|---|---|
| exp 109 better than AR-ref at 201M (CI excludes 0) | AutoPareto found an **exposure-controlled quality improvement** (post-hoc tested; say so) | A co-headline: "the full-size recipe from the seed-42 campaign improves held-out BPB by X at matched tokens" |
| Δ ≈ 0 | No detectable difference. The selection question is closed honestly. | One sentence in §2.5 |
| exp 109 worse | Strengthens the negative quality result | One sentence |

**Essential?** **Yes.** It is cheap, directly answers concern #3, and is the only experiment that could *add* a positive quality finding.

**Caveat to state either way.** The test is post hoc. The recipe is chosen with knowledge of the archive; it was, however, in the preregistered 11-recipe pool.

---

## E1b — Tokens vs. updates (optional)

**Question.** Do update count and token count play separable roles in the AP-Q vs AR-ref gap?

**Design**
- AR-ref and AP-Q at 88,604,672 tokens with `TOTAL_BATCH_SIZE` = 65,536 (1,352 updates), seeds 6–8.
- 6 runs, about 1.0 A40-h.
- Changing the batch changes the recipe. State this explicitly: it is an *intervention on the recipe*, not a pure manipulation of updates.

**Interpretation**
- If Δ at 88.6M moves substantially when updates double, update count is part of the mechanism.
- Use with the July ablation: batch alone gave −0.113 BPB at about fixed tokens (05 §3.6).

---

## E2 — Stratified selection-bias re-run (optional for camera-ready; core of the follow-up paper)

**Question.** How large is discovery optimism as a function of discovery rank, and how well do discovery ranks predict re-run ranks?

**Design**
- 12 candidates, chosen by completed-run discovery rank:
  - AP seed 0: ranks 1, 5, 12, 20.
  - AP seed 42: ranks 1, 5, 12, 20.
  - AR main campaign: ranks 1, 5, 10, 15.
- Each at **300 s fixed time** (the discovery contract), seeds 6–8. That is 36 runs × 8.2 min ≈ **4.9 A40-h ≈ $2.2**.

**Analysis**
- Optimism (discovery − re-run mean) against rank.
- Spearman correlation of discovery vs re-run rank, within and across campaigns.
- The fraction of "top-5" candidates whose re-run mean falls outside the campaign's top 5.

**Interpretation**
- It quantifies the winner's curse across the distribution, not only at the top (05 §3.8).
- If optimism at rank 1 far exceeds that at rank 12, the reported discovery bests are mostly selection.

---

## E6 — Scaling pilot (optional; only after E0, E1 and E4 succeed)

The full specification, feasibility and blockers are in `07_SCALING_FEASIBILITY.md` §E. Summary:

**Questions**
- Does the dominant discovered edit (the smaller batch) keep its advantage after 4× more training at the same model size?
- Does AP-Q's equal-token quality deficit against AR-ref grow or shrink?
- What is the first FLOP-matched comparison of AP-Q and AR-ref above the discovery compute?

**Design.** Seeds 3–5 throughout, paired with the existing 201M runs, which serve as the 1× rung once a code-path check passes. Analyse these seeds as a block separate from E1's seeds 6–8; do not pool the two.

| Tier | Arm | Budgets | Runs |
|---|---|---|---|
| **1a (core)** | AR-ref | 4× (805,306,368 tokens) | 3 |
| **1a** | AP-Q | 4× (equal tokens) | 3 |
| **1a** | AP-Q | 7× (1,409,286,144 tokens; FLOP-matched to AR-ref at 4×, −0.7%) | 3 |
| 1b (add-on) | Baseline | 4× | 3 |
| 1b | Batch only | 1× and 4× | 6 |
| 1b | Everything except batch | 1× and 4× | 6 |
| — | Code-path check | 1×; skip if E0 used the same executor | 1 |

**Size and cost**
- Tier 1a: 9 runs, about 7.2 A40-h (about $3.20).
- Tier 1b: 15 runs, about 10.4 A40-h (about $4.60).
- Both tiers: about 17.9 A40-h (about $7.90).
- Hard cap 22 A40-h. Calendar cutoff **2026-11-06**. **If time is short, run Tier 1a only.**

**Blockers to clear first** (`07` §D)
- The Phase-1 runtime is hard-coded to the 201M plan and device batch 64.
- Tokenizer files are not local.
- Confirm that the data cache holds at least 1.41B tokens (about 18–26 shards).
- Two Phase-1 implementation files no longer match their recorded hashes.

**Decision rules (preregistered)**
- "Persists": same sign on all 3 seeds and |mean| > 0.01.
- "Vanished": |mean| < 0.005 and the interval includes 0.
- Anything else: "inconclusive".
- The smallest reliably detectable change between rungs is about 0.02 BPB.

**Limits.** Nothing about larger models, other recipes, hardware or downstream transfer.

**Priority.** Below E0, E1 and E4. Run it only on a separate pod, or after the core runs finish. Report it as exploratory (Part 3 or an appendix).

**Long-term study** (`07` §F)
- A preregistered promotion operator.
- A 4-rung ladder at 4^k · C0.
- Edit-level paired promotions, sampled across all discovery effect levels.
- Fresh-seed re-measurement at the discovery budget, to separate regression to the mean from loss of the effect with scale.
- About 860–1,190 A40-h ($380–$525), plus agent API.

---

## E8 — Harness invariants (essential before any new agent campaign)

**Changes** (`study/contracts_2026/candidate_process.py`, controller)
1. Treat a normal exit of the candidate script as a failure. Do not re-enter the `while True` loop at line ~660 except for the warm-up stop.
2. Assert that the checkpointed model's step count equals the observed optimizer updates.
3. Require quality-valid runs to pass a sanity bound, e.g. BPB < baseline + 0.5, before they become parent-eligible.
4. Add noise margins to Pareto acceptance, e.g. require an improvement larger than 2× the measured repeat SD. A09/A10 show that identical models differ by 0.0014 BPB and 10% decode (05 §2.5).
5. Enforce the contract's learning-rate progress variable in the harness. Either inject the progress value, or reject a candidate whose LR schedule does not follow token progress under the fixed-token contract. In lineages G3/G4 most fixed-token candidates kept a time-based schedule (05 §5).

**Tests:** unit tests plus one smoke run with a deliberately self-terminating candidate.

**Cost:** under 0.5 A40-h.

---

## E3 — Replicated contract study (future paper)

**Design**
- Two contracts, fixed-time and fixed-token, with **5 seed-paired campaigns each**, 10 attempts per campaign, under one frozen and bug-fixed harness (after E8) and one pinned agent version.
- Pre-declared outcomes: those in `paper/NEXT_EXPERIMENT_PLAN.md`, plus lever coverage. The campaign is the unit of analysis.

**Cost**
- About 100 attempts × 8.2 min ≈ 14 A40-h.
- Verification of each campaign's best recipe at matched tokens, 3 seeds: about 5 A40-h.
- Agent: about 100 × 2 calls × (0.5M input, about 0.44M cached; 7k output) ≈ **100M input (≈88M cached) and ≈1.4M output tokens**. Add about 20% for requeues.

**Not feasible as a camera-ready experiment.** Credits ran out mid-Pair-A, and the harness needs E8 first.

## E7 — Counterfactual next-proposal resampling (future, cheap)

- Hold an existing history fixed at 3 points per arm, swap only the contract paragraph, and resample the next proposal 30 times. That is 2 × 3 × 30 = 180 agent calls, with no GPU.
- It estimates P(lever | contract, history), separating framing from evidence.
- This is the correct test of the "contract-induced blind spot" hypothesis, which the existing data do **not** support (05 §5).

---

## Redundant or low-value experiments (do not run for camera-ready)

1. **Re-running the existing 201M panel at seeds 3–5.** It is integrity-verified, and every number reproduces (`REPRO` §A).
2. **Extending any existing Pair-A trajectory.** Each is a single adaptive trajectory with protocol drift and the B.1 harness defect. More attempts add no replication.
3. **More seeds for AP-Q vs AR-ref at 201M beyond E1.** t = 14.7 at n = 3 already. E1 adds seeds 6–8 at 201M.
4. **Re-measuring memory or KV cache.** Deterministic across seeds.
5. **Classical seeds 0/1 at fixed time.** They do not answer any reviewer concern. A fair tournament is a separate study.
6. **Downstream re-evaluation without per-item logging.** It cannot produce paired CIs.

## Stopping and integrity rules (all experiments)

- Fixed run lists. No seeds are added in response to results.
- Each run must pass the P4 integrity gates. A failed run is re-run **once** on the same seed; if it fails again it is reported as failed.
- The analysis follows the pre-registration (P3). Deviations are reported.
- If E0 fails its tolerance, stop.

## Calendar through 2026-11-13

| Dates | Work |
|---|---|
| Oct 28–29 | Decide the scope (E1 + E4 recommended). Reconcile the manuscript source (`09` §0.4 item 1). Restore the environment (P1). Pin the data revision (P2). Write the pre-registration (P3). |
| Oct 30 | E0 (2 runs). Halt if it fails tolerance. |
| Oct 30 – Nov 1 | E1 (24 runs) and E4 (9 runs): about 6.3 A40-h. Fits in one day on one A40; spread over two days for buffer. |
| Nov 2 | Optional: E1b (6 runs) or E2 (36 runs). |
| Nov 3–6 | Analysis, figure, Part 1 rewrite (`09`). E6 only if everything above is done. |
| Nov 7–10 | Human rewrite and verification of Part 2. Public artifact release (`REPRO` §F). |
| Nov 11–12 | Compile; per-part page checks; PDF QA; copy to `/Users/Rami/Documents/dev/autopareto_automlr_FINAL.pdf`. |
| Nov 13 | Submit camera-ready. |
