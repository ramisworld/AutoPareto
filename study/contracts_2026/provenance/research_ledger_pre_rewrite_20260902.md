# AutoPareto research ledger

## Purpose

This is the maintained interpretation layer for the AutoPareto research project. It
records why decisions were made, what the evidence currently supports, what remains
uncertain, and which tables and figures belong in a future paper. SQLite remains the
authority for experiment rows, exact metrics, candidate diffs, hypotheses, failures,
events, hashes, and timestamps. Values in this document must be regenerated or checked
against SQLite before publication.

This file is not an instruction file for an isolated research agent. AutoPareto agents
receive only their registered program, protocol, candidate interface, same-seed
baseline, and campaign-isolated history.

The canonical current-state and forward research roadmap is
[`PROJECT_STATE.md`](../PROJECT_STATE.md#autopareto-remaining-research-roadmap). This
ledger remains the historical evidence and interpretation record; future phase status
and sequencing belong in the canonical roadmap.

## NeurIPS 2026 AutoMLR submission target

The intended near-term venue is the non-archival NeurIPS 2026 Workshop for Autonomous
Machine Learning Research (AutoMLR). Its official call requires both a qualifying ML
result for which an autonomous agent made a decisive contribution and a detailed
account of the autonomous system. The submission has three separately limited parts:
four pages for the research result, four pages for system design, and one page for
reflections. The current official deadlines are 2026-08-22 for the abstract and
2026-08-29 for the complete submission. At least one author must attend in Sydney if
the paper is accepted.

This target does not change the frozen scientific protocol or agent program. The
paper's central claim must be selected from confirmed evidence rather than chosen to
match a desired narrative. During double-blind review, linked code, data, and dashboard
artifacts must preserve author anonymity. A de-anonymized public repository and
production dashboard belong after acceptance or under whatever final workshop policy
is then authoritative.

Paper evidence is organized as follows:

- Part 1: the autonomous research question, matched-budget methods, confirmed Pareto
  result, causal ablations, downstream quality, deployment behavior, and limitations;
- Part 2: the agent program, campaign isolation, parent selection, evaluator-owned
  measurements, protection system, failure handling, prompts, tools, and every material
  human intervention;
- Part 3: what autonomous experimentation changes about the roles of proposal,
  verification, judgment, and scientific responsibility.

The workshop is the immediate target; it is not equivalent to NeurIPS main-track
acceptance or an invited talk. The work should remain suitable for a later archival
submission after broader confirmation and scale transfer.

## Research question

Can an autonomous coding researcher discover useful transformer designs when it
preserves trade-offs among validation quality, cached inference speed, and allocated
inference memory instead of advancing only a single quality winner?

The project does not assume that one model is universally best. It searches for a
Pareto frontier: models where improving one measured objective requires sacrificing at
least one other objective.

## Fixed discovery contract

- Hardware: one NVIDIA A40.
- Model family: approximately 50M-parameter GPT baseline, with the complete candidate
  `train.py` open to architecture, optimizer, schedule, batching, and compatible cache
  adapter changes.
- Training: from scratch in bfloat16 for 300 measured seconds.
- Search attempts: exactly 25 per campaign.
- Search seeds: 42, 0, and 1.
- Objectives: lower validation BPB, higher protected cached-decode tokens/second, and
  lower protected peak allocated inference memory.
- Every failure consumes its attempt.
- Candidate histories are isolated by method and campaign.
- Cache correctness is mandatory and cannot be traded against the three objectives.

The full current rules are in `RESEARCH_PROTOCOL.md`. The protected model and cache
contract is in `docs/candidate_interface.md`.

## Methods

- Baseline: the unchanged shared model and training recipe.
- AutoResearch: arbitrary-code agent research advancing only strict BPB improvements.
- AutoPareto: arbitrary-code agent research choosing a current frontier parent and
  preserving every non-dominated quality-speed-memory result.
- Fixed-space controls: Random Search, TPE, CMA-ES, MOTPE, and NSGA-II under matched
  attempts, hardware, timing, seeds, and failure accounting.
- Planned hybrid: AutoPareto plus MOTPE, only after its mechanism is separately
  specified and preregistered.

Arbitrary-code and fixed-space methods have unequal research surfaces and must be
reported as separate tracks rather than as a pure algorithm-ranking contest.

## Completed evidence before AutoPareto

The historical AutoResearch campaign provided exploratory evidence that smaller total
training batch size produced a major quality improvement under the five-minute budget.
Confirmation and ablation runs later showed that the batch-only change explained most
of the observed mean BPB improvement, while the non-batch bundle did not improve mean
quality alone. Those campaigns used earlier cache-correctness rules and remain
historical evidence rather than perfectly matched AutoPareto replicates.

## Cache-correctness incident and resolution

### Original symptom

Full-prefix inference and incremental cached decoding sometimes produced different
bfloat16 logits. Early checks compared a candidate's FlashAttention-style full-prefix
path with the protected SDPA cached path. Both intended to compute the same model, but
different kernels, tensor shapes, and floating-point operation orders created
deterministic numerical drift.

### Diagnostic finding

Matching generated tokens did not imply identical raw logits. Conversely, a bfloat16
greedy-token mismatch did not automatically prove a logically broken cache. The
project therefore needed to distinguish kernel/precision drift from a genuine cache
implementation error.

### Final policy

For canonical candidates, protected correctness compares an uncached full-prefix SDPA
reference with cached SDPA decoding at multiple prefixes. Bfloat16 remains the fast
first check. Any bfloat16 correctness failure, including a greedy-token mismatch,
automatically triggers strict float32 math-SDPA adjudication. A candidate is compatible
only if the strict float32 path passes both greedy-token and logit-equivalence checks.

Float32 is an adjudication test, not a performance mode and not an output correction.
Official speed and memory remain bfloat16 measurements.

## Human-intervention and verification record

AutoPareto is an autonomous proposal-and-experiment loop inside a human-designed and
human-audited scientific harness; it is not claimed to be an entirely human-free
research process. Material interventions that must be disclosed in the paper include:

- humans fixed the benchmark, objectives, attempt budgets, search/confirmation seeds,
  protected files, failure accounting, and campaign-isolation rules;
- humans diagnosed and repaired the cached/full correctness evaluator after bfloat16
  kernel drift was initially mistaken for cache incompatibility;
- humans introduced strict float32 math-SDPA adjudication, while retaining bfloat16 for
  all official speed and memory measurements;
- humans reviewed campaign pauses caused by correctness failures and authorized resume
  only after preserving the checkpoint and error;
- humans deliberately rewound the early seed-42 continuation to make corrected attempt
  4 the parent of the replacement attempt 5; this historical intervention and evaluator
  amendment remain disclosed rather than presented as uninterrupted autonomy;
- humans stopped and restarted RunPod instances, synchronized authoritative SQLite
  state, recovered checkpoints, and repaired infrastructure without selecting candidate
  model edits during an active isolated campaign;
- after seed-0 attempt 17 exposed an ephemeral tokenizer cache, humans added a
  pre-reservation persistent-cache check so the same infrastructure failure cannot
  consume a later scientific attempt;
- humans curate dashboard language and will verify every paper claim, figure, citation,
  checksum, and reproduction instruction.

Within each isolated campaign, the research agent selected a valid frontier parent,
formulated the hypothesis, edited candidate `train.py`, interpreted the protected
result, and continued or stopped according to the frozen program. SQLite, candidate
diffs, program hashes, agent actions, logs, and events are the audit evidence for this
division of labor.

### Scientific amendment

Seed-42 attempts 1--4, 5--9, and 10--25 carry successive evaluator hashes as the
correctness system was repaired. The objectives and bfloat16 performance metrics did
not change. This amendment must be disclosed, and selected candidates must be
remeasured with one final evaluator during confirmation.

## AutoPareto seed 42

Campaign `autopareto-seed42-20260727` completed 25 of 25 attempts. Every final
authoritative attempt completed training and protected inference. Its final
three-objective frontier contains 11 models.

### Current highlighted models

| Attempt | Role | BPB | Cached tok/s | Allocated MiB | Parameters |
|---:|---|---:|---:|---:|---:|
| 22 | best quality | 1.093981 | 151.32 | 249.89 | 50,332,176 |
| 14 | safe quality balance | 1.097077 | 177.09 | 203.13 | 36,962,574 |
| 21 | lower-memory quality balance | 1.100385 | 178.24 | 189.01 | 31,850,638 |
| 15 | intermediate efficiency | 1.112001 | 207.34 | 122.45 | 18,874,476 |
| 16 | primary efficiency | 1.122841 | 242.21 | 113.32 | 17,301,610 |
| 17 | fastest and smallest | 1.181091 | 302.44 | 74.51 | 9,175,112 |

These are seed-42 discoveries, not cross-seed confirmed conclusions.

### Interpretation

- Attempt 22 is the observed quality anchor, but its isolated weight-decay gain is
  small and needs replication.
- Attempt 14 preserves near-best quality while improving speed and memory.
- Attempt 21 trades 0.30% BPB against attempt 14 for 0.65% more speed and 7% less
  memory. Neither strictly dominates the other.
- Attempt 15 is a strong compact midpoint. It is preferred over attempt 19 for costly
  confirmation because it is about 19% faster and 19% lower-memory with only 0.07%
  worse BPB.
- Attempt 16 is the main deployment-efficiency hypothesis: relative to attempt 14 it
  is about 36.8% faster and 44.2% lower-memory at a 2.35% BPB cost.
- Attempt 17 establishes the observed efficiency boundary. Its BPB remains better
  than the seed-42 baseline, but its quality cost relative to the best AutoPareto model
  is substantial.

## Seed-42 architectural change-effect matrix

This matrix reports observed parent-to-child changes. “Single variable” means the
recorded candidate changed one intended architectural or optimizer control while
derived shape stayed fixed. “Compound” means a sizing rule changed depth, width, or
head layout together. A single seed supports a hypothesis, not a causal population
claim.

| Parent → child | Change | BPB | Speed | Memory | Classification |
|---|---|---:|---:|---:|---|
| 12 → 14 | depth 8→7, width fixed at 512 | +0.06% | +13.85% | −7.62% | single variable |
| 14 → 21 | two KV heads→one | +0.30% | +0.65% | −6.95% | single variable |
| 14 → 15 | depth 7→6, width 512→384, heads 4/2→3/1 | +1.36% | +17.08% | −39.72% | compound |
| 15 → 16 | depth 6→5, width fixed at 384 | +0.97% | +16.82% | −7.45% | single variable |
| 16 → 17 | depth 5→4, width 384→256, query heads 3→2 | +5.19% | +24.87% | −34.25% | compound |
| 14 → 19 | head dimension 128→64 with width/head-layout changes | +1.29% | −1.55% | −25.95% | compound |
| 10 → 22 | weight decay 0.2→0.1 | −0.02% | −1.00% | unchanged | single variable |

Positive BPB means worse quality; positive speed means faster; negative memory means
less allocated memory. The live dashboard derives these percentages from SQLite and
the frozen case-study specification rather than copying the values above.

## AutoPareto seed 0

Campaign `autopareto-seed0-20260731` completed exactly 25 registered attempts. Twenty-
four attempts completed training and protected cached inference. Attempt 17 failed
before training because `/root/.cache/autoresearch/tokenizer/tokenizer.pkl` disappeared;
the failure consumed its attempt and is infrastructure evidence, not model evidence.
SQLite integrity and foreign-key checks pass.

The mechanically selected discovery representatives are:

| Attempt | Role | BPB | Cached tok/s | Allocated MiB | Parameters |
|---:|---|---:|---:|---:|---:|
| 22 | best quality | 1.093294 | 312.84 | 159.88 | 29,360,392 |
| 12 | lowest memory within 1% quality | 1.096304 | 258.26 | 132.20 | 24,576,298 |
| 25 | high-throughput quality | 1.097434 | 408.40 | 141.38 | 26,214,662 |
| 24 | aggressive efficiency / hypervolume | 1.116324 | 351.09 | 105.20 | 17,891,526 |

Attempt 8 reached the lowest absolute memory (`80.51` MiB), but its `1.157314` BPB
places it outside the predeclared one-percent quality budget.

Relative to the same-seed baseline, attempt 25 reduced BPB by about 9.58%, increased
cached throughput by about 148.9%, and reduced allocated inference memory by about
43.4%. Relative to the similar-quality seed-42 attempt 14, it measured about 130.6%
faster and 30.4% lower-memory for a 0.033% BPB cost. These are cross-campaign discovery
comparisons, not repeatability estimates or confirmed claims.

### Seed-0 mechanistic findings and caveats

- Reducing total training batch from `2^19` to `2^17` produced the major initial BPB
  improvement. Because the inference graph is unchanged, its small throughput change
  is runtime variation rather than a batching effect on inference.
- The depth-8 to depth-7 step improved all three observed objectives with width fixed.
- The compound depth-7/width-512 to depth-6/width-384 step produced a large speed and
  memory gain, but depth and width effects cannot be separated by that comparison.
- Attempt 7 was described as width 320, but head-dimension rounding produced actual
  width 384. Its parent-to-child model change is therefore a clean depth-6 to depth-5
  reduction.
- Attempts 9 and 10 both built depth-4, width-384, three-head models. Attempt 10 is a
  duplicate-architecture measurement, not evidence for a width increase. Their speed
  difference demonstrates why single-run throughput deltas require caution.
- Width 256 to 512 at depth 4 recovered quality at a large memory cost.
- Depth 4/width 512 to depth 3/width 384 improved speed and memory but cost quality.
- Width 384 to 512 at depth 3 produced attempt 25's unusually strong measured
  throughput and quality recovery. This is a promising GPU-utilization hypothesis that
  requires confirmation before causal wording is used.
- Optimizer-only changes can support quality claims, but cannot cause inference-graph
  memory changes. Their observed speed differences are treated as benchmark variation.

The frozen interpretation and actual architectures are stored in
`evaluation/seed0_case_study.json`. All 25 candidate sources, hypotheses, intended
changes, diffs, parents, and decisions are local. All 24 successful checkpoints were
recovered from persistent storage and independently hash-recorded in
`results/preserved_checkpoints/autopareto-seed0-20260731-manifest.json`.

## AutoPareto seed 1

Campaign `autopareto-seed1-20260731` completed exactly 25 registered attempts. Every
attempt completed training and protected cached inference. Seed 1 used the same final
program SHA-256, evaluator SHA-256, and protected-manifest SHA-256 as seed 0, so these
two campaigns are the cleanest direct discovery replicates. All hypotheses, intended
changes, exact diffs, parents, decisions, logs, events, candidate sources, and
checkpoints are local.

The mechanically selected representatives are:

| Attempt | Role | BPB | Cached tok/s | Allocated MiB | Parameters |
|---:|---|---:|---:|---:|---:|
| 25 | best quality | 1.095755 | 253.70 | 132.20 | 24,576,298 |
| 21 | fastest within 1% quality | 1.096740 | 257.14 | 132.20 | 24,576,298 |
| 10 | lowest memory within 1% quality | 1.106084 | 249.20 | 113.32 | 17,301,610 |
| 15 | largest hypervolume contribution / extreme boundary | 1.342847 | 534.79 | 49.82 | 3,538,980 |

Attempt 16 reached `1,073.01` cached tok/s and `48.57` MiB, but its `1.501478` BPB is
a severe quality collapse. Attempts 15 and 16 are scientifically useful boundary
measurements, not recommended language models.

### Seed-1 mechanistic findings and caveats

- The initial `2^19` to `2^17` total-batch reduction again produced a large finite-time
  quality gain without changing the inference graph.
- Depth 8 to 7 at width 512, followed by depth 6 to 5 at width 384, improved all three
  measured objectives in their respective parent-to-child comparisons.
- Simultaneously reducing depth and rounded width produced larger speed and memory
  gains but eventually caused substantial quality loss.
- Holding width 384 while reducing depth 5 to 4 avoided much of the quality collapse
  observed when width simultaneously fell to 256.
- One shared KV head reduced parameters and allocated memory. It did not establish a
  reliable throughput gain and incurred a modest BPB cost.
- Weight-decay, matrix-LR, and embedding-LR refinements improved BPB while leaving the
  inference architecture unchanged. Their speed variation is not treated as a causal
  optimizer effect.
- The 2-layer and 1-layer models map the outer efficiency boundary; they demonstrate
  that speed and memory can be optimized into a regime where language quality is no
  longer acceptable.

The exact architecture and effect registrations are frozen in
`evaluation/seed1_case_study.json`. Every seed-1 checkpoint and candidate source is
hash-recorded in
`results/preserved_checkpoints/autopareto-seed1-20260731-manifest.json`.

## Three-seed discovery interpretation

Across the three isolated searches, the most consistent discovery is a family of
shallower models that trades capacity for cached throughput and allocated memory. The
evidence also repeatedly shows that smaller total training batch improves quality in
the fixed five-minute regime, while optimizer-only changes affect quality rather than
the inference graph. KV-head sharing consistently reduces parameter and memory cost,
but its speed effect is not yet reliable.

These are repeated discovery patterns, not confirmed causal population effects. Exact
recipes differ between campaigns, inference throughput has measurement variation, and
seed 42 includes a disclosed evaluator amendment. The pooled confirmation selector
therefore hashes and deduplicates exact candidate sources and preserves distinct model
recipes rather than averaging unrelated attempts.

## Preregistered case study and confirmation

The three campaign case-study specifications are `evaluation/seed42_case_study.json`,
`evaluation/seed0_case_study.json`, and `evaluation/seed1_case_study.json`. Exact
confirmation recipes are selected by
`scripts/select_autopareto_representatives.py`, pooled across campaigns, and
deduplicated by the SHA-256 of the complete candidate `train.py`. The frozen current
selection is stored in `results/autopareto_confirmation_selection.json`. Each exact
recipe is retrained independently on confirmation seeds 0, 1, and 2. Unrelated
architectures are never averaged into a synthetic model.

The compatible AutoResearch comparison is labelled the **AutoResearch compatible
representative**. The lower-BPB historical AutoResearch winner remains quality-only:
its cached inference was incompatible, so it is retained for audit but never placed on
the quality–speed–memory plot. Database IDs are audit identifiers, not model names.

The final dashboard point for a confirmed recipe uses its aggregate across the three
confirmation checkpoints and exposes each seed and its range. Search-seed attempt
numbers are local identifiers and are never treated as shared recipes.

## Post-search language characterization

The frozen downstream suite is `evaluation/downstream_suite.json`. It runs only after
authorized AutoPareto and classical searches finish and representatives are selected.
It includes a second held-out ClimbMix BPB shard, LAMBADA, BLiMP, PIQA, ARC-Easy, and
fixed qualitative completions. Results never influence search proposals. Task metrics
remain separate; no undocumented combined quality number is created.

The held-out BPB shard tests whether search improvements generalize beyond the
validation shard repeatedly observed during discovery. Qualitative completions are
illustrations and cannot establish general model quality.

The frozen execution selection contains 11 comparisons: one seed-1 baseline, one
compatible AutoResearch reference, four confirmed AutoPareto roles represented by the
median-BPB confirmation checkpoint, and the lowest-BPB valid compatible result from
each of TPE, CMA-ES, MOTPE, Random Search, and NSGA-II. This rule is mechanical and was
recorded before viewing any downstream score. AutoResearch is labelled a compatible
reference rather than a 3/3-confirmed claim because one historical confirmation run
was inference-incompatible. Missing original weights are never silently substituted:
the audit first seeks a hash-matching preserved file and otherwise records a distinct
exact-recipe checkpoint-recovery run.

For qualitative evidence, every selected model receives the same six prompts at seeds
0, 1, and 2 with 128 generated tokens, temperature 0.8, and top-k 50. SQLite and raw
JSON preserve the prompt, completion verbatim, token IDs, token count, elapsed time,
checkpoint/candidate/evaluator hashes, and decoding settings. The dashboard permits
side-by-side comparison. Paper examples must disclose the fixed selection and may not
be cherry-picked as proof of aggregate quality.

## Quantization study

The preregistered design is `evaluation/quantization_suite.json`. Quantization begins
only after full-precision searches, representative selection, and confirmation. It
compares BF16 against pinned INT8 weight-only execution and, when a verified A40
kernel exists, INT4 weight-only execution.

The study separately reports weight storage, file size, total allocated/reserved VRAM,
KV-cache growth, speed, prefill latency, held-out BPB, downstream tasks, correctness,
and qualitative generations. Unsupported or slower kernels remain recorded. Quantized
measurements never replace the full-precision headline frontier.

## Dashboard reporting contract

The default Explore view shows four explicit AutoPareto trade-offs from each of seeds
42, 0, and 1, plus subdued same-seed baselines and one compatible AutoResearch
reference. AutoPareto seed colors and separate role outlines keep method, campaign,
and scientific role distinct. Only the selected point is persistently labelled.

Readers can switch to the full frontier, confirmed frontier, every completed result,
one method, one campaign, or one seed. The Evidence view presents the derived
change-effect matrix and marks single-seed findings as observed rather than causal.
The public dashboard will eventually use a frozen read-only data export rather than a
writable research database.

## Current limitations

- AutoPareto has completed all three discovery seeds; the four selected primary recipes
  have three-seed confirmation, while unselected recipes are not confirmed.
- The seed-42 evaluator changed during cache-correctness repair.
- Several architectural patterns recur across seeds, but causal effect sizes are not
  yet isolated from token exposure, parameter count, batch size, training recipe, and
  seed variance.
- Downstream language tasks are complete for the frozen 11-model historical panel
  (99 aggregate metrics and 198 verbatim generations) and the separate 21-checkpoint
  equal-token panel (189 aggregate metrics and 378 verbatim generations). Both are
  preserved in SQLite and raw JSON; the scopes must not be silently combined.
- Quantization has not run.
- All five seed-42 classical controls have completed. The preregistered seed-0/1
  classical repeats have not run.
- Larger-model transfer and cross-GPU replication have not run.
- A 15M--50M model trained for five minutes cannot support broad frontier-model
  capability claims; the contribution is search methodology and measured deployment
  trade-offs.

## Next evidence sequence

The canonical forward sequence is maintained in
[`PROJECT_STATE.md`](../PROJECT_STATE.md#autopareto-remaining-research-roadmap). The
primary confirmation, downstream suite, and stabilized deployment characterization are
now complete. The next planned scientific phase is the fair equal-token comparison;
quantization, cross-hardware replication, and scale transfer are deferred or
conditional according to the canonical roadmap.

The exact 15 classical campaign registrations, attempt limit, method order, training
seeds, and matched algorithm seeds are frozen in
`evaluation/classical_campaigns.json`. They were registered on 2026-08-05 at zero
attempts after every optimizer passed offline proposal validation. Registration is an
audit action only and did not consume GPU time. The quantization audit schema and plan
gate are implemented; its execution status and priority are governed by the canonical
roadmap.

## TPE seed-42 fixed-space result — 2026-08-06

Campaign `tpe-seed42-20260805` completed its exact 25-attempt budget with training seed
42 and algorithm seed 20260767. Twenty-three attempts completed training and protected
cached inference; attempts 1 and 20 were genuine candidate OOM failures and count toward
the budget. The frozen search-space SHA-256 is
`bc20cc05ababd12616c0571b476bfb949d530096aac22d00ad97c2f58137ceee`.

TPE minimized validation BPB only. Attempt 21 achieved the best quality (1.097702 BPB,
212.29 cached tok/s, 204.89 MiB); attempt 22 was fastest (216.16 tok/s); attempt 15
used the least memory (95.04 MiB); and attempt 23 was the practical balance (1.099956
BPB, 199.12 tok/s, 142.70 MiB). These four exact checkpoints and their SHA-256 manifest
are preserved. They are single-seed discovery representatives, not confirmations.

The first ten startup proposals reached 1.103328 best BPB; guided attempts 11–25
improved this to 1.097702. All five best-quality valid trials used depth 6, total batch
131072, unembedding LR 0.002, zero warmup, final-LR fraction 0.05, weight decay 0.3,
and the SSSS window. This is descriptive optimizer behavior, not causal evidence: TPE
changed many controls at once and the sample is small. The complete curve,
configurations, Pareto set, failures, and frequencies are frozen in
`results/tpe-seed42-20260805-analysis.json`.

This does not contradict Centaur: Centaur combines an LLM with CMA-ES in a fixed search
space. AutoPareto edits arbitrary candidate code and optimizes three objectives, while
this TPE control is restricted to 15 knobs and optimizes quality alone. Direct method
claims require matched scope, objectives, budgets, and repeated seeds.

## CMA-ES seed-42 fixed-space result — 2026-08-06

Campaign `cmaes-seed42-20260805` completed 25/25 attempts with no failures. Every
candidate completed protected cached inference, and all Optuna trials map exactly to
their SQLite rows and sampled configurations. CMA-ES minimized validation BPB only.

Its complete three-objective frontier is attempts 15, 19, 22, and 23. Attempt 15 has
the best quality (1.108287 BPB, 174.72 tok/s, 131.07 MiB). Attempt 22 is both fastest
and lowest-memory (1.131397 BPB, 205.48 tok/s, 119.95 MiB). Attempt 23 provides a
nearby speed-quality balance (1.129408 BPB, 202.72 tok/s, 125.01 MiB), while attempt
19 is the intermediate quality step (1.121736 BPB, 175.78 tok/s, 150.42 MiB).

Among the five best-quality candidates, head dimension 64, embedding LR 0.45,
unembedding LR 0.004, final-LR fraction 0.05, and warmdown ratio 0.5 appeared in all
five. These are small-sample, multi-knob associations rather than isolated causal
effects. The four frontier checkpoints are hash-verified locally; the full analysis is
`results/cmaes-seed42-20260805-analysis.json` and its checkpoint manifest is
`results/preserved_checkpoints/cmaes-seed42-20260805-manifest.json`.

On seed 42, TPE found better best quality than CMA-ES (1.097702 versus 1.108287 BPB),
while CMA-ES produced a competitive efficiency point. This is one discovery seed and
does not support a general method-ranking claim.

## MOTPE seed-42 fixed-space result — 2026-08-07

Campaign `motpe-seed42-20260805` completed its exact 25-attempt budget with training
seed 42 and algorithm seed 20260767. Attempt 1 was a genuine candidate OOM and counts
toward the budget; attempts 2–25 completed protected cached inference. SQLite integrity
and foreign keys pass, and Optuna contains the matching 25 trials. Unlike TPE and
CMA-ES, MOTPE jointly optimized validation BPB, cached decode speed, and peak inference
memory.

Its complete three-objective frontier is attempts 21–25. Attempt 22 has the best quality
(1.098112 BPB, 226.08 tok/s, 142.70 MiB), attempt 25 is fastest (1.100810 BPB,
229.00 tok/s, 142.70 MiB), attempt 23 has the lowest memory (1.109570 BPB,
214.17 tok/s, 122.45 MiB), and attempt 24 is the low-memory quality balance
(1.108565 BPB, 211.66 tok/s, 122.45 MiB). The random-startup phase reached 1.102669
best BPB; guided attempts 11–25 improved it to 1.098112 while moving the efficiency
frontier substantially outward. The full frozen analysis is
`results/motpe-seed42-20260805-analysis.json`.

The final database and optimizer snapshots were written before automatic Pod shutdown
and their SHA-256 values match the watchdog report. However, the original MOTPE model
weights were not preserved: `results/checkpoints` resolved to the container-local
`/tmp/autoresearch-checkpoints`, which RunPod cleared on stop. Exact configurations,
candidate sources, measurements, logs, optimizer state, and database evidence survive,
so discovery measurements remain valid; downstream and quantization work requires
retraining selected exact recipes. This infrastructure failure is recorded in
`results/preserved_checkpoints/motpe-seed42-20260805-manifest.json` and must not be
hidden. The runner now repairs the legacy symlink and rejects non-`/workspace`
checkpoint storage before reserving any future scientific attempt.

## Random Search seed-42 fixed-space result — 2026-08-07

Campaign `random-search-seed42-20260805` completed its exact 25-attempt budget with
training seed 42 and algorithm seed 20260767. Attempts 1 and 12 were genuine candidate
OOM failures and count toward the budget; the other 23 attempts completed training and
protected cached inference. SQLite integrity and foreign keys pass, and the matching
Optuna study contains exactly 25 trials. Every proposal was an independent sample from
the same frozen 15-control space; Random Search has no startup-to-guided transition.

The complete three-objective frontier is attempts 6, 7, 9, 10, 22, and 25. Attempt 10
has the best quality (`1.103809` BPB, `204.56` cached tok/s, `179.89` MiB), attempt 7
is fastest (`1.227225` BPB, `208.89` tok/s, `182.38` MiB), and attempt 25 uses the
least memory (`1.210954` BPB, `174.94` tok/s, `131.07` MiB). These are distinct
trade-offs: the speed and memory extremes pay substantial quality costs. The six
nondominated checkpoints are preserved locally and SHA-256 recorded in
`results/preserved_checkpoints/random-search-seed42-20260805-manifest.json`.

Among the five best-quality valid trials, total batch 131072 appeared in all five,
while head dimension 64 and the SSSS window each appeared in four. Because every
candidate changed many controls at once, these are descriptive associations rather
than causal effects. The full configurations, convergence trace, failures, frontier,
and parameter frequencies are frozen in
`results/random-search-seed42-20260805-analysis.json`.

On the single matched discovery seed, Random Search did not match the best quality of
TPE (`1.097702`), MOTPE (`1.098112`), or the compatible AutoResearch representative
(`1.094976`), and it did not reach AutoPareto's strongest efficiency points. This is
useful negative-control evidence, not a cross-seed method-ranking claim.

## NSGA-II seed-42 fixed-space result — 2026-08-08

Campaign `nsgaii-seed42-20260805` completed its exact 25-attempt budget with training
seed 42 and algorithm seed 20260767. Attempts 1 and 10 were genuine candidate OOM
failures and count; the remaining 23 attempts completed training and protected cached
inference. SQLite integrity and foreign-key checks pass. NSGA-II jointly optimized
validation BPB, cached decode speed, and peak inference memory with population size
eight; attempts 1-8 formed the initial population and attempts 9-25 used completed
population history.

The complete nondominated set is attempts 6, 12, 13, 14, 16, 19, 20, 24, and 25.
Attempt 13 has the best quality (`1.107744` BPB, `166.05` tok/s, `249.89` MiB),
attempt 20 is fastest (`214.15` tok/s, `1.162595` BPB, `182.38` MiB), attempt 12 is
the lowest-memory frontier point (`133.57` MiB, `1.143242` BPB, `183.17` tok/s), and
attempt 14 is the selected quality-speed representative (`1.114167` BPB, `188.96`
tok/s, `270.89` MiB). On this single discovery seed, NSGA-II did not match the best
quality of TPE, MOTPE, or AutoPareto. This is not a general cross-seed ranking.

The completed campaign initially remained only on the stopped original Pod volume
after an automatic migration exposed an empty replacement volume. RunPod zero-GPU
recovery was used to retrieve the authoritative database, optimizer state, logs,
candidates, events, and four representative checkpoints. The full non-causal analysis is frozen in
`results/nsgaii-seed42-20260805-analysis.json`, and the dashboard representatives are
registered in `evaluation/nsgaii_seed42_case_study.json`.

## Paper claim discipline

Current supportable wording is: “Three independent AutoPareto discovery searches found
distinct quality-speed-memory trade-offs under a fixed A40 five-minute budget; the
seed-0 search included a 408 tok/s candidate at 1.0974 BPB and 141 MiB, while seed 1
mapped the outer efficiency boundary and its associated quality collapse.” It is not yet
supportable to claim that AutoPareto reliably beats every method, that any
architectural change generalizes across seeds or scales, or that the compact models
retain equivalent downstream language ability.

## Resource-aware primary confirmation panel — 2026-08-10

The preregistered per-campaign selector remains intact and preserves 11 unique exact
AutoPareto recipes. Confirming all 11 on three seeds would require 33 runs. Before any
confirmation run, a separately disclosed staged analysis applied the same four-role
selector globally to that frozen pool. This produced a primary panel of four recipes:

- experiment 134: global best discovery quality;
- experiment 137: fastest candidate within 1% of global best discovery quality;
- experiment 124: lowest-memory candidate within that same 1% quality budget;
- experiment 136: largest remaining exclusive normalized hypervolume contribution.

The exact sources, hashes, discovery measurements, selection rule, and seeds are frozen
in `results/autopareto_primary_confirmation_plan.json`. Each recipe is retrained
unchanged on seeds 0, 1, and 2, producing 12 planned runs. This is a resource-aware
staged confirmation and does not replace, hide, or retroactively redefine the complete
11-recipe discovery selection. Any paper must disclose that distinction.

Confirmation uses one immutable one-attempt campaign per recipe and seed. It makes no
new proposal, reads no downstream evaluation, and records every success, failure,
checkpoint, evaluator hash, and measurement. Causal ablations, downstream tasks,
deployment workloads, and quantization remain blocked until this primary confirmation
stage is complete and reviewed.

## Primary AutoPareto confirmation completed — 2026-08-12

All 12 preregistered exact-source runs completed on the A40. Every run finished its
five-minute training budget, produced a persistent checkpoint, and passed protected
cached inference. No recipe, evaluator, selection rule, or seed was changed after the
panel was frozen.

Across seeds 0, 1, and 2:

- experiment 134, selected for quality, averaged `1.094745` BPB, `299.38` cached
  tok/s, and `159.88` MiB;
- experiment 137, selected for speed within the one-percent discovery quality budget,
  averaged `1.098827` BPB, `360.01` tok/s, and `141.38` MiB;
- experiment 124, selected for memory within that quality budget, averaged `1.097739`
  BPB, `244.57` tok/s, and `132.20` MiB;
- experiment 136, selected for exclusive hypervolume contribution, averaged `1.125060`
  BPB, `368.90` tok/s, and `105.20` MiB.

The discovery value of `408.40` tok/s for experiment 137 was not reproduced on every
seed; its confirmed seed values were `327.63`, `391.22`, and `361.18` tok/s. The
architecture nevertheless retained strong quality and a mean of `360.01` tok/s. This
distinction must remain explicit in the paper.

The authoritative structured result is
`results/autopareto_primary_confirmation_analysis.json`. The 12 local checkpoints
were verified byte-for-byte against the remote SHA-256 values. The next evidence gate
is the frozen downstream language suite on mechanically selected confirmed models,
followed by deployment characterization and the separately reported quantization
study. Downstream scores must not retroactively alter this confirmation panel.

## Downstream evaluator infrastructure correction — 2026-08-14

The first official downstream invocation stopped before producing any scientific
result. Held-out BPB slicing preserved the correct token values but produced a
strided, non-contiguous memory layout. Frozen candidates flatten targets with
`Tensor.view`, which requires contiguous storage, so the first baseline raised an
infrastructure exception before any task score or generation was committed.

The failed run remains recorded. No candidate, checkpoint, selection, dataset, task,
decoding setting, or metric definition changed. The evaluator now materializes
aligned causal inputs and targets with `contiguous()` and validates finite losses, the
exact nine-metric set, all 18 fixed generations per model, JSON round trips, atomic
SQLite persistence, database integrity, and raw-result completeness before advancing.
Regression tests verify unchanged tokens and causal alignment. The changed evaluator
SHA-256 distinguishes the failed infrastructure run from corrected official runs.

### Frozen downstream completion — 2026-08-14

All 11 preregistered representatives completed under the unchanged suite and
evaluator: 99 task metrics, 198 fixed qualitative generations, and 11 raw JSON
bundles. Every run contains exactly nine metrics and 18 generations; SQLite integrity
is `ok` with no foreign-key violations. Two later zero-output infrastructure
reservations are preserved under `results/downstream_interruptions/`: an external
funding shutdown and a missing `/root/.cache` binding after Pod migration. The wrapper
now restores and verifies `/workspace/.cache/autoresearch` before every evaluation.

AutoPareto confirmed quality achieved the panel's best LAMBADA perplexity
(`13.965839`) and ARC-Easy length-normalized accuracy (`0.344276`). AutoPareto fast
quality achieved the best LAMBADA exact match (`0.211915`), and AutoPareto low memory
achieved the best BLiMP macro accuracy (`0.721701`). TPE (`1.153413`) and Random Search
(`1.154778`) slightly outperformed AutoPareto confirmed quality (`1.157651`) on the
separate held-out BPB slice. These are distinct trade-offs, not universal dominance,
and none of these scores influenced search or representative selection.

### Equal-token verification completion — 2026-08-28

The later Phase 1 verification is a separate seven-recipe panel: Baseline, compatible
AutoResearch, TPE, MOTPE, and AutoPareto recipes 292, 299, and 294. Each recipe was
trained at exactly 201,326,592 tokens on seeds 3, 4, and 5. All 21 training and
downstream runs completed. The panel contains 189 aggregate task metric rows and 378
fixed qualitative generations in `results/phase1/downstream_200m.db` and
`results/phase1/downstream_200m_evaluations/`.

Across seeds, AutoPareto 292 recorded 29,360,392 parameters, 1.141837 held-out BPB,
385.019 steady-state-equivalent training seconds, 383.022 stabilized cached decode
tokens/s, and 1.992 MiB incremental KV memory. AutoResearch recorded 50,332,176
parameters, 1.099803 held-out BPB, 696.163 seconds, 192.933 tokens/s, and 3.984 MiB
KV memory. AP292 is therefore the current balanced efficiency candidate, but its
held-out BPB and BLiMP micro accuracy (0.720174 versus AutoResearch's 0.745592) are
worse. Its ARC-Easy accuracy (0.369108 versus 0.365039), PIQA accuracy (0.582880
versus 0.586688), and LAMBADA exact match (0.214956 versus 0.204412) are close on
this panel. These values are verification evidence, not a claim of universal
AutoResearch superiority or inferiority.

The full per-seed training, inference, memory, downstream, and caveat audit is in
`docs/PHASE1_200M_RESULTS.md` and `docs/DOWNSTREAM_EVALUATION.md`. The original
five-minute panel remains historical evidence; it is not merged with this equal-token
panel because its training exposure, seeds, and evaluator SHA differ.

## Frozen final deployment workload study — 2026-08-14

After downstream completion and before observing final deployment results, five exact
full-precision checkpoints were frozen: the same-seed baseline and confirmed
AutoPareto representatives for best quality, lowest memory within the one-percent
quality budget, fastest within that budget, and largest exclusive hypervolume
contribution. The selection and artifact hashes are recorded in
`results/final_deployment_selection.json`; the workload definition is frozen in
`evaluation/final_deployment_suite.json`.

The matrix is 5 models × 4 contexts × 4 inference batches. Every workload generates
128 tokens and must record one warm-up plus five measured repetitions or one explicit
failure. A fresh baseline is measured in the same session so hardware/runtime drift
cannot be confused with model gains. Historical deployment evidence is preserved in
a separate database session rather than overwritten. This stage does not change the
completed search, confirmation panel, downstream selection, or headline metrics.

## Timing-stability amendment before deployment execution — 2026-08-16

The frozen final deployment matrix had not begun when a methodological audit found
that discovery experiment 137's repeated throughput continued rising during the
measurement window. The likely mechanism is GPU compilation, clock, or thermal state
still settling after the fixed warm-up count. Consequently, the original matrix is
preserved as a preregistration artifact but is superseded before execution; no result
was deleted or rewritten.

The replacement evaluator first runs a two-model A40 canary: baseline experiment 4
and discovery experiment 137, context 256, inference batch 1, and 256 generated
tokens. Each of three passes adaptively warms up in blocks of five until two adjacent
blocks have no more than 2% median throughput drift and no more than 5% within-block
coefficient of variation. Failure to stabilize by 30 warm-ups invalidates the
environment. Each successful pass then records ten repetitions and requires its two
five-run measurement blocks to pass the same stability test.

Synchronized wall-clock decode throughput is the scientific headline. CUDA-event
timing is a diagnostic for host-side overhead. GPU clocks, performance state,
temperature, power, driver, and memory state are captured outside timed regions. All
raw warm-ups, measurements, failures, and environment evidence are append-only in a
versioned SQLite session. The final five-model workload matrix will be refrozen only
after both canary models pass; therefore no final corrected speed result exists yet.

## Stabilized timing canary result — 2026-08-16

Both preregistered models passed session 4 without a correctness, stability, timing,
or telemetry failure. The baseline used 35 adaptive warm-ups and experiment 137 used
40; each then contributed 30 measured repetitions across three passes.

For context 256, batch 1, and 256 generated tokens, synchronized wall-clock medians
were `166.8675` tok/s for baseline experiment 4 and `419.9379` tok/s for AutoPareto
experiment 137. Allocated VRAM was `249.8799` versus `141.3784` MiB; reserved VRAM
was `320.0` versus `192.0` MiB; incremental KV-cache memory was `3.9844` versus
`1.4941` MiB. Thus the frozen AutoPareto checkpoint was `2.516x` faster while using
`43.42%` less allocated memory and `62.50%` less incremental KV-cache memory in this
same-session A40 workload. This is strong hardware-specific deployment evidence, not
yet a cross-hardware or cross-workload generalization claim.

The independent verifier returned no errors, SQLite integrity was `ok`, and there
were no foreign-key violations. Raw outputs are preserved under
`results/deployment_evaluations/4/`; every warm-up and measured repetition is stored
append-only in `results/results.db`. A replacement stabilized full deployment matrix
may now be frozen. The superseded fixed-warm-up matrix must not be executed.

## Consolidated infrastructure and evaluator findings

The project encountered and preserved several failures that materially improved the
final methodology:

1. Early cached/full checks compared a FlashAttention-style full-prefix path with an
   SDPA cached path. Kernel and reduced-precision accumulation differences produced
   non-zero logits even when greedy tokens matched. The reference and cached paths
   were aligned to SDPA; ambiguous BF16 cases now trigger strict float32 math-SDPA
   adjudication instead of being silently accepted or rejected.
2. Checkpoints originally pointed at ephemeral `/tmp` storage and were lost when Pods
   stopped. Checkpoint storage is now required to resolve under persistent
   `/workspace`, with byte size and SHA-256 checked before evaluation.
3. Pod migrations repeatedly cleared `/root/.cache`. The launcher now binds that
   runtime path to `/workspace/.cache/autoresearch` and verifies all 11 data shards,
   tokenizer bytes, and at least one training shard before scientific execution.
4. The first downstream run used correct token values in non-contiguous tensor views;
   frozen candidates called `Tensor.view` and failed. The evaluator now materializes
   contiguous aligned slices and regression-tests that token identity and causal
   alignment are unchanged.
5. Fixed inference warm-ups allowed A40 compilation and clock state to improve during
   measured repetitions. The final protocol adaptively stabilizes two consecutive
   five-run blocks, measures three independent ten-run passes, records external GPU
   telemetry, and independently rechecks measured-block stability.

These are disclosed engineering and measurement findings, not model improvements.
Failed infrastructure invocations remain distinguishable from scientific runs by
status, evaluator hash, raw artifacts, and database session.

## Stabilized final deployment study frozen — 2026-08-17

After the successful canary, the replacement final deployment matrix was frozen
without observing any matrix result. It contains five exact full-precision
checkpoints and 16 workloads per checkpoint: four contexts crossed with four
inference batches. The exact selection and evaluator dependency hashes are stored in
`results/final_deployment_stabilized_selection.json`; the workload and stability rules
are stored in `evaluation/final_deployment_stabilized_suite.json`.

Every successful workload must contain three measured passes of ten repetitions plus
10--30 adaptive warm-ups per pass. Every unsupported, OOM, correctness, or stability
failure is explicit. Evidence is append-only and cannot be resumed across a Pod or
GPU-state interruption. The next scientific action is this 80-workload A40 study;
quantization remains blocked until its results are retrieved and audited.

## Stabilized deployment recovery and final evidence — 2026-08-17

Session 5 (`final-deployment-stabilized-a40-20260817`) executed the frozen 80-workload
A40 matrix and recorded 62 stable pairs plus 18 terminal unstable pairs. Session 5 is
preserved as immutable historical evidence; its failed status is not rewritten.

Those exact 18 pairs were retried in session 6
(`final-deployment-stabilized-a40-20260817-retry-unstable-20260817`) under the same
checkpoint selection, workload definition, evaluator, and adaptive stability policy.
Four retries became stable and fourteen remained terminally unstable: eleven reported
measured-throughput drift and three did not stabilize within the 30-warm-up limit. No
retry correctness failure was recorded. Final audited evidence is therefore 66/80
stable pairs (82.5%) and 14/80 explicit unstable pairs (17.5%). No speed or memory
value is estimated for an unstable pair.

“Unstable” is a timing conclusion, not model incompatibility. The retry supersedes
only the exact pair it targeted; both sessions remain available for audit. The
reproducible CPU-only summary is `results/final_deployment_stabilized_summary.json`,
generated by `python3 scripts/analyze_stabilized_recovery.py`. SQLite integrity and
foreign-key checks pass. The paper should report both stable and unresolved outcomes,
including failure categories, rather than selecting only successful workloads.

No additional GPU retry is authorized by this ledger. Any larger-warm-up sensitivity
test must be separately preregistered and versioned, and must not alter sessions 5 or 6.

## Phase 1 pre-Stage-B timing correction — 2026-08-24

The NON-SCIENTIFIC equal-token smoke exposed a path mismatch: the staged Phase 1
executor still called the legacy `inference_benchmark.py`, which uses exactly two fixed
warm-ups and ten measured repetitions. The later protected deployment implementation
was not being used for Phase 1 final inference. No training result or frozen recipe was
changed.

An inference-only seven-checkpoint rebenchmark was therefore registered and run on a
single NVIDIA A40 at the Phase 1 headline workload (BF16, fixed prompt, 256 input
tokens, batch 1, greedy 256 generated tokens). It reused the frozen
`autopareto_inference_stabilized_v2` evaluator: five-run adaptive warm-up blocks,
adjacent median drift <=2%, within-block CV <=5%, maximum 30 warm-ups per pass, three
passes, and ten measured repetitions per pass. A measured pass is rejected if its two
five-repetition blocks fail the same gate. The scientific speed summary is the median
of all 30 successful synchronized wall-clock measurements; maximum repetition speed is
diagnostic only. Cached/full correctness passed for all loaded checkpoints.

The first session and a versioned retry are preserved under
`results/phase1/stabilized_inference/`. The retry produced one complete standardized
row: the MOTPE representative's exact-recipe recovery checkpoint (source experiment
234, checkpoint experiment 300) measured 268.3438 tok/s with pass medians 267.8348,
269.5926, and 268.2667 and warm-up counts 25, 10, and 10. Baseline and AutoResearch
failed to stabilize by 30 warm-ups; TPE and AutoPareto 292 failed a measured-pass gate;
AutoPareto 299 and 294 each failed a later measured pass. These are explicit timing
unresolved outcomes, not model correctness failures, and no partial median is promoted
to the paper headline.

AutoPareto 299's smoke observations around 480 tok/s are not a single-repetition
maximum artifact: a stable pass in the first session was 476.7233 tok/s, and two retry
passes were 493.6442 and 493.4024 tok/s. The retry's third pass failed with 17.7933%
median drift and 8.4659% CV, so 299 has no canonical stabilized value yet. The old
fixed-two-warmup discovery/confirmation values remain historical and unchanged. The
new canonical comparison table, which records unresolved rows explicitly, is
`results/phase1/phase1_stabilized_inference_rebenchmark.md`.

This is a pre-Stage-B measurement-protocol correction. Phase 1 remains
`PREPARING / PREFLIGHT`; no Stage B or Stage C training job has started.

## Phase 1 measurement hardening before Stage B — 2026-08-25

The pre-Stage-B audit identified four infrastructure risks before any scientific
result: CUDA-event allocation was inside every timed optimizer step; the telemetry
sampler queried PyTorch CUDA allocator state; evaluation and stabilized inference ran
between training jobs; and the previously registered order was only partially
counterbalanced.

The amended protocol removes scientific CUDA-event instrumentation, keeps synchronized
monotonic wall timing authoritative, and records steady-state-equivalent training time
as the primary efficiency metric. The complete first-to-final step sum remains a
secondary operational metric. A same-excluded-token sensitivity estimate is also
recorded. Telemetry is CUDA-runtime-free and its 0.5-second overhead is tested by an
excluded control/instrumented smoke with a preregistered ABBA order.

Training and BPB evaluation are now separated from deployment inference. Inference
instability therefore does not invalidate token accounting, BPB, or training timing,
but no deployment-throughput claim is allowed until the stabilized inference gate and
cache-correctness check pass. Each run receives isolated compiler-cache paths, and
successful run directories are immutable.

The revised machine-readable registration is
`evaluation/phase1_equal_token_preflight.json`, format
`autopareto_phase1_equal_token_preflight_v3`. The human-readable amendment is
`docs/PHASE1_MEASUREMENT_AMENDMENT.md`. No frozen model recipe or scientific result
was changed; Phase 1 remains `PREPARING / PREFLIGHT`.

## Phase 1 equal-token experiment complete — 2026-08-26

All 21 preregistered scientific runs completed on one NVIDIA A40: seven frozen
representatives crossed with fresh seeds 3, 4, and 5. Every run consumed exactly
201,326,592 training tokens from its seed's runtime-verified common token stream.
Baseline completed 384 optimizer steps at accumulation 4; every other model completed
1,536 optimizer steps at accumulation 1. All 84 milestone checkpoint hashes, final
held-out evaluations, source hashes, timing artifacts, and cache-correctness checks
passed the final integrity audit.

The preregistered primary training metric is steady-state-equivalent seconds: the
post-step-11 sustained token rate extrapolated to the complete common token budget.
Complete first-to-final scientific wall time remains secondary and includes every
scientific optimizer step. The primary deployment metric is the median of 30
synchronized-wall-clock decode repetitions after three passes satisfy the frozen
2% adjacent-median / 5% within-block-CV gate; maximum throughput is diagnostic only.

Five initial Stage-C inference sessions were explicitly rejected by that gate. Each
failed JSON was retained append-only with its SHA-256, and a complete new session was
run with identical rules. All five versioned retries stabilized and passed cache
correctness. No warm-up allowance, stability threshold, workload, model recipe, or
scientific result was changed. The failure/retry history is retained under each run
directory and in `results/phase1/stage_c_inference_recovery-session-1.json` and
`results/phase1/stage_c_inference_recovery.json`.

Across three seeds, mean held-out BPB / steady-state-equivalent seconds / stabilized
decode tok/s were: baseline 1.146590 / 688.50 / 196.82; AutoResearch 1.099803 / 696.16 /
192.93; TPE 1.119402 / 540.53 / 258.89; MOTPE 1.148253 / 398.23 / 270.01; AutoPareto
292 1.141837 / 385.02 / 383.02; AutoPareto 299 1.159309 / 312.78 / 488.26; and
AutoPareto 294 1.190057 / 295.68 / 495.49. These results support a trade-off frontier,
not universal superiority: AutoResearch has the best quality, while AutoPareto 294/299
are fastest in training and deployment at worse BPB. The complete per-seed evidence,
sample SDs, and paired candidate-minus-baseline contrasts are in
`results/phase1/phase1_equal_token_results.json`.
