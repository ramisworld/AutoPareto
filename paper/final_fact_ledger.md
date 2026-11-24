# Internal audit artifact: not a submission asset

# AutoPareto final fact ledger

Every value below was recomputed from frozen artifacts on 2026-09-06/07. Sources: `paper/generated/equal_token_runs.json` (21 matched-exposure runs), `paper/generated/db_experiments.json` (318 archive rows), `evaluation/phase1_public_manifest.json`, `benchmark_config.py`, `inference_benchmark.py`, `study/contracts_2026/protocol.md`.

## Identifier resolution

| Question | Answer | Source | Confidence |
| --- | --- | --- | --- |
| What is A22? | Attempt 22 of campaign `autopareto-seed0-20260731`; source `results/candidates/autopareto-seed0-20260731/0022-train.py`, SHA-256 `55a7fd8bf32924f2...` | db_experiments row id 134 | HIGH |
| What is global experiment 134? | The same row: A22, seed 0, completed, val BPB 1.093294337780448, 29,360,392 params, 159,776,768 fixed-time tokens | db_experiments id 134 | HIGH |
| Is A22 = experiment 134? | Yes | db_experiments | HIGH |
| What is AP292? | The frozen source recipe of experiment 134 as retrained under the matched-exposure protocol. The numeral is inherited from archive row 292, `autopareto-primary-best-quality-exp134-seed1-20260810`, an earlier exact-source confirmation of experiment 134 on seed 1. It is a configuration label, not a separately discovered architecture. | db_experiments ids 291-293 descriptions; equal_token_runs `model: autopareto_292` | HIGH |
| AP294 / AP299 | Frozen recipes of A24 (exp 136) and A25 (exp 137); labels inherited from archive rows 294 and 299 | db_experiments 294-299 | HIGH |
| AP292 source hash | `55a7fd8bf32924f2a35ebc095c5206f3503e671a9d7a9646b94979c8ae49259a` | equal_token_runs; phase1_public_manifest | HIGH |
| Why was AP292 selected? | Primary role `best_quality` over the discovery archive | campaign id `autopareto-primary-best-quality-exp134-seed0-20260810`; confirmation descriptions | HIGH |
| When was it selected? | Campaign identifiers dated 2026-08-10; confirmation runs executed 2026-08-12 00:09 onward; discovery finished 2026-08-04. Selection preceded confirmation. | db_experiments started_at | HIGH |
| Why the AutoResearch reference? | Best completed attempt of the recovered quality-only archive (`autoresearch-diagnostic5-seed42-20260724`, attempt 5, exp 68, val BPB 1.0949760675); the two better-scoring rows in that campaign are crashes | db_experiments 64-68 | HIGH |
| AutoResearch source hash | `b3a961eeea05dcec...`, `results/candidates/autoresearch-diagnostic5-seed42-20260724/0005-train.py` | equal_token_runs | HIGH |
| Difference from baseline | Depth 8 / width 512 / 4+4 heads both; window pattern SSSS vs baseline SSSL; total batch 131,072 with accumulation 1 vs baseline 524,288 with accumulation 4 | docs/PHASE1_200M_RESULTS.md panel table | HIGH |

## Headline values (recomputed, n=3 seeds 3/4/5)

| Quantity | AP292 | AutoResearch | Derived |
| --- | --- | --- | --- |
| Parameters | 29,360,392 | 50,332,176 | 41.66% fewer |
| Held-out BPB (mean) | 1.141837 (SD 0.006035) | 1.099803 (SD 0.001943) | +0.042034 cost |
| Cached decode tok/s | 383.022 (SD 4.900) | 192.933 (SD 1.380) | 1.985x |
| Peak allocated MiB | 162.8799 (SD 0) | 250.8818 (SD 0) | 35.08% less |
| Incremental KV MiB | 1.9922 (SD 0) | 3.9844 (SD 0) | exactly 50% less |
| Counted tokens | 201,326,592 | 201,326,592 | exact, verified per run |
| Optimizer updates | 1,536 | 1,536 | identical |
| Total batch size | 131,072 | 131,072 | identical |
| Steady-state train s | 385.02 | 696.16 | extrapolated, not wall time |

All published macro values in `generated/results_macros.tex` match these recomputations.

## Protocol constants

| Constant | Value | Why | Source |
| --- | --- | --- | --- |
| Fixed-time budget | 300.000 s | Immutable protected constant; fits a 25-attempt campaign in one GPU session | `benchmark_config.TRAINING_BUDGET_SECONDS`; protocol.md | 
| Confirmation tokens | 201,326,592 = 192 x 2^20 = 1536 x 131,072 = 384 x 524,288 | Divisible by both panel batch settings so no partial update | phase1_public_manifest; verified per run |
| Contract N tokens | 88,604,672 = 169 x 524,288 | floor(median of three seed-3 baseline T runs / 524,288) x 524,288 | protocol.md "Calibration and freezing N_STAR" |
| Confirmation seeds | 3, 4, 5 | Disjoint from discovery seeds (0, 1, 42) and earlier confirmation seeds (0, 1, 2) | phase1_public_manifest; campaign records |
| Decode probe | BF16, batch 1, 256 in / 256 out, greedy, fixed protected prompt, 30 measured repetitions in 3 passes | Registered immutable constants | benchmark_config.py; final_inference.json |
| Memory statistic | `max_memory_allocated()` after `reset_peak_memory_stats()` per repetition; weights resident and included; reserved excluded; covers prefill + all decode steps | inference_benchmark.py `one_measured_repetition` | HIGH |
| Incremental KV | validated cache bytes after final decode step minus after prefill (255 positions) | inference_benchmark.py | HIGH |
| Memory variance | Exactly constant across all 30 repetitions and all 3 seeds for both headline recipes | verified directly from final_inference.json | HIGH |
| Panel size | 7 recipes | baseline + AutoResearch + TPE + MOTPE + 3 AutoPareto primary roles; the lowest-memory role (exp 124 / AP289) was not rerun at 200M | docs/PHASE1_200M_RESULTS.md | HIGH |

## Hypothesis status

- **H1** is a RETROSPECTIVE audit hypothesis. Formulated after inspecting historical results; recorded in `paper/research_question_freeze.md` (2026-09-06), explicitly "a retrospective analysis freeze, not a new experimental preregistration". Do not present as preregistered.
- **H2** was registered before the paired campaigns in `study/contracts_2026/protocol.md` in NONDIRECTIONAL form. The directional wording in the writing brief is interpretive, not agent-originated.
- Agent hypotheses (per-attempt predictions recorded before each patch) are preserved in `generated/budget_attempts.json` and the experiments table; they are distinct from both H1 and H2.

## Edit-category attribution (75 AutoPareto attempts)

32 architectural, 21 optimizer, 10 optimization logic, 8 batching, 4 schedule. Derived from source diffs; multi-category edits assigned to the dominant change.

## Confirmed lineage (seed 0)

baseline(8) -> A1(113) -> A4(116) -> A6(118) -> A7(119) -> A8(120) -> A18(130) -> A19(131) -> A20(132) -> A22(134). Secondary branch A22 -> A24(136) -> A25(137). A17(129) crashed and is retained. Parentage verified against `parent_experiment_id`.

## Unresolved

- Ownership of some historical infrastructure repairs (inference stabilization, warm-up handling, relaunch decisions) cannot be attributed to human operator vs assisted debugging from preserved artifacts. These are excluded from claims of autonomous contribution.
- The current controller performs warm-up in the same worker process with reconstructed state, not the separate process specified in the registered program text. Disclosed in the paper as an execution deviation.
