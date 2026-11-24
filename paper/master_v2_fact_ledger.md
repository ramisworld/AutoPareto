# Master v2 fact ledger

Internal audit artifact. Every value re-derived from frozen artifacts on 2026-09-07 by `analysis/master_v2/build_master.py` and direct reads of `paper/generated/equal_token_runs.json` (21 verification runs), `paper/generated/db_experiments.json` (318 archive rows), `evaluation/phase1_public_manifest.json`, `benchmark_config.py`, `inference_benchmark.py`, and `study/contracts_2026/protocol.md`. Where prose in the submitted workshop PDF disagreed with raw artifacts, the artifacts win.

## 1. Identifier resolution

| Question | Verified answer | Source |
| --- | --- | --- |
| What is A22 | Attempt 22 of campaign `autopareto-seed0-20260731` | db_experiments id 134 |
| What is experiment 134 | The same row: seed 0, completed, val BPB 1.093294337780448, 29,360,392 params, 159,776,768 fixed-time tokens | db_experiments id 134 |
| What is AP292 | The frozen source recipe of experiment 134 retrained under the verification protocol. Numeral inherited from archive row 292 (`autopareto-primary-best-quality-exp134-seed1-20260810`), an earlier exact-source confirmation of exp 134 on seed 1. A configuration label, not a separate architecture. | db_experiments 291-293 descriptions; equal_token_runs `model: autopareto_292` |
| AP299 / AP294 | Frozen recipes of A25 (exp 137) and A24 (exp 136); labels inherited from archive rows 299 and 294 | db_experiments 294-299 |
| Selected AutoResearch recipe | `autoresearch-diagnostic5-seed42-20260724/0005-train.py`, archive exp 68, val BPB 1.0949760675. Best *completed* attempt of the recovered quality-only archive; the two lower-BPB rows in that campaign (exp 64, 65) are crashes. | db_experiments 64-68 |
| Selection timing | Role-encoded campaign identifiers created 2026-08-10; confirmation executed from 2026-08-12 00:09; discovery ended 2026-08-04. Selection preceded verification. | db_experiments started_at |

## 2. Source hashes (frozen recipes)

| Recipe | Source path | SHA-256 (first 8) |
| --- | --- | --- |
| Baseline | results/candidates/baseline-a40-seed0/0002-train.py | 428e22f1 |
| AutoResearch | results/candidates/autoresearch-diagnostic5-seed42-20260724/0005-train.py | b3a961ee |
| TPE | results/candidates/tpe-seed42-20260805/0021-train.py | 8547b278 |
| MOTPE | results/candidates/motpe-seed42-20260805/0022-train.py | 62882a20 |
| AP292 | results/candidates/autopareto-seed0-20260731/0022-train.py | 55a7fd8b |
| AP299 | results/candidates/autopareto-seed0-20260731/0025-train.py | 36e3ad91 |
| AP294 | results/candidates/autopareto-seed0-20260731/0024-train.py | 6d0d8d1d |

Every hash was verified equal in three places: the frozen file on disk, the per-run `training_metrics.json` `source_sha256`, and `evaluation/phase1_public_manifest.json`.

## 3. Discovery-stage exposure (the mechanism, newly surfaced in v2)

Under the 300-second fixed-time contract, the seven later-verified recipes consumed:

| Recipe | Archive exp | Discovery tokens | Discovery val BPB |
| --- | --- | --- | --- |
| AutoResearch | 68 | 88,866,816 | 1.0949760675 |
| Baseline | 4 | 92,798,976 | 1.2127299917 |
| TPE | 183 | 113,770,496 | 1.0977022288 |
| MOTPE | 234 | 152,961,024 | 1.0981116330 |
| AP292 | 134 | 159,776,768 | 1.0932943378 |
| AP299 | 137 | 197,656,576 | 1.0974344667 |
| AP294 | 136 | 251,789,312 | 1.1163236466 |

Spread: 251.79M / 88.87M = 2.833x. AP292 versus AutoResearch: 159.78M / 88.87M = 1.798x. AP292 discovery margin over AutoResearch: 1.0949760675 - 1.0932943378 = 0.0016817. This pairing (a 1.80x exposure difference producing a 0.0017 BPB margin) is the paper's central motivating fact and was not stated in the workshop version.

## 4. Matched-exposure verification (n = 3 seeds: 3, 4, 5)

| Recipe | Params | Held-out BPB (SD) | Decode tok/s (SD) | Peak MiB | Incr. KV MiB | Steady s | Updates | Batch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 50,332,176 | 1.146590 (0.004673) | 196.821 (2.446) | 250.8818 | 3.9844 | 688.50 | 384 | 524,288 |
| AutoResearch | 50,332,176 | 1.099803 (0.001943) | 192.933 (1.380) | 250.8818 | 3.9844 | 696.16 | 1,536 | 131,072 |
| TPE | 39,846,284 | 1.119402 (0.004271) | 258.894 (3.423) | 205.8809 | 2.9883 | 540.53 | 1,536 | 131,072 |
| MOTPE | 26,345,772 | 1.148253 (0.003102) | 270.006 (5.401) | 145.5684 | 2.2412 | 398.23 | 1,536 | 131,072 |
| AP292 | 29,360,392 | 1.141837 (0.006035) | 383.022 (4.900) | 162.8799 | 1.9922 | 385.02 | 1,536 | 131,072 |
| AP299 | 26,214,662 | 1.159309 (0.000370) | 488.258 (8.298) | 142.3799 | 1.4941 | 312.78 | 1,536 | 131,072 |
| AP294 | 17,891,526 | 1.190057 (0.002179) | 495.491 (3.953) | 106.9424 | 1.1206 | 295.68 | 1,536 | 131,072 |

Every run verified at exactly 201,326,592 counted tokens. Derived AP292 vs AutoResearch: params -41.66%, memory -35.08%, decode 1.985x, incremental KV exactly -50.00%, held-out BPB +0.042034.

Rank change, discovery to verification: AP292 1 to 3; AutoResearch 2 to 1; AP299 3 to 6; TPE 4 to 2; MOTPE 5 to 5; AP294 6 to 7; Baseline 7 to 4.

Memory and incremental KV are byte-identical across all 30 measured repetitions and all three seeds for every recipe, so their sample SD is exactly zero. Verified directly from each `final_inference.json`.

## 5. Downstream suite (9 metrics, all recipes, all seeds)

Metrics present for every run: heldout_climbmix_bpb/bits_per_byte, lambada_openai/{perplexity, exact_match}, blimp/{macro_accuracy, micro_accuracy}, piqa/{accuracy, length_normalized_accuracy}, arc_easy/{accuracy, length_normalized_accuracy}. BLiMP macro and micro accuracy are numerically identical for every run because all 67 configurations contain exactly 1,000 pairs (67,000 total); the paper reports one column and says why.

Means and seed SDs are in `paper/master_v2/generated/master_downstream_table.tex` and `master_downstream_sd_table.tex`. Honest reading recorded in the paper: AP292 leads on ARC-E accuracy, ARC-E length-normalized, and PIQA length-normalized, and is within 0.03 of the best LAMBADA accuracy; it trails on raw PIQA accuracy (-0.38) and LAMBADA perplexity (+0.06). Those gaps are smaller than seed SDs that reach 1.4 to 2.3 points, so no separation is claimed. BLiMP does separate (72.02 vs 74.56, SDs 0.36 and 0.59) and correlates with held-out BPB (r = -0.969) more strongly than with depth (r = 0.717).

## 6. Protocol constants

| Constant | Value | Justification in artifacts |
| --- | --- | --- |
| Fixed-time budget | 300.000 s | `benchmark_config.TRAINING_BUDGET_SECONDS`, immutable protected constant |
| Verification budget | 201,326,592 = 192 x 2^20 = 1536 x 131,072 = 384 x 524,288 | Divisible by both panel batch settings, so no partial update |
| Contract N budget | 88,604,672 = 169 x 524,288 | floor(median of three seed-3 baseline T runs / 524,288) x 524,288, per protocol.md |
| Verification seeds | 3, 4, 5 | Disjoint from discovery seeds (0, 1, 42) and earlier confirmation seeds (0, 1, 2) |
| Decode probe | BF16, batch 1, 256 in / 256 out, greedy, fixed protected prompt, 30 measured repetitions in 3 passes | benchmark_config.py; per-run final_inference.json |
| Peak memory | `max_memory_allocated()` after `reset_peak_memory_stats()` per repetition, checkpoint resident, reserved excluded, covers prefill and all decode steps | inference_benchmark.py `one_measured_repetition` |
| Incremental KV | validated cache bytes after final decode step minus after prefill (255 positions) | inference_benchmark.py |
| Panel size | 7 recipes | baseline + AutoResearch + TPE + MOTPE + 3 AutoPareto primary roles. The lowest-memory role (exp 124, archive rows 288-290) was not rerun at 201.3M. |

## 7. Budget study (4 + 4 = 8 completed attempts)

Verified from `runs/pair-a-official-20260906-fixed-warmup-v2`: four T attempts and four N attempts, single training seed 3, one adaptive trajectory per arm, 0 of 8 Pareto-eligible. Values as in the paper's Table 4. Registered protocol is nondirectional (`study/contracts_2026/protocol.md`). No causal claim is made anywhere in the manuscript.

## 8. Hypothesis provenance

- The exposure-confound question is retrospective: `paper/research_question_freeze.md` (2026-09-06) states explicitly that it is "a retrospective analysis freeze, not a new experimental preregistration". The master manuscript therefore presents it as an audit question, not a preregistered prediction, and never labels it H1/preregistered.
- The contract-behavior question is prospective and registered in nondirectional form before the paired campaigns launched.
- Per-attempt agent hypotheses are preserved separately and are not conflated with either.

## 9. Artifact availability (stated exactly this way in the paper)

Preserved and released: frozen recipe sources with hashes, protocol manifests, aggregate per-seed result summaries. Preserved, not released: model checkpoints, raw training logs and telemetry, the private SQLite registry. The manuscript makes no claim that checkpoints or raw logs are public, and no claim of availability "through the submission system".

## 10. Unresolved

- Ownership of several historical infrastructure repairs (inference stabilization, warm-up handling, relaunch decisions) cannot be attributed to human operator versus assisted debugging from preserved artifacts. Excluded from autonomous-contribution claims.
- The registered program text specifies separate-process warm-up; the executed controller uses same-worker warm-up with reconstructed state and verified initial-state hashes. Disclosed in the manuscript as an execution deviation.
