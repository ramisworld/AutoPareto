# Phase 6 amendment A1: local storage, timing-anomaly gate and lm-eval scope for P2

Approved by the author on 2026-11-16 ("Approve local-storage amendment A1 for P2, provided it
preserves all scientific protocols, hashes and measurements. Validate it before running P2.").

- **Written before** any P2 study run.
- **Parent plan:** `evaluation/phase6_unified_plan.json`, sha256 dc1e605e..., left unchanged.
  P0 and P1 run under the parent plan only.
- **A1 plan:** `evaluation/phase6_amendment_a1_plan.json`. It records the parent hash, the hashes of
  this file and of the new code, and the manifests named below.
- **Background:** the measurements behind A1 are in `PHASE6_AMENDMENT_LOCAL_STORAGE_DRAFT.md` and
  `PHASE6_INCIDENTS.md`.

## 1. What A1 changes: where bytes are read and written (P2 only)

| Item | Parent plan | Under A1 |
|---|---|---|
| Env A (training, held-out evaluator) | `/workspace/autopareto-camera-ready/.venv` on the network volume | `/opt/autopareto_envA` on local disk. It is built with `uv sync --locked` from the same `uv.lock`, and its `uv pip freeze` must equal the volume environment's (the hash is recorded in the A1 plan). |
| Training data and tokenizer | `~/.cache/autoresearch` resolves to `/workspace/.cache/autoresearch` | `~/.cache/autoresearch` resolves to `/root/autoresearch_cache_local`, a byte copy. Every file is checked against the E1 data manifest by the existing E1 data-identity gate. |
| Fixed-time run directories | written on the volume | each attempt runs in `/root/phase6_local/runs/<dir>`. It is then copied to `results/camera_ready_2026/phase6/runs/<dir>`, and every file's SHA-256 is compared. |
| Compile caches | ~2,000 files per run on the volume | archived as one uncompressed `compile_cache.tar` in the volume copy, with its hash checked |
| lm-eval | not run yet | Env C (`/opt/autopareto_lmeval`) and the dataset cache `/root/phase6_lmeval_cache` (offline) on local disk. Checkpoints are staged by SHA-256, as P0 inference already is. |

- **Env A equality, established 2026-11-16.** `uv sync --locked` reproduces the lockfile exactly.
  The volume environment also holds 13 packages installed outside the lockfile: datasets 5.0.1 and
  its dependencies aiohttp, aiohappyeyeballs, aiosignal, async-timeout, attrs, dill, frozenlist,
  multidict, multiprocess, propcache, xxhash and yarl. These were added to the local copy at the
  same versions, with `--no-deps`. Both freezes are now byte-identical (sha256 16ca2a74...).
- **lm-eval dataset revisions in the cache:**
  - The four shared tasks are byte-for-byte the revisions pinned by the frozen suite:
    - lambada_openai 900124bf
    - ai2_arc 210d026f
    - piqa 142f6d73
    - blimp 877fba08
  - The new tasks use these revisions:
    - hellaswag 218ec52e
    - winogrande 01e74176
    - openbookqa 388097ea
    - sciq 2c94ad3e
    - super_glue/boolq 3de24cf8
    - wikitext_document_level 64723477
- **Every copy is recorded.** Each volume copy holds `a1_copy_manifest.json`, which lists every file
  with its SHA-256. Each run record gives both the local and the volume path.
- **Nothing is deleted.**
- **Unchanged:** code, package versions, data bytes, checkpoints, seeds, the historical launcher
  (aa9dc8b2), 300 s of synchronised step time after step 10, the E1 gates, the evaluator, metrics
  and the one-retry rule.
- **When the switch happens:** the data symlink is re-pointed only after the P0/P1 chain has
  finished, so no P0 or P1 run is affected.

## 2. Gates added for P2 fixed-time runs

### Timing-anomaly gate (every attempt)

- **How it is measured:** per-step times (`dt`) are parsed from `training.log` for steps after
  step 10. Let m be their median. The anomaly is the total time by which steps slower than 2m
  exceed m.
- **When an attempt fails:** if the anomaly exceeds 3.0 s (1% of the 300 s window), or if fewer
  than 100 steps can be parsed.
- **What then happens:** under the existing rule, the failed attempt is kept and the run is
  repeated once. Every attempt's anomaly is reported.
- **Reason:** a stall inside the window cuts the tokens trained and so changes a fixed-time result.
  In P1, GPT A24 seed 6 lost about 33 s this way.

### E1 comparability gate (AR-ref seed-6 drift control, which runs first)

- **Pass condition:** total training tokens within [0.99 x min, 1.01 x max] of E1's AR-ref
  seeds 6 to 8, and held-out BPB within [min - 0.003, max + 0.003].
- **If it fails:** P2 stops and is reported. The reused E1 300 s results for AR-ref and GPT A22
  would then not be comparable with pod5 runs, and the author decides what follows.

## 3. lm-eval scope (author's decision)

- **New tasks on all 39 matched-token checkpoints:** ARC-Challenge, HellaSwag, WinoGrande,
  OpenBookQA, SciQ and BoolQ (`acc`, `acc_norm`), plus WikiText (bits per byte, word and byte
  perplexity).
  - Zero-shot, lm-evaluation-harness 0.4.13.
  - The scoring is the frozen evaluator's `ModelScorer`, wrapped by `scripts/phase6_lmeval.py`.
  - It runs at the frozen suite's runtime settings: scoring batch 8 and at most 2,048 tokens,
    both read from `evaluation/downstream_suite.json`.
- **Not repeated on all 39:** ARC-Easy, PIQA, LAMBADA and BLiMP. They are taken from the frozen
  evaluator, which has already scored them on the full test sets:
  - the 15 reused checkpoints in Phase 5A;
  - the 24 new checkpoints in P1.
- **Cross-check, run first:**
  - **Checkpoints:** AR-ref, GPT A22 and Opus A23, all seed 6.
  - **Tasks:** lm-eval's ARC-Easy `acc`, PIQA `acc`, LAMBADA `acc` and BLiMP macro accuracy.
  - **Pass condition:** each within 0.5 points of the frozen evaluator's value.
  - **If it fails:** stop and report before the 39-checkpoint run.
- **Candidate source for each checkpoint:** the one the frozen evaluator used.
  - Reused checkpoints: as listed in `phase5/A/downstream/downstream_models.json`.
  - New checkpoints: the recipe's `source_path`.
  - MOTPE seed 8: the attempt-3 checkpoint from amendment A2 (`PHASE6_AMENDMENT_A2.md`). Only
    this lookup was added to `run_phase6_p2.py`, and the A1 plan was regenerated before any P2 run.
- **WikiText caveat:** ClimbMix is a web corpus, so overlap with WikiText cannot be excluded.
  WikiText is reported with that caveat.

## 4. Validation before P2 (non-study)

- **Pilot:** one fixed-time run under A1, Opus A27 seed 6, written to `results/camera_ready_2026/
  phase6_pilot/a1/`. It exercises:
  - the local environment and data;
  - both gates;
  - the evaluator on A27's converted checkpoint (the pre-registered int8 risk);
  - the verified copy.
- **Its numbers are not study results.**
- **Not run under A1:** optional P3 runs (lineage and supplementary workloads) are not part of A1.
  They need separate approval.
