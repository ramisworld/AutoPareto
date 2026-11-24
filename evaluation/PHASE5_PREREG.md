# Pre-registration: Phase 5 (5A matched-token Opus frontier; 5B inference ablations)

Frozen 2026-11-08, before any Phase 5 run. The user authorised both phases on 2026-11-08. The commit that adds this
file and both plans is the registration. Plan hashes are recorded in the audit document (doc 24). Any later change is a
disclosed deviation.

Machine-readable plans:
- `evaluation/phase5a_opus_fixed_tokens_plan.json`
- `evaluation/phase5b_inference_ablation_plan.json`

Nothing in the E0/E1 plans, the Pair-A campaign records or earlier results is changed.

## Phase 5A: the three Opus frontier recipes at 201,326,592 training tokens

### Question
At exactly the tokens and updates of E1's 201.3M cell, on the same paired fresh seeds (6, 7, 8), data stream, evaluator
and A40, how do the Opus 5.5 frontier recipes (attempts 23, 24 and 27) compare with AutoResearch (AR-ref) and with AP292
(AP-Q)?

### Relation to the 2026-11-02 E1 extension (`E1_OPUS_EXTENSION_PREREG.md`)
That extension covered the single winner at all four E1 cells. Its frozen adapter refuses all three Opus sources:
- the error is `expected exactly one anchor: 'total_tokens = step * TOTAL_BATCH_SIZE'`;
- Opus rewrote the training-loop driver for Contract T.

Under that plan's own rule, the extension is therefore not executable. It stays unrun and is reported as such. Phase 5A is
a new, narrower study:
- one cell;
- all three frontier recipes;
- two documented changes, given below.

### Design
| Factor | Level |
|---|---|
| Recipes | Opus attempt 23 (the winner by the frozen confirmation rule), 24 and 27 (the extended frontier) |
| Budget | 201,326,592 tokens = 1,536 updates at TOTAL_BATCH_SIZE 131,072. All three Opus recipes already use 2^17 with DEVICE_BATCH_SIZE 64, the same as AR-ref and AP-Q, so tokens, updates and the loader call sequence all match. |
| Seeds | 6, 7, 8. These are paired with E1, fresh for every recipe and never used in discovery (seed 3). The seed is forced by the protected launcher, as in E1. |
| Comparators | E1's six 201.3M runs, reused unchanged (AR-ref and AP-Q, seeds 6–8, checkpoint hashes in the plan) |
| Drift control | AR-ref, 201.3M, seed 6, run first through the E1 executor unchanged |
| New runs | 9 treatment runs + 1 control |
| Order | The control, then seed blocks 6, 7, 8, with a cyclic Latin square over recipes within each block |

### Changes required to run the fixed-token protocol
Both are scripted and gated, and nothing is hand-edited.

1. **Driver mapping** (`scripts/opus_contract_t_to_phase1_source.py`). It restores the panel baseline's text for exactly
   these Contract-T driver lines:
   - the `contract_progress` call;
   - the perf_counter clock without warm-up exclusion;
   - the fallback time stop;
   - the NaN `val_bpb` placeholder;
   - the Contract-T constants and prints.

   It also moves the recipe's own post-training storage lines verbatim to just before the restored final evaluation.

   The unchanged protected transform then sets token-fraction LR progress and the exact-token stop, as for AR-ref and AP-Q.

   Gates (all pass; committed mapped sources and diffs are in `evaluation/phase5a_sources/`):
   - definitions are byte-identical apart from the `prepare` import;
   - statement-level changes are confined to the expected set: the synchronize position, the moved storage lines and
     `model.eval()`;
   - the transform accepts seeds 6–8;
   - the driver edit is identical for all three recipes;
   - the frozen adapter's own gates still hold.
2. **Deployment step** (`scripts/phase5a_export_deployed_checkpoints.py`). The Phase-1 recorder checkpoints inside the
   loop, before the recipe's post-training storage conversion. That conversion is bf16 matrices for all three recipes;
   attempt 27 also quantises its tables to int8. In Pair-A the scored model was the converted one. The export applies the
   recipe's own statements, with bit-exact load and reload gates.
   - For recipes 23 and 24 (matrices only), the protected loader's direct load must equal the deployed model bit for bit.
     That proves the step is a no-op for what the evaluator sees.
   - The unchanged Phase-1 evaluator scores the deployed checkpoints, including held-out BPB.

**LR semantics.** Contract T used time progress with every step counted. The token cell uses token fraction, as AR-ref
and AP-Q did in E1.

### Outcomes and analysis (fixed)
- **Primary:** for each recipe r, the held-out BPB difference Δr = BPB(r) − BPB(AR-ref).
  - It is paired over seeds 6–8 and reported as a mean with a t-based 95% CI (df = 2, t = 4.303).
  - Holm adjustment is applied across the 3 recipes.
  - "Confirmed difference at 201.3M tokens" only if the CI excludes 0; otherwise "inconclusive".
- **Secondary:**
  - the same contrasts on validation BPB;
  - Δ against AP-Q;
  - each recipe's mean, sd and CI;
  - the August panel (seeds 3–5) as an unpaired descriptive reference only;
  - the frozen downstream suite on the 9 Opus and 6 E1 final checkpoints (LAMBADA, BLiMP, PIQA, ARC-E), reported as
    descriptive means, sds and paired differences.
- **Training throughput:**
  - steady-state tokens/s per recipe;
  - ρr = tok/s(r) / tok/s(AR-ref).
- **Exploratory exchange rate.** kr is the token multiple AR-ref needs to match r's held-out BPB, read from AR-ref's E1
  curve (piecewise linear in ln tokens, per seed). A value outside the measured range is reported as a bound.
  - ρr > kr predicts a fixed-time win.
  - This is a prediction, not a measurement.
- **Fixed-time vs fixed-token:**
  - No fixed-time Opus run is made in this harness.
  - Fixed-time statements cite the Pair-A campaign (seed 3, one run each) and are never called confirmed.
- **Drift rule.** The control must reproduce its E1 cell within 0.005 BPB, with identical tokens and updates. Otherwise
  every Opus-vs-comparator contrast is reported as session-confounded, descriptively only.
- **Integrity:**
  - every E1 fixed-token gate applies (the deployed-manifest substitution is described in the plan);
  - each run gets one retry on the same seed, otherwise it is reported as failed;
  - no seeds are added;
  - no recipe is selected or dropped.
- **Estimated cost:** 3.1 A40-h, about $1.4 at $0.44/h.

## Phase 5B: what causes the Opus inference speed-up (no retraining)

### Question
Which inference-time mechanisms produce the Opus adapters' ~31× decode-only speed-up and their ~19× slower prefill?

The test applies to the existing Pair-A checkpoints of attempts 23, 24 and 27 (300 s, seed 3, hashes in the plan). Each
mechanism is switched off while the model, weights and protected harness stay the same.

### Variants
Each variant is the frozen source with only `build_inference_adapter` replaced
(`scripts/phase5b_inference_ablation.py`). V0 is the unmodified source.

| Variant | Change (same function, same key set, same scale) |
|---|---|
| V0 as_submitted | control |
| V1 no_cuda_graph | the same compiled kernels launched eagerly per token; no per-prompt capture |
| V2 no_fusion | graph kept; the same decode helper functions run uncompiled (no Inductor fusion) |
| V3 sdpa_attention | graph and fusion kept; masked SDPA instead of the custom split-key decode attention |
| V4 eager_unfused | neither graph nor fusion |
| V5 linear_kv_cache | full-length linear K/V buffers with the same window mask, instead of 272-slot rings |

Window-size changes are excluded, because they change the trained function.

### Protocol
- **Harness.** The frozen protected benchmark `study.contracts_2026.inference_benchmark` in candidate-only mode:
  - the registered 8 × 1,024-token prompts;
  - 128 greedy tokens, batch 1, bf16;
  - precision-aware correctness with FP32 adjudication;
  - a 20 s discarded warm-up;
  - 8 windows of at least 8 s.

  Its sha must equal the one Phase 2A recorded.
- **Hardware:** the same A40 as Phase 2.
- **Repeats and order.** Each (checkpoint, variant) runs twice: forward order, then fully reversed. The same-session
  baseline (the Pair-A baseline T checkpoint, canonical adapter) runs at the start, middle and end. That is 39 runs, each
  a separate process.
- **Kernel census** (`scripts/phase5b_kernel_census.py`, diagnostic and not timing):
  - GPU kernels and host launch calls per decode token;
  - attempt 23 × 6 variants, plus the baseline.

### Outcomes and analysis (fixed)
- **Per run:**
  - decode tok/s with the frozen stability class;
  - prefill ms;
  - end-to-end tok/s, i.e. 128 / (prefill + 127-token decode), median over repetitions;
  - peak memory;
  - K/V bytes;
  - correctness;
  - the greedy sample.
- **Factors** (multiplicative, with both repeats shown; no significance tests):
  - graph = V0/V1;
  - fusion = V0/V2;
  - joint = V0/V4;
  - interaction;
  - attention kernel = V0/V3;
  - ring = V0/V5 (speed) and V5/V0 (memory);
  - architecture under eager execution = V4/baseline.

  The same factors are computed for prefill and end-to-end. Which variant maximises end-to-end throughput is reported.
- **Stability.** Runs that fail the frozen stability rule are reported and not re-run. No threshold changes.
  - If host timing is unstable (pod4 failed the CPU screen), that is reported as a limitation.
  - No pod migration.
- **Predictions recorded before measurement** (P1–P6 in the plan). They carry no consequence for reporting.
- **Estimated cost:** about 2 A40-h, about $0.9.

## Order of work and stop rules
1. 5A training (control first).
2. 5A downstream.
3. 5B.
4. Verified off-pod backup and analysis.

Work stops and is reported on:
- an unexpected infrastructure failure;
- an integrity failure that repeats on retry;
- an estimated cost of more than 2× this registration.

No manuscript is changed and nothing is submitted.
