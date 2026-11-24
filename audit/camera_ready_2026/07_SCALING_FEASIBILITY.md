# 07 — Scaling feasibility: can the agent-discovered recipes be tested at larger scale?

**Purpose.** This audit supports a possible follow-up paper, *"Do AI-discovered optimizations survive scaling?"* It records what the existing evidence shows and what the current tooling can run.

> **Lead-auditor consistency note.** Two estimates in this file use unpaired or mixed-seed inputs. Cite the seed-matched estimates from `05_STATISTICAL_AND_CAUSAL_AUDIT.md`:
> 1. **The 300-second AP292 − AutoResearch contrast.** This file gives −0.0054; it mixes AutoResearch's seed-42 discovery run with its seed-0/2 re-runs. Use the paired value **−0.0067** (95% CI [−0.0130, −0.0003], seeds 0–2), from `05` §1.2.
> 2. **The batch-only share of AutoResearch's 300-second gain.** This file gives 96% against the 5-seed early baseline. The seed-matched value is **97.8%**, with the non-batch-only effect at +0.0023 rather than +0.0119, from `05` §3.6.
>
> The conclusions are unchanged. The gap between the two baseline campaigns (about 0.005 BPB on the same seeds) is a session-level exposure offset (`05` §2.1).
>
> 3. **Checkpoints.** Statements below that 11 of the 21 final 201M checkpoints are "not stored locally" refer to the working tree. The inventory (`01` §0 item 5, §C.3) found all 21 `final.pt` files as hash-verified LFS objects in `.git/lfs/objects`, so they are recoverable locally. The 63 milestone checkpoints were not located.

**Method.**
- The audit is read-only. No training, GPU, cloud, ssh or paid API call was made, and no repository script was run.
- SQLite files were opened only as `file:<path>?mode=ro&immutable=1`.
- No repository file was written except this one.
- Audit date: 2026-10-28. The repository was on branch `autopareto/professor-ready-2026` at commit `6bcb69e`, and the working tree had uncommitted changes.

**Labels**
- **OBSERVED**: read directly from a record. The file, line or JSON key is cited.
- **DERIVED**: computed by this audit. The arithmetic is shown, or the scratch script that computed it is named.
- **INFERRED**: an interpretation or an extrapolation.
- **UNKNOWN**: the evidence is not available locally.

## Abbreviations

| Name | Path or meaning |
|---|---|
| `BASE` | `results/candidates/baseline-a40-seed0/0002-train.py` (sha256 `428e22f1…`). The registered baseline. |
| `ARS` | `results/candidates/autoresearch-diagnostic5-seed42-20260724/0005-train.py` (`b3a961ee…`). The AutoResearch reference. |
| `TPES` / `MOTS` | `results/candidates/tpe-seed42-20260805/0021-train.py` (`8547b278…`) and `results/candidates/motpe-seed42-20260805/0022-train.py` (`62882a20…`) |
| `AP292S` / `AP299S` / `AP294S` | `results/candidates/autopareto-seed0-20260731/{0022,0025,0024}-train.py` (`55a7fd8b…`, `36e3ad91…`, `6d0d8d1d…`) |
| `P1` | `results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/`. `P1/runs/<run>/` holds the per-run files. |
| `RDB` | `results/results.db`, opened read-only and immutable |
| `PA` | `runs/pair-a-official-20260914-new-a40-continuation-20260916-r2/`. The latest Pair-A namespace. |
| `PAv2` | `runs/pair-a-official-20260906-fixed-warmup-v2/`. An earlier, superseded Pair-A namespace. |
| `SCR` | `/private/tmp/claude-501/-Users-Rami-Documents-dev-autopareto/084ded0c-4778-442b-9b7f-c93e451f1f5d/scratchpad/audit/scaling/`. The audit scripts. This folder is ephemeral and outside the repo. |
| code-FLOPs | The training script's own per-token estimate, `estimate_flops()` (`BASE` L208–222): 6·(non-embedding parameters including `lm_head`) + 12·heads·head_dim·Σ windows. |
| `C0` | 2.1×10¹⁶ code-FLOPs, the size of one 300-second discovery run on the A40. See C1. |

All seven frozen sources match the SHA-256 values in `evaluation/phase1_public_manifest.json` → `models[*].source_sha256` (OBSERVED; checked with `shasum -a 256`).

**Audit scripts.** All are in `SCR`. All are read-only with respect to the repo.

| Script | What it does |
|---|---|
| `extract_phase1.py` (output `phase1_runs.json`) | Collects per-run throughput, memory and milestones for the 21 runs |
| `extract_pair_a.py` (output `pair_a.json`) | Tabulates the Pair-A attempts |
| `model_math.py` | Replicates the code's parameter-count and FLOP formulas. It reproduces all seven logged parameter counts and "Estimated FLOPs per token" values exactly. |
| `milestones_table.py` | Milestone curves, and 5-minute versus 201M-token comparisons |
| `pair_a_decompose.py` | Splits each Pair-A edit's effect into a token part and a per-token part |
| `recon_pair_a.py` | Rebuilds the Pair-A sources from the recorded patches |
| `ladder.py` (output `ladder_output.txt`) | Costs and memory for the scale ladders |

---

## Summary

1. **One knob sets model size, and nothing sets the training budget.**
   - The code is a nanochat-style "depth dial". Width = ⌈`DEPTH`·`ASPECT_RATIO`/`HEAD_DIM`⌉·`HEAD_DIM`, and heads, value-embedding layers and attention windows follow from depth.
   - AdamW learning rates are scaled automatically by (width/768)^−½. The Muon learning rate does not depend on width.
   - Batch size, learning-rate values and weight decay are absolute numbers.
   - Every frozen recipe measures schedule progress in wall-clock time (300 s). No rule ties tokens to parameters.
   - (OBSERVED, part A.)
2. **All seven panel recipes change only constants, so a script can promote them by editing those constants.** The difficulty is that much of what the agent discovered is model size.
   - AP292, AP299 and AP294 are three sizes, (d4,w512), (d3,w512) and (d3,w384), of one recipe. They share the same non-size edits: `TOTAL_BATCH_SIZE` 2¹⁷ and learning rates 12.5–17% lower than the baseline's.
   - Under a depth dial, AP292 and AP294 become the same recipe. Under a width dial, AP294 and AP299 do.
   - (DERIVED from diffs.)
3. **The 201M-token panel is not a scale-up for the AutoPareto recipes.**
   - Compared with each recipe's own 300-second runs, the panel gave AP292 1.28× the tokens, AP299 1.03× and AP294 0.94×. It gave the baseline 2.17× and AutoResearch 2.27×.
   - The prompt's "about 88M tokens" in 300 s applies only to the baseline and AutoResearch. The AutoPareto recipes processed 157–215M tokens in 300 s.
   - (DERIVED, part B1.)
4. **The gain from a smaller batch shrinks with more training.**
   - The AutoResearch-minus-baseline validation gap was −0.107 BPB at 300 s and −0.061 at 201M tokens (paired seeds −0.057, −0.064, −0.061).
   - Changing only the batch size reproduces 96% of AutoResearch's 300-second gain. Changing only the other settings makes the baseline worse by 0.012.
   - (OBSERVED and DERIVED, parts B1 and B6.)
5. **The milestones inside the 201M-token runs cannot stand in for shorter runs.**
   - A checkpoint whose learning rate has not finished decaying scores 0.006–0.061 BPB worse than a complete run at a similar or smaller token count.
   - Recipes can be compared at a common learning rate only at the 25% milestone. (DERIVED, part B3.)
6. **No existing run separates model size from compute above `C0`.**
   - The 300-second comparisons hold compute roughly fixed. The 201M-token panel holds tokens fixed but lets compute vary from 0.67× to 2.29× `C0`.
   - The only runs larger than 50M parameters are 300-second hyperparameter-search runs, plus two single-seed fixed-token runs at 85.9M and 135.3M parameters in a superseded Pair-A namespace.
   - (OBSERVED, parts B4 and B7.)
7. **At `C0`, the agent chose roughly compute-optimal model sizes.**
   - Among 138 runs with batch 2¹⁷, the best 300-second validation BPB is at 24–30M parameters.
   - A rule of about 20 tokens per FLOP-relevant parameter, applied at `C0`, reproduces AP294's and AP299's exact shapes.
   - The edits that shrank the model in the Pair-A fixed-time arm help only by buying more tokens. After subtracting the token effect, every one of them is neutral or harmful per token, by +0.001 to +0.044 BPB.
   - (DERIVED and INFERRED, parts B4 and B5.)
8. **Compute is cheap. Engineering and provenance are the binding constraints.**
   - On one A40 at $0.44/h, the 7-recipe panel × 3 seeds costs 12.8 / 46 / 179 GPU-hours ($6 / $20 / $79) at 4× / 16× / 64× the tokens.
   - Promoting six recipe families to 4×, 16× and 64× `C0` (models up to about 150M parameters) costs about 130 GPU-hours ($57).
   - Memory fits once `DEVICE_BATCH_SIZE` is 32 on the top rung.
   - (DERIVED, part C.)
9. **The blockers are engineering and provenance work** (part D):
   - The data cache holds an unknown number of tokens. At least 287M are confirmed.
   - The tokenizer files are not stored locally.
   - The staged executor is hard-coded to the 201M-token plan.
   - The Phase-1 runtime assumes 201M tokens and a device batch of 64. It fails for `TOTAL_BATCH_SIZE` of 2²⁰ or more.
   - The Pair-A runtime limits batch size to 2¹⁹ or less and fixes the time contract at 300 s.
   - The Phase-1 transform rejects Pair-A sources.
   - 11 of the 21 final 201M checkpoints and all 63 milestone checkpoints are not stored locally.
   - Two Phase-1 implementation files no longer match their recorded hashes.
10. **Camera-ready pilot (part E), optional and capped.** It is placed after E0, E1 and E4 of `08_EXPERIMENT_PLAN.md`.
    - **Tier 1a, the core:** AutoResearch and AP292 at 4× tokens, plus AP292 at a 7× budget that matches AutoResearch's 4× compute. 9 runs, about 7 GPU-hours (about $3).
    - **Tier 1b, an add-on:** a 2×2 factorial of batch × non-batch settings at 1× and 4× tokens. 15 runs, about 10 GPU-hours.
    - Seeds 3–5, paired with the existing 201M panel.
    - It tests whether the ranking of AP292 against AutoResearch, and the dominant discovered edit, survive 4× training.
    - It says nothing about larger models.
11. **Long-term study (part F).**
    - A preregistered promotion operator.
    - A 4-rung ladder at 4^k·`C0` (k = 0–3).
    - Edit-level paired promotions. Edits are sampled across all levels of discovery-time effect, not just the winners.
    - Effects are re-measured at the discovery budget with fresh seeds. This separates regression to the mean from loss of the effect with scale.
    - Cost: about 860–1,190 GPU-hours ($380–$525), plus agent API cost.

---

## A. How the training code parameterizes scale

### A1. One size knob, and no budget rule (OBSERVED)

**Size**
- `DEPTH` is the only size constant (`BASE` L449–450, under the "# Model size" heading).
- `build_model_config(depth)` (`BASE` L469–477) sets:
  - `model_dim = ceil(depth·ASPECT_RATIO / HEAD_DIM)·HEAD_DIM`;
  - `num_heads = model_dim / HEAD_DIM`;
  - `n_kv_head = num_heads`.
  - In `TPES` and `MOTS`, `n_kv_head` comes instead from `KV_HEAD_RATIO` (L437, L476–480, L483). Its value is 1, so the result is still full multi-head attention.

**Derived structure**
- Value embeddings sit on alternating layers and always include the last layer: `has_ve` (`BASE` L47–49).
  - At depth 3 this means layers [0, 2]. At depth 4 it means [1, 3]. The parity therefore flips with depth (DERIVED, `SCR/model_math.py`).
- The window pattern repeats over the layers, and the last layer is forced to full attention (L195–206).
  - "S" means half the sequence (1,024 tokens). "L" means the full sequence (2,048).
- The MLP is always 4× width (L102–103), with ReLU² activation (L107).
- The logit softcap is 15 (L282–285).
- The rotary table covers 10× the sequence length (L144).

**Initialization**
- Weights are drawn uniformly with s = √3·width^−½ (L156).
- Output projections start at zero (L161, L163).
- Nothing in the initialization depends on depth.

**Optimizer split**
- Muon trains every parameter inside `transformer.h`, grouped by shape (L239, L257–262).
  - Muon settings:
    - Newton–Schulz steps: 5 (6 in `ARS`, L261).
    - `beta2` is 0.95 (NorMuon).
    - Weight decay is cautious.
    - The learning rate is multiplied by max(1, rows/cols)^½ (L412).
  - There is no width rule for Muon.
- AdamW trains `lm_head`, `wte`, the value embeddings and two per-layer scalars.
  - Their learning rates are multiplied by `dmodel_lr_scale = (model_dim/768)^−0.5` (L248–256).

**Batch**
- `grad_accum_steps = TOTAL_BATCH_SIZE // (DEVICE_BATCH_SIZE·2048)` (L495–497). Gradient accumulation is therefore supported, subject to an assert that the batch divides evenly (L496).
- `DEVICE_BATCH_SIZE` is 64 in all seven recipes.
- A device batch of 128 did not fit on the A40 for the 50M baseline (`PROJECT_STATE.md` L959).

**Schedule**
- In the frozen sources, `progress = total_training_time / TIME_BUDGET` with `TIME_BUDGET = 300` (`BASE` L555; `prepare.py` L31). The clock skips steps 0–10, including compilation (L578–579).
- Learning-rate warmup, warmdown and final fraction are all fractions of `progress` (L518–525).
- Muon weight decay falls linearly to 0 with `progress` (L531–532).
- Muon momentum ramps from 0.85 to 0.95 over **300 optimizer steps** (L527–529). This is an absolute step count.

**Fixed elsewhere**
- Sequence length is 2,048 (`prepare.py` L30).
- The vocabulary has 8,192 tokens (`prepare.py` L45).
- The validation set is 20,971,520 tokens (`prepare.py` L32).

**No token-budget rule**
- No constant links tokens to parameters. Under the 300-second contract, a larger model automatically gets fewer tokens.
- The Phase-1 transform shows that switching the schedule from time to tokens is a mechanical step:
  - it rewrites `progress` to `step·TOTAL_BATCH_SIZE/PHASE1_TOKEN_BUDGET` (`protected/phase1_source_transform.py` L145–152);
  - it stops at an exact step count (L199–206);
  - the 201M-token runs used this form (`P1/runs/phase1-autopareto_292-seed3/phase1_train.py` L561–562, L584, L647).
- The Muon momentum ramp stays step-based. The preflight records this as deliberate (`scripts/prepare_phase1_equal_tokens.py` L290–297, `schedule_audit`).

### A2. The seven frozen recipes (OBSERVED from the sources; derived shapes from `SCR/model_math.py`)

| Constant (line in `BASE`) | Baseline | AutoResearch | TPE | MOTPE | AP292 | AP299 | AP294 |
|---|---|---|---|---|---|---|---|
| `DEPTH` (L450) | 8 | 8 | 6 | 6 | 4 | 3 | 3 |
| `ASPECT_RATIO` (L433) | 64 | 64 | 80 | 48 | 128 | 160 | 128 |
| Realized width / heads (DERIVED) | 512 / 4 | 512 / 4 | 512 / 4 | 384 / 3 | 512 / 4 | 512 / 4 | 384 / 3 |
| Effective width/depth ratio (DERIVED) | 64 | 64 | **85.3** (480 rounded up to 512) | **64** (288 rounded up to 384) | 128 | **170.7** (480 rounded up to 512) | 128 |
| `HEAD_DIM` (L434) | 128 | 128 | 128 | 128 | 128 | 128 | 128 |
| KV heads | = heads | = heads | ratio 1 (= heads) | ratio 1 (= heads) | = heads | = heads | = heads |
| `WINDOW_PATTERN` (L435) | SSSL | SSSS | SSSS | L | SSSL | SSSL (realized S,S,L) | SSSL (realized S,S,L) |
| `TOTAL_BATCH_SIZE` (L438) | 2¹⁹ (4 accumulation steps) | 2¹⁷ | 131072 (comment says "~524K"; stale) | 131072 (same stale comment) | 2¹⁷ | 2¹⁷ | 2¹⁷ |
| Embedding / unembedding / matrix / scalar LR (L439–442) | 0.6 / 0.004 / 0.04 / 0.5 | 0.7 / 0.006 / 0.04 / 0.4 | 0.3 / 0.002 / 0.03 / 0.25 | 0.45 / 0.006 / 0.04 / 0.25 | 0.5 / 0.0035 / 0.035 / 0.5 | same as AP292 | same as AP292 |
| Weight decay (L443) | 0.2 | 0.1 | 0.3 | 0.1 | 0.2 | 0.2 | 0.2 |
| Warmdown / final LR fraction (L446–447) | 0.5 / 0 | 0.6 / 0.078125 | 0.3 / 0.05 | 0.7 / 0.05 | 0.5 / 0 | 0.5 / 0 | 0.5 / 0 |
| Muon Newton–Schulz steps | 5 | **6** (L261) | 5 (L450) | 5 | 5 | 5 | 5 |
| Parameters | 50,332,176 | 50,332,176 | 39,846,284 | 26,345,772 | 29,360,392 | 26,214,662 | 17,891,526 |
| Code-FLOPs per token (`P1/runs/*/training.log`, "Estimated FLOPs per token") | 2.3908e8 | 2.3279e8 | 1.8245e8 | 1.3920e8 | 1.3212e8 | 1.0696e8 | 6.9600e7 |
| Value-embedding share of parameters (DERIVED) | 33% | 33% | 32% | 36% | 29% | 32% | 35% |

*Line numbers for `TPES` and `MOTS` are shifted by one to three lines (L434–454). `ARS` and the AutoPareto sources use the `BASE` line numbers.*

**Implications for the parameter-to-token convention (DERIVED).** About a third of all parameters are value embeddings. These cost no FLOPs, so "parameters" and "FLOP-relevant parameters" differ by 1.7–2.1×. The parameter-to-token ratio must therefore be defined explicitly. This audit uses N_flop, the non-embedding parameters including `lm_head`, which is the quantity `estimate_flops` counts.

### A3. Relative versus absolute edits (OBSERVED from diffs against `BASE`; classification INFERRED)

**Relative (ratio) settings.** These keep their meaning at any size:
- `ASPECT_RATIO`;
- `KV_HEAD_RATIO`;
- the window pattern, as a fraction of the sequence;
- warmup and warmdown ratios and `FINAL_LR_FRAC`;
- value-embedding alternation;
- the 4× MLP;
- the 1/√width AdamW learning-rate scaling.

**Absolute settings:**
- `DEPTH`;
- `TOTAL_BATCH_SIZE` and `DEVICE_BATCH_SIZE`;
- every learning-rate value, weight decay and the Adam betas;
- the number of Newton–Schulz steps;
- the 300-step momentum ramp;
- the 300-second `TIME_BUDGET`.

**What each recipe changed relative to the baseline:**
- **AP292, AP299 and AP294** changed two absolute settings, `DEPTH` and `TOTAL_BATCH_SIZE`; one relative setting, `ASPECT_RATIO`; and three absolute learning rates, each lowered by 12.5–17%.
- **AutoResearch** changed only non-size constants, plus Newton–Schulz steps 5→6. This makes it the cleanest test of whether non-size edits transfer.
- **TPE and MOTPE** are points sampled from a hyperparameter grid. They have no edit intent.

**The agent reasoned in absolute widths and depths.** For example, its stated rationales include:
- "Explore the next distinct rounded width bucket at four layers" (`RDB` experiment 130, `description`);
- "tuned depth-three width-384 point" (experiment 136);
- "test a wider three-layer model" (experiment 137).

The recipe's intended meaning is therefore partly absolute, and any rule for promoting it is a choice the agent never specified (INFERRED).

### A4. Can each recipe be promoted mechanically?

**Mechanical procedure.** Edit `DEPTH`, optionally `ASPECT_RATIO`, and `DEVICE_BATCH_SIZE`. Keep every other constant. Apply the Phase-1 time-to-token transform.

| Recipe | Mechanical? | What is ambiguous |
|---|---|---|
| Baseline | Yes. This is the reference depth dial. | At odd depth with ratio 64, rounding inflates the width/depth ratio (d9 → width 640, ratio 71). Whether `TOTAL_BATCH_SIZE` 2¹⁹ is right at larger budgets is an open question for the baseline too. |
| AutoResearch | Yes | 1. The batch of 2¹⁷ is absolute: keep it, or scale it? 2. With SSSS, only the forced last layer is full-attention, so the full-attention share falls as 1/depth (1/8 at d8, 1/16 at d16). Is the edit "all layers local", or "1/8 of layers global"? 3. The 300-step momentum ramp covers 20% of a 201M-token run but 0.3% of a 64× run. |
| TPE | Yes | Rounding: the effective width/depth ratio is 80–96, depending on depth. Depth 6 belongs to a sampled point and has no intent behind it. Batch of 2¹⁷. |
| MOTPE | Yes, but the model's shape drifts | Severe rounding: the effective ratio is 64 at d6, 48 at d8 and 51 at d10. The model also becomes deep and narrow as it grows, for example d14 × 768 at 64× `C0` (part C3). |
| AP292 | Yes. Ratio 128 never rounds. | The discovered edit **is** the shallow depth. A depth dial (keep the ratio) and a width dial (keep depth 4) give different models. Under the depth dial, AP294 collapses into AP292's family. Batch of 2¹⁷. |
| AP299 | Yes | Rounding: the effective ratio is 160–179. The agent's stated intent was "width 512 at depth 3", which is absolute. |
| AP294 | Yes, but it is not a distinct recipe once promoted | It differs from AP292 only in `DEPTH` (3 against 4), and from AP299 only in `ASPECT_RATIO` (128 against 160) (DERIVED from diffs). |

**Pair-A edits are harder to promote.** Several use absolute layer indices or literal constants:
- `has_ve` returns `layer_idx in (n_layer - 3, n_layer - 1)` and `layer_idx in (1, 3)`;
- `n_kv_head=1`;
- a factorized output head of rank 384;
- a literal `85458944` in `progress` (fixed-token attempt 2);
- bf16 `lm_head` storage.

These are visible in `PA/*/history.jsonl` → `candidate_patch`. They need to be rewritten in relative terms by a human or an LLM, which is not mechanical (INFERRED).

---

## B. Existing evidence relevant to scale transfer

### B1. 300-second discovery against the 201M-token panel: the same model trained longer

**Sources.**
- 300-second rows: `RDB` `experiments` (`val_bpb`, `total_training_tokens`).
- 201M-token rows: `P1/runs/*/checkpoint_evaluations.json` → `checkpoints[3].validation_bpb`.

**Metric.** Both columns use the same evaluator: `prepare.evaluate_bpb` on shard 06542. Every schedule ran to completion.

**Labels.** Cells are OBSERVED. Ratios, Δ, FLOPs and slopes are DERIVED (`SCR/milestones_table.py`).

| Recipe | 300-second rows (`RDB` ids; seeds) | 300-second tokens (mean) | 300-second val BPB, mean ± SD | 201M val BPB, mean ± SD (seeds 3–5) | Token ratio | ΔBPB | Code-FLOPs, 300 s → 201M (×`C0`) | ΔBPB per e-fold of tokens |
|---|---|---:|---:|---:|---:|---:|---|---:|
| Baseline | 3–7 (seeds 0–4) | 92.7M | 1.2069 ± 0.0080 | 1.0934 ± 0.0038 | 2.17 | −0.1134 | 2.22e16 → 4.81e16 (1.06 → 2.29) | −0.146 |
| AutoResearch | 68, 75, 77 (seeds 42, 0, 2) | 88.6M | 1.1001 ± 0.0044 | 1.0327 ± 0.0005 | 2.27 | −0.0674 | 2.06e16 → 4.69e16 (0.98 → 2.23) | −0.082 |
| TPE | 183 (seed 42) | 113.8M | 1.0977 (one run) | 1.0572 ± 0.0030 | 1.77 | −0.0405 | 2.08e16 → 3.67e16 | −0.071 |
| MOTPE | 234, 300 (seed 42, run twice) | 153.9M | 1.0978 ± 0.0004 | 1.0793 ± 0.0026 | 1.31 | −0.0185 | 2.14e16 → 2.80e16 | −0.069 |
| AP292 | 291–293 (seeds 0–2) | 156.9M | 1.0947 ± 0.0025 | 1.0777 ± 0.0019 | 1.28 | −0.0170 | 2.07e16 → 2.66e16 (0.99 → 1.27) | −0.068 |
| AP299 | 297–299 (seeds 0–2) | 195.9M | 1.0988 ± 0.0008 | 1.0971 ± 0.0007 | 1.03 | −0.0017 | 2.10e16 → 2.15e16 | (−0.062, ill-conditioned) |
| AP294 | 294–296 (seeds 0–2) | 215.3M | 1.1251 ± 0.0018 | 1.1284 ± 0.0008 | **0.94** | +0.0033 | 1.50e16 → 1.40e16 (0.71 → 0.67) | (−0.049, ill-conditioned) |

**Worked arithmetic for AP292.** ln(201,326,592 / 156.9M) = 0.2494, and −0.0170 / 0.2494 = −0.068 per e-fold. The code-FLOPs are 1.3212e8 × 156.9M = 2.07e16.

**Findings**
- **Token ratios (DERIVED).** For the AutoPareto recipes, "equal tokens" mostly means similar tokens, not more tokens.
  - The preflight records the 300-second token counts for the registered rows: AP292 159,252,480; AP299 196,345,856; AP294 209,715,200 (`evaluation/phase1_equal_token_preflight.json` → `models[*].historical_five_minute.total_training_tokens`).
- **Rankings flip (DERIVED).**
  - At 300 s, which is roughly equal compute: AP292 (1.0947) < TPE (1.0977) ≈ MOTPE (1.0978) < AP299 (1.0988) < AutoResearch (1.1001) < AP294 < baseline.
  - At 201M tokens, where tokens are equal but compute is not: AutoResearch (1.0327) < TPE < AP292 < MOTPE < baseline < AP299 < AP294.
- **Large-batch training gains more from extra tokens.**
  - The baseline, with a 2¹⁹ batch, improves at −0.146 BPB per e-fold. The 2¹⁷-batch recipes improve at −0.049 to −0.082 (DERIVED).
  - The AutoResearch-minus-baseline gap shrinks from −0.1068 at 300 s to −0.0608 at 201M (paired seeds: −0.0571, −0.0641, −0.0611; SD 0.0035; held-out: −0.0468 ± 0.0044) (DERIVED from `P1` per-run values).
- **A naive extrapolation (INFERRED, not evidence).** If both slopes stayed log-linear, the gap would close near e^(0.0608 / (0.146 − 0.082)) × 201M ≈ 520M tokens. Slopes flatten in reality, so this only motivates the pilot's 4× rung.
- **Confounds.**
  - The two columns use different seed sets.
  - They ran on different pods: throughput differs (AP294 ran at 834,416 tokens/s in discovery experiment 136 but 694,039–729,853 in its confirmations, ids 294–296, `RDB` → `training_tokens_per_second`).
  - The schedules count progress differently: time in one, tokens in the other.
  - TPE and MOTPE have a single 300-second seed.
  - Even so, AP299 at matched tokens (196M against 201M) reproduces within 0.0017 BPB. The time-to-token translation is therefore not a large distortion (DERIVED).
- **Mislabelled metric (OBSERVED, a side finding for audit 03).**
  - `results/phase1/discovery_heldout_bpb.json` L4 labels its numbers "held-out ClimbMix bits per byte".
  - All seven values (L7–13) are identical to `RDB` `val_bpb`, which is the shard-06542 search metric. For example, baseline 1.21272999168195 equals `RDB` id 4.
  - They are not the shard-06541 held-out metric used at 201M.

### B2. Validation logged at milestones inside the 201M-token runs (OBSERVED)

**What exists.** Every one of the 21 runs has checkpoints at 50,331,648, 100,663,296, 150,994,944 and 201,326,592 tokens (`scripts/prepare_phase1_equal_tokens.py` L26; `protected/phase1_runtime.py` L531–537).

`P1/runs/<run>/` holds the following:
- `checkpoint_evaluations.json` → `checkpoints[i].{token_count, optimizer_step, validation_bpb, cumulative_steady_state_seconds}`. This is the validation BPB at each milestone.
- `heldout_bpb`, at the final checkpoint only.
- `training.log`: every optimizer step's smoothed training loss, learning-rate multiplier (`lrm`), throughput and data `epoch`.
- `step_timing.jsonl`: the duration of every step.
- `milestones/manifest.json`, with checkpoint hashes.

**What is missing.**
- The milestone `.pt` files are not stored locally. Only their hashes and sizes are recorded (`results/snapshots/…/inventory.json` → `phase1.files` with `kind == "checkpoint"`: 188 entries, of which 84 belong to the 21 scientific runs and 104 to non-scientific smoke runs).
- There is no held-out or downstream evaluation at any intermediate milestone.

**Data never repeated.** Every log ends at `epoch: 1` (DERIVED, `SCR/extract_phase1.py`).

**Validation BPB at each milestone.** Means over seeds 3–5; the learning-rate multiplier at the milestone step is in brackets (DERIVED from `training.log`).

| Recipe | 50.3M | 100.7M | 151.0M | 201.3M |
|---|---:|---:|---:|---:|
| Baseline | 1.3538 [1.0] | 1.2127 [1.0] | 1.1365 [0.51] | 1.0934 [0.01] |
| AutoResearch | 1.2026 [1.0] | 1.1254 [0.85] | 1.0710 [0.46] | 1.0327 [0.08] |
| TPE | 1.2238 [1.0] | 1.1590 [1.0] | 1.1182 [0.84] | 1.0572 [0.05] |
| MOTPE | 1.2393 [1.0] | 1.1591 [0.73] | 1.1117 [0.39] | 1.0793 [0.05] |
| AP292 | 1.2404 [1.0] | 1.1772 [1.0] | 1.1178 [0.50] | 1.0777 [0.0] |
| AP299 | 1.2517 [1.0] | 1.1918 [1.0] | 1.1356 [0.50] | 1.0971 [0.0] |
| AP294 | 1.2758 [1.0] | 1.2174 [1.0] | 1.1644 [0.50] | 1.1284 [0.0] |

**Gap to AutoResearch at the four milestones (DERIVED):**
- Baseline: +0.151, +0.087, +0.066, +0.061.
- AP292: +0.038, +0.052, +0.047, +0.045.
- TPE: +0.021, +0.034, +0.047, +0.025. TPE does not decay until progress 0.7, so the 75% point flatters the others.

**Where comparisons are clean.** At 25%, every recipe is still at learning-rate multiplier 1.0, so this is the only milestone where all recipes can be compared at the same learning rate. Weight decay is also at 0.75× its starting value for all, but the starting values differ by recipe.

### B3. Learning-rate schedule confound: intermediate milestones are not shorter runs (DERIVED)

Each milestone is compared with a complete 300-second run that has a similar token count. The 300-second values are the ones in B1.

| Recipe | Milestone (tokens, LR multiplier) | Milestone val BPB | Complete 300-second run (tokens) | Complete-run val BPB | Gap |
|---|---|---:|---|---:|---:|
| AP292 | 151.0M, 0.50 | 1.1178 | 152–159M | 1.0947 | +0.023 |
| MOTPE | 151.0M, 0.39 | 1.1117 | 153–155M | 1.0978 | +0.014 |
| AutoResearch | 100.7M, 0.85 | 1.1254 | **88.6M (fewer)** | 1.1001 | +0.025 |
| TPE | 100.7M, 1.00 | 1.1590 | 113.8M | 1.0977 | +0.061 |
| Baseline | 100.7M, 1.00 | 1.2127 | 92.7M (fewer) | 1.2069 | +0.006 |

**What this shows.** A checkpoint taken before the learning rate has finished decaying overstates the loss by 0.006–0.061 BPB. The effect depends on the recipe, so even milestone rankings are distorted.
- Between 50M and 201M tokens, MOTPE and AP292 swap places, and the baseline climbs from last to fifth (B2 ranks, DERIVED).

**Valid uses of the milestones (INFERRED):**
- the shape of each run's own loss curve;
- cross-recipe comparisons at 25%, where all learning rates are equal.

**Not valid:** treating a milestone as a shorter run, or fitting a token-scaling law to it.

### B4. Runs at other model sizes

**Equal-time size sweep (DERIVED from `RDB`).**
- Input: the 138 completed 300-second rows with `TOTAL_BATCH_SIZE` 131072. Batch size was read from each row's JSON `description`, or by parsing its candidate source.
- The table gives the best validation BPB in each parameter range:

| Parameters | Best val BPB | Row (method) | Tokens | Tokens/parameter |
|---|---:|---|---:|---:|
| < 12M | 1.1532 | 142 (AutoPareto) | 215M | 18.7 |
| 12–16M | 1.1216 | 149 (AutoPareto) | 206M | 14.0 |
| 16–20M | 1.1054 | 122 (AutoPareto) | 204M | 10.4 |
| 20–25M | 1.0948 | 290 (confirmation) | 179M | 7.3 |
| **25–30M** | **1.0928** | 293 (AP292 confirmation) | 159M | 5.4 |
| 30–40M | 1.0977 | 183 (TPE) | 114M | 2.9 |
| 40–52M | 1.0950 | 68 (AutoResearch) | 89M | 1.8 |
| 52–70M | 1.1142 | 276 (NSGA-II) | 80M | 1.4 |
| 70–90M | 1.1405 | 245 (random search) | 64M | 0.9 |

- **The size AP292 found is close to compute-optimal at `C0` (INFERRED).** The minimum lies at roughly 20–50M parameters, about where AP292 is.
- The estimate is noisy: each cell is one seed with recipes mixed, and taking the best value in each range favours lucky runs.

**Larger models in the 300-second hyperparameter searches (OBSERVED, `RDB`; parsed in `SCR/classical_runs.json`).**
- There are 118 such runs with 13.2M–86.5M parameters, all at device batch 64.
- These rows supply the throughput and memory measurements used in part C.
- In each, a larger model got fewer tokens.

**Superseded `PAv2` fixed-token arm (OBSERVED, `PAv2/pair-a-fixed-tokens-seed3/history.jsonl`).**
- All runs used 88,604,672 tokens, a 2¹⁹ batch and seed 3, with progress measured in tokens (the patch adds `progress = min(total_tokens / N_STAR, 1.0)`).

| Model | Parameters | Device batch / accumulation | Training time | Throughput | Val BPB |
|---|---:|---|---:|---:|---:|
| d10 × 640 | 85.85M | 32 / 8 | 527 s | 168,258 tokens/s | 1.1631 |
| d12 × 768 | 135.27M | 16 / 16 | 811 s | 109,259 tokens/s | 1.1643 |

- **Comparison (OBSERVED).**
  - The namespace's registered d8 baseline scored 1.1937 at the same token count (`PAv2/pair-a-fixed-tokens-seed3/campaign_view.json` → `baseline.protected_validation_bpb`).
  - The d10 model with a 2¹⁸ batch scored 1.1243, and with a 2¹⁷ batch 1.1402.
- **Interpretation (INFERRED, single seed).** At a fixed 88.6M tokens, growing the model from 50M to 86M parameters helped by about 0.03. Growing it further to 135M gave nothing, which suggests the run was data-limited. The best batch size also varied with model size.

### B5. Pair-A fixed-time edits that shrank the model

**Fixed-time arm, seed 3 (OBSERVED, `PA/pair-a-fixed-time-seed3/history.jsonl`).** The 22 attempts are in `result.{num_params,total_tokens,validation_bpb}`.

| Quantity | Value |
|---|---|
| Registered baseline (`PA/provenance/baseline_registration_new_a40.json` → `protected_validation_bpb`) | 1.19989 at 85,458,944 tokens |
| Best attempt (attempt 14) | 1.1016 at 24.6M parameters and 126.6M tokens |
| Frontier members at the end of the run (`PA/pair-a-fixed-time-seed3/campaign_view.json` → `frontier`) | attempts 14, 15, 17, 18, 21, 22; 17.2–24.6M parameters; 126.6–167.0M tokens |

**Separating the token effect from the per-token effect (DERIVED, `SCR/pair_a_decompose.py`).**
- **Method.** Subtract from each child-minus-parent change the part explained by tokens: s·ln(D_child/D_parent). The slope s is −0.062 to −0.082 BPB per e-fold, taken from the 2¹⁷-batch recipes in B1.

| Edit (attempt) | Δ val BPB | ln of token ratio | Per-token residual (s = −0.07) |
|---|---:|---:|---:|
| Batch 2¹⁹ → 2¹⁸ (2) | −0.0240 | −0.006 | **−0.0244** |
| Depth 8 → 7 (3) | −0.0056 | +0.114 | +0.0024 |
| MLP 4× → 3× (4) | −0.0051 | +0.093 | +0.0014 |
| Short window = sequence/4 (10) | −0.0067 | +0.043 | −0.0036 |
| Batch 2¹⁸ → 2¹⁷ (12) | −0.0621 | −0.009 | **−0.0628** |
| Depth 7 → 6 (13) | +0.0071 | +0.452 | **+0.0387** |
| Width ratio 64 → 80 (14) | −0.0109 | −0.321 | −0.0334 (more capacity helps per token) |
| Head dimension 128 → 64 (16) | +0.0157 | +0.055 | +0.0196 |
| MLP 3× → 2× (17) | +0.0037 | +0.101 | +0.0107 |
| Depth 6 → 5 (18) | +0.0036 | +0.141 | +0.0135 |
| Head dimension 128 → 96 (22) | +0.0106 | +0.017 | +0.0118 |

**Reading the table (INFERRED).**
- Every edit that shrank depth, MLP or head size was neutral or harmful per token (+0.001 to +0.044 across the s range). Those edits were selected under the 300-second contract because they bought more tokens.
- The two batch edits are the only large per-token gains.
- Each edit here is a single seed. At 300 s the seed SD is 0.002–0.008 (B1), so effects under about 0.01 are noise.

**Fixed-token arm, `PA/pair-a-fixed-tokens-seed3/history.jsonl` (OBSERVED unless noted).**
- **Budget.** All 20 completed attempts ran exactly 85,458,944 tokens at batch 524,288 (163 updates each). `N_STAR` is set at `PA/provenance/baseline_registration_new_a40.json` → `contract_parameters.n_star`.
- **The agent never changed `TOTAL_BATCH_SIZE`.**
- **Quality.** The best was 1.2030 (attempt 11), against a baseline of 1.1999.
- **Schedule.**
  - The protocol and the agent's program both require token-normalized progress (`study/contracts_2026/protocol.md` L103; `programs/program_fixed_tokens.md` L10–14).
  - Only attempt 2 implemented it, and that attempt was dominated, at 1.2160 against its parent's 1.2047.
  - Every retained descendant kept `progress = total_training_time / TIME_BUDGET`. The call to `contract_progress(...)` returns a value that is never used (`study/contracts_2026/study_runtime.py` L24–36).
- **Consequence (DERIVED).** Faster candidates stopped before their learning rate had finished decaying. Time progress at the stop was between 0.773 (attempt 21) and 0.987 (attempts 1–2; computed as `result.measured_training_seconds`/300). The quality of fixed-token candidates is therefore confounded by incomplete annealing.
- **A related observation (INFERRED).** The instructed protocol change was rejected on one seed with Δ = 0.011, roughly 1.4 × the 300-second seed SD. This shows how selecting on single-seed noise can discard a correct change, which matters for search pressure (part F).

### B6. Ablations: batch only, and everything except batch (OBSERVED, `RDB`, 300 s, seeds 0–2)

| Recipe (source, sha256) | `RDB` ids | Tokens | Val BPB, mean ± SD |
|---|---|---:|---:|
| Baseline (`BASE`) | 3–7 (5 seeds) | 92.7M | 1.2069 ± 0.0080 |
| Batch only: `BASE` with `TOTAL_BATCH_SIZE` 2¹⁷ (`results/candidates/ablation-batch-only-seeds012-20260725/0001-train.py`, `d388c661…`; a one-line diff) | 78–80 | 88.1M | **1.1040 ± 0.0019** |
| Everything except batch: `ARS` with 2¹⁹ (`…/ablation-nonbatch-only-seeds012-20260725/0001-train.py`, `0863cad9…`; a one-line diff) | 81–83 | 94.4M | 1.2188 ± 0.0164 |
| AutoResearch (`ARS`) | 68, 75, 77 | 88.6M | 1.1001 ± 0.0044 |

**Effects (DERIVED).**
- Batch only: −0.1029, which is 96% of AutoResearch's −0.1068.
- Everything except batch: +0.0119.
- Interaction: −0.0039.

**Context.** The first retained AutoPareto step was also the batch change (`RDB` id 113, "batching", 1.2137 → 1.1063, Δ −0.107). The later learning-rate refinements in the AP292 line each moved validation BPB by 0.0001–0.0020 (ids 131, 132, 134), which is within single-seed noise.

These ablations exist only at 300 s and about 90M tokens. Nothing tests them at a longer budget.

### B7. Which comparisons separate longer training from a larger model, and tokens from FLOPs (DERIVED)

| Comparison | Model size | Tokens | FLOPs | Schedule complete? | What it separates |
|---|---|---|---|---|---|
| 300 s against 201M, same recipe (B1) | Fixed | 0.94–2.27× | ∝ tokens | Yes, both | Longer training at a fixed size. Tokens and FLOPs move together. Confounded by different seeds and pods. |
| 201M panel, across recipes | 17.9–50.3M | Fixed | 1.40–4.81e16 | Yes | Exposure is matched, but size and FLOPs move together. Size cannot be separated from compute. |
| 300 s, across recipes | 17.9–50.3M | 88.6–215M | 2.06–2.22e16 (AP294 1.50e16) | Yes | Compute is roughly matched. Size and tokens trade off: the compute-optimal question at `C0`. |
| Milestones within a 201M run | Fixed | 50–201M | ∝ tokens | **No**, except at 100% | Not a shorter-run substitute (B3) |
| Equal-time size sweep (B4) | 11.5–70.8M | 64–215M | ≈ `C0` | Yes | Compute-optimal size at `C0` only |
| Pair-A fixed-time arm | 15–40M | 86.5–175M | Roughly fixed | Yes | Edit effects at fixed time, with the token effect mixed in (B5) |
| Pair-A fixed-token arm | 33.6–47.2M | Fixed 85.46M | Varies | **No** for faster candidates | Size at fixed tokens, confounded by the schedule |
| `PAv2` fixed-token runs | 50 / 86 / 135M | Fixed 88.6M | 2.1 / 3.8 / 6.1e16 | Yes | A larger model at fixed tokens; one seed |
| Batch ablations | Fixed | ≈ fixed | ≈ fixed | Yes | The batch effect alone, at `C0` only |

**Bottom line (DERIVED).**
- No measured comparison matches FLOPs above `C0`.
- The only roughly FLOP-matched pairs are the 300-second ones. For example, AP292 at 2.07e16 is better than AutoResearch at 2.06e16 by 0.0054 validation BPB.
- At 201M tokens, AutoResearch is better by 0.0451 (paired seeds: 0.044, 0.047, 0.045) but uses 1.76× the FLOPs.

**What a log-linear extrapolation suggests (INFERRED).** Projecting AP292 to AutoResearch's 201M compute gives 1.0777 − 0.068 × ln(4.687/2.660) = 1.0392. That would leave AutoResearch about 0.0065 better at equal compute, a sign flip from the 300-second result. The pilot's FLOP-matched arm tests exactly this.

The reviewer matrix already says "no matched-FLOP claim" (`paper/reviewer_attack_matrix.md` L11) and asks that fixed tokens be called "exposure matched" (L63).

---

## C. Compute and feasibility, from measured throughput

### C1. Throughput and FLOP accounting

**Steady-state throughput at 201M tokens** (`P1/runs/*/training_metrics.json` → `steady_state_training_tokens_per_second`; means are also in `docs/PHASE1_200M_RESULTS.md` L85–91) (OBSERVED).

| Recipe | Tokens/s |
|---|---:|
| Baseline | 292,414 |
| AutoResearch | 289,195 |
| TPE | 372,464 |
| MOTPE | 505,550 |
| AP292 | 522,900 |
| AP299 | 643,659 |
| AP294 | 681,101 |

**Achieved code-FLOPs per second (DERIVED, FLOPs per token × tokens/s).**
- Six recipes reach 6.73–7.04e13. Example: AP292 at 1.321221e8 × 522,900 = 6.91e13.
- AP294 reaches only 4.74e13, because of its narrow 384-wide, 3-head shape.
- Across the 118 hyperparameter-search rows, the median code-FLOPs per second ranges from 55 to 80 TFLOP/s, depending on shape (d9 × 768 reaches 80).
- The `PAv2` runs with gradient accumulation reached 72.1 TFLOP/s (d10, device batch 32) and 75.3 TFLOP/s (d12, device batch 16).

**Planning assumptions.**
- Throughput: 70 TFLOP/s, with a range of 60–80.
- `C0` = 300 s × 70 TFLOP/s ≈ 2.1e16. The six 300-second runs measured 2.06–2.22e16 (B1).
- Per-run overhead: 300 s (DERIVED). Compilation took 91–157 s (`step_timing.jsonl`, step 0), and evaluation and saving took 75–112 s (`end_to_end_runtime_seconds` − `total_scientific_step_wall_seconds`).

**Two FLOP conventions are in use.**
- `estimate_flops` (`BASE` L208–222) leaves out `wte` and the value embeddings, and adds attention.
- Pair-A records 6·N_total·D (`study/contracts_2026/study_runtime.py` L49–50; `result.training_flops_estimate`). That includes the cost-free value embeddings, so it overstates AP292's FLOPs per token by 1.33× (6 × 29.36M = 1.76e8 against 1.32e8).
- Any FLOP-matched design should preregister `estimate_flops`.

**Hardware MFU.** The printed MFU divides by the H100 peak (`BASE` L463, L587), so it is meaningless on an A40. True hardware MFU is lower than the code-FLOP ratio, because the attention term counts the whole window and ignores causal masking (INFERRED).

**Pricing.** $0.44/hour for an A40 (`PROJECT_STATE.md` L960). This excludes idle and storage time.

### C2. (i) 4×, 16× and 64× the tokens at a fixed model size

**Base and method.** The token budget is a multiple of 201,326,592. Time = D/(tokens/s) + 300 s (DERIVED, `SCR/ladder.py`).

| Recipe | 4× (0.805B tokens) | 16× (3.22B) | 64× (12.89B) |
|---|---|---|---|
| Baseline | 0.85 h, $0.37 | 3.14 h, $1.38 | 12.32 h, $5.42 |
| AutoResearch | 0.86 h, $0.38 | 3.18 h, $1.40 | 12.46 h, $5.48 |
| TPE | 0.68 h, $0.30 | 2.49 h, $1.09 | 9.69 h, $4.26 |
| MOTPE | 0.53 h, $0.23 | 1.85 h, $0.82 | 7.16 h, $3.15 |
| AP292 | 0.51 h, $0.22 | 1.79 h, $0.79 | 6.93 h, $3.05 |
| AP299 | 0.43 h, $0.19 | 1.47 h, $0.65 | 5.64 h, $2.48 |
| AP294 | 0.41 h, $0.18 | 1.40 h, $0.61 | 5.34 h, $2.35 |
| **All 7 × 3 seeds** | **12.8 GPU-h, $6** | **46.0 GPU-h, $20** | **178.6 GPU-h, $79** |

**Example (baseline at 64×).** 12,884,901,888 / 292,414 = 44,063 s, or 12.24 h; plus 300 s gives 12.32 h; × $0.44 = $5.42.

**Memory and steps.** At a fixed size, peak memory equals the 201M-token values (`training_metrics.json` → `peak_training_cuda_memory_mb`: 8,006–22,805 MiB), which is under the 46,068 MiB A40 limit. The number of optimizer steps at 64× is 24,576 for the baseline and 98,304 for the 2¹⁷-batch recipes (DERIVED).

### C3. (ii) Larger models at roughly compute-optimal budgets: 4×, 16× and 64× `C0`

**Method (DERIVED, `SCR/ladder.py`).**
- Promote with the depth dial: keep `ASPECT_RATIO` and every other constant, and choose the `DEPTH` whose D/N_flop is closest to 20, where D = C/(FLOPs per token).
- The ratio of 20 is an external Chinchilla-style heuristic, and the parameter convention alone moves the answer by up to about 2× (A2), so treat the sizes as approximately compute-optimal.

**Rule check at 1× `C0`.** The rule picks d3 × 384 for the ratio-128 family, which is **AP294's exact shape**, and d3 × 512 for the ratio-160 family, which is **AP299's exact shape**. For the baseline and AutoResearch it picks d5 × 384, about half the registered baseline's size. The 50M baseline is roughly compute-optimal only at about 4–7× `C0`: the rule picks d7 × 512 at 4×.

**The three rungs.**

| Family (non-size constants as in A2) | 4× `C0` = 8.4e16 | 16× `C0` = 3.36e17 | 64× `C0` = 1.34e18 |
|---|---|---|---|
| Baseline (ratio 64, SSSL, batch 2¹⁹) | d7 × 512; 47.2M parameters (26.2M FLOP-relevant); 0.39B tokens; 749 steps | d9 × 640; 80.9M (49.5M); 0.86B | d12 × 768; 135.3M (91.2M); 1.95B; 3,721 steps; **device batch 32** |
| AutoResearch (ratio 64, SSSS, batch 2¹⁷) | d7 × 512; 47.2M; 0.40B; 3,087 steps | d9 × 640; 80.9M; 0.89B | d12 × 768; 135.3M; 2.01B; 15,303 steps; **device batch 32** |
| TPE (ratio 80) | d6 × 512; 39.8M; 0.46B | d8 × 640; 70.8M; 0.99B | d10 × 896; 147.7M; 1.81B; **device batch 32** |
| MOTPE (ratio 48, L) | d8 × 384; 33.0M; 0.47B | d10 × 512; 60.8M; 0.99B | d14 × 768; 155.7M; 1.50B; **device batch 32** |
| AP292 / AP294 (ratio 128) | d5 × 640; 50.8M (29.8M); 0.36B | d6 × 768; 73.9M (48.8M); 0.91B | d8 × 1024; 151.0M (109.1M); 1.72B; **device batch 32** |
| AP299 (ratio 160) | d4 × 640; 40.6M; 0.45B | d5 × 896; 84.9M; 0.82B | d6 × 1024; 117.4M; 2.23B; device batch 64 |
| AP292 width dial (depth fixed at 4) | ratio 160, width 640; 40.6M; 0.45B | ratio 224, width 896; 67.9M; 1.02B | ratio 352, width 1408; 141.3M; 1.85B |

**Cost.**
- Each run takes 0.33 h, 1.33 h or 5.33 h at 70 TFLOP/s (ranges 0.29–0.39, 1.17–1.56 and 4.67–6.22 h), which is $0.15, $0.59 or $2.35.
- Six families × 3 seeds × three rungs = **130 GPU-h, $57**.

**Rounding.** Rounding shifts the effective width/depth ratio of the ratio-64 family to 71–73 at d7 and d9. Using even depths avoids this.

**Extrapolation risk.** Throughput beyond 135M parameters has not been measured (UNKNOWN).

### C4. Memory on the A40 (46,068 MiB; `study/contracts_2026/provenance/a40_environment.json` → `runtime.gpus[0].memory_total_mib`)

**Fit (DERIVED).**
- Model: peak training MiB ≈ 2,194 + 5.0145·d·w at device batch 64.
- Data: 25 runs with full multi-head attention and head dimension 128 (7 from the 201M panel and 18 from the hyperparameter searches), with d·w from 1,152 to 6,400. RMS residual 87 MiB, maximum residual 318 MiB.
- Checks:

| Run | Observed | Predicted |
|---|---:|---:|
| AP292 | 12,473 | 12,464 |
| Baseline | 22,805 | 22,733 |
| d10 × 640 | 34,253 | 34,287 |

**Planning model (INFERRED).** Peak ≈ 16 bytes × parameters + (device batch/64) × (2,194 + 5.0145·d·w). The headroom target is ≤ 40,000 MiB.
- At device batch 64, anything with d·w above about 7,500 needs device batch 32. That covers d12 × 768 (9,216), d8 × 1024 (8,192) and d14 × 768 (10,752). The predicted peaks at device batch 32 are 23–30 GiB.
- Above d·w = 6,400 this is an **extrapolation**. The `PAv2` d12 × 768 run at device batch 16 completed, which is consistent with the model.

**Gradient accumulation** is built in (A1). The token stream does not depend on device batch: `make_dataloader` fills rows one after another from the same document buffer (`prepare.py` L309–341) (INFERRED). A runtime check exists only for 4×64 against 1×64 microbatches (`P1/token_stream_verification.json` → `runtime_verification.both_accumulation_regimes_equal`). Re-verify with `scripts/verify_phase1_token_stream.py` for any new device batch.

**Evaluation coupling.** `evaluate_bpb` uses `DEVICE_BATCH_SIZE` (`BASE` L613), and its step count is `EVAL_TOKENS // (B·2048)` (`prepare.py` L358). The device batch must therefore divide 10,240, or the evaluation token count changes. 16, 32 and 64 are all safe.

### C5. Data availability (ClimbMix-400B-shuffle)

**Local state (OBSERVED).**
- No dataset shard or tokenizer file is stored locally. There is no `*.parquet` or `tokenizer.pkl` under the repo, and `~/.cache/autoresearch` does not exist.
- `data/` holds only CSV exports.

**The pod cache (OBSERVED).**
- It held **10 training shards plus 1 validation shard**, 917,173,911 parquet bytes for training (`study/contracts_2026/provenance/data_manifest.json` → `files`; `P1/token_stream_verification.json` → `runtime_verification.training_files`).
- Default download: 10 shards (`prepare.py` L377). The full dataset has 6,543 shards (L42–43).

**Tokens per shard: UNKNOWN.**
- **Lower bound (OBSERVED):** no logged 300-second run ever wrapped to epoch 2. The largest is 286.9M tokens in `results/logs/autopareto-seed1-20260731-0013.log` (DERIVED over 282 logs). The 10 shards therefore hold at least 287M tokens.
- **Estimate (INFERRED):** 55–80M tokens per shard, from about 4.0 bytes per token (AP292's final smoothed training loss 3.011 / (validation BPB 1.0766 × ln 2) = 4.04) and 2.5–3.5× parquet compression.

**Shards needed (INFERRED).**

| Run | Tokens | Shards |
|---|---:|---:|
| 4× tokens | 0.81B | 10–15 (probably more than the current cache) |
| 16× tokens | 3.2B | 40–60 |
| 64× tokens | 12.9B | 160–235 (about 15–22 GB) |
| 64× `C0` | 1.5–2.2B | 19–40 |

**Hazards (OBSERVED).**
1. **The loader repeats data silently.** `_document_batches` increments `epoch` and starts over (`prepare.py` L265–273), and only the log shows it. A guard must fail any run that reaches `epoch > 1`.
2. **Downloads are not pinned to a revision.** `prepare.py` downloads from `/resolve/main` (L41), whereas the held-out shard is pinned to revision `915333b4…` (`evaluation/downstream_suite.json` → `heldout_bpb.dataset_revision`). New shards should be pinned and their hashes recorded.
3. **The cache path is shared.** `CACHE_DIR` is hard-coded to `~/.cache/autoresearch` (L38), so a scaling run should use a separate home directory or container. That keeps the frozen 10-shard cache (aggregate hash `1211827…`) unchanged.
4. **The held-out shard must stay out of training.** Shard 06541 stays out as long as fewer than 6,542 shards are downloaded (L94–97). The downstream evaluator caches it separately (`downstream_evaluator.py` L478–500).

**Useful property (INFERRED).** Shards are read in sorted order (`prepare.py` L121), so a longer run's first 201,326,592 tokens are byte-identical to the Phase-1 stream. Verify against `first_microbatch_sha256` `f0fcc459…`. The 201M-token runs are therefore a valid 1× rung for paired comparisons on seeds 3–5.

### C6. Tokenizer constraints

**Facts (OBSERVED).**
- Vocabulary 8,192, with 4 reserved special tokens; the beginning-of-sequence token is `<|reserved_0|>` (`prepare.py` L45, L50–51).
- The tokenizer was trained on at most 10⁹ characters of the first training shards (L125–138, L163).
- Hashes: `tokenizer.pkl` `888c9fa8…`, `token_bytes.pt` `23517e05…` (`study/contracts_2026/provenance/tokenizer_manifest.json` → `tokenizer`).
- Neither file is stored locally.

**Implications (INFERRED).**
- Comparisons across rungs need the identical tokenizer, so restore it from a pod volume or retrain it and confirm the hash. Whether retraining reproduces the hash is UNKNOWN.
- A vocabulary of 8,192 is small for models of 100M parameters and up. Its unembedding share of FLOPs falls from 19% (AP292) to 6% (AP292 promoted to d8 × 1024) (DERIVED).
- Keep the vocabulary fixed. Changing it would confound the transfer measurement.

### C7. Whether the evaluators work at larger sizes

**Validation BPB.** Computed inside the training script (A1, C4). It scales trivially.

**Held-out BPB** (`downstream_evaluator.py` L503–587).
- Protocol: 2,560 sequential segments of 2,049 tokens from shard 06541, scored in batches of 8 (`evaluation/downstream_suite.json` → `runtime`).
- Unlike validation, which packs whole documents into rows that start at a document boundary, these segments cut documents at arbitrary points. This likely explains why held-out BPB sits 0.053–0.069 above validation BPB, with an offset that differs by recipe (DERIVED from `P1` means; explanation INFERRED).
- The two metrics can rank recipes differently. MOTPE minus baseline is −0.0142 on validation but +0.0017 on held-out (paired means, DERIVED). Preregister one primary metric.

**Downstream suite.**
- Took 147–252 s per checkpoint at 17.9–50.3M parameters (DERIVED from `P1/downstream_200m.db`, `downstream_evaluation_runs` rows 14–34, `finished_at − started_at`).
- At about 150M parameters, expect 10–20 minutes (INFERRED, scaling with FLOPs per token).
- ARC-Easy and PIQA sit near chance at these sizes (`docs/PHASE1_200M_RESULTS.md` L127–135), so BPB has to be the primary endpoint below about 100M parameters (INFERRED).

**Cached decode and inference memory.**
- The canonical adapter (`protected/inference_adapter.py`) recomputes masked attention in PyTorch, not with FlashAttention.
- Inference memory is about 2 bytes per parameter plus the KV cache, for example about 38 MiB of KV cache for d8 × 1024 at 1,152 tokens (DERIVED). That is negligible on the A40.
- Batch-1 decode speed falls with depth, so the shallow-wide family is likely to keep its decode advantage after promotion (INFERRED).
- **Risk.** The bf16 check that cached and uncached logits agree has a mean tolerance of 0.03, with an fp32 check as backup (`benchmark_config.py` L21–23). Errors grow with depth, and one historical candidate already failed at 0.0341 (`PROJECT_STATE.md` L711, L851).
- **Provenance.** `protected/inference_adapter.py` has uncommitted changes: checkpoint `format_version` 1→2, plus a value-embedding resolver (`git diff`). The Phase-1 checkpoints use format 1. The version used for any new run must be pinned.

**Checkpoint size (DERIVED).** About 3.1 bytes per parameter (matrices in fp32, embeddings in bf16). AP292's 29.4M parameters give 92.3 MB; a 151M-parameter promoted model gives about 0.52 GB, or about 2.6 GB per run with four milestones.

---

## D. Which recipes can be promoted at each level, and what blocks them

### Promotion levels

| Level | What changes | Recipes ready | Status |
|---|---|---|---|
| **L0: token ladder** (same model, more tokens) | Only `PHASE1_TOKEN_BUDGET` | All 7, plus both ablation sources (B6) | Promotable now through `protected/phase1_source_transform.py` (any budget divisible by `TOTAL_BATCH_SIZE`, L62–67). Blocked by B1–B7 below. |
| **L1: size promotion by depth dial** | `DEPTH`, plus `DEVICE_BATCH_SIZE` when memory requires | All 7; AP292 and AP294 share one family | Mechanical, with the ambiguity flags in A4 |
| **L1b: size promotion by width dial** | `ASPECT_RATIO` at fixed depth | The AutoPareto recipes, as a sensitivity arm | Mechanical (C3 last row) |
| **L2: promotion of single edits** (parent and child) | Any recorded edit | AutoPareto and AutoResearch: 130 edits, each with `code_diff` and parent (`RDB`); Pair-A: 42 completed edits | Sources can be recovered, but the absolute edits need rewriting (A4) |
| **L3: agent search at larger budgets** | New campaigns | — | Needs a new contract and calibration. Expensive in API cost. |

**Recoverability of Pair-A sources (DERIVED, `SCR/recon_pair_a.py`).** Applying the recorded `candidate_patch` diffs in order to `study/contracts_2026/historical_reference/baseline_source/train.py` (sha256 `b2fb8788…`) reproduces all 42 completed Pair-A candidate sources byte for byte (22 + 20 matches on `candidate_source_after_sha256`). No Pair-A source is lost.

### Exact blockers

| # | Blocker | Evidence | Affects | Fix |
|---|---|---|---|---|
| B1 | The data cache may hold fewer tokens than any rung ≥4× needs. Data repeats silently. Downloads are unpinned. | C5 | L0–L2 | Download pinned shards into a separate cache, record a hash manifest, add an `epoch == 1` check, and verify the 201M prefix hash |
| B2 | Tokenizer files are not stored locally | C6 | All | Restore, then verify `888c9fa8…` and `23517e05…` |
| B3 | The staged executor is tied to the 201M plan: status must be `PREPARING / PREFLIGHT`, and the model order and seeds are fixed | `scripts/run_phase1_equal_tokens.py` L34–56 | L0–L1 | A new plan file or a generalized executor. Do not edit the frozen plan. |
| B4 | The Phase-1 runtime hard-codes the 201M budget for `steady_state_equivalent_seconds` (L27), hard-codes a device batch of 64 in the milestone manifest (L560), and raises an error for `TOTAL_BATCH_SIZE` ≥ 2²⁰, because 11 × 524,288 is not divisible by it (L375–377) | `protected/phase1_runtime.py` | L0 metadata; L1 when batch scales | Parameterize these values in new files |
| B5 | The Pair-A runtime allows `TOTAL_BATCH_SIZE` ≤ 2¹⁹ only, fixes the time contract at 300 s, and requires the fixed-token budget to be a multiple of 524,288 | `study/contracts_2026/study_runtime.py` L6; `batching.py` L14–16, L65–70; `train_launcher.py` L29; `contracts.py` L66 | L2 (Pair-A path), L3 | Use the Phase-1 path, or add a new contract |
| B6 | The Phase-1 transform rejects Pair-A sources, because their import line does not include `evaluate_bpb` (validation runs out of process) | `phase1_source_transform.py` L69–73; `historical_reference/baseline_source/train.py` L26 | L2 | An adapter from Pair-A sources to Phase-1 (an execution-only shim) |
| B7 | Missing checkpoints. 10 of 21 final 201M checkpoints are stored locally and verified. AutoResearch seed 4 is truncated (3,014,656 bytes; hash mismatch). 10 are missing (baseline ×3, TPE ×3, MOTPE ×3, AutoResearch seed 5). All 63 milestone checkpoints exist only on the RunPod volume, whose current state is UNKNOWN. | `results/snapshots/…/scientific_final_checkpoints/` hashed against `P1/runs/*/training_metrics.json` → `checkpoint_sha256`; `inventory.json` | Re-evaluating the 201M panel, for example on a new sealed shard. Not needed for promotion from scratch. | Restore from the volume if it still exists |
| B8 | Continuing training from the 201M finals is **not** a valid route to 4× runs. The learning rate has already decayed to its final fraction and Muon weight decay to 0. | A1 schedule | L0 | Retrain from scratch with the stretched schedule |
| B9 | Implementation drift. The local `protected/phase1_train_launcher.py` (`1808f371…`) and `scripts/evaluate_phase1_checkpoints.py` (`e2de6aae…`) differ from the hashes recorded at preflight (`6b39dbcc…`, `2c2d0e8d…`, in `evaluation/phase1_equal_token_preflight.json` → `implementation_sha256`). Which version actually ran is UNKNOWN. The other 5 implementation files and `prepare.py` (`7132ec46…`) match. | Hashes computed in this audit | L0 comparability with the 201M rung | Pin the versions, and run one reproduction check (part E) |
| B10 | Promotion choices: depth dial or width dial; batch kept literally or scaled; rounding; the 300-step momentum ramp; absolute indices in Pair-A edits | A4 | L1, L2 | Preregister an operator with ambiguity flags (part F) |
| B11 | Throughput is unmeasured above 135M parameters, and memory is extrapolated above d·w = 6,400 | C1, C4 | Top rung of L1 | A one-step memory and throughput probe per new shape. This is non-scientific and must be labelled as such. |
| B12 | Two metrics with different rankings: validation (packed documents) and held-out (stream) | C7 | All | Preregister the primary metric |

---

## E. A minimal camera-ready pilot (run only under the conditions below)

**Conditions that protect the core reviewer-response experiments**
- **Hardware.** Use a separate pod, or wait until all core GPU work has finished. Never share a GPU with any run whose timing is measured.
- **Storage.** Use a separate `HOME`/cache, because of `prepare.py` L38. Write results only to a new namespace, for example `results/scale_pilot_2026/`.
- **Frozen files.** Do not touch `results/results.db`, the Phase-1 database, any frozen source, or any `protected/*.py` file. Put any generalization in new files.
- **Seeds 3–5.** These were already used for verification. They pair the pilot with the existing 201M runs, and they leave the fresh seeds that `paper/NEXT_EXPERIMENT_PLAN.md` L18 reserves for the replicated contract study unused.
- **Limits.** A hard cap of **22 GPU-hours**, and a calendar cutoff of **2026-11-06**, one week before the 2026-11-13 deadline. If the pilot misses either, nothing from it goes in the camera-ready.
- **Preregistration.** Commit a preregistration file before the first run. It must list the source hashes, budgets, seeds, data manifest, primary metric and decision rules.
- **Reporting.** Report the pilot as exploratory, in Part 3 or an appendix. The headline claims stay unchanged.

**Coordination with `08_EXPERIMENT_PLAN.md` (OBSERVED there)**
- **Ordering.** Its E6 slot points to this section, and runs only after E0, E1 and E4 succeed (08 L199–201). This pilot follows the same ordering.
- **Seeds.** E1 runs AP-Q (AP292) against AR-ref at ≤ 201M tokens on fresh seeds 6–8 (08 L86, L91). This pilot stays on seeds 3–5, so that every 4× cell pairs, seed for seed, with the existing 201M panel. E1's seeds 6–8 then give an independent replicate at 1×. Analyse the two seed sets as separate blocks, and do not pool them.
- **Code-path check.** If E0's 201M reproduction cell (08 L57) uses the same executor, skip this pilot's code-path check run.
- **Runner.** 08 L99 assumes a fixed-token runner whose budget can be configured. Blockers B3 and B4 (part D) show what that requires.

**Design.** All sources are frozen. The only change is the token budget, applied by `protected/phase1_source_transform.py`.
- **Tier 1a, the core.** It extends the AP292-against-AutoResearch comparison above 201M tokens.
- **Tier 1b, an add-on.** It tests whether the batch-size discovery survives longer training.

| Tier | Arm | Source (sha256) | Budget(s) | Steps | Runs (seeds 3–5) |
|---|---|---|---|---|---:|
| 1a | AutoResearch | `ARS` (`b3a961ee…`) | 4× = 805,306,368 tokens | 6,144 | 3 |
| 1a | AP292, equal tokens | `AP292S` (`55a7fd8b…`) | 4× | 6,144 | 3 |
| 1a | AP292, equal FLOPs | `AP292S` | **7× = 1,409,286,144 tokens** (milestones at multiples of 352,321,536; 2,688 steps each) | 10,752 | 3 |
| 1b | Baseline | `BASE` (`428e22f1…`) | 4× | 1,536 | 3 |
| 1b | Batch only | `…/ablation-batch-only-seeds012-20260725/0001-train.py` (`d388c661…`) | 1× = 201,326,592 and 4× | 1,536 and 6,144 | 6 |
| 1b | Everything except batch | `…/ablation-nonbatch-only-seeds012-20260725/0001-train.py` (`0863cad9…`) | 1× and 4× | 384 and 1,536 | 6 |
| — | Code-path check (skip if E0 covered it) | `BASE` | 1× | 384 | 1 (seed 3) |

**Why 7× matches AutoResearch's compute (DERIVED).** AP292 at 7×: 1.321221e8 × 1,409,286,144 = 1.862e17 code-FLOPs. AutoResearch at 4×: 2.327869e8 × 805,306,368 = 1.875e17, so the gap is −0.7%. The baseline at 4× uses 1.925e17. All of these are about 9× `C0`.

**Reusing the existing 201M runs.** The 201M runs of baseline, AutoResearch and AP292 serve as the 1× rung. That reuse holds only if the code-path check reproduces `P1/runs/phase1-baseline-seed3` (validation 1.089689) within 0.005 BPB. Otherwise, re-run the three 1× cells (+9 runs, about 2.6 GPU-h).

**Cost (DERIVED).**

| Arm | Seconds per run |
|---|---:|
| Baseline, 4× | 2,754 |
| Batch only, 4× (288,690 tokens/s, `RDB` ids 78–80) | 2,790 |
| Batch only, 1× | 697 |
| Everything except batch, 4× (293,860 tokens/s, ids 81–83) | 2,740 |
| Everything except batch, 1× | 685 |
| AutoResearch, 4× | 2,785 |
| AP292, 4× | 1,540 |
| AP292, 7× | 2,695 |

- **Tier 1a.** Per seed: (2,785 + 1,540 + 2,695) s + 3 × 300 s = 7,920 s. Across 3 seeds that is 6.6 h; downstream evaluation of 9 checkpoints at about 250 s each adds 0.6 h. Total **about 7.2 GPU-h, about $3.20**, for 9 runs.
- **Tier 1b.** Per seed: (2,754 + 697 + 2,790 + 685 + 2,740) s + 5 × 300 s = 11,166 s. Across 3 seeds that is 9.3 h; evaluating 15 checkpoints adds 1.0 h. Total **about 10.4 GPU-h, about $4.60**, for 15 runs.
- **Code-path check:** 0.3 GPU-h.
- **All tiers:** about **17.9 GPU-h, about $7.90**, for 25 runs. The 22 GPU-h cap leaves about 20% contingency.
- **If time runs short,** run Tier 1a alone.
- **Data (INFERRED).** About 1.41B tokens, roughly 18–26 shards. Download pinned shards 10–29, about 1.8 GB. Tier 1b alone needs only the 4× budget, about 10–15 shards.

**Preregistered analysis**
- **Primary metric:** held-out BPB, using the same evaluator as the 201M panel (sha256 `696f9c46…`, `docs/DOWNSTREAM_EVALUATION.md` L42).
- **Secondary metrics:** validation BPB and the downstream suite.
- **Contrasts.** All are differences paired by seed, reported with every seed shown.
  1. **Batch main effect at 1× and 4×** in the 2×2 design. Does it shrink?
  2. **AutoResearch minus baseline at 1× and 4×.** At 1× it is −0.0468 ± 0.0044 held-out and −0.0608 ± 0.0035 validation.
  3. **AP292 minus AutoResearch at equal tokens, at 1× and 4×.** At 1× it is +0.0420 ± 0.0050 held-out.
  4. **AP292 at 7× minus AutoResearch at 4× (equal FLOPs).** Compare this with the 300-second roughly-equal-compute contrast (AP292 better by 0.0054 validation BPB).
  5. **Each recipe's slope in BPB per e-fold, from 1× to 4×,** against its 300-second-to-1× slope (B1).
- **Decision rules.**
  - "Persists": same sign on all 3 seeds and |mean| > 0.01.
  - "Vanished": |mean| < 0.005 and the paired interval includes 0.
  - Anything else: "inconclusive".
- **Resolution.** Seed SDs of the paired differences are 0.0014–0.0056 (DERIVED). With n = 3, the t-interval half-width is about 0.006–0.014, so a change of about 0.02 BPB in a contrast between rungs is the smallest reliably detectable effect (INFERRED).

**What the pilot can establish**
- Whether the dominant discovered edit (the smaller batch) and the non-batch edits keep, lose or reverse their advantage after 4× more training at the same size.
- Whether AP292's quality deficit against AutoResearch at equal tokens grows or shrinks.
- The first FLOP-matched comparison of AP292 and AutoResearch above `C0` (about 9× `C0`). This answers "no matched-FLOP claim" in the reviewer matrix.

**What it cannot establish**
- Anything about larger models, since parameters are not scaled.
- Whether other edits or recipes transfer, or how agents behave at larger budgets.
- Whether results generalize across hardware, datasets or tokenizers.
- Effects smaller than about 0.02 BPB.
- The mechanism behind any effect.
- Downstream transfer, since these sizes score near chance.

**Optional Tier 2.** Not recommended for the camera-ready; it belongs to the follow-up.
- Runs: one rung at 16× `C0`, promoted with the depth dial: baseline and AutoResearch at d10 × 640, and AP292 at d6 × 768, with 2 seeds.
- Cost: about 8 GPU-h.
- Fit: all three fit at device batch 64. d10 × 640 measured 34,253 MiB (`RDB` id 167); d6 × 768 is predicted at about 26 GiB.
- Caveat: it brings in the depth-dial/width-dial ambiguity (A4), so it is not "minimal and defensible" under the camera-ready deadline.

---

## F. Long-term design: "Do AI-discovered optimizations survive scaling?"

### F1. The promotion operator P(source, rung k, mode)

The operator is preregistered, versioned code, frozen before any run at rung 1 or above.

1. **Parse.** Read constants via the AST, as `scripts/prepare_phase1_equal_tokens.py` L149–183 does; never from comments, because the batch comments in `TPES` and `MOTS` are stale. Diff the result against the registered parent.
2. **Classify.** Put each diff into one class: size (depth, width ratio, MLP ratio, head dimension, KV heads), batch, learning rate, optimizer algorithm (Newton–Schulz steps, betas, weight decay), schedule, attention pattern, structure (value-embedding layout, factorization), or numerics (dtype). The classifier is frozen; `paper/NEXT_EXPERIMENT_PLAN.md` L28 already asks for a frozen diff classifier.
3. **Size rule.** Use one of three modes:
   - `depth`: keep the width ratio, head dimension and KV ratio, and choose `DEPTH` = argmin over d of |ln(D_k(d)/N_flop(d)/20)|, with D_k = C_k/(FLOPs per token).
   - `width`: keep `DEPTH` and step the width ratio up to reach the same N_flop.
   - `fixed`: keep the size; this is the token ladder.
   The AutoPareto recipes are run under both `depth` and `width`, and the difference between the two is reported as a finding.
4. **Budget.** D_k is rounded down to a multiple of 4·`TOTAL_BATCH_SIZE`, so that milestones land exactly. Training is from scratch with the token-normalized schedule (the Phase-1 transform).
5. **Batch.**
   - Primary mode, `literal`: keep the recipe's absolute batch, and the baseline keeps 2¹⁹.
   - Secondary mode, `sqrt`: batch ∝ (D_k/D_0)^½, rounded to a power of two ≤ 2²¹. This needs fix B4.
6. **Learning rates and optimizer.** Keep them literally. The code already applies the 1/√width scaling to AdamW. A μP-style variant is optional, as a sensitivity arm only.
7. **Device batch.** The largest of {64, 32, 16} whose predicted peak is ≤ 40,000 MiB, which divides `TOTAL_BATCH_SIZE`/2048, and which divides 10,240.
8. **Ambiguity flags.** Every promoted source carries these flags; flagged edits are analysed both with and without them.

| Flag | Meaning |
|---|---|
| A1 | Size is itself the edit |
| A2 | The effective width ratio drifts by more than 5% from rounding |
| A3 | Absolute batch |
| A4 | Step-count constant (the 300-step momentum ramp) |
| A5 | Absolute layer index or literal constant in a structural edit (Pair-A) |
| A6 | The full-attention share depends on depth |

9. **Output.** The promoted source, its hash, and a promotion manifest listing every diff with its class, rule and flags.

### F2. The scale ladder

**Main ladder.** C_k = 4^k·`C0`, with `C0` = 2.1e16 code-FLOPs, about 300 A40-seconds. The table gives the reference sizes at D/N_flop ≈ 20 (DERIVED); family-specific shapes are in C3.

| Rung | Compute | Time per run on an A40 at 70 TFLOP/s | N_flop | Tokens |
|---|---:|---:|---:|---:|
| 0 | 2.1e16 | 5 min | 13M | 0.27B |
| 1 | 8.4e16 | 20 min | 26M | 0.53B |
| 2 | 3.36e17 | 80 min | 53M | 1.06B |
| 3 | 1.34e18 | 5.3 h | 106M | 2.1B |
| 4 (optional) | 5.4e18 | about 21 h (UNKNOWN on a faster GPU) | 212M | 4.2B |

**Orthogonal arms**
- A token-only ladder at fixed size: the 7-recipe panel at 4× and 16× tokens (C2).
- A small isoFLOP check, 3 sizes × 2 families, at rungs 1 and 2. It verifies the ratio of 20 empirically, because the parameter convention is ambiguous (A2).

**Hardware.** One A40 class and one software stack, pinned as in `study/contracts_2026/provenance/a40_environment.json`. If rung 4 moves to other hardware, rung 3 must be re-run there as a calibration rung.

### F3. Proxy and verified metrics

**Proxy.** This is what the agent optimized:
- 300-second validation BPB on shard 06542, from a single seed;
- normalized decode throughput;
- inference memory.

**Verified, at each rung:**
1. Validation BPB at the rung's own budget, with complete schedules.
2. Held-out BPB on shard 06541, in the stream protocol.
3. A **sealed new shard**, evaluated once at the end of the study. A candidate is 06540, which no run has used locally (INFERRED from the shard lists in C5); this must be confirmed before sealing.
4. The downstream suite at rungs 2 and above.
5. Decode throughput and memory for each promoted architecture. These depend on the architecture, so one measurement per architecture is enough.

**Derived quantities**
- **Overoptimization:** E[Δ_proxy − Δ_verified,0] over the selected edits. This is the winner's-curse effect at the discovery budget.
- **Scale attenuation:** E[Δ_verified,0 − Δ_verified,k].

### F4. Transfer by edit category

**Edit pool (OBSERVED).**
- 75 AutoPareto edits: architecture 32, optimizer 21, optimization 10, batching 8, schedule 4 (`RDB` `change_category`).
- 55 AutoResearch edits (optimization 25, optimizer 13, schedule 9, architecture 5, batching 2).
- 42 completed Pair-A edits, with finer categories in `history.jsonl` → `change_category`.
- All the AutoPareto and AutoResearch edits have `code_diff` and parents. The Pair-A sources can be reconstructed (D).

**Sample.** 48 edits: 8 classes × 6, stratified by thirds of base effect, **including rejected and dominated edits**. They come from about 12 lineages, about 60 distinct sources in all.

**Statistics.**
- Paired effect at each rung: Δ_k(e) = mean over seeds of [BPB(child) − BPB(parent)].
- Errors-in-variables regression Δ_k = α_k + β_k·Δ_0 + ε. The base effect Δ_0 is noisy, so plain regression would bias β_k.
- Sign retention among edits with Δ_0 < 0.
- Kendall τ of edit effects between rungs.
- A hierarchical model of β_k by class.

**Expected precision (INFERRED).** If each edit's Δ has a standard error of about 0.003 and base effects vary by 0.015–0.02 across edits, β_k has a standard error of about 0.04 overall and 0.1–0.15 per class.

**Hypotheses to preregister.** These are tests, not findings.
- Size and batch edits depend on the budget. Their effects should shrink, then flip, as compute grows (B1, B4–B6).
- Schedule edits expressed as fractions transfer.
- Learning-rate edits partly transfer.
- For structural edits, no prediction is made.

### F5. Dependence on search pressure

**Why it matters (INFERRED).**
- The maximum of K noisy draws exceeds the mean by about σ√(2 ln K). With K = 25 and a 300-second seed σ of 0.002–0.008, that is about 0.005–0.02 BPB.
- This is as large as most of the learning-rate and architecture gains the agents kept (B5, B6).
- So the study must re-measure base effects at rung 0 with fresh seeds. Otherwise regression to the mean will be mistaken for loss of the effect with scale.

**Covariates per edit**
- attempt index;
- number of siblings evaluated before the edit was kept;
- selection rank;
- |Δ_proxy|/σ_seed;
- campaign length;
- contract (fixed time or fixed tokens).

**Manipulation.** New campaigns that differ in search budget: 10 against 25 attempts, and a strict against a lenient acceptance threshold. Test whether β_k falls as selection intensity rises.

### F6. Campaigns and seeds

**Campaigns.**
- The edits should come from campaign-replicated searches. The design in `paper/NEXT_EXPERIMENT_PLAN.md` L9–20 is suitable: 5 campaigns per contract, 10 attempts each, with paired seeds.
- Add **8 campaigns × 25 attempts at rung 0 and 4 × 25 at rung 1** for the search-pressure arm.
- The campaign is the unit of analysis. Do not merge the existing single trajectories into the new data (`NEXT_EXPERIMENT_PLAN.md` L43).

**Training seeds.**
- 3 per promoted source at rungs 0–2.
- 2 at rung 3, plus a third when the interval straddles the decision threshold.

### F7. Costs (DERIVED; 5 min of overhead per run at rungs 0–2 and 10 min at rung 3)

| Block | Runs | GPU-h |
|---|---|---:|
| Rung-0 re-measurement | 60 sources × 3 seeds × 10 min | 30 |
| Rung 1 | 60 × 3 × 25 min | 75 |
| Rung 2 | 60 × 3 × 85 min | 255 |
| Rung 3 | 60 × 2 × 330 min = 660 GPU-h ungated; about 50% with gating (S1) | 330 |
| Panel token ladder (C2, 4× and 16×) | 7 × 3 × 2 | 59 |
| IsoFLOP check | 12 × 25 min + 12 × 85 min | 22 |
| New agent campaigns | 200 attempts at rung 0 (about 12 min each) + 100 at rung 1 (about 30 min) | 90 |
| **Total** | | **≈ 861 (≈ $379); 1,191 ungated (≈ $524)** |

**Elapsed time.** About 36 days on one A40, about 9 days on four. Add agent API cost (UNKNOWN). Budget it explicitly: the Pair-A campaign stopped at 22 of its 25 registered attempts per arm (`study/contracts_2026/pair_registration.json` → `attempts_per_campaign`; `PA/*/history.jsonl`, 22 lines each).

### F8. Stopping rules (preregistered)

- **S1 (gate between rungs).** An edit moves to rung k+1 if its rung-k paired effect has P(beneficial) ≥ 0.1, or if it belongs to the preregistered random "always promote" third. The random third keeps β unbiased.
- **S2 (category futility).** Stop promoting a class, except its random third, when its pooled transfer ratio at rung k has a 90% upper bound below 0, or a lower bound above 0.8 at rungs 1–2.
- **S3 (global).** If, after rung 1, the 95% interval for the overall β lies inside [−0.2, 0.2] or inside [0.8, 1.2], run rung 3 only for the random third.
- **S4 (runs).** No hidden retries; divergence, NaN and OOM count as recorded failures. One re-run with the same seed and the same hash is allowed only for a documented infrastructure failure. This mirrors `study/contracts_2026/protocol.md`, "Candidate failure and attempt consumption".
- **S5 (data).** Abort any run that reaches `epoch > 1`.
- **S6 (budget).** A hard GPU-hour cap per rung. If it is reached, report the design as completed so far.
- **S7 (blinding).** Freeze the operator, classifier and analysis code before rung 1. Open the sealed shard once, at the end.

---

## Appendix: formulas used

- **Width:** w = ⌈d·r/128⌉·128.
- **Parameters:** total = 2·V·w + n_ve·V·kv + d·(2w² + 2w·kv + 2·m·w²) + n_ve·32·h_kv + 2d, with V = 8,192, m = 4 and kv = h_kv·128.
  - N_flop = total − V·w − n_ve·V·kv − 2d.
- **FLOPs per token:** 6·N_flop + Σ_layers 12·h·128·min(window, 2048).
- **Checks:** these formulas reproduce all seven logged parameter counts and FLOPs per token exactly (`SCR/model_math.py`).
- **Time and cost:** time = D/(tokens/s) + 300 s; cost = $0.44 × hours.
- **Memory:** M_MiB ≈ 16·N/2²⁰ + (B/64)·(2,194 + 5.0145·d·w).
- **Slope:** ΔBPB per e-fold = ΔBPB/ln(D₂/D₁) at a fixed model.
