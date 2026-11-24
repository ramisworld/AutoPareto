# Phase 6 pre-registration: unified benchmark of the 13 frozen candidates (2026-11-16)

Authorised by the author on 2026-11-16 ("AutoPareto: Final Unified Benchmark and Camera-Ready
Experiment Campaign"): one RunPod A40, a planning budget of about 24 A40-hours (30 h contingency).
This is a verification campaign. No search, no agent call, no new architecture. Machine-readable
plan: `evaluation/phase6_unified_plan.json`. Written before any Phase 6 GPU measurement.

## 1. The 13 frozen candidates

| Key | Display name | Frozen source (sha256 prefix) | Seeds 6-8 at 201,326,592 tokens |
|---|---|---|---|
| baseline | Baseline | `results/candidates/baseline-a40-seed0/0002-train.py` (428e22f1) | new (P1) |
| arref | AR-ref | `results/candidates/autoresearch-diagnostic5-seed42-20260724/0005-train.py` (b3a961ee) | E1, reused |
| rs | Random Search | `results/candidates/random-search-seed42-20260805/0010-train.py` (a30482c3) | new |
| tpe | TPE | `results/candidates/tpe-seed42-20260805/0021-train.py` (8547b278) | new |
| motpe | MOTPE | `results/candidates/motpe-seed42-20260805/0022-train.py` (62882a20) | new |
| nsgaii | NSGA-II | `results/candidates/nsgaii-seed42-20260805/0013-train.py` (5d4ad2f6) | new |
| cmaes | CMA-ES | `results/candidates/cmaes-seed42-20260805/0015-train.py` (7163c7b2) | new |
| gpt_a22 | GPT A22 (AP292) | `results/candidates/autopareto-seed0-20260731/0022-train.py` (55a7fd8b) | E1, reused |
| gpt_a24 | GPT A24 (AP294) | `results/candidates/autopareto-seed0-20260731/0024-train.py` (6d0d8d1d) | new |
| gpt_a25 | GPT A25 (AP299) | `results/candidates/autopareto-seed0-20260731/0025-train.py` (36e3ad91) | new |
| opus_a23 | Opus A23 | Phase 5A mapped source of Pair-A attempt 23 | Phase 5A, reused |
| opus_a24 | Opus A24 | Phase 5A mapped source of Pair-A attempt 24 | Phase 5A, reused |
| opus_a27 | Opus A27 | Phase 5A mapped source of Pair-A attempt 27 | Phase 5A, reused |

**Classical picks.** Each classical method contributes its lowest archived validation-BPB recipe
among valid, completed, inference-compatible trials. This is the rule in
`autopareto/research-infrastructure:scripts/prepare_downstream_selection.py:160-176` and
`docs/RESEARCH_LEDGER.md:375-379`, committed on 2026-08-17 (3e3645c), before Phase 1 and long before
these evaluations. Its output (`results/downstream_selection.json`, selection sha 89c396ff) names
trials 247 (Random Search #10), 183 (TPE #21), 234 (MOTPE #22), 275 (NSGA-II #13) and 202 (CMA-ES
#15); every source hash above was re-checked. The TPE and MOTPE picks equal the Phase 1 panel
recipes. Caveat recorded in advance: Random Search #10 is byte-identical to TPE #10 and MOTPE #10
(all three methods share trials 1-10), and its search-time margin over its runner-up (0.0008 BPB) is
smaller than the 0.0011 spread of that recipe across the three runs.

The later 22-attempt GPT-5.6 campaign is not a candidate.

## 2. Environments (one per comparison)

| Comparison | Environment | Reason |
|---|---|---|
| Training (token and fixed-time cells), held-out and validation BPB, frozen downstream evaluator | Env A: `uv.lock` 262a4d55, Python 3.10.18, torch 2.9.1+cu128 | the environment of E1 and Phase 5A, whose 15 runs are reused |
| Inference (all modes and workloads) | Env B: `/opt/autopareto_runtime`, torch 2.8.0+cu128, triton 3.4.0 | the environment of every earlier 1,024/128 measurement (Phase 2A, 5B) |
| Expanded downstream suite (lm-evaluation-harness) | Env C: Env B plus a pinned `lm_eval` | lm-eval's dependencies are not in `uv.lock` |

Hardware: one A40 (`GPU-6de81321`, driver 595.91.07), a different host from E1 and Phase 5A (pod 4).
Drift controls (below) measure the effect of the new host on the reused results.

## 3. P0: correctness and the unified inference benchmark (absolute priority)

**Protocol (primary).** `autopareto_search_inference_bracketed_v2`, called unchanged through
`scripts/phase6_unified_inference.py`: 1,024 input tokens (the eight registered prompts), 128
generated, batch 1, bfloat16 autocast, greedy. Discarded 20 s warm-ups; eight rounds of baseline
before / candidate / baseline after, each window at least 8 s of decode. Bracketing baseline: the
registered Pair-A baseline (`study/contracts_2026/historical_reference/baseline_source/train.py`,
checkpoint f7ca17c6), canonical adapter, exactly as in Phase 2A. Default allocator.

**Recorded per run:** prefill latency and throughput; decode tokens/s (median of 8 windows, 95%
bootstrap CI), decode latency per token; end-to-end latency and throughput (prefill + first token +
127 decode calls); peak allocated and reserved memory (candidate-only pass), K/V cache bytes, weight
bytes, parameter count; cold start (definition load, checkpoint read, model build, adapter build,
first prefill, first decode calls) with a fresh compiler cache; speed-up over the bracketing baseline
with its 95% CI.

**Checkpoint staging.** Every checkpoint (and the bracketing baseline) is read from the pod's local
overlay disk, at `/root/phase6_stage/<sha256>.pt`. The staged file's SHA-256 is checked against the
recorded hash of the reused or new run before every run, and the events log keeps the original path.
Reason: on 2026-11-16 the network volume stalled for minutes on single 64 MiB chunks. Checkpoint
read time depends on storage, so it is reported on its own row and left out of the
compile and graph-capture overhead comparison.

**Correctness (every run, before timing).** All 8 prompts x 128 generated tokens against the
candidate's own uncached `model.forward`, teacher-forced in one pass over prompt + generated tokens.
This covers positions 1,024-1,151 and the ring-cache wrap. A generated token that is not the
reference argmax is a hard mismatch unless the reference prefers its own argmax by at most 0.25
logits (a bfloat16 tie). The timed function's own outputs (`one_measured_repetition`) must equal the
validated tokens. **Decision rule:** any hard mismatch, non-finite output or timed-output difference
means the run is recorded as `incorrect_not_timed` and contributes no speed. Max and mean logit
errors are reported for every run.

**Modes.**
- `native` (primary, all 13 x seeds 6-8 = 39 runs): the candidate's own adapter where it defines one
  (Opus A23/A24/A27: CUDA graphs, fused regions, ring K/V cache), otherwise the protected canonical
  adapter. Opus's engineering is not given to any other candidate.
- `uncached` (architecture-only control, all 13, seed 6): no K/V cache; each step recomputes
  `model.forward` over the full prefix. Model-neutral; labelled as a control, not an agent result.
- Opus eager reference (seed 6): Opus's own adapter with CUDA graph, compiled regions, split-key
  attention and ring cache all switched off (the Phase 5B mechanism, all four switches at once).
- `canonical` forced on Opus (seed 6, correctness only): documents whether the protected canonical
  adapter computes the Opus function (expected: no; it omits the token-shift mix and QK gain).

**Order.** P0a (now): the 15 existing checkpoints, seed-major (seed 6 for all, then 7, then 8),
models in a fixed shuffled order rotated by one per seed. P0b (after P1): the 24 new checkpoints in
the same scheme, then three cross-session anchors (AR-ref, GPT A22, Opus A23, seed 6) re-measured
to quantify any session offset. Serial on the single GPU, `nvidia-smi` idle check before each run.

## 4. P1: matched-token training (seeds 6-8, 201,326,592 tokens)

24 new runs: baseline, rs, tpe, motpe, nsgaii, cmaes, gpt_a24, gpt_a25 x seeds 6, 7, 8, through the
E1 code path unchanged (`run_one` of `scripts/run_phase1_equal_tokens.py`, the protected Phase-1
launcher and source transform, the E1 integrity gates of `gate_token_run` with each recipe's own
batch). Milestones at 25/50/75/100% of the budget, each scored for validation BPB (shard 06542) and
held-out BPB (shard 06541, revision 915333b4). Recorded: tokens, optimiser updates, batch, steady-state
throughput, peak training memory, wall time. Baseline uses its own TOTAL_BATCH_SIZE (2^19, 384
updates); every other recipe uses 2^17 (1,536 updates). One retry per run; failures stay on record.

**Drift controls (3 runs):** AR-ref, GPT A22 and Opus A23 at seed 6, 201.3M tokens, rerun on this
host (the Opus run through Phase 5A's `execute_opus_attempt`, deployment export included).
Rule (the Phase 5A rule, unchanged): the reused E1 values stand if each rerun's held-out BPB is
within 0.005 of its E1 value (Phase 5A's same-seed control differed by 0.0010). Training throughput
on the new host is reported beside the reused values; if it differs from E1 by more than 3%, the
throughput columns use only same-host runs or say which host each value comes from.

**Frozen downstream evaluator** (`downstream_evaluator.py` 696f9c46, suite 3dce2c73: ARC-Easy, PIQA,
LAMBADA, BLiMP) on the 24 new checkpoints, as in Phase 5A.

## 5. P2: fixed-time 300 s and the expanded downstream suite

**Fixed time.** The E1 fixed-time cell unchanged (historical launcher aa9dc8b2, 300 s of synchronised
step time after step 10, gate window [300, 315] s): baseline, rs, tpe, motpe, nsgaii, cmaes, gpt_a24,
gpt_a25, opus_a23, opus_a24, opus_a27 x seeds 6-8 (33 runs) plus one AR-ref seed 6 drift control.
AR-ref and GPT A22 E1 300 s runs are reused. **Opus:** its Pair-A loop counted wall time from the
start of training (compilation included) and was stopped by the Pair-A boundary. The Phase 5A
mapped source (`evaluation/phase5a_sources/*.phase1.train.py`, reviewed and byte-exactly
re-derived in Phase 5A) restores exactly that driver to the panel baseline's text, so in the E1
fixed-time cell Opus is timed by the identical rule. Risk recorded in advance: the frozen evaluator
may not load a checkpoint saved after Opus's storage conversion (A27 stores int8 tables); the E1
gate (checkpoint vs in-process validation within 0.002) decides, and a failing cell is rescored on a
deployed re-export only under a documented amendment.

**Expanded downstream suite** on the 39 matched-token checkpoints with lm-evaluation-harness (pinned
version, recorded): ARC-Easy, ARC-Challenge, PIQA, HellaSwag, WinoGrande, OpenBookQA, SciQ, BoolQ,
LAMBADA (OpenAI), BLiMP (67 sub-tasks, category breakdown), zero-shot, `acc` and `acc_norm`, plus
WikiText (document-level, test split, bits per byte). Model access goes through a thin lm-eval `LM`
subclass that scores with the same forward pass as the frozen evaluator; it is cross-checked against
the frozen evaluator on the four shared tasks. ClimbMix is a web corpus, so WikiText overlap cannot
be excluded; it is reported as such, and all 13 candidates share the same training data.

## 6. P3: ablations and supplementary workloads

- Phase 5B raw records re-verified. Correction recorded in advance: Phase 5B's "identical greedy
  samples" check compared null hashes (no sample file was written), so it established nothing.
- Speed-uncertain lineage steps: Pair-A campaign checkpoints of attempts 17, 18, 19, 22 and 23
  (seed 3) re-measured with the primary protocol, three repeats each.
- Supplementary workloads (seed 6, native mode): 256/128, 512/128 and 1,024/128 at batch 4 where
  the adapter supports it, with the timed function generalised only in its token counts.

## 7. Statistics and reporting

Per candidate: mean and SD over seeds 6-8 and a t-based 95% CI (n = 3, so CIs are wide). Pairwise
contrasts are paired by seed (the seeds fix the data order). No claim of superiority where the CI
of a difference includes zero; no p-value hunting; downstream differences within seed noise are
called ties. Chance and majority baselines are printed beside every task. Inference: per-run window
CIs and the across-seed spread. Fixed-time and matched-token quality are never shown in one
comparison without labels. Every table and figure is generated by script from the raw outputs.

## 8. Budget (A40-hours, to be recomputed after the P0 and P1 pilots)

| Block | Estimate |
|---|---|
| P0: pilots, 39 native, 13 uncached, Opus eager and canonical, 3 anchors | 5.2 |
| P1: 27 training runs (24 new, 3 drift controls), frozen downstream on 24 | 7.7 |
| P2: 34 fixed-time runs, lm-eval on 39 | 9.0 |
| P3: lineage and supplementary workloads | 3.3 |
| Total | about 25 |

The author is alerted before the projected total exceeds 24 h. No required result is dropped to
save time without the author's decision.
