# 22 — Phase 2: independent re-verification of the Opus winners and the frozen downstream suite (2026-11-07)

This is Phase 2 of the PI's three-phase plan. It ran on the same A40 as Phase 1 (pod4, `GPU-3adf8a47…`, doc 21).

Constraints honoured:
- No training ran. Only inference and evaluation of existing checkpoints.
- No protocol, threshold, timer, prompt, scoring rule or data split was changed.
- No historical database or result was written to.
- SSH endpoints and pod identifiers are not recorded here.

## Generated outputs
All generated outputs are in `results/camera_ready_2026/phase2/`, produced by `scripts/analyze_phase2_verification.py`:
- `PHASE2_HEADLINE.md` and `phase2_headline_table.tex`: the paper table.
- `PHASE2_TABLES.md`: detail tables 1–4.
- `phase2_verification.json`: every number, plus the SHA-256 of every input.
- `raw/`: the archived raw evidence (§8).

## 1. What ran

| Step | What | Where (Env) | UTC | GPU time |
|---|---|---|---|---|
| 2A | Identity checks, protected BPB, and 3 bracketed inference repeats each for Opus attempts 23, 24, 27 (repeat order rotated) | Env B: trusted tree, torch 2.8, registered cache | 03:28–05:20 | 1 h 52 min |
| 2B | Frozen downstream suite on 6 checkpoints | Env A: `uv run --locked`, torch 2.9.1 | 05:20–05:38 | 18 min |
| 2C | Held-out BPB by token position for 9 checkpoints, plus the BLiMP paradigm map | Env A | 05:38–05:43 | 5 min |

Details:
- **2A** also re-measured the protected BPB of the paired Pair-A baseline T checkpoint.
- **2A protocol:** `autopareto_search_inference_bracketed_v2`, against paired T checkpoint `f7ca17c6`.
- **2B evaluator and suite:** `downstream_evaluator.py` `696f9c46` with suite `3dce2c73`, exactly as in the August 201M panel. Results went into a new database, `phase2_downstream.db`.
- **2C** used the suite's exact 2,560 × 2,048-token held-out segments and the frozen `ModelScorer`. This step is exploratory and was added post hoc.
- **Drivers:** `raw/pod_phase2/scripts/`, namely `p2a_winners.sh`, `p2b_downstream.sh`, `p2c_context_position.py` and `p2_all.sh`.
- **Aborted first launch:** the first 2A launch (`…T032656Z`) aborted before any measurement. The candidate contexts lacked the registered `prepare.py` (sha `4f2ba9cb`). The relaunch installed and hash-checked it, which is exactly how the registered baseline context is laid out. The aborted directory is kept and marked `ABORTED.txt`.
- **Identity checks before any measurement:**
  - Checkpoints: 23 `f098e0bd`, 24 `e5687149`, 27 `520af249`, baseline T `f7ca17c6`.
  - Sources: `408228f4`, `d7c7d51b`, `b8eaf5e9`.
  - Phase-1 checkpoints: AP292 seed 3 `9dbea197`, seed 4 `a4017dcc`, seed 5 `1559944f`; AutoResearch seed 3 `84687769`. All four equal the hashes recorded in the August bundles.

## 2. Headline table

| Model | Budget | Pair-A val BPB | Held-out BPB | LAMBADA | BLiMP | PIQA | ARC-E | Decode speedup (stable) | E2E speedup | Peak MiB |
|---|---|---|---|---|---|---|---|---|---|---|
| Opus 5.5 attempt 23 | 300 s (197M tok) | 1.0962 | 1.158 | 0.231 | 0.711 | 0.572 | 0.349 | 31.8× (2/3) | 4.6× | 113 |
| Opus 5.5 attempt 24 | 300 s (198M tok) | 1.0995 | 1.161 | 0.219 | 0.722 | 0.577 | 0.345 | 30.9× (3/3) | 5.3× | 97 |
| Opus 5.5 attempt 27 | 300 s (198M tok) | 1.0962 | 1.158 | 0.228 | 0.712 | 0.589 | 0.372 | 31.5× (1/3) | 5.3× | 99 |
| Pair-A baseline | 300 s (85M tok) | 1.2011 | 1.249 | 0.170 | 0.652 | 0.573 | 0.348 | 1× (2/3) | 1× | 337 |
| AutoResearch | 300 s (88M tok) | — | 1.168 | 0.203 | 0.702 | 0.560 | 0.353 | — | — | — |
| AutoPareto AP292 | 201M tok, seeds 3–5 | — | 1.142 ± 0.006 | 0.215 ± 0.019 | 0.720 ± 0.004 | 0.583 ± 0.003 | 0.369 ± 0.004 | — | — | — |
| AutoResearch | 201M tok, seeds 3–5 | — | 1.100 ± 0.002 | 0.204 ± 0.009 | 0.746 ± 0.006 | 0.587 ± 0.014 | 0.365 ± 0.014 | — | — | — |
| Phase-1 baseline | 201M tok, seeds 3–5 | — | 1.147 ± 0.005 | 0.205 ± 0.006 | 0.714 ± 0.008 | 0.574 ± 0.004 | 0.353 ± 0.015 | — | — | — |

How to read the table:
- **Fixed-time rows** are one training run each, so their differences carry no seed variance.
  - Opus and the Pair-A baseline come from the Pair-A harness, seed 3.
  - AutoResearch 300 s is E0, from the Phase-1 harness, seed 0.
- **Matched-token rows** are the August panel, reused: mean ± sd over seeds 3–5.
- **Single-run 95% intervals:** held-out BPB ±0.007, LAMBADA ±0.012, BLiMP ±0.003, PIQA ±0.023, ARC-E ±0.019.
- **Decode speedup:** decode only, bracketed against the paired baseline, median of 3 repeats. The parenthesis gives the repeats that passed the frozen 2% drift rule.
- **E2E speedup:** a 1,024-token prefill plus 128 greedy tokens. It uses the frozen `end_to_end_tokens_per_second` field, median of 3 repeats. The reference is the baseline recipe measured as a candidate on this pod (doc 21 §8).

## 3. Winner re-verification (2A)
- **Protected BPB reproduces bit for bit:**

  | Model | Campaign record | Re-measured |
  |---|---|---|
  | Attempt 23 | 1.0961979650 | identical |
  | Attempt 24 | 1.0994636270 | identical |
  | Attempt 27 | 1.0962273901 | identical |
  | Baseline T | 1.2011009456 | identical |

- **Decode:**
  - Opus decodes at 4,570–4,592 tok/s, against a paired baseline of 135–151 tok/s, for a normalized speedup of 30.3–33.9× across all 9 repeats.
  - The campaign medians (30.5–31.4×) fall inside that range.
  - The candidate's own decode is very steady: block-to-block drift ≤ 0.2% and CV ≤ 0.1% in every repeat.
- **Instability, reported as measured and with no threshold changed:**
  - Stable repeats: attempt 24 3/3, attempt 23 2/3, attempt 27 1/3.
  - The three failing repeats had normalized drift of 0.040, 0.059 and 0.030, against a limit of 0.02.
  - **Root cause:** in every failing repeat the drift comes from the baseline brackets, not the candidate. The worst baseline bracket in each failing repeat drifted 0.030–0.082. The baseline decodes at about 140 tok/s, a speed limited by CPU kernel launches. Pod4 is the host that failed the CPU screen and gave REJECT on the Phase-1 probe (doc 21 §3, §8).
  - No speed claim depends on these repeats: the speedup stayed between 30× and 34× in every one of them.
- **End-to-end:**
  - Opus reaches 648–791 tok/s against the baseline recipe's 146 tok/s, which is **4.6–5.3×, not about 31×**.
  - **Why:** prefill of 1,024 tokens takes 140–143 ms for Opus (median) against 7.5 ms for the baseline. Opus's adapter captures the CUDA graph for its decode step inside every `prefill()` call (`_build_decode_graph`). The frozen harness charges that capture to prefill.
  - The median over all ~2,300 timed repetitions per repeat gives the same picture: 748–762 against 143 tok/s, 5.2–5.3×.
  - The end-to-end figure is noisy between repeats: the campaign record was 650–816 tok/s.
- **Memory:** peak allocated inference memory is 97–113 MiB against 337 MiB, 3.0–3.5× lower. The KV cache is 3.6 MiB against 18.0 MiB.
- **Correctness:** passed in 9/9 repeats.
  - Attempts 23 and 27 pass through FP32 adjudication: their bf16 logit tolerance is exceeded, but greedy tokens match and the FP32 check passes.
  - Attempt 24 passes on the bf16 fast path.
  - These modes are the same as in the campaign records.

## 4. Downstream suite (2B)

### Fixed time (300 s)
Against the Pair-A baseline, all three Opus winners are better on:
- held-out BPB, by 0.088–0.092;
- LAMBADA, by 0.049–0.061;
- BLiMP, by 0.059–0.070.

PIQA and ARC-E differ from the baseline by less than one single-run CI. Opus trains on 2.3× more tokens in the same 300 s.

Against AutoResearch 300 s (a different harness and seed):
- held-out BPB is lower by 0.006–0.010;
- LAMBADA is higher by 0.016–0.028;
- BLiMP is higher by 0.009–0.020;
- PIQA and ARC-E are within a CI.

The 201M panel's seed-to-seed sd is about 0.002–0.006 BPB and 0.004–0.008 BLiMP. So single-seed gaps of this size are suggestive, not established.

### Matched tokens
This comparison is approximate and crosses harnesses. Opus's 197–198M tokens are 98% of the panel's 201.3M, but they come from a different trainer.

| Metric | Opus | AutoResearch 201M | AP292 201M | Phase-1 baseline 201M |
|---|---|---|---|---|
| Held-out BPB | 1.158–1.161 | 1.100 ± 0.002 | 1.142 ± 0.006 | 1.147 ± 0.005 |
| BLiMP | 0.711–0.722 | 0.746 | ≈ Opus | ≈ Opus |
| LAMBADA | 0.219–0.231 | 0.204–0.215 (all three references) | | |

Opus is worse on held-out BPB than all three matched-token references. It is below AutoResearch on BLiMP, roughly level with AP292 and the Phase-1 baseline, and at or above all three on LAMBADA.

**Per token, the 3-layer Opus model is less sample-efficient than the 8-layer AutoResearch model. Its fixed-time advantage comes from throughput.**

### Reproducibility
AP292 seed 3 was re-evaluated on pod4, with a different host and Python stack from August. It reproduces all 9 August metrics exactly, |Δ| = 0.

## 5. BLiMP by phenomenon (`PHASE2_TABLES.md` Table 3)
This breaks BLiMP into its 13 dataset `linguistics_term` phenomena (67 paradigms, 67,000 pairs). Single runs carry Wilson 95% CIs; the 201M panel carries mean ± t-based 95% half-width over 3 seeds.

**Opus against the Pair-A baseline.** Opus is higher on all 13 phenomena.
- Largest gains: subject–verb agreement +0.12, NPI licensing +0.11 to +0.13, irregular forms +0.10 to +0.11.
- Smallest gains: quantifiers and ellipsis, where the CIs overlap.

**Opus against AutoResearch 300 s.**
- Subject–verb agreement is higher for all three winners, with disjoint CIs: 0.803–0.806 against 0.774.
- NPI licensing is higher: 0.570–0.590 against 0.550.
- Most other phenomena overlap.
- Island effects vary most between winners: 0.510–0.578.

**Against AutoResearch at 201M,** Opus is lower on most phenomena. This matches §4.

## 6. Short- versus long-context test (2C; exploratory, post hoc)
**Question.** Opus uses windows of 256, 256 and 2,048 tokens (3 layers, pattern SSSL). Every reference model's shortest window is 1,024 tokens:
- Pair-A baseline and AP292: SSSL, half context;
- AutoResearch: SSSS.

Every model, Opus included, has a full-context last layer. A 256-token window can only matter for target positions ≥ 256.

**Data and check.**
- All 9 models were scored on the same 2,560 held-out segments (same hash).
- Each model's overall BPB equals its frozen-suite held-out BPB to 6 decimals.

**BPB gain from late positions [1024, 2048) over early positions [128, 256):**

| Model | Gain |
|---|---|
| Opus 23 / 24 / 27 | −0.089 / −0.092 / −0.090 |
| AutoResearch 300 s | −0.085 |
| AutoResearch 201M seed 3 | −0.087 |
| Pair-A baseline | −0.069 |
| AP292 201M seeds 3 / 4 / 5 | −0.097 / −0.109 / −0.095 |

**Paired bootstrap over segments** (1,000 reps). How much more Opus gains from late positions than each reference, with 95% CI:

| Reference | Opus's extra late-position gain |
|---|---|
| Pair-A baseline | 0.020–0.023 more, CI excludes 0 |
| AutoResearch 300 s | 0.003–0.006 more, CI excludes 0 |
| AutoResearch 201M | 0.002–0.005 more |
| AP292 seed 3 | 0.006–0.009 less, CI excludes 0 |

Across AP292 seeds, Opus's shortfall is 0.003–0.020.

**Reading.**
- There is **no sign that the 256-token windows cost long-range modelling** compared with the AutoResearch models or the Pair-A baseline. Opus's gap to AutoResearch 201M is flat across positions from 128 onwards (+0.064 to +0.073).
- Against AP292, the gap widens slightly with position, by at most about 0.02 bits/byte at late positions. This is confounded with depth (3 against 4 layers), harness and seed, and rests on one Opus seed.
- LAMBADA is not a test of this: its passages are mostly shorter than 256 tokens.
- Separately, at positions < 128 Opus is worse than the Pair-A baseline by 0.073–0.079 and better than both AutoResearch models. No window can affect that bucket, and this test does not explain the difference.

## 7. Not done, and why
- **Codex G4 (attempt 14) was not re-benchmarked.** Its checkpoint (`63bf0276…`) exists nowhere (doc 21), and re-benchmarking it would need retraining. As instructed, it was not retrained.
- **The Opus attempt 1 checkpoint is also absent.** It is not one of the winners.
- **AutoResearch 201M seeds 4 and 5 were not re-scored in 2C.** In the September snapshot, seed 4's checkpoint is truncated (3 MB) and seed 5's is missing. Their August downstream results are intact and are used in §4–5. The context test uses AutoResearch 201M seed 3 only.
- **No end-to-end or memory numbers for the matched-token models.** The bracketed speed protocol is defined for Pair-A candidates against the paired T checkpoint; it was not extended.
- **Qualitative generations were saved but not analysed.** There are 18 per model, in `raw/pod_phase2/downstream/runs/*/results.json`.

## 8. Evidence
- **`raw/winner_verification_and_host_probe.tar.xz`** holds the exact bytes of the 2A directory, the aborted 2A directory and the Phase-1 probe repeats.
- **`raw/pod_phase2/`** holds the downstream DB, bundles and logs, the context-position JSON (per-segment nats and bytes), the BLiMP map, the drivers and the master logs. It leaves out checkpoint copies.
- **`raw/SHA256SUMS`** lists 81 files. The 77 archived files were verified against it after extraction. The 4 checkpoint lines record the copies used on the pod.
- **Off-pod backup:** the hash-verified 15-minute pull copied the full originals to the Mac backup, `pod4_GPU-3adf8a47_live/` (all files verified, 05:48 UTC).

## 9. Conclusions (paper-ready)
1. **The Contract-T result holds.** The protected BPB of all three Opus winners and of the paired baseline reproduces exactly on a fresh A40.
2. **Decode speedup is about 31× and robust, but end-to-end speedup is about 5×.** Decode-only speedup was 30–34× in all 9 repeats, while end-to-end speedup with a 1,024-token prefill is 4.6–5.3×.
   - Prefill is about 19× slower than the baseline's, because Opus captures a CUDA graph per prompt.
   - Speed claims must say which of the two they mean.
   - The 3 of 9 repeats that failed the frozen stability rule failed because the baseline brackets drifted on a noisy host. They are reported, not re-run until they pass.
3. **Memory:** peak inference memory is 3.0–3.5× lower.
4. **Quality at fixed time.** At a fixed 300 s, Opus beats the Pair-A baseline clearly and AutoResearch modestly on language modelling, LAMBADA and BLiMP. It does not improve PIQA or ARC-E.
5. **Quality at matched tokens.** At about matched tokens, Opus is worse than AutoResearch on held-out BPB (1.158–1.161 against 1.100) and BLiMP. Its quality gain is a throughput effect, not better sample efficiency.
6. **Sliding windows.** The 256-token windows show no measurable long-range penalty against AutoResearch, and at most a small one against AP292. This test is exploratory.
