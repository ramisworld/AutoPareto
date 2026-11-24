# 26. Inference-editing audit: why only Opus rewrote the decode path (2026-11-15)

Internal audit for camera-ready revision 8. Read-only. No experiment, GPU job or network call.
Evidence root: `/Users/Rami/Documents/dev/autopareto`. Historical code is cited as git objects on the local
branch `autopareto/research-infrastructure` (`git show <commit>:<path>`), because the working-tree
`protected/inference_adapter.py` is a later, uncommitted version.

- `EXT` = `runs/pair-a-official-20260914-new-a40-continuation-20260916-r2/pair-a-fixed-time-seed3` (GPT-5.6, extended).
- `OPUS` = `runs/pair-a-claude-opus55-t-seed3-new-a40-continuation-20261105c/pair-a-fixed-time-seed3`.

## Answer

**Opus's inference work came from search decisions, under harness conditions that differed in several
recorded ways. It did not come from a different editable surface.**

- **Same editable surface.** Every AutoPareto agent could write only `train.py`. In every era that file could
  define `build_inference_adapter(model)`, and the protected benchmark then timed the candidate's own
  `prefill` and `decode_one`.
- **Different decisions.** GPT-5.6 never used that hook in 97 attempts: 75 original attempts plus 22 extended
  ones. Opus used it from attempt 8 onward.
- **Not attributable to model family.** The extended arms also differed in:
  - reasoning setting;
  - execution environment;
  - call timeout from Opus A10 onward;
  - harness repairs;
  - GPU instance;
  - number of attempts.

  With one trajectory per arm, the difference cannot be attributed to the model family.

## A. Original GPT-5.6 AutoPareto (75 attempts, 2026-07-28 to 08-04)

1. **Starting `train.py` had no inference code.**
   - The baseline is `git show 5ea452c:train.py` (sha256 `428e22f1…`, 630 lines).
   - It has a full-sequence `GPT.forward` (lines 268–295) using FA3 `flash_attn_func` (line 93).
   - It has no KV cache, no decode loop, no generate function and no adapter.
   - `torch.compile` is applied only to training (line 508) and to the optimizer steps (lines 305 and 316).
2. **The agent could replace the per-step decode implementation.**
   - `ece3dbd:inference_benchmark.py` lines 48–58 execute the candidate code above the setup marker.
   - `ece3dbd:protected/inference_adapter.py` lines 260–273 (`build_adapter`) use the candidate's
     `build_inference_adapter(model)` if it exists. Otherwise they use the protected `CanonicalGPTAdapter`.
   - The protected timed loop (`one_measured_repetition`, benchmark lines 256–296) calls `adapter.prefill` and
     then 255 calls to `adapter.decode_one`.
   - On the canonical path, which all 75 attempts used, the protected code re-implements the model. It calls
     candidate submodules, including `MLP.forward` (adapter line 181), but never calls the candidate's
     `GPT.forward`.
   - So architecture and configuration edits changed the timed computation, while edits to forward code would
     not have.
3. **A candidate adapter would have been accepted and measured.**
   - The hook is in protected code from `75af7b7` (2026-07-24).
   - It is documented in `c200ccb:docs/candidate_interface.md` lines 43–52.
   - Gates: three methods, cached logits matched to the candidate's own forward, and exact `cache_nbytes`.
   - No code rejected compile or CUDA-graph use.
   - Practical cost: an inference failure made the attempt a dominated crash, and from program `7d6817f0…` it
     **paused the campaign**.
   - The FP32 adjudication added on 07-30/31 re-ran the candidate's FA3 forward in float32. That is the same
     latent defect Opus later hit at A8 (Section C).
4. **The research program already offered an inference adapter.**
   - Program `7d6817f0…` = `git show 8548308:programs/program_autopareto.md`; `af42d041…` differs only by
     the pause rule.
   - It lists "inference-oriented model structure and a candidate cache adapter that satisfies the protected
     interface" (lines 38–39).
   - It gives the same `build_inference_adapter` contract (lines 57–72).
   - It frames the adapter as needed when "an architecture … changes the canonical fields" (line 58).
   - It warns "Do not force architectural novelty for its own sake" (line 41).
5. **GPT-5.6 made no inference-implementation change.**
   - 0 of 75 candidates define an adapter, cache, graph or compile change. All 75 queue files reproduce the
     database `code_diff`.
   - Every edit sits in one of three places:
     - the hyperparameter block;
     - `n_kv_head` in `build_model_config`;
     - Muon step plumbing.
   - Category counts: architecture 32, optimizer 21, optimization 10, batching 8, schedule 4. The offered
     "systems" category was used 0 times.
   - Terms: "CUDA graph", "compile" (outside unchanged context lines), "fuse", "kernel", "quantiz" and "int8"
     appear in 0 attempts.
   - "KV cache" appears in 19 attempts. Every one predicts or reports cache shrinkage from GQA, depth or
     width, for example seed-0 A3: "halve KV-cache width and reduce K/V projection work".
   - Full term table: revision-8 scratch output `original_term_hits.tsv`, reproduced by grep of
     `queue/autopareto-seed*/`.
6. **The instructions encouraged and discouraged inference work.**
   - Encouraged:
     - the adapter in the research surface;
     - "may be retained with slightly worse BPB when it provides a real speed or memory advantage";
     - a `systems` category.
   - Discouraged:
     - "Cache correctness is a mandatory gate";
     - the campaign pauses on an inference failure;
     - "must perform the same model computation … must not branch on … call order, repetition";
     - a stale protected docstring saying architecture changes need "a reviewed protected adapter addition".
   - No program mentions CUDA graphs, compilation for inference, kernels or quantization.
7. **Correction to the paper (applied in revision 8).**
   - The 75 discovery attempts were timed by `inference_benchmark.py` in `cached_decode` mode: 256 in /
     256 out, 2 warm-ups, 10 repetitions (`ece3dbd:benchmark_config.py` lines 1–13). The database shows
     `inference_metric_mode='cached_decode'` for all 74 measured attempts.
   - `autopareto_inference_stabilized_v2` is the later panel protocol. It is in `benchmark_stability.py`,
     whose docstring says "Discovery measurements used a fixed warm-up count. Final paper measurements use
     this adaptive protocol".
   - The appendix had both phases under stabilized_v2.
8. **AutoResearch.**
   - Inference was measured, but only as a gate and a record. Its program says the measurements "must never
     affect this decision".
   - The hook did not exist at its starting commit `4c302e5`, so the 50-attempt campaign could not supply
     inference code.

## B. Extended experiments: GPT-5.6 (22 attempts) versus Opus (27)

Common to both arms (per-attempt records in `EXT/history.jsonl` and `OPUS/history.jsonl`):

| Condition | GPT-5.6 | Opus |
|---|---|---|
| Starting source | `b2fb8788…` (no adapter, no graphs) | identical |
| Program | `8acffb40…` | identical |
| Agent-visible files | `campaign_view.json, prepare.py, program.md, study_runtime.py, train.py` | identical |
| Writable files | `train.py` | identical |
| Contract | T, 300.000 protected s, training seed 3 | identical |
| Benchmark | `autopareto_search_inference_bracketed_v2`, 1,024 in / 128 out | identical |
| Parent rule | same-arm, controller-eligible | identical program text |
| Response schema | `hypothesis`, `intended_change`, `change_category`, optional `parent_experiment_id`; two calls per attempt | identical |

The extended program lists "an inference adapter compatible with the protected interface" in the research
surface and gives the same hook, adding "If the architecture changes the canonical cache fields, define
`build_inference_adapter(model)`". It also adds:

- "The baseline implementation is a starting point, not a constraint";
- "Think deeply and prefer significant, well-motivated experiments over cosmetic or repetitive edits".

Recorded differences (manifests `EXT/../provenance/final_pair_a_continuation_manifest_20260917_posttraining_failure_receipt_fix.json`
and `OPUS/../provenance/claude_t_continuation_manifest_20261105c_timeout900_attempt9_candidate_failure.json`):

| Difference | GPT-5.6 | Opus |
|---|---|---|
| Model and reasoning | `gpt-5.6-sol`, high | `claude-opus-5-5`, xhigh |
| Client | Codex CLI 0.153.0 | Claude Code 2.1.318 |
| Execution | Docker container, shell inside (`sandbox_mode` "danger-full-access") | host, file tools only (`acceptEdits`), no shell |
| Call timeout | 600 s throughout | 600 s for A1–A9 and 900 s from A10. Raised after the A9 edit call timed out three times "each time … writing an inference adapter" (amendment, PI-approved) |
| Evaluator | `f05d47a1…` throughout | `f05d47a1…` for A1–A7, then `c9b632e6…` |
| Evaluator change | none | A8's first run crashed the evaluator: FP32 adjudication sent FP32 tensors into the candidate's FA3 forward. Fixed (commit `f2873fc`) and the same A8 candidate re-run |
| Protected-adapter repairs | at T7 and T8: shared value embeddings and single-head broadcast (audit 06 §B.5) | none |
| Attempts | 22. Stopped when the Codex sandbox failed at global position 45 (`controller_state.json`: "Scientist sandbox command failed"); never resumed | 25 scheduled plus 2 PI-approved extension attempts to reach 25 valid (`provenance/extension_attempts_amendment.json`) |
| Interleaving | T and N arms alternate | T arm only |
| GPU | A40 `GPU-35264a0f…` | A40 `GPU-4702993a…` / `GPU-b74ec8bd…` |

**GPT-5.6's 22 attempts.**
- No attempt defines `build_inference_adapter`, a CUDA graph or compiled decode code (`rg` over `EXT/sources`: 0 files).
- Its 22 hypotheses contain "adapter", "graph", "compile" and "fus" 0 times.
- "Kernel" appears once (A22) and "KV cache" three times. Every use concerns architectural effects.
- All 22 categories are `architecture_*` (18) or `optimization_*` (4).
- Patches have 2–15 changed lines, and most intended changes start "change only …".
- Its decode speed-up came from smaller models only: 0.96× (A1, GQA, slower) to 1.58× (A19).
- Best validation BPB: A14, 1.1016 at 1.32×. The largest single gain was A12 (second batch halving, −0.062).
- It hit a capacity limit at A13 (depth 6, width 384, 15.0M parameters, BPB +0.007) and widened back at A14.
- The search ended because of an infrastructure failure, not a recorded decision. So "it could not discover
  graph replay" is not supported. It had the hook, the 600 s timeout, and 22 opportunities.

## C. Opus: attempt-by-attempt (hypothesis → change → measurement → verdict → children)

Speed-up = normalized decode-only throughput, campaign record. Memory = peak allocated inference memory, in MiB.
"Verdict" compares the measurement with the attempt's own recorded expectation or falsification rule.

| A | Parent | What the agent hypothesized (record) | What changed (code) | Measured | Verdict vs its prediction | Children |
|---|---|---|---|---|---|---|
| 1 | B | run is "optimizer-update-limited" | batch 2^19→2^18 | BPB 1.1791 (−0.021), 1.00×, 337 | supported | 2, 3 |
| 2 | 1 | depth 8→6 | depth 6 and a fallback break | invalid: `CandidateExitedBeforeStop` (its break raced the protected stop) | not measured | – |
| 3 | 1 | per-layer latency; BPB 1.172–1.185, ≥1.15× | depth 6, STOP_GUARD_S 10 | 1.1622, 1.33×, 289 | supported (BPB better than range) | 4 |
| 4 | 3 | depth 4: BPB 1.152–1.168, ~1.8–2.0× | depth 4 | 1.1273, 1.93×, 243 | supported (BPB better) | 5 |
| 5 | 4 | depth 3: BPB 1.115–1.135, 2.35–2.45× | depth 3 | 1.1177, 2.49×, 221 | supported | 6, 7 |
| 6 | 5 | depth 2: BPB 1.110–1.135, 3.2–3.5× | depth 2 | 1.1391, 3.64×, 192 | speed supported, quality falsified ("The quality prediction was falsified"; "do not reduce depth further") | – |
| 7 | 5 | second batch halving, BPB −0.004 to −0.015 | batch 2^18→2^17 | 1.0995 (−0.018), 2.47×, 221 | supported | 8 |
| 8 | 7 | eager adapter cuts memory ≥30 MB; "If this candidate is valid, CUDA-graph decode will be the next isolated speed change" | first `ChunkedKVInferenceAdapter`: chunked prefill logits, preallocated K/V | 1.0995, 2.57×, 169 (−52) | supported | 9, 10 |
| 9 | 8 | CUDA-graph decode | per-state graph capture | invalid: `cache_nbytes` omitted 43,024 B of graph I/O tensors | not measured | – |
| 10 | 8 | "Batch-1 cached decode in attempt 8 is launch-bound"; ≥2× | graph replay with corrected accounting | 1.1000, 8.62× (1,268 tok/s), **429** | speed supported; memory falsified (predicted 175–192 MB) | 11 |
| 11 | 10 | memory back to ~168–187 MB at equal speed | one persistent capture stream (shared cuBLAS workspace), autocast-matched prefill | 1.0998, 8.43×, 156 | supported | 12, 13 |
| 12 | 11 | third halving lowers BPB to 1.085–1.096 | batch 2^16 (device batch 32) | 1.1131 (+0.013) | falsified | – |
| 13 | 11 | weight and rotary storage, 120–140 MB, 9.0–11.5× | matrix weights stored in bf16 (`PrecisionLinear`), rotary table 10×→2× sequence length | 1.0998, 10.15×, 129 | supported | 14 |
| 14 | 13 | "bound by the number of serialized small kernels"; ≥2,100 tok/s | TorchInductor-compiled decode regions captured in the graph | 1.0998, 16.77× (2,490 tok/s), 129 | supported | 15, 16, 17 |
| 15 | 14 | sqrt-LR rule lowers BPB | matrix LR ×√2 | 1.1024 (+0.0026) | falsified | – |
| 16 | 14 | 2 KV heads: −≥8 MB, ≤+0.004 BPB | GQA | 1.1069 (+0.007), 108 | memory supported, quality falsified | – |
| 17 | 14 | short windows give 3–7% more tokens at +0.001 to +0.004 BPB | windows 1,024→256 | 1.0992 (−0.0007), +3.5% training tokens/s | supported (quality better) | 18 |
| 18 | 17 | ring K/V buffer for windowed layers: +5–12% decode | 272-slot ring cache | 1.0996, 24.74× (3,269 tok/s), 117 | speed exceeded prediction (speed-uncertain status) | 19 |
| 19 | 18 | split-key decode attention: 3,700–4,100 tok/s | custom decode attention in the graph | 1.0991, 29.57× (4,124 tok/s) | supported (speed-uncertain) | 20 |
| 20 | 19 | fixed q gain 1.35: BPB 1.0955–1.0985 | QK gain | 1.0969 (−0.0022), 27.57× | supported | 21 |
| 21 | 20 | last-row prefill logits: −28 to −38 MiB | prefill returns last logits | 1.0975, 113 (−3.9 MiB) | falsified (memory) | 22 |
| 22 | 21 | fusing eager glue: +30–70% decode | GEMV-bounded compiled regions | 1.0976, 32.00× (4,618 tok/s) | falsified by its own rule (CI upper bound <4,700 tok/s) | 23 |
| 23 | 22 | token-shift gate: BPB 1.0915–1.0955, falsified if >1.0962 | 13-parameter smear gate | 1.0962 (−0.0014), 30.84×, 113 | at its own falsification boundary (1.09620 ≤ 1.0962); gain smaller than predicted | 24, 25, 26 |
| 24 | 23 | shared VE table: memory 109.8–110.4 MB, BPB ≤+0.002 | one shared value-embedding table | 1.0995 (+0.0033), 31.40×, **97** | memory supported, quality slightly worse than range | – |
| 25 | 23 | attention output gate: BPB 1.0940–1.0958 | head-wise output gate | 1.1330 (+0.037), training 360k tok/s | falsified (dominated) | – |
| 26 | 23 | FA3 prefill: −28 to −36 MB | prefill via `flash_attn_func` | 1.0962, 111 (−2 MiB) | falsified (memory) | 27 |
| 27 | 26 | int8 gather tables: memory 103.5–104.4 MB | int8 embedding tables | 1.0962, 30.54× (4,580 tok/s), 99 (103.9 MB) | supported | – |

**The research arc in the records:**

1. **Training efficiency (A1–A7).** Opus cut the batch and then depth, one step at a time (8→6→4→3). Depth 2
   (A6) was faster but worse. Opus recorded the quality falsification, returned to A5 and halved the batch
   again (A7).
2. **Switch to inference (A8 onward).** The reason is recorded in its own words: A10's evidence review
   calls graph replay "the only lever that can raise speed without a quality cost". A8 already planned
   graph decode as "the next isolated speed change".
3. **Mechanisms (A10–A22).**
   - CUDA graph replay (A10);
   - storage changes (A11, A13);
   - compiled fusion (A14, A22);
   - ring K/V cache (A18);
   - custom decode attention (A19).
4. **Late quality edits (A17, A20, A23).** Small BPB gains from windows, QK gain and the token-shift gate.
5. **Frontier.** The last three edits produced the frontier:
   - A23 (best validation BPB);
   - A24 (least memory);
   - A27 (int8 tables, BPB tied with A23).

## D. Should this be central in the paper?

Yes, as one paragraph in the "How they searched" section, stated as observed decisions under equal access,
with the harness differences named.

- **Supported:**
  - GPT-5.6 had the same hook in both eras and never used it in 97 attempts.
  - Opus used it from A8 and reached ~4,600 decode tokens/s.
- **Not supported:**
  - that the model family caused the difference (one trajectory per arm; reasoning, timeout, isolation and
    repairs differ);
  - that GPT-5.6 "could not" find graph replay.
- Corrections applied to the manuscript:
  - "a revised harness that let GPT-5.6 and Claude Opus 5.5 also rewrite the inference code" → access was
    the same in both eras;
  - "train.py … and, later, the inference code" → the starting file never held inference code;
  - the appendix protocol paragraph → discovery used `cached_decode`, not stabilized_v2.
