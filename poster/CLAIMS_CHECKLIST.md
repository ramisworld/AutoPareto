# Poster claims checklist

## Version 7 (A1 portrait): every number, mapped to the manuscript

Same build-time checks as v6: `poster_data.load()` asserts every value against the paper's macros. v7 adds one
assertion, the ablation session's starting-model rate (149 tok/s, macro `CRIABaseDecode`). It also checks the
verified line's 31.6× and 1.8× against `CRSixDecXVsBaseRecipe` and `CRSixGptDecRatio`.

**Unchanged from v6** (see the v6 table below for sources):
- the header;
- the latency panel (880 / 192 ms, 7.3 + 873, 164 + 28, ~32×, 4.6×);
- the lineage milestones and the A10 quote;
- the stage speed-ups (1.0, 2.0, 11.6, 30.8×) and profiler counts;
- the Pareto data;
- the memory values;
- the limits and AI-use text.

**New or reworded in v7:**

| Poster element | Value | Paper | Source if not printed |
|---|---|---|---|
| Loop return: "each attempt reads all earlier ones and picks a parent to edit" | | §1.1, "Search loop" ("Each attempt reads all earlier ones, picks a parent…") | |
| Verified line: starting model, GPT A22, Opus A27 | 145, 266 (1.8×), 4,583 tok/s (31.6×) | Table 6; 1.8× §1.2; 31.6× App. E (`CRSixDecXVsBaseRecipe`) | |
| Time per decoded token, stages 1 to 4 | 6.71, 3.40, 0.58, 0.22 ms | derived: 1 ÷ (149 tok/s × stage speed-up); 149 tok/s is `CRIABaseDecode` | `cr_v2.json` phase5.ablation.baseline.decode_tps (149.06) and medians of `speedup` |
| Speed-up column: "= 6.71 ms ÷ time" | | definition of the speed-up above | |
| Stage 3: "same 290 kernels" | 290 | §1.5 (290 unfused kernels) | census: V2 and V4 both 290 |
| Ablation box: final engine | 30.8× | Fig. 4b, §1.5 | |
| Ablation box: fusion without graph replay; "the two interact" | 2.7× | Fig. 4b; §1.5 ("the effects interact") | |
| Ablation box: without custom attention, without ring K/V cache | 23.8×, 29.3× | Fig. 4b | |
| Ablation box: re-timed in the 13-recipe test | 31.4 to 32.1× | §1.2, §1.5 | |
| Equal-tokens block | AR-ref 1.099 "best in every seed", GPT A22 1.141, Opus A27 1.158 | Table 6; §1.4 (all 12 others worse in every seed) | |
| Equal-time block: BPB · tokens seen in 300 s | AR-ref 1.157 · 88M, GPT A22 1.155 · 158M, Opus A27 1.159 · 198M | Table 6; 88.0M and 158.1M in the E1 figure caption (full-statistics appendix) | 198M: unified_13 `fixed_tokens_mean` (197.9M) |
| Equal-time: "not resolved: 95% CIs of differences include 0" | Opus A27 −0.004 to +0.008; GPT A22 −0.020 to +0.016 | §1.4 | |
| Starting model at 300 s, off scale | 1.240 | Table 6 | |
| Subtitle: "faster-training recipes see more tokens in the same time" | | §1.4 (training speed sets tokens seen in 300 s) | |
| Memory line | 337, 243, 96 MiB | Table 6 | |

| QR code (header) | `https://openreview.net/forum?id=U9ICjFvtN8` | the paper's OpenReview page (URL given by the author) | `qr/openreview_forum_M.txt`; decoded back to exactly this URL by `qr/qr_decode.swift` at 36 to 300 dpi |
| Contact line | rameonix@gmail.com, rami.co.nz, @vicilah on X | | given by the author |

**Removed claims.**
- "Fusion may reduce intermediate memory traffic (not measured)" no longer appears, so v7 makes no claim about memory
  traffic.
- "Starting model 93M tokens" is no longer printed.

## Version 6 (A1 portrait): every number, mapped to the manuscript

**How the numbers are checked.** All values are read by `poster_data.load()` from the frozen records and asserted
against the paper's macros at build time.

**Columns.**
- **Paper:** where the final PDF prints the value.
- **Records only:** the paper does not print the value; the source file is named instead.

| Poster element | Value | Paper | Source if not printed |
|---|---|---|---|
| Headline: decode speed-up | ~32× (31.4 to 32.1×) | §1.2, Fig. 1 | |
| Headline: complete generation | 4.6× | §1.5, Fig. 4c | |
| Starting model: prefill + decode = total | 7.3 + 873 = 880 ms | §1.5, Fig. 4c | |
| Opus (mean of A23, A24, A27): prefill + decode = total | 164 + 28 = 192 ms | §1.5 (163 to 165, 28), Fig. 4c | |
| Prefill holds the CUDA graph capture; recorded, not scored | | §1.5, Table 14 (A11) | |
| Quality at 300 s: starting model, Opus A27, AR-ref | 1.240, 1.159, 1.157 | Table 6 | |
| Opus A27 not resolved from AR-ref | +0.002, 95% CI −0.004 to +0.008 | §1.4 | |
| Training budget, hardware | 300 s, one A40 | §1.1 | |
| Lineage (search records) | A4 1.9×, A10 8.6×, A14 16.8×, A18 24.7×, A19 29.6×, A22 32.0× | Fig. 2, Fig. 4a | |
| A18, A19, A22 speed-uncertain (open markers) | | Fig. 4a hatching | `cr_v2.json` quality_status |
| A10 hypothesis quote | "Batch-1 cached decode in attempt 8 is launch-bound" | paraphrased ("launch-bound", §1.3) | attempt-10 record, verbatim |
| Verified decode rates: starting model, GPT A22, Opus A27 | 145, 266, 4,583 tok/s | Table 6 | |
| Rate ÷ starting model | 1.8×, 31.6× | 1.8× vs AR-ref (§1.2; AR-ref = 145); 31.6× (App. E, mean of Opus) | unified_13 (A27 alone: 31.6) |
| Layers: starting model, GPT A22, Opus A27 | 8, 4, 3 | §1.1, §1.2, Fig. 2 | Opus A27 `DEPTH = 3` in its source |
| GPT-5.6 wrote no inference code | 0 of 75, 0 of 22 | Table 2 | |
| AutoResearch kept the model size | 50.3M | §1.2 ("kept the 50.3M-parameter model") | |
| Decoder stages: speed-ups | 1.0, 2.0, 11.6, 30.8× | Fig. 4b, §1.5 | |
| Decoder stages: GPU kernels per token | 625, 290, 290, 78 | §1.5 | |
| Decoder stages: CPU launch calls per token | 618, 284, 1 graph + 2, 1 graph + 2 | | `cr_v2.json` phase5.ablation.census |
| Stage 2 keeps Opus's own attention and ring cache, graph and fusion off | | §1.5 ("without both") | `evaluation/PHASE5_PREREG.md` (V4) |
| Fusion by TorchInductor | | | PHASE5_PREREG.md (V2: "no Inductor fusion"); Opus A23 source |
| Ablation bars | 30.8, 29.3, 23.8, 11.6, 2.7, 2.0× | Fig. 4b | |
| Independently verified band | 31.4 to 32.1× | §1.2 | |
| Tokens seen in 300 s: AR-ref, GPT A22 | 88M, 158M | §1.4 (88.0M, 158.1M) | |
| Tokens seen in 300 s: starting model, Opus A27 | 93M, 198M | | unified_13 `fixed_tokens_mean` |
| Held-out BPB after 300 s | 1.240, 1.157, 1.155, 1.159 | Table 6 | per-seed dots: unified_13 rows |
| Held-out BPB after 201.3M tokens | 1.144, 1.099, 1.141, 1.158 | Table 6 | per-seed dots: unified_13 rows |
| AR-ref best in every seed at equal tokens | | §1.4 | |
| 300 s leaders not resolved | 95% CIs include 0 | §1.4 | |
| Pareto panels | all 13 recipes, seed means | Fig. 3 (same data, same frontier rule) | |
| Memory: AR-ref, GPT A22, Opus A27 | 337, 243, 96 MiB | Table 6 | |

**Independent fact-check of v6.** A read-only agent checked about 95 statements and encodings and found all
numbers correct. Its 17 findings were all fixed:
- **Errors and overclaims:**
  - The objective box said "held-out" BPB; agents were scored on validation BPB. Now "bits per byte, unseen text".
  - Stage 2 was called "smaller model, eager", but it keeps Opus's attention and ring cache. It is now "Opus, graph
    and fusion off", in a neutral colour.
  - The headline now credits the smaller model too.
  - "AutoResearch kept its architecture" now reads "kept its size": its path includes an attention-window change.
- **Verified panel:** speed-up labels are now rate ratios that match the bars.
- **Headline panel:** now names its source (13-recipe test; Opus is the mean of three recipes).
- **Labels and keys:**
  - quality-strip label order matches the markers;
  - the lineage has a key for the line and dots, and its phase names say "mostly";
  - the lanes use one colour per meaning, with a four-item legend and "blocks schematic".
- **Removed:**
  - the 201.3M line from the 300 s token axis;
  - "back to back", which was not measured;
  - "seed spreads overlap", now replaced by the paper's 95% CI test.
- **Kept and documented:** values printed only in the records (CPU launch counts, 93M and 198M, TorchInductor, the
  A10 quote, "Sydney" from the author's brief).

## Version 5 (A1 landscape)

**Visuals and data.** Same visuals and data as version 4, re-flowed into three columns; the same `poster_data.load()`
assertions apply.

**New text, checked against the paper:**

| v5 text | Support |
|---|---|
| Figure 1 caption: the prefill "was recorded but not scored" | §1.5 (prefill recorded, not scored) |
| Figure 2 caption: "Grey: 8 other recipes (Baseline and GPT A24 lie off scale)" | computed from the plotted data |
| Figure 3 caption: "GPT-5.6 wrote no inference code in any campaign" | Table 2 (0 of 75, 0 of 22) |
| Take-home 1: "every agent found the same lever, smaller batches and models" | §1.2 wording |
| Take-home 2: 31.4 to 32.1× in 96 to 110 MiB, same greedy tokens up to BF16 near-ties | §1.2, §1.5 |
| Take-home 3: at 300 s "the leaders are not separated"; 32× is 4.6× for a whole generation | §1.4, §1.5 |

## Version 4 (A1 portrait)

v4 prints a subset of the values already checked for v1 and v3, from the same `poster_data.load()` assertions.

**New encodings, checked by an independent read-only agent:**
- **Pareto shading:** exactly the set dominated in quality and speed by the frontier of seed means. The dashed line is
  its boundary.
- **Contract arrows:** each runs from (tokens trained in 300 s, held-out BPB at 300 s) to (201.3M tokens, held-out BPB
  at 201.3M). Source: unified_13 `fixed_tokens_mean`.
- **CUDA lanes:** row lengths are proportional to 1/decode speed-up (ablation medians 2.0, 11.6, 30.8×). The blocks
  are schematic.

**Discovery chart:**
- the line follows Opus's recorded parent path, A1 to A23;
- milestone values come from the search records;
- open markers are speed-uncertain records.

**Verified strip and 2×2 grid:**
- the verified strip shows 13-recipe speed-ups. The GPT-5.6 range of 1.9 to 2.5× is from the records; the paper
  prints 1.9× from search records.
- the 2×2 grid arrows are ratios of the cell medians shown. Launch and kernel counts come from the profiler census of
  Opus A23; launch counts are in the records only.

**Fact-check findings, all fixed:**
- "At 300 s the recipes are not separated" now reads "the leaders": the paper resolves MOTPE and NSGA-II as worse.
- "Fusion removes the kernels" now reads "merges" (290 to 78 kernels).
- The hero answer now credits both the smaller model and the engine.
- The arrow gains are now ratios of the cells (×1.35, not ×1.39).
- "Record once" now reads "record per prompt": the graph is captured in every prefill.
- The fusion inset's memory claim, which nothing measured, is removed.
- "Diagnosis" now reads "hypothesis before A10".
- A key for open markers is added, and leader lines tie each label to its marker.
- The Pareto panels now state "seed means".
- The hero's decode swatch is now two-tone, matching both bars.
- "873 to 28 ms" under 32× is replaced by "scored: 31.4 to 32.1×".

## Version 3 (landscape, 85 × 61 cm)

All numbers come from `poster_data.load()`, with the same assertions as versions 1 and 2, plus these:
- κ equals the reciprocal of the paper's interaction macros (`CRIAInteractLo/Hi`, 0.51 to 0.52);
- the Opus per-attempt records reproduce the paper's 19 valid attempts after A7 and the 0.0033 gain;
- Table 2 of the paper contains "0 of 75", "0 of 22" and "19 of 25 valid";
- the A8 and A10 prefill times are 6.4 and 81.3 ms.

| Poster item | Source | In the paper? |
|---|---|---|
| BPB, the objective vector, Pareto dominance | §1.1 terms and definitions | yes (as formulas: poster) |
| Training contracts θ_T and θ_N, model and optimizer code unchanged | §1.4, App. E ("byte-identical") | yes |
| Bracketed speed-up ŝ (median of 8 window ratios to the interpolated baseline) | App. E, inference protocol | yes |
| Figure 2: 25 valid Opus attempts; validation BPB, decode and whole-generation speed-up, peak memory | `cr_v2.json` agents.opus.rows | Fig. 2, Fig. 4a, App. Fig. 7 |
| After A7, 19 valid attempts improve BPB by only 0.0033 | `CRThreeLateValid`, `CRThreeLateGain` | §1.3 |
| Figure 3: per-token time = 1/decode speed-up (ablation medians); counts from the census | `phase5.ablation` | speed-ups and 625/290/78 kernels: yes; CPU launch counts: records only |
| κ = S_{G+F}/(S_G S_F) ∈ [1.92, 1.96] | ablation speed-ups, labelled derived | the paper says "the effects interact"; κ is the reciprocal of its unprinted macro |
| Fusion alone 1.39×, on top of graph replay 2.66× (medians of per-recipe ratios) | `phase5.ablation` | records only |
| Removing split-key attention costs 1.29×, the ring K/V cache 1.05× | `CRIAAttnLo`, `CRIARingLo` | App. E |
| Table 1: 75 + 22 GPT-5.6 attempts, 0 own inference code; Opus 27 (25 valid), 19 of 25; fastest re-timed GPT A24, A25 2.5×, Opus 31.4 to 32.1× | Table 2 of the paper; unified_13 | GPT 2.5×: records only |
| Table 2: params, BPB ± SD at 201.3M tokens and 300 s, decode tok/s, memory | unified_13 summary | Table 6, Fig. 1 |
| T_gen = T_prefill + 127 t_step; S_gen = (7.3 + 873)/(164 + 28) = 4.6× | unified_13 means | §1.5 |
| lim S_gen = 880/164 = 5.4× (with this test's prefill) | arithmetic on the means, labelled derived | no |
| Same greedy tokens up to BF16 near-ties (99.8% of 39,936; 83 near-ties; 0 hard mismatches) | `CRSixAgreePct`, `CRSixNearTies` | §1.5; "not identical logits" (App. E) |

**Independent fact-check of version 3.** A read-only agent checked about 115 statements. It found 3 errors, 3
overclaims, 3 points supported only by the records and 6 style points. All are fixed in the current file:

- **Errors:**
  - "protocol of Figure 1" now reads "the bracketed protocol".
  - "adds a further 1.29×" now reads "removing ... costs 1.29×".
  - The fusion gains are now medians of per-recipe ratios: 1.39× and 2.66×.
  - v is no longer stated as 1,024/128 for every campaign.
- **Overclaims:**
  - "Exact up to BF16 rounding" now reads "same greedy tokens up to BF16 near-ties".
  - Finding 1 now uses the paper's "same lever" wording.
  - The Opus fastest-recipe cell now lists A23, A24 and A27.
- **Labelling:** κ and the Amdahl-style bound are now labelled "derived".
- **Style:**
  - "falsifiable" is removed.
  - Figure 2 says "25 valid attempts (A2 and A9 failed)" and "improves by only 0.0033".
  - Figure 3's rows are configurations ("fusion only", "graph only"), the graph rows show two CPU marks, and the
    caption says that the ticks and the GPU band are schematic.
  - Figure 4's frontier is called a two-objective frontier.
  - The prefill sentence now compares 164 ms with the baseline's 7.3 ms.
  - The contracts note that model and optimizer code stay unchanged.

## Version 2 (61 × 85 cm)

Version 2 prints a subset of version 1's checked numbers (the tables below), from the same `poster_data.load()`
assertions. Its new wording, checked against the paper:

| Version 2 text | Support | Status |
|---|---|---|
| "A 300 s training budget rewards speed, so the agents shrank batches and models." | §1.2: every agent found the same lever (smaller batches and smaller models); AutoResearch shrank only the batch, hence "the agents", not "every agent shrank the model" | matches |
| "its recipes decode 32× faster than the starting model, yet a whole generation is only 4.6× faster" | 31.4 to 32.1×; 4.6× (§1.5) | matches |
| "In 300 s, GPT A22's smaller model sees 1.8× more data, but learns less from each token." | 158.1M vs 88.0M tokens (1.80); worse at every budget and seed (E1) | matches |
| GPT-5.6 rail: A1 batch ¼; A4 to A8 fewer layers (8 to 4); A18 to A22 width, learning rates; GPT A22 4 layers, 29.4M; no inference code in any campaign | Fig. 2; §1.3; Table 2 | matches |
| Opus bars: A4 1.9×, A10 8.6×, A14 16.8×, A18 24.7×, A19 29.6×, A22 32.0× (hatched: speed-uncertain) | Fig. 2, Fig. 4a | matches |
| Memory: AR-ref 337, GPT A22 243, Opus A23 110, A24 105, A27 96 MiB | Fig. 1 | matches |
| "Only decode was scored. At batch one it is launch-bound: hundreds of tiny GPU kernels per token, each launched by the CPU." | §1.5 | matches |
| Grid: 284/290, 2/290, 75/78, 2/78 (launches/kernels); 2.0×, 11.6×, 2.7×, 30.8× | census and ablations (records; Fig. 4b) | matches |
| "Opus records its CUDA graph in the unscored prefill (7 to 164 ms)." | §1.5: 7.3 and 163 to 165 ms | matches |
| Footer: "each compared search ran once"; AI-use line as in version 1 | §1.3; disclosure | matches; AI-use line for the author to confirm |

## Version 1 (A1)

**How the numbers are checked.**
- **Sources:** every value is read by `poster_data.py` from the paper's frozen records:
  - `results/camera_ready_2026/phase6/analysis/unified_13.json`;
  - `paper/camera_ready/data/cr_v2.json`;
  - the attempt records, through `analysis/cr_r9.py`.
- **Paper match:** each printed value is asserted equal to the macro the paper prints (`paper/generated/camera_ready/*.tex`),
  so the build fails if a poster number drifts from the paper.
- **Quotes:** each quotation is asserted verbatim against its attempt record by the same `_q()` check the paper's
  Figure 2 uses.

**Paper locations** below are given as sections and figures of the final PDF:
`/Users/Rami/Documents/dev/autopareto_automlr_FINAL.pdf`, revision 14.

## Header and takeaway

| Poster | Value and source | Paper | Status |
|---|---|---|---|
| 32× faster decoding, decode steps only | Opus A23, A24, A27: 31.4 to 32.1× (`CRSixSpeedLo/Hi`, unified_13 `inf_speedup_vs_bracket_mean`) | §1.2, §1.5, Fig. 1 | matches |
| 4.6× faster complete generation | 880.06 / 192.19 ms = 4.58 (`CRSixEtoEXLo`) | §1.5, Fig. 4c | matches |
| 50.3M starting model, one A40, 1,024 in, 128 out | Setup; benchmark protocol | §1.1, Fig. 1, App. E | matches |
| "The 300 s budget shaped what the agents discovered"; GPT-5.6 smaller, faster architectures; Opus also rewrote the inference engine | every agent found the same lever under 300 s; GPT A22 4 layers, 29.4M; Opus "rewrote the inference engine" | §1.2 | matches (scoped to these searches) |

## 1 · The budget shapes the scientist

| Poster | Value and source | Paper | Status |
|---|---|---|---|
| Smaller batches: more updates; smaller models: more tokens | "Every agent found the same lever" | §1.2 | matches |
| Search loop: reads earlier attempts, picks a parent, records a hypothesis, edits train.py; protected 300 s training on one A40; three objectives; failures recorded; frontier = no other attempt at least as good on all three and better on one | Setup and search loop | §1.1, App. Fig. 5 | matches |
| AutoResearch: same loop, quality only | Eight approaches | §1.1 | matches |
| E1 curves at 88.6M, 157.3M, 201.3M tokens; seeds 6 to 8 | `cr_v2.json` `e1.curves` | App. Fig. 6 | matches |
| +0.042 at equal tokens; GPT A22 worse at every budget and seed | `e1.per_cell` (`CRSixGptTokD`) | §1.4, Fig. 6 caption | matches |
| 300 s: AR-ref 88.0M, GPT A22 158.1M tokens | `CREOneTokAR`, `CREOneTokAP` | §1.4 | matches |
| GPT A22 minus AR-ref at 300 s: −0.002 (95% CI −0.020 to +0.016), "not separated" | `CRSixGptTimeD/CI` | §1.4 | matches; no claim of equality |
| Quotes: AutoResearch #5, GPT-5.6 A1, GPT-5.6 A8, Opus A1 | verbatim in attempt records | Fig. 2 | asserted |
| Start 8 layers, 50.3M; GPT A22 4 layers, 29.4M, trains 1.81× as many tokens/s; Opus 3 layers, 22 to 26M | unified_13 params; `CRSixGptTrainRatio`; Opus sources `DEPTH = 3` | §1.1, §1.2, Fig. 1 | matches (3 layers: Fig. 2 A5 and sources) |

## 2 · One benchmark for all 13 recipes

| Poster | Value and source | Paper | Status |
|---|---|---|---|
| Pre-registered test on seeds 6 to 8, reusing E1 and Opus runs, 201.3M tokens and 300 s, timed on one A40 | Phase 6 plan | §1.4, App. E | matches |
| AR-ref learns most per token; at 300 s none resolved as better than AR-ref; no recipe wins everything | 12 others worse by 0.014 to 0.088 in every seed | §1.4 | matches |
| Scatter points, means, frontier of seed means, rings (a) and (b) | unified_13 rows and summary; same Pareto rule as the paper | Fig. 3 | same data and rule |
| Memory: 337, 337, 260, 289, 226, 337, 210, 243, 183, 221, 110, 105, 96 MiB | `inf_peak_allocated_mib_mean` | Fig. 1 | matches |
| BF16, batch 1, greedy, 1,024 in, 128 out; median of 8 bracketed windows | protocol | §1.4, App. E | matches |
| 11 of 39 timing flags; dropping them moves no mean by more than 5.4% and leaves the frontier unchanged | `CRSixUnstableRuns`, `CRSixStableShiftMax` | §1.4, App. E | matches |
| 99.8% of 39,936 tokens match, 83 BF16 near-ties, 0 hard mismatches | `CRSixAgreePct`, `CRSixTokensChecked`, `CRSixNearTies` | §1.5 | matches |

## 3 · Two agents, two discoveries

| Poster | Value and source | Paper | Status |
|---|---|---|---|
| Both shrank the model, about 1.9× faster decoding; only Opus then rebuilt the decoder | `CRNineGptDecX`; Opus A4 1.93× in search records | §1.3 | matches |
| GPT-5.6 seed 0, 25 attempts; Opus seed 3, 27 attempts | Fig. 2 subtitles | §1.1, Fig. 2 | matches |
| GPT: A1 batch ¼; A4 to A8 remove layers 8 to 4; A8 loses 0.061 BPB; A18 width 512; A19 to A22 learning rates; GPT A22 4 layers, 29.4M | attempt records (`CREightAPEightWorse`) | §1.3, Fig. 2 | matches |
| 0 of 75 attempts and 0 of 22 defined the generation hook | Table 2 (last row) | §1.3, App. D | matches |
| GPT-5.6 tied decode speed to model size | §1.3 wording | §1.3 | matches |
| Bars: search records, steps accumulate (each gain depends on the earlier ones), hatched = speed-uncertain | Fig. 4a | Fig. 4a | matches |
| Opus: A1 batch halved; A3 to A5 depth 6, 4, 3: 1.9× by A4, 2.5× at A5; A6 depth 2: BPB +0.021, quoted from A7's record ("the quality prediction was falsified" refers to A6) | Opus search records; asserted in `poster_data.py` | §1.3, Fig. 2 (A6 +0.021), Fig. 4a | matches |
| Opus A8 own inference code, memory −52 MiB | Fig. 2 | Fig. 2 | matches |
| A10 8.6×, A13 10.1×, A14 16.8×, A18 24.7×, A19 29.6×, A22 32.0× (search records; A18, A19, A22 speed-uncertain) | `cr_v2.json` Opus rows | Fig. 2, Fig. 4a | matches |
| Quotes: GPT A1, A4 (labelled A4), A18; Opus A1, A7 record, A10 hypothesis ("Batch-1 cached decode in attempt 8 is launch-bound", recorded as HYPOTHESIS) | verbatim in attempt records | Fig. 2; the A10 quote is verbatim in the attempt record; the paper paraphrases it ("launch-bound", §1.3 and FAQ) | asserted |
| Re-timed: GPT A22 1.9×; Opus 31.4 to 32.1× | unified_13 `inf_speedup_vs_bracket_mean` (GPT A22 1.931) | Opus: §1.2, Fig. 1. GPT A22: the paper prints 1.9× from search records and 1.8× against AR-ref in the test | GPT value from records |
| One campaign per agent, different harnesses: no ranking of models | | §1.1, §1.3 | matches |

## 4 · Why 32× decoding is 4.6× generation

| Poster | Value and source | Paper | Status |
|---|---|---|---|
| Only decode steps were scored; at batch one they are launch-bound | | §1.5 | matches |
| Starting model: 625 kernels, 618 CPU launches per token | profiler census `baseline` (`CRIAKernBase`; 618 = `CRIALaunchBase`) | §1.5 prints 625; 618 is in the census record | 618 from records |
| Unfused: 290 GPU kernels; fused: 78 | census (`CRIAKernEager`, `CRIAKernFused`) | §1.5 | matches |
| CPU launch calls: 284 (no graph, no fusion), 75 (fusion only), 2 with the graph (1 graph replay call) | census `host_kernel_launch_calls_per_token`, `host_graph_launch_calls_per_token` | records only | from records |
| Graph replay keeps the kernel count (78 with and without); it removes launches | census; asserted in `analysis/cr_v2.py` | §1.5 | matches |
| Speed-ups 2.0×, 2.7×, 11.6×, 30.8× (ablation session, medians of A23, A24, A27); "the two interact" | `phase5.ablation.speedup` | Fig. 4b, §1.5 ("the effects interact") | matches; no mechanism claimed |
| Split-key decode attention 1.29×, ring K/V cache 1.05× | `CRIAAttnLo`, `CRIARingLo` | App. E (Opus retraining and ablations) | matches |
| Fusion explanation: small kernels between matrix-vector multiplies, compiled with TorchInductor (helpers from A14, regions spanning all work between two multiplies from A22, attention included); K/V writes stay separate | Opus A23 source, `ChunkedKVInferenceAdapter` docstring and region comments | §1.5 (fusion merges kernels) | from source; schematic, labelled as such |
| Whole generation: baseline 880 ms (7.3 prefill + 873 decode), Opus 192 ms (164 prefill + 28 decode) | unified_13 `inf_prefill_ms_mean`, `inf_e2e_ms_mean` | §1.5, Fig. 4c | matches |
| Prefill rose because Opus records its CUDA graph there; A11 quote "Prefill time rose from 6.4 ms to 81 ms" | attempt record | §1.5, Table 14 | matches |

## Footer

| Poster | Paper | Status |
|---|---|---|
| Limits: three seeds; each compared search ran once; the 32× compares engines and models, measures speed not quality; correctness scope | §1.2, §1.3, §1.5, Part 2 limitations | matches |
| Records: frozen records, no re-runs; release pending | Part 2 reproducibility | matches |
| References: Karpathy autoresearch; AI-use line (Claude wrote and ran the camera-ready experiments under approved plans, the analysis, draft text and the poster; same family as the Opus agent, no role in its proposals) | references, disclosure statement | follows the disclosure; author to confirm |

## Independent check

A read-only agent checked about 120 numbers, 9 quotes and 40 claims against the paper, the records and Opus's source.
It found every headline number consistent. Its other findings, and what was done:

| # | Finding | Severity | Done |
|---|---|---|---|
| 1 | "A7 hits a quality limit" with the A7 quote: the quote is in A7's record but describes A6 (depth 2, +0.0215 BPB); A7 improved | error | Now "A6 depth 2: BPB +0.021", quote labelled "A7 record"; asserted in `poster_data.py` |
| 2 | "One campaign per agent": GPT-5.6 ran three original campaigns and a later one | error | Now "Each compared search ran once" |
| 3 | Mechanism for the graph and fusion interaction is not in the paper | overclaim | Caption now states the numbers only ("the two interact") |
| 4 | AI-use line understated Claude's role | overclaim | Expanded to match the paper's disclosure |
| 5 | "The budget shapes what autonomous research agents discover" is general | overclaim | Takeaway now "The 300 s budget shaped what the agents discovered"; the section title keeps the metaphor |
| 6 | GPT A22 re-timed 1.9× is from the record (1.931); the paper prints 1.9× from search records | unsupported | Kept, marked as from records; the data check no longer conflates the two |
| 7 | CPU launch counts (618, 284, 75, 2) are in the census record, not the paper | unsupported | Kept, marked as from records |
| 8 | A10 "launch-bound" sentence is a recorded hypothesis | unsupported | Prefixed "hypothesis:" |
| 9 | Fusion schematic omitted attention and K/V writes | unsupported | Attention region added; text names A14 helpers, A22 regions and eager K/V writes |
| 10 | A18 bar (1.45× step) vs ring-cache ablation (1.05×) | style | Bar caption: steps accumulate, so each gain depends on the earlier ones |
| 11 | "Every attempt kept" and the loose frontier definition | style | Now "Pareto archive", "failures recorded", and the paper's definition |
| 12 | "no recipe is resolved as better" | style | Now "better than AR-ref" |
| 13 | "rebuilt GPU execution" | style | Now "rewrote the inference engine" (paper wording) |
| 14 | "every speed change came from the architecture" | style | Removed; the paper's "tied decode speed to model size" kept |
| 15 | A4 quote under "A4 to A8" | style | Prefixed "A4:" |
| 16 | Footer said all comparisons use the 13-recipe test | style | Now names E1 and search records too |
| 17 | "Sydney" is not in the paper | unsupported | Kept: the venue is in the author's brief (ICC Sydney, 11 December 2026) |

**Two points about the paper itself (not changed; the paper is final):**
- §1.3 cites "the quality prediction was falsified" as A7. The sentence is in A7's record, but it describes A6.
- Appendix E (timing-stability flags) says the starting model "launches 625 kernels per decoded token from the CPU".
  The census records 625 GPU kernels and 618 CPU launch calls.
