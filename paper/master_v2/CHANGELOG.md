# Master v2 changelog

Base: the AutoMLR workshop submission, backed up unmodified under `paper_backups/submitted-automlr-20260907-020829/`. The submitted PDF and source were not overwritten.

## Scientific content

- **Surfaced the mechanism as a measured quantity.** Re-auditing the archive showed that the seven verified recipes consumed between 88.9M and 251.8M tokens under the 300-second contract, a 2.83x spread, and that AP292 beat the AutoResearch reference by 0.0017 validation BPB while training on 1.80x as many tokens. The workshop version argued the confound abstractly; the master version measures it and builds the paper on it. New columns in Table 1, new Figure 3 panel A, new paragraph "Exposure was not held fixed".
- **Restored the complete downstream table.** All 7 verified recipes against 8 distinct metrics, with a seed-SD companion table in the appendix. The workshop version had compressed this into prose. No "Favored" column and no win counts; the interpretation separates the four task-accuracy metrics (differences within seed variation) from BLiMP (separated, and tracking held-out BPB at r = -0.969).
- **Held-out BPB retained** as the quality metric everywhere, with discovery validation BPB shown alongside it rather than substituted for it.
- **Corrected the causal overclaim** in the significance discussion. The phrasing about the contract paying for one part and the edit earning the other was removed. The manuscript now states that verification changed exposure, seeds, and evaluation data together, so the reversal is not attributed to exposure alone, and that what the experiment establishes is the failure of the fixed-time ranking to survive an exposure-controlled retest.
- **Artifact availability corrected.** A new table separates preserved from released artifacts. Checkpoints and raw logs are stated as preserved but not released; the claim that artifacts are available "through the submission system" was removed.
- **Related work rewritten** around the actual gap (auditable source-edit trajectory plus resource-sensitive contract plus post-search matched-exposure verification), with the concession that if such verification exists in prior work the contribution reduces to the measurement.

## Structure

- Replaced the AutoMLR three-part template with a venue-neutral structure: Introduction, research system, autonomous search, matched-exposure verification, downstream evaluation, resource contracts as an experimental variable, auditability and reproducibility, related work, discussion and broader impact, conclusion, appendices.
- Introduction now ends with three contributions rather than a scattered list.
- Broader impact folded into the discussion, as is conventional outside the workshop template. Any AutoMLR-formatted derivative should restore it as a separate section.
- Reproducibility promoted from appendix filler to its own section with an artifact-status table.

## Presentation

- **Font fix.** The workshop PDF rendered Section 1.4 at `\footnotesize` because prose had been shrunk to hold a page limit. Every prose paragraph in the master version is at normal body size; the only size directives left apply to table bodies.
- **Figure 2 legibility fixed at the root cause.** Labels overflowed the axes, so `bbox_inches='tight'` grew the canvas to 10 inches and `width=\linewidth` scaled the fonts down to roughly 55%. Now saved at its declared canvas with a wider label panel and 7 pt labels.
- **New Figure 3**, discovery versus verification, with exposure encoded in marker size and annotation.
- **Removed the budget trajectory figure**, which implied sequential ancestry the records contradict.
- **Pagination cleaned.** Tables no longer float ahead of their sections, and the near-empty conclusion page in an intermediate build was removed by tightening prose rather than by shrinking type.

## Abstract

Rewritten around one idea: the budget becomes part of the objective. Opens on the mechanism, states the measured exposure spread, gives the reversal and the surviving efficiency result, notes the registered contract study as exploratory, and states what is preserved rather than overclaiming release.

## New artifacts

- `paper/master_v2_fact_ledger.md`
- `paper/master_v2_visual_audit.md`
- `paper/NEXT_EXPERIMENT_PLAN.md`
- `analysis/master_v2/build_master.py`

## Revision 2 (human-debugging pass)

- **New Section 5, "Instrumentation: what had to be repaired."** Documents the two human debugging efforts the measurements depend on, with the exact numbers from `docs/RESEARCH_LEDGER.md` sections 20.2 to 20.7 and `study/contracts_2026/provenance/a40_inference_diagnosis.md`:
  - The untimed warm-up leak. The original launcher ran the first eleven optimizer updates as real training but excluded them from the timed interval, donating 11 x 524,288 = 5,767,168 tokens per candidate, scaled by whatever batch the agent chose. Historical baseline 92,798,976 tokens / 177 updates; later calibration 93,847,552 / 179.
  - The cold-start regression introduced by the first repair. Separate-process warm-up correctly discarded learned state but destroyed the CUDA context, TorchInductor kernels, and allocator warmth, so the first scientific update cost 20.946 / 20.809 / 25.674 s inside the protected 300 s and exposure fell to 82,837,504 / 82,837,504 / 81,264,640 tokens (158 / 158 / 155 updates).
  - How it was caught: arithmetic, not instrumentation. 179 - 11 = 168 updates = 88,080,384 tokens was the predicted post-warm-up exposure; the measurement said 158 updates and 82.8M. The 10-update, 5,242,880-token shortfall matches 10 x 1.783 s = 17.8 s, inside the observed 20.8 to 25.7 s first-update cost.
  - The final same-worker lifecycle with hash-verified state reconstruction (initial-state hash b2db8781), first scientific update 1.711 s against a 1.783 s steady rate, and the recalibration 88,604,672 / 88,080,384 / 88,604,672 that fixes N* = 88,604,672 = 169 x 524,288. The predicted 168 and measured 169 differ by one update.
- **New Section 5.2, "Measuring cached decoding."** Why the correctness gate exists at all, why the reference must be protected SDPA rather than the candidate's own FlashAttention-3 forward, why BF16 (8-bit mantissa, relative resolution 3.9e-3) legitimately fails a 0.03 mean-logit screen at 0.04805 with a 0.15288 peak at step 14 while all greedy tokens agree, why the threshold was not loosened, and how float32 adjudication (24-bit mantissa, bounds 1e-5 mean / 1e-4 max, observed 2.18e-6) settles it. Includes the FlashAttention integration bug: float32 Q/K/V routed into a kernel that supports only fp16, bf16, and fp8_e4m3, fixed by keeping FlashAttention for official BF16 timing and routing float32 adjudication through SDPA. Also states the residual risk honestly: five of six fresh diagnostic sessions on the current paired study were rejected before producing a usable scalar, which is why that study promotes no throughput value.
- **Agent hypotheses quoted verbatim** in Section 3. The A8 hypothesis explicitly cites the exposure incentive ("because the smaller model processes more tokens in the fixed training time"), which is direct evidence that the contract was legible to the agent and entered its stated reasoning. A18 and A22 are quoted to show the compress, measure, restore-width, tune sequence.
- **Downstream analysis deepened to the architecture level.** AP292 (4 blocks, SSSL, one full-context layer in four) against AutoResearch (8 blocks, SSSS, no full-context layer): receptive field runs the wrong way to explain the BLiMP gap, and BLiMP pairs sit inside the half-context window, so the operative difference is sequential composition budget. The phenomena where AP292 loses (distractor agreement, NPI scope and licensing, filler-gap, ellipsis) are the ones needing at least two dependent relational hops; ARC-Easy and PIQA lean on lexical and world-knowledge association carried by the width-512 MLPs and the embeddings, which A18 restored. Flagged as a mechanistic reading consistent with the data, not an established cause, because depth and held-out BPB are entangled in this panel.
- **KV closed form added.** 255 x 2 x 2 x L x h_kv x d bytes reproduces the measured incremental KV column for all seven recipes exactly (AP292: 2,088,960 bytes = 1.9922 MiB), which validates the memory instrument rather than merely reporting it.
- **Duplication removed.** The held-out BPB column was repeated in Tables 1 and 2 and is now only in Table 1. The introduction restated Section 4's efficiency numbers verbatim and no longer does. The appendix inference paragraph duplicated Section 5.2 and now only carries the workload specifics. Redundant restatements in the conclusion and discussion were cut.
- **Layout.** The two instrumentation sections were merged under one numbered heading, and the appendix `\clearpage` was removed, eliminating a near-empty page that held only the conclusion.
- **Abstract** now closes on reproducibility, naming what is preserved (hypotheses, patches, lineage, seeds, recipe and checkpoint hashes, contracts, raw per-repetition measurements) separately from what is released (recipe sources, manifests, per-seed results).
