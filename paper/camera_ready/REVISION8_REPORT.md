# Revision 8: final verification report (2026-11-15)

Final PDF: `/Users/Rami/Documents/dev/autopareto_automlr_FINAL.pdf`, 25 pages. Main paper: Part 1 is 4 pages, Part 2 is 4 pages and Part 3 is 1 page. No experiment, GPU job or agent call was run, and no record was changed. Before writing, every file in the evidence tree was checked: none had been modified in the previous 5 hours.

## 1. Inference-editing investigation (full audit: `audit/camera_ready_2026/26_INFERENCE_EDITING_AUDIT.md`)

- **Same editable surface.** Every AutoPareto agent could write only `train.py`. In every AutoPareto campaign, original and extended, that file could define `build_inference_adapter(model)`. The protected benchmark then timed the candidate's own `prefill` and `decode_one` inside its own loop.
  - The hook has existed in protected code since `75af7b7` (2026-07-24).
  - The original program text lists "a candidate cache adapter that satisfies the protected interface" (`8548308:programs/program_autopareto.md`, lines 38–39). The extended program says "an inference adapter compatible with the protected interface".
  - Neither program mentions CUDA graphs, compilation, kernels or quantization.
  - Neither starting `train.py` contained any generation code.
- **Different decisions.**
  - **GPT-5.6:** used the hook in 0 of 75 original attempts and 0 of 22 extended attempts. Its edits sat in the hyperparameter block, the KV-head setting or the optimizer. It never used the "systems" category, and in the extended run it called decode speed "architecture-dependent" (A12). Its extended speed-up came from smaller models only, at most 1.58×. That run stopped when the Codex sandbox failed, not by any decision.
  - **Opus:** wrote its first adapter at A8 and announced graph decode as its next step. A9 failed an accounting check. A10 introduced graph replay. 19 of its 25 valid sources define the hook.
- **Harness differences between the extended arms:**
  - reasoning setting: high vs xhigh;
  - environment: container with a shell vs host with file tools only;
  - call timeout: 600 s vs 900 s from A10, raised after Opus's A9 calls timed out while it was writing an adapter;
  - evaluator: an FP32-check fix after A8's first run crashed;
  - protected-adapter repairs for GPT-5.6's architectures;
  - attempts: 22 vs 27;
  - GPU instance.
- **The original harness also made inference work costly.** A failed cache check counted as a dominated crash and paused the whole campaign.
- **Conclusion:** search decisions, made under harness conditions that differed in several recorded ways. Not a different editable surface. With one trajectory per arm, the difference is not attributed to the model family. The paper says this in Section 1.4 ("Same access, different strategies") and in the new Appendix D, with Table 3.
- **Corrections this exposed:**
  - Original discovery was timed by `cached_decode` (2 warm-ups, 10 repetitions), not by `stabilized_v2`.
  - The claim that only the extended harness allowed inference edits was wrong and is removed.

## 2. Figures and tables

| Item | Status | Content |
|---|---|---|
| Figure 1 `cr_overview` | new, replaces the revision-7 comparison | Original block (panel, seeds 3–5): Baseline, AR-ref, TPE, MOTPE, AP-Q. Extended block (retraining, seeds 6–8; fresh decode re-measurement): Baseline, AR-ref, AP-Q, Opus A23, A24, A27. Columns: held-out BPB, training throughput, decode throughput, memory. Decode and memory have a separate axis per block. |
| Figure 2 `cr_tracks` | new, replaces the Opus milestone figure | Four decision logs: AutoResearch, GPT-5.6 original (AP-Q), GPT-5.6 extended, Opus. One row per attempt, recorded parent links only. Real attempt IDs, no milestone numbering. |
| Table 1 `table_verify_main` | new | Held-out BPB, ARC-Easy, PIQA, LAMBADA and BLiMP on shared seeds 6–8 for AR-ref, AP-Q, A23, A24 and A27, with bold marking the best. |
| Figure 3 `cr_inference` | kept, retitled | Ablation session named in the panel titles. Text separates 30.8× (ablation session) from 30.9–31.8× (primary re-measurement). |
| Appendix | Table 3 (access, new), Figure 9 (Opus lineage relabelled), Table 8 (bold per seed group), FAQ | |

## 3. What is highlighted, and from which protocol

- **AP-Q vs AR-ref (panel, seeds 3–5, decode 256 in / 256 out):**
  - 1.81× training throughput;
  - 1.99× decode throughput (383 vs 193 tok/s);
  - 35% less inference memory;
  - 29.4M vs 50.3M parameters.
- **Opus frontier (fresh A40 re-measurement, 1,024 in / 128 out, raw repeat records):**

  | Recipe | Decode (tok/s) | Speed-up | Note |
  |---|---|---|---|
  | A23 | 4,570 | 31.8× | largest speed-up |
  | A24 | 4,578 | 30.9× | least memory, 97 MiB, 22.0M parameters |
  | A27 | 4,592 | 31.5× | highest absolute rate |

  The bracketing baseline ran at 135–151 tok/s. A full generation is 5.3× faster.
- **Matched tokens:** AR-ref is best. AP-Q trails by 0.038–0.042 and Opus by 0.059–0.061 held-out BPB.
- **Equal time:** AP-Q sees 1.80× the tokens of AR-ref, and the difference is not resolved. "Not resolved" is stated as not evidence of equivalence.
- **Downstream (seeds 6–8):**
  - AR-ref leads held-out BPB and BLiMP;
  - AP-Q and A27 share the best ARC-Easy mean;
  - A23 has the best PIQA and A27 the best LAMBADA mean.

  These are observed means. Seed SDs reach 2.3 points.
- **Benchmarks are never mixed.** Raw throughputs from the two decode benchmarks never share an axis, and the text says AP-Q's 383 and A23's 4,570 form no ratio.

## 4. Part 2 and abstract (approved by the author, 2026-11-15: "Approve both")

- The Part 2 text is unchanged: the source is identical to revision 7.
- One glossary correction (P2-01): the phase entry dropped "may also edit inference code".
- Abstract: one word-level accuracy fix. "in later campaigns, its inference code" became "in a later campaign, its inference code", because only the Opus campaign edited inference code.

## 5. Checks

- 4/4/1;
- 0 em dashes;
- no bare "31×";
- no AP-23 or AP-24;
- no placeholders or TODOs;
- no undefined references or citations;
- no overfull boxes;
- "Pair-A" appears once in Part 1, as its definition;
- every Part 1 page ends at a paragraph boundary, so no float interrupts a sentence;
- author block and disclosure unchanged;
- repository link present and described as "to be released".

Every Figure 2 label, every quoted hypothesis and every program quote is asserted against its record by `analysis/cr_r8.py`. Every plotted value is written with its source to `data/cr_overview.json` and `data/cr_tracks.json`.

## 6. Limitations that remain

- Each extended arm ran once, so the strategy contrast is not replicated.
- The correctness checks cover neither the timed decode window nor the 1,024-token ring cache.
- Random search, NSGA-II and CMA-ES were never retrained.
- One GPU type; recipes of 18–50M parameters.
- Downstream accuracy differences are within seed noise.
