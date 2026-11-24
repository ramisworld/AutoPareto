# Revision 10: the 13-recipe test (Phase 6)

- **Status:** a draft, uncommitted. Nothing is submitted, pushed or published.
- **Where it is:**
  - PDF: `/Users/Rami/Documents/dev/autopareto_automlr_FINAL.pdf`, 28 pages, sha256 `e20dc08c…`
    (rebuilt 2026-11-18 with the Part 2 updates and the new Figure 1).
  - Build log: the full pipeline (`paper/build_camera_ready.sh`) exited 0.
  - Snapshot of the previous source: `camera_ready/source_before_revision10/main.tex`.
- **Checks:** pages 4/4/1, all within limits; 0 undefined references; 0 overfull boxes; no em
  dashes.

## Phase 6 (GPU work) is complete

- **When:** one A40 (pod5), 2026-11-16 01:09 to 2026-11-17 12:40 UTC.
- **Hours:** 35.52 pod-hours, inside the author's cap of 37.5 h. About 12 h of that was idle, across
  four incidents (`evaluation/PHASE6_BUDGET_RECOMPUTE.md`).
- **Every planned measurement was recorded:**

| Measurement | Count |
|---|---|
| Native inference runs (1,024 in, 128 out) | 39 |
| Inference controls | 19 |
| Anchors | 3 |
| Matched-token cells (201.3M tokens) | 39, plus 3 drift controls |
| 300 s cells | 39, plus 1 drift control |
| Frozen-downstream checkpoints | 39 |
| lm-eval checkpoints | 39 |

- **Amendments.** A1 to A4, each approved by the author before it ran (`evaluation/PHASE6_AMENDMENT_A*.md`).
- **Incidents.** 1 to 7. None changed a measurement (`evaluation/PHASE6_INCIDENTS.md`).
- **Copy on the Mac.** Every record and log is in `results/camera_ready_2026/phase6/`, checksums
  identical to the pod's. Pod-local console logs are in `console_logs/`.
  - Not copied: checkpoints and compile caches, which stay on the network volume.
- **The pod is idle but still running and billing.** Stopping it is your call, because it is a cloud
  infrastructure action.

## Results now in the paper

| Finding | Detail |
|---|---|
| Matched tokens | AR-ref has the best held-out BPB (1.099). All 12 others are worse: +0.014 to +0.088, every 95% CI above 0, worse on all 3 seeds. |
| 300 s | No recipe is resolved as better than AR-ref (GPT A22 −0.002 [−0.020, +0.016]). 4 are resolved as worse. |
| Frontier, matched tokens | AR-ref, TPE, GPT A22, Opus A27 and Opus A24 are on it on every seed. |
| Frontier, 300 s | The quality differences are within seed noise, so speed sets the frontier. |
| Decode | Opus 4,579 to 4,591 tok/s, 31.4x to 32.1x. GPT A22 1.8x AR-ref with 28% less memory. |
| Whole generation (32x vs ~5x) | Opus records its CUDA graph in the untimed prefill: 146 to 167 ms against 7 ms. A whole 1,024+128 generation is 4.6x here and 5.3x in the 5B session, whose prefill was 136 ms. |
| Correctness | Every timed token, positions 1,024 to 1,151 past the ring-cache wrap, matched the uncached model: 0 hard mismatches in 39 runs. This closes the old "correctness does not cover the timed window" caveat. |
| Downstream | AR-ref leads BLiMP. Single-task leads are within seed SD. Of the lm-eval tasks, 5 sit near chance; WikiText perplexity ranks recipes like held-out BPB (Spearman 0.98). |

## What changed in the paper

**Part 1** (fits on 4 pages):
- **Figure 1** (`cr_compare13`, changed 2026-11-18 at the author's request): the revision-9 row layout,
  now filled entirely from the 13-recipe test.
  - **Rows:** all 13 recipes, seeds 6 to 8 on one A40, so there is no "not run" note and no seed
    mix.
  - **Columns:** held-out BPB after 201.3M tokens and after 300 s (one shared axis), training speed
    (300 s runs), decode speed and peak memory.
  - **Stars:** recipes on the BPB–decode Pareto frontier in every seed under that budget.
  - **Removed:** the unexplained bold names.
- **Moved to the appendix:** the 13-recipe table (`table_unified`, now Table 6), whose frontier
  marks now match the stars, and the scatter `cr_unified` (Figure 5), whose captions name the
  recipes on the frontier of the means but not on it in every seed.
- **Training-speed ratios in §1.3 and the FAQ** now come from the 300 s runs (`\CRSixGptTrainRatio`
  = 1.81, unchanged; Opus "about 2.3×", was 2.27).
- **§1.3, §1.5 and §1.6** are rewritten from the 13-recipe data, and the old E1 compact figure and
  seed-group table were removed. Every number comes from generated macros (`cr_macros_p6.tex`).
- "Original" and "extended" no longer appear in Part 1.

**Abstract** (your text; **please confirm or revert both edits**):
- "Nearly twice as fast ... 35% less memory" became "1.8 times as fast ... 28% less inference
  memory". These are the 1,024/128 benchmark's values.
- Added: "or 4.6 times for a whole generation". The abstract is now 244 words.

**Appendix:**
- New paragraph, "The 13-recipe test", in the reproducibility statement. It gives:
  - the plan hash;
  - the drift controls;
  - the four amendments;
  - the driver change after the pod restart, with the anchor evidence: speed-ups within 2.1%;
  - the one unreliable training-throughput run.
- New Table `tab:paired6`: paired differences against AR-ref for both contracts. It notes the
  intervals are per comparison and not adjusted for the 12 comparisons.
- New Table `tab:lmeval`: lm-eval results with rank correlations and the BLiMP template explanation
  (amendment A4).
- Inference protocol: discloses the 11 of 39 runs flagged unstable. Those differ from their
  recipe's stable seeds by up to 10%, only among recipes below 200 tok/s. The paragraph also covers
  the pre-registered controls.
- The environment table gains three 13-recipe rows, with a dagger for the second driver.
- Glossary, approaches table, per-seed caption and scatter caption no longer say these recipes were
  "never retrained" or "never measured".
- Reviewer questions: four answers are updated to the 13-recipe evidence, and one new question
  covers whether 201.3M tokens favours larger models.
- **Disclosure** (a humanauthor block): the AI assistant's role now includes running the 13-recipe
  test. This is factual and **flagged for your review**.

**Build:** `analysis/cr_p6.py` and `analysis/cr_p6_paper.py` were added to `build_camera_ready.sh`,
and `cr_v2.py` and `cr_v3.py` were extended.

## Waiting on you

1. **Abstract:** keep or revert the two edits above.
2. **Part 2:** approved and applied on 2026-11-18. To keep 4 pages, items 3 and 4 were shortened
   and item 1 was left out, because Table 4 records it. Please read the final wording of item 3,
   the new intervention (viii), in your own voice. Outcomes are in
   `camera_ready/PART2_REVISION10_PROPOSED.md`.
3. **"Original" and "extended" naming:** it remains in Part 2 and the appendix, defined in the
   glossary, so the paper is consistent. The same file has a one-pass rename proposal.
4. **Pod5:** stop it once you are satisfied.
5. **Commit:** everything is uncommitted on branch `autopareto/camera-ready-opus` in the
   camera-ready worktree.

## Not done

- The poster is not updated.
- The markers-on layout was not checked.
- Some revision-9 macros are now unused. They are harmless.
