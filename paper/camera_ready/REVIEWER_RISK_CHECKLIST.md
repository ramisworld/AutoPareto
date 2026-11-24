# Reviewer-risk checklist (revision 14, 2026-11-19)

- **PDF:** `/Users/Rami/Documents/dev/autopareto_automlr_FINAL.pdf`, 27 pages, sha256 `9eb21985…`.
- **Revision 14:** only §1.3 ("Why only Opus reached 32×") and §1.4 ("Speed measurements") reworded; see the
  changelog and `REVISION14.diff`. Risks B2, B3 and B5 below still apply as written.
- **Build:** full pipeline exit 0; pages 4/4/1; 0 overfull boxes; 0 undefined references; no em dashes.
- **Untouched:** Part 2's wording, Part 3 and the disclosure text are unchanged since revision 10. Revision 13 added one
  `\par` in §2.3 (line spacing only) and removed the page break before the references.
- **Changes:** `CHANGELOG_CAMERA_READY.md` (revisions 11 to 13), `REVISION12.diff` (revisions 11 and 12) and
  `REVISION13.diff` (this polish).
- **Revision 13 visual audit:** every page of the new PDF was rendered and inspected (see the changelog).

**How this list was made.** Two independent checks ran on the built paper; neither edited anything.

- **Claims audit:** a read-only agent checked Part 1, the appendix and the FAQ against `unified_13.json`, the raw inference
  records and the scripts.
- **Visual check:** a read-only agent inspected all 28 pages and measured font sizes from the PDF itself.

Every finding is listed below with what was done about it.

## A. Fixed in this revision

| Risk | Fix |
|---|---|
| FAQ said every ablation variant passed with identical greedy tokens; two runs (Opus A24, plain K/V cache) passed only after FP32 adjudication | Stated |
| Appendix said the 900 s timeout applied "from attempt 9"; the table, the FAQ and Part 2 say from A10 | Corrected to attempt 10 |
| Part 1 said "at 300 s quality differences are within seed noise", but four recipes are resolved worse | Removed; the four are named, with "only the baseline after adjustment" |
| Per-comparison intervals over 12 comparisons | Post hoc Bonferroni: 8 of 12 matched-token gaps survive, 1 of 4 at 300 s; stated in Part 1, Table 7 and the FAQ |
| FAQ claimed perfect argmax agreement, contradicting 83 near-ties | Rewritten; Part 1 and the appendix give 99.8%, 83 near-ties, 0 hard mismatches |
| "1,024-token ring cache" | Corrected: Opus's rings hold 256-token windows (verified in all three sources) |
| Opus's 32× could read as a better model | Part 1 says it is mostly the inference engine, the other ten recipes use the reference decoder, and it measures speed, not quality |
| The 32× and 4.6× use different references | Appendix names both: the registered baseline for speed-ups, the Baseline recipe for whole generation; dividing by the latter gives 31.6× |
| End-to-end 4.6× vs 5.3× from different sessions | Figure 4 panels name their sessions; the ablation 5.3× is explained by its 136 ms prefill; Table 12's caption says it is another session |
| "Seeds 6 to 8 on one A40" | Only the new runs and all timing were on one A40; reused E1 and Opus runs were trained elsewhere |
| "Within 0.002" drift control was 0.00204 | Now "within 0.0021" |
| FAQ counted E1 and the 13-recipe test as two independent 300 s results; they share runs | Corrected; the seeds 0–2 validation result (GPT A22 slightly better) is now cited |
| "Best recipe of every approach" | Now excepts GPT-5.6's later campaign (appendix) |
| Single-task leads "within seed SDs" were wrong for GPT A24 vs AR-ref on ARC-Easy (+1.9 points) | Part 1 now says margins over the runner-up (≤ 1.3 points) are within SDs; Table 6's caption states the ARC-Easy gap |
| "Best held-out BPB" for AR-ref | Now "at equal tokens" (GPT A22 has the best 300 s mean) |
| "Four campaigns" vs "three campaigns of 25" | Now "never used the hook in any campaign" |
| 11 timing flags | Exact rule, counts (10 drift, 1 CV-only), sources, time clustering, no asserted cause, stable-only and anchor checks; rings in Figure 3 |
| Reproduction instructions lacked the 13-recipe test | Added to the appendix how-to and `REPRODUCE.md` (flags checked against each parser; nothing executed) |
| Figure layout | Smaller figure padding; larger Figure 1 sub-headers (5.8 pt) and Figure 9 text (5.75 pt); Figure 3 markers no longer cut, off-scale baseline drawn as a square; Figure 4 labels separated; Figure 8 crossing line and tick crowding fixed; Table 15 caret; appendix how-to line spacing; gap before Table 3 |

## B. Open risks a reviewer could raise (disclosed, not fixable without new runs or author decisions)

1. **Three seeds.** Matched-token gaps are consistent (every recipe worse on all three seeds), but only 8 of 12
   survive Bonferroni. The 300 s comparison cannot show equivalence, and E1 does not resolve the gap's slope with tokens.
2. **Engine versus model.** Opus's speed comes mainly from its own decoder; the other ten recipes were timed on the
   protected reference decoder. Porting Opus's engine to them was not done, so the 32× compares engines and models
   together. This is disclosed in Part 1 and Part 2's limitations.
3. **Timing flags.** 11 of 39 inference runs are flagged and reported as measured. Frontier order among recipes
   below ~400 tok/s is not resolved. The cause is not identified, because no clock, CPU-load or profiler traces were
   recorded.
4. **Driver change mid-test.** The NVIDIA driver changed from 595.91.07 to 595.112.07 after a host restart. Anchors
   agree within 2.1% on speed-ups and 6% on absolute rates. Disclosed in the appendix and Table 3.
5. **Agents are not compared fairly.** GPT-5.6 and Opus each ran once, under different harnesses, timeouts and
   reasoning settings. The paper draws no ranking; a reviewer may still ask for replication.
6. **Selection and the winner's curse.** The full-size exp 109 candidate was never confirmed (Part 2). Panel selection
   came after scores were known (Table 10).
7. **Correctness scope.** The check covers greedy decoding, batch one and this 1,024/128 workload only. It establishes
   token agreement up to BF16 near-ties, not identical logits.
8. **Release pending.** Several reproducibility levels need files that are not public yet (Part 2 says so).
9. **Fixed-time runs used local disk (amendment A1).** Their storage differs from the matched-token runs. Disclosed.
10. **Downstream tasks.** Five of seven lm-eval tasks are near chance. GPT A24's ARC-Easy lead over AR-ref exceeds the
    seed SDs, a small counter-example to "AR-ref leads"; it is disclosed in Table 6.

## C. Needs the author (not changed: Part 2 or author-voice text)

1. **Part 2, P2-09** says the 13-recipe test adds "the best recipe of every other approach". GPT-5.6's later campaign
   was not retrained; suggested addition: "except GPT-5.6's later campaign".
2. **Part 2 layout.** Fixed in revision 13 (one `\par`, no wording change), as the polish brief asked for spacing
   fixes on every page. Revert it if you prefer Part 2's markup untouched.
3. **Abstract.** "Of its registered baseline" became "of the 50-million-parameter starting model", so it is true for
   both 32× and 4.6×. The revision-10 edits (1.8×, 28%, 4.6×) and revision 11's removal of "with its own benchmark"
   also still await confirmation.
4. **Glossary (P2-01 block).** The "Inference benchmark" entry was rewritten in revision 11.
5. **Item (viii) of Part 2, the original/extended naming, and the disclosure line** are carried over from revision 10.
6. **Two stale cross-references:** fixed on 2026-11-19 with the author's approval ("Appendices G and I use …",
   "Appendices I and J call …"); see the revision-13 changelog entry.

## D. Minor items left as they are

- The appendix is ragged-right by design.
- Some appendix pages end with white space, because the tables and figures are placed with `[H]`.
- Figure 6 (E1) ticks are about 5 pt.
- Figure 7's panels name the agents differently ("Claude Opus 5.5" vs "Opus").
- Table 4 is the only table with a side caption.
- Some revision-9 macros are now unused; this is harmless.
- The poster has not been updated.
