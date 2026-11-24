> **Superseded by revisions 12 and 13**: see `REVIEWER_RISK_CHECKLIST.md` and the revision-12 entry of
> `CHANGELOG_CAMERA_READY.md`. In particular, the Opus ablation figure is back in Part 1 (Figure 4).

# Revision 11: one inference benchmark throughout

- **Status:** a draft, uncommitted. Nothing is submitted, pushed or published.
- **PDF:** `/Users/Rami/Documents/dev/autopareto_automlr_FINAL.pdf`, 28 pages, sha256 `0d71cf1b…`.
- **Checks:** full build (`paper/build_camera_ready.sh`) exit 0; pages 4/4/1; 0 undefined references;
  0 overfull boxes; no em dashes.
- **Untouched:** Part 2, Part 3 and the disclosure are byte-identical to revision 10. No run, measurement or
  generated value changed; the only new numbers come from a post hoc check on the existing 13-recipe records.
- **Details:** `CHANGELOG_CAMERA_READY.md` (revision 11) and `REVISION11.diff`.

## The author's brief, item by item

| Item | Done |
|---|---|
| 1. One benchmark | Old Figure 7 (256/256 vs 1,024/128) and old Table 8 (panel, 256/256 decode) removed. The abstract, Figure 2 caption and §1.4 no longer speak of each agent's "own benchmark". No text says a recipe lacks a 1,024/128 measurement. The 256/256 history is kept in a short appendix paragraph ("Earlier inference measurements"), the glossary and Table 2. |
| 2. Main-paper figures | Figure 1 unchanged in content. The scatter is now Part 1 Figure 3, with per-seed points and the frontier; its caption calls it a two-objective view and points to Figure 1 and Table 6 for memory. Labels enlarged and kept off the frontier line. |
| 3. Evidence kept | Lineages, Opus mechanisms, ablation figure (now Appendix Figure 6), matched-token and fixed-time comparisons, E1 figure (Appendix Figure 5), all 13-recipe tables, downstream and statistics tables, provenance and limitations. Part 1 now states E1's conclusion. |
| 4. Downstream | Part 1 summary: AR-ref leads BLiMP (narrowly) and WikiText perplexity; WikiText ranks recipes like held-out BPB; single-task leads are within seed SDs; five of seven further tasks are near chance. Full results stay in Tables 6 and 9. |
| 5. Inference reliability | Part 1 paragraph and rings in Figure 3 for the 11 runs outside the 2% drift rule. A new script check shows dropping them moves no recipe's mean by more than 5.4% and leaves the frontier of the means unchanged; the appendix gives the details. **The brief was cut off in item 5, so anything after "outside its" is not yet done.** |

## Judgement calls for you to check

1. **Opus figure moved to the appendix.** Adding the scatter pushed Part 1 to 5 pages. Moving the 32x/5x
   mechanism figure (now Appendix Figure 6) was the smallest change that kept Figure 1, the scatter and the
   lineage figure in Part 1. §1.6 keeps every number. The alternative is to shrink the lineage figure instead.
2. **Abstract:** "In a later campaign with its own benchmark" became "In a later campaign".
3. **Glossary (P2-01 block):** "Decode benchmarks" became "Inference benchmark", and the campaign entry lost its
   benchmark parentheses.
4. **Opus stars:** §1.3 now says the three Opus recipes are 0.3% apart in decode speed, so which of them is
   starred is not resolved.
5. **E1 in Part 1:** "the two effects roughly cancel" summarises the exploratory split; the split itself is in
   Appendix Figure 5c and the FAQ.

## Still open from revision 10

- Abstract edits of revision 10 ("1.8 times ... 28%", "4.6 times for a whole generation").
- Reading item (viii) of Part 2 in your own voice; the "original"/"extended" rename; the disclosure line.
- Stopping pod5; committing; the poster.
