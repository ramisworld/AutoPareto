# Part 2: proposed factual updates after the 13-recipe test (revision 10)

**Status (2026-11-18):** the author approved all seven ("Apply them"). They are applied, except
that Part 2 must stay within 4 pages, so items 1, 3 and 4 had to be fitted:

| # | Outcome | Text in `main.tex` |
|---|---|---|
| 1 | **Not applied.** The Tools paragraph is exactly full; one more word pushes the next paragraph to page 6 and Part 2 to 5 pages. Table 4, which the paragraph cites, already has the 13-recipe training, inference and lm-eval rows. | unchanged from revision 6 |
| 2 | Applied as proposed. | "the recipes for the panel, E1, the Opus tests and the 13-recipe test" |
| 3 | **Applied, shortened.** The amendment details are in Appendix E's "13-recipe test" paragraph. The proposed "changed execution, not measurement" was replaced, because A1 also added a stall gate and fixed the lm-eval task list. | "(viii) 13-recipe test. Its four amendments moved storage to local disk and added a stall gate (A1), re-ran one failed run (A2), supplied a database setting (A3) and narrowed the lm-eval cross-check (A4). I approved each before it ran, and none changed the frozen training or evaluation code (Appendix E)." |
| 4 | **Applied, shortened** to fit the last line of the Statistics item. | "The 13-recipe analysis reads only run records and finds no value missing." |
| 5 | Applied as proposed. | as proposed |
| 6 | Applied as proposed. | as proposed |
| 7 | Applied as proposed. | as proposed |

- **Spacing:** `\looseness=-1` was added to Part 2's paragraphs. This changes line breaking only,
  not wording, and Part 1 already uses it.
- **Appendix consistency:** A1's description in Appendix E now also says it "fixed the lm-eval task
  list before any lm-eval run".
- **Comments:** each edited block carries a `% Revision 10` comment recording the approval.
- **Please read item 3 in the author's voice:** it says "I approved each before it ran".

The original proposals follow, unchanged. Each fixes a statement that the 13-recipe test (Phase 6)
has made incomplete or out of date, and changes nothing else. Line numbers refer to `paper/main.tex`
as built for revision 10 before these edits.

1. **P2-03, software (line 148)**
   - Now: "The matched-token verification (201.3M-token panel, E1, Opus retraining) used Python
     3.10.18 with PyTorch 2.9.1."
   - Proposed: "The matched-token verification (201.3M-token panel, E1, Opus retraining, 13-recipe
     test) used Python 3.10.18 with PyTorch 2.9.1; the 13-recipe test timed inference with PyTorch
     2.8.0 and scored its further tasks with lm-evaluation-harness 0.4.13."
   - Source: `run_metadata.json` and the inference records in
     `results/camera_ready_2026/phase6/`.

2. **P2-06, author's role (line 167)**
   - Now: "the recipes for the panel, E1 and the Opus tests"
   - Proposed: "the recipes for the panel, E1, the Opus tests and the 13-recipe test"

3. **Interventions: a new item (viii), after (vii) Opus retraining (line 181)**
   - Proposed: "**(viii) 13-recipe test.** Four documented amendments changed execution, not
     measurement: local disk for the 300-second runs after network-storage stalls (A1), one
     re-run after a stale network file handle (A2), a database setting the frozen downstream runner
     omitted (A3), and an lm-eval cross-check restricted to the three tasks whose template matches
     the frozen evaluator (A4). I approved each before it ran (Appendix~\ref{app:repro})."
   - Basis: the approvals are recorded in `evaluation/PHASE6_AMENDMENT_A1..A4.md` and
     `PHASE6_INCIDENTS.md`. **Please check that the last sentence is true in your own words before
     approving.**

4. **Verification (line 193)**
   - Append: "The 13-recipe test's analysis (`analysis/cr_p6.py`) reads only its run records and
     reports any missing value as missing; none is missing."

5. **Significance (line 197)**
   - Now: "E1 tests GPT A22, not the best possible AutoPareto recipe, and our conclusions do not
     extend to unconfirmed recipes."
   - Proposed: "E1 tests GPT A22, not the best possible AutoPareto recipe. The 13-recipe test adds
     GPT A24 and A25 and the best recipe of every other approach, at one budget per contract. Our
     conclusions do not extend to unconfirmed recipes."

6. **Reproducibility (line 202)**
   - Now: "retrain GPT A22, AR-ref and Opus's recipes under the recorded token or time contract."
   - Proposed: "retrain all 13 recipes under the recorded token or time contract."

7. **Limitations, P2-14 (line 211)**
   - Now: "The Opus retraining covers one budget and replaces time-fraction learning-rate progress with
     the token fraction."
   - Proposed: "The Opus retraining covers one token budget (the 13-recipe test adds 300 s) and
     replaces time-fraction learning-rate progress with the token fraction."

## The "original" and "extended" naming

- **Where it remains.**
  - Part 2 uses "original" and "extended" 15 times.
  - The appendix uses them about 22 times, and the glossary defines both.
  - Part 1 no longer uses either term.
- **Why the rest is left alone.** The paper stays consistent as it is, because both terms are
  defined in the glossary.
- **If the framing should go everywhere,** a single rename would do it:
  - "original experiments" → "the first campaigns (July–August 2026)";
  - "extended experiments" → "the two-agent campaigns (September–November 2026)".
- **What applying it involves.** It touches Part 2, so it needs approval. I would then apply it
  across the appendix, the generated tables (`cr_v3.py` "Phase, seed" column) and the glossary in
  one pass.
