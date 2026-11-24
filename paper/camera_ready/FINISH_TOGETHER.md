# Finish together: what only you can do

Updated 2026-11-15 after camera-ready revision 9 (final visual presentation): one candidate comparison (Figure 1), three research histories (Figure 2), equal-token plot with one downstream table (Figure 3, Table 1), and the Opus decode path (Figure 4). Names are GPT A22/A24/A25 everywhere. Earlier, after revision 8: Part 1 now runs: what the agents discovered (Figure 1), how they searched (Figure 2, four decision logs), whether it held up (Table 1, matched tokens and downstream tasks), and why Opus decodes faster (Figure 3). An audit found that every AutoPareto agent had the same inference-code hook (audit 26). Page limits (4 / 4 / 1) hold with the markers off.

**Part 2, revision 9 (approved by you on 2026-11-15, "Approve them"):** name substitutions only ("AP-Q" became "GPT A22" in eight places) plus glossary entries for the new names and the two decode benchmarks; see `PART2_HUMAN_AUTHORSHIP.md`. **Part 2, revision 8:** revision 8 left the Part 2 text unchanged. One factual correction to the glossary entry "Original and extended experiments" (tag P2-01) and the abstract fix "in a later campaign" were approved by you on 2026-11-15 ("Approve both"). Nothing has been committed, submitted or published.

**Deadline.** The workshop site (rechecked 2026-11-14) still lists the camera-ready as due **2026-11-13**, which has passed, and says exact time zones would be posted. Check on OpenReview right away whether the camera-ready field is still open.

**Appendices.** The site sets "nine pages, excluding references" and says nothing about appendices. The accepted submission had its appendices in the same PDF, and this draft keeps that layout (Parts 1 to 3 on pp. 1 to 9; appendices and references on pp. 10 to 25). If the portal says appendices count, ask me to split them into a supplement PDF.

## A. Part 2 passages to rewrite in your own words

The workshop expects Part 2 to be human-written. Every passage drafted or changed by Claude is wrapped in `\humanauthor[ID]`. To see them as red boxes, set `\showhumanmarkerstrue` in `paper/main.tex`. Facts and evidence for each ID: `PART2_HUMAN_AUTHORSHIP.md`.

| ID | Section | Status after revision 4 | J |
|---|---|---|---|
| P2-01 | Glossary (Appendix A) | **Expanded:** BPB, update, Pareto frontier, prefill/K-V cache, CUDA graph, kernel fusion, ablation, CI | |
| P2-02, P2-03, P2-04, P2-06 | 2.1, 2.2, 2.3, 2.4 | Your approved text. Wording unchanged; revision 5 sets the (i) to (vii) labels in P2-06 in bold | |
| P2-05 | Loop figure caption | **Your approved caption was replaced.** The figure moved to Part 1 (Figure 1) and got a shorter caption. Your text is preserved in `source_before_revision4/main.tex` | Is the move OK? |
| P2-05b | End of 2.3 | **New:** 2 sentences pointing to Figure 1 and to both trees (the search is a tree) | |
| P2-07 | 2.5 Verification | One phrase added: lineage edges are checked against recorded parent links | |
| P2-08 | Table 2 caption | **New table:** chronological funnel (pre-specified / post hoc / pre-registered); the old table moved to Appendix Table 7 | |
| P2-09 | Selection | **New opening on the winner's curse**; adds "E1 tests AP-Q, not the best possible AutoPareto recipe" | J: why exp 124 was dropped, if you remember |
| P2-10 | Variability | First sentence moved into P2-09 | |
| P2-11 | Data integrity | Revision 5 drops the duplicate "ids 88 to 90" sentence | |
| P2-12 | 2.7 Significance | "came from exposure" softened to "reflects its extra exposure" | J |
| P2-13 | 2.8 Interpretability | Adds the predicted / changed / measured / concluded distinction | |
| P2-14 | 2.9 Limitations | **Rewritten** in two parts: "Addressed since review" and "Remaining" | J: which limits matter most |
| P2-15 | 2.10 Pair-A pilot | **Revision 5:** restructured; **corrects N\*** (88,604,672 in the first launches, 85,458,944 after the 2026-09-09 recalibration) | Check the correction |
| P2-15b | 2.10 Pair-A pilot | **New in revision 5:** what the two arms explored, from every launch (Appendix Table 9); a tendency, not an effect | J: keep it in Part 1 and Part 3? |
| P2-16 | Measurement scope | Shortened again in revision 5; adds that the program calls prefill "diagnostic" | J |
| P2-17 | 2.6 Reproducibility | **New:** three reproduction levels and what each needs | J: what you will release |

## B. Other items only you can do

| # | Item | Where |
|---|---|---|
| 1 | Affiliation placeholder removed (name and email only). Add "Independent Researcher" only if the portal asks for an affiliation | `main.tex` author block |
| 2 | Review the rewritten Disclosure statement (three roles); the placeholder is gone, and the passage is tagged `Disclosure r6` in the marker view | Appendix I |
| 3 | The two checkbox declarations were removed at your request. Restore them from `source_before_revision6/main.tex` only if the Overleaf template or OpenReview requires them in the PDF (I could not open the template without signing in) | Appendix I |
| 4 | Part 3 is AI-drafted. It gained a clearly marked future direction ("agents that ask for their own confirmation"); rewrite it in your voice if you want | Section 3 |
| 5 | Artifact release: decide redaction and publish, or drop the release sentence; then remove `\releasepending`. Before any public release, review the sensitive items in section D | Appendix C |
| 6 | Paste the exact OpenReview review texts, so the checklist can be checked against their wording (it was written from your summaries) | `REVIEWER_RESPONSE_CHECKLIST.md` |
| 7 | Check the camera-ready page for extra fields (poster, slides, licence) | OpenReview |
| 8 | Poster: the draft does not yet show Phase 5 or the new figures | `paper/camera_ready/poster/` |
| 9 | Rotate the RunPod key flagged on 2026-11-09 | RunPod |

## C. Things I do once you say so

| # | Item |
|---|---|
| 1 | Commit the manuscript (revisions 2 to 6, all uncommitted) |
| 2 | Split the appendices into a supplement PDF, if the portal requires it |
| 3 | Rebuild and re-check page limits after your Part 2 rewrite |
| 4 | Upload or submit: **only on your explicit go-ahead** |

## D. Sensitive items found while mapping the repository (review before any public release)

These are files in the worktree, not in the PDF. The repo-mapping pass skipped them and quoted none of their contents.

- `scripts/pod/pull_pod_state_to_mac.py` and `audit/camera_ready_2026/13_FRESH_POD_SETUP.md` §2–3 contain SSH or host details.
- `study/contracts_2026/bridge.py` and `campaign_controller.py` hard-code a default `--a40-host` and a default SSH identity path.
- The pod ID appears in committed paths, for example in `analysis/cr_common.py`.
- `audit/camera_ready_2026/25_HISTORICAL_AGENT_PROVENANCE.md` notes that the local Codex logs contain the pod SSH login and a RunPod API key.
- `paper/AUTHOR_METADATA_PRIVATE.md` is tracked in git.
