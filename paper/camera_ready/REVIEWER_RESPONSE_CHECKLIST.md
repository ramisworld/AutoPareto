# Reviewer-response checklist (camera-ready revision 5, 2026-11-14)

The four reports were supplied by the author as summaries in chat (the OpenReview text is not stored locally). Section numbers refer to `paper/main.tex` revision 4. "Status" says whether a concern was answered by **evidence** (an experiment or archived record), by **presentation** only, or **remains a limitation**. No concern is marked resolved because its wording was removed. No new experiment was run for this revision. Revision 5 renumbers the main figures: Figure 1 is a compact loop, Figure 2 shows six milestones of the Opus search (the full lineage is Appendix Figure 8), Figure 3 is the full-width E1 figure with Table 1 beside its caption, and Figure 4 sits inside §1.5.

## Area Chair: readability

| # | Criticism | What changed | Evidence | Status |
|---|---|---|---|---|
| AC1 | Excessively dense technical terminology | Abstract rewritten as one 244-word paragraph modelled on the structure of Primer's abstract (problem, what we built, what the agents found, why it matters, measured consequences, outcome), with no internal identifiers (revision 5). Part 1 rebuilt around reader questions: §1.1 question, §1.2 how it works, §1.3 discoveries, §1.4 verification, §1.5 speed mechanism, §1.6 result. Dates, hashes and run counts moved to Part 2 and Appendix C. | n/a | Presentation. Part 2 is still dense in places; it must be rewritten by the author anyway (workshop rule). |
| AC2 | Original jargon and niche vocabulary unexplained | Every term is defined in plain English at first use: contract, exposure, optimizer update, Pareto frontier, parent, prefill, K/V cache, decode-only vs end-to-end, kernel, CUDA graph, kernel fusion, ablation, confidence interval. The glossary (Appendix A) gained these terms. | n/a | Presentation |
| AC3 | BPB not sufficiently defined | §1.1 defines BPB, says why lower is better, and separates validation BPB (seen by agents) from held-out BPB (confirmation only). | n/a | Presentation |
| AC4 | AutoPareto never outlined clearly | New §1.2 "How AutoPareto works": what the human fixes, the agent's loop, the tree structure, the Pareto frontier, the two parent rules. §1.3 walks through six milestones of one search (Figure 2). The loop figure moved from Part 2 to Part 1 (Figure 1). | `study/contracts_2026/campaign_controller.py`, `programs/program_fixed_time.md` | Presentation |
| AC5 | Assumes familiarity with Karpathy's autoresearch | §1.1 explains the autoresearch protocol (same 300 s on one GPU, score the resulting model) and why it couples speed with data. | `references.bib` `autoresearch` | Presentation |
| AC6 | Hard to follow even for experts | Each number in Part 1 now comes with its interpretation. Captions state the conclusion. The abstract, §1.6, P2-12 and Part 3 carry one message. | n/a | Presentation |

## Reviewer NDEA: causal attribution

| # | Criticism | What changed | Evidence | Status |
|---|---|---|---|---|
| N1 | The original confirmation changed token exposure, seeds and evaluation data at once | §1.4 "Why a new test was needed" states the confound. E1 holds seeds (6–8), data order (seed-independent by construction) and the held-out shard fixed, and varies only the budget: 88.6M, 157.3M and 201.3M tokens with equal updates, plus 300 s. Table 1 and Figure 3. | Plan `evaluation/camera_ready_e0_e1_plan_amendment1.json` (sha256 `89f29015`, 2026-11-04); `results/camera_ready_2026/amendment1/E1/` (24 runs) | **Evidence** for AP-Q vs AR-ref |
| N2 | The reversal could not be attributed to exposure alone | E1: AP-Q is worse at every matched budget (+0.038 to +0.042, 9/9 seed–budget pairs); at 300 s the difference is not resolved (−0.002 [−0.020, +0.016]). Exploratory split: extra tokens −0.041, architecture +0.040. The FAQ entry "Does E1 prove that exposure is the only cause?" answers **no**. | `e1_analysis.json`; `e1_exposure_decomposition.json` | Evidence for "not better per token". **Remaining:** tokens and updates move together; architecture and optimizer edits are bundled; the split is exploratory (§1.4, P2-14, FAQ). |
| N3 | The fixed-time vs fixed-token study is too small (four attempts per arm) | Not enlarged, since no new experiments were allowed. Revision 5 reads all five Codex launches from their histories (Appendix Table 9): in each of the three launches with architecture edits the fixed-token arm's best model was 1.7 to 2.2 times larger, while batch reductions do not separate the arms. §1.3, P2-15b and Part 3 report this as a tendency from single runs under changing harnesses, not an estimated effect. P2-14 calls the pilot too small. | Pair-A launch histories; `table_contract_pilot()` | **Remains a limitation** (tendency reported, no effect claimed) |
| N4 | Historical optimizer comparisons uncontrolled (seed, search space, budget, failures) | The seed-42 spread figure was removed from Part 1, which now states "no search-policy ranking". P2-14 and Appendix B say the comparators differ in search space, budget, seed and failure history. | n/a | **Remains a limitation** (stated) |
| N5 | Architectural discoveries not novel | Appendix B states that the architectural changes are not novel in themselves; the contribution is framed as an audit of what a resource contract rewards. | n/a | Accepted |

## Reviewer tKK: training budgets

| # | Criticism | What changed | Evidence | Status |
|---|---|---|---|---|
| T1 | Original verification changed seeds and data alongside exposure | As N1. | E1 | **Evidence** |
| T2 | Only one matched-token horizon | E1 has three matched budgets plus a fresh-seed 300 s control, shown in the abstract, Table 1 and Figure 3. The 201.3M gap replicates the original panel on new seeds and another GPU (difference −0.0005). | E1; `replication_vs_phase1_panel` | **Evidence.** **Remaining:** whether the gap grows with tokens is inconclusive (+0.0031 [−0.0017, +0.0078]); one model scale. |
| T3 | Prospective contract comparisons preliminary | Kept preliminary (P2-15, P2-15b, P2-14, Part 3), with the archival tendency described in N3. | Pair-A launch histories | **Remains a limitation** |
| T4 | Exact agent behaviour not reproducible with proprietary models | New §2.6 defines three levels (artifacts, recipes, agent campaigns) and says Level 3 is comparable, not identical. Appendix C lists the real entry points as copyable command blocks, every flag checked against the script's argument parser; `paper/camera_ready/REPRODUCE.md` gives the full commands and private dependencies. | `paper/camera_ready/REPRODUCE.md` (paths checked to exist; commands not executed) | **Remains a limitation** (stated precisely) |

## Reviewer wwml: selection bias and uncertainty

| # | Criticism | What changed | Evidence | Status |
|---|---|---|---|---|
| W1 | The reversal may come from seed and evaluation-data changes | As N1. | E1 | **Evidence** |
| W2 | Winner's-curse bias from selecting discovery winners | P2-09 now opens with "Selection and the winner's curse": every re-run candidate scored 0.0014–0.0087 BPB worse than at discovery, which is why independent confirmation was needed. The Opus retraining covers that campaign's whole frontier, so there was no selection within it. | `cr_stats.json`; Phase 5A plan | Partly evidence. **Remaining:** AP-Q itself was chosen after discovery. |
| W3 | Training-quality variability under the time contract insufficiently quantified | P2-10 reports the pooled fixed-time seed SD (0.0022), the baseline seed effect (0.024), session effects (0.0035–0.0070), token-count spread (up to 18.7%) and E1 seed SD (≤0.0045). §1.3 says Opus's last 19 valid attempts gained only 0.0033 in total, and Appendix Table 11 notes that AP-Q's steps after A18 move BPB by less than the pooled seed SD. | `cr_stats.json`; E1 | **Evidence** (existing records) |
| W4 | Shortlisting may have excluded competitive candidates | Table 2 is now a chronological funnel marking each step as pre-specified, post hoc or pre-registered; the full counts are in Appendix Table 7. Exp 109 is named as unconfirmed. P2-09: "our conclusions do not extend to unconfirmed recipes". | Selection manifests (`72d58664`) | **Remains a limitation** (stated) |
| W5 | The verification panel covered only selected recipes | As W4. E1 tests AP-Q vs AR-ref; the Opus retraining tests Opus's whole frontier. | E1; Phase 5A | **Remains a limitation** for historical recipes |
| W6 | Parent eligibility, branching and selection need precision | §1.2 gives both parent rules (historical: frontier only; Pair-A: any quality-valid attempt). P2-04 keeps the exact loop, and P2-05b points to both trees. Figure 2 is drawn only from recorded parent IDs, which `cr_v3.py` asserts edge by edge. | Pair-A `history.jsonl` `parent_experiment_id` | **Evidence** |
| W7 | External reproduction depends on artifact access | §2.6, Appendix C, `REPRODUCE.md` and a README pointer state exactly what needs the pending release. | `REPRODUCE.md` | **Remains a limitation** until the author publishes the artifact |
| W8 | Do not imply a greedy single-branch optimizer | §1.2 says the search is a tree. Figure 2 shows the A6 and A7 branches from A5 and the A24/A26/A27 branches; Appendix Figure 8 shows every branch (A12 and A13 on A11, A15 to A17 on A14, A24 to A26 on A23). | Pair-A records | **Evidence** |

## Claim corrections made during the audit

- "ties" (Opus at 201.3M vs AR-ref at 300 s) → "does not differ resolvably (95% CIs span zero)"; no equivalence wording anywhere.
- The 31× speed-up is always tied to the Pair-A baseline and the decode-only protocol. The paper says it does not isolate the architecture (graph and fusion were not applied to the baseline or AR-ref), and that AP-Q's 1.99× uses another protocol.
- Correctness scope (revision 5): "Outputs were unchanged" is gone. §1.5, the FAQ and Appendix C say the checks passed on the 64- and 256-token prefixes they cover and reach neither the timed window nor the 1,024-token ring cache, so full-workload equivalence is not established.
- Fusion's end-to-end effect is stated from the measured prefill (96 → 136 ms), not from an assumed mechanism.
- Inference edits are described as "labelled inference changes", the agent's own label, each within 0.0015 BPB of its parent (asserted in code).
- Validation and held-out BPB are labelled at every use in Part 1.
- Pre-registered vs exploratory vs post hoc is marked at each result (Table 1 caption, §1.4 "Mechanism (exploratory)", "post hoc" for the Opus 300 s comparison, Table 2 selection column).
