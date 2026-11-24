# Part 2: passages the human author must rewrite and verify

The workshop requires Part 2 ("System Design and Meta-Analysis") to be **human-written**.

**Revision 9 (2026-11-15), approved.** Part 2 received name substitutions only: eight occurrences of "AP-Q" became "GPT A22" (2.3, 2.4 (i), 2.4 (v), the closing sentence of 2.4, 2.5 Verification, 2.5 Selection, 2.6, 2.7); in 2.5 Verification the article changed with it ("an AP-Q checkpoint" became "a GPT A22 checkpoint"). The glossary (P2-01) has a new "Candidate names" mapping, a new "Decode benchmarks" entry and one added sentence in "Original and extended experiments". The word diff is `REVISION9.diff`. Asked "Approve the Part 2 and glossary name changes?", the author replied "Approve them" (2026-11-15).

**Revision 8 (2026-11-15).** The Part 2 text is unchanged (the source is identical to revision 7). One factual correction in the glossary (P2-01): the phase entry no longer implies that only the extended harness allowed inference-code edits. Approved: after the revision 8 report, the author was asked "Approve the glossary and abstract wording fixes?" and replied "Approve both" (2026-11-15).

**Revision 7 (2026-11-15).** The author stated in chat that Part 2 has been reviewed and approved. Revision 7 changed only experiment names (historical to original, Pair-A to extended), one cross-reference and one new glossary entry; the list and `PART2_REVISION7.diff` are in `CHANGELOG_CAMERA_READY.md`. Asked "Approve the Part 2 name substitutions?", the author replied "Approve them" on 2026-11-15.

**Revision 6 (2026-11-15).** Every Part 2 passage was restructured again and now carries an `r6` tag, including your previously approved P2-02, P2-03, P2-04 and P2-06. The full provisional text, a change table and the facts to confirm are in `PART2_REVISION6_PROVISIONAL.md`, and the word diff against revision 5 is `PART2_REVISION6.diff`. The facts and evidence listed below still apply.

Every Part 2 paragraph in `paper/main.tex` was drafted or changed by the AI assistant for this camera-ready draft. Each one is wrapped in `\humanauthor[P2-xx]{…}`, which renders as a red-framed box headed "HUMAN AUTHOR MUST REWRITE/VERIFY [P2-xx]". Captions carry the same tag through `\humancaption{P2-xx}`.

**How to finish Part 2**
1. Rewrite each passage in your own words.
2. Check each fact against the evidence listed for it below.
3. Remove the wrapper.
4. When no wrappers remain, set `\showhumanmarkersfalse`, or delete the macro definitions.

The submitted Part 2 was also AI-drafted (`paper/README.md`: "Part 2 is explicitly provisional AI-written prose"), so none of the old text can be kept as human-written either.

## Revision 3 (2026-11-09): passages changed by Phase 5

Phase 5 (doc 24) changed facts in nine boxed passages. All 16 are still AI-drafted and still need your rewrite; this table lists only what is new.

| ID | What changed | Evidence to check |
|---|---|---|
| P2-01 | Glossary: "matched-exposure retraining" now includes Opus's recipes; new entry "Opus retraining / ablations" | `evaluation/PHASE5_PREREG.md` |
| P2-03 | **Correction:** the reported Pair-A arms (Codex continuation, Opus) and the ablations used PyTorch **2.8.0**+cu128; the September pilot recorded 2.9.1. The Opus retraining used E1's environment | `runtime.torch` in `runs/pair-a-official-20260914-new-a40-continuation-20260916-r2/provenance/final_pair_a_continuation_manifest_20260916_r2.json` and `runs/pair-a-claude-opus55-t-seed3-new-a40-continuation-20261105c/provenance/claude_t_continuation_manifest_20261105c.json`; `"torch"` in `results/camera_ready_2026/phase5/B/kernel_census/*.json`; Appendix Table 3 |
| P2-06 | New (vii): the driver mapping and storage export needed for the Opus retraining, registered with the plan; "Opus-test recipes" added to what humans chose | doc 24 §2; `scripts/opus_contract_t_to_phase1_source.py`; `scripts/phase5a_export_deployed_checkpoints.py` |
| P2-07 | Phase 5 analysis reproduces byte for byte from the archived raw records; the figure list now excludes the hand-drawn Figure 3 and includes Table 4 | `results/camera_ready_2026/phase5/README.md` (verification commands) |
| P2-10 | 5 of 39 ablation runs (all without the CUDA graph) failed the frozen stability rule; reported, not re-run | `phase5_analysis.json` → `phase5b.unstable_runs` |
| P2-11 | The Opus retraining's 10 runs passed every gate on the first attempt | `results/camera_ready_2026/phase5/A/runs/*/phase5a_run_record.json` |
| P2-12 | **J** Opus sentence: repeated the strategy, worse still per token, most decode gain from graph replay whose capture was untimed | Part 1 §1.4 |
| P2-14 | **J** The cross-harness caveat is gone (superseded by 5A); new limits: one budget, token-fraction instead of time-fraction LR progress, ablations on one workload and one host, interacting factors | doc 24 §4 |
| P2-16 | Adds the ablation confirmation (graph 11.2–11.7× of decode speed; capture = 97% of prefill) and now points to Figure 5b for the per-attempt trajectory and Figure 2c for the ablations | `PHASE5_TABLES.md`; Figure 2c |

**Further box edits after the independent review (2026-11-09):**

| ID | Change |
|---|---|
| P2-03 | Ablations: only PyTorch 2.8.0 is claimed, since Python is not recorded |
| P2-07 | "Every result number", because hashes, dates and protocol constants are typed by hand |
| P2-10 | The Opus re-verification had 6 of 9 repeats pass the drift rule (A27: 1 of 3) |
| P2-14 | Graph replay and fusion were not applied to the baseline or AR-ref |
| P2-16 | Besides backend, early-exit repair and FP32 fix, the isolation, call timeout and attempt budget also differed (§2.4) |

**Historical agent provenance (2026-11-09, doc 25).** The author's Codex session logs show the following:
- **Tool:** the Codex desktop app ran the historical campaigns, except seed-42 A1–A5 (Cursor).
- **Model:** gpt-5.6-sol at high effort for every reported attempt. Seed-42 A1–A5 were proposed in Cursor's agent (same model and setting) while Codex usage was exhausted; the replayed A5 (exp 92) only re-ran that proposal.
- **Transcripts:** they survive privately (they contain credentials).

The drafts of P2-02, P2-06 (last sentence narrowed to "no recorded scientific intervention"), P2-13 and P2-14 were corrected to match. By the author's decision, the seed-0 overnight restart is documented in doc 25 only, not in the paper.

**Page budget after revision 3.** With the markers Part 2 ends near the bottom of page 8; without them it ends about 40% down page 8. A human rewrite has roughly half a page of slack.

## Revision 2 (2026-11-08): the 16 passages and what each needs from you

Revision 2 adds E1 (pre-registered token sweep, doc 23) and the Claude Opus 5.5 fixed-time arm (docs 20–22). Facts changed in 10 passages and one passage (P2-16) is new. The detailed fact lists for P2-01 to P2-15 further down still apply; the table below supersedes them where they differ.

**Rule for every passage.** Write it in your own words. Keep only facts you have checked against the listed evidence. Do not keep a sentence because it is already there: none of the boxed text is human-written. Where a passage calls for judgement (marked **J**), the draft offers no opinion of yours; supply your own or cut the sentence.

| ID | Where | Status in rev. 2 | What it must convey | Evidence to check | Your input |
|---|---|---|---|---|---|
| P2-01 | Glossary | changed: adds E1, decode-only vs end-to-end | Plain definitions of every term Part 1 uses | `main.tex` Part 1 | Cut entries you think readers do not need |
| P2-02 | §2.1 Agent and harness | changed: adds the Opus arm | Who the agent was in each era, what it could see and write, how isolation worked | Codex: `runs/pair-a-official-20260914-new-a40-continuation-20260916-r2/controller_state.json` `scientist_config` (codex-cli 0.153.0, gpt-5.6-sol, high, 600 s, docker). Opus: `runs/pair-a-claude-opus55-t-seed3-new-a40-continuation-20261105c/controller_state.json` (Claude Code 2.1.318, claude-opus-5-5, xhigh, host, file tools, 900 s); canary: `runs/claude_canary_20261104_cli2.1.318_report.json`; doc 12 §2.3 | **J** what you know about the historical agent (Codex or Cursor) |
| P2-03 | §2.2 Tools | changed: adds the E1 environment | Hardware and software per era | Appendix Table 4 (generated); E1 stage manifest `environment` | None |
| P2-04 | §2.3 Research loop | unchanged facts; one sentence cut for space | The 8-step loop and its parent/retention rules | code references listed further down | Verify the code references |
| P2-05 | Figure 3 caption | changed: Opus isolation | What the loop figure shows | `figures/research_loop.pdf` | None |
| P2-06 | §2.4 Human interventions | changed: adds (iv) Opus arm and (vi) E1 | Every human decision that touched the search or the evidence | Opus arm: doc 20 §§10–21 (FP32 adjudication fix §12; timeout 600→900 s §14; attempt 9 charged §16; extension to 25 valid §16; automatic requeue §20; recorded attempts on two A40s, Opus A1 / A2–27 and Codex A1–9 / A10–22, from the `gpu_uuid` in each `history.jsonl`). E1 deviation: doc 23 §1 | **J** whether you want to name these as your decisions |
| P2-07 | §2.5 Verification | changed: E1 analysis and Phase 2 re-verification | What was independently re-derived and how | doc 23 §§2–3 (cross-check max |Δ| = 0); doc 22 §3 (BPB bit-identical), §4 (AP292 9/9 metrics); `analysis/cr_v2.py` | None |
| P2-08 | Table 3 caption | unchanged | The selection funnel | as below | None |
| P2-09 | Selection paragraph | unchanged | Dates and reasons for each narrowing step | as below | **J** why exp 124 was dropped, if you remember |
| P2-10 | Variability | changed: E1 SD; speed-uncertain rates | Size of seed, session and measurement noise | `cr_v2.json` `e1.seed_consistency`; Opus 7/25 vs Codex 1/22 speed-uncertain from the two `history.jsonl` files; doc 20 §21 for the cause | None |
| P2-11 | Data integrity | changed: E1 gates; one sentence cut | Which value is canonical and what was checked at run time | doc 23 §1 (24/24, 0 retries) | None |
| P2-12 | §2.6 Significance | changed | Why the result matters | Part 1 numbers | **J** entirely yours: this is interpretation. Suggested points only: the sweep found no budget where AP-Q is better per token; a second agent repeated the strategy; measurement scope shaped what was optimised |
| P2-13 | §2.7 Interpretability | unchanged | What the records let a reader trace | as below | Verify the A7/A8 intent-vs-build finding |
| P2-14 | §2.8 Limitations | changed | What the evidence cannot show | E1 has tokens and updates confounded (no E1b); exploratory split assumptions (doc 23 §6); Pair-A asymmetries (one seed per agent; Codex stopped at 22; Opus 900 s timeout and 2 extra attempts; cross-harness matched-token comparison, doc 22 §4) | **J** which limits matter most to you |
| P2-15 | §2.9 Pair-A (H2 pilot) | changed: shortened; "the fix is" became a fact | The H2 pilot and why it is uninterpretable | doc 06 B.1–B.8; the early-exit repair: doc 12 line 30, doc 20 §1 attempt 2 (`CandidateExitedBeforeStop`) | Verify the repair is described correctly |
| P2-16 | §2.9 Measurement scope | **new** | The decode-only metric left prefill untimed; Opus's adapter moved graph capture there; two further harness gaps | doc 20 §21 (audit, caveat, harness gaps); doc 22 §3 (prefill 140–143 ms vs 7.5 ms; e2e 4.6–5.3×); Figure 2c; program text "The protected evaluator owns workload, correctness, synchronization, timing" | **J** whether "within the permitted surface" matches your reading of the program |

**Outside Part 2 (also yours):** the affiliation; the Disclosure statement (now names Claude Opus 5.5 as the writing assistant and as a research agent, marked "[Author to verify.]"); the two accountability boxes; Part 3 is AI-drafted reflection text, so rewrite it in your voice if the workshop treats it as personal reflection.

**Abbreviations used below**
- `SNAPDB` = `/Users/Rami/Documents/dev/autopareto/results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db`. Open it read-only: `mode=ro&immutable=1`.
- `P1JSON` = `…/phase1/phase1_equal_token_results.json`.
- `AUDIT` = `audit/camera_ready_2026/`.
- `CR` = `paper/camera_ready/data/`. It holds `cr_evidence.json`, `cr_stats.json` and `lineage_edit_labels.json`.

**Page budget.** With the markers, Part 2 occupies pages 5–8 and ends about half-way down page 8. Without them it still ends on page 8. The limit is 4 pages, so a human rewrite has roughly half a page of slack.

---

## P2-01: Glossary (box before §2.1)

**Facts**
- Contract: the fixed-time contract is 300 s. The fixed-token contract is N* tokens in the pilot and 201,326,592 tokens in the panel.
- Exposure is the number of tokens processed before evaluation.
- Matched exposure:
  - AP-Q and AR-ref both ran 1,536 updates at batch 131,072.
  - The baseline ran 384 updates at batch 524,288.
- Campaign: 25 attempts per AutoPareto seed.
- Attempt: failures consume an attempt.
- Retained = non-dominated within the campaign plus the registered same-seed baseline, on raw BPB, decode and memory.
- Pair-A: speed-uncertain records are not Pareto-eligible.
- Held-out shard: `shard_06541.parquet`.
- Timing covers 255 decode steps.
- Old IDs: AP292, AP299, AP294; exp 134, 137, 136; A22, A25, A24; exp 68.

**Evidence**
- `AUDIT/06_AUTONOMOUS_SEARCH_AUDIT.md` A.2 (`validate_autopareto_proposal`, `raw_pareto_rows`, program lines 80–81) and A.4.
- `AUDIT/supporting/REPRODUCIBILITY_AND_PROVENANCE.md` A.2 (255 decode tokens).
- `09` §0.1–0.2.

**Suggested content.** Keep one line per term and define it in plain words. Consider moving the glossary to Appendix A if Part 2 needs space (09 page-budget note).

## P2-02: §2.1 Agent and Harness

**Facts: Pair-A**
- Codex CLI 0.153.0, `gpt-5.6-sol`, reasoning high.
- Fresh container per call; 600 s timeout, raised to 900 s in restart 2 (09-09 lineage).
- Five staged files are visible; only `train.py` is writable.

**Facts: historical campaigns**
- They ran in a repository-level coding session. The handoff allowed "Codex or Cursor".
- `coding_model_name='gpt-5.6-sol'` and `reasoning_level='high'` are the argparse defaults of `runner.py`. They are also recorded for the classical campaigns.
- No transcripts exist. `agent_actions` has 0 rows for AutoPareto.
- Isolation was by instruction only ("campaign-view only") plus hashing of protected files. Any change outside `train.py` raises an error.

**Evidence**
- `AUDIT/06` A.1, A.2 ("What the agent saw"), A.4, A.5. The A.5 table lists the Pair-A manifests.
- `runs/pair-a-official-20260906-fixed-warmup-v2/provenance/final_pre_study_manifest_pair_a_20260906_fixed_warmup_v2.json`, key `scientist`. It was verified for this draft:
  - `codex-cli 0.153.0`;
  - `timeout_seconds 600.0`;
  - `fresh_ephemeral_invocations: true`.
- `study/contracts_2026/context_allowlist.json`.
- REPRO M17.

**Must verify personally**
- Which tool (Codex or Cursor) and which model actually ran the historical campaigns. If you know, state it and say how you know.
- The 7.5 h gap between seed-0 A16 and A17 is unexplained (06 A.2/C.3).

## P2-03: §2.2 Tools

**Facts**
- The panel ran on Python 3.10.18, torch 2.9.1+cu128, CUDA 12.8 and driver 570.211.01, on one physical A40. It was read from all 21 `run_metadata.json` and `final_inference.json` files.
- The Pair-A manifest records Python 3.12, torch 2.9.1+cu128 and CUDA 12.8. The driver is not recorded.
- The discovery-era environment is not recorded. Tracebacks show `python3.10` (e.g. `SNAPDB experiments.crash_or_failure_reason` for exp 65).
- One canary session used 3.12.3 with torch 2.8.0+cu128.
- At least four drivers appear across eras.

**Evidence**
- REPRO §D.1.
- `paper/generated/camera_ready/table_env.tex`. Its panel and Pair-A rows are generated from those records.

**Suggested content.** Two sentences plus a pointer to Appendix Table 4. The old text said "Python 3.12" for everything (D3/D12); that is wrong for the panel.

## P2-04: §2.3 Research Loop (algorithm box and two follow-up sentences)

**Facts**
- Historical campaigns:
  - The parent must be in `autopareto_frontier_ids`, the current non-dominated set over the same-seed baseline plus completed and valid campaign rows (`runner.py` lines 675–743 on the `research-infrastructure` branch).
  - The runner checks the agent's `retained`/`dominated` label (lines 1367–1396). There is no noise margin.
- Pair-A:
  - `campaign_controller.py` lines 611–633 require the same arm and `parent_eligible`.
  - Eligibility in the paper lineage: every `quality_valid_speed_uncertain` record was parent-eligible.
  - "Quality-valid" means finite BPB plus passing cache correctness (`campaign.py` lines 146–165).
- A09 and A10 are a natural replicate:
  - same model, 19,661,000 parameters;
  - BPB 1.1068 vs 1.1054;
  - decode 324.7 vs 291.0 tok/s;
  - A10 was retained.

**Evidence**
- `AUDIT/06` A.2 and A.4.
- `AUDIT/05` §2.5.
- `CR/cr_stats.json` key `A09_A10`.
- `SNAPDB experiments` ids 121 and 122.

**Must verify.** The code line references, and the claim that failures consume the attempt. Seed-42 departs from that rule (P2-06).

## P2-05: Figure 3 caption and figure text

**Facts**
- The schematic is hand-authored in `analysis/generate_figures.py` (search for `research_loop`).
- The box texts were changed for era accuracy:
  - agent identity: Pair-A verified, historical unverified;
  - trust boundary: "protected files hash-checked every attempt (Pair-A also containerized)";
  - warm-up applies to Pair-A only;
  - the bracketed decode protocol applies to Pair-A only.

**Must verify**
- REPRO A.1 flags the trust-boundary box as a design claim. The Pair-A scientist container ran Codex with `sandbox_mode: danger-full-access` inside Docker, and an earlier probe on another pod found "no enforceable candidate filesystem/process isolation boundary".
- Decide whether "trust boundary" is defensible, or rename the box, for example "protected-evaluation boundary (hash-checked)".
- The loop-back arrow from box 8 is drawn to box 6. It should reach the agent (box 3); fix it in `generate_figures.py` if you keep the figure.

## P2-06: §2.4 Human Interventions

**Facts**
1. **Seed-42 rollback.**
   - A5–A8 ran on 2026-07-28. Their event files are `results/events/autopareto-seed42-20260727-000{6,7,8}.jsonl`.
   - Experiment ids 88–90 are absent from every preserved DB.
   - A4 was re-measured on a recreated checkpoint (`/workspace/autopareto-diagnostics/attempt4-recreation/0004-recreated.pt`). Event 9458 changed its decision from dominated to retained.
   - A5 was replayed as exp 92.
   - The A10 decision was rewritten after FP32 adjudication (event 11340).
   - Program `af42d041` (A1–A4) was replaced by `7d6817f0` (commit 8548308).
   - Seeds 0 and 1 started after a human review of seed 42 (`PROJECT_STATE.md` lines 1169–1170 on the `research-infrastructure` branch).
2. **AR50.**
   - 27 trained candidates failed a cache check that was later judged faulty (commit 8ef7ab2).
   - Four of them beat their parent on logged BPB (#17, #18, #29, #30).
   - AR-ref comes from the 5-attempt diagnostic under program `9dde08ed`.
3. **Pair-A.**
   - Adapter amendments, requeues and model-configuration churn before launch.
   - N19 and N20 were charged as candidate failures, while N5, N6, N8, T7 and T8 were repaired.
   - The warm-up ran in the same process (`warmup_lifecycle.same_worker_process: true`).
4. **Failure accounting.** A17 (exp 129) consumed a seed-0 attempt. The identical seed-42 failure (exp 91) was relabelled as setup.
5. **The AP-Q lineage.** Program, manifest (`1df6db4c`) and evaluator were constant across seed 0.

**Evidence**
- `AUDIT/06` A.1, A.2, B.5, C.1, C.3.
- `AUDIT/03` D17.
- REPRO M16.

**Must verify personally.** These are your own actions. Confirm each one, add any intervention you remember that the records miss, and state who performed each.

## P2-07: §2.5 Verification (first paragraph)

**Facts**
- The audit checks on the 21 panel runs come from the submitted text and REPRO A.2.
- For this revision, the scripts `analysis/cr_evidence.py`, `cr_tables.py`, `cr_figures.py` and `lineage_labels.py` generate every Part 1 number, Tables 1–4 and Figures 1, 2 and 4.
- `cr_stats.json` → `cross_check_vs_audit` shows 118 quantities with maximum |Δ| = 1.1e-16. The count is printed through the macro `\CRNCrossChecked`.
- Hand-authored items: Figure 3, the glossary, and the environment rows outside the panel and the pilot.
- This replaces the submitted "Every table and vector figure in this paper is regenerated" (D14).

**Evidence**
- `CR/cr_stats.json`.
- `paper/camera_ready/BUILD.md`.

## P2-08: Table 3 (selection funnel) caption and contents

**Facts**
- The rule script `scripts/select_autopareto_representatives.py` was committed in c200ccb (2026-07-27 21:12 +1200), before seed-42 A1 (2026-07-28 03:29 UTC).
- `results/autopareto_confirmation_selection.json`:
  - first committed in ea3b3d7 (2026-08-05);
  - SHA-256 `72d58664…`;
  - lists 11 representatives: 103, 108, 109, 124, 134, 136, 137, 147, 152, 158 and 162.
- The plan `results/autopareto_primary_confirmation_plan.json` (cc756d1, 2026-08-10 11:42 +1200) selects 124, 134, 136 and 137.
- The panel preflight is dated 2026-08-21/25, and the panel ran Aug 25–27.

**Evidence**
- Read through `git show autopareto/research-infrastructure:results/…`. The hashes are in `CR/cr_evidence.json` under `selection`.
- `AUDIT/06` Part D.

**Must verify.** Why exp 124 was dropped from the panel. No rule was found; if you remember the reason, state it.

## P2-09: Selection paragraph

**Facts**
- As in P2-08, plus the confirmation campaign `created_at` values of 2026-08-11 23:43 to 2026-08-12 01:15 in DB time (UTC presumed) (D19).
- Exp 109:
  - 50,332,176 parameters;
  - `TOTAL_BATCH_SIZE = 2**16` (frozen source `results/candidates/autopareto-seed42-20260727/0022-train.py`);
  - val 1.0940 on 84.1M tokens;
  - AR-ref: 1.0950 on 88.9M tokens.
- Exp 65 vs exp 68 fixed-time re-run means: 1.1017 vs 1.1014.

**Evidence**
- `CR/cr_stats.json` keys `exp109`, `exp68_tokens`, `fixed_time`.
- `AUDIT/05` §3.6.
- `AUDIT/06` D.4.

**Note.** "Most plausible exposure-neutral candidate" is the audit's inference (05 §0.4). Keep or soften it. Planned experiment E4 tests it.

## P2-10: Variability paragraph

**Facts**
- Selection optimism ranges from 0.0014 to 0.0087 across six candidates.
- Agent campaigns: best-to-second gap ≤ 0.0003; best-to-fifth gap ≤ 0.0021 (AP seed 0: 0.00215).
- Pooled fixed-time seed SD is 0.00215, printed as 0.0022.
- The baseline differs by 0.0244 between seed 0 and seed 42 (exp 8 vs exp 13).
- Token spread of exp 136 is 18.7%, computed as range/mean: 251.8M at discovery against 209.7M–220.5M on re-runs.
- **Session offset (added):** the same baseline code at the same seeds differs by 0.0035–0.0070 BPB between campaigns `baseline-a40-seed0` (ids 3–7) and `baseline-a40-cached-fresh-20260722` (ids 8–12). The later campaign also processed about 3% fewer tokens. Source: `CR/cr_stats.json` keys `baseline_session_offset_by_seed` and `baseline_session_tokens`; `AUDIT/05` §2.1.
- Part 1 now cites this offset as a caveat on the −0.0067 fixed-time margin, because AR-ref's 300 s re-runs (2026-07-27) and AP-Q's (2026-08-12) ran in different sessions.

**Evidence**
- `CR/cr_stats.json` keys `optimism`, `archive_bests`, `pooled_fixed_time_sd`, `baseline_seed0_minus_seed42`, `token_spread_pct`.
- `AUDIT/05` §2–3.

**Note.** The 09 draft said "top-5 gaps (≤ 0.002)" and "SD 0.0021". The primary data give 0.0021 and 0.0022 after rounding; see the changelog.

## P2-11: Data integrity paragraph

**Facts**
- Panel validation values: P1JSON is canonical. DB rows 301–321 differ from it by at most 2.43e-5 (D1).
- Held-out values are identical in both.
- Training runs 14973 and 16716 are still in the `running` state (D8). This was re-read from `SNAPDB` and stored in `CR/cr_evidence.json` under `training_runs_left_running`.
- Ids 88–90 are missing.
- `token_stream_verification.json` has `status` = `RUNTIME_VERIFIED` (D21). It was re-read for this draft.
- **Added:** held-out BPB exceeds validation BPB by 0.053 (baseline) to 0.069 (MOTPE) at 201M tokens, so the offset is recipe-dependent. Source: `CR/cr_stats.json` key `heldout_minus_val_201M`. It is consistent with `AUDIT/11_VALIDITY_VERIFICATION.md` §1.6, which appeared in the worktree during this session.

## P2-12: §2.6 Significance

The submitted text said "removes the apparent quality win". It was rewritten to "reveals that the quality ranking itself depends on the contract" (09 §2.6). This is an interpretation, so write it in your own voice.

## P2-13: §2.7 Interpretability

**Facts**
- Hypotheses and patches are retained for all campaigns: `SNAPDB experiments.hypothesis`, `intended_change` and `code_diff`.
- Transcripts exist only for Pair-A (`runs/**/scientist_diagnostics/*/stdout.jsonl`).
- `change_category` is supplied with each proposal (REPRO M5). The label is applied inconsistently: "MATRIX_LR 0.04→0.03" is `optimizer` at A02 and A05 but `optimization` at A11.
- The recorded intent differs from the built width:
  - A7: the intended change says width 384→320, but 384 was built;
  - A8: the intended change says 320→256, but the build went 384→256.
  - Sources: `SNAPDB experiments.intended_change` for ids 119 and 120, and `CR/lineage_edit_labels.json`.

**Note.** The intent-vs-build finding is new in this draft and nobody has reviewed it yet. Check it against the frozen sources and logs: `results/logs/autopareto-seed0-20260731-0007.log` reports `n_embd` 384.

## P2-14: §2.8 Limitations

**Facts.** These follow 09 §2.8 and 04 concern 1:
- tokens and updates are confounded;
- the seed sets differ between the two contracts;
- selection narrowed after discovery;
- exp 109 was never confirmed;
- historical agent identity is unverified;
- downstream results exist only as aggregates (no item-level records);
- only one GPU type was used.

**Must decide.** Whether to state the E1 and E4 plans here or only in Part 3. They are currently only in Part 3.

## P2-15: §2.9 Pair-A pilot (moved from Part 1)

**Facts**
- There are five lineages, G0–G4, from 2026-09-05 to 2026-09-17.
- N* = 88,604,672 is the median of the calibration runs 88,604,672, 88,080,384 and 88,604,672 (`runs/pair-a/provenance/final_pre_study_manifest_repaired_20260903.json`, key `calibration.runs`). It was verified for this draft.
- The reported prefix is `runs/pair-a-official-20260906-fixed-warmup-v2`:
  - T1 = 1.174807, T2 = 1.103402;
  - T3 and T4 = 3.215915 and 3.215901, with candidate-reported step 0 and 1,264 protected updates.
- The defect is in `study/contracts_2026/candidate_process.py` lines 660–697. A `while True: exec(training_code)` loop re-executes the training body.
- The controller still labelled T3/T4 `quality_valid…`, `parent_eligible: true`.
- T4's hypothesis: "The prior 65,536-token batch failed because retaining the parent learning rates doubled update intensity".
- Batch-size behaviour flips across lineages (05 §5, 06 B.7).

**Evidence**
- `AUDIT/06` B.1–B.8.
- `AUDIT/05` §5.
- `AUDIT/03` C17/D7.

**Must verify.**
- That the fix described has actually been made, or is only planned. The text currently says "the fix is…".
- The planned replicated design (E3) wording.

---

## P2-16: §2.9 What the speed metric does not time (new in revision 2)

**Facts**
- The Pair-A evaluator scores decode-only throughput: 127 cached transitions after a 1,024-token prefill, normalized by bracketing baseline measurements (`autopareto_search_inference_bracketed_v2`). Prefill and end-to-end speed are recorded per attempt but not scored.
- The program text makes the inference adapter part of the candidate surface and says the protected evaluator owns timing.
- From attempt 10 on, Opus's adapter (`ChunkedKVInferenceAdapter` in `train.py`) captures the decode CUDA graph inside `prefill()` (`_build_decode_graph`). Decode-only went 2.6× (A8) → 8.6× (A10) → ~31×; prefill went ~6 ms → 81–169 ms over attempts 10–27; end-to-end stayed 4.3–5.6× (campaign records; `cr_v2.json` `agents.opus.rows`).
- Re-measured on a fresh A40: decode-only 30.9–31.8×, end-to-end 4.6–5.3×, prefill 140–143 ms against the baseline recipe's 7.5 ms (doc 22 §3).
- Codex's prefill stayed ≤ 27 ms and its decode-only and end-to-end speed-ups coincide (≤ 1.58×).
- Harness gaps found in the read-only audit (doc 20 §21): timed-window outputs are not checked; the correctness prefixes (64 and 256 tokens) do not exercise the adapter's prefill ring wrap at T = 1,024.
- The Opus arm inherited the Codex arm's frozen manifest except the agent backend, six controller files (doc 12 §2.3) including the early-exit repair, and the mid-campaign FP32 adjudication fix (doc 20 §12).

**Must decide.** Whether to recommend a specific fix (scored end-to-end metric, prefill budget, or output checks) or only name the gap.

---

## Outside Part 2, also for the human author

- **Author block:** the name and email come from `paper/AUTHOR_METADATA_PRIVATE.md`, which says to use them for the camera-ready. The affiliation is a red placeholder.
- **Human-accountability checkboxes** at the end of the Disclosure: tick them only after you have personally verified the paper.
- **`\releasepending`** marks Appendix A "Code and artifacts". The listed release items do not exist publicly yet; see the release checklist in `CHANGELOG_CAMERA_READY.md`.
- **Part 1** may be AI-drafted, but every claim in it is yours to defend. The numbers are generated; the interpretations ("best explains", "most plausible") are judgement calls.


## Revision 4 additions (2026-11-14)

- **P2-05b** (end of 2.3): points to Figure 1 (the loop, now in Part 1) and says every campaign is a tree, with Opus's branches in Figure 2 and AP-Q's in Appendix Figure 8. Evidence: `parent_experiment_id` in the Pair-A `history.jsonl`; `cr_evidence.json` seed-0 lineage.
- **P2-17** (2.6 Reproducibility): three levels. Level 1 is the build script, which needs the full evidence tree; the E1 and Phase 5 analyses already rerun from committed records. Level 2 covers the E1 and Phase 5A executors, where the shard revision, flash-attention kernel and lockfile are not public or not pinned. Level 3 is the Pair-A controller, which is comparable but not identical. The historical campaigns have no entry point. Evidence: `REPRODUCE.md` (paths checked; commands not executed).
- **P2-09** now opens with the winner's curse. Every re-run candidate scored 0.0014–0.0087 BPB worse than at discovery (`\CROptMin`–`\CROptMax`), and the conclusions are limited to the recipes actually confirmed.
- **P2-14** is now split into "Addressed since review" (the confounded reversal, the single horizon, fixed-time variability) and "Remaining".
- **P2-05** caption: the loop figure moved to Part 1 with a shorter caption. The author-approved caption is preserved in `source_before_revision4/main.tex`.


## Revision 5 changes for your review (2026-11-14)

Substantive wording changes are marked **S**. Punctuation or formatting only is marked **F**.

| ID | What changed | Type |
|---|---|---|
| P2-06 (your approved text) | Roman-numeral item labels (i) to (vii) set in bold so the list scans. Wording unchanged | F |
| P2-07 | Two semicolons became full stops | F |
| P2-08 caption | Semicolons became full stops | F |
| P2-09 | Inline lead-in "Not confirmed." before the exp 109 sentence. Four semicolons became full stops | F |
| P2-10 | Semicolons became full stops; the last sentence now joins "failed the frozen stability rule and are reported, not re-run" | F |
| P2-11 | Removed "Experiment ids 88 to 90 are missing, consistent with the seed-42 rollback" (it repeats P2-06) | S (deletion) |
| P2-12 | One semicolon became a full stop | F |
| P2-13 | Compressed, same content. Now says "AP-Q's recorded intended changes for A7 and A8" to make clear which campaign | S (minor) |
| P2-14 | Pilot sentence now "shows a tendency but is too small for a behavioural conclusion". Removed "Proprietary agent behaviour limits exact reproduction" (covered in 2.6) | S |
| P2-15 | Restructured. **Correction:** N* was 88,604,672 in the first launches and 85,458,944 after recalibration on a new A40 from 2026-09-09; the 2026-09-14 restart used 85,458,944. Removed the sentence "Later restarts disagree on whether either arm explores batch size, so no contract effect on agent behaviour can be estimated", replaced by P2-15b | S |
| P2-15b (new) | "What the two arms explored": the archival reading of every launch (Appendix Table 9) and why it is a tendency, not an effect | S (new) |
| P2-16 | Shortened, because Section 1.5 now explains the mechanism. Adds that the program calls prefill "diagnostic" | S |
| P2-17 | Semicolons became full stops | F |

**Facts behind P2-15b** (all asserted by `table_contract_pilot()` in `analysis/cr_v3.py`):
- Five launch lineages, matching the audit registry: Sep 5 (`runs/pair-a`), Sep 6 fresh, Sep 6 warm-up v2 (final file: `...-continuation-20260907-new-a40-r2`), Sep 9, Sep 14 (final file: `...-continuation-20260916-r2`).
- Best-BPB model per arm (fixed time vs fixed tokens): Sep 6 warm-up 50.3M vs 85.9M; Sep 9 16.3M vs 35.7M; Sep 14 24.6M vs 47.2M. Sep 6 fresh: neither arm edited the architecture.
- Only fixed-token attempts exceeded the 50.3M starting model (10 and 12 layers, Sep 6 warm-up, A1 and A2). The A1 hypothesis says "by increasing model capacity".
- Batch reductions: fixed time in Sep 6 fresh, Sep 6 warm-up and Sep 14; fixed tokens in Sep 5, Sep 6 fresh, Sep 6 warm-up and Sep 9.
- The Sep 6 warm-up fixed-time arm has 2 valid of 8 attempts (six hit the early-exit defect, step 0, BPB 3.216).

**Must decide.** Whether you want the pilot tendency in Part 1 at all (one sentence now) and in Part 3 (one sentence now). It is deliberately absent from the abstract.
