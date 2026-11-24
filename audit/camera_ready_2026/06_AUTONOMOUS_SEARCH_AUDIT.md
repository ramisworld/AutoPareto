# 06 — Autonomous Search Audit (camera-ready 2026)

Auditor stance: skeptical, read-only. Scope: how every AutoPareto/AutoResearch search was actually run, what the agent saw and could change, what humans changed, and how the confirmation recipes were chosen. Prior AI-written analyses (PROJECT_STATE.md, docs/RESEARCH_LEDGER.md, FORENSIC_RESEARCH_AUDIT.md, AUTOPARETO_TRAJECTORY_AUDIT.md, paper/*.md) were treated as claims and checked against code, data and git.

## 0. Conventions and evidence base

**Labels.** Every material statement carries one label:
- **[OBSERVED]**: read directly from a file, DB row, JSON key or git object.
- **[DERIVED]**: computed from observed data. The helper scripts are listed in §E.
- **[INFERRED]**: a reasoned conclusion that is not directly recorded.
- **[UNKNOWN]**: the artifacts do not settle it.

**Sources and shorthands.**
- `DB` = `results/results.db`, opened with `file:...?mode=ro&immutable=1`. `SNAP` = `results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db`.
- `RI:` = `git show autopareto/research-infrastructure:<path>`. This branch holds the 93 historical commits from 2026-07-20 to 2026-08-28. The checked-out branch `autopareto/professor-ready-2026` has 6 squashed commits, and `study/`, `runs/`, `paper/` and `analysis/` are untracked there (`git status`). So **no Pair-A code, program or run has a git timestamp**. Pair-A provenance rests on manifests and file mtimes only [OBSERVED].
- `exp N` = `experiments.id`. `A k` = `experiment_number` within a campaign.

**Times.** DB `created_at`, `started_at` and history JSON are UTC. Git dates carry +1200 (NZST). File mtimes are local (+1200).

**Redaction.** Manifests contain SSH endpoints. None are reproduced; where one is needed it appears as `<runpod-ssh-endpoint>`.

**Submitted-PDF text.** The path given in the brief (`scratchpad/pdftext/...`) does not exist. The extracted text was used from `scratchpad/audit/repro/pdftext/automlr_autopareto.txt`. Its Part 2 and Section 1.3 wording matches `paper/main.tex`; only the abstract differs [OBSERVED].

---

## Executive summary: findings ranked by impact on the paper

1. **Harness bug: the H2 fixed-time prefix holds two degenerate attempts the paper does not disclose.** T3 and T4 of the reported 4-attempt prefix, and T5–T8 of its continuation, scored validation BPB ≈ 3.2159.
   - **Mechanism.** In the protected worker, `while True: exec(training_code)` re-executes the training body whenever a candidate's own loop exits before the trusted stop. A freshly initialized model is then checkpointed and evaluated as "quality-valid" (§B.1).
   - **Trigger.** The agent's own T1 timing rewrite, which the program text invited and every later T attempt inherited.
   - **Effect on the agent.** It read the bogus 3.2159 results as learning-rate instability and steered its next edits accordingly.
2. **The 22-attempt "latest" namespace is a different study, not a continuation of the paper's prefix.** `runs/pair-a-official-20260914-new-a40-continuation-20260916-r2` is a fresh restart: new baseline calibration, N* = 85,458,944 instead of 88,604,672, new program hashes `8acffb40…`/`825e53e4…`, different A40s. Its attempts 1–4 are not the paper's T1–T4 or N1–N4. The paper's lineage ends at 8 attempts per arm (§B.2–B.4).
3. **Batch-size behaviour flips across the three independent Pair-A lineages.** In the paper's lineage, both arms cut the batch (T first). In the 09-09 lineage only N cut it. In the latest lineage only T cut it, and N never changed it in 22 attempts. The paper's descriptive contrast is not reproducible across restarts (§B.7–B.8).
4. **Seed-42 historical AutoPareto was rolled back and partly re-run. This is undisclosed, and the project notes mischaracterize it.**
   - Attempts 6–8, and evidently 5, ran on 2026-07-28 as exp 88–91 (live-event files survive) and were deleted from the DB.
   - Attempt 4's evaluation was redone on a *recreated* checkpoint and its decision flipped from `dominated` to `retained`. Attempt 10 also had its decision rewritten after an evaluator change.
   - The program text changed after attempt 4.
   - PROJECT_STATE calls the deleted evidence "orphaned database event duplicates" belonging to "the explicitly discarded earlier attempt-5 record". The events show at least three further attempts (§A.2, §C).
5. **The AutoResearch comparator was not purely quality-only.**
   - In the 50-attempt campaign, 27 attempts that finished training failed a BF16 cache-logit check later recognized as faulty (a FlashAttention vs SDPA reference mismatch). They were stored as `crash` with `val_bpb = NULL` and discarded.
   - 4 of these beat their parent on the training-log `val_bpb`. #29 scored 1.095429 against 1.096768 for its retained parent.
   - The comparator (exp 68) comes from a separate 5-attempt diagnostic under a different program (§A.1, §D.4).
6. **Historical agent identity is self-declared, not verified.** `coding_model_name='gpt-5.6-sol'` and `reasoning_level='high'` are CLI defaults of `runner.py`, recorded even for non-LLM classical, confirmation and ablation campaigns. The handoff explicitly allowed "Codex or Cursor". There are no transcripts, no token usage and no `agent_actions` rows for any of the 75 AutoPareto attempts. The paper's "Codex CLI 0.153.0, gpt-5.6-sol, fresh ephemeral container" description is verified only for Pair-A (§A.6).
7. **Recipe selection narrowed in undocumented stages.**
   - 11 preregistered per-campaign representatives were cut to 4 seed-0 recipes on 2026-08-10, after all discovery results were known.
   - Those 4 were cut to 3 for the Phase-1 panel (the paper's Table 1) around 2026-08-21–25, after the seeds-0/1/2 confirmation outcomes. The lowest-memory role (exp 124 / AP289) was dropped without a recorded rule.
   - The paper's "selection preceded confirmation" is true only for the seeds-0/1/2 confirmation (§D).
8. **Within-lineage protocol drift in Pair-A.**
   - The protected inference adapter changed 6 times in the latest lineage, with candidate re-runs.
   - Inference incompatibilities were adjudicated inconsistently: N5, N6, N8, T7 and T8 were repaired and retried; N19 and N20 were charged as candidate failures.
   - The scientist timeout went from 600 to 900 s and the campaign view was compacted mid-campaign in the 09-09 lineage.
   - Provider quota and capacity failures forced requeues (§B.5).

---

# Part A — Search procedure, per era

## A.0 Campaign inventory

| Era | Campaign (DB `campaigns.campaign_id` / runs namespace) | Attempts | Seed | Dates (UTC) | Status |
|---|---|---|---|---|---|
| Hist. AutoResearch | `autoresearch-seed42-20260723` | 50 (`attempt_limit=50`) | 42 | 2026-07-22 12:46 → 07-23 09:19 | 19 completed / 29 crash / 2 interrupted; 10 retained [DERIVED from DB exp 14–63] |
| Hist. AutoResearch | `autoresearch-diagnostic5-seed42-20260724` | 5 | 42 | 2026-07-24 04:45 → 05:38 | 2 crash (both *retained*), 3 completed (discarded) [OBSERVED exp 64–68] |
| Hist. confirmation/ablation | `confirmation-exp65/68-seeds012-20260725`, `ablation-{batch,nonbatch}-only-seeds012-20260725` | 3 each | 0,1,2 | 2026-07-27 03:02 → 04:37 | exp 72–83 [OBSERVED] |
| Hist. AutoPareto | `autopareto-seed42-20260727` | 25 (+ 4 deleted, see A.2) | 42 | 2026-07-28 03:29 → 07-31 02:51 | 25 completed; 22 retained / 3 dominated [OBSERVED exp 84–87, 92–112] |
| Hist. AutoPareto | `autopareto-seed0-20260731` | 25 | 0 | 2026-08-03 08:27 → 08-04 00:00 | 24 completed + A17 crash; 22 retained [OBSERVED exp 113–137] |
| Hist. AutoPareto | `autopareto-seed1-20260731` | 25 | 1 | 2026-08-04 04:13 → 10:40 | 25 completed; 18 retained / 7 dominated [OBSERVED exp 138–162] |
| Classical | `{random-search,tpe,cmaes,motpe,nsgaii}-seed42-20260805` | 25 each | 42 | 2026-08-06 → 08-07 | completed; OOMs: TPE 2, MOTPE 1, NSGA-II 2, random 2 [OBSERVED exp 163–287] |
| Classical | same methods, seeds 0 and 1 | 0 | 0,1 | registered 2026-08-05 03:05 | `status='ready'`, never run [OBSERVED `campaign_state`] |
| Primary confirmation | `autopareto-primary-*-exp{124,134,136,137}-seed{0,1,2}-20260810` | 12 runs | 0,1,2 | 2026-08-11 23:43 → 08-12 01:23 | exp 288–299 [OBSERVED] |
| Phase-1 matched-token panel | `phase1-*` (7 recipes × seeds 3,4,5) | 21 | 3,4,5 | ~2026-08-25/27 | SNAP exp 301–321 (downstream rows bulk-registered 2026-08-27 02:22) [OBSERVED] |
| Pair-A (paper lineage) | `runs/pair-a-official-20260906-fixed-warmup-v2` → `…-continuation-20260907` → `…-continuation-20260907-new-a40-r2` | 8 per arm | 3 | 2026-09-06 → 09-07 | §B |
| Pair-A (restart 2) | `runs/pair-a-official-20260909-new-a40` | 7 per arm | 3 | 2026-09-09 → 09-10 | §B |
| Pair-A (restart 3) | `runs/pair-a-official-20260914-new-a40` → `…-continuation-20260916` → `…-continuation-20260916-r2` | 22 per arm | 3 | 2026-09-14 → 09-17 | §B |

## A.1 Historical AutoResearch (50-attempt campaign + 5-attempt diagnostic)

**Editable surface and protection.**
- [OBSERVED] The candidate is a copy of `train.py` in `queue/<campaign>/<N>-train.py`, copied into an isolated worktree (`RI:runner.py` lines 885–893).
- [OBSERVED] Protected files are hashed before and after each run: `prepare.py`, `benchmark_config.py`, `benchmark_stability.py`, `inference_benchmark.py`, `deployment_benchmark.py`, `autopareto_db.py`, `event_log.py`, `runner.py`, `protected/inference_adapter.py`, `protected/train_launcher.py` and `protected/smoke_gpu.py` (`RI:benchmark_config.py` lines 35–46). Hashing and validation code is at `RI:runner.py` lines 147–180.
- [OBSERVED] Any other file change in the isolate raises "candidate changed files outside train.py" (`RI:runner.py` lines 978–983).
- [OBSERVED] `protected/train_launcher.py` wraps `prepare.evaluate_bpb` (lines 222–234).

**Prompt / program.**
- [OBSERVED] The 50-attempt campaign used `programs/program_autoresearch.md` at SHA-256 `a634a5959268…`. The DB stores this snapshot (`campaigns.program_instructions`, recomputed hash matches); the file is also in git at commits 6761228–75af7b7.
- Short excerpts (snapshot lines 8, 11, 20–21):
  - "Run in a clean Codex task using GPT-5.6-Sol High."
  - "Submit exactly 50 actual proposed changes"
  - "Always branch from the retained quality winner. Retain a completed candidate only when its `val_bpb` is strictly lower. Revert equal, worse, or failed changes."
- [OBSERVED] The diagnostic used `programs/program_autoresearch_diagnostic_5.md`, SHA `9dde08ed…`.
  - Section A embeds Karpathy's `program.md` verbatim (lines 8–123).
  - The addendum changes the rule (lines 212–216): "If training completed and produced `val_bpb` before post-training inference failed, preserve and report that `val_bpb` … Apply the strict quality-only retention rule to that preserved `val_bpb`."
  - Line 233: "do not target or promise a number such as 0.98."
- [OBSERVED] The git copy of `program_autoresearch.md` was rewritten later. At 65120a5 (2026-07-26) it became `258c495e…`, a 25-attempt version. **That version was never run.**
- [OBSERVED] `campaigns.program_sha256` is overwritten on every attempt by `attach_experiment_metadata` (`RI:runner.py` lines 509–530, `ON CONFLICT … DO UPDATE SET program_sha256=excluded.program_sha256`). Only `experiments.program_sha256` per row is authoritative.

**Contract.**
- [OBSERVED] Fixed 300 s (`RI:prepare.py` line 31, `TIME_BUDGET = 300`). Protected window 300–315 s (`RI:benchmark_config.py` lines 3 and 24).
- [OBSERVED] LR progress = `min(total_training_time / TIME_BUDGET, 1.0)`, with time accumulated only for `step > 10` (`RI:train.py` lines 564, 587 and 612).

**Parent choice.**
- [OBSERVED] Greedy "branch from retained winner" exists only as program text.
- [OBSERVED] Code checks only that the parent belongs to the campaign or is a baseline (`RI:runner.py` lines 480–496). The decision validator enforces "retained iff strictly lower `val_bpb` than *the declared parent*" (lines 1339–1365).
- [DERIVED] In practice every AR50 attempt branched from the current winner. Parents in the DB: 13, 17, 18, 19, 22, 26, 28, 29, 32, 34, 61.

**Failures and retries.**
- [OBSERVED] Exp 14 (A1) is `interrupted`. Agent action id 1 records "sqlite3.InterfaceError binding the pathlib.Path value for program_path; GPU remained idle".
- [DERIVED] 29 attempts are `crash`. 27 of them completed training: the training log prints `val_bpb`, and the inference log ends with `RuntimeError: cached/uncached correctness failed: tokens_match=True logits_match=False max_abs_error≈0.12–0.18` (e.g. `results/logs/autoresearch-seed42-20260723-0008-inference.log`). Their DB rows carry `val_bpb=NULL` and `decision='discarded'`. Exceptions: #43 `tokens_match=False`, and #41–#42 `AssertionError` in training.
- [DERIVED] Four of those discarded rows beat their parent on the logged `val_bpb`:

  | AR50 attempt | logged `val_bpb` | parent `val_bpb` |
  |---|---|---|
  | #17 | 1.097180 | 1.097674 |
  | #18 | 1.097602 | 1.097674 |
  | #29 | 1.095429 | 1.096768 |
  | #30 | 1.095598 | 1.096768 |

  #29 is lower than any AR50 retained value; the best retained is #48 = exp 61 at 1.096754.
- [OBSERVED] The failing check compared the candidate's training-time attention path against the protected SDPA cache. This was corrected only on 2026-07-29 (`git show 8ef7ab2 -- protected/inference_adapter.py`: "Comparing those backends measures backend round-off rather than cache correctness").
- [INFERRED] AR50 was therefore gated by an inference check later judged faulty, so it was not a pure quality-only greedy search. `RI:RESEARCH_PROTOCOL.md` lines 227–229 concede AR50 "used the earlier cache-compatibility rule".

**What the agent saw.**
- [OBSERVED] No campaign-view isolation existed for AutoResearch. The agent worked in the repository with shell access. AR50 proposals 6–50 are logged as `tool_name='codex'`, "launched through scripts/run_runner.sh in tmux autoresearch-0006" (`agent_actions` ids 8–52).
- [OBSERVED] The diagnostic program orders the agent to "Inspect the complete 50-attempt history" (line 179).

**Agent identity.**
- [OBSERVED] `experiments.coding_model_name='gpt-5.6-sol'` and `reasoning_level='high'` are the argparse defaults (`RI:runner.py` lines 1882–1883 and 1902–1903).
- [OBSERVED] AR50 ran as **two agent tasks on two pods**:
  - Agent action id 7 (2026-07-22 13:57 UTC): "Campaign checkpoint reached after experiment 5; no experiment 6 … will be started in this task".
  - Then commits 7133809 "Add safe runner metadata preflight" and 9d905ce "Point helpers at new A40 pod" (2026-07-23 11:32 and 13:15 +1200).
  - #6 started 2026-07-23 01:33 UTC.
- [UNKNOWN] The provider-side model, the reasoning effort actually used, and token cost (`agent_actions.input_tokens` is NULL for all 67 rows).

## A.2 Historical AutoPareto (seeds 42, 0, 1)

**Editable surface / protection.** [OBSERVED] Same runner and protected manifest as A.1. Seed 42 ran under three manifests: `c95816e8` (A1–A4), `55913999` (A5–A10) and `4b1160cd` (A11–A25). Seeds 0 and 1 used `1df6db4c` throughout (`experiments.protected_manifest_sha256`).

**Program.**
- [OBSERVED] `programs/program_autopareto.md`, SHA `7d6817f0…` (`RI:` lines 1–140), for seed-42 A5–A25 and all of seeds 0 and 1. **Seed-42 A1–A4 ran under `af42d041…`** (`experiments.program_sha256`, exp 84–87).
- [OBSERVED] The only change, commit 8548308 on 2026-07-30, inserts the "pause after post-training inference failure" sentence (`git diff c200ccb 8548308 -- programs/program_autopareto.md`).
- Key text:
  - Line 21: "Use only `scripts/run_runner.sh campaign-view` … Do not inspect … AutoResearch, confirmation, ablation, classical-search, or another AutoPareto campaign."
  - Lines 80–81: "A candidate is `retained` exactly when it is non-dominated within the registered same-seed baseline plus this campaign."
  - Lines 30–39: the full research surface, including architecture and the inference adapter.

**AutoResearch vs AutoPareto program diff.** [DERIVED] `diff progsnap/autoresearch-seed42-20260723.md progsnap/autopareto-seed42-20260727.md` gives 194 lines. The non-objective differences:

| Aspect | AutoResearch (`a634a595`, 50) | AutoPareto (`7d6817f0`, 25) |
|---|---|---|
| Budget | 50 attempts | 25 attempts |
| Runner call | `uv run python runner.py run` | `scripts/run_runner.sh` |
| History access | none restricted | "campaign-view only" |
| Research surface | "Keep the protected candidate interface" | explicit architecture / adapter list and `build_inference_adapter` contract |
| Failure rule | "Revert … failed changes" | inference failure → pause campaign |
| Parent rule | retained winner | explicit frontier parent; runner-enforced |
| Feedback | `val_bpb` decision only | three objectives + frontier IDs |

So the objective string was **not** the only difference, as the paper concedes.

**Parent choice.** [OBSERVED] `validate_autopareto_proposal` (`RI:runner.py` lines 721–743):
- the previous attempt must be terminal with a `retained`/`dominated` decision;
- `parent_experiment_id` must be in `autopareto_frontier_ids`, i.e. the *current* non-dominated set over the registered same-seed baseline plus the campaign's rows that are `completed`, `valid_result=1`, cached-mode, and have all three metrics (lines 675–718).

**Acceptance / Pareto.** [OBSERVED] The agent records `retained`/`dominated` and the runner rejects a wrong label (`validate_autopareto_decision`, lines 1367–1378). Dominance is computed on raw `val_bpb`↓, median cached tok/s↑ and peak allocated MiB↓ with no noise margin (`raw_pareto_rows`, lines 1381–1396).

**Failures, retries, rollbacks.**
- [OBSERVED] The program says every failure consumes its attempt. Seed-0 A17 (exp 129) crashed and was consumed.
- [OBSERVED] Seed 42 departs from this.

**Seed-42 departures, in time order:**
1. [OBSERVED] A4 (exp 87) failed the cache gate at 2026-07-28 04:19 UTC. Event 9185 shows the agent labelled it `dominated`: "failed the mandatory protected cached/full compatibility gate at max mean logit error 0.0313604".
2. [OBSERVED] Attempts beyond 4 then ran and were later deleted.
   - Raw event files `results/events/autopareto-seed42-20260727-0006.jsonl`, `-0007.jsonl` and `-0008.jsonl` hold training/inference events for **experiment ids 89, 90 and 91** dated 2026-07-28 04:41–05:15 UTC.
   - Ids 88–90 exist in no preserved DB. `results/archive/safety-20260730-before-attempt5-recovery.sqlite` has `sqlite_sequence.experiments=90` but no rows 88–90.
   - The 242 DB events removed between `safety-20260730T230009Z-before-orphan-event-cleanup.db` and `safety-20260731-before-post-seed42-preparation.db` are exactly the events of ids 89, 90 and 91 [DERIVED].
   - PROJECT_STATE.md calls these "stale live-event rows belonging to the explicitly discarded earlier attempt-5 record" (lines 1259–1260) and "242 orphaned database event duplicates" (lines 1293–1295). The event files show attempts 6, 7 and 8 also ran [DERIVED]. Their proposals and results are lost [UNKNOWN].
3. [OBSERVED] Evaluator repair. Commit 8ef7ab2 (2026-07-29 19:00 +1200) changed the canonical correctness reference to protected SDPA.
4. [OBSERVED] A4 re-measurement. Events 9441–9459 (2026-07-30 02:59–03:02 UTC) re-benchmark exp 87; the first attempt crashed on the tokenizer.
   - The checkpoint used is `/workspace/autopareto-diagnostics/attempt4-recreation/0004-recreated.pt` (`experiments.checkpoint_path`, exp 87; PROJECT_STATE.md lines 1214–1219). [INFERRED] A recreated checkpoint means the candidate was retrained, so exp 87's speed/memory and its `val_bpb` come from different training runs.
   - Event 9458 rewrote the decision: `"previous_decision": "dominated"` → `"corrected_decision": "retained"`.
   - The handoff forbade exactly this: "never retrain, renumber, or delete an attempt" (`git show 8ef7ab2 -- handoffs/autopareto-seed42.md`).
5. [OBSERVED] A5 resubmitted. Exp 91 crashed on 2026-07-30 11:34 UTC with a missing tokenizer and was re-labelled into hidden campaign `setup-autopareto-seed42-20260730-tokenizer-cache`. A5 was rerun as exp 92 "exactly" (PROJECT_STATE.md lines 1229–1233, 1257–1260): the lost proposal was replayed, not regenerated.
6. [OBSERVED] A10 decision rewrite. Exp 97 failed at 21:39 UTC (event 11325 `campaign_paused_after_inference_failure`). The evaluator gained automatic FP32 adjudication (commits d358f39 and ece3dbd). Exp 97 was re-benchmarked and set `retained` (event 11340, 22:59 UTC). Later attempts A20 and A22 used it as parent.
7. [OBSERVED] Three evaluator hashes across seed 42: `62b3730c` (A1–A4), `5d6c2092` (A5–A9) and `0b252757` (A10–A25), as `RI:RESEARCH_PROTOCOL.md` lines 231–238 disclose.

**What the agent saw.**
- [OBSERVED] `campaign_view_command` (`RI:runner.py` lines 1258–1291) prints the registration, frontier IDs, every row of this campaign plus the baseline, and their `agent_actions`. There are none for AutoPareto.
- [OBSERVED] Isolation was **instruction-only**. The agent ran as a repository-level coding session ("To launch either Codex or Cursor as the isolated research agent, provide only: Read `handoffs/autopareto-seed42.md`…", `RI:PROJECT_STATE.md` lines 1156–1160). The handoff asks it not to read PROJECT_STATE.md or other campaigns (`RI:handoffs/autopareto-seed42.md` lines 58–69), but every file was on disk.
- [UNKNOWN] Whether any session read forbidden files. No transcripts are preserved.

**Information parity across seeds.**
- [OBSERVED] Seeds 0 and 1 began only after seed-42 results had been reviewed ("After the seed-42 campaign is complete, review its evidence before authorizing other seeds", `RI:PROJECT_STATE.md` lines 1169–1170).
- [INFERRED] The *humans* could adjust seeds 0 and 1 on seed-42 evidence, for example the program, evaluator and FP32 adjudication. Seeds are not exchangeable replicates of one protocol.

**Agent identity.**
- [OBSERVED] All 75 rows say `gpt-5.6-sol`/`high` (argparse defaults, as in A.1). `RI:handoffs/autopareto-seed42.md` lines 89–92 anticipate "If control must pass from Codex to Cursor…". PROJECT_STATE line 1162 mentions "a tool-credit or agent transition".
- [OBSERVED] Unexplained gaps exist, e.g. seed-0 A16→A17 from 14:52 to 22:21 UTC, and seed-42 A7→A8 from 13:42 to 20:01 UTC.
- [UNKNOWN] Which tool or session ran which attempt, and whether a transition happened mid-campaign.
- [OBSERVED] `agent_actions` has 0 rows for AutoPareto experiments, so no visible messages, tokens or cost are preserved. This conflicts with the protocol's "Report … available LLM usage and cost" (`RI:RESEARCH_PROTOCOL.md` line 222).

## A.3 Classical baselines

- **Space.** [OBSERVED] A frozen 15-dimensional discrete grid (`RI:search/common.py` lines 21–35):
  - `DEPTH` 6–10, `ASPECT_RATIO` 48/64/80, `HEAD_DIM` 64/128, `WINDOW_PATTERN` L/SL/SSSL/SSSS, `KV_HEAD_RATIO` 1/2/4;
  - `TOTAL_BATCH_SIZE` 2^17–2^19;
  - four learning rates, weight decay, warm-up, warm-down, final LR fraction, Muon NS steps.
  - Space hash `bc20cc05…` (`campaigns.search_space_sha256`).
  - `DEVICE_BATCH_SIZE` is fixed at 64 (`RI:RESEARCH_PROTOCOL.md` lines 87–91).
- **Samplers.** [OBSERVED] Optuna `RandomSampler`; `TPESampler(multivariate=True)`; `CmaEsSampler`; MOTPE = TPE with 3 directions; `NSGAIISampler(population_size=8)` (`RI:search/run_search.py` lines 91–105).
- **Objective.** [OBSERVED] Single-objective methods minimize `val_bpb`, with failure = 100.0. MOTPE and NSGA-II use (`val_bpb`, tok/s, MiB) with penalty (100, 0, 1e6) when incompatible (lines 122–137).
- **Budget and seeds.** [OBSERVED] 25 trials each (lines 172–174). Training seed 42, algorithm seed 20260767. Seeds 0 and 1 were registered but **never executed** (`campaign_state.status='ready'`, `next_experiment_number=1`).
- **Agent identity.** [OBSERVED] Not LLM-driven, yet `campaigns.coding_model_name='gpt-5.6-sol'`. This shows the field is not evidence of agent identity.
- **Comparability.** [DERIVED]
  - Same 300 s contract, hardware class and evaluator family.
  - Different, much smaller search space. Batch floor is 2^17, against 2^16 for AutoPareto seed-42 A11–A12. No code edits are possible.
  - One seed per method, against three for AutoPareto.
  - Classical ran after the AutoPareto evaluator fixes (FP32 adjudication). AR50 ran before them.
  - Different failure accounting: OOM counts, no inference-pause.

  Comparable only as "same benchmark, different space". The paper's "do not use them to infer search-policy superiority" is warranted. Its "lowest archived BPB" ranking compares AutoPareto seed 0 (best 1.0933) with AutoResearch seed 42 (1.0950) and classical seed 42; AutoPareto's own seed-42 best is 1.0940 (exp 109) [DERIVED].

## A.4 Pair-A (2026-09) procedure: T and N arms

**Editable surface.** [OBSERVED]
- `study/contracts_2026/context_allowlist.json`: visible files are exactly `train.py`, `prepare.py`, `study_runtime.py`, `program.md` and `campaign_view.json`; writable is only `train.py`.
- The scientist runs in Docker with `--read-only`, `--cap-drop=ALL`, `--network=bridge` and a bind mount of the per-position context (`runs/pair-a-official-20260906-fixed-warmup-v2/pair-a-fixed-time-seed3/scientist_diagnostics/*/invocation.json`, `command_summary`).
- Trusted code is hashed per manifest (`immutable_deployment` in `runs/pair-a-official-20260906-fixed-warmup-v2/provenance/final_pre_study_manifest_pair_a_20260906_fixed_warmup_v2.json`).

**Two-phase scientist.**
- [OBSERVED] Each attempt is a `select_parent` invocation followed by an `edit_child` invocation (invocation metadata `phase`; `study/contracts_2026/campaign_controller.py` lines 373–441, 1146–1196).
- [OBSERVED] Wrapper prompt (RESEARCH_LEDGER.md §20.12, lines 803–806): "Read program.md completely FIRST … Return one JSON object with exactly the bridge proposal fields hypothesis, intended_change, and change_category, plus optional parent_experiment_id."
- [OBSERVED] In `select_parent` the staged `train.py` is the *baseline* (`sha256sum train.py` → `b2fb8788…` in the T-A4 transcript `scientist-1788666887344213000-2807970b2b6f/stdout.jsonl`). Parents' code is visible only through `candidate_patch` in the view.

**Programs (paper prefix).**
- [OBSERVED] The program files were not preserved, but both were recovered byte-exactly from the agents' own `sed -n '1,240p' program.md` outputs. The recovered SHA-256 matches the manifest hashes:
  - T = `a1ea17a98dc5…` (170 lines) from `…/pair-a-fixed-time-seed3/scientist_diagnostics/scientist-1788661656107600000-53256fe647c0/stdout.jsonl`;
  - N = `65a304a20b1a…` (163 lines) from `…/pair-a-fixed-tokens-seed3/scientist_diagnostics/scientist-1788660438842025000-178e01ae0959/stdout.jsonl`.
- [DERIVED] **T vs N differ only in the contract paragraph**: T lines 9–30 against N lines 9–23.
  - T: "a 300.000-second measured training budget … stops on the first completed update whose elapsed protected time is at or beyond 300.000 seconds … `progress = min(measured_training_time / 300, 1)`".
  - N: "Every candidate must process exactly `N_STAR` counted training tokens … `progress = min(total_tokens / N_STAR, 1)`".
  - The objective paragraph is identical in both, starting "Your goal is to discover the highest-value changes to `train.py` that improve validation quality, cached-decode throughput and inference memory under this arm's fixed resource contract" (T lines 138–147).
  - The wrapper text additionally contains the campaign id (`pair-a-fixed-time-seed3` vs `pair-a-fixed-tokens-seed3`), which itself names the contract [OBSERVED, `campaign_controller.py` line 406 template].
- [OBSERVED] Later programs: T `8acffb40…` and N `825e53e4…` (current `study/contracts_2026/programs/`). Again they differ only in the contract paragraph [DERIVED, `diff`]. Against the prefix they add:
  - the parent-eligibility caveat "If its quality is worse than the registered baseline while speed is uncertain, it remains recorded evidence and is parent-eligible only when the controller marks it so";
  - a mandatory "review the complete same-arm evidence" checklist (`diff` lines 57–58, 135–141, 154–162).

**Contract and LR-progress variable.**
- [OBSERVED] T: 300.000 s protected clock, stop at the first completed update past 300 s (program T lines 9–24). N: exactly N* counted tokens, N* = 88,604,672 in the paper lineage (manifest `contracts.N.active_n_star`).
- [OBSERVED] Under protected execution `contract_progress` is replaced. T returns `completed_timing_seconds/300` from the trusted parent; N returns `total_tokens/n_star` (`candidate_process.py` lines 185–216, `note_progress` lines 397–407).
- [OBSERVED] A candidate may ignore the return value. The latest lineage's T sources keep `progress = min(total_training_time / TIME_BUDGET, 1.0)` with `step > 10` gating (`…-r2/pair-a-fixed-time-seed3/sources/pair-a-fixed-time-seed3-attempt-22.train.py`). So the LR-progress variable in T is candidate-chosen and differs between lineages.

**Allowed batches.** [OBSERVED] Both arms share `ALLOWED_TOTAL_BATCH_SIZES = (32768, 65536, 131072, 262144, 524288)` (`study_runtime.py` line 6; enforced again in `candidate_process.py` lines 431–432; `pair_registration.json` line 23). N* = 169 × 524,288 is divisible by every allowed size [DERIVED].

**Parent choice.** [OBSERVED] The agent proposes `parent_experiment_id`; the controller requires same arm and `parent_eligible` (`campaign_controller.py` lines 611–633). Eligibility rules:
- Paper prefix: every `quality_valid_speed_uncertain` record is parent-eligible (manifest `lineage.quality_valid_speed_uncertain_source_parent_eligible: true`). Quality-valid means only "finite BPB + cache correctness passed" (`campaign.py` lines 146–165).
- Later (current `campaign.py` lines 166–186): speed-uncertain parents must also satisfy BPB ≤ baseline. Dominated records are not parent-eligible (lines 240–252).

**Acceptance / Pareto.** [OBSERVED] `decide_candidate` (`campaign.py` lines 209–257) applies dominance on protected BPB, normalized bracketed decode throughput (with CI-based speed relation) and peak allocated memory, among the baseline plus same-arm completed records. The frontier is recomputed from the append-only evidence (`reconcile_same_arm_archive`, lines 260–312).

**Failures.** [OBSERVED] Manifest `failure_semantics`:
- scientific candidate failure consumes the round and the campaign continues;
- infrastructure or integrity failure stops fail-closed;
- `silent_retries: false`.

In practice infrastructure failures were followed by human-approved requeues of the *same* candidate (§B.5).

**What the agent saw.**
- [OBSERVED] The campaign view projects, per same-arm record: hypothesis, intended change, category, `candidate_patch`, protected BPB, status/error, quality/parent/Pareto flags, and training telemetry (tokens, protected seconds, updates, microbatches, params, batch fields, `warmup_lifecycle`) (`context.py` lines 62–160).
- [OBSERVED] It does **not** include `candidate_reported` counters. That matters for §B.1.

**Information parity between arms.**
- [OBSERVED] Initial bytes identical: `starting_source_sha256 = b2fb8788…` in both arms. Same `prepare.py` (`4f2ba9cb…`) and `study_runtime.py` (`50dc7126…`) (`scientist.handoff.agent_context_hashes`). Separate histories.
- [DERIVED] Views differ by arm history, which is the design.

**Warm-up.**
- [OBSERVED] The program states the eleven warm-up updates run "in a separate process … The scientific process is fresh" (T lines 26–30).
- [OBSERVED] What executed was same-process: `warmup_lifecycle.same_worker_process: true`, `warmup_model_discarded: true` in every position file. Validation provenance was `historical_protected_in_process_evaluate_bpb`, not the "disposable model host" of program line 100 (`position-004-pair-a-fixed-time-seed3.json`, `protected_validation_provenance`).
- The paper discloses the warm-up gap but not the in-process validation.

## A.5 Agent identity, version and reasoning level

| Campaign | Recorded identity | Source of record | Verified? |
|---|---|---|---|
| AR50, diag5, AP seeds 42/0/1 | `gpt-5.6-sol` / `high` | `experiments.coding_model_name` = runner CLI default (`RI:runner.py` lines 1882–1883, 1902–1903) | No. Also recorded for classical/confirmation/ablation; handoff allows Codex **or Cursor**; no transcripts [OBSERVED] |
| Pair-A pilot manifests | 2026-09-04: `gpt-5.6-terra`/**medium**; 09-05: `sol`/high; 09-05: `terra`/high; 09-05 17:43Z on: `sol`/high | `runs/pair-a/provenance/final_pre_study_manifest_pair_a_2026090{4,5}*.json`, `…_20260906_sol_high_bracketed.json` | Configuration was changed repeatedly before launch; RESEARCH_LEDGER §20.13 (lines 812–816) had registered terra/medium as frozen [OBSERVED] |
| `runs/ABANDONED_NONSTUDY/pair-a-official-20260905` | `gpt-5.6-terra` / high | 4 `invocation.json` | Abandoned after Ctrl-C (`ABANDONED_NONSTUDY.md`) [OBSERVED] |
| All official Pair-A namespaces | `gpt-5.6-sol` / high / `codex-cli 0.153.0` (441 invocations) | `invocation.json` `metadata`; Codex launched with `model_reasoning_effort=high` (`campaign_controller.py` line 441) | Client-side only. Codex authenticated "Logged in using ChatGPT" via an `auth.json` relay (RESEARCH_LEDGER.md lines 818–831); stdout events contain no model field [OBSERVED]. Provider-side model version **[UNKNOWN]** |

**Provider-side evidence.** [OBSERVED]
- "Codex provider quota exhausted during child Scientist execution" (09-09 lineage, position 8).
- "T6 Scientist child-edit exceeded the frozen 600-second bound" → timeout raised to **900 s** mid-campaign, then the "campaign view … compact[ed]" when it timed out again (`runs/pair-a-official-20260909-new-a40/provenance/final_operational_manifest_pair_a_20260909_t6_timeout900.json` and `…t6_compact_view.json`).
- "Sol/high Scientist capacity failure" (latest lineage, position 31; `…-r2/provenance/final_pair_a_continuation_manifest_20260916_r2_requeue031.json`, `operational_amendments[6]`).

No model or reasoning change occurred *within* the paper lineage [OBSERVED].

## A.6 Comparability summary

[DERIVED] The historical AR and AP campaigns differ in all of the following, so the archive comparison is diagnostic only:
- program text and length (50 vs 25);
- seed (AR: 42 only);
- pod (AR spans two);
- evaluator version (AR50 pre-correction);
- feedback and isolation regime;
- agent session structure.

Pair-A is far better controlled: hash-locked context, per-call containers, identical starting bytes and programs differing only in the contract. But see Part B for harness failures and lineage drift. Historical vs Pair-A also differ:
- inference workload: 256/256 cached decode historically vs 1,024 input / 128 generated bracketed (manifest `inference`);
- scientist architecture: a long-running session vs ephemeral two-phase calls.

---

# Part B — Pair-A forensics

## B.1 Priority item: why T3–T8 report BPB ≈ 3.2159

**Records.** [OBSERVED] `runs/pair-a-official-20260906-fixed-warmup-v2/pair-a-fixed-time-seed3/history.jsonl` and its continuation `…-continuation-20260907-new-a40-r2/pair-a-fixed-time-seed3/history.jsonl`:

| T attempt (line) | position | parent | TBS / DBS | edit | protected BPB | protected updates / tokens | `candidate_reported` (position file) |
|---|---|---|---|---|---|---|---|
| 1 (l.1) | 2 | `baseline-calibration-repaired-20260903-run1` | 262144 / 64 | batch ½ + agent timing rewrite | 1.174807 | 335 / 87,818,240 | step 334, `total_training_time` 299.95 |
| 2 (l.2) | 4 | attempt-01 | 131072 / 64 | batch ½ | 1.103402 | 661 / 86,638,592 | step 660, 299.97 |
| 3 (l.3) | 5 | attempt-02 | 65536 / 32 | batch ½ | **3.215915** | 1264 / 82,837,504 | **step 0, updates 0, microbatches 1, time 0**, `total_tokens` 82,771,968 |
| 4 (l.4) | 7 | attempt-02 | 65536 / 32 | batch ½ + LR ×1/√2 | **3.215901** | 1264 / 82,837,504 | step 0 … (same signature) |
| 5 (l.5, r2) | 10 | attempt-02 | 65536 / 32 | batch ½ + LR ½ | **3.215950** | 1267 | step 0 … |
| 6 (l.6) | 11 | attempt-02 | 131072 / 64 | SwiGLU MLP | **3.215840** | 643 | step 0 … |
| 7 (l.7) | 14 | attempt-02 | 131072 / 64 | fused SwiGLU | **3.215928** | 644 | step 0 … |
| 8 (l.8) | 16 | attempt-02 | 131072 / 64 | short window 1024→512 | **3.215901** | 684 | step 0 … |

**1. Cause: a harness defect, triggered by an agent edit and inherited through the lineage.**

- **(a) The defect.** [OBSERVED] `study/contracts_2026/candidate_process.py` lines 660–697 (current file `700937bb…`): `while True: exec(training_code, …)`. It breaks only on `CandidateStop`; a `CandidateWarmupStop` continues. If the candidate's script returns normally (its own `break` fires before the trusted parent sends `STOP`), the loop re-executes the setup and training body. That builds a **fresh model**, resets `step=0` and `optimizer_updates=0`, and runs one microbatch. The parent then answers `STOP` inside `optimizer.step()` (lines 446–451). The worker checkpoints `candidate_namespace["model"]` (line 744), i.e. the fresh model.
  - The scientific initial-state hash is recorded only once per phase (lines 302–306), so the third pass is not caught.
  - The protected counters keep accumulating across passes, so the record shows 1264 updates for a model that received ≈1.
- **(b) Fingerprint.** [DERIVED] Across all 177 position files with `candidate_reported`, the signature `step==0 and optimizer_updates==0` occurs in exactly the 19 files with BPB > 3, and in none with BPB < 2. The 19 files are 8 unique runs plus copies in the preflight-bug namespaces.
  - The leftover `total_tokens` 82,771,968 = 1263 × 65,536 is the candidate's count when its own loop broke. The protected count is 1264.
  - `microbatches: 1` is the third pass.
  - The near-zero logits fit an untrained `lm_head` (std 0.001 init, source line 154): bf16 `max_logit_absolute_error` is 0.000244 for T3, against 0.111 for T2 (`inference_result.correctness`).
  - [INFERRED] 3.2159 = log2(8192)/4.042 bytes per token, the uniform-prediction BPB.
- **(c) Trigger.**
  - [OBSERVED] T1's own patch (history l.1 `candidate_patch`; intended change "align progress, timing, and counters with Contract T") replaced `if step > 10: total_training_time += dt` with `total_training_time = t1 - t_start_training`, and the stop with `if total_training_time >= TIME_BUDGET: break`.
  - [INFERRED] The candidate's clock now includes startup steps and runs ≈0.3 s ahead of the parent clock. At T2's 0.45 s per update the race happened to go to the parent; at 0.24 s per update (T3–T5) the candidate always wins; T6–T8 lost the race at 0.45 s.
  - [OBSERVED] The same deletion of the `step > 10` gating appears in the two other degenerate T runs: `pair-a-official-20260906-fresh` T2 (3.215839) and `prestudy_validation/pair-a-api-repair-20260905-r10` T1 (3.216018). It does not appear in their healthy counterparts.
  - [INFERRED] The N arm is immune: the token boundary is deterministic and the parent always stops first.
  - [OBSERVED] The program does not forbid this. It asks the candidate to "finish with … `total_training_time`, `step` …" (T lines 95–97) and says the parent "owns the authoritative update boundary" (lines 82–84). It never warns against an earlier self-stop.
- **(d) Not divergence.** [DERIVED] BPB is identical to 4 decimals across different learning rates (T3, T4, T5) and different architectures (T6–T8). The candidate's NaN/>100-loss fast-fail would have aborted a divergent run.
- **(e) Not an evaluation fault.** [INFERRED] The evaluator correctly scored the near-untrained checkpoint it was handed.
- **Persistence.** [OBSERVED] The 09-08/09-09 manifests list the same `candidate_process.py` hash (`ca4d1eaa…`) as the paper lineage, and the defect is still present in the current file. It stays latent in later lineages only because those T agents kept `step > 10` gating (r2 T a22 source).
- [UNKNOWN] The exact text of `ca4d1eaa…`; it is not preserved locally. The data signature matches the current code path.

**2. Parents.** [OBSERVED, history `parent_experiment_id`] T1←baseline; T2←T1; T3←T2; T4←T2; T5–T8←T2. **No attempt selected a degenerate parent.**

**3. Did "quality-valid" eligibility let broken candidates seed later attempts?**
- [OBSERVED] Eligibility allowed it. T3–T8 are `parent_eligible: true` (`source_registry.jsonl` lines for attempts 3–8; history `pareto_decision.parent_eligible`).
- [OBSERVED] T5, T7 and T8 were even `quality_valid_speed_valid`, `pareto_eligible: true`, decision `dominated`. Quality validity checks only finiteness plus cache correctness (`campaign.py` lines 146–165).
- [OBSERVED] No broken candidate was used as a parent. The bogus results did **steer** the agent:
  - T4 hypothesis: "The prior 65,536-token batch failed because retaining the parent learning rates doubled update intensity".
  - T5: "The 65,536-token trials collapsed because their learning rates produced excessive cumulative parameter movement".
  - T7: "The prior SwiGLU trial was confounded by a separate gate projection…".
  - The campaign view hid `candidate_reported` and showed plausible protected counters (`context.py` lines 62–78), so the agent could not diagnose the harness.
- [OBSERVED] The later BPB ≤ baseline gate (`campaign.py` lines 166–186) would exclude speed-uncertain degenerate records.

**4. Disclosure.**
- [OBSERVED] The submitted PDF does not mention T3/T4 or any BPB ≈ 3.216. The prefix table `paper/generated/budget_results_table.tex` (T3/T4 = 3.2159) is **not** `\input` in `paper/main.tex` (the only inputs are lines 15, 21, 58, 71 and 98).
- [OBSERVED] Section 1.3 gives `\TimeBest{}` = T2 1.103402 and describes T as "keeps the 50.33M-parameter architecture and changes batching and learning rate".
- [OBSERVED] Internal notes did see the anomaly:
  - `paper/FINDINGS_FOR_HUMAN.md`: "Its third and fourth trials complete at BPB about3.216";
  - `paper/master_v2_visual_audit.md` line 12 removed the trajectory figure because its axis "is dominated by the two T outliers near 3.216".
- **Recommended wording.** Report T3 and T4 as harness-invalid (re-executed training body, checkpoint of an untrained model). Say the fixed-time prefix contains only two informative attempts. Say T4's learning-rate change was a response to corrupted feedback.

## B.2 Namespace roles

| Namespace | Role | Positions / attempts per arm | Programs T / N | N* | Notes |
|---|---|---|---|---|---|
| `runs/ABANDONED_NONSTUDY/pair-a-official-20260905` | abandoned pilot (terra/high) | 1–2 | `212a947f`/`bc7f1b38` | 88,604,672 | Ctrl-C; N1 `KeyError: 'N_STAR'` [OBSERVED `ABANDONED_NONSTUDY.md`] |
| `runs/pair-a` (+ `prestudy_debug/pair-a-interrupted-20260905`) | pre-official pilot | 2 / 2 | `79642f26`/`e9c853a2` | 88,604,672 | T1, T2, N1 `scientific_failure`; N2 1.139966 |
| `runs/prestudy_validation/*` | non-study validations | 1 | various | — | r10 T1 degenerate 3.216018 |
| `pair-a-official-20260905-fresh` | registered, never ran | 0 | — | — | no `history.jsonl` |
| `pair-a-official-20260906-fresh` | first official start (sol/high bracketed) | 2 / 2 | `7a7b87b6`/`f14d0e45` | 88,604,672 | T2 degenerate 3.215839; superseded |
| **`pair-a-official-20260906-fixed-warmup-v2`** | **paper's 4-attempt prefix** (manifest frozen 2026-09-06T02:00Z) | 4 / 4 (positions 1–8, one A40 `GPU-0188884d…`) | `a1ea17a9`/`65a304a2` | 88,604,672 | `paper/generated/budget_attempts.json` marks only this `current: true` |
| `…-continuation-20260907` | same lineage after pod loss | 5 / 5 | same | same | position 9 (N5, candidate `89d92ce8`) abandoned unconsumed; re-dispatched on `GPU-1fba6e1f…` (`migration/abandoned_position_009.json`) |
| `…-continuation-20260907-preflight-{path,manifest-path,runtime-parent}-bug` | failed continuation preflights | 4 / 4 (copies) | same | same | no new attempts |
| `…-continuation-20260907-new-a40-r2` | same lineage, third A40 | 8 / 8 | same | same | position 11 (T6, candidate `a35ade32`) abandoned on pod loss, re-dispatched on `GPU-764a760a…`; new-A40 calibration gave 88,080,384 but N* kept |
| `pair-a-official-20260908-new-a40` | dry run only | 0 | `21826014`/`29b3452d` | 80,216,064 | `dry-run-position-1` |
| `pair-a-official-20260909-new-a40` | **restart 2** (fresh attempt 1) | 7 / 7 | `21826014`/`29b3452d` | 85,458,944 | timeout 600→900 s, view compaction, quota requeue (§A.5) |
| `pair-a-official-20260912-{fresh,api-repair}` | registered restarts, never ran | 0 | `8acffb40`/`825e53e4` | 84,934,656 | `supersedes.reason`: "new fresh campaign after current-A40 closure and context implementation" |
| `pair-a-official-20260914-new-a40` → `…-continuation-20260916` → **`…-continuation-20260916-r2`** | **restart 3**; latest | 9/10 → 22 / 22 | `8acffb40`/`825e53e4` | 85,458,944 | position 20 (T10) abandoned on pod loss and re-dispatched; amendments in §B.5 |

## B.3 Full per-arm histories

**Paper lineage** (`…-continuation-20260907-new-a40-r2`, 8 positions per arm). [OBSERVED]
- **T**: 8 completed, 0 failed, 1 abandoned-unconsumed dispatch (position 11). **6 of 8 are harness-degenerate** (T3–T8).
- **N**: 8 completed, 0 failed, 1 abandoned-unconsumed dispatch (position 9).

  | N attempt | BPB | parent | Decision |
  |---|---|---|---|
  | N1 | 1.163068 | baseline | |
  | N2 | 1.164258 | N1 | |
  | N3 | 1.124312 | N1 | |
  | N4 | 1.140167 | N3 | |
  | N5 | 1.124997 | N3 | retained |
  | N6 | 1.128183 | N3 | retained |
  | N7 | 1.135569 | N4 | dominated |
  | N8 | 1.133894 | N3 | |

**Latest lineage** (`…-continuation-20260916-r2`, 22 positions per arm). [OBSERVED]
- **T**: 22 completed; 20 retained, T13 speed-uncertain, T20 dominated. TBS 524288 (T1) → 262144 (T2–T11) → 131072 (T12–T22). Best BPB 1.101613 (T14).
- **N**: 20 completed + 2 `scientific_failure` (N19 at position 37, N20 at position 39); 16 retained, 4 dominated (N2, N7, N16, N17). TBS 524288 throughout. Best BPB 1.203034 (N11).
- **Requeues / retries**: §B.5.

## B.4 Why the paper reports only 4; are 5–22 comparable?

- [OBSERVED] Analysis cutoff 2026-09-06T23:59:00+12:00 (`paper/research_question_freeze.md` line 4; `paper/FINDINGS_FOR_HUMAN.md` "Cutoff status"). At that time the paper lineage had exactly 4 per arm; positions 9 and later are dated 2026-09-07. **The paper reports what existed at submission.**
- [DERIVED] The latest namespace's attempts 5–22 are **not** a continuation of the reported 1–4. That lineage restarted from attempt 1 with:
  - a different baseline (`baseline-calibration-new-a40-20260914-run1` vs `…repaired-20260903-run1`);
  - a different N* (85,458,944 vs 88,604,672);
  - different programs (eligibility caveat + review checklist), controller, context and adapter;
  - different hardware.

  Its own attempts 1–4 differ: T1 is GQA at TBS 524288 and BPB 1.2021, against batch-halving at 1.1748 in the paper.
- [DERIVED] Within the paper lineage, attempts 5–8 are protocol-comparable in contract (same N*, same programs) but **not** in hardware: three A40 epochs after positions 8 and 10. Under contract T, hardware changes exposure.
  - The T arm's 5–8 are harness-degenerate anyway.
  - Warm-up is the same-process design throughout (`warmup_lifecycle` in every position file).

## B.5 Amendments, requeues and abandonments

The table covers the latest lineage plus the paper lineage's abandoned positions 9 and 11 [OBSERVED].

| Position | Arm/attempt | Record | What changed | Disposition |
|---|---|---|---|---|
| 6 | N3 | `runs/pair-a-official-20260914-new-a40/controller_state.json.pre_n3_adapter_requeue.json`, `…_2.json` (`positions["6"].status = reconciliation_required`) | base manifest `ec105648…` (positions 1–5; **file not preserved**) → `2ab309eebb9d` | requeued twice; content of change **[UNKNOWN]** |
| 9 | N5 | `operational_amendments[0]` (`…_adapter_fix_20260915.json`) | `protected/inference_adapter.py` `54112be4→0d8769d7`; **new scientist JSON schemas** added | same candidate re-run ("rearmed") |
| 12 | N6 | `operational_amendments[1]` | parent-selection invocation timeout | requeued |
| 12 | N6 | `operational_amendments[2]` (`n6_adapterfix`) | adapter `0d8769d7→f81f434f` (stage-shared value embeddings) | same candidate retried |
| 14 | T7 | `operational_amendments[3]` (`t7_shared_value_embedding_fix`) | adapter `f81f434f→7fa072de` | retried |
| 15 | N8 | `operational_amendments[4]` (`t7_early_shared_value_embedding_fix`) | adapter `7fa072de→3e9d1e02` | retried |
| 16 | T8 | `operational_amendments[5]` (`t8_single_head_value_broadcast_fix`) | adapter `3e9d1e02→80f4d8c9` | retried |
| 20 | T10 | `migration/abandoned_position_020.json`; r2 `continuation.abandoned_unconsumed_invocation` | pod lost (`GPU-98579d60…` → `GPU-35264a0f…`) | candidate `5dfe07da` re-dispatched |
| 31 | T16 | `operational_amendments[6]` (`requeue031`) | "Sol/high Scientist capacity failure" | requeued |
| 37 / 38 | N19 | `operational_amendments[7]` (`posttraining_failure_receipt_fix`) | `train_launcher.py` `b8cdb19b→aed6df93` | N19 **charged as candidate failure** |
| 39 | N20 | `operational_amendments[8]` | `bridge.py`, `train_launcher.py` | N20 charged as candidate failure |
| 9 (paper lineage) | N5 | `…-continuation-20260907/migration/abandoned_position_009.json` | pod lost before receipt | same candidate `89d92ce8` re-dispatched |
| 11 (paper lineage) | T6 | `…-new-a40-r2/migration/abandoned_position_009.json` (sic; `global_position: 11`) | pod lost | same candidate `a35ade32` re-dispatched |

**Validity.**
- [INFERRED] Adapter changes alter the correctness gate that decides eligibility. Candidates N5, N6, N8, T7 and T8 became valid only after humans extended the protected code; N19 and N20 were not given the same treatment.
- [INFERRED] Within-lineage comparability of "valid vs invalid" outcomes is therefore human-mediated. Every amendment declares `scientific_contract_changed: false`. That is accurate for training, not for eligibility.

## B.6 Single trajectory? Prompts? Batch sizes?

- [OBSERVED] **Each arm is a single adaptive trajectory.** Every scientist call reads its arm's cumulative `campaign_view.json`; parents are prior same-arm attempts.
  - The paper prefix is n = 1 trajectory per arm. The paper says this: "one adaptive trajectory per arm, not 8 independent samples".
  - The three restarts are different protocol versions and should not be pooled as replicates [DERIVED].
- [DERIVED] **Prompts differ only in the contract paragraph** (§A.4), plus the campaign id string in the wrapper.
- [OBSERVED] **Same TOTAL_BATCH_SIZE values are allowed** in both arms (§A.4).
- [OBSERVED] The prefix program's warm-up sentence ("separate process") is identical in both arms but was not executed as written.

## B.7 Every batch-size edit, per arm

Values are `result.total_batch_size` per history line [OBSERVED].

| Lineage | T arm | N arm |
|---|---|---|
| Paper (`fixed-warmup-v2` → r2 continuation) | T1 524,288→262,144; T2 →131,072; T3 →65,536 (from T2, DBS 64→32); T4 →65,536 (from T2) + LR ×1/√2; T5 →65,536 (from T2) + LR ½; T6–T8 stay 131,072 | N1, N2 stay 524,288 (capacity edits); N3 →262,144 (from N1); N4 →131,072 (from N3); N5, N6, N8 stay 262,144; N7 stays 131,072 |
| `20260906-fresh` | T1 →262,144; T2 →262,144 (from baseline) | N1 →131,072; N2 →65,536 |
| `20260909-new-a40` | none (524,288 in all 7) | N3 →262,144; N4–N7 stay 262,144 |
| latest (`20260914` → r2) | T2 →262,144; T12 →131,072 | **none** (524,288 in all 22) |

## B.8 What drove behaviour: wording, incentives, observations, or chance?

**Evidence available.**
- [OBSERVED] Wording differs only in the contract paragraph.
- [OBSERVED] Agent rationales reference the contract explicitly. T1: "doubling optimizer-update frequency within the fixed 300-second budget".
- [OBSERVED] Earlier observations drive the T arm after T2. T4, T5 and T7 react to harness-corrupted results (§B.1).
- [DERIVED] Across lineages the direction flips: §B.7, and N1 capacity rises (85.85M) in the paper lineage but falls to 39.85M GQA in the 09-09 and latest lineages. Path dependence and chance dominate, and one-trajectory-per-arm designs cannot separate them.

**Needed to separate the hypotheses** [INFERRED]:
- multiple independent trajectories per arm under one frozen protocol, with randomized arm order and seeds;
- first-attempt-only replicates: many independent attempt-1 calls per arm from the identical context, which isolates wording and incentive from history;
- crossed designs: T wording with N feedback and vice versa, or shuffled-history controls;
- replaying the identical view to the scientist several times to estimate sampling noise;
- the harness bug fixed and invalid runs excluded before any comparison.

---

# Part C — Human interventions

## C.1 Dated timeline

Git dates are +1200; DB and history times are UTC. Evidence is OBSERVED unless marked.

| Date | Change | Evidence | Affects | Validity threat |
|---|---|---|---|---|
| 2026-07-20/22 | A40 baseline; cached evaluator; A40 cached-correctness calibration | commits 5ea452c, da4da1c, 357c390 | all historical | low (pre-search) |
| 2026-07-22 | AR50 launched (program `a634a595`) | campaigns row; exp 14 metadata failure | AR | — |
| 2026-07-23 | AR50 split: task 1 stopped at #5; runner metadata preflight; **new A40 pod**; task 2 ran #6–50 | agent_actions id 7; commits 093f605, 7133809, 9d905ce, a2680be | AR50 | hardware/session change mid-campaign |
| 2026-07-24 | 5-attempt diagnostic preregistered *after* AR50 inspection; program changes failure rule | commits 3d22539, 75af7b7; program `9dde08ed` | AR comparator | rule change; post-hoc continuation |
| 2026-07-24 | diag5 exp 64 infrastructure crash (quota-truncated checkpoint) still `retained` | exp 64 `crash_or_failure_reason` | AR lineage | moderate |
| 2026-07-25/27 | Confirmation of exp 65 and 68 + batch ablations at seeds 0–2 | `RI:RESEARCH_PROTOCOL.md` lines 112–121; exp 72–83 | AR reference | — |
| 2026-07-27 | AutoPareto program/controls hardened (`af42d041`); seed 42 registered | c200ccb; campaigns `created_at` 09:09 UTC | seed 42 | — |
| 2026-07-28 | Seed-42 A1–A4; A4 cache failure; **A5–A8 ran then were deleted** | exp 84–87; event files `-0006/7/8.jsonl` | seed 42 | **high** (undisclosed rollback) |
| 2026-07-29 | Canonical correctness reference → protected SDPA (fixes FA-vs-SDPA false failures) | 8ef7ab2 | all later evaluations; explains AR50 failures | evaluator change between campaigns |
| 2026-07-30 | A4 re-measured on **recreated** checkpoint; decision flipped to retained; pause rule added (program `7d6817f0`); A5 replayed; setup crash re-labelled | events 9441–9459; 8548308, 4843940; exp 91/92 | seed 42 | high |
| 2026-07-30/31 | BF16 failures get FP32 adjudication; A10 paused, re-measured, retained; 242 events deleted | d358f39, ece3dbd; events 11325–11341; archive DBs | seed 42 A10–A25 | moderate (disclosed as evaluator amendment) |
| 2026-07-31 → 08-03 | Seed-0/1 readiness after human review of seed 42 | 92f85f4; campaigns created 08-03 04:53 UTC | seeds 0/1 | humans saw seed-42 results first |
| 2026-08-05/07 | Classical searches (seed 42 only); GPU preflight guard | ea3b3d7, 4716748 | classical | — |
| 2026-08-05 | Per-campaign representative selection (11 recipes) | `results/autopareto_confirmation_selection.json` first in ea3b3d7 | Part D | — |
| 2026-08-10 | **Discovery freeze; global 4-role primary plan** (all seed 0) | cc756d1; `results/discovery-freeze-20260810.json`; `results/autopareto_primary_confirmation_plan.json` | confirmation | post-hoc narrowing (§D.2) |
| 2026-08-11/12 | Primary confirmation seeds 0–2 (exp 288–299) | DB | Part D | — |
| 2026-08-14/17 | Stabilized deployment protocol `autopareto_inference_stabilized_v2`; failure-retry sessions | 3e3645c, d8131bd, 5bf86b5 | deployment numbers | evaluator change after selection |
| ~2026-08-21/25 | Phase-1 seven-recipe panel; exp 124 role dropped | `RI:docs/PHASE1_EQUAL_TOKEN_PREFLIGHT.md` lines 1–31 (dated 2026-08-21) | paper Table 1 | §D.3 |
| 2026-08-25/27 | Phase-1 runs; measurement amendment | `RI:docs/PHASE1_MEASUREMENT_AMENDMENT.md`; SNAP exp 301–321 | paper Table 1 | panel file first committed later (e74771d, 2026-08-28) |
| 2026-09-03/04 | Pair-A registered with **terra/medium**; N* = 88,604,672 from 3 calibration runs | RESEARCH_LEDGER.md lines 640–669, 813–817 | Pair-A | — |
| 2026-09-05 | Model switched sol/high ↔ terra/high; pilots fail (`N_STAR` env) | `runs/pair-a/provenance/*20260905*.json`; ABANDONED_NONSTUDY | Pair-A | pre-launch protocol churn |
| 2026-09-06 | `20260906-fresh` (T2 degenerate) superseded by **fixed-warmup-v2** (same-process warm-up) | manifests `…sol_high_bracketed.json`, `…fixed_warmup_v2.json` | paper prefix | warm-up deviates from program text |
| 2026-09-07 | Two pod losses; abandoned positions 9 and 11 re-dispatched; N* kept across hardware | continuation manifests | paper lineage 5–8 | hardware change within lineage |
| 2026-09-08/09 | Restart 2 with new calibration (N* 85,458,944); new programs | `…20260909_new_a40.json` | — | not pooled with paper |
| 2026-09-09/10 | Restart 2: quota requeue, timeout 600→900 s, view compaction, Codex schema/parser changes | operational manifests (§A.5) | restart 2 | scientist-side changes mid-campaign |
| 2026-09-12 | Restarts registered (fresh / api-repair), never ran; `candidate_process.py` → `700937bb` | manifests | — | — |
| 2026-09-14 → 09-17 | Restart 3 (latest): 9 amendments, requeues, pod loss, failure reclassification | §B.5 | latest lineage | moderate–high |

## C.2 Agent-generated vs human-designed

- **Agent-generated** [OBSERVED: hypothesis, intended change and `code_diff`/`candidate_patch` fields; mechanical decisions validated by code]:
  - every historical AutoPareto/AutoResearch edit and hypothesis, including the AP292 lineage (seed-0 A1→A4→A6→A7→A8→A18→A19→A20→A22; parents exp 113/116/118/119/120/130/131/132 confirmed in DB);
  - parent choices within code-enforced constraints;
  - Pair-A edits, including the harmful T1 timing rewrite.
- **Human-designed** [OBSERVED]:
  - objectives and the three-metric rule;
  - the 300 s contract and N* calibration rule;
  - programs and handoffs, including the rule changes between AR50 and diag5 and between seed-42 A4 and A5;
  - evaluators and their repairs (correctness reference, FP32 adjudication, stabilized inference, Pair-A adapter extensions);
  - eligibility amendments;
  - rollback and replay decisions;
  - representative and panel selection (Part D);
  - the matched-token confirmation design;
  - failure classification (N19/N20 vs N5/N6/N8/T7/T8).
- **Mixed / unclear** [UNKNOWN]:
  - who executed historical runs (handoff allows Codex or Cursor; sessions unrecorded);
  - whether historical sessions read forbidden files;
  - authorship of some "assisted debugging" during the seed-42 recovery.

## C.3 Discoveries with doubtful autonomy

1. **Seed-42 A4 and everything descending from it.** A5–A9 via exp 87; A20 and A22 via exp 97. The trajectory continued only because humans re-measured A4, on a recreated checkpoint, and A10, and rewrote both decisions. The discarded A5–A8 results may have been visible to the replay session [UNKNOWN].
2. **The AutoResearch comparator.** Its lineage depends on human rule changes. The diag5 retained crashes exp 64 and 65 only because the new program preserved their `val_bpb`.
3. **Latest-lineage N5, N6, N8, T7 and T8.** Valid only after protected-adapter extensions written after the failure.
4. **Pair-A T-arm behaviour after T2** (paper lineage). Driven by harness-corrupted feedback, not by the contract.

The headline AP292 (seed-0 A22) has **no** recorded human intervention inside its campaign [OBSERVED: seed-0 rows have constant program, manifest and evaluator; A17 crash consumed]. Caveats [UNKNOWN]:
- the session or tool identity is unverified;
- the unexplained A16→A17 gap of 7.5 h is not documented.

---

# Part D — Selection policy

## D.1 Preregistered AutoPareto representative rule

- [OBSERVED] `RI:RESEARCH_PROTOCOL.md` lines 198–218 define four roles per campaign:
  - lowest-BPB frontier;
  - fastest frontier within 1% of best BPB;
  - lowest-memory frontier within 1%;
  - largest exclusive hypervolume contribution.

  Then: "Representatives are pooled across the three AutoPareto replicates, deduplicated again, and trained on confirmation seeds 0, 1, and 2."
- [OBSERVED] The implementation `scripts/select_autopareto_representatives.py` was committed in c200ccb (2026-07-27 21:12 +1200). That precedes seed-42 A1 (2026-07-28 03:29 UTC), so **the rule was predetermined** [DERIVED].
- [OBSERVED] Output: `results/autopareto_confirmation_selection.json`, rule `preregistered_autopareto_representatives_v1`, **11 representatives**: exp 103, 108, 109, 124, 134, 136, 137, 147, 152, 158 and 162. Identical at ea3b3d7, cc756d1 and e74771d.

## D.2 Narrowing from 11 to 4 (2026-08-10)

- [OBSERVED] `results/autopareto_primary_confirmation_plan.json` (cc756d1, 2026-08-10 11:42 +1200) says: "Apply the preregistered four-role selector globally to the frozen, deduplicated per-campaign representative pool."
  - It selects exp 124, 134, 136 and 137, **all from seed 0**.
  - Disclosure text: "This four-recipe primary panel is a resource-aware staged confirmation, not a replacement or undisclosed post-hoc selection."
- [DERIVED] This global application is **not** the preregistered pooling rule. It was introduced after all three discovery campaigns had finished (seed 1 ended 2026-08-04 10:40 UTC).
- [INFERRED] Global max-over-seeds selection is mechanical but creates winner's-curse bias.
- [OBSERVED] Execution followed the plan: campaigns `…-20260810` were created 2026-08-11 23:43 → 08-12 01:15 UTC (exp 288–299). The paper's "campaign identifiers created on 2026-08-10 and executed from 2026-08-12" is accurate only if "created" refers to the date in the identifier string and the plan commit. The DB `created_at` is 2026-08-11 UTC.

## D.3 Phase-1 seven-recipe panel (the paper's Table 1)

- [OBSERVED] `RI:docs/PHASE1_EQUAL_TOKEN_PREFLIGHT.md` ("Prepared: 2026-08-21") lines 13–31 and `RI:evaluation/phase1_equal_token_preflight.json` (`prepared_on` 2026-08-25, `panel_policy.all_seven_models_frozen_before_stage_B: true`). The panel:
  - baseline (exp 4);
  - AutoResearch exp 68 ("the frozen best-compatible representative");
  - TPE exp 183 (`tpe_quality`);
  - MOTPE exp 234 (`motpe_quality`);
  - AP292, AP299 and AP294, i.e. confirmation rows 292, 299 and 294 of exp 134, 137 and 136.
- [OBSERVED] The lowest-memory role (exp 124; confirmation rows 288–290; deployment representative 289) is **absent**. No rule for dropping it was found. `paper/final_fact_ledger.md` line 51 only states it "was not rerun at 200M", while PROJECT_STATE.md lines 53–57 list 289 among the "important confirmation rows".
- [OBSERVED] The preflight line "AutoPareto 294 is included in Stage B; it is no longer optional" (line 16) shows the panel composition was revised.
- [DERIVED] The panel was set **after** the seeds-0/1/2 confirmation results (8f330a8, 2026-08-12) and the deployment study (to 2026-08-17).
- [DERIVED] The preflight JSON's first git appearance is e74771d (2026-08-28 14:47 +1200), after the Phase-1 runs. Only smoke runs dated 2026-08-23/25 (e.g. `NON_SCIENTIFIC-smoke-autopareto_299-20260824-010756`) corroborate an earlier freeze.
- **Verdict.** The paper's "three AutoPareto recipes selected before confirmation" is true relative to the Phase-1 runs, but the role subset was chosen with confirmation outcomes known [INFERRED]. Disclose the drop of the lowest-memory role.

## D.4 AutoResearch comparator: experiment 65 vs 68

- [OBSERVED] Both come from `autoresearch-diagnostic5-seed42-20260724`:
  - exp 65 (A2): `val_bpb` 1.0948561, `status='crash'`, `inference_status='incompatible'`, `decision='retained'`. This is the lowest observed AutoResearch `val_bpb`.
  - exp 68 (A5): 1.0949760675, compatible, `decision='discarded'` because it did not beat exp 65.
- [OBSERVED] `RI:RESEARCH_PROTOCOL.md` lines 112–121, preregistered before AutoPareto: "experiment 65 exactly: lowest observed quality, inference-incompatible on seed 42; experiment 68 exactly: best fully compatible observed candidate".
- [OBSERVED] Seed 0–2 confirmations:
  - exp 65: all three `crash` (inference), `val_bpb` 1.1037, 1.0989, 1.1025 (exp 72–74);
  - exp 68: 2 compatible + 1 crash, 1.1029, 1.0990, 1.1024 (exp 75–77).
- [INFERRED] Exp 68 was chosen because a deployment comparison needs cached-inference compatibility, and the BPB difference to exp 65 is 0.00012 [DERIVED: 1.0949761 − 1.0948561]. Two caveats:
  - exp 65's incompatibility was measured with the pre-2026-07-29 correctness reference, later judged faulty, and was never re-adjudicated in FP32;
  - "best completed attempt of the quality-only AutoResearch archive" describes a candidate the AutoResearch rule itself discarded.
- [DERIVED] Given the AR50 gating in A.1, the AutoResearch archive's quality frontier is itself evaluator-contaminated. The 0.04 BPB held-out gap in the paper is far larger, so this does not overturn H1, but the comparator's "quality-only" label needs qualification.

## D.5 TPE / MOTPE representatives

[OBSERVED] TPE exp 183 is `tpe-seed42` A21 with the minimum `val_bpb` 1.0977. MOTPE exp 234 is the quality representative, and exp 300 is an exact-recipe checkpoint recovery (`downstream-recovery-motpe-exp234-seed42-20260812`). Random search, CMA-ES and NSGA-II have no Phase-1 representative [OBSERVED `docs/PHASE1_200M_RESULTS.md` lines 155–157].

---

# E. Reproducibility of this audit

All helper code lives in `/private/tmp/claude-501/-Users-Rami-Documents-dev-autopareto/084ded0c-4778-442b-9b7f-c93e451f1f5d/scratchpad/audit/agent/`. Each item lists what it does and the sections it supports:

- `q.py`: read-only SQLite access (immutable URI).
- `ar_logs.py` (output `ar_logs.out`): AR50 logged vs stored `val_bpb` (A.1, D.4).
- `hist.py` and `ns.py`: Pair-A history summaries and the namespace lineage table (B.2–B.7).
- `manif.py`: manifest amendment diffs (B.5), with the SSH-endpoint redactor.
- `recover_prog.py`: byte-exact recovery of programs `a1ea17a9…` and `65a304a2…` from agent transcripts into `progs/` (A.4).
- `progsnap/`: DB-stored historical program snapshots (A.1, A.2).
- `ri/`: `git show` extracts of the research-infrastructure branch.

**Inline checks** (no stored script; reproduce with the commands named here):
- The B.1 degeneracy signature comes from an inline scan of `runs/**/position-*.json`.
- The C.1 count of 242 deleted events comes from comparing the two archive DBs.

No repository script was run. No file other than this report was written in the repository.

## Camera-ready corrections recommended

1. **H2 prefix.** Disclose T3/T4 as harness-invalid (§B.1). State that the T arm has two informative attempts. Fix the harness: treat a normal script exit as a failure and assert the checkpointed model's state lineage. Note the cross-lineage flip of batch-size behaviour if the later lineages are mentioned.
2. **Agent and harness section.** Limit "Codex CLI 0.153.0 / gpt-5.6-sol / fresh container" to Pair-A. For the historical campaigns, state that a repository-level coding session was used (Codex or Cursor permitted), with instruction-based isolation and self-declared model labels.
3. **Human interventions.** Add the seed-42 rollback of A5–A8, the recreated-checkpoint re-measurement of A4, the decision rewrites for A4 and A10, the A5 replay, the AR50 program and evaluator differences, and the Pair-A adapter amendments and requeues.
4. **Selection.** Describe the 11 → 4 global narrowing (2026-08-10, post-discovery) and the 4 → 3 Phase-1 panel (exp 124 dropped). Restrict "selection preceded confirmation" to what the timestamps support.
5. **AutoResearch comparator.** Qualify "quality-only". AR50 discarded 27 trained candidates on a faulty inference check, 4 of them better than their parents. The comparator comes from a separate, post-hoc 5-attempt diagnostic under a different program.
