# AutoPareto: reproducibility and provenance audit (camera-ready 2026)

Audit date: 2026-10-28. Auditor: read-only reproducibility review.

## Scope and method

The audit covers six things:

- **Submitted PDF.** `/Users/Rami/Documents/dev/automlr_autopareto.pdf`: SHA-256 `60b419b1e8a22fa7c01d606f151c24bde685baadf546a9e5915d727b3dac4023`, 10 pages, PDF `CreationDate` 2026-09-06T22:59:19Z.
- **LaTeX tree.** `paper/`.
- **Analysis pipeline.** `analysis/`.
- **Local evidence.** `results/`, `runs/`, `study/` and `evaluation/`.
- **Git history.**
- **The public GitHub release.**

The audit followed these rules:

- **Read-only.** No repository script was executed.
- **SQLite access.** Databases were opened only as `file:<path>?mode=ro&immutable=1`.
- **Git.** Only `log`, `show`, `ls-tree`, `diff`, `rev-parse` and `remote -v` were used.
- **Network.** There was one network read: unauthenticated GitHub REST GETs for repository visibility, branches and the tree. No credentials were used and nothing was written.
- **Recomputation.** All recomputation used my own scripts in the session scratchpad. The scratchpad is ephemeral, so each script is listed in the Appendix with what it computes.
- **PDF text.** The PDF-text path given in the brief did not exist. I extracted the text with PyMuPDF to `…/scratchpad/audit/repro/pdftext/automlr_autopareto.txt`.
- **Secrets.** SSH endpoints, IP addresses, GPU UUIDs, host names and the repository owner handle are deliberately not reproduced here.

Every statement carries one of four labels:

- **OBSERVED:** read directly from a file, DB row or git object.
- **DERIVED:** computed by me from observed raw data.
- **INFERRED:** reasoned from indirect evidence.
- **UNKNOWN:** cannot be determined from the available artifacts.

---

## 0. Bottom line

1. **Headline numbers reproduce exactly from raw data.** (DERIVED)
   - Inputs: the 21 per-run bundles in `results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/runs/phase1-*`, plus `downstream_200m.db` in the same directory.
   - Every Table 1 cell is reproduced: 7 recipes × 5 columns.
   - 1.99× = 1.985256.
   - 41.7 % = 41.6668.
   - 35.1 % = 35.0771.
   - 50 % = exactly 50.000: 1.9921875 vs 3.984375 MiB.
   - BPB cost = 0.0420340.
   - All 21 runs pass all 8 integrity checks: tokens, steps × tokens per step, one timing row per step, source hash in three places, correctness, checkpoint pairing, 9-metric suite and cutoff.
   - The same values also reproduce from public files on the live public `main` branch.
2. **Correlations reproduce.** (DERIVED)
   - r(BLiMP, held-out BPB) = −0.968817, printed −0.969.
   - r(BLiMP, depth) = 0.717303, printed 0.717.
   - Both reproduce from raw DB rows and from public aggregates.
   - The BLiMP contrast (51 lower / 1 tied / 15 higher of 67) reproduces from raw per-configuration JSON, which is private.
3. **Figure 1 values reproduce, but 4 labels are wrong.** (DERIVED) All 26 nodes match the DB and the public CSV: 25 BPB and 25 decode values, plus the A17 "crash" marker. All 11 parent arrows on the highlighted paths match as well. However, 4 of the 25 hand-typed edit labels state widths that contradict the raw training logs (M2).
4. **The main problems are provenance and packaging, not arithmetic.**
   - The submitted abstract has no pipeline source (M1).
   - The public release docs and the release verifier are stale or broken (M9).
   - The analysis pipeline, the Pair-A study and the paper are untracked in git (M12).
   - The paper's environment statement mixes eras (M4).
   - The "source-diff audit" edit counts are agent self-labels (M5).
   - The paper's appendix misdescribes warm-up token accounting for the panel (M3).

## Mismatch register

| ID | Severity | Location | Finding | Basis |
|---|---|---|---|---|
| M1 | High | Submitted PDF abstract (p.1, ll.1–15) vs `paper/generated/abstract.tex` | **Three different abstracts exist.** None of the build systems produces the submitted one. The submitted text matches only `paper_backups/automlr_overleaf_reconstructed/generated/abstract.tex` (1,189 B, mtime 2026-09-07 10:50). `paper/generated/abstract.tex` (1,422 B) holds a different abstract. So does `generated/abstract.tex` inside `autopareto_final_overleaf.zip` and `paper/working_source_bundle.zip` (1,556 B). `analysis/generate_reports.py:24–31` hard-codes the local version and overwrites `abstract.tex` on every refresh (line 35), as `paper/EDITING_GUIDE.md:25,39` warns. `claim_sentences.json` and `claim_ledger.md` therefore cover an abstract that was **not** submitted. A word-level diff shows the rest of the PDF body is identical to `paper/main.pdf`, except for "Affiliation/Address/email", which a modified `neurips_2026.sty` adds. The abstract's numbers (1.99×, 41.7 %, 35.1 %) are macro-driven and correct. | OBSERVED (diffs), DERIVED (word diff) |
| M2 | High | Fig 1 labels A07–A10 (PDF p.3); `analysis/generate_figures.py:44` | **Four width labels are wrong.** Each candidate's width is `ceil(DEPTH·ASPECT_RATIO/128)·128` (candidate `train.py:470–471`, `HEAD_DIM = 128`). `results/logs/autopareto-seed0-20260731-00NN.log` "Model config" gives these values. **A07:** labelled "width 384→320"; the log shows n_embd 384, so the width is unchanged at 384. **A08:** labelled "width 320→256"; the actual change is 384→256. **A09:** labelled "width 256→320"; the actual change is 256→384. **A10:** labelled "width 320→384"; the width stays 384. A10's source differs from A09's only in `ASPECT_RATIO` 80→96, so A09 and A10 are functionally identical (19,661,000 params each). The labels are hand-typed, contrary to `paper/EDITING_GUIDE.md:187`. The node values and arrows are correct. | OBSERVED (logs, sources), DERIVED |
| M3 | Medium | Appendix "Training protocol" (main.tex:137; PDF ll.269–270) | **Warm-up claim does not hold for the panel.** The appendix says "counted tokens exclude the eleven disposable warm-up updates". In the 21 confirmation runs, steps 0–10 are scientific, counted updates: the `training.log` first line reads `step 00000 … remaining_tokens: 201,326,592`, and `training_metrics.json:excluded_initial_tokens = 1,441,792` is a *timing* exclusion inside the 201,326,592 total (`historical_timing_boundary`). Disposable same-process warm-up exists only in the Pair-A controller (`history.jsonl` `result.warmup_lifecycle.warmup_model_discarded = true`). | OBSERVED |
| M4 | Medium | §2.2 Tools (main.tex:85; PDF ll.154–156) | **"Python 3.12" is wrong for the headline panel.** The paper says "Training and all measurements … driver 570.211.01 … PyTorch 2.9.1+cu128 … CUDA 12.8, Python 3.12". The 21 panel runs used **Python 3.10.18** (`run_metadata.json:software_versions.python`), as did their downstream evaluations (`downstream_evaluation_runs.environment_json`). Python 3.12 applies only to the Pair-A manifest. Drivers 570.195.03, 570.211.01, 580.126.09 and 580.159.04 all occur across eras. One Aug-16 canary session used torch 2.8.0+cu128. Section D has the details. | OBSERVED |
| M5 | Medium | §1.3 (main.tex:54; PDF ll.99–101) | **The edit counts are agent self-labels.** The paper says "the source-diff audit labels 32 … architectural, 21 … optimizer, 10 … optimization logic, eight … batching, and four … schedule". These counts equal the raw `experiments.change_category` tally exactly, in both the snapshot DB and the 2026-08-05 safety DB `results/archive/safety-20260805T025129Z-before-seed1-closure.db`. The campaign program tells the agent to pass `--change-category "<architecture\|optimizer\|schedule\|batching\|systems\|other>"`. The labels are applied inconsistently: the same edit "MATRIX_LR 0.04→0.03" is `optimizer` at A02/A05 but `optimization` at A11. All 10 "optimization logic" edits are learning-rate changes. No analysis script computes this classification. | OBSERVED, INFERRED (self-labels) |
| M6 | Low–Med | §2.5 (main.tex:105; PDF ll.199–200) | **The selection-date wording is inaccurate.** The paper says "campaign identifiers created on 2026-08-10 and executed from 2026-08-12". The DB's `campaigns.created_at` for the 12 primary confirmation campaigns runs from 2026-08-11 23:43:02 to 2026-08-12 01:15:33 (DB clock, presumably UTC; INFERRED). The string "20260810" appears only as an identifier suffix. Prior selection *is* evidenced, but only in private git: `results/autopareto_primary_confirmation_plan.json` (commit `cc756d1`, 2026-08-10 11:42 +1200) and the selection manifest `results/autopareto_confirmation_selection.json` (commit `ea3b3d7`, 2026-08-05; SHA-256 `72d58664…` matches the plan's `source_selection_sha256`). Neither file is public or in the working tree. | OBSERVED |
| M7 | Low | Fig 2 caption | **One caption claim is visibly false.** The caption says "the horizontal bars are narrower than the markers". In the rendered figure, AP299's horizontal error bar (decode SD 8.30 tok/s) clearly extends beyond its marker, and MOTPE's (SD 5.40) is also visible. | OBSERVED (render) |
| M8 | Medium | §2.5 claim (main.tex:105) | **"Every table and vector figure … is regenerated from that pipeline" is only partly true.** The pipeline does emit every file, but some content is hand-written. Table 2's rows are hard-coded strings (`generate_tables.py:60`). Fig 3 is a hard-coded schematic (`generate_figures.py:102–137`). Fig 1's edit labels are hard-coded and wrong (M2). The archive BPBs in main.tex:41 are typed prose: `search_comparison_table.tex` is generated but not `\input`. Every typed number matches the raw data except those in M2. | OBSERVED, DERIVED |
| M9 | High | Appendix "Code and artifacts" (main.tex:131) vs public repo | **Stale branch ref.** The local `origin/main` ref is stale: it points to `9cb262e` (40 files) and contains no recipe sources, protocol manifests, phase-1 summaries or figure data. The live public `main` is `ad0b627` (merge of PR #1, 2026-08-28), whose tree `122dda90…` equals local `fa54db5^{tree}` (100 files). That tree does contain those artifacts for Table 1 and Fig 2. **Missing from public main.** Nothing for the Pair-A study (Table 2, H2) and none of the analysis pipeline. **Stale docs.** `RELEASE.md` and `REPRODUCIBILITY.md` on main are still the 2026-08-11 versions. They say the confirmation panel and downstream evaluation are "Not reproducible from this release yet" and that the release "does not contain … the autonomous-search implementation". **Broken verifier.** `scripts/verify_release.py:63–67` requires `"Confirmation retraining:** planned, not started."` in the README and forbids the string "500". The current README has neither the phrase nor the absence: "688.500" occurs on README:76. The documented verification step therefore fails. | OBSERVED (git, API), DERIVED (static read of the verifier) |
| M10 | Medium | Appendix: "anonymized artifact bundle … through the submission system" | **No reviewer bundle exists locally.** Only manuscript-source bundles are present. The README inside `paper/working_source_bundle.zip` says: "contains rendered evidence, not the private raw data or analysis pipeline". `autopareto_final_overleaf.zip` has an **empty `figures/`** directory, so it does not compile on its own. No evidence shows that a bundle was uploaded: `paper/audit/openreview_group.json` has no supplementary-material field. Meanwhile, the full result set is public under an identifying GitHub handle. | OBSERVED, UNKNOWN (upload) |
| M11 | Low | `paper/generated/` internal files | **Two stale or inconsistent internal outputs.** `pdf_qa.json` describes a different 9-page PDF (SHA `cb0054fb…`, title "AutoPareto: Separating Architecture from Resource Allocation…"), so the submitted PDF was never QA'd by `pdf_qa.py`. The macros `\APWall`/`\ARWall` (508.23/813.45) are sums of per-step wall time, while the public "Complete scientific wall s" (512.61/819.53) is the start-to-end interval. They share a name but differ in meaning; both macros are unused in the PDF. | OBSERVED |
| M12 | Medium | Pipeline reproducibility | **Builds depend on mutable local state.** `analysis/build_dataset.py:57–72` and `common.py:68–79` gate the budget rows on `paper/audit/first_observed_records.json`, a local first-observation timestamp ledger (all 22 entries are 2026-09-06T06:15:52Z). Without that file, a re-run marks the Pair-A rows as post-cutoff. `analyze_budget_effect.py:45` then raises `KeyError` on `snapshots['T']`, and even past that, `generate_tables.py:19` would format `None`. Either way the build fails (DERIVED by static reading). `runs/` now holds 38 `history.jsonl` files against 17 at build time. `git ls-files analysis paper runs study` returns 0: none of these trees is version-controlled anywhere. The 144 key input files are unchanged since the build-time inventory, so the current outputs remain valid. | OBSERVED, DERIVED |
| M13 | Medium | Executed-code identity | **Panel harness.** All 21 `run_metadata.json` record git commit `5bf86b57`, but the phase-1 harness files did not exist in that commit: they were untracked on a dirty pod checkout, and `remote_worktree.patch` covers only 10 tracked files. **Evaluators are hash-linked to public files.** `final_inference.json:evaluator_sha256 = e3f8c8be…` equals public `deployment_benchmark.py`; the downstream `evaluator_sha256 696f9c46…` equals public `downstream_evaluator.py`. **Preflight vs public harness.** The preflight `implementation_sha256` (`evaluation/phase1_equal_token_preflight.json`) matches public code for 5 of 7 files; `protected/phase1_train_launcher.py` and `scripts/evaluate_phase1_checkpoints.py` differ. **Pair-A.** 13 of the 35 executed-file hashes in the v2 manifest's `immutable_deployment` + `inference.evaluator_sha256` are not preserved anywhere locally: `campaign_controller.py`, `campaign.py`, `bridge.py`, `candidate_process.py`, `candidate_sandbox.py`, `context.py`, `inference_benchmark.py` (the Pair-A evaluator), `isolation.py`, `remote_runner.py`, `train_launcher.py`, both program `.md` files and `prompt_snapshot_manifest.json`. The two program texts *are* recoverable byte-exact from the history records (`paper/audit/prompts/{T,N}/program.md` hash to `a1ea17a9…`/`65a304a2…`). | OBSERVED |
| M14 | Medium | Data (main.tex:135) | **Training data is not pinned in code.** The paper says training and validation use dataset revision `915333b4…`. In code, the pin is enforced only for held-out and downstream evaluation (`evaluation/downstream_suite.json`; `downstream_evaluator.py:155–159, 485–490`). `prepare.py:41` downloads training shards from `…/resolve/main`, which is unpinned. Shard content hashes exist only in `study/contracts_2026/provenance/data_manifest.json` (untracked, recorded 2026-08-31, after the panel). Runtime token-stream fingerprints exist only in the private snapshot `token_stream_verification.json`. | OBSERVED |
| M15 | Low–Med | Local checkpoints | **Only 10 of 21 phase-1 final checkpoints are usable locally.** The 9 AP runs and AR seed 3 are present and hash-verified against `training_metrics.json:checkpoint_sha256`. AR seed 4's `final.pt` is truncated (3.0 MB vs 159.4 MB recorded). Ten directories are empty: AR seed 5, Baseline ×3, TPE ×3 and MOTPE ×3. | DERIVED |
| M16 | Low | Fig 1 A17 "crash" | **A17's failure was infrastructure, not science.** The crash was `FileNotFoundError … tokenizer/tokenizer.pkl`, so the "depth 4→3" edit was never trained. Failure accounting also differs between campaigns: the identical failure in seed 42 (exp 91) was reclassified as setup ("no scientific attempt consumed"), while A17 consumed a seed-0 attempt. | OBSERVED |
| M17 | Medium | §2.1/Disclosure (main.tex:82, 141, 144) | **The harness description covers only Pair-A.** The paper describes "Codex CLI 0.153.0 … fresh ephemeral container for every attempt"; this is OBSERVED only in the Pair-A manifest. For the historical discovery campaigns, the DB records just the model name and reasoning level (`gpt-5.6-sol`/`high`); one AutoResearch row has null. `PROJECT_STATE.md:1170` says either "Codex or Cursor" could be launched as the research agent from a handoff file, with agent transitions allowed mid-campaign. The historical client and harness are therefore UNKNOWN. | OBSERVED, UNKNOWN |

---

## A. Provenance table

### A.0 Which source produced the submitted PDF

| Object | SHA-256 (prefix) | Abstract variant | Notes |
|---|---|---|---|
| `/Users/Rami/Documents/dev/automlr_autopareto.pdf` (submitted) | `60b419b1` | "When an autonomous researcher is evaluated under a fixed wall-clock budget…" | OBSERVED: CreationDate 2026-09-06T22:59:19Z; mtime 2026-09-07 11:15. |
| `paper_backups/automlr_autopareto-before-1.4-fix.pdf` | `9609433a` | same as submitted | DERIVED: differs only in §1.4 line breaks. |
| `paper_backups/automlr_overleaf_reconstructed/` (`main.tex` 28,967 B; `generated/abstract.tex` 1,189 B) | n/a | **matches submitted** | OBSERVED: identical to `paper/` except the abstract, two blank lines and the `.sty` ("Affiliation/Address/email"). Its `figures/` and other `generated/*` files equal `paper/`'s (`diff -rq`). |
| `paper/main.pdf` = `paper_backups/submitted-automlr-20260907-020829/main.pdf` | `88aaef39` | "A resource contract can act as a hidden research incentive…" | OBSERVED |
| `paper/main.tex` + `paper/generated/abstract.tex` | `abstract.tex` 1,422 B | as `paper/main.pdf` | OBSERVED: produced by `analysis/generate_reports.py:24–35`. |
| `autopareto_final_overleaf.zip` (`main.pdf` `f46e4d2d`), `paper/working_source_bundle.zip` (`main.pdf` `17be82cb`) | as listed | "Autonomous code-editing researchers are commonly evaluated…" | OBSERVED: the two `main.pdf` files have the same size but different bytes. The final zip has an empty `figures/`. The bundle's `SHA256.json` figure hashes equal `paper/figures/*.pdf`. |

Figure provenance (DERIVED). The text extracted from `paper/figures/autopareto_lineage.pdf` (`c0c422de…`), `pareto_front.pdf` (`8d0e915c…`) and `research_loop.pdf` (`322e3430…`) equals the figure text in the submitted PDF. Those hashes also equal `paper/audit/submission_asset_check.json`.

### A.1 Figures

| Artifact | Generator (script:lines) | Inputs (files / DB tables / keys) | Inputs local? | Recomputed from local raw data | Public-recomputable | Mismatches |
|---|---|---|---|---|---|---|
| **Fig 1: lineage** (`figures/autopareto_lineage.pdf`; main.tex:49–52) | `analysis/generate_figures.py:25–83`, function `main()` | **Data.** `paper/generated/historical_experiments.json`, dumped by `build_dataset.py:84–92` from `downstream_200m.db` table `experiments` (`campaign_id='autopareto-seed0-20260731'` plus baseline `id=8`). Fields: `val_bpb`, `inference_tokens_per_second`, `parent_experiment_id`, `status`, `decision`. **Hand-typed parts.** Edit labels (`:44`), green path `{8,113,116,118,119,120,130,131,132,134}` (`:45`) and amber path `{134,136,137}` (`:46`). | Yes | **Values.** All 26 nodes (25 BPB and 25 decode values, plus the A17 "crash") match the DB to the printed precision. Path edges match the recorded parents (`lineage.py`). **Labels.** 21 of 25 labels match parent→child constant diffs and logs. 4 are wrong (M2). A10 is a de facto replicate of A09: identical code except a no-op constant; BPB 1.1068 vs 1.1054, decode 324.7 vs 291.0 tok/s (−10.4 %). That gives a noise floor for fixed-time discovery measurements. | Yes for values, parents and decisions (`data/experiment_records.csv`, 284 rows; a 0-mismatch check). Labels cannot be checked publicly: no sources for A01–A21, A23. The public `parameter_count` does expose A09 = A10. | M2, M16 (DERIVED) |
| **Fig 2: Pareto** (`figures/pareto_front.pdf`; main.tex:61–64) | `generate_figures.py:84–93` | `paper/generated/pareto_analysis.json` (`analyze_pareto.py:4–14`), built from `equal_token_runs.json` (`build_dataset.py:36–54`). Raw sources: per-run `training_metrics.json`, `final_inference.json` (`median_cached_decode_tokens_per_second`, `peak_inference_cuda_memory_mb`) and DB `downstream_task_results` (`task_name='heldout_climbmix_bpb'`). | Yes | Means and SDs reproduce. Nondominance reproduces: 6 circles, and Baseline (square) is dominated by TPE and AP292. `equal_token_runs.json` differs from my raw recomputation by 0.0 on bpb/speed/memory/steady/params/KV. | Yes: `results/phase1/phase1_equal_token_results.csv`/`.json` (per-run values and SDs) | M7 (caption) (OBSERVED, DERIVED) |
| **Fig 3: research loop** (`figures/research_loop.pdf`; main.tex:90–93) | `generate_figures.py:102–137`; a schematic, every string hard-coded | No data inputs. Its claims can be checked against the Pair-A manifest `runs/pair-a-official-20260906-fixed-warmup-v2/provenance/final_pre_study_manifest_pair_a_20260906_fixed_warmup_v2.json`: `scientist.configuration` (codex-cli 0.153.0, `gpt-5.6-sol`, `high`, `timeout_seconds 600`), `inference.protocol_version` (bracketed v2) and `lineage`. | Yes | Agent configuration, warm-up lifecycle, protocol and arm isolation match the manifest and the 8 history records. | No: no Pair-A artifacts are public | **Trust boundary (UNKNOWN).** The "TRUST BOUNDARY" box is a design claim. The scientist container ran Codex with `sandbox_mode: danger-full-access` inside Docker. An earlier probe (`study/contracts_2026/provenance/remote_isolation_probe_20260901.json`, a different pod) concluded "No enforceable candidate filesystem/process isolation boundary is available". No equivalent probe exists for the Pair-A pod. |

### A.2 Tables

| Artifact | Generator | Inputs | Inputs local? | Recomputed from raw | Public-recomputable | Mismatches |
|---|---|---|---|---|---|---|
| **Table 1** (`generated/main_results_table.tex`; main.tex:56–59) | `generate_tables.py:31` (means from `pareto_analysis.json`) | **Per-run files** (21 runs `phase1-<model>-seed{3,4,5}/`, same chain as Fig 2): `training_metrics.json:parameter_count`, `steady_state_equivalent_seconds` and `final_inference.json` keys. **DB:** `downstream_200m.db` tables `experiments` (`phase='phase1_downstream'`), `downstream_evaluation_runs` (`status='completed'`) and `downstream_task_results` (held-out BPB). | Yes | **All 35 cells match** (table below). **Decode:** `median_cached_decode_tokens_per_second` equals the median of the 30 measured `inference_repetitions[].decode_tokens_per_second` in every run (max \|Δ\| = 0). **Steady s:** recomputed from `step_timing.jsonl` (steps ≥ 11) as Σwall×total/steady tokens; max \|Δ\| 1e-12. **Memory and KV:** constant across all 30 reps × 3 seeds, as the paper states. | Yes: CSV columns `heldout_bpb`, `stabilized_decode_tokens_per_second`, `peak_inference_cuda_memory_mb`, `steady_state_equivalent_seconds`, plus manifest `parameter_count`. 0 mismatches. | none (DERIVED) |
| **Table 2: controls** (`generated/controls_table.tex`; main.tex:96–99) | `generate_tables.py:60–61` | **Hard-coded strings**, except "Frozen prefix" (`budget_analysis.json:arms.{T,N}.attempts`, from `analyze_budget_effect.py:14–36`). | Yes | Each row checked against the raw Pair-A records (`runs/pair-a-official-20260906-fixed-warmup-v2/*/history.jsonl`, manifest, `paper/audit/prompts/{T,N}/*`). **Initial code:** `train.py` `b2fb8788…`, `prepare.py` `4f2ba9cb…` and `study_runtime.py` `50dc7126…` are identical in T and N. **Agent:** `gpt-5.6-sol`/`high`/seed 3 in all 8. **Stop rules:** T 300.09–300.62 s; N 88,604,672 tokens. **LR progress:** T `progress = min(measured_training_time / 300, 1)` (program.md:18); N `min(total_tokens / N_STAR, 1)` (program.md:14). **Prefix:** 4+4 completed. **Protocol:** `autopareto_search_inference_bracketed_v2` in all 8. **GPU:** one GPU UUID in all 8. **Parent policy:** `lineage.quality_valid_speed_uncertain_source_parent_eligible: true`. | No | none in content; see M8 (hard-coded) (OBSERVED, DERIVED) |

**Table 1 recomputed from raw data** (DERIVED; means over seeds 3–5). Every value matches the submitted PDF at its printed precision.

| Recipe | Params | Held-out BPB (SD) | Decode tok/s (SD) | Peak alloc MiB | Steady s | Incr. KV MiB |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 50,332,176 | 1.146590 (0.004673) | 196.8207 (2.4465) | 250.8818 | 688.5001 | 3.9843750 |
| AutoResearch | 50,332,176 | 1.099803 (0.001943) | 192.9332 (1.3795) | 250.8818 | 696.1633 | 3.9843750 |
| TPE | 39,846,284 | 1.119402 (0.004271) | 258.8942 (3.4233) | 205.8809 | 540.5262 | 2.9882812 |
| MOTPE | 26,345,772 | 1.148253 (0.003102) | 270.0064 (5.4014) | 145.5684 | 398.2326 | 2.2412109 |
| AP292 | 29,360,392 | 1.141837 (0.006035) | 383.0218 (4.9005) | 162.8799 | 385.0191 | 1.9921875 |
| AP299 | 26,214,662 | 1.159309 (0.000370) | 488.2581 (8.2976) | 142.3799 | 312.7850 | 1.4941406 |
| AP294 | 17,891,526 | 1.190057 (0.002179) | 495.4913 (3.9530) | 106.9424 | 295.6825 | 1.1206055 |

Other values checked against the same raw records (OBSERVED in raw):

- **Panel-wide constants.** Held-out shard `shard_06541.parquet`, revision `915333b4…`, shard SHA `51da1a9b…`, 2,560 segments and 5,242,880 tokens are identical in all 21 `checkpoint_evaluations.json`.
- **Measurement protocol.** Every run has 30 measured repetitions in 3 passes of 10, protocol `autopareto_inference_stabilized_v2`, and `correctness.passed = true`.
- **Decode token count (DERIVED).** 255 decode tokens are timed: `decode_tokens_per_second × decode_seconds` = 255.
- **Recipe hashes.** The source hashes printed in the appendix (`428e22f1`, `b3a961ee`, `8547b278`, `62882a20`, `55a7fd8b`, `36e3ad91`, `6d0d8d1d`) equal the frozen file, `training_metrics.json:source_sha256` and `evaluation/phase1_public_manifest.json` for all 21 runs.
- **Model configurations.** AP292's config (4 layers, n_embd 512, 4/4 heads, `SSSL`) and the AR config (8, 512, 4/4, `SSSS`) match each run's `training.log` "Model config".

### A.3 Every macro in `paper/generated/results_macros.tex`

All macros are written by `analysis/generate_tables.py:main()`, which writes the file and `headline_values.json` at line 30. Recomputation is DERIVED with my scripts. "Public" means recomputable from files on the live public `main`.

| # | Macro (line) | Value | Used in submitted PDF | Generator (lines) → inputs | Raw recomputation | Public | Status |
|---|---|---|---|---|---|---|---|
| 1 | `\APParams` (1) | 29.36 | §1.4 | :11–13 → `parameters` ← `training_metrics.json:parameter_count` | 29.360392 | yes (manifest) | match |
| 2 | `\APBPB` (2) | 1.141837 | §1.3, §1.4 | :11–13 → `bpb` ← DB `downstream_task_results.metric_value` (`heldout_climbmix_bpb`; `build_dataset.py:51`) | 1.1418371 | yes (CSV) | match |
| 3 | `\APSpeed` (3) | 383.02 | §1.3, §1.4 | `final_inference.json:median_cached_decode_tokens_per_second` | 383.0218 | yes | match |
| 4 | `\APTime` (4) | 385.02 | no | `training_metrics.json:steady_state_equivalent_seconds` | 385.0191 | yes | match (unused) |
| 5 | `\APWall` (5) | 508.23 | no | `training_metrics.json:total_scientific_step_wall_seconds` | 508.2294 | partly (public "scientific wall" is the interval, 512.61) | match; naming (M11) |
| 6 | `\APMemory` (6) | 162.88 | §1.4 | `final_inference.json:peak_inference_cuda_memory_mb` | 162.8798828 | yes | match |
| 7 | `\ARParams` (7) | 50.33 | §1.4 | as #1 | 50.332176 | yes | match |
| 8 | `\ARBPB` (8) | 1.099803 | §1.3, §1.4 | as #2 | 1.0998031 | yes | match |
| 9 | `\ARSpeed` (9) | 192.93 | §1.3, §1.4 | as #3 | 192.9332 | yes | match |
| 10 | `\ARTime` (10) | 696.16 | no | as #4 | 696.1633 | yes | match (unused) |
| 11 | `\ARWall` (11) | 813.45 | no | as #5 | 813.4451 | partly (interval 819.53) | match; naming (M11) |
| 12 | `\ARMemory` (12) | 250.88 | §1.4 | as #6 | 250.8818359 | yes | match |
| 13 | `\DecodeRatio` (13) | 1.99 | abstract, §1.4 | :14 | 1.985256 | yes | match |
| 14 | `\MemoryReduction` (14) | 35.1 | abstract, §1.4 | :15 | 35.0771 | yes | match |
| 15 | `\ParameterReduction` (15) | 41.7 | abstract, §1.4 | :16 | 41.6668 | yes | match |
| 16 | `\BPBCost` (16) | 0.042034 | §1.4 | :17 | 0.0420340 | yes | match |
| 17 | `\PanelN` (17) | 21 | no ("21" is typed in §2.5) | :18 → Σ`eligible` (`build_dataset.py:44`, 8 checks) | 21/21 pass | partly (checks need per-run files) | match |
| 18 | `\BudgetN` (18) | 8 | §1.3 | :18 → `budget_analysis.json:rows` (`analyze_budget_effect.py:13`) ← `runs/pair-a-official-20260906-fixed-warmup-v2/*/history.jsonl` + `paper/audit/first_observed_records.json` | 8 (4 T + 4 N, all `completed`) | no | match; fragile (M12) |
| 19 | `\TimeN` (19) | 4 | no (only in the unsubmitted abstract) | :18 | 4 | no | match |
| 20 | `\TokenN` (20) | 4 | no | :18 | 4 | no | match |
| 21 | `\TimeBest` (21) | 1.103402 | no | :19 | 1.1034022 (T2) | no | match |
| 22 | `\TokenBest` (22) | 1.124312 | no | :19 | 1.1243123 (N3) | no | match |
| 23 | `\BudgetParetoN` (23) | 0 | §1.3 | :20 | 0: all 8 are `quality_valid_speed_uncertain`, `pareto_eligible: false` | no | match |
| 24 | `\BLiMPLower` (24) | 51 | §1.3 | :21 ← `downstream_contrast.json` (`analyze_downstream.py:17–33`) ← DB `downstream_task_results.raw_json.configurations[*].accuracy` | 51 | no | match |
| 25 | `\BLiMPTied` (25) | 1 | no | :21 | 1 (exact tie) | no | match |
| 26 | `\BLiMPHigher` (26) | 15 | no | :21 | 15 | no | match |
| 27 | `\APKV` (27) | 1.992 | §1.3, §1.4 | :22–24 ← mean `inference_repetitions[].incremental_kv_cache_memory_mb` (`build_dataset.py:49`) | 1.9921875 | yes, rounded (`docs/PHASE1_200M_RESULTS.md` per-run "KV MiB") | match |
| 28 | `\ARKV` (28) | 3.984 | §1.3, §1.4 | as #27 | 3.984375 | same | match |
| 29 | `\KVReduction` (29) | 50 | §1.4 | :24 | 50.000 exactly | yes | match |
| 30 | `\BLiMPHeldoutCorr` (30) | −0.969 | §1.3 | :25 ← `downstream_mechanism.json` (`analyze_downstream.py:36–44`, `np.corrcoef`) | −0.968817 | yes (`public_downstream_metrics.json`) | match |
| 31 | `\BLiMPDepthCorr` (31) | 0.717 | §1.3 | :25; depth = AST `DEPTH` constant of the frozen source | 0.717303 | yes (public sources) | match |
| 32 | `\BLiMPParamCorr` (32) | 0.680 | no | :25 | 0.680457 | yes | match |
| 33 | `\BatchOnlyBPB` (33) | 1.1040 | no | :26–29 ← `db_ablation_runs` ⋈ `db_training_runs` | 1.103968 (snapshot DB = `results/results.db`) | yes (CSV `val_bpb`, 3 rows) | match |
| 34 | `\BatchOnlyTokens` (34) | 88.1 | no | :26–29 | 88.080 M | no (no token column in public CSV) | match |
| 35 | `\NonBatchOnlyBPB` (35) | 1.2188 | no | :26–29 | 1.218802 | yes | match |
| 36 | `\NonBatchOnlyTokens` (36) | 94.4 | no | :26–29 | 94.372 M | no | match |

16 of the 36 macros are unused in the submitted PDF: #4, 5, 10, 11, 17, 19–22, 25, 26 and 32–36 (OBSERVED by grep). `headline_values.json` equals `headline_values.previous.json`, so the last refresh produced no headline change (OBSERVED).

### A.4 `primary_result.tex` and the abstract

- **`primary_result.tex`.** Its three sentences are hard-coded in `analysis/generate_reports.py:32–34`, with macros. The literals "seeds 3 to 5" and "201,326,592" match raw (`training_metrics.json:total_training_tokens`, all 21). The file is identical in `paper/`, the reconstructed Overleaf source and the zips (OBSERVED).
- **Submitted abstract.** It uses `\DecodeRatio`, `\ParameterReduction` and `\MemoryReduction` (verified above). Its source file exists only under `paper_backups/automlr_overleaf_reconstructed/` (M1).

### A.5 Hand-typed numbers in main.tex prose (not macros)

| Number (main.tex line) | Raw source | Check |
|---|---|---|
| 1.0933 / 1.0950 / 1.0977 / 1.0981 / 1.1038 / 1.1077 / 1.1083 (41) | DB `experiments` min `val_bpb` with `valid_result=1 AND exclude_from_analysis=0`, by method | DERIVED match (also from public CSV) |
| 50 + 5 AR attempts; 3 × 25 AP; classical 25 each (via `search_comparison_table.tex`, which is not `\input`) | DB counts | DERIVED match. The AR main campaign had 19 completed of 50; diagnostic5 had 3 of 5. |
| 1,536 × 131,072 and 384 × 524,288 (43, 137) | `training_metrics.json` | DERIVED match |
| A8 1.0966→1.1573; A18 1.0957; A22 1.0933; 408.4 tok/s (54) | DB ids 119, 120, 130, 134, 137 | DERIVED match |
| 32/21/10/8/4 (54) | DB `change_category` tally | DERIVED match in count; provenance wrong (M5) |
| 36.91/36.50, 58.29/58.67, 21.50/20.44, 72.02/74.56 (68) | DB `downstream_task_results` means | DERIVED match |
| 67,000 minimal pairs | BLiMP `micro_accuracy.sample_count`; 67 configs × `count` 1000 | DERIVED match. The BLiMP split is `train` (DB `split`); the paper does not state the split. |
| 88,604,672; 524,288 quantum (68, 141) | `runs/pair-a/provenance/final_pre_study_manifest_repaired_20260903.json` `/calibration/selection`: runs 88,604,672 / 88,080,384 / 88,604,672 → median → floor to quantum (169 × 524,288) | DERIVED match |
| 50.33M → 85.85M → 135.27M (68) | Pair-A N1/N2 `result.num_params` | DERIVED match |
| 29,360,392 / 50,332,176; depth/width/heads/window (133) | `training_metrics.json`; `training.log` | OBSERVED match |
| Driver 570.211.01; PyTorch 2.9.1+cu128; CUDA 12.8; Python 3.12 (85, 137) | See D | Python mismatch (M4) |
| "counted tokens exclude the eleven disposable warm-up updates" (137) | `training.log`, `training_metrics.json` | mismatch (M3) |
| Created 2026-08-10, executed from 2026-08-12 (105) | DB `campaigns.created_at` | wording mismatch (M6) |

---

## B. Public-release audit

### B.1 Visibility and URL

- **Visibility.** The repository is public (OBSERVED, 2026-10-28). An unauthenticated `GET https://api.github.com/repos/<owner>/autopareto` returned `private: false`, `visibility: public`, `default_branch: main`, `created_at 2026-08-11T01:28:48Z` and `pushed_at 2026-08-28T03:33:04Z`. The repository page returned HTTP 200.
- **URL.** `https://github.com/<owner>/autopareto.git` (owner redacted). The owner handle equals the local `git config user.name`, so the URL identifies the author. `git remote -v` contains **no embedded credentials** (OBSERVED).
- **Live branches** (OBSERVED via API):
  - `main` → `ad0b627`, "Merge pull request #1 from <owner>/autopareto/professor-ready-2026", 2026-08-28T03:25:05Z. Its parents are `9cb262e` and `fa54db5`.
  - `autopareto/professor-ready-2026` → `6bcb69e`.
  - The local remote-tracking refs `origin/master`, `origin/agenthub` and `origin/exp/H100/mar8` no longer exist on the remote.
- **Stale local ref.** The local `origin/main` ref is stale at `9cb262e` (OBSERVED). The tree of live `main` (`122dda90…`) is **byte-for-byte the tree of local `fa54db5`** (DERIVED: blob-SHA comparison over all 100 paths). Branch `6bcb69e` differs from it only in `docs/ABSTRACT_DRAFT.md`.

### B.2 Files on `origin/main` as recorded locally (`git ls-tree -r origin/main` = `9cb262e`, 40 files; OBSERVED)

```
.gitignore  LICENSE  README.md  RELEASE.md  REPRODUCIBILITY.md  requirements-release.txt
configs/README.md  configs/manifest.json  configs/baseline_seed0.json
configs/seed0_A08_low_memory_boundary.json  configs/seed0_A12_lowest_memory_within_1pct.json
configs/seed0_A22_best_validation_bpb.json  configs/seed0_A24_hypervolume_representative.json
configs/seed0_A25_fastest_within_1pct.json  configs/seed1_A10_lowest_memory_within_1pct.json
configs/seed1_A15_efficiency_boundary.json  configs/seed1_A16_degradation_boundary.json
configs/seed1_A21_fastest_within_1pct.json  configs/seed1_A25_best_validation_bpb.json
configs/seed42_A14_balanced.json  configs/seed42_A15_efficiency.json  configs/seed42_A16_primary_efficiency.json
configs/seed42_A17_fastest_smallest.json  configs/seed42_A21_kv_shared.json  configs/seed42_A22_best_validation_bpb.json
data/README.md  data/experiment_records.csv  data/inference_measurements.csv  data/deployment_measurements.csv
plots/ablation_validation_bpb.{png,svg}  plots/deployment_workload.{png,svg}  plots/discovery_frontier.{png,svg}
protocol/ledger_schema.md  protocol/research_protocol.md
scripts/export_public_data.py  scripts/generate_release_plots.py  scripts/verify_release.py
```

### B.3 Additional files on the live public `main` (`ad0b627` = `fa54db5` tree, +60 files, 100 total; OBSERVED)

`README.md` is replaced by the 2026-08-28 version. The new files are:

```
.python-version  PROJECT_STATE.md  pyproject.toml  train.py  prepare.py
autopareto_db.py  benchmark_config.py  benchmark_stability.py  deployment_benchmark.py  downstream_evaluator.py
event_log.py  inference_benchmark.py
docs/ABSTRACT_DRAFT.md  docs/DOWNSTREAM_EVALUATION.md  docs/NEXT_EXPERIMENTS.md  docs/PHASE1_200M_RESULTS.md  docs/RESEARCH_LEDGER.md
evaluation/downstream_suite.json  evaluation/phase1_equal_token_preflight.json  evaluation/phase1_public_manifest.json
evaluation/phase1_stabilized_inference_rebenchmark.json  evaluation/phase1_token_stream_audit.json
plots/README.md  plots/ap292_vs_autoresearch.{png,svg}  plots/discovery_to_verification.{png,svg}
plots/downstream_comparison.{png,svg}  plots/main_pareto_frontier.{png,svg}
protected/inference_adapter.py  protected/phase1_runtime.py  protected/phase1_source_transform.py  protected/phase1_train_launcher.py
results/candidates/autopareto-seed0-20260731/{0022,0024,0025}-train.py
results/candidates/autoresearch-diagnostic5-seed42-20260724/0005-train.py
results/candidates/baseline-a40-seed0/0002-train.py  results/candidates/motpe-seed42-20260805/0022-train.py
results/candidates/tpe-seed42-20260805/0021-train.py
results/phase1/discovery_heldout_bpb.json  results/phase1/phase1_200m_downstream_selection.json
results/phase1/phase1_200m_downstream_summary.json  results/phase1/phase1_equal_token_results.{csv,json,md}
results/phase1/phase1_stabilized_inference_rebenchmark.{csv,json,md}  results/phase1/public_downstream_metrics.json
scripts/ensure_persistent_cache.py  scripts/ensure_persistent_checkpoint_storage.py  scripts/evaluate_phase1_checkpoints.py
scripts/generate_publication_figures.py  scripts/prepare_phase1_equal_tokens.py  scripts/run_phase1_equal_tokens.py
scripts/run_phase1_stabilized_inference.py  scripts/verify_phase1_token_stream.py
```

### B.4 Internal consistency of the public release (OBSERVED unless marked)

- **Integrity.** The four `RELEASE.md` hashes still match (`data/*.csv`, `configs/manifest.json`). The 7 frozen sources hash to the manifest digests.
- **What it supports (DERIVED).** Table 1, Fig 2, the headline ratios and both correlations are recomputable from public files alone. The public `public_downstream_metrics.json` is rounded to 6 decimal places.
- **Stale and broken documentation.** `RELEASE.md`, `REPRODUCIBILITY.md` and `data/README.md` still describe the 2026-08-11 discovery-only release. They contradict the 2026-08-28 README and the newly added files, and `scripts/verify_release.py` fails on the current README (M9).
- **Undocumented generator.** No script in the public tree, the private history, or the working tree writes the formats `autopareto_phase1_equal_token_results_v1` or `autopareto_public_phase1_downstream_metrics_v1`. I searched for both strings. The generator of the public summaries is UNKNOWN. Its values were checked against raw by `analysis/verify_evidence.py:14–22` (`paper/generated/independent_verification.json:passed = true`) and by me.

### B.5 What an outside researcher is missing (OBSERVED absence on live `main`)

1. **The paper's own pipeline.** Missing: `analysis/*.py`, `paper/requirements.txt`, `paper/refresh_paper.sh` and `paper/generated/*.json`. The public `scripts/generate_publication_figures.py` draws *different* figures. The paper source itself (`main.tex`, `generated/*.tex`, `figures/`) is also absent.
2. **Per-run raw records for the 21 panel runs.** Missing:
   - `run_metadata.json`, `training_metrics.json`, `timing_summary.json` and `step_timing.jsonl`.
   - `final_inference.json`, with its 30 reps, stabilization blocks and GPU telemetry.
   - `checkpoint_evaluations.json` (milestone curves), `events.jsonl` and the derived `phase1_train.py`.
   - Per-run downstream rows (189), BLiMP per-configuration JSON (needed for 51/67) and generations.
   - The runtime token-stream fingerprints (`token_stream_verification.json`).

   Only aggregates are public, plus a per-run markdown table in `docs/PHASE1_200M_RESULTS.md:59–80`.
3. **Discovery histories.** The CSV has metrics, parent, category and decision for 284 rows. Missing: hypotheses, intended changes, agent interpretations, code diffs or sources (7 of about 275 searched candidates are public; `configs/*.json` are "not independently runnable"), failure texts, campaign program texts and campaign views.
4. **Post-Aug-10 rows.** The 12 fixed-time primary confirmations (exp 288–299), the MOTPE recovery retrain (exp 300) and the stabilized deployment sessions 3–6 (6,060 DB rows vs 233 public deployment rows).
5. **Selection manifests.** `autopareto_confirmation_selection.json` and `autopareto_primary_confirmation_plan.json`.
6. **Everything for the Pair-A study.** Protocol, registered programs, executed manifest, N\* calibration, 8 histories plus all relaunch namespaces, prompts, controller code and the scientist Dockerfile and image digest.
7. **Classical search code.** `search/*.py` exists only in the private branch `autopareto/research-infrastructure`; the working tree has only `.pyc` files. Search-space definitions and optimizer-state DBs are also missing.
8. **Environment and data pins.** Missing: `uv.lock` (private history, SHA `262a4d55…`), a per-era environment record, the data manifest with shard SHA-256s and the tokenizer hashes.
9. **Release metadata.** Checksums for the 60 files added on 2026-08-28, a release tag or DOI, and up-to-date `RELEASE.md`/`REPRODUCIBILITY.md` with a passing verifier.
10. **Checkpoints.** Excluded by design. Re-measuring inference or downstream metrics without retraining needs them.

### B.6 Anonymized reviewer bundle

**Not present locally** (OBSERVED; see M10). The only bundles are manuscript sources:

- `autopareto_overleaf.zip` (2026-09-06 23:34).
- `autopareto_final_overleaf.zip` (2026-09-07 00:13; empty `figures/`).
- `paper/working_source_bundle.zip`.
- `paper_backups/paper-20260906-234643/working_source_bundle.zip`.

`analysis/package_paper.py` packages manuscript inputs only. Whether anything was uploaded to OpenReview is UNKNOWN.

### B.7 Anonymity and operational notes (OBSERVED; values not reproduced)

- **Identifying content on the public repo.** Live `main` exposes the author's handle and absolute local paths containing the author's name: `PROJECT_STATE.md:387,878` and `evaluation/phase1_equal_token_preflight.json:105,204,293`. `PROJECT_STATE.md:389` contains a RunPod SSH login endpoint (an identifier, not a key).
- **No credentials found.** A pattern scan for private keys and API tokens found none. The scan is pattern-based, not exhaustive.

---

## C. Reproducibility classification of major results

Classes used:

- **FR:** fully reproducible with public artifacts.
- **RC:** reproducible with extra compute or dependencies.
- **PR:** partially reproducible.
- **NR:** not reproducible from available artifacts.

Proprietary agent behaviour (`gpt-5.6-sol` through a provider; hidden provider state; unrecorded client version for the historical campaigns) cannot be reproduced exactly in any case (INFERRED; the paper acknowledges this at main.tex:114).

| Result | Re-derive the reported numbers | Re-execute | Class | Why |
|---|---|---|---|---|
| **Discovery campaigns** (AutoPareto 3×25, AutoResearch 50+5; Fig 1, §1.3) | Public CSV reproduces all Fig 1 values, parents, decisions, archive bests and category counts (DERIVED) | Agent trajectory not replayable. The harness is not recorded per attempt (M17), the discovery-era environment is unrecorded (D), and only 7 sources are public. | **PR** | The numbers are recomputable, but the search process is not. Fixed-time outcomes also depend on hardware speed: A09 and A10 are code-identical, yet decode differs by 10 %. |
| **Classical archives** (Random/TPE/MOTPE/NSGA-II/CMA-ES, seed 42) | Best BPB per method recomputable from the public CSV (DERIVED) | Code is in a private branch only; search spaces are recorded only by hash (`campaigns.search_space_sha256`). | **PR** (RC once `search/` and the spaces are published) | The samplers are seeded (`campaign_state.algorithm_seed`), but the 300 s wall-clock budget ties results to GPU speed. |
| **Fixed-time confirmations** (seeds 0–2, exp 288–299; earlier exp65/68) | Not public: rows postdate the 2026-08-10 CSV cut. Only `results/phase1/discovery_heldout_bpb.json` (held-out BPB of rows 292/294/299/77/183/300/4) is public. | Requires the private DB, the exp124 source and the confirmation launcher (private `cc756d1`). | **NR** (public); RC with private artifacts plus an A40 | The paper uses these runs only for identifiers and seed disjointness. |
| **Matched-token panel** (21 runs; Table 1, Fig 2, ratios) | **Exact** from `results/phase1/phase1_equal_token_results.csv` + `evaluation/phase1_public_manifest.json` (DERIVED) | Public harness, frozen sources and evaluators (hash-linked). Needs an A40, roughly 4.0 h of training (Σ`end_to_end_runtime_seconds`), 1.1 h of downstream evaluation, ClimbMix shards (unpinned in `prepare.py`), `uv.lock` (private), and the tokenizer rebuilt or obtained. | **FR** (numbers) / **RC** (re-run) | Bitwise BPB is not guaranteed under GPU nondeterminism (INFERRED). Throughput is GPU, driver and clock dependent. The integrity checks need the private per-run files. |
| **Downstream suite** (9 metrics × 21) | Model means and SDs public, giving the §1.3 values and r = −0.969 and 0.717. Per-run values and the BLiMP 51/67 contrast are private only. | Evaluator public and hash-linked (`696f9c46…`); dataset revisions pinned. Needs the 21 checkpoints (not public; only 10 usable locally, M15) or retraining. | **PR** | The per-configuration data behind "51 of 67" is not public. |
| **Inference and memory measurements** | Per-run medians and peak memory public (CSV); per-run KV in the docs table (3 decimal places). The 30 per-rep timings, stabilization blocks and telemetry are private. | `deployment_benchmark.py` is public (`e3f8c8be…`). Needs checkpoints and an A40. | **PR** / RC (re-measure) | Memory is deterministic and should reproduce exactly. Decode should reproduce only within noise; one canary needed repairs and retries (sessions 3–6). |
| **Pair-A budget study** (8 attempts; Table 2, H2) | Nothing public. Locally fully re-derivable: I reproduced every value in §1.3 and Table 2 from the `history.jsonl` files. | Proprietary agent; 13 executed controller and evaluator files are not preserved at their executed hash (M13); pod-side trusted state. The registered `protocol.md` is untracked, and its hash appears in no manifest. | **NR** | The 22-attempt continuation (`runs/pair-a-official-20260914-new-a40-continuation-20260916-r2`, T 22 completed; N 20 completed + 2 failures) is also local-only. |
| Ablations (macros #33–36; unused in the PDF) | BPB from the public CSV; tokens private | Needs sources and compute | **PR** | none |

---

## D. Environment specification

### D.1 Recorded environment by era

| Era (dates) | Python | PyTorch | CUDA | NVIDIA driver | GPU / host | Source of record |
|---|---|---|---|---|---|---|
| Discovery campaigns: AP, AR, classical (2026-07-19 → 08-07) | **not recorded**. Crash tracebacks show `/usr/lib/python3.10/` (INFERRED 3.10.x). | not recorded; `pyproject.toml` pins `torch==2.9.1` (cu128 index) (INFERRED) | not recorded | not recorded | "NVIDIA A40" only (`deployment_sessions.hardware_json`) | DB `experiments` has no environment columns; `experiment_events` payloads contain no versions (OBSERVED) |
| Fixed-time confirmations (08-11/12) | not recorded | not recorded | n/a | not recorded | A40 | as above |
| Original downstream, 11 representatives (08-13/14) | 3.10.18 | 2.9.1+cu128 | 12.8 | n/a | A40; `datasets` 5.0.1 | `downstream_evaluation_runs.environment_json` |
| Inference canary v2, session 4 (08-16) | **3.12.3** | **2.8.0+cu128** | 12.8 | 570.195.03 | A40 | `deployment_sessions.environment_json`; `deployment_measurements.gpu_telemetry_json` |
| Final stabilized deployment, sessions 5–6 (08-16/17) | not recorded | not recorded | n/a | 570.211.01 (s5); **580.126.09** (s6) | A40 | `gpu_telemetry_json` |
| **Matched-token panel, 21 runs (08-26/27)** | **3.10.18** | 2.9.1+cu128 | 12.8 | **570.211.01** | One physical A40: one GPU UUID across two pod identifiers. Kernel 6.8.0-52. Git `5bf86b57` (dirty). | `run_metadata.json:software_versions`; `final_inference.json:gpu_telemetry[].driver_version` |
| Panel downstream (08-27) | 3.10.18 | 2.9.1+cu128 | 12.8 | n/a | kernel 6.8.0-134 (a different host or kernel from training) | `environment_json` |
| A40 environment record (08-31) | 3.10.18 | 2.9.1+cu128 | 12.8 (nvcc 12.8) | **580.159.04** | A40; triton 3.5.1, kernels 0.12.1, pyarrow 23.0.1 | `study/contracts_2026/provenance/a40_environment.json` |
| N\* calibration v3 (09-01/02) | 3.10.18 | 2.9.1+cu128 | 12.8 | 570.195.03 | A40 | `n_star_freeze_v3_20260902.json`; `compute_ledger.json` |
| **Pair-A official v2 (09-06)** | **3.12** (`/usr/local/bin/python`) | 2.9.1+cu128 | 12.8 | **not recorded** | One A40, a different physical GPU from the panel. pyarrow 23.0.1; flash-attn3 rev `e29f138f…`. | v2 manifest `runtime`, `registered_artifacts`; history records carry only the GPU UUID |
| Paper analysis (09-06/07) | 3.10.20 (`paper/.venv`) | — | — | — | CPU | `paper/requirements.txt`: matplotlib 3.10.9, numpy 2.2.6, pypdf 6.17.0, beautifulsoup4 4.15.0 |

All rows are OBSERVED unless marked otherwise.

**Consistency (DERIVED).**

- The headline panel is internally consistent: one Python, torch, CUDA, driver and GPU, and one git commit string across all 21 runs.
- Across eras the environment is **not** consistent: at least 4 drivers, 2 Python minor versions and 2 torch versions, on several physical A40s.
- The paper's single environment sentence (main.tex:85) is accurate for the panel except Python, and accurate for Pair-A except the unrecorded driver.
- The discovery-era environment is UNKNOWN.

### D.2 Lockfiles, containers, data and tokenizer

- **Lockfile.**
  - `uv.lock` (2,297 lines; SHA-256 `262a4d55…`) is the "authoritative environment spec" in `a40_environment.json` (`install_command: uv sync --frozen --offline`). It exists only in the private branch `autopareto/research-infrastructure` (commit `e74771d`). It is absent from the working tree and the public tree (OBSERVED).
  - Public files: `.python-version` = `3.10`; `pyproject.toml` (SHA `147f82fd…`, matches the recorded hash) uses lower bounds except `torch==2.9.1`, `cmaes==0.12.0` and `optuna==4.5.0`.
  - Pair-A's Python 3.12 runtime has no lockfile at all (OBSERVED absence).
  - `requirements-release.txt` pins matplotlib 3.10.8, while `paper/requirements.txt` pins 3.10.9 (OBSERVED).
- **Docker.**
  - The scientist image is `autopareto-scientist-codex:0.153.0`, digest `sha256:877dae34…` (v2 manifest). It is built from `study/contracts_2026/scientist_runtime.Dockerfile` (base `node:22-bookworm-slim@sha256:83f487e0…`, `@openai/codex@0.153.0`). The Dockerfile is untracked and unpublished; the image is local only (OBSERVED).
  - There is no container for training or inference: they ran in pod virtual environments (OBSERVED).
- **Dataset revision.**
  - Held-out evaluation: `karpathy/climbmix-400b-shuffle@915333b4f8b8684f39aeaafea600fea6f43fb703`, `shard_06541.parquet`, SHA `51da1a9b…` (recorded per run).
  - Downstream revisions: LAMBADA `900124bf…`, BLiMP `877fba08…`, PIQA `142f6d73…`, ARC-Easy `210d026f…` (`evaluation/downstream_suite.json`; DB `dataset_revision`).
  - **Training shards are fetched from `resolve/main`** (`prepare.py:41`). Their content hashes are only in the untracked `data_manifest.json` (aggregate `12118270…`; 10 training shards plus validation `shard_06542`).
  - The panel's token-stream identity was runtime-verified *conditional on identical tokenizer and parquet bytes* (`token_stream_verification.json`, private) (OBSERVED).
- **Tokenizer.**
  - `tokenizer.pkl` SHA-256 `888c9fa8…` and `token_bytes.pt` `23517e05…` (`data_manifest.json`; v2 manifest `tokenizer_sha256`).
  - **Not recorded per run** for the discovery or panel eras (UNKNOWN).
  - The tokenizer cache was re-linked mid-discovery, after the exp-91 and A17 crashes. Byte-identity before and after is UNKNOWN.

---

## E. Statistical analysis scripts

All scripts named below exist **locally only** in the untracked `analysis/` directory (OBSERVED).

| Statistic | Script / function (lines) | Exists? | Re-derived by me |
|---|---|---|---|
| Recipe means (Table 1, Fig 2) | `analysis/analyze_pareto.py:4–14`, using `common.py:30–32` `summary()` (`statistics.mean`) | yes, local only | yes, exact |
| Sample SDs (Fig 2 bars; public ± values) | `common.py:32` (`statistics.stdev`, n − 1) | yes, local only. The public-summary generator was not found (B.4). | yes (e.g. AP292 decode SD 4.9005; BPB SD 0.006035) |
| Nondominance (Fig 2 markers) | `common.py:59–63` `dominates()`; `analyze_pareto.py:12–13` | yes | yes (6 nondominated; Baseline dominated by TPE and AP292) |
| Ratios and percentage reductions | `generate_tables.py:14–17, 22–24` | yes | yes |
| Pearson r: BLiMP vs held-out BPB, depth, params | `analyze_downstream.py:36–44` (`np.corrcoef` over the 7 recipe means; depth = AST `DEPTH`) | yes | **−0.968817, 0.717303, 0.680457**, from raw DB means and again from public aggregates. Spearman as a robustness check (not reported): −0.893 and 0.698. |
| BLiMP per-configuration contrast | `analyze_downstream.py:17–33` | yes | 51 / 1 / 15 |
| Budget prefix statistics | `analyze_budget_effect.py:14–36`; `generate_tables.py:18–20` | yes | yes |
| Archive best BPB | `generate_tables.py:33–46` (table not `\input`) | yes | yes |
| Edit-category counts (32/21/10/8/4) | **none**: a raw tally of `experiments.change_category` | **no** | tally reproduced; provenance issue (M5) |
| N\* derivation | the calibration selection inside the manifest (`floor(median/524288)·524288`) | recorded, not a script in `analysis/` | yes |
| Integrity checks | `build_dataset.py:44`; `verify_evidence.py:7–25` | yes | yes (all 21 pass) |
| Bootstrap CIs and significance tests | none reported. Held-out BPB has bootstrap CIs in the data (`confidence_low/high`, 1,000 reps per `downstream_suite.json`). | n/a | consistent with the paper's "no significance claims" |

No script, local or public, regenerates `results/phase1/phase1_equal_token_results.json` or `public_downstream_metrics.json` (OBSERVED by string search). Their values are nonetheless verified against raw data.

---

## F. Minimal additions to the public artifact, ranked by value per effort

1. **Repair the public release metadata.** Value: very high. Effort: under 1 h.
   - Update `RELEASE.md`, `REPRODUCIBILITY.md` and `data/README.md` to match the 2026-08-28 content.
   - Fix `scripts/verify_release.py:63–67`.
   - Add `SHA256SUMS` covering all 100+ files.
   - Tag the camera-ready commit; optionally mint a Zenodo DOI.
   - Remove absolute local paths and the SSH endpoint from `PROJECT_STATE.md` and the preflight JSON.
   - The camera-ready may now cite the URL.
2. **Selection manifests.** Value: high. Effort: about 15 min. Publish `results/autopareto_confirmation_selection.json` (`ea3b3d7`, SHA `72d58664…`) and `results/autopareto_primary_confirmation_plan.json` (`cc756d1`) with their commit IDs and dates. This directly supports "selection preceded confirmation" and fixes the evidence behind M6.
3. **Per-run records for all 21 matched-token runs.** Value: high. Effort: 1–2 h. Size: about 15–20 MB; the snapshot tarball without checkpoints is 12.9 MB compressed. Include:
   - the run bundle files `run_metadata.json`, `training_metrics.json`, `timing_summary.json`, `step_timing.jsonl`, `checkpoint_evaluations.json`, `events.jsonl`, `phase1_train.py`, and `final_inference.json` with all 30 reps and stabilization blocks;
   - `token_stream_verification.json`;
   - the 189 per-run `downstream_task_results` rows, including BLiMP `raw_json`.

   Redact GPU UUIDs and absolute paths. This makes the panel integrity checks, the KV figure and the 51/67 count publicly verifiable.
4. **The analysis pipeline in "public mode".** Value: high. Effort: 2–4 h.
   - Publish `analysis/*.py`, `paper/requirements.txt` and the generated JSON, and commit them to git.
   - Remove the dependency on `paper/audit/first_observed_records.json`, or publish that file.
   - Derive the Fig 1 labels from logs or AST instead of `generate_figures.py:44` (fixes M2).
   - Generate the abstract from the submitted text (fixes M1).
5. **Environment lock and per-era environment table.** Value: medium–high. Effort: about 1 h. Publish `uv.lock` (`262a4d55…`), a lock for the Pair-A Python 3.12 runtime, the scientist Dockerfile with its image digest, and the table in D.1. Correct main.tex:85 (fixes M4) and main.tex:137 (fixes M3).
6. **Data and tokenizer pins.** Value: medium. Effort: under 1 h.
   - Publish `data_manifest.json` (per-shard SHA-256 values), the tokenizer hashes and the held-out shard hash.
   - Change `prepare.py:41` to resolve revision `915333b4…`.
   - Ship the panel's microbatch fingerprints.

   Together these let outsiders verify data identity (fixes M14).
7. **Discovery histories with failures.** Value: high. Effort: 2–4 h, including the redaction review.
   - For all 275 searched attempts (AP, AR, classical): hypothesis, intended change, agent-reported `change_category` (labelled as such; fixes M5), parent, decision, status, `crash_or_failure_reason`, and the full source or `code_diff` (about 26 KB each, about 7 MB in total).
   - Campaign `program_instructions` and their hashes.
   - Flag the missing `0017-train.py` and the corrupt `baseline-a40-seed0/0003-train.py`.
   - Add the post-Aug-10 confirmation rows (exp 288–300).
   - Add the classical `search/*.py` and the search-space definitions from the private branch.
8. **Pair-A histories.** Value: high. Effort: 3–6 h, mostly redaction.
   - `history.jsonl` for both arms (about 5.5 MB each), `campaign_view.json`, `source_registry.jsonl`, `sources/` and the executed prompts (`paper/audit/prompts/{T,N}`).
   - The v2 manifest (redact the SSH host and GPU UUID), the schedule and the N\* calibration records.
   - An index of every relaunch namespace with its status, so abandoned relaunches are visible.
   - The 22-attempt continuation if the camera-ready uses it.
9. **Executed Pair-A controller code.** Value: medium. Effort: medium–high. Recover the 13 files whose executed hashes are missing locally (M13), from the preserved pod snapshot if it still exists, or declare them unrecoverable. From now on, commit `study/` before every launch and record `git rev-parse HEAD` plus `git diff` per attempt.
10. **Checkpoints.** Value: medium. Effort: high (storage). Publish the 21 final checkpoints (2.25 GB by Σ`checkpoint_bytes`) on the HF Hub or Zenodo with their recorded SHA-256 values. Recover the 11 missing or truncated ones first (M15). This enables re-measuring inference, memory and downstream metrics without retraining.

---

## Appendix: recomputation scripts (ephemeral scratchpad)

Directory: `/private/tmp/claude-501/-Users-Rami-Documents-dev-autopareto/084ded0c-4778-442b-9b7f-c93e451f1f5d/scratchpad/audit/repro/`. Every script is read-only, and every one opens SQLite in `mode=ro&immutable=1`.

- **`recompute_table1.py`.** Recomputes the Table 1 means and SDs, the integrity checks, the reps-median and steady-state replays, the environment sets and all headline ratios from the 21 run directories plus the snapshot DB. Writes `table1_per_run.json` and `table1_summary.json`.
- **`recompute_downstream.py`.** Recomputes the downstream means, BLiMP 51/1/15, Pearson and Spearman r, splits, revisions and sample counts.
- **`lineage.py`.** Checks the Fig 1 values against the DB, the recorded parents and the constant diffs behind each edit label. Model widths were confirmed separately from `results/logs/autopareto-seed0-20260731-00NN.log`.
- **`pair_a.py`.** Summarizes the 8 Pair-A records.
- **`env_eras.py`.** Extracts the environment and driver records per era.
- **`inv_check.py`.** Checks 144 key inputs against `paper/generated/artifact_inventory.json`: 0 changed.
- **`q.py`, `db_schema.py`.** Ad hoc SQL queries and schema dumps.
- **`extract_pdf.py`, `render.py`.** Text extraction and rasterizing of the figure regions.

**Public-only checks.** I also recomputed Table 1, the ratios, the correlations, the Fig 1 values and the archive bests from `git show fa54db5:<path>` files alone, with 0 mismatches. `results/results.db` (2026-08-17) is a strict subset of the snapshot DB: every common row in 10 tables is identical. It is not used by the paper pipeline (`analysis/common.py:9` points at the snapshot DB). `results/phase1/downstream_200m.db` is a 0-byte placeholder.
